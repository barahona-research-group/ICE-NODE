"""."""
from __future__ import annotations

from typing import Callable, Tuple, Optional, List

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom

from .base_models import (ObsStateUpdate, NeuralODE_JAX)
from .base_models_koopman import SKELKoopmanOperator, VanillaKoopmanOperator
from .embeddings import (InpatientEmbedding, EmbeddedInAdmission, InpatientLiteEmbedding,
                         LiteEmbeddedInAdmission, DeepMindPatientEmbedding,
                         DeepMindEmbeddedAdmission)
from .model import (InpatientModel, ModelConfig, ModelRegularisation,
                    Precomputes)
from ..ehr import (Admission, InpatientObservables, AdmissionPrediction,
                   DatasetScheme, DemographicVectorConfig, CodesVector, LeadingObservableExtractorConfig,
                   PatientTrajectory,
                   TrajectoryConfig)
from ..utils import model_params_scaler


class InICENODEConfig(ModelConfig):
    state: int = 50
    lead_predictor: str = "monotonic"


class InICENODERegularisation(ModelRegularisation):
    L_taylor: float = 0.0
    taylor_order: int = 0


class MonotonicLeadingObsPredictor(eqx.Module):
    _mlp: eqx.nn.MLP

    def __init__(self, input_size: int,
                 leading_observable_config: LeadingObservableExtractorConfig,
                 key: jax.random.PRNGKey, **mlp_kwargs):
        super().__init__()
        out_size = len(leading_observable_config.leading_hours) + 1
        width = mlp_kwargs.get("width_size", out_size * 5)
        self._mlp = eqx.nn.MLP(input_size,
                               out_size,
                               width_size=width,
                               depth=2,
                               key=key)

    def __call__(self, lead):
        y = self._mlp(lead)
        risk = y[-1]
        p = jnp.cumsum(jnn.softmax(y[:-1]))
        return risk * p


class SigmoidLeadingObsPredictor(eqx.Module):
    _t: jnp.ndarray = eqx.static_field()
    _mlp: eqx.nn.MLP

    def __init__(self, input_size: int,
                 leading_observable_config: LeadingObservableExtractorConfig,
                 key: jax.random.PRNGKey, **mlp_kwargs):
        super().__init__()
        self._t = jnp.array(leading_observable_config.leading_hours)
        width = mlp_kwargs.get("width_size", len(self._t) * 5)
        self._mlp = eqx.nn.MLP(input_size,
                               4,
                               width_size=width,
                               depth=3,
                               key=key)

    def __call__(self, lead):
        [vscale, hscale, vshift, hshift] = self._mlp(lead)
        return vscale * jnn.sigmoid(hscale * (self._t - hshift)) + vshift


class MLPLeadingObsPredictor(eqx.Module):
    _mlp: eqx.nn.MLP

    def __init__(self, input_size: int,
                 leading_observable_config: LeadingObservableExtractorConfig,
                 key: jax.random.PRNGKey, **mlp_kwargs):
        super().__init__()
        width = mlp_kwargs.get(
            "width_size",
            len(leading_observable_config.leading_hours) * 5)
        self._mlp = eqx.nn.MLP(input_size,
                               len(leading_observable_config.leading_hours),
                               width_size=width,
                               final_activation=jnn.sigmoid,
                               depth=3,
                               key=key)

    def __call__(self, lead):
        return self._mlp(lead)


class InICENODE(InpatientModel):
    """
    The InICENODE model. It is composed of the following components:
        - f_emb: Embedding function.
        - f_obs_dec: Observation decoder.
        - f_dx_dec: Discharge codes decoder.
        - f_dyn: Dynamics function.
        - f_update: Update function.
    """
    _f_emb: Callable[[Admission], EmbeddedInAdmission]
    _f_obs_dec: Callable
    _f_lead_dec: Callable
    _f_dx_dec: Callable
    _f_dyn: Callable
    _f_update: Callable

    config: InICENODEConfig = eqx.static_field()

    def __init__(self, config: InICENODEConfig, schemes: Tuple[DatasetScheme],
                 demographic_vector_config: DemographicVectorConfig,
                 leading_observable_config: LeadingObservableExtractorConfig,
                 key: "jax.random.PRNGKey"):
        self._assert_demo_dim(config, schemes[1], demographic_vector_config)
        (emb_key, obs_dec_key, lead_key, dx_dec_key, dyn_key,
         update_key) = jrandom.split(key, 6)
        f_emb = self._make_embedding(
            schemes=schemes,
            demographic_vector_config=demographic_vector_config,
            config=config,
            key=emb_key)
        f_dx_dec = self._make_dx_dec(config=config,
                                     dx_size=len(schemes[1].outcome),
                                     key=dx_dec_key)

        self._f_obs_dec = self._make_obs_dec(config=config,
                                             obs_size=len(schemes[1].obs),
                                             key=obs_dec_key)
        self._f_lead_dec = self._make_lead_dec(
            config=config,
            input_size=config.state,
            leading_observable_config=leading_observable_config,
            key=lead_key)
        self._f_dyn = self._make_dyn(config=config, key=dyn_key)
        self._f_update = self._make_update(config=config,
                                           obs_size=len(schemes[1].obs),
                                           key=update_key)

        super().__init__(config=config, _f_emb=f_emb, _f_dx_dec=f_dx_dec)

    @staticmethod
    def _make_lead_dec(config, input_size, leading_observable_config, key,
                       **kwargs):
        if config.lead_predictor == "monotonic":
            return MonotonicLeadingObsPredictor(input_size,
                                                leading_observable_config,
                                                key=key,
                                                **kwargs)
        elif config.lead_predictor == "sigmoid":
            return SigmoidLeadingObsPredictor(input_size,
                                              leading_observable_config,
                                              key=key,
                                              **kwargs)
        elif config.lead_predictor == "mlp":
            return MLPLeadingObsPredictor(input_size,
                                          leading_observable_config,
                                          key=key,
                                          **kwargs)
        else:
            raise ValueError(
                f"Unknown leading predictor type: {config.lead_predictor}")

    @staticmethod
    def _make_dyn(config, key):
        dyn_input_size = config.state + config.emb.inp_proc_demo
        f_dyn = eqx.nn.MLP(in_size=dyn_input_size,
                           out_size=config.state,
                           activation=jnn.tanh,
                           depth=2,
                           width_size=config.state * 5,
                           key=key)
        f_dyn = model_params_scaler(f_dyn, 1e-2, eqx.is_inexact_array)
        return NeuralODE_JAX(f_dyn, timescale=1.0)

    @staticmethod
    def _make_update(config, obs_size, key):
        return ObsStateUpdate(config.state, obs_size, key=key)

    @staticmethod
    def _make_dx_dec(config, dx_size, key):
        return eqx.nn.MLP(config.state,
                          dx_size,
                          config.state * 5,
                          depth=1,
                          key=key)

    @staticmethod
    def _make_obs_dec(config, obs_size, key):
        return eqx.nn.MLP(config.state,
                          obs_size,
                          obs_size * 5,
                          depth=1,
                          key=key)

    @staticmethod
    def _make_embedding(config, demographic_vector_config, schemes, key):
        return InpatientEmbedding(
            schemes=schemes,
            demographic_vector_config=demographic_vector_config,
            config=config.emb,
            key=key)

    @eqx.filter_jit
    def _init_state(self, dx):
        return jnp.hstack((jnp.zeros((self.config.state - len(dx),)), dx))

    @eqx.filter_jit
    def _safe_integrate(self, delta, state, int_e, precomputes):
        # dt = float(jax.block_until_ready(delta))
        # if dt < 0.0:
        # logging.error(f"Time diff is {dt}!")
        # return state
        # if dt < 1 / 3600.0:
        # logging.debug(f"Time diff is less than 1 second: {dt}")
        # return state
        # else:
        second = jnp.array(1 / 3600.0)
        delta = jnp.where((delta < second) & (delta >= 0.0), second, delta)
        return self._f_dyn(delta, state, args=dict(control=int_e))[-1]

    def sample_state_trajectory(self, state: jnp.ndarray, int_e: jnp.ndarray,
                                t0: float, t1: float,
                                sampling_time: Optional[jnp.ndarray],
                                precomputes: Precomputes) -> List[jnp.ndarray]:
        if sampling_time is None:
            return []
        mask = (sampling_time > t0) & (sampling_time <= t1)
        if jnp.sum(mask) == 0:
            return []
        sampling_time = sampling_time[mask]
        trajectory = []
        t = t0
        for t_sample in sampling_time:
            delta_i = t_sample - t
            state = self._safe_integrate(delta_i, state, int_e, precomputes)
            t = t_sample
            trajectory.append(state)
        return trajectory

    def decode_obs_trajectory(
            self, trajectory: PatientTrajectory) -> PatientTrajectory:
        obs = eqx.filter_vmap(self._f_obs_dec)(trajectory.value)
        return PatientTrajectory(time=trajectory.time, value=obs)

    def decode_lead_trajectory(
            self, trajectory: PatientTrajectory) -> PatientTrajectory:
        lead = eqx.filter_vmap(self._f_lead_dec)(trajectory.value)
        return PatientTrajectory(time=trajectory.time, value=lead)

    def step_segment(self, state: jnp.ndarray, int_e: jnp.ndarray,
                     obs: InpatientObservables, lead: InpatientObservables,
                     t0: float, t1: float,
                     sampling_time: Optional[jnp.ndarray],
                     precomputes: Precomputes):
        preds = []
        lead_preds = []
        trajectory = []
        t = t0
        for t_obs, val, mask in zip(obs.time, obs.value, obs.mask):
            trajectory.extend(
                self.sample_state_trajectory(state, int_e, t, t_obs,
                                             sampling_time, precomputes))
            # if time-diff is more than 1 seconds, we integrate.
            state = self._safe_integrate(t_obs - t, state, int_e, precomputes)
            pred_obs = self._f_obs_dec(state)
            pred_lead = self._f_lead_dec(state)
            state = self._f_update(state, pred_obs, val, mask)
            t = t_obs
            preds.append(pred_obs)
            lead_preds.append(pred_lead)

        trajectory.extend(
            self.sample_state_trajectory(state, int_e, t, t1, sampling_time,
                                         precomputes))
        state = self._safe_integrate(t1 - t, state, int_e, precomputes)

        if len(preds) > 0:
            pred_obs_val = jnp.vstack(preds)
            pred_lead_val = jnp.vstack(lead_preds)
        else:
            pred_obs_val = jnp.empty_like(obs.value)
            pred_lead_val = jnp.empty_like(lead.value)

        return state, (InpatientObservables(obs.time, pred_obs_val, obs.mask),
                       InpatientObservables(lead.time, pred_lead_val,
                                            lead.mask)), trajectory

    def __call__(
            self, admission: Admission,
            embedded_admission: EmbeddedInAdmission, precomputes: Precomputes,
            regularisation: InICENODERegularisation,
            store_embeddings: Optional[TrajectoryConfig]
    ) -> AdmissionPrediction:
        state = self._init_state(embedded_admission.dx0)
        int_e = embedded_admission.inp_proc_demo
        obs = admission.observables
        lead = admission.leading_observable
        pred_obs_l = []
        pred_lead_l = []
        t0 = admission.interventions.t0
        t1 = admission.interventions.t1

        trajectory = [state]
        if store_embeddings is not None:
            dt = store_embeddings.sampling_rate
            sampling_time = jnp.arange(0.0, t1[-1], dt)
        else:
            sampling_time = None
        for i in range(len(t0)):
            state, pred, traj = self.step_segment(state, int_e[i], obs[i],
                                                  lead[i], t0[i], t1[i],
                                                  sampling_time, precomputes)
            pred_obs, pred_lead = pred

            pred_obs_l.append(pred_obs)
            pred_lead_l.append(pred_lead)
            trajectory.extend(traj)

        pred_dx = CodesVector(self._f_dx_dec(state), admission.outcome.scheme)

        if len(trajectory) > 0:
            trajectory = PatientTrajectory(time=sampling_time,
                                           value=jnp.vstack(trajectory))
        else:
            trajectory = None

        return AdmissionPrediction(admission=admission,
                                   outcome=pred_dx,
                                   observables=pred_obs_l,
                                   leading_observable=pred_lead_l,
                                   trajectory=trajectory)

    @property
    def dyn_params_list(self):
        return self.params_list(self._f_dyn)


class InICENODELiteConfig(InICENODEConfig):
    pass


class InICENODELite(InICENODE):
    """
    The InICENODE model. It is composed of the following components:
        - f_emb: Embedding function.
        - f_obs_dec: Observation decoder.
        - f_dyn: Dynamics function.
        - f_update: Update function.
    """
    _f_emb: Callable[[Admission], EmbeddedInAdmission]
    _f_obs_dec: Callable
    _f_lead_dec: Callable
    _f_init: Callable
    _f_dyn: Callable
    _f_update: Callable

    config: InICENODELiteConfig = eqx.static_field()

    def __init__(self, config: InICENODELiteConfig,
                 schemes: Tuple[DatasetScheme],
                 demographic_vector_config: DemographicVectorConfig,
                 leading_observable_config: LeadingObservableExtractorConfig,
                 key: "jax.random.PRNGKey"):
        super_key, init_key = jrandom.split(key, 2)
        super().__init__(config, schemes, demographic_vector_config,
                         leading_observable_config, super_key)
        self._f_init = self._make_init(config=config, key=init_key)

    @staticmethod
    def _make_dx_dec(config, dx_size, key):
        return None

    @staticmethod
    def _make_init(config, key):
        init_input_size = config.emb.inp_proc_demo + config.emb.dx
        return eqx.nn.MLP(init_input_size,
                          config.state,
                          config.emb.dx * 5,
                          depth=2,
                          key=key)

    @eqx.filter_jit
    def _init_state(self, int_demo_emb, dx0):
        return self._f_init(jnp.hstack((int_demo_emb, dx0)))

    def __call__(
            self, admission: Admission,
            embedded_admission: EmbeddedInAdmission, precomputes: Precomputes,
            regularisation: InICENODERegularisation,
            store_embeddings: Optional[TrajectoryConfig]
    ) -> AdmissionPrediction:
        int_e = embedded_admission.inp_proc_demo

        state = self._init_state(int_demo_emb=int_e[0],
                                 dx0=embedded_admission.dx0)

        obs = admission.observables
        lead = admission.leading_observable
        pred_obs_l = []
        pred_lead_l = []
        t0 = admission.interventions.t0
        t1 = admission.interventions.t1
        trajectory = [state]
        if store_embeddings is not None:
            dt = store_embeddings.sampling_rate
            sampling_time = jnp.arange(0.0, t1[-1], dt)
        else:
            sampling_time = None
        for i in range(len(t0)):
            state, pred, traj = self.step_segment(state, int_e[i], obs[i],
                                                  lead[i], t0[i], t1[i],
                                                  sampling_time, precomputes)
            pred_obs, pred_lead = pred

            pred_obs_l.append(pred_obs)
            pred_lead_l.append(pred_lead)
            trajectory.extend(traj)

        if len(trajectory) > 0:
            trajectory = PatientTrajectory(time=sampling_time,
                                           value=jnp.vstack(trajectory))
        else:
            trajectory = None

        return AdmissionPrediction(admission=admission,
                                   outcome=None,
                                   observables=pred_obs_l,
                                   leading_observable=pred_lead_l,
                                   trajectory=trajectory)


class InGRUConfig(InICENODELiteConfig):
    pass


class InGRUJump(InICENODELite):
    # TODO: as for the original paper, use multi-layer embeddings with skip
    # connections.
    _f_emb: Callable[[Admission], EmbeddedInAdmission]
    _f_obs_dec: Callable
    _f_lead_dec: Callable
    _f_init: Callable
    # GRU. Alternatives: SRU, LSTM, ..
    _f_update: Callable
    _f_dyn: Callable

    config: InGRUConfig = eqx.static_field()

    @staticmethod
    def _make_embedding(config, demographic_vector_config, schemes, key):
        return InpatientLiteEmbedding(
            schemes=schemes,
            demographic_vector_config=demographic_vector_config,
            config=config.emb,
            key=key)

    @staticmethod
    def _make_dyn(config, key):
        return eqx.nn.GRUCell(input_size=config.emb.demo,
                              hidden_size=config.state,
                              key=key)

    @staticmethod
    def _make_init(config, key):
        init_input_size = config.emb.demo + config.emb.dx
        return eqx.nn.MLP(init_input_size,
                          config.state,
                          config.emb.dx * 5,
                          depth=2,
                          key=key)

    @eqx.filter_jit
    def _init_state(self, demo_emb, dx0):
        return self._f_init(jnp.hstack((demo_emb, dx0)))

    def __call__(
            self, admission: Admission,
            embedded_admission: LiteEmbeddedInAdmission,
            precomputes: Precomputes, regularisation: ModelRegularisation,
            store_embeddings: Optional[TrajectoryConfig]
    ) -> AdmissionPrediction:
        obs = admission.observables
        lead = admission.leading_observable
        if len(obs) == 0:
            pred_obs = InpatientObservables.empty(obs.value.shape[1])
            pred_lead = InpatientObservables.empty(lead.value.shape[1])
            return AdmissionPrediction(admission=admission,
                                       outcome=None,
                                       observables=pred_obs,
                                       leading_observable=pred_lead,
                                       trajectory=None)

        demo_e = embedded_admission.demo

        state = self._init_state(demo_emb=embedded_admission.demo,
                                 dx0=embedded_admission.dx0)

        pred_obs_l = []
        pred_lead_l = []
        for i in range(len(obs.time)):
            state = self._f_dyn(demo_e, state)
            pred_obs = self._f_obs_dec(state)
            pred_lead = self._f_lead_dec(state)
            state = self._f_update(state, pred_obs, obs.value[i], obs.mask[i])
            pred_obs_l.append(pred_obs)
            pred_lead_l.append(pred_lead)

        pred_obs = InpatientObservables(obs.time, jnp.vstack(pred_obs_l),
                                        obs.mask)
        pred_lead = InpatientObservables(lead.time, jnp.vstack(pred_lead_l),
                                         lead.mask)

        return AdmissionPrediction(admission=admission,
                                   outcome=None,
                                   observables=pred_obs,
                                   leading_observable=pred_lead,
                                   trajectory=None)


class InGRU(InGRUJump):

    @staticmethod
    def _make_embedding(config, demographic_vector_config, schemes, key):
        return DeepMindPatientEmbedding(
            schemes=schemes,
            demographic_vector_config=demographic_vector_config,
            config=config.emb,
            key=key)

    @staticmethod
    def _make_dyn(config, key):
        return eqx.nn.GRUCell(input_size=config.emb.sequence,
                              hidden_size=config.state,
                              key=key)

    def __call__(
            self, admission: Admission,
            embedded_admission: DeepMindEmbeddedAdmission,
            precomputes: Precomputes, regularisation: ModelRegularisation,
            store_embeddings: Optional[TrajectoryConfig]
    ) -> AdmissionPrediction:
        obs = admission.observables
        lead = admission.leading_observable
        if len(obs) == 0:
            pred_obs = InpatientObservables.empty(obs.value.shape[1])
            pred_lead = InpatientObservables.empty(lead.value.shape[1])
            return AdmissionPrediction(admission=admission,
                                       outcome=None,
                                       observables=pred_obs,
                                       leading_observable=pred_lead,
                                       trajectory=None)

        state = self._init_state(demo_emb=embedded_admission.demo,
                                 dx0=embedded_admission.dx0)
        sequence_e = embedded_admission.sequence
        pred_obs_l = []
        pred_lead_l = []
        for seq_e in sequence_e:
            pred_obs = self._f_obs_dec(state)
            pred_lead = self._f_lead_dec(state)
            state = self._f_dyn(seq_e, state)
            pred_obs_l.append(pred_obs)
            pred_lead_l.append(pred_lead)

        pred_obs = InpatientObservables(obs.time, jnp.vstack(pred_obs_l),
                                        obs.mask)
        pred_lead = InpatientObservables(lead.time, jnp.vstack(pred_lead_l),
                                         lead.mask)

        return AdmissionPrediction(admission=admission,
                                   outcome=None,
                                   observables=pred_obs,
                                   leading_observable=pred_lead,
                                   trajectory=None)


class InRETAINConfig(ModelConfig):
    mem_a: int = eqx.static_field(default=45)
    mem_b: int = eqx.static_field(default=45)
    lead_predictor: str = "monotonic"


class InRETAIN(InpatientModel):
    _f_gru_a: Callable
    _f_gru_b: Callable
    _f_att_a: Callable
    _f_att_b: Callable
    _f_lead_dec: Callable
    _f_obs_dec: Callable
    _f_dx_dec: Callable
    _f_init: Callable
    config: InRETAINConfig = eqx.static_field()

    def __init__(self, config: InRETAINConfig, schemes: Tuple[DatasetScheme],
                 demographic_vector_config: DemographicVectorConfig,
                 leading_observable_config: LeadingObservableExtractorConfig,
                 key: "jax.random.PRNGKey"):
        self._assert_demo_dim(config, schemes[1], demographic_vector_config)
        keys = jrandom.split(key, 9)

        f_emb = self._make_embedding(
            config=config,
            schemes=schemes,
            demographic_vector_config=demographic_vector_config,
            key=keys[0])
        f_dx_dec = self._make_dx_dec(config=config,
                                     dx_size=len(schemes[1].outcome),
                                     key=keys[1])

        self._f_lead_dec = InICENODE._make_lead_dec(
            config=config,
            input_size=config.emb.sequence,
            leading_observable_config=leading_observable_config,
            key=keys[2])
        self._f_obs_dec = self._make_obs_dec(config=config,
                                             obs_size=len(schemes[1].obs),
                                             key=keys[3])

        self._f_gru_a = eqx.nn.GRUCell(config.emb.sequence,
                                       config.mem_a,
                                       use_bias=True,
                                       key=keys[4])
        self._f_gru_b = eqx.nn.GRUCell(config.emb.sequence,
                                       config.mem_b,
                                       use_bias=True,
                                       key=keys[5])

        self._f_att_a = eqx.nn.Linear(config.mem_a,
                                      1,
                                      use_bias=True,
                                      key=keys[6])
        self._f_att_b = eqx.nn.Linear(config.mem_b,
                                      config.emb.sequence,
                                      use_bias=True,
                                      key=keys[7])

        self._f_init = self._make_init(config=config, key=keys[8])

        super().__init__(config=config, _f_emb=f_emb, _f_dx_dec=f_dx_dec)

    @staticmethod
    def _make_init(config, key):
        init_input_size = config.emb.demo + config.emb.dx
        return eqx.nn.MLP(init_input_size,
                          config.mem_a + config.mem_b + config.emb.sequence,
                          (config.mem_a + config.mem_b) * 3,
                          depth=2,
                          key=key)

    @staticmethod
    def _make_embedding(config, demographic_vector_config, schemes, key):
        return DeepMindPatientEmbedding(
            schemes=schemes,
            demographic_vector_config=demographic_vector_config,
            config=config.emb,
            key=key)

    @staticmethod
    def _make_dx_dec(config, dx_size, key):
        return eqx.nn.MLP(config.emb.sequence,
                          dx_size,
                          config.emb.dx * 5,
                          depth=1,
                          key=key)

    @staticmethod
    def _make_obs_dec(config, obs_size, key):
        return eqx.nn.MLP(config.emb.sequence,
                          obs_size,
                          obs_size * 5,
                          depth=1,
                          key=key)

    @property
    def dyn_params_list(self):
        return self.params_list(
            (self._f_gru_a, self._f_gru_b, self._f_att_a, self._f_att_b))

    @eqx.filter_jit
    def _gru_a(self, x, state):
        return self._f_gru_a(x, state)

    @eqx.filter_jit
    def _gru_b(self, x, state):
        return self._f_gru_b(x, state)

    @eqx.filter_jit
    def _att_a(self, x):
        return self._f_att_a(x)

    @eqx.filter_jit
    def _att_b(self, x):
        return self._f_att_b(x)

    @eqx.filter_jit
    def _dx_dec(self, x):
        return self._f_dx_dec(x)

    @eqx.filter_jit
    def _lead_dec(self, x):
        return self._f_lead_dec(x)

    @eqx.filter_jit
    def _obs_dec(self, x):
        return self._f_obs_dec(x)

    @eqx.filter_jit
    def _init_state(self, demo_emb, dx0):
        state = self._f_init(jnp.hstack((demo_emb, dx0)))
        splitter = (self.config.mem_a, self.config.mem_a + self.config.mem_b)
        state_a, state_b, context = jnp.hsplit(state, splitter)
        return state_a, state_b, jnp.tanh(self._att_b(state_b)) * context

    def __call__(
            self, admission: Admission,
            embedded_admission: DeepMindEmbeddedAdmission,
            precomputes: Precomputes, regularisation: ModelRegularisation,
            store_embeddings: Optional[TrajectoryConfig]
    ) -> AdmissionPrediction:

        obs = admission.observables
        lead = admission.leading_observable

        if len(obs) == 0:
            pred_obs = InpatientObservables.empty(obs.value.shape[1])
            pred_lead = InpatientObservables.empty(lead.value.shape[1])
            pred_dx = CodesVector(jnp.zeros(len(admission.outcome.scheme)),
                                  admission.outcome.scheme)
            return AdmissionPrediction(admission=admission,
                                       outcome=pred_dx,
                                       observables=pred_obs,
                                       leading_observable=pred_lead,
                                       trajectory=None)

        state_a0, state_b0, context = self._init_state(embedded_admission.demo,
                                                       embedded_admission.dx0)

        # step 1 @RETAIN paper

        # v1, v2, ..., vT
        # Merge controls with embeddings
        cv_seq = embedded_admission.sequence

        pred_obs_l = [self._obs_dec(context)]
        pred_lead_l = [self._lead_dec(context)]

        for i in range(1, len(cv_seq)):
            # e: i, ..., 1
            e_seq = []

            # beta: i, ..., 1
            b_seq = []

            state_a = state_a0
            state_b = state_b0
            for j in reversed(range(i)):
                # step 2 @RETAIN paper
                state_a = self._gru_a(cv_seq[j], state_a)
                e_j = self._att_a(state_a)
                # After the for-loop apply softmax on e_seq to get
                # alpha_seq

                e_seq.append(e_j)

                # step 3 @RETAIN paper
                h_j = state_b = self._gru_b(cv_seq[j], state_b)
                b_j = self._att_b(h_j)

                b_seq.append(jnp.tanh(b_j))

            # alpha: i, ..., 1
            a_seq = jax.nn.softmax(jnp.hstack(e_seq))

            # step 4 @RETAIN paper

            # v_i, ..., v_1
            context = cv_seq[:i][::-1]
            context = sum(a * (b * v)
                          for a, b, v in zip(a_seq, b_seq, context))

            # step 5 @RETAIN paper
            pred_obs_l.append(self._obs_dec(context))
            pred_lead_l.append(self._lead_dec(context))

        pred_obs = InpatientObservables(obs.time, jnp.vstack(pred_obs_l),
                                        obs.mask)
        pred_lead = InpatientObservables(lead.time, jnp.vstack(pred_lead_l),
                                         lead.mask)

        pred_dx = CodesVector(self._f_dx_dec(context),
                              admission.outcome.scheme)
        return AdmissionPrediction(admission=admission,
                                   outcome=pred_dx,
                                   observables=pred_obs,
                                   leading_observable=pred_lead,
                                   trajectory=None)


class InSKELKoopmanConfig(InICENODELiteConfig):
    pass


class InSKELKoopmanRegularisation(ModelRegularisation):
    L_rec: float = 1.0


class InSKELKoopmanPrecomputes(Precomputes):
    A: jnp.ndarray


class InSKELKoopman(InICENODELite):

    def __init__(self, config: InSKELKoopmanConfig,
                 schemes: Tuple[DatasetScheme],
                 demographic_vector_config: DemographicVectorConfig,
                 leading_observable_config: LeadingObservableExtractorConfig,
                 key: "jax.random.PRNGKey"):
        super().__init__(config=config,
                         schemes=schemes,
                         demographic_vector_config=demographic_vector_config,
                         leading_observable_config=leading_observable_config,
                         key=key)

    @staticmethod
    def _make_dyn(config, key):
        return SKELKoopmanOperator(input_size=config.state,
                                   control_size=config.emb.inp_proc_demo,
                                   koopman_size=config.state * 5,
                                   key=key)

    @eqx.filter_jit
    def _safe_integrate(self, delta, state, int_e, precomputes):
        second = jnp.array(1 / 3600.0)
        delta = jnp.where((delta < second) & (delta >= 0.0), second, delta)
        return self._f_dyn(x=state, t=delta, u=int_e, A=precomputes.A)

    def precomputes(self, *args, **kwargs):
        return InSKELKoopmanPrecomputes(A=self._f_dyn.compute_A())

    @property
    def dyn_params_list(self):
        return self.params_list((self._f_dyn.R, self._f_dyn.Q, self._f_dyn.N))

    def step_segment(self, state: jnp.ndarray, int_e: jnp.ndarray,
                     obs: InpatientObservables, lead: InpatientObservables,
                     t0: float, t1: float,
                     sampling_time: Optional[jnp.ndarray],
                     precomputes: InSKELKoopmanPrecomputes):
        preds = []
        lead_preds = []
        trajectory = []
        rec_loss = []
        t = t0
        for t_obs, val, mask in zip(obs.time, obs.value, obs.mask):
            # if time-diff is more than 1 seconds, we integrate.
            rec_loss.append(self._f_dyn.compute_phi_loss(x=state, u=int_e))
            trajectory.extend(
                self.sample_state_trajectory(state, int_e, t, t_obs,
                                             sampling_time, precomputes))

            state = self._safe_integrate(t_obs - t, state, int_e, precomputes)
            pred_obs = self._f_obs_dec(state)
            pred_lead = self._f_lead_dec(state)
            state = self._f_update(state, pred_obs, val, mask)
            t = t_obs
            preds.append(pred_obs)
            lead_preds.append(pred_lead)

        rec_loss.append(self._f_dyn.compute_phi_loss(x=state, u=int_e))
        trajectory.extend(
            self.sample_state_trajectory(state, int_e, t, t1, sampling_time,
                                         precomputes))
        state = self._safe_integrate(t1 - t, state, int_e, precomputes)

        if len(preds) > 0:
            pred_obs_val = jnp.vstack(preds)
            pred_lead_val = jnp.vstack(lead_preds)
        else:
            pred_obs_val = jnp.empty_like(obs.value)
            pred_lead_val = jnp.empty_like(lead.value)

        return state, (InpatientObservables(obs.time, pred_obs_val, obs.mask),
                       InpatientObservables(lead.time, pred_lead_val,
                                            lead.mask)), rec_loss, trajectory

    def __call__(
            self, admission: Admission,
            embedded_admission: EmbeddedInAdmission, precomputes: Precomputes,
            regularisation: InSKELKoopmanRegularisation,
            store_embeddings: Optional[TrajectoryConfig]
    ) -> AdmissionPrediction:
        int_e = embedded_admission.inp_proc_demo

        state = self._init_state(int_demo_emb=int_e[0],
                                 dx0=embedded_admission.dx0)

        obs = admission.observables
        lead = admission.leading_observable
        pred_obs_l = []
        pred_lead_l = []
        rec_loss = []
        t0 = admission.interventions.t0
        t1 = admission.interventions.t1
        trajectory = [state]
        if store_embeddings is not None:
            dt = store_embeddings.sampling_rate
            sampling_time = jnp.arange(0.0, t1[-1], dt)
        else:
            sampling_time = None
        for i in range(len(t0)):
            state, pred, rloss, traj = self.step_segment(
                state, int_e[i], obs[i], lead[i], t0[i], t1[i], sampling_time,
                precomputes)
            pred_obs, pred_lead = pred
            rec_loss.extend(rloss)
            pred_obs_l.append(pred_obs)
            pred_lead_l.append(pred_lead)
            trajectory.extend(traj)

        if len(trajectory) > 0:
            trajectory = PatientTrajectory(time=sampling_time,
                                           value=jnp.vstack(trajectory))
        else:
            trajectory = None
        return AdmissionPrediction(admission=admission,
                                   outcome=None,
                                   observables=pred_obs_l,
                                   leading_observable=pred_lead_l,
                                   trajectory=trajectory,
                                   auxiliary_loss={'L_rec': rec_loss})

    @eqx.filter_jit
    def pathwise_params_stats(self):
        stats = super().pathwise_params_stats()
        real_eig_A, imag_eig_A = self._f_dyn.compute_A_spectrum()
        desc = [f'_f_dyn.A.lam_{i}' for i in range(real_eig_A.shape[0])]
        stats.update({
            k: {
                'real': lamr,
                'imag': lami
            }
            for k, lamr, lami in zip(desc, real_eig_A, imag_eig_A)
        })
        for component, v in (('real', real_eig_A), ('imag', imag_eig_A)):
            stats.update({
                f'_f_dyn.A.{component}_lam': {
                    'mean': jnp.nanmean(v),
                    'std': jnp.nanstd(v),
                    'min': jnp.nanmin(v),
                    'max': jnp.nanmax(v),
                    'l1': jnp.abs(v).sum(),
                    'l2': jnp.square(v).sum(),
                    'nans': jnp.isnan(v).sum(),
                }
            })
        return stats


class InVanillaKoopmanConfig(InSKELKoopmanConfig):
    pass


class InVanillaKoopmanRegularisation(InSKELKoopmanRegularisation):
    pass


class InVanillaKoopmanPrecomputes(InSKELKoopmanPrecomputes):
    pass


class InVanillaKoopman(InSKELKoopman):

    @staticmethod
    def _make_dyn(config, key):
        return VanillaKoopmanOperator(input_size=config.state,
                                      control_size=config.emb.inp_proc_demo,
                                      koopman_size=config.state * 5,
                                      key=key)
