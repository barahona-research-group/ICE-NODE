"""."""
from __future__ import annotations
from typing import Callable, Tuple
import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jrandom
import equinox as eqx

from ..utils import model_params_scaler
from ..ehr import (Admission, InpatientObservables, AdmissionPrediction,
                   DatasetScheme, DemographicVectorConfig, CodesVector,
                   LeadingObservableConfig, PatientTrajectory)
from .embeddings import (InpatientEmbedding, InpatientEmbeddingConfig,
                         EmbeddedInAdmission)

from .model import InpatientModel, ModelConfig, ModelRegularisation
from .base_models import (ObsStateUpdate, NeuralODE_JAX)


class InICENODEConfig(ModelConfig):
    mem: int = 15
    obs: int = 25
    lead: int = 5
    lead_predictor: str = "monotonic"


class InICENODERegularisation(ModelRegularisation):
    L_taylor: float = 0.0
    taylor_order: int = 0


class MonotonicLeadingObsPredictor(eqx.Module):
    _mlp: eqx.nn.MLP

    def __init__(self, config: InICENODEConfig,
                 leading_observable_config: LeadingObservableConfig,
                 key: jax.random.PRNGKey):
        out_size = len(leading_observable_config.leading_hours) + 1
        self._mlp = eqx.nn.MLP(config.lead,
                               out_size,
                               config.lead * 5,
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

    def __init__(self, config: InICENODEConfig,
                 leading_observable_config: LeadingObservableConfig,
                 key: jax.random.PRNGKey):
        self._t = jnp.array(leading_observable_config.leading_hours)
        self._mlp = eqx.nn.MLP(config.lead,
                               4,
                               config.lead * 5,
                               depth=3,
                               key=key)

    def __call__(self, lead):
        [vscale, hscale, vshift, hshift] = self._mlp(lead)
        return vscale * jnn.sigmoid(hscale * (self._t - hshift)) + vshift


class MLPLeadingObsPredictor(eqx.Module):
    _mlp: eqx.nn.MLP

    def __init__(self, config: InICENODEConfig,
                 leading_observable_config: LeadingObservableConfig,
                 key: jax.random.PRNGKey):
        self._mlp = eqx.nn.MLP(config.lead,
                               len(leading_observable_config.leading_hours),
                               config.lead * 5,
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
                 leading_observable_config: LeadingObservableConfig,
                 key: "jax.random.PRNGKey"):
        self._assert_demo_dim(config, schemes[1], demographic_vector_config)
        (emb_key, obs_dec_key, lead_key, dx_dec_key, dyn_key,
         update_key) = jrandom.split(key, 6)
        f_emb = InpatientEmbedding(
            schemes=schemes,
            demographic_vector_config=demographic_vector_config,
            config=config.emb,
            key=emb_key)
        f_dx_dec = eqx.nn.MLP(config.emb.dx,
                              len(schemes[1].outcome),
                              config.emb.dx * 5,
                              depth=1,
                              key=dx_dec_key)

        self._f_obs_dec = eqx.nn.MLP(config.obs,
                                     len(schemes[1].obs),
                                     config.obs * 5,
                                     depth=1,
                                     key=obs_dec_key)
        if config.lead_predictor == "monotonic":
            self._f_lead_dec = MonotonicLeadingObsPredictor(
                config, leading_observable_config, key=lead_key)
        elif config.lead_predictor == "sigmoid":
            self._f_lead_dec = SigmoidLeadingObsPredictor(
                config, leading_observable_config, key=lead_key)
        elif config.lead_predictor == "mlp":
            self._f_lead_dec = MLPLeadingObsPredictor(
                config, leading_observable_config, key=lead_key)
        else:
            raise ValueError(
                f"Unknown leading predictor type: {config.lead_predictor}")

        dyn_state_size = config.obs + config.emb.dx + config.mem + config.lead
        dyn_input_size = dyn_state_size + config.emb.inp_proc_demo

        f_dyn = eqx.nn.MLP(in_size=dyn_input_size,
                           out_size=dyn_state_size,
                           activation=jnn.tanh,
                           depth=2,
                           width_size=dyn_state_size * 5,
                           key=dyn_key)
        f_dyn = model_params_scaler(f_dyn, 1e-2, eqx.is_inexact_array)
        self._f_dyn = NeuralODE_JAX(f_dyn, timescale=1.0)
        self._f_update = ObsStateUpdate(dyn_state_size,
                                        len(schemes[1].obs),
                                        key=update_key)

        super().__init__(config=config, _f_emb=f_emb, _f_dx_dec=f_dx_dec)

    @eqx.filter_jit
    def join_state(self, mem, obs, lead, dx):
        if mem is None:
            mem = jnp.zeros((self.config.mem, ))
        if obs is None:
            obs = jnp.zeros((self.config.obs, ))
        if lead is None:
            lead = jnp.zeros((self.config.lead, ))

        return jnp.hstack((mem, obs, lead, dx))

    @eqx.filter_jit
    def split_state(self, state: jnp.ndarray):
        s1 = self.config.mem
        s2 = self.config.mem + self.config.obs
        s3 = self.config.mem + self.config.obs + self.config.lead
        return jnp.hsplit(state, (s1, s2, s3))

    @eqx.filter_jit
    def _safe_integrate(self, delta, state, int_e):
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

    def step_segment(self, state: jnp.ndarray, int_e: jnp.ndarray,
                     obs: InpatientObservables, lead: InpatientObservables,
                     t0: float, t1: float):
        preds = []
        lead_preds = []
        t = t0
        for t_obs, val, mask in zip(obs.time, obs.value, obs.mask):
            # if time-diff is more than 1 seconds, we integrate.
            state = self._safe_integrate(t_obs - t, state, int_e)
            state_components = self.split_state(state)
            obs_e, lead_e = state_components[1], state_components[2]
            pred_obs = self._f_obs_dec(obs_e)
            pred_lead = self._f_lead_dec(lead_e)
            state = self._f_update(state, pred_obs, val, mask)
            t = t_obs
            preds.append(pred_obs)
            lead_preds.append(pred_lead)

        state = self._safe_integrate(t1 - t, state, int_e)

        if len(preds) > 0:
            pred_obs_val = jnp.vstack(preds)
            pred_lead_val = jnp.vstack(lead_preds)
        else:
            pred_obs_val = jnp.empty_like(obs.value)
            pred_lead_val = jnp.empty_like(lead.value)

        return state, (InpatientObservables(obs.time, pred_obs_val, obs.mask),
                       InpatientObservables(lead.time, pred_lead_val,
                                            lead.mask))

    def __call__(self, admission: Admission,
                 embedded_admission: EmbeddedInAdmission,
                 regularisation: InICENODERegularisation,
                 store_embeddings: bool) -> AdmissionPrediction:
        state = self.join_state(None, None, None, embedded_admission.dx0)
        int_e = embedded_admission.inp_proc_demo
        obs = admission.observables
        lead = admission.leading_observable
        pred_obs_l = []
        pred_lead_l = []
        trajectory_l = []
        t0 = admission.interventions.t0
        t1 = admission.interventions.t1
        for i in range(len(t0)):
            state, (pred_obs,
                    pred_lead) = self.step_segment(state, int_e[i], obs[i],
                                                   lead[i], t0[i], t1[i])

            pred_obs_l.append(pred_obs)
            pred_lead_l.append(pred_lead)
            trajectory_l.append(PatientTrajectory(time=t1[i], state=state))

        pred_dx = CodesVector(self._f_dx_dec(self.split_state(state)[3]),
                              admission.outcome.scheme)

        return AdmissionPrediction(admission=admission,
                                   outcome=pred_dx,
                                   observables=pred_obs_l,
                                   leading_observable=pred_lead_l,
                                   trajectory=trajectory_l)

    @property
    def dyn_params_list(self):
        return self.params_list(self._f_dyn)


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

    config: InICENODEConfig = eqx.static_field()

    def __init__(self, config: InICENODEConfig, schemes: Tuple[DatasetScheme],
                 demographic_vector_config: DemographicVectorConfig,
                 leading_observable_config: LeadingObservableConfig,
                 key: "jax.random.PRNGKey"):
        super_key, init_key, dyn_key, update_key = jrandom.split(key, 4)
        super().__init__(config, schemes, demographic_vector_config,
                         leading_observable_config, super_key)

        self._f_dx_dec = None

        init_input_size = config.emb.inp_proc_demo + config.emb.dx
        state_size = config.mem + config.obs + config.lead
        dyn_input_size = state_size + config.emb.inp_proc_demo

        f_dyn = eqx.nn.MLP(in_size=dyn_input_size,
                           out_size=state_size,
                           activation=jnn.tanh,
                           depth=2,
                           width_size=state_size * 5,
                           key=dyn_key)
        f_dyn = model_params_scaler(f_dyn, 1e-2, eqx.is_inexact_array)
        self._f_dyn = NeuralODE_JAX(f_dyn, timescale=1.0)
        self._f_init = eqx.nn.MLP(init_input_size,
                                  state_size,
                                  config.emb.dx * 5,
                                  depth=2,
                                  key=init_key)
        self._f_update = ObsStateUpdate(state_size,
                                        len(schemes[1].obs),
                                        key=update_key)

    @eqx.filter_jit
    def join_state(self, mem, obs, lead, int_demo_emb=None, dx0=None):
        if mem is None or obs is None or lead is None:
            return self._f_init(jnp.hstack((int_demo_emb, dx0)))

        return jnp.hstack((mem, obs, lead))

    @eqx.filter_jit
    def split_state(self, state: jnp.ndarray):
        s1 = self.config.mem
        s2 = self.config.mem + self.config.obs
        return jnp.hsplit(state, (s1, s2))

    def __call__(self, admission: Admission,
                 embedded_admission: EmbeddedInAdmission,
                 regularisation: InICENODERegularisation,
                 store_embeddings: bool) -> AdmissionPrediction:
        int_e = embedded_admission.inp_proc_demo

        state = self.join_state(None,
                                None,
                                None,
                                int_demo_emb=int_e[0],
                                dx0=embedded_admission.dx0)

        obs = admission.observables
        lead = admission.leading_observable
        pred_obs_l = []
        pred_lead_l = []
        # trajectory_l = []
        t0 = admission.interventions.t0
        t1 = admission.interventions.t1
        for i in range(len(t0)):
            state, (pred_obs,
                    pred_lead) = self.step_segment(state, int_e[i], obs[i],
                                                   lead[i], t0[i], t1[i])

            pred_obs_l.append(pred_obs)
            pred_lead_l.append(pred_lead)
            # trajectory_l.append(PatientTrajectory(time=t1[i], state=state))

        return AdmissionPrediction(admission=admission,
                                   outcome=None,
                                   observables=pred_obs_l,
                                   leading_observable=pred_lead_l,
                                   trajectory=None)
        # trajectory=trajectory_l)
