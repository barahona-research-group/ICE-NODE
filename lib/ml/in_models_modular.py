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
                         EmbeddedInAdmission, InpatientLiteEmbedding,
                         LiteEmbeddedInAdmission, DeepMindPatientEmbedding,
                         DeepMindPatientEmbeddingConfig,
                         DeepMindEmbeddedAdmission)

from .model import InpatientModel, ModelConfig, ModelRegularisation
from .in_models import InICENODE, InICENODERegularisation
from .base_models import (ObsStateUpdate, NeuralODE_JAX)
from ..base import Data, Config


class InModularICENODEConfig(ModelConfig):
    mem: int = 15
    obs: int = 25
    lead: int = 5
    lead_predictor: str = "monotonic"

    @property
    def state_size(self):
        return self.mem + self.obs + self.lead + self.emb.dx

    @property
    def state_splitter(self):
        s1 = self.mem
        s2 = s1 + self.obs
        s3 = s2 + self.lead
        return (s1, s2, s3)


class InModularICENODE(InICENODE):
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

    config: InModularICENODEConfig = eqx.static_field()

    def __init__(self, config: InModularICENODEConfig,
                 schemes: Tuple[DatasetScheme],
                 demographic_vector_config: DemographicVectorConfig,
                 leading_observable_config: LeadingObservableConfig,
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
            input_size=config.lead,
            leading_observable_config=leading_observable_config,
            key=lead_key)
        self._f_dyn = self._make_dyn(config=config, key=dyn_key)
        self._f_update = self._make_update(config=config,
                                           obs_size=len(schemes[1].obs),
                                           key=update_key)

        InpatientModel.__init__(self,
                                config=config,
                                _f_emb=f_emb,
                                _f_dx_dec=f_dx_dec)

    @staticmethod
    def _make_dyn(config, key):
        dyn_input_size = config.state_size + config.emb.inp_proc_demo
        f_dyn = eqx.nn.MLP(in_size=dyn_input_size,
                           out_size=config.state_size,
                           activation=jnn.tanh,
                           depth=2,
                           width_size=config.state_size * 5,
                           key=key)
        f_dyn = model_params_scaler(f_dyn, 1e-2, eqx.is_inexact_array)
        return NeuralODE_JAX(f_dyn, timescale=1.0)

    @staticmethod
    def _make_update(config, obs_size, key):
        return ObsStateUpdate(config.state_size, obs_size, key=key)

    @staticmethod
    def _make_obs_dec(config, obs_size, key):
        return eqx.nn.MLP(config.obs,
                          obs_size,
                          config.obs * 5,
                          depth=1,
                          key=key)

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
        return jnp.hsplit(state, self.config.state_splitter)

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


class InModularICENODELiteConfig(InModularICENODEConfig):

    @property
    def state_size(self):
        return self.mem + self.obs + self.lead

    @property
    def state_splitter(self):
        s1 = self.mem
        s2 = s1 + self.obs
        return (s1, s2)


class InModularICENODELite(InModularICENODE):
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

    config: InModularICENODELiteConfig = eqx.static_field()

    def __init__(self, config: InModularICENODELiteConfig,
                 schemes: Tuple[DatasetScheme],
                 demographic_vector_config: DemographicVectorConfig,
                 leading_observable_config: LeadingObservableConfig,
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
                          config.state_size,
                          config.emb.dx * 5,
                          depth=2,
                          key=key)

    @eqx.filter_jit
    def join_state(self, mem, obs, lead, int_demo_emb=None, dx0=None):
        if mem is None or obs is None or lead is None:
            return self._f_init(jnp.hstack((int_demo_emb, dx0)))

        return jnp.hstack((mem, obs, lead))

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


class InModularGRUConfig(InModularICENODELiteConfig):
    pass

class InModularGRUJump(InModularICENODELite):
    # TODO: as for the original paper, use multi-layer embeddings with skip
    # connections.
    _f_emb: Callable[[Admission], EmbeddedInAdmission]
    _f_obs_dec: Callable
    _f_lead_dec: Callable
    _f_init: Callable
    # GRU. Alternatives: SRU, LSTM, ..
    _f_update: Callable
    _f_dyn: Callable

    config: InModularGRUConfig = eqx.static_field()

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
                              hidden_size=config.state_size,
                              key=key)

    @staticmethod
    def _make_init(config, key):
        init_input_size = config.emb.demo + config.emb.dx
        return eqx.nn.MLP(init_input_size,
                          config.state_size,
                          config.emb.dx * 5,
                          depth=2,
                          key=key)

    @eqx.filter_jit
    def join_state(self, mem, obs, lead, demo_emb=None, dx0=None):
        if mem is None or obs is None or lead is None:
            return self._f_init(jnp.hstack((demo_emb, dx0)))

        return jnp.hstack((mem, obs, lead))

    def __call__(self, admission: Admission,
                 embedded_admission: LiteEmbeddedInAdmission,
                 regularisation: InICENODERegularisation,
                 store_embeddings: bool) -> AdmissionPrediction:
        demo_e = embedded_admission.demo

        state = self.join_state(None,
                                None,
                                None,
                                demo_emb=embedded_admission.demo,
                                dx0=embedded_admission.dx0)

        obs = admission.observables
        lead = admission.leading_observable
        pred_obs_l = []
        pred_lead_l = []
        for i in range(len(obs.time)):
            state = self._f_dyn(demo_e, state)
            state_components = self.split_state(state)
            obs_e, lead_e = state_components[1], state_components[2]
            pred_obs = self._f_obs_dec(obs_e)
            pred_lead = self._f_lead_dec(lead_e)
            state = self._f_update(state, pred_obs, obs.value[i], obs.mask[i])
            pred_obs_l.append(pred_obs)
            pred_lead_l.append(pred_lead)

        if len(obs) == 0:
            pred_obs = InpatientObservables.empty(obs.value.shape[1])
            pred_lead = InpatientObservables.empty(lead.value.shape[1])
        else:
            pred_obs = InpatientObservables(obs.time, jnp.vstack(pred_obs_l),
                                            obs.mask)
            pred_lead = InpatientObservables(lead.time,
                                             jnp.vstack(pred_lead_l),
                                             lead.mask)

        return AdmissionPrediction(admission=admission,
                                   outcome=None,
                                   observables=pred_obs,
                                   leading_observable=pred_lead,
                                   trajectory=None)


class InModularGRU(InModularGRUJump):

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
                              hidden_size=config.state_size,
                              key=key)

    def __call__(self, admission: Admission,
                 embedded_admission: DeepMindEmbeddedAdmission,
                 regularisation: InICENODERegularisation,
                 store_embeddings: bool) -> AdmissionPrediction:
        state = self.join_state(None,
                                None,
                                None,
                                demo_emb=embedded_admission.demo,
                                dx0=embedded_admission.dx0)
        sequence_e = embedded_admission.sequence
        pred_obs_l = []
        pred_lead_l = []
        for seq_e in sequence_e:
            state_components = self.split_state(state)
            obs_e, lead_e = state_components[1], state_components[2]
            pred_obs = self._f_obs_dec(obs_e)
            pred_lead = self._f_lead_dec(lead_e)
            state = self._f_dyn(seq_e, state)
            pred_obs_l.append(pred_obs)
            pred_lead_l.append(pred_lead)

        obs = admission.observables
        lead = admission.leading_observable
        if len(obs) == 0:
            pred_obs = InpatientObservables.empty(obs.value.shape[1])
            pred_lead = InpatientObservables.empty(lead.value.shape[1])
        else:
            pred_obs = InpatientObservables(obs.time, jnp.vstack(pred_obs_l),
                                            obs.mask)
            pred_lead = InpatientObservables(lead.time,
                                             jnp.vstack(pred_lead_l),
                                             lead.mask)

        return AdmissionPrediction(admission=admission,
                                   outcome=None,
                                   observables=pred_obs,
                                   leading_observable=pred_lead,
                                   trajectory=None)
