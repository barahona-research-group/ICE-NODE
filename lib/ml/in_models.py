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
                   LeadingObservableConfig)
from .embeddings import (InpatientEmbedding, InpatientEmbeddingDimensions,
                         EmbeddedInAdmission)

from .model import InpatientModel, ModelDimensions
from .base_models import (ObsStateUpdate, NeuralODE_JAX)


class InICENODEDimensions(ModelDimensions):
    mem: int = 15
    obs: int = 25

    def __init__(self, emb: InpatientEmbeddingDimensions, mem: int, obs: int,
                 leading_observable_config: LeadingObservableConfig):
        super().__init__(emb=emb)
        self.emb = emb
        self.mem = mem
        self.obs = obs
        self.lead = len(leading_observable_config.leading_hours)


class LeadingObsPredictor(eqx.Module):
    mlp: eqx.nn.MLP
    dims: InICENODEDimensions = eqx.static_field()

    def __init__(self, dims: InICENODEDimensions, key: jax.random.PRNGKey):
        self.dims = dims
        self.mlp = eqx.nn.MLP(dims.lead,
                              dims.lead + 1,
                              dims.lead * 5,
                              depth=2,
                              key=key)

    def __call__(self, lead):
        y = self.mlp(lead)
        risk = y[-1]
        p = jnp.cumsum(jnn.softmax(y[:-1]))
        return risk * p


class InICENODE(InpatientModel):
    """
    The InICENODE model. It is composed of the following components:
        - f_emb: Embedding function.
        - f_obs_dec: Observation decoder.
        - f_dx_dec: Discharge codes decoder.
        - f_dyn: Dynamics function.
        - f_update: Update function.
    """
    f_emb: Callable[[Admission], EmbeddedInAdmission]
    f_obs_dec: Callable
    f_lead_dec: Callable
    f_dx_dec: Callable
    f_dyn: Callable
    f_update: Callable

    schemes: Tuple[DatasetScheme] = eqx.static_field()
    dims: InICENODEDimensions = eqx.static_field()
    demographic_vector_config: DemographicVectorConfig = eqx.static_field()
    leading_observable_config: LeadingObservableConfig = eqx.static_field()

    def __init__(self, dims: InICENODEDimensions,
                 schemes: Tuple[DatasetScheme],
                 demographic_vector_config: DemographicVectorConfig,
                 leading_observable_config: LeadingObservableConfig,
                 key: "jax.random.PRNGKey"):
        self.leading_observable_config = leading_observable_config
        self._assert_demo_dim(dims, schemes[1], demographic_vector_config)
        (emb_key, obs_dec_key, lead_key, dx_dec_key, dyn_key,
         update_key) = jrandom.split(key, 5)
        f_emb = InpatientEmbedding(
            schemes=schemes,
            demographic_vector_config=demographic_vector_config,
            dims=dims.emb,
            key=emb_key)
        f_dx_dec = eqx.nn.MLP(dims.emb.dx,
                              len(schemes[1].outcome),
                              dims.emb.dx * 5,
                              depth=1,
                              key=dx_dec_key)

        self.f_obs_dec = eqx.nn.MLP(dims.obs,
                                    len(schemes[1].obs),
                                    dims.obs * 5,
                                    depth=1,
                                    key=obs_dec_key)
        self.f_lead_dec = LeadingObsPredictor(dims, key=lead_key)

        dyn_state_size = dims.obs + dims.emb.dx + dims.mem
        dyn_input_size = dyn_state_size + dims.emb.inp_proc_demo

        f_dyn = eqx.nn.MLP(in_size=dyn_input_size,
                           out_size=dyn_state_size,
                           activation=jnn.tanh,
                           depth=2,
                           width_size=dyn_state_size * 5,
                           key=dyn_key)
        f_dyn = model_params_scaler(f_dyn, 1e-2, eqx.is_inexact_array)
        self.f_dyn = NeuralODE_JAX(f_dyn, timescale=1.0)
        self.f_update = ObsStateUpdate(dyn_state_size,
                                       len(schemes[1].obs),
                                       key=update_key)

        super().__init__(dims=dims,
                         schemes=schemes,
                         demographic_vector_config=demographic_vector_config,
                         f_emb=f_emb,
                         f_dx_dec=f_dx_dec)

    @eqx.filter_jit
    def join_state(self, mem, obs, lead, dx):
        if mem is None:
            mem = jnp.zeros((self.dims.mem, ))
        if obs is None:
            obs = jnp.zeros((self.dims.obs, ))
        if lead is None:
            lead = jnp.zeros((self.dims.lead, ))

        return jnp.hstack((mem, obs, lead, dx))

    @eqx.filter_jit
    def split_state(self, state: jnp.ndarray):
        s1 = self.dims.mem
        s2 = self.dims.mem + self.dims.obs
        s3 = self.dims.mem + self.dims.obs + self.dims.lead
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
        return self.f_dyn(delta, state, args=dict(control=int_e))[-1]

    def step_segment(self, state: jnp.ndarray, int_e: jnp.ndarray,
                     obs: InpatientObservables, lead: InpatientObservables,
                     t0: float, t1: float):
        preds = []
        lead_preds = []
        t = t0
        for t_obs, val, mask in zip(obs.time, obs.value, obs.mask):
            # if time-diff is more than 1 seconds, we integrate.
            state = self._safe_integrate(t_obs - t, state, int_e)
            _, obs_e, lead_e, _ = self.split_state(state)
            pred_obs = self.f_obs_dec(obs_e)
            pred_lead = self.f_lead_dec(lead_e)
            state = self.f_update(state, pred_obs, val, mask)
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

    def __call__(
            self, admission: Admission,
            embedded_admission: EmbeddedInAdmission) -> AdmissionPrediction:
        state = self.join_state(None, None, None, embedded_admission.dx0)
        int_e = embedded_admission.inp_proc_demo
        obs = admission.observables
        lead = admission.leading_observable
        pred_obs_l = []
        pred_lead_l = []
        t0 = admission.interventions.t0
        t1 = admission.interventions.t1
        for i in range(len(t0)):
            state, (pred_obs, pred_lead) = self.step_segment(
                state, int_e[i], obs[i], lead[i], t0[i], t1[i])

            pred_obs_l.append(pred_obs)
            pred_lead_l.append(pred_lead)

        pred_dx = CodesVector(self.f_dx_dec(self.split_state(state)[2]),
                              admission.outcome.scheme)
        return AdmissionPrediction(admission=admission,
                                   outcome=pred_dx,
                                   observables=pred_obs_l,
                                   leading_observable=pred_lead_l)
