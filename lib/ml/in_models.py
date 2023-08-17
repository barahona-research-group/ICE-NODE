"""."""
from __future__ import annotations
from typing import Callable
import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jrandom
import equinox as eqx

from ..utils import model_params_scaler
from ..ehr import (Admission, InpatientObservables, AdmissionPrediction,
                   DatasetScheme, DemographicVectorConfig,
                   CodesVector)
from .embeddings import (InpatientEmbedding, InpatientEmbeddingDimensions,
                         EmbeddedInAdmission)

from .model import InpatientModel, ModelDimensions
from .base_models import (ObsStateUpdate, NeuralODE_JAX)


class InICENODEDimensions(ModelDimensions):
    mem: int = 15
    dx: int = 30
    obs: int = 25

    def __init__(self, emb: InpatientEmbeddingDimensions, mem: int, obs: int):
        super().__init__(emb=emb)
        self.emb = emb
        self.mem = mem
        self.dx = emb.dx
        self.obs = obs


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
    f_dx_dec: Callable
    f_dyn: Callable
    f_update: Callable

    scheme: DatasetScheme = eqx.static_field()
    dims: InICENODEDimensions = eqx.static_field()
    demographic_vector_config: DemographicVectorConfig = eqx.static_field()

    def __init__(self, dims: InICENODEDimensions,
                 source_scheme: DatasetScheme, target_scheme: DatasetScheme,
                 demographic_vector_config: DemographicVectorConfig,
                 key: "jax.random.PRNGKey"):
        (emb_key, obs_dec_key, dx_dec_key, dyn_key,
         update_key) = jrandom.split(key, 5)
        f_emb = InpatientEmbedding(
            source_scheme=source_scheme,
            target_scheme=target_scheme,
            demographic_vector_config=demographic_vector_config,
            dims=dims.emb,
            key=emb_key)
        f_dx_dec = eqx.nn.MLP(dims.dx,
                              len(target_scheme.outcome),
                              dims.dx * 5,
                              depth=1,
                              key=dx_dec_key)

        self.f_obs_dec = eqx.nn.MLP(dims.obs,
                                    len(target_scheme.obs),
                                    dims.obs * 5,
                                    depth=1,
                                    key=obs_dec_key)
        dyn_state_size = dims.obs + dims.dx + dims.mem
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
                                       len(target_scheme.obs),
                                       key=update_key)

        super().__init__(dims=dims,
                         source_scheme=source_scheme,
                         target_scheme=target_scheme,
                         demographic_vector_config=demographic_vector_config,
                         f_emb=f_emb,
                         f_dx_dec=f_dx_dec)

    @eqx.filter_jit
    def join_state(self, mem, obs, dx):
        if mem is None:
            mem = jnp.zeros((self.dims.mem, ))
        if obs is None:
            obs = jnp.zeros((self.dims.obs, ))
        return jnp.hstack((mem, obs, dx))

    @eqx.filter_jit
    def split_state(self, state: jnp.ndarray):
        s1 = self.dims.mem
        s2 = self.dims.mem + self.dims.obs
        return jnp.hsplit(state, (s1, s2))

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
                     obs: InpatientObservables, t0: float, t1: float):
        preds = []
        t = t0
        for t_obs, val, mask in zip(obs.time, obs.value, obs.mask):
            # if time-diff is more than 1 seconds, we integrate.
            state = self._safe_integrate(t_obs - t, state, int_e)
            _, obs_e, _ = self.split_state(state)
            pred_obs = self.f_obs_dec(obs_e)
            state = self.f_update(state, pred_obs, val, mask)
            t = t_obs
            preds.append(pred_obs)

        state = self._safe_integrate(t1 - t, state, int_e)

        if len(preds) > 0:
            pred_obs_val = jnp.vstack(preds)
        else:
            pred_obs_val = jnp.empty_like(obs.value)

        return state, InpatientObservables(obs.time, pred_obs_val, obs.mask)

    def __call__(self, admission: Admission,
                 embedded_admission: EmbeddedInAdmission) -> AdmissionPrediction:
        state = self.join_state(None, None, embedded_admission.dx0)
        int_e = embedded_admission.inp_proc_demo
        obs = admission.observables
        pred_obs_l = []
        t0 = admission.interventions.t0
        t1 = admission.interventions.t1
        for i in range(len(t0)):
            t = t0[i], t1[i]
            state, pred_obs = self.step_segment(state, int_e[i], obs[i], *t)
            pred_obs_l.append(pred_obs)
        pred_dx = CodesVector(self.f_dx_dec(self.split_state(state)[2]),
                              admission.outcome.scheme)
        return AdmissionPrediction(admission=admission,
                                   outcome=pred_dx,
                                   observables=pred_obs_l)
