"""."""
from __future__ import annotations
from typing import List, Callable

from absl import logging
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx

from ..utils import model_params_scaler
from ..ehr import (Patient, AdmissionPrediction, DemographicVectorConfig,
                   MIMICDatasetScheme, CodesVector)
from ..embeddings import (PatientEmbedding, PatientEmbeddingDimensions,
                          EmbeddedAdmission)

from .base_models import (StateUpdate, NeuralODE_JAX)
from .abstract_model import OutpatientModel, ModelDimensions


class ICENODEDimensions(ModelDimensions):
    mem: int = 15
    dx: int = 30

    def __init__(self, emb: PatientEmbeddingDimensions, mem: int):
        super().__init__(emb=emb)
        self.emb = emb
        self.mem = mem
        self.dx = emb.dx


#     @classmethod
#     def sample_model_config(cls, trial: optuna.Trial):
#         return {'state_size': trial.suggest_int('s', 10, 100, 10)}


class ICENODE(OutpatientModel):
    f_dyn: Callable
    f_update: Callable

    def __init__(self, dims: ICENODEDimensions, scheme: MIMICDatasetScheme,
                 demographic_vector_config: DemographicVectorConfig,
                 key: "jax.random.PRNGKey"):
        (emb_key, dx_dec_key, dyn_key, up_key) = jrandom.split(key, 4)
        f_emb = PatientEmbedding(
            scheme=scheme,
            demographic_vector_config=demographic_vector_config,
            dims=dims.emb,
            key=emb_key)
        f_dx_dec = eqx.nn.MLP(dims.dx,
                              len(scheme.outcome),
                              dims.dx * 5,
                              depth=1,
                              key=dx_dec_key)

        dyn_state_size = dims.dx + dims.mem
        dyn_input_size = dyn_state_size + dims.emb.demo
        f_dyn = eqx.nn.MLP(in_size=dyn_input_size,
                           out_size=dyn_state_size,
                           activation=jnn.tanh,
                           depth=2,
                           width_size=dyn_state_size * 5,
                           key=dyn_key)
        ode_dyn_f = model_params_scaler(f_dyn, 1e-3, eqx.is_inexact_array)

        self.ode_dyn = NeuralODE_JAX(ode_dyn_f, timescale=1.0)

        self.f_update = StateUpdate(state_size=dims.mem,
                                    embeddings_size=dims.emb.dx,
                                    key=up_key)

        super().__init__(dims=dims,
                         scheme=scheme,
                         demographic_vector_config=demographic_vector_config,
                         f_emb=f_emb,
                         f_dx_dec=f_dx_dec)

    def join_state_emb(self, state, emb):
        if state is None:
            state = jnp.zeros((self.dims.mem, ))
        return jnp.hstack((state, emb))

    def split_state_emb(self, state: jnp.ndarray):
        return jnp.hsplit(state, (self.dims.mem, ))

    def _integrate(self, state: jnp.ndarray, int_time: jnp.ndarray,
                   ctrl: jnp.ndarray):
        try:
            return self.ode_dyn(int_time, state, args=dict(control=ctrl))[-1]
        except Exception:
            dt = float(jax.block_until_ready(int_time))
            if dt < 1 / 3600.0 and dt > 0.0:
                logging.debug(f"Time diff is less than 1 second: {int_time}")
            else:
                logging.error(f"Time diff is {int_time}!")
            return state

    @eqx.filter_jit
    def _update(self, *args):
        return self.f_update(*args)

    @eqx.filter_jit
    def _decode(self, dx_e: jnp.ndarray):
        return self.f_dx_dec(dx_e)

    @staticmethod
    def _time_diff(t1, t2):
        """
        This static method is created to simplify creating a variant of
        ICE-NODE (i.e. ICE-NODE_UNIFORM) that integrates with a
        fixed-time interval. So only this method that needs to be overriden.
        """
        return t1 - t2

    def __call__(self, patient: Patient,
                 embedded_admissions: List[EmbeddedAdmission]):
        adms = patient.admissions
        state = self.join_state_emb(None, embedded_admissions[0].emb)
        t0_date = adms[0].admission_time[0]
        preds = []
        for i in range(1, len(adms)):
            adm = adms[i]
            demo = embedded_admissions[i].demo

            # Integrate
            t0 = adm[i - 1].days_since(t0_date)[1]
            t1 = adm.days_since(t0_date)[1]
            dt = self._time_diff(t1, t0)
            delta_disch2disch = jnp.array(dt)
            state = self._integrate(state, delta_disch2disch, demo)
            mem, dx_e_hat = self.split_state_emb(state)

            # Predict
            dx_hat = CodesVector(self._decode(dx_e_hat), adm.outcome.scheme)

            # Update
            dx_e = embedded_admissions[i].dx
            mem = self._update(mem, dx_e_hat, dx_e)
            state = self.join_state_emb(mem, dx_e)

            preds.append(AdmissionPrediction(admission=adm, outcome=dx_hat))
        return preds


class ICENODE_UNIFORM(ICENODE):

    @staticmethod
    def _time_diff(t1, t2):
        return 7.0


class ICENODE_ZERO(ICENODE_UNIFORM):

    @eqx.filter_jit
    def _integrate(self, state, int_time, ctrl):
        return state
