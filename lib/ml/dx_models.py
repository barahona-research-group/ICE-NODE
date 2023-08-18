"""."""
from __future__ import annotations
from typing import List, Callable, Tuple

from absl import logging
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx

from ..utils import model_params_scaler
from ..ehr import (Patient, AdmissionPrediction, DemographicVectorConfig,
                   DatasetScheme, CodesVector)
from .embeddings import (OutpatientEmbedding, EmbeddedOutAdmission)

from .base_models import (StateUpdate, NeuralODE_JAX)
from .model import OutpatientModel, ModelDimensions


class ICENODEDimensions(ModelDimensions):
    mem: int = 15


#     @classmethod
#     def sample_model_config(cls, trial: optuna.Trial):
#         return {'state_size': trial.suggest_int('s', 10, 100, 10)}


class ICENODE(OutpatientModel):
    f_dyn: Callable
    f_update: Callable
    dims: ICENODEDimensions = eqx.static_field()

    def __init__(self, dims: ICENODEDimensions, schemes: Tuple[DatasetScheme],
                 demographic_vector_config: DemographicVectorConfig,
                 key: "jax.random.PRNGKey"):
        self._assert_demo_dim(dims, schemes[1], demographic_vector_config)
        (emb_key, dx_dec_key, dyn_key, up_key) = jrandom.split(key, 4)

        f_emb = OutpatientEmbedding(
            schemes=schemes,
            demographic_vector_config=demographic_vector_config,
            dims=dims.emb,
            key=emb_key)
        f_dx_dec = eqx.nn.MLP(dims.emb.dx,
                              len(schemes[1].outcome),
                              dims.emb.dx * 5,
                              depth=1,
                              key=dx_dec_key)

        dyn_state_size = dims.emb.dx + dims.mem
        dyn_input_size = dyn_state_size + dims.emb.demo
        f_dyn = eqx.nn.MLP(in_size=dyn_input_size,
                           out_size=dyn_state_size,
                           activation=jnn.tanh,
                           depth=2,
                           width_size=dyn_state_size * 5,
                           key=dyn_key)
        ode_dyn_f = model_params_scaler(f_dyn, 1e-2, eqx.is_inexact_array)

        self.f_dyn = NeuralODE_JAX(ode_dyn_f, timescale=1.0)

        self.f_update = StateUpdate(state_size=dims.mem,
                                    embeddings_size=dims.emb.dx,
                                    key=up_key)

        super().__init__(dims=dims,
                         schemes=schemes,
                         demographic_vector_config=demographic_vector_config,
                         f_emb=f_emb,
                         f_dx_dec=f_dx_dec)

    def join_state_emb(self, state, emb):
        if state is None:
            state = jnp.zeros((self.dims.mem, ))
        return jnp.hstack((state, emb))

    def split_state_emb(self, state: jnp.ndarray):
        return jnp.hsplit(state, (self.dims.mem, ))

    @eqx.filter_jit
    def _integrate(self, state, delta, ctrl):
        second = jnp.array(1 / (3600.0 * 24.0))
        delta = jnp.where((delta < second) & (delta >= 0.0), second, delta)
        return self.f_dyn(delta, state, args=dict(control=ctrl))[-1]

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
                 embedded_admissions: List[EmbeddedOutAdmission]):
        adms = patient.admissions
        state = self.join_state_emb(None, embedded_admissions[0].dx)
        t0_date = adms[0].admission_dates[0]
        preds = []
        for i in range(1, len(adms)):
            adm = adms[i]
            # Integrate
            # days between first admission and last discharge
            t0 = adms[i - 1].days_since(t0_date)[1]
            # days between first admission and current discharge
            t1 = adms[i].days_since(t0_date)[1]
            # days between last discharge and current discharge
            dt = self._time_diff(t1, t0)
            delta_disch2disch = jnp.array(dt)

            demo_e = embedded_admissions[i - 1].demo
            state = self._integrate(state, delta_disch2disch, demo_e)
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


class GRUDimensions(ModelDimensions):
    pass


class GRU(OutpatientModel):
    f_update: Callable
    dims: GRUDimensions = eqx.static_field()

    def __init__(self, dims: GRUDimensions, schemes: Tuple[DatasetScheme],
                 demographic_vector_config: DemographicVectorConfig,
                 key: "jax.random.PRNGKey"):
        self._assert_demo_dim(dims, schemes[1], demographic_vector_config)
        (emb_key, dx_dec_key, up_key) = jrandom.split(key, 3)
        f_emb = OutpatientEmbedding(
            schemes=schemes,
            demographic_vector_config=demographic_vector_config,
            dims=dims.emb,
            key=emb_key)
        f_dx_dec = eqx.nn.MLP(dims.emb.dx,
                              len(schemes[1].outcome),
                              dims.emb.dx * 5,
                              depth=1,
                              key=dx_dec_key)

        self.f_update = eqx.nn.GRUCell(dims.emb.dx + dims.emb.demo,
                                       dims.emb.dx,
                                       use_bias=True,
                                       key=up_key)

        super().__init__(dims=dims,
                         schemes=schemes,
                         demographic_vector_config=demographic_vector_config,
                         f_emb=f_emb,
                         f_dx_dec=f_dx_dec)

    def weights(self):
        return [self.f_update.weight_hh, self.f_update.weight_ih]

    @eqx.filter_jit
    def _update(self, mem: jnp.ndarray, dx_e_prev: jnp.ndarray,
                demo: jnp.ndarray):
        x = jnp.hstack((dx_e_prev, demo))
        return self.f_update(x, mem)

    @eqx.filter_jit
    def _decode(self, dx_e_hat: jnp.ndarray):
        return self.f_dx_dec(dx_e_hat)

    def __call__(self, patient: Patient,
                 embedded_admissions: List[EmbeddedOutAdmission]):
        adms = patient.admissions
        state = jnp.zeros((self.dims.emb.dx, ))
        preds = []
        for i in range(1, len(adms)):
            adm = adms[i]
            demo = embedded_admissions[i - 1].demo
            dx_e_prev = embedded_admissions[i - 1].dx
            # Step
            state = self._update(state, dx_e_prev, demo)
            # Predict
            dx_hat = CodesVector(self._decode(state), adm.outcome.scheme)

            preds.append(AdmissionPrediction(admission=adm, outcome=dx_hat))
        return preds


class RETAINDimensions(ModelDimensions):
    mem_a: int = eqx.static_field(default=45)
    mem_b: int = eqx.static_field(default=45)

    # @staticmethod
    # def sample_model_config(trial: optuna.Trial):
    #     sa = trial.suggest_int('sa', 100, 350, 50)
    #     sb = trial.suggest_int('sb', 100, 350, 50)
    #     return {'state_size': (sa, sb)}


class RETAIN(OutpatientModel):

    f_gru_a: Callable
    f_gru_b: Callable
    f_att_a: Callable
    f_att_b: Callable
    dims: RETAINDimensions = eqx.static_field()

    def __init__(self, dims: RETAINDimensions, schemes: Tuple[DatasetScheme],
                 demographic_vector_config: DemographicVectorConfig,
                 key: "jax.random.PRNGKey"):
        self._assert_demo_dim(dims, schemes[1], demographic_vector_config)
        k1, k2, k3, k4, k5, k6 = jrandom.split(key, 6)

        f_emb = OutpatientEmbedding(
            schemes=schemes,
            demographic_vector_config=demographic_vector_config,
            dims=dims.emb,
            key=k1)
        f_dx_dec = eqx.nn.MLP(dims.emb.dx,
                              len(schemes[1].outcome),
                              dims.emb.dx * 5,
                              depth=1,
                              key=k2)
        self.f_gru_a = eqx.nn.GRUCell(dims.emb.dx + dims.emb.demo,
                                      dims.mem_a,
                                      use_bias=True,
                                      key=k3)
        self.f_gru_b = eqx.nn.GRUCell(dims.emb.dx + dims.emb.demo,
                                      dims.mem_b,
                                      use_bias=True,
                                      key=k4)

        self.f_att_a = eqx.nn.Linear(dims.mem_a, 1, use_bias=True, key=k5)
        self.f_att_b = eqx.nn.Linear(dims.mem_b,
                                     dims.emb.dx,
                                     use_bias=True,
                                     key=k6)

        super().__init__(dims=dims,
                         schemes=schemes,
                         demographic_vector_config=demographic_vector_config,
                         f_emb=f_emb,
                         f_dx_dec=f_dx_dec)

    def weights(self):
        return [
            self.f_gru_a.weight_hh, self.f_gru_a.weight_ih,
            self.f_gru_b.weight_hh, self.f_gru_b.weight_ih,
            self.f_att_a.weight, self.f_att_b.weight
        ]

    @eqx.filter_jit
    def _gru_a(self, x, state):
        return self.f_gru_a(x, state)

    @eqx.filter_jit
    def _gru_b(self, x, state):
        return self.f_gru_b(x, state)

    @eqx.filter_jit
    def _att_a(self, x):
        return self.f_att_a(x)

    @eqx.filter_jit
    def _att_b(self, x):
        return self.f_att_b(x)

    @eqx.filter_jit
    def _dx_dec(self, x):
        return self.f_dx_dec(x)

    def __call__(self, patient: Patient,
                 embedded_admissions: List[EmbeddedOutAdmission]):
        adms = patient.admissions
        state_a0 = jnp.zeros(self.dims.mem_a)
        state_b0 = jnp.zeros(self.dims.mem_b)
        preds = []

        # step 1 @RETAIN paper

        # v1, v2, ..., vT
        v_seq = jnp.vstack([adm.dx for adm in embedded_admissions])

        # c1, c2, ..., cT. <- controls
        c_seq = jnp.vstack([adm.demo for adm in embedded_admissions])

        # Merge controls with embeddings
        cv_seq = jnp.hstack([c_seq, v_seq])

        for i in range(1, len(adms)):
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

            b_seq = jnp.vstack(b_seq)

            # alpha: i, ..., 1
            a_seq = jax.nn.softmax(jnp.hstack(e_seq))

            # step 4 @RETAIN paper

            # v_i, ..., v_1
            v_context = v_seq[:i][::-1]
            c_context = sum(a * (b * v)
                            for a, b, v in zip(a_seq, b_seq, v_context))

            # step 5 @RETAIN paper
            logits = CodesVector(self._dx_dec(c_context),
                                 adms[i].outcome.scheme)

            preds.append(AdmissionPrediction(admission=adms[i],
                                             outcome=logits))

        return preds
