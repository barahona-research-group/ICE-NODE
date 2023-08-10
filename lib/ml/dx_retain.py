"""JAX implementation of RETAIN algorithm."""
from __future__ import annotations
from typing import List, TYPE_CHECKING, Callable

import jax
import jax.random as jrandom
import jax.numpy as jnp

import equinox as eqx

from ..ehr import (Patient, AdmissionPrediction, MIMICDatasetScheme,
                   DemographicVectorConfig)
from ..embeddings import (PatientEmbeddingDimensions, PatientEmbedding,
                          EmbeddedAdmission)
from .abstract_model import OutpatientModel, ModelDimensions

if TYPE_CHECKING:
    import optuna


class RETAINDimensions(ModelDimensions):
    mem_a: int = 45
    mem_b: int = 45

    def __init__(self, emb: PatientEmbeddingDimensions, mem_a: int,
                 mem_b: int):
        super().__init__(emb=emb)
        self.emb = emb
        self.mem_a = mem_a
        self.mem_b = mem_b

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

    def __init__(self, dims: RETAINDimensions, scheme: MIMICDatasetScheme,
                 demographic_vector_config: DemographicVectorConfig,
                 key: "jax.random.PRNGKey"):
        k1, k2, k3, k4, k5, k6 = jrandom.split(key, 6)

        f_emb = PatientEmbedding(
            scheme=scheme,
            demographic_vector_config=demographic_vector_config,
            dims=dims.emb,
            key=k1)
        f_dx_dec = eqx.nn.MLP(dims.emb.dx,
                              len(scheme.outcome),
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
                         scheme=scheme,
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
                 embedded_admissions: List[EmbeddedAdmission]):
        adms = patient.admissions
        state_a0 = jnp.zeros(self.dims.mem_a)
        state_b0 = jnp.zeros(self.dims.mem_b)
        preds = []

        # step 1 @RETAIN paper

        # v1, v2, ..., vT
        v_seq = jnp.vstack([adm.dx0 for adm in embedded_admissions])

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

            preds.append(AdmissionPrediction(adm=adms[i], outcome=logits))

        return preds
