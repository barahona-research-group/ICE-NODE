"""."""

from typing import List, Callable

import jax
import jax.random as jrandom
import jax.numpy as jnp
import equinox as eqx

from ..ehr import (Patient, AdmissionPrediction, MIMICDatasetScheme,
                   DemographicVectorConfig, CodesVector)
from ..embeddings import (PatientEmbeddingDimensions, PatientEmbedding,
                          EmbeddedAdmission)
from .abstract_model import OutpatientModel, ModelDimensions


class GRUDimensions(ModelDimensions):
    mem: int = 45

    def __init__(self, emb: PatientEmbeddingDimensions, mem: int):
        super().__init__(emb=emb)
        self.emb = emb
        self.mem = mem


class GRU(OutpatientModel):
    f_update: Callable

    def __init__(self, dims: GRUDimensions, scheme: MIMICDatasetScheme,
                 demographic_vector_config: DemographicVectorConfig,
                 key: "jax.random.PRNGKey"):
        (emb_key, dx_dec_key, up_key) = jrandom.split(key, 3)
        f_emb = PatientEmbedding(
            scheme=scheme,
            demographic_vector_config=demographic_vector_config,
            dims=dims.emb,
            key=emb_key)
        f_dx_dec = eqx.nn.MLP(dims.emb.dx,
                              len(scheme.outcome),
                              dims.emb.dx * 5,
                              depth=1,
                              key=dx_dec_key)

        self.f_update = eqx.nn.GRUCell(dims.emb.dx + dims.emb.demo,
                                       dims.mem,
                                       use_bias=True,
                                       key=up_key)

        super().__init__(dims=dims,
                         scheme=scheme,
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
                 embedded_admissions: List[EmbeddedAdmission]):
        adms = patient.admissions
        state = jnp.zeros((self.dims.mem, ))
        preds = []
        for i in range(1, len(adms)):
            adm = adms[i]
            demo = embedded_admissions[i].demo
            dx_e_prev = embedded_admissions[i - 1].dx
            # Step
            state = self._update(state, dx_e_prev, demo)
            # Predict
            dx_hat = CodesVector(self._decode(state), adm.outcome.scheme)

            preds.append(AdmissionPrediction(admission=adm, outcome=dx_hat))
        return preds
