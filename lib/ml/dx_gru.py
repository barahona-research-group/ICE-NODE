"""."""

from functools import partial
from typing import Any, List, Callable

import equinox as eqx
import jax.numpy as jnp

from .abstract_model import AbstractModel

from ..ehr import Subject_JAX, BatchPredictedRisks


class GRU(AbstractModel):
    f_update: Callable

    def __init__(self, key: "jax.random.PRNGKey", **kwargs):
        super().__init__(**kwargs)
        self.f_update = eqx.nn.GRUCell(self.dx_emb.embeddings_size +
                                       self.control_size,
                                       self.state_size,
                                       use_bias=True,
                                       key=key)

    def weights(self):
        return [self.f_update.weight_hh, self.f_update.weight_ih]

    def __call__(self,
                 subject_interface: Subject_JAX,
                 subjects_batch: List[int],
                 args=dict()):
        dx_for_emb = subject_interface.dx_batch_history_vec(subjects_batch)
        G = self.dx_emb.compute_embeddings_mat(dx_for_emb)
        emb = partial(self.dx_emb.encode, G)

        loss = {}
        risk_prediction = BatchPredictedRisks()
        state0 = jnp.zeros(self.state_size)
        for subject_id in subjects_batch:
            ctrl_f = partial(subject_interface.subject_control, subject_id)

            adms = subject_interface[subject_id]

            # Exclude last input for irrelevance (not more future predictions)
            dx_vec = [adm.dx_vec for adm in adms[:-1]]
            emb_seqs = list(map(emb, dx_vec))

            # Merge controls with embeddings
            # c1, c2, ..., cT. <- controls
            c_seq = jnp.vstack(
                [ctrl_f(adm.admission_date) for adm in adms[:-1]])

            # Merge controls with embeddings
            emb_seqs = jnp.hstack([c_seq, emb_seqs])

            loss[subject_id] = []
            state = state0

            # Exclude first adm, we need to predict them for a future step.
            for i, (dx_emb_i, adm_i) in enumerate(zip(emb_seqs, adms[1:])):
                state = self.f_update(dx_emb_i, state)
                logits = self.dx_dec(state)
                risk_prediction.add(subject_id=subject_id,
                                    admission=adm_i,
                                    prediction=logits)
                if args.get('return_embeddings', False):
                    risk_prediction.set_subject_embeddings(
                        subject_id=subject_id, embeddings=state)

        return {'predictions': risk_prediction}
