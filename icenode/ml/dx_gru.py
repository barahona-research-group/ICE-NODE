"""."""

from functools import partial
from typing import Any, List, Callable

import equinox as eqx
import jax
import jax.random as jrandom
import jax.numpy as jnp

from .. import metric

from .abstract import AbstractModel
from ..ehr import Subject_JAX


@jax.jit
def dx_loss(y: jnp.ndarray, dx_logits: jnp.ndarray):
    return -jnp.sum(y * jax.nn.log_softmax(dx_logits) +
                    (1 - y) * jnp.log(1 - jax.nn.softmax(dx_logits)))


class GRU(AbstractModel):
    f_update: Callable

    def __init__(self, key, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_update = eqx.nn.GRUCell(self.dx_emb.embeddings_size +
                                       self.control_size,
                                       self.state_size,
                                       use_bias=True,
                                       key=key)

    def __call__(self, subject_interface: Subject_JAX,
                 subjects_batch: List[int], args):

        G = self.dx_emb.compute_embeddings_mat()
        emb = partial(self.dx_emb.encode, G)

        loss = {}
        risk_prediction = metric.BatchPredictedRisks()
        state0 = jnp.zeros(self.state_size)
        for subject_id in subjects_batch:
            get_ctrl = partial(subject_interface.subject_control, subject_id)

            adms = subject_interface.batch_nth_admission([subject_id])
            adms = [adms[n_i][subject_id] for n_i in sorted(adms)]

            # Exclude last input for irrelevance (not more future predictions)
            dx_vec = [adm.dx_vec for adm in adms[:-1]]

            # Control inputs
            ctrl = [get_ctrl(adm.admission_date) for adm in adms[:-1]]

            # Exclude first one, we need to predict them for a future step.
            dx_outcome = [adm.dx_outcome for adm in adms[1:]]

            admission_id = [adm.admission_id for adm in adms[1:]]

            emb_seqs = map(emb, dx_vec)

            loss[subject_id] = []
            state = state0
            for i, (dx_emb_i, ctrl_i) in enumerate(zip(emb_seqs, ctrl)):
                y_i = dx_outcome[i]

                state = self.f_update(jnp.hstack((dx_emb_i, ctrl_i)), state)
                logits = self.dx_dec(state)
                risk_prediction.add(subject_id=subject_id,
                                    admission_id=admission_id,
                                    index=i,
                                    prediction=logits,
                                    ground_truth=y_i)
                if args.get('return_embeddings', False):
                    risk_prediction.set_subject_embeddings(
                        subject_id=subject_id, embeddings=state)

                loss[subject_id].append(dx_loss(y_i, logits))

        loss = [sum(l) / len(l) for l in loss.values()]

        return {
            'loss': sum(loss) / len(loss),
            'risk_prediction': risk_prediction
        }

    def detailed_loss(self, loss_mixing, params, res):

        dx_loss_ = res['loss']
        l1_loss = metric.l1_absolute(params)
        l2_loss = metric.l2_squared(params)
        l1_alpha = loss_mixing['L_l1']
        l2_alpha = loss_mixing['L_l2']

        loss = dx_loss_ + (l1_alpha * l1_loss) + (l2_alpha * l2_loss)

        return {
            'dx_loss': dx_loss_,
            'loss': loss,
            'l1_loss': l1_loss,
            'l2_loss': l2_loss,
        }

    def eval_stats(self, res):
        return {}

    @classmethod
    def create_model(cls, config, subject_interface, train_ids):
        dx_emb = cls.create_embedding(emb_config=config['emb']['dx'],
                                      emb_kind=config['emb']['kind'],
                                      subject_interface=subject_interface,
                                      train_ids=train_ids)

        return cls(subject_interface=subject_interface,
                   dx_emb=dx_emb,
                   **config['model'])
