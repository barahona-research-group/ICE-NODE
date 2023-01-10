"""JAX implementation of RETAIN algorithm."""
from __future__ import annotations
from functools import partial
from typing import Any, List, TYPE_CHECKING, Callable

import jax
import jax.random as jrandom
import jax.numpy as jnp

import equinox as eqx

if TYPE_CHECKING:
    import optuna

from ..ehr import Subject_JAX
from .. import embeddings as E
from .. import metric
from .. import utils

from .abstract import AbstractModel


@jax.jit
def dx_loss(y: jnp.ndarray, dx_logits: jnp.ndarray):
    return -jnp.sum(y * jax.nn.log_softmax(dx_logits) +
                    (1 - y) * jnp.log(1 - jax.nn.softmax(dx_logits)))


class RETAIN(AbstractModel):
    state_a_size: int
    state_b_size: int

    gru_a: Callable
    gru_b: Callable
    att_a: Callable
    att_b: Callable

    def __init__(self, state_a_size: int, state_b_size: int,
                 key: "jax.random.PRNGKey", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_a_size
        self.state_b_size

        k1, k2, k3, k4 = jrandom.split(key, 4)
        self.gru_a = eqx.nn.GRUCell(self.dx_emb.embeddings_size +
                                    self.control_size,
                                    state_a_size,
                                    use_bias=True,
                                    key=k1)
        self.gru_b = eqx.nn.GRUCell(self.dx_emb.embeddings_size +
                                    self.control_size,
                                    state_b_size,
                                    use_bias=True,
                                    key=k2)

        self.att_a = eqx.nn.Linear(state_a_size, 1, use_bias=True, key=k3)
        self.att_b = eqx.nn.Linear(state_b_size,
                                   self.dx_emb.embeddings_size,
                                   use_bias=True,
                                   key=k3)

    def __call__(self, subject_interface: Subject_JAX,
                 subjects_batch: List[int], args):
        G = self.dx_emb.compute_embeddings_mat()
        emb = jax.vmap(partial(self.dx_emb.encode, G))

        loss = {}
        risk_prediction = metric.BatchPredictedRisks()
        state_a0 = jnp.zeros(self.state_a_size)
        state_b0 = jnp.zeros(self.state_b_size)

        for subj_id in subjects_batch:
            adms = subject_interface[subj_id]
            get_ctrl = partial(subject_interface.subject_control, subj_id)

            dx_vec = jnp.vstack([adm.dx_vec for adm in adms])
            dx_outcome = [adm.dx_outcome for adm in adms]
            admission_id = [adm.admission_id for adm in adms]

            # step 1 @RETAIN paper

            # v1, v2, ..., vT
            v_seq = emb(dx_vec)

            # c1, c2, ..., cT. <- controls
            c_seq = jnp.vstack([get_ctrl(adm.admission_date) for adm in adms])

            # Merge controls with embeddings
            v_seq = jnp.hstack([c_seq, v_seq])

            loss[subj_id] = []

            for i in range(1, len(v_seq)):
                # e: i, ..., 1
                e_seq = []

                # beta: i, ..., 1
                b_seq = []

                state_a = state_a0
                state_b = state_b0
                for j in reversed(range(i)):
                    # step 2 @RETAIN paper
                    g_j = state_a = self.gru_a(v_seq[j], state_a)
                    e_j = self.att_a(g_j)
                    # After the for-loop apply softmax on e_seq to get
                    # alpha_seq

                    e_seq.append(e_j)

                    # step 3 @RETAIN paper
                    h_j = state_b = self.gru_b(v_seq[j], state_b)
                    b_j = self.att_b(h_j)

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
                logits = self.dx_dec(c_context)
                risk_prediction.add(subject_id=subj_id,
                                    admission_id=admission_id,
                                    index=i,
                                    prediction=logits,
                                    ground_truth=dx_outcome[i])
                if args.get('return_embeddings', False):
                    risk_prediction.set_subject_embeddings(
                        subject_id=subj_id, embeddings=c_context)

                loss[subj_id].append(dx_loss(dx_outcome[i], logits))

        # Loss of all visits 2, ..., T, normalized by T
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

    @staticmethod
    def sample_model_config(trial: optuna.Trial):
        return {
            'state_a_size': trial.suggest_int('sa', 100, 350, 50),
            'state_b_size': trial.suggest_int('sb', 100, 350, 50)
        }

    @staticmethod
    def sample_training_config(trial: optuna.Trial):
        return AbstractModel.sample_training_config(trial)
