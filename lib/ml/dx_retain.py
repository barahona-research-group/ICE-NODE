"""JAX implementation of RETAIN algorithm."""
from __future__ import annotations
from functools import partial
from typing import Any, List, TYPE_CHECKING, Callable, Tuple

import jax
import jax.random as jrandom
import jax.numpy as jnp

import equinox as eqx

if TYPE_CHECKING:
    import optuna

from ..ehr import Subject_JAX, BatchPredictedRisks
from .abstract_model import AbstractModel


class RETAIN(AbstractModel):
    gru_a: Callable
    gru_b: Callable
    att_a: Callable
    att_b: Callable

    def __init__(self, key: "jax.random.PRNGKey", *args, **kwargs):
        kwargs['state_size'] = tuple(kwargs['state_size'])
        super().__init__(*args, **kwargs)

        k1, k2, k3, k4 = jrandom.split(key, 4)
        self.gru_a = eqx.nn.GRUCell(self.dx_emb.embeddings_size +
                                    self.control_size,
                                    self.state_size[0],
                                    use_bias=True,
                                    key=k1)
        self.gru_b = eqx.nn.GRUCell(self.dx_emb.embeddings_size +
                                    self.control_size,
                                    self.state_size[1],
                                    use_bias=True,
                                    key=k2)

        self.att_a = eqx.nn.Linear(self.state_size[0],
                                   1,
                                   use_bias=True,
                                   key=k3)
        self.att_b = eqx.nn.Linear(self.state_size[1],
                                   self.dx_emb.embeddings_size,
                                   use_bias=True,
                                   key=k4)

    def weights(self):
        return [
            self.gru_a.weight_hh, self.gru_a.weight_ih, self.gru_b.weight_hh,
            self.gru_b.weight_ih, self.att_a.weight, self.att_b.weight
        ]

    @staticmethod
    def decoder_input_size(expt_config):
        return expt_config["emb"]["dx"]["embeddings_size"]

    def __call__(self,
                 subject_interface: Subject_JAX,
                 subjects_batch: List[int],
                 args=dict()):
        dx_for_emb = subject_interface.dx_batch_history_vec(subjects_batch)
        G = self.dx_emb.compute_embeddings_mat(dx_for_emb)
        emb = partial(self.dx_emb.encode, G)

        risk_prediction = BatchPredictedRisks()
        state_a0 = jnp.zeros(self.state_size[0])
        state_b0 = jnp.zeros(self.state_size[1])

        for subj_id in subjects_batch:
            adms = subject_interface[subj_id]
            get_ctrl = partial(subject_interface.subject_control, subj_id)

            # step 1 @RETAIN paper

            # v1, v2, ..., vT
            v_seq = jnp.vstack([emb(adm.dx_vec) for adm in adms])

            # c1, c2, ..., cT. <- controls
            c_seq = jnp.vstack([get_ctrl(adm.admission_date) for adm in adms])

            # Merge controls with embeddings
            cv_seq = jnp.hstack([c_seq, v_seq])

            for i in range(1, len(v_seq)):
                # e: i, ..., 1
                e_seq = []

                # beta: i, ..., 1
                b_seq = []

                state_a = state_a0
                state_b = state_b0
                for j in reversed(range(i)):
                    # step 2 @RETAIN paper
                    state_a = self.gru_a(cv_seq[j], state_a)
                    e_j = self.att_a(state_a)
                    # After the for-loop apply softmax on e_seq to get
                    # alpha_seq

                    e_seq.append(e_j)

                    # step 3 @RETAIN paper
                    h_j = state_b = self.gru_b(cv_seq[j], state_b)
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
                                    admission=adms[i],
                                    prediction=logits)
                if args.get('return_embeddings', False):
                    risk_prediction.set_subject_embeddings(
                        subject_id=subj_id, embeddings=c_context)

        return {'predictions': risk_prediction}

    @staticmethod
    def sample_model_config(trial: optuna.Trial):
        sa = trial.suggest_int('sa', 100, 350, 50)
        sb = trial.suggest_int('sb', 100, 350, 50)
        return {'state_size': (sa, sb)}
