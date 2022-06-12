"""JAX implementation of RETAIN algorithm."""

from functools import partial
from typing import Any, List

import haiku as hk
import jax
import jax.numpy as jnp
import optuna
from absl import logging

from ..ehr_model.jax_interface import DxInterface_JAX
from ..embeddings.gram import AbstractEmbeddingsLayer
from ..metric.common_metrics import l2_squared, l1_absolute
from ..utils import wrap_module
from ..ehr_predictive.abstract import AbstractModel
from ..ehr_model.ccs_dag import ccs_dag

from .risk import BatchPredictedRisks


@jax.jit
def dx_loss(y: jnp.ndarray, dx_logits: jnp.ndarray):
    return -jnp.sum(y * jax.nn.log_softmax(dx_logits) +
                    (1 - y) * jnp.log(1 - jax.nn.softmax(dx_logits)))


class RETAIN(AbstractModel):

    def __init__(self, subject_interface: DxInterface_JAX,
                 dx_emb: AbstractEmbeddingsLayer, state_a_size: int,
                 state_b_size: int):

        self.subject_interface = subject_interface
        self.dx_emb = dx_emb

        self.dimensions = {
            'dx_emb': dx_emb.embeddings_dim,
            'dx_in': len(ccs_dag.dx_ccs_idx),
            'dx_out': len(ccs_dag.dx_flatccs_idx),
            'state_a': state_a_size,
            'state_b': state_b_size
        }

        gru_a_init, gru_a = hk.without_apply_rng(
            hk.transform(
                wrap_module(hk.GRU, hidden_size=state_a_size, name='gru_a')))
        self.gru_a = jax.jit(gru_a)

        gru_b_init, gru_b = hk.without_apply_rng(
            hk.transform(
                wrap_module(hk.GRU, hidden_size=state_b_size, name='gru_b')))
        self.gru_b = jax.jit(gru_b)

        # Followed by a softmax on e1, e2, ..., -> alpha1, alpha2, ...
        att_layer_a_init, att_layer_a = hk.without_apply_rng(
            hk.transform(
                wrap_module(hk.Linear, output_size=1, name='att_layer_a')))
        self.att_layer_a = jax.jit(att_layer_a)

        # followed by a tanh
        att_layer_b_init, att_layer_b = hk.without_apply_rng(
            hk.transform(
                wrap_module(hk.Linear,
                            output_size=self.dimensions['dx_emb'],
                            name='att_layer_b')))
        self.att_layer_b = jax.jit(att_layer_b)

        decode_init, decode = hk.without_apply_rng(
            hk.transform(
                wrap_module(hk.Linear,
                            output_size=self.dimensions['dx_out'],
                            name='decoder')))
        self.decode = jax.jit(decode)

        self.initializers = {
            'gru_a': gru_a_init,
            'gru_b': gru_b_init,
            'att_layer_a': att_layer_a_init,
            'att_layer_b': att_layer_b_init,
            'decode': decode_init
        }

    def init_params(self, prng_key):
        state_a = jnp.zeros(self.dimensions['state_a'])
        state_b = jnp.zeros(self.dimensions['state_b'])

        dx_emb = jnp.zeros(self.dimensions['dx_emb'])

        return {
            "dx_emb": self.dx_emb.init_params(prng_key),
            "gru_a": self.initializers['gru_a'](prng_key, dx_emb, state_a),
            "gru_b": self.initializers['gru_b'](prng_key, dx_emb, state_b),
            "att_layer_a": self.initializers['att_layer_a'](prng_key, state_a),
            "att_layer_b": self.initializers['att_layer_b'](prng_key, state_b),
            "decode": self.initializers["decode"](prng_key, dx_emb)
        }

    def state_size(self):
        return self.dimensions['state']

    def dx_out_index(self) -> List[str]:
        index2code = {
            i: c
            for c, i in self.subject_interface.dx_ccs_idx.items()
        }
        return list(map(index2code.get, range(len(index2code))))

    def __call__(self, params: Any, subjects_batch: List[int]):
        G = self.dx_emb.compute_embeddings_mat(params["dx_emb"])
        emb = jax.vmap(partial(self.dx_emb.encode, G))

        loss = {}
        risk_prediction = BatchPredictedRisks()
        state_a0 = jnp.zeros(self.dimensions['state_a'])
        state_b0 = jnp.zeros(self.dimensions['state_b'])

        for subj_id in subjects_batch:
            adms = self.subject_interface.subject_admission_sequence(subj_id)

            dx_ccs = jnp.vstack([adm.dx_ccs_codes for adm in adms])
            dx_flatccs = [adm.dx_flatccs_codes for adm in adms]
            admission_id = [adm.admission_id for adm in adms]

            logging.debug(len(dx_ccs))

            # step 1 @RETAIN paper

            # v1, v2, ..., vT
            v_seq = emb(dx_ccs)

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
                    g_j, state_a = self.gru_a(params['gru_a'], v_seq[j],
                                              state_a)
                    e_j = self.att_layer_a(params['att_layer_a'], g_j)
                    # After the for-loop apply softmax on e_seq to get
                    # alpha_seq

                    e_seq.append(e_j)

                    # step 3 @RETAIN paper
                    h_j, state_b = self.gru_b(params['gru_b'], v_seq[j],
                                              state_b)
                    b_j = self.att_layer_b(params['att_layer_b'], h_j)

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
                logits = self.decode(params['decode'], c_context)
                risk_prediction.add(subject_id=subj_id,
                                    admission_id=admission_id,
                                    index=i,
                                    prediction=logits,
                                    ground_truth=dx_flatccs[i])

                loss[subj_id].append(dx_loss(dx_flatccs[i], logits))

        # Loss of all visits 2, ..., T, normalized by T
        loss = [sum(l) / len(l) for l in loss.values()]

        return {
            'loss': sum(loss) / len(loss),
            'risk_prediction': risk_prediction
        }

    def detailed_loss(self, loss_mixing, params, res):

        dx_loss_ = res['loss']
        l1_loss = l1_absolute(params)
        l2_loss = l2_squared(params)
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
    def create_model(cls, config, patient_interface, train_ids,
                     pretrained_components):
        dx_emb = cls.create_embedding(
            emb_config=config['emb']['dx'],
            emb_kind=config['emb']['kind'],
            patient_interface=patient_interface,
            train_ids=train_ids,
            pretrained_components=pretrained_components)

        return cls(subject_interface=patient_interface,
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


if __name__ == '__main__':
    from ..hyperopt.hpo_utils import capture_args, run_trials
    run_trials(model_cls=RETAIN, **capture_args())
