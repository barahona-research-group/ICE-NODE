from functools import partial
from typing import Any, List

import haiku as hk
import jax
import jax.numpy as jnp

from ..ehr_model.jax_interface import DxInterface_JAX
from ..ehr_model.ccs_dag import ccs_dag
from ..embeddings.gram import AbstractEmbeddingsLayer
from ..metric.common_metrics import l2_squared, l1_absolute
from ..utils import wrap_module

from .abstract import AbstractModel
from .risk import BatchPredictedRisks


@jax.jit
def dx_loss(y: jnp.ndarray, dx_logits: jnp.ndarray):
    return -jnp.sum(y * jax.nn.log_softmax(dx_logits) +
                    (1 - y) * jnp.log(1 - jax.nn.softmax(dx_logits)))


class GRAM(AbstractModel):

    def __init__(self, subject_interface: DxInterface_JAX,
                 dx_emb: AbstractEmbeddingsLayer, state_size: int):

        self.subject_interface = subject_interface
        self.dx_emb = dx_emb

        self.dimensions = {
            'dx_emb': dx_emb.embeddings_dim,
            'dx_out': len(ccs_dag.dx_flatccs_idx),
            'state': state_size
        }

        gru_init, gru = hk.without_apply_rng(
            hk.transform(
                wrap_module(hk.GRU, hidden_size=state_size, name='gru')))
        self.gru = jax.jit(gru)

        out_init, out = hk.without_apply_rng(
            hk.transform(
                wrap_module(hk.Linear,
                            output_size=self.dimensions['dx_out'],
                            name='out')))
        self.out = jax.jit(out)

        self.initializers = {'gru': gru_init, 'out': out_init}

    def init_params(self, rng_key):
        state = jnp.zeros(self.dimensions['state'])
        dx_emb = jnp.zeros(self.dimensions['dx_emb'])

        return {
            "dx_emb": self.dx_emb.init_params(rng_key),
            "gru": self.initializers['gru'](rng_key, dx_emb, state),
            "out": self.initializers['out'](rng_key, state)
        }

    def state_size(self):
        return self.dimensions['state']

    def __call__(self, params: Any, subjects_batch: List[int], **kwargs):

        G = self.dx_emb.compute_embeddings_mat(params["dx_emb"])
        emb = partial(self.dx_emb.encode, G)

        loss = {}
        risk_prediction = BatchPredictedRisks()
        state0 = jnp.zeros(self.dimensions['state'])
        for subject_id in subjects_batch:
            adms = self.subject_interface.batch_nth_admission([subject_id])
            n = sorted(adms)
            adms = [adms[n_i][subject_id] for n_i in n]

            # Exclude last input for irrelevance (not more future predictions)
            dx_ccs = [adm.dx_ccs_codes for adm in adms[:-1]]

            # Exclude first one, we need to predict them for a future step.
            dx_flatccs = [adm.dx_flatccs_codes for adm in adms[1:]]

            admission_id = [adm.admission_id for adm in adms[1:]]

            emb_seqs = map(emb, dx_ccs)

            loss[subject_id] = []
            state = state0
            for i, dx_emb in enumerate(emb_seqs):
                y_i = dx_flatccs[i]
                output, state = self.gru(params['gru'], dx_emb, state)
                logits = self.out(params['out'], output)
                risk_prediction.add(subject_id=subject_id,
                                    admission_id=admission_id,
                                    index=i,
                                    prediction=logits,
                                    ground_truth=y_i)

                loss[subject_id].append(dx_loss(y_i, logits))

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
    def create_model(cls, config, patient_interface, train_ids):
        dx_emb = cls.create_embedding(emb_config=config['emb']['dx'],
                                      emb_kind=config['emb']['kind'],
                                      patient_interface=patient_interface,
                                      train_ids=train_ids)

        return cls(subject_interface=patient_interface,
                   dx_emb=dx_emb,
                   **config['model'])


if __name__ == '__main__':
    from ..hyperopt.optuna_job import capture_args, run_trials
    run_trials(model_cls=GRAM, **capture_args())
