from functools import partial
from typing import Any, List

import pandas as pd
import haiku as hk
import jax
import jax.numpy as jnp

from ..ehr_model.jax_interface import SubjectDiagSequenceJAXInterface
from ..ehr_model.mimic.concept import DiagSubject
from ..ehr_model.ccs_dag import CCSDAG

from ..embeddings.gram import AbstractEmbeddingsLayer
from ..metric.common_metrics import l2_squared, l1_absolute
from ..utils import wrap_module

from .abstract import AbstractModel


@jax.jit
def diag_loss(y: jnp.ndarray, diag_logits: jnp.ndarray):
    return -jnp.sum(y * jax.nn.log_softmax(diag_logits) +
                    (1 - y) * jnp.log(1 - jax.nn.softmax(diag_logits)))


class GRAM(AbstractModel):

    def __init__(self, subject_interface: SubjectDiagSequenceJAXInterface,
                 diag_emb: AbstractEmbeddingsLayer, state_size: int):

        self.subject_interface = subject_interface
        self.diag_emb = diag_emb

        self.dimensions = {
            'diag_emb': diag_emb.embeddings_dim,
            'diag_in': len(subject_interface.diag_ccs_idx),
            'diag_out': len(subject_interface.diag_flatccs_idx),
            'state': state_size
        }

        gru_init, gru = hk.without_apply_rng(
            hk.transform(
                wrap_module(hk.GRU, hidden_size=state_size, name='gru')))
        self.gru = jax.jit(gru)

        out_init, out = hk.without_apply_rng(
            hk.transform(
                wrap_module(hk.Linear,
                            output_size=self.dimensions['diag_out'],
                            name='out')))
        self.out = jax.jit(out)

        self.initializers = {'gru': gru_init, 'out': out_init}

    def init_params(self, rng_key):
        state = jnp.zeros(self.dimensions['state'])
        diag_emb = jnp.zeros(self.dimensions['diag_emb'])

        return {
            "diag_emb": self.diag_emb.init_params(rng_key),
            "gru": self.initializers['gru'](rng_key, diag_emb, state),
            "out": self.initializers['out'](rng_key, state)
        }

    def state_size(self):
        return self.dimensions['state']

    def __call__(self, params: Any, subjects_batch: List[int], **kwargs):

        G = self.diag_emb.compute_embeddings_mat(params["diag_emb"])
        emb = partial(self.diag_emb.encode, G)
        diag_seqs = self.subject_interface.diag_sequences_batch(subjects_batch)

        loss = {}
        diag_detectability = {}
        state0 = jnp.zeros(self.dimensions['state'])
        for subject_id, _diag_seqs in diag_seqs.items():
            # Exclude last one for irrelevance
            hierarchical_diag = _diag_seqs['diag_ccs_vec'][:-1]
            # Exclude first one, we need to predict them for a future step.
            diag_flatccs = _diag_seqs['diag_flatccs_vec'][1:]
            admission_id = _diag_seqs['admission_id'][1:]

            emb_seqs = map(emb, hierarchical_diag)

            diag_detectability[subject_id] = {}
            loss[subject_id] = []
            state = state0
            for i, diag_emb in enumerate(emb_seqs):
                y_i = diag_flatccs[i]
                output, state = self.gru(params['gru'], diag_emb, state)
                logits = self.out(params['out'], output)
                diag_detectability[subject_id][i] = {
                    'admission_id': admission_id[i],
                    'true_diag': y_i,
                    'pred_logits': logits
                }
                loss[subject_id].append(diag_loss(y_i, logits))

        loss = [sum(l) / len(l) for l in loss.values()]

        return {
            'loss': sum(loss) / len(loss),
            'diag_detectability': diag_detectability
        }

    def detailed_loss(self, loss_mixing, params, res):

        diag_loss_ = res['loss']
        l1_loss = l1_absolute(params)
        l2_loss = l2_squared(params)
        l1_alpha = loss_mixing['L_l1']
        l2_alpha = loss_mixing['L_l2']

        loss = diag_loss_ + (l1_alpha * l1_loss) + (l2_alpha * l2_loss)

        return {
            'diag_loss': diag_loss_,
            'loss': loss,
            'l1_loss': l1_loss,
            'l2_loss': l2_loss,
        }

    def eval_stats(self, res):
        return {}

    @staticmethod
    def create_patient_interface(processed_mimic_tables_dir: str,
                                 data_tag: str):
        static_df = pd.read_csv(
            f'{processed_mimic_tables_dir}/static_df.csv.gz')
        adm_df = pd.read_csv(f'{processed_mimic_tables_dir}/adm_df.csv.gz')
        diag_df = pd.read_csv(f'{processed_mimic_tables_dir}/diag_df.csv.gz',
                              dtype={'ICD9_CODE': str})
        # Cast columns of dates to datetime64
        adm_df['ADMITTIME'] = pd.to_datetime(
            adm_df['ADMITTIME'], infer_datetime_format=True).dt.normalize()
        adm_df['DISCHTIME'] = pd.to_datetime(
            adm_df['DISCHTIME'], infer_datetime_format=True).dt.normalize()

        patients = DiagSubject.to_list(adm_df, diag_df)

        # CCS Knowledge Graph
        k_graph = CCSDAG()

        return SubjectDiagSequenceJAXInterface(patients, k_graph)

    @classmethod
    def create_model(cls, config, patient_interface, train_ids,
                     pretrained_components):
        diag_emb = cls.create_embedding(
            emb_config=config['emb']['diag'],
            emb_kind=config['emb']['kind'],
            patient_interface=patient_interface,
            train_ids=train_ids,
            pretrained_components=pretrained_components)

        return cls(subject_interface=patient_interface,
                   diag_emb=diag_emb,
                   **config['model'])


if __name__ == '__main__':
    from ..hyperopt.hpo_utils import capture_args, run_trials
    run_trials(model_cls=GRAM, **capture_args())
