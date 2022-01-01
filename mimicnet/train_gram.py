from functools import partial
from typing import Any, Dict, List

import pandas as pd
import haiku as hk
import jax
import jax.numpy as jnp
import optuna

from .jax_interface import SubjectDiagSequenceJAXInterface
from .gram import DAGGRAM
from .metrics import l2_squared, l1_absolute
from .concept import Subject
from .dag import CCSDAG
from .utils import wrap_module
from .abstract_model import AbstractModel
from .glove import glove_representation


@jax.jit
def diag_loss(y: jnp.ndarray, diag_logits: jnp.ndarray):
    return -jnp.sum(y * jax.nn.log_softmax(diag_logits) +
                    (1 - y) * jnp.log(1 - jax.nn.softmax(diag_logits)))


class GRAM(AbstractModel):
    def __init__(self, subject_interface: SubjectDiagSequenceJAXInterface,
                 diag_gram: DAGGRAM, state_size: int):

        self.subject_interface = subject_interface
        self.diag_gram = diag_gram

        self.dimensions = {
            'diag_gram': diag_gram.basic_embeddings_dim,
            'diag_dag': len(subject_interface.diag_multi_ccs_idx),
            'diag_out': len(subject_interface.diag_single_ccs_idx),
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
        diag_gram = jnp.zeros(self.dimensions['diag_gram'])

        return {
            "diag_gram": self.diag_gram.init_params(rng_key),
            "gru": self.initializers['gru'](rng_key, diag_gram, state),
            "out": self.initializers['out'](rng_key, state)
        }

    def state_size(self):
        return self.dimensions['state']

    def diag_out_index(self) -> List[str]:
        index2code = {
            i: c
            for c, i in self.subject_interface.diag_single_ccs_idx.items()
        }
        return list(map(index2code.get, range(len(index2code))))

    def __call__(self, params: Any, subjects_batch: List[int], **kwargs):

        G = self.diag_gram.compute_embedding_mat(params["diag_gram"])
        gram = partial(self.diag_gram.encode, G)
        diag_seqs = self.subject_interface.diag_sequences_batch(subjects_batch)

        loss = {}
        diag_detectability = {}
        state0 = jnp.zeros(self.dimensions['state'])
        for subject_id, _diag_seqs in diag_seqs.items():
            # Exclude last one for irrelevance
            hierarchical_diag = _diag_seqs['diag_multi_ccs_vec'][:-1]
            # Exclude first one, we need to predict them for a future step.
            flat_diag = _diag_seqs['diag_single_ccs_vec'][1:]
            gram_seqs = map(gram, hierarchical_diag)

            diag_detectability[subject_id] = {}
            loss[subject_id] = []
            state = state0
            for i, diag_gram in enumerate(gram_seqs):
                y_i = flat_diag[i]
                output, state = self.gru(params['gru'], diag_gram, state)
                logits = self.out(params['out'], output)
                diag_detectability[subject_id][i] = {
                    'diag_true': y_i,
                    'pre_logits': logits
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
    def create_patient_interface(processed_mimic_tables_dir: str):
        static_df = pd.read_csv(
            f'{processed_mimic_tables_dir}/static_df.csv.gz')
        adm_df = pd.read_csv(f'{processed_mimic_tables_dir}/adm_df.csv.gz')
        diag_df = pd.read_csv(f'{processed_mimic_tables_dir}/diag_df.csv.gz',
                              dtype={'ICD9_CODE': str})
        proc_df = pd.read_csv(f'{processed_mimic_tables_dir}/proc_df.csv.gz',
                              dtype={'ICD9_CODE': str})
        # Cast columns of dates to datetime64
        static_df['DOB'] = pd.to_datetime(
            static_df.DOB, infer_datetime_format=True).dt.normalize()
        adm_df['ADMITTIME'] = pd.to_datetime(
            adm_df.ADMITTIME, infer_datetime_format=True).dt.normalize()
        adm_df['DISCHTIME'] = pd.to_datetime(
            adm_df.DISCHTIME, infer_datetime_format=True).dt.normalize()

        patients = Subject.to_list(static_df, adm_df, diag_df, proc_df, None)

        # CCS Knowledge Graph
        k_graph = CCSDAG()

        return SubjectDiagSequenceJAXInterface(patients, set(), k_graph)

    @classmethod
    def create_model(cls, config, patient_interface, train_ids):
        diag_glove, proc_glove = glove_representation(
            diag_idx=patient_interface.diag_multi_ccs_idx,
            proc_idx=patient_interface.proc_multi_ccs_idx,
            ccs_dag=patient_interface.dag,
            subjects=[patient_interface.subjects[i] for i in train_ids],
            **config['glove'])

        diag_gram = DAGGRAM(
            ccs_dag=patient_interface.dag,
            code2index=patient_interface.diag_multi_ccs_idx,
            basic_embeddings=diag_glove,
            ancestors_mat=patient_interface.diag_multi_ccs_ancestors_mat,
            **config['gram']['diag'])

        return cls(subject_interface=patient_interface,
                   diag_gram=diag_gram,
                   **config['model'])

    @staticmethod
    def sample_training_config(trial: optuna.Trial):
        return AbstractModel._sample_training_config(trial, epochs=10)


if __name__ == '__main__':
    from .hpo_utils import capture_args, run_trials
    kwargs = {'model_class': GRAM, **capture_args()}
    run_trials(**kwargs)
