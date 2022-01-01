from typing import Iterable, Dict, Any

from absl import logging

import optuna

from .train_gram import (loss_fn, eval_fn, GRAM, create_patient_interface)
from .gram import DAGGRAM
from .metrics import EvalFlag
from .glove import glove_representation
from .hpo_utils import (capture_args, run_trials, sample_model_config_gram)

logging.set_verbosity(logging.INFO)


def create_model(config, patient_interface, train_ids):
    diag_glove, _ = glove_representation(
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

    return GRAM(subject_interface=patient_interface,
                diag_gram=diag_gram,
                **config['model'])


if __name__ == '__main__':
    kwargs = capture_args()
    logging.info('[LOADING] Patients JAX Interface.')
    patient_interface = create_patient_interface(kwargs['mimic_processed_dir'])
    logging.info('[DONE] Patients JAX Interface')

    kwargs.update({
        'eval_flags': EvalFlag.CM,
        'patient_interface': patient_interface,
        'loss_fn': loss_fn,
        'eval_fn': eval_fn,
        'sample_config': sample_model_config_gram,
        'create_model': create_model
    })

    run_trials(**kwargs)

### IMPORTANT: https://github.com/optuna/optuna/issues/1647
