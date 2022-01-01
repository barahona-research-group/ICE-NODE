from typing import Iterable, Dict, Any
from absl import logging

import optuna

from .train_snonet_diag import (loss_fn, eval_fn, SNONETDiag)
from .jax_interface import create_patient_interface, Ignore
from .gram import DAGGRAM
from .metrics import EvalFlag
from .glove import glove_representation
from .hpo_utils import (capture_args, run_trials, sample_model_config_ode)

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

    return SNONETDiag(subject_interface=patient_interface,
                      diag_gram=diag_gram,
                      **config['model'],
                      tay_reg=config['training']['tay_reg'],
                      diag_loss=config['training']['diag_loss'])


if __name__ == '__main__':
    kwargs = capture_args()
    logging.info('[LOADING] Patients JAX Interface.')
    patient_interface = create_patient_interface(kwargs['mimic_processed_dir'],
                                                 Ignore.TESTS)
    logging.info('[DONE] Patients JAX Interface')

    kwargs.update({
        'eval_flags': EvalFlag.POST | EvalFlag.CM,
        'patient_interface': patient_interface,
        'loss_fn': loss_fn,
        'eval_fn': eval_fn,
        'sample_config': sample_model_config_ode,
        'create_model': create_model
    })

    run_trials(**kwargs)
