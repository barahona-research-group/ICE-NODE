from absl import logging

import optuna

from .train_snonet_lite import (loss_fn, eval_fn, SNONETLite)
from .jax_interface import create_patient_interface
from .gram import DAGGRAM
from .metrics import EvalFlag
from .glove import glove_representation
from .hpo_utils import (capture_args, run_trials, sample_glove_params,
                        sample_gram_params, sample_training_params)

logging.set_verbosity(logging.INFO)


def sample_config(trial: optuna.Trial):
    training_params = {
        **sample_training_params(trial), 'epochs':
        2,
        'diag_loss':
        trial.suggest_categorical('dx_loss', ['balanced_focal', 'bce']),
        'tay_reg':
        3
    }
    training_params['loss_mixing'].update({
        'diag_alpha':
        trial.suggest_float('dx_alpha', 1e-4, 1, log=True),
        'dyn_reg':
        trial.suggest_float('dyn_reg', 1e-3, 1e3, log=True)
    })
    model_params = {
        'ode_dyn': trial.suggest_categorical(
            'ode_dyn',
            ['mlp', 'gru', 'res']),  # Add depth conditional to 'mlp' or 'res'
        'state_size': trial.suggest_int('s', 100, 350, 50),
        'init_depth': trial.suggest_int('init_d', 1, 5),
        'bias': True,
        'max_odeint_days': trial.suggest_int('mx_ode_ds', 8 * 7, 16 * 7, 7)
    }

    if model_params['ode_dyn'] == 'gru':
        model_params['ode_depth'] = 0
    else:
        model_params['ode_depth'] = trial.suggest_int('ode_d', 1, 5)

    return {
        'glove': sample_glove_params(trial),
        'gram': {
            'diag': sample_gram_params('dx', trial),
            'proc': sample_gram_params('pr', trial)
        },
        'model': model_params,
        'training': training_params
    }


def create_model(config, patient_interface, train_ids):
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

    proc_gram = DAGGRAM(
        ccs_dag=patient_interface.dag,
        code2index=patient_interface.proc_multi_ccs_idx,
        basic_embeddings=proc_glove,
        ancestors_mat=patient_interface.proc_multi_ccs_ancestors_mat,
        **config['gram']['proc'])
    return SNONETLite(subject_interface=patient_interface,
                      diag_gram=diag_gram,
                      proc_gram=proc_gram,
                      **config['model'],
                      tay_reg=config['training']['tay_reg'],
                      diag_loss=config['training']['diag_loss'])


if __name__ == '__main__':
    kwargs = capture_args()
    logging.info('[LOADING] Patients JAX Interface.')
    patient_interface = create_patient_interface(kwargs['mimic_processed_dir'])
    logging.info('[DONE] Patients JAX Interface')

    kwargs.update({
        'eval_flags': EvalFlag.POST | EvalFlag.CM,
        'patient_interface': patient_interface,
        'loss_fn': loss_fn,
        'eval_fn': eval_fn,
        'sample_config': sample_config,
        'create_model': create_model
    })

    run_trials(**kwargs)
