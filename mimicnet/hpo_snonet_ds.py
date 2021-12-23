import argparse
import os
from pathlib import Path
import random
from typing import Iterable
from functools import partial

from absl import logging
from tqdm import tqdm

import jax
from jax.experimental import optimizers

import optuna
from optuna.storages import RDBStorage
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from sqlalchemy.pool import NullPool

from .train_snonet_ds import (loss_fn, eval_fn, SNONETDiagStat)
from .jax_interface import create_patient_interface, Ignore
from .gram import DAGGRAM
from .metrics import evaluation_table, EvalFlag
from .utils import (parameters_size, tree_hasnan, write_config)
from .glove import glove_representation

logging.set_verbosity(logging.INFO)


def sample_config(trial: optuna.Trial):
    config = {
        'glove_config': {
            'diag_vector_size': trial.suggest_int('diag_vector_size', 100, 250,
                                                  50),
            'proc_vector_size': 50,
            'iterations': 30,
            'window_size_days': 2 * 365
        },
        'gram_config': {
            'diag': {
                'attention_method':
                trial.suggest_categorical('attention_method', ['tanh', 'l2']),
                'attention_dim':
                trial.suggest_int('attention_dim', 100, 250, 50)
            }
        },
        'model': {
            'ode_dyn':
            trial.suggest_categorical(
                'ode_dyn', ['mlp', 'gru', 'res'
                            ]),  # Add depth conditional to 'mlp' or 'res'
            'state_size':
            trial.suggest_int('state_size', 100, 500, 50),
            'init_depth':
            trial.suggest_int('init_depth', 1, 5),
            'bias':
            True,
            'max_odeint_days':
            trial.suggest_int('max_odeint_days', 8 * 7, 16 * 7, 7)
        },
        'training': {
            'batch_size':
            trial.suggest_int('batch_size', 5, 25, 5),
            'epochs':
            2,
            'lr':
            trial.suggest_float('lr', 1e-5, 1e-2, log=True),
            'diag_loss':
            trial.suggest_categorical('diag_loss', ['balanced_focal', 'bce']),
            'tay_reg':
            3,  # Order of regularized derivative of the dynamics function (None for disable).
            'loss_mixing': {
                'diag_alpha': trial.suggest_float('diag_alpha',
                                                  1e-4,
                                                  1,
                                                  log=True),
                'l1_reg': trial.suggest_float('l1_reg', 1e-7, 1e-1, log=True),
                'l2_reg': trial.suggest_float('l2_reg', 1e-6, 1e-1, log=True),
                'dyn_reg': trial.suggest_float('dyn_reg', 1e-3, 1e3, log=True)
            }
        }
    }

    if config['model']['ode_dyn'] == 'gru':
        config['model']['ode_depth'] = 0
    else:
        config['model']['ode_depth'] = trial.suggest_int('ode_depth', 1, 5)

    return config


def run_trials(study_name: str, store_url: str, num_trials: int,
               mimic_processed_dir: str, output_dir: str, cpu: bool,
               job_id: str):

    storage = RDBStorage(url=store_url, engine_kwargs={'poolclass': NullPool})
    study = optuna.create_study(study_name=study_name,
                                direction="maximize",
                                storage=storage,
                                load_if_exists=True,
                                sampler=TPESampler(),
                                pruner=HyperbandPruner())

    if cpu:
        jax.config.update('jax_platform_name', 'cpu')
    else:
        jax.config.update('jax_platform_name', 'gpu')

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logging.info('[LOADING] Patients JAX Interface.')
    patient_interface = create_patient_interface(mimic_processed_dir,
                                                 Ignore.TESTS)
    logging.info('[DONE] Patients JAX Interface')

    rng = random.Random(42)
    subjects_id = list(patient_interface.subjects.keys())
    rng.shuffle(subjects_id)

    # splits = train:val:test = 0.7:.15:.15
    splits = int(.7 * len(subjects_id)), int(.85 * len(subjects_id))

    train_ids = subjects_id[:splits[0]]
    valid_ids = subjects_id[splits[0]:splits[1]]
    codes_by_percentiles = patient_interface.diag_single_ccs_by_percentiles(
        20, train_ids)
    eval_freq = 1

    def objective(trial: optuna.Trial):
        trial.set_user_attr('job_id', job_id)

        trial_dir = os.path.join(output_dir, f'trial_{trial.number:03d}')
        Path(trial_dir).mkdir(parents=True, exist_ok=True)

        logging.info('[LOADING] Sampling & Initializing Models')

        config = sample_config(trial)
        write_config(config, os.path.join(trial_dir, 'config.json'))

        logging.info(f'Trial {trial.number} HPs: {trial.params}')
        diag_glove, _ = glove_representation(
            diag_idx=patient_interface.diag_multi_ccs_idx,
            proc_idx=patient_interface.proc_multi_ccs_idx,
            ccs_dag=patient_interface.dag,
            subjects=[patient_interface.subjects[i] for i in train_ids],
            **config['glove_config'])

        diag_gram = DAGGRAM(
            ccs_dag=patient_interface.dag,
            code2index=patient_interface.diag_multi_ccs_idx,
            basic_embeddings=diag_glove,
            ancestors_mat=patient_interface.diag_multi_ccs_ancestors_mat,
            **config['gram_config']['diag'])

        ode_model = SNONETDiagStat(subject_interface=patient_interface,
                                   diag_gram=diag_gram,
                                   **config['model'],
                                   tay_reg=config['training']['tay_reg'],
                                   diag_loss=config['training']['diag_loss'])

        prng_key = jax.random.PRNGKey(rng.randint(0, 100))
        params = ode_model.init_params(prng_key)

        trial.set_user_attr('parameters_size', parameters_size(params))
        logging.info('[DONE] Sampling & Initializing Models')

        lr = config['training']['lr']
        loss_mixing = config['training']['loss_mixing']
        opt_init, opt_update, get_params = optimizers.adam(step_size=lr)
        opt_state = opt_init(params)
        loss = partial(loss_fn, ode_model, loss_mixing)
        eval_ = partial(eval_fn, ode_model, loss_mixing)

        def update(
                step: int, batch: Iterable[int],
                opt_state: optimizers.OptimizerState
        ) -> optimizers.OptimizerState:
            params = get_params(opt_state)
            grads = jax.grad(loss)(params, batch)
            return opt_update(step, grads, opt_state)

        batch_size = config['training']['batch_size']
        batch_size = min(batch_size, len(train_ids))

        epochs = config['training']['epochs']
        iters = int(epochs * len(train_ids) / batch_size)
        trial.set_user_attr('steps', iters)

        for step in tqdm(range(iters)):
            rng.shuffle(train_ids)
            train_batch = train_ids[:batch_size]

            opt_state = update(step, train_batch, opt_state)
            if tree_hasnan(get_params(opt_state)):
                trial.set_user_attr('nan', 1)
                return float('nan')

            if step % eval_freq != 0: continue

            params = get_params(opt_state)
            trn_res = eval_(params, train_batch)
            val_res = eval_(params, valid_ids)
            eval_df = evaluation_table(trn_res, val_res,
                                       EvalFlag.CM | EvalFlag.POST,
                                       codes_by_percentiles)
            eval_df.to_csv(os.path.join(trial_dir, f'step{step:03d}_eval.csv'))
            logging.info(eval_df)

            auc = eval_df.loc['AUC', 'Validation']
            trial.report(auc, step)
            trial.set_user_attr(f'eval{step:03d}', eval_df.to_json())
            trial.set_user_attr("progress", (step + 1) / iters)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return auc

    study.optimize(objective, n_trials=num_trials)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--mimic-processed-dir',
                        required=True,
                        help='Absolute path to MIMIC-III processed tables')
    parser.add_argument('-o',
                        '--output-dir',
                        required=True,
                        help='Aboslute path to log intermediate results')
    parser.add_argument('-n',
                        '--num-trials',
                        type=int,
                        required=True,
                        help='Number of HPO trials.')

    parser.add_argument('-s',
                        '--store-url',
                        required=True,
                        help='Storage URL, e.g. for PostgresQL database')

    parser.add_argument('--study-name', required=True)

    parser.add_argument('--job-id', required=False)

    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    study_name = args.study_name
    store_url = args.store_url
    num_trials = args.num_trials
    mimic_processed_dir = args.mimic_processed_dir
    output_dir = args.output_dir
    job_id = args.job_id or 'unknown'
    cpu = args.cpu

    run_trials(study_name=study_name,
               store_url=store_url,
               num_trials=num_trials,
               mimic_processed_dir=mimic_processed_dir,
               output_dir=output_dir,
               job_id=job_id,
               cpu=cpu)

### IMPORTANT: https://github.com/optuna/optuna/issues/1647
