import os
import argparse
import random
from pathlib import Path
from typing import Iterable, Dict, Any
from functools import partial

from absl import logging
from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax.experimental import optimizers

import optuna
from optuna.storages import RDBStorage
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from sqlalchemy.pool import NullPool
from optuna.integration import MLflowCallback

import mlflow

from .metrics import evaluation_table
from .utils import (parameters_size, tree_hasnan, write_config)


def sample_gram_params(prefix: str, trial: optuna.Trial):
    return {
        'attention_method':
        trial.suggest_categorical(f'{prefix}_att_f', ['tanh', 'l2']),
        'attention_dim':
        trial.suggest_int(f'{prefix}_att_d', 100, 250, 50),
    }


def sample_glove_params(trial: optuna.Trial):
    return {
        'diag_vector_size': trial.suggest_int('dx', 100, 250, 50),
        'proc_vector_size': 50,
        'iterations': 30,
        'window_size_days': 2 * 365
    }


def sample_training_params(trial: optuna.Trial):
    return {
        'batch_size': trial.suggest_int('B', 2, 42, 5),
        'optimizer': trial.suggest_categorical('opt', ['adam', 'sgd']),
        'lr': trial.suggest_float('lr', 1e-6, 1e-2, log=True),
        'loss_mixing': {
            'l1_reg': trial.suggest_float('l1', 1e-7, 1e-1, log=True),
            'l2_reg': trial.suggest_float('l2', 1e-6, 1e-1, log=True),
        }
    }


def capture_args():
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

    parser.add_argument(
        '--optuna-store',
        required=True,
        help='Storage URL for optuna records, e.g. for PostgresQL database')

    parser.add_argument(
        '--mlflow-store',
        required=True,
        help='Storage URL for mlflow records, e.g. for PostgresQL database')

    parser.add_argument('--study-name', required=True)

    parser.add_argument('--job-id', required=False)

    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    return {
        'study_name': args.study_name,
        'optuna_store': args.optuna_store,
        'mlflow_store': args.mlflow_store,
        'num_trials': args.num_trials,
        'mimic_processed_dir': args.mimic_processed_dir,
        'output_dir': args.output_dir,
        'job_id': args.job_id or 'unknown',
        'cpu': args.cpu
    }


def objective(sample_config, create_model, patient_interface, train_ids,
              test_ids, valid_ids, rng, eval_freq, loss_fn, eval_fn,
              eval_flags, job_id, output_dir, codes_by_percentiles,
              trial: optuna.Trial):
    trial.set_user_attr('job_id', job_id)
    trial_dir = os.path.join(output_dir, f'trial_{trial.number:03d}')

    trial.set_user_attr('dir', trial_dir)

    Path(trial_dir).mkdir(parents=True, exist_ok=True)

    logging.info('[LOADING] Sampling & Initializing Models')
    config = sample_config(trial)

    mlflow.log_params(trial.params)
    write_config(config, os.path.join(trial_dir, 'config.json'))

    logging.info(f'Trial {trial.number} HPs: {trial.params}')

    model = create_model(config, patient_interface, train_ids)

    prng_key = jax.random.PRNGKey(rng.randint(0, 100))
    params = model.init_params(prng_key)
    logging.info('[DONE] Sampling & Initializing Models')

    trial.set_user_attr('parameters_size', parameters_size(params))

    loss_mixing = config['training']['loss_mixing']
    lr = config['training']['lr']
    if config['training']['optimizer'] == 'adam':
        optimizer = optimizers.adam
    else:
        optimizer = optimizers.sgd

    opt_init, opt_update, get_params = optimizer(step_size=lr)
    opt_state = opt_init(params)

    loss = partial(loss_fn, model, loss_mixing)
    eval_ = partial(eval_fn, model, loss_mixing)

    def update(
            step: int, batch: Iterable[int],
            opt_state: optimizers.OptimizerState) -> optimizers.OptimizerState:
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
        tst_res = eval_(params, test_ids)

        eval_df, eval_dict = evaluation_table(trn_res, val_res, tst_res,
                                              eval_flags, codes_by_percentiles)
        mlflow.log_metrics(eval_dict, step=step)
        eval_df.to_csv(os.path.join(trial_dir, f'step{step:03d}_eval.csv'))
        logging.info(eval_df)

        auc = eval_df.loc['AUC', 'VAL']

        # nan is returned when no predictions actually made.
        if jnp.isnan(auc):
            continue

        trial.report(auc, step)
        trial.set_user_attr("progress", (step + 1) / iters)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return auc


def run_trials(patient_interface, eval_flags, loss_fn, eval_fn, sample_config,
               create_model, study_name: str, optuna_store: str,
               mlflow_store: str, num_trials: int, mimic_processed_dir: str,
               output_dir: str, cpu: bool, job_id: str):

    storage = RDBStorage(url=optuna_store,
                         engine_kwargs={'poolclass': NullPool})
    study = optuna.create_study(study_name=study_name,
                                direction="maximize",
                                storage=storage,
                                load_if_exists=True,
                                sampler=TPESampler(),
                                pruner=HyperbandPruner())

    mlflc = MLflowCallback(tracking_uri=mlflow_store, metric_name="VAL-AUC")

    study.set_user_attr('metric', 'auc')

    if cpu:
        jax.config.update('jax_platform_name', 'cpu')
    else:
        jax.config.update('jax_platform_name', 'gpu')

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)
    subjects_id = list(patient_interface.subjects.keys())
    rng.shuffle(subjects_id)

    # splits = train:val:test = 0.7:.15:.15
    splits = int(.7 * len(subjects_id)), int(.85 * len(subjects_id))

    train_ids = subjects_id[:splits[0]]
    valid_ids = subjects_id[splits[0]:splits[1]]
    test_ids = subjects_id[splits[1]:]

    eval_freq = 1
    codes_by_percentiles = patient_interface.diag_single_ccs_by_percentiles(
        20, train_ids)

    @mlflc.track_in_mlflow()
    def objective_f(trial: optuna.Trial):
        return objective(sample_config=sample_config,
                         create_model=create_model,
                         patient_interface=patient_interface,
                         train_ids=train_ids,
                         test_ids=train_ids,
                         valid_ids=valid_ids,
                         rng=rng,
                         eval_freq=eval_freq,
                         loss_fn=loss_fn,
                         eval_fn=eval_fn,
                         eval_flags=eval_flags,
                         job_id=job_id,
                         output_dir=output_dir,
                         codes_by_percentiles=codes_by_percentiles,
                         trial=trial)

    study.optimize(objective_f, n_trials=num_trials, callbacks=[mlflc])
