import os
import argparse
import random
from pathlib import Path
from typing import Iterable, Dict, Any, Optional, Tuple
from functools import partial
from datetime import datetime, timedelta
import pickle
import copy
from absl import logging
from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax.experimental import optimizers

import optuna
from optuna.trial import FrozenTrial
from optuna.storages import RDBStorage
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from optuna.integration import MLflowCallback
from sqlalchemy.pool import NullPool
import mlflow

from .metrics import evaluation_table
from .utils import (parameters_size, tree_hasnan, write_config, write_params)
from .abstract_model import AbstractModel


class ResourceTimeout(Exception):
    pass


def mlflow_callback_noexcept(callback):
    def apply(study, trial):
        try:
            return callback(study, trial)
        except Exception as e:
            logging.warning(f'MLFlow Exception supressed: {e}')

    return apply


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
    parser.add_argument('--trials-time-limit',
                        type=int,
                        required=True,
                        help='Number of maximum hours for all trials')
    parser.add_argument(
        '--training-time-limit',
        type=int,
        required=True,
        help='Number of maximum hours for training in single trial')

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

    parser.add_argument('--pretrained-components', required=False)

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
        'cpu': args.cpu,
        'trials_time_limit': args.trials_time_limit,
        'training_time_limit': args.training_time_limit,
        'pretrained_components': args.pretrained_components
    }


def mlflow_set_tag(key, value, frozen):
    if not frozen:
        mlflow.set_tag(key, value)


def mlflow_log_metrics(eval_dict, step, frozen):
    if not frozen:
        mlflow.log_metrics(eval_dict, step=step)


def objective(model_cls: AbstractModel, pretrained_components,
              patient_interface, train_ids, test_ids, valid_ids, rng,
              eval_freq, job_id, output_dir, codes_by_percentiles,
              trial_stop_time: datetime, frozen: bool, trial: optuna.Trial):
    trial.set_user_attr('job_id', job_id)

    mlflow_set_tag('job_id', job_id, frozen)
    mlflow_set_tag('trial_number', trial.number, frozen)

    if frozen:
        trial_dir = os.path.join(output_dir,
                                 f'frozen_trial_{trial.number:03d}')
    else:
        trial_dir = os.path.join(output_dir, f'trial_{trial.number:03d}')

    Path(trial_dir).mkdir(parents=True, exist_ok=True)

    trial.set_user_attr('dir', trial_dir)

    logging.info('[LOADING] Sampling & Initializing Models')
    config = model_cls.sample_experiment_config(trial, pretrained_components)

    try:
        mlflow.log_params(trial.params)
    except Exception as e:
        logging.warning(f'Supressed error when logging config sample: {e}')

    write_config(config, os.path.join(trial_dir, 'config.json'))

    logging.info(f'Trial {trial.number} HPs: {trial.params}')

    model = model_cls.create_model(config, patient_interface, train_ids,
                                   pretrained_components)

    prng_key = jax.random.PRNGKey(rng.randint(0, 100))
    params = model.init_params(prng_key)
    logging.info('[DONE] Sampling & Initializing Models')

    trial.set_user_attr('parameters_size', parameters_size(params))
    mlflow_set_tag('parameters_size', parameters_size(params), frozen)

    loss_mixing = config['training']['loss_mixing']
    lr = config['training']['lr']
    if config['training']['optimizer'] == 'adam':
        optimizer = optimizers.adam
    else:
        optimizer = optimizers.sgd

    opt_init, opt_update, get_params = optimizer(step_size=lr)
    opt_state = opt_init(params)

    loss_ = partial(model.loss, loss_mixing)
    eval_ = partial(model.eval, loss_mixing)

    def update(
            step: int, batch: Iterable[int],
            opt_state: optimizers.OptimizerState) -> optimizers.OptimizerState:
        params = get_params(opt_state)
        grads = jax.grad(loss_)(params, batch)
        return opt_update(step, grads, opt_state)

    batch_size = config['training']['batch_size']
    batch_size = min(batch_size, len(train_ids))

    epochs = config['training']['epochs']
    iters = int(epochs * len(train_ids) / batch_size)
    trial.set_user_attr('steps', iters)
    mlflow_set_tag('steps', iters, frozen)

    for step in tqdm(range(iters)):

        if datetime.now() > trial_stop_time:
            trial.set_user_attr('timeout', 1)
            mlflow_set_tag('timeout', 1, frozen)
            break

        rng.shuffle(train_ids)
        train_batch = train_ids[:batch_size]

        opt_state = update(step, train_batch, opt_state)
        if tree_hasnan(get_params(opt_state)):
            trial.set_user_attr('nan', 1)
            mlflow_set_tag('nan', 1, frozen)
            return float('nan')

        if not (step % eval_freq == 0 or step == iters - 1):
            continue

        trial.set_user_attr("progress", (step + 1) / iters)
        mlflow_set_tag("progress", (step + 1) / iters, frozen)

        params = get_params(opt_state)

        if frozen or step == iters - 1:
            raw_res = {
                'TRN': eval_(params, train_batch),
                'VAL': eval_(params, valid_ids),
                'TST': eval_(params, test_ids)
            }
        else:
            raw_res = {
                'TRN': eval_(params, train_batch),
                'VAL': eval_(params, valid_ids)
            }

        eval_df, eval_flat = evaluation_table(raw_res, codes_by_percentiles)
        try:
            mlflow_log_metrics(eval_flat, step, frozen)
        except Exception as e:
            logging.warning(f'Exception when logging metrics to mlflow: {e}')

        eval_df.to_csv(os.path.join(trial_dir, f'step{step:04d}_eval.csv'))

        # Only dump parameters for frozen trials.
        if frozen:
            fname = os.path.join(trial_dir, f'step{step:04d}_params.pickle')
            write_params(get_params(opt_state), fname)

        logging.info(eval_df)

        auc = eval_df.loc['MICRO-AUC', 'VAL']

        # nan is returned when no predictions actually made.
        if jnp.isnan(auc):
            continue

        trial.report(auc, step)

        if trial.should_prune():
            raise optuna.TrialPruned()
    return auc


def run_trials(model_cls: AbstractModel, pretrained_components: str,
               study_name: str, optuna_store: str, mlflow_store: str,
               num_trials: int, mimic_processed_dir: str, output_dir: str,
               cpu: bool, job_id: str, trials_time_limit: int,
               training_time_limit: int):

    termination_time = datetime.now() + timedelta(hours=trials_time_limit)
    logging.set_verbosity(logging.INFO)
    logging.info('[LOADING] patient interface')
    patient_interface = model_cls.create_patient_interface(mimic_processed_dir)
    logging.info('[DONE] patient interface')

    storage = RDBStorage(url=optuna_store,
                         engine_kwargs={'poolclass': NullPool})
    study = optuna.create_study(study_name=study_name,
                                direction="maximize",
                                storage=storage,
                                load_if_exists=True,
                                sampler=TPESampler(),
                                pruner=HyperbandPruner())
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

    eval_freq = 20
    codes_by_percentiles = patient_interface.diag_single_ccs_by_percentiles(
        20, train_ids)

    def objective_f(trial: optuna.Trial):
        trial_stop_time = datetime.now() + timedelta(hours=training_time_limit)
        if trial_stop_time + timedelta(minutes=20) > termination_time:
            trial.set_user_attr('timeout', 1)
            raise ResourceTimeout('Time-limit exceeded, abort.')

        return objective(model_cls=model_cls,
                         pretrained_components=pretrained_components,
                         patient_interface=patient_interface,
                         train_ids=train_ids,
                         test_ids=test_ids,
                         valid_ids=valid_ids,
                         rng=rng,
                         eval_freq=eval_freq,
                         job_id=job_id,
                         output_dir=output_dir,
                         codes_by_percentiles=codes_by_percentiles,
                         trial_stop_time=trial_stop_time,
                         trial=trial,
                         frozen=num_trials <= 0)

    if num_trials > 0:
        mlflow_uri = f'{mlflow_store}_bystudy/{study_name}'
        mlflc = MLflowCallback(tracking_uri=mlflow_uri, metric_name="VAL-AUC")
        study.optimize(mlflc.track_in_mlflow()(objective_f),
                       n_trials=num_trials,
                       callbacks=[mlflow_callback_noexcept(mlflc)],
                       catch=(RuntimeError, ))
    else:
        number = -num_trials
        trials = study.trials
        for trial in trials:
            if trial.number == number:
                trial = copy.deepcopy(trial)
                objective_f(trial)
                return
        raise RuntimeError(f'Trial number {number} not found')
