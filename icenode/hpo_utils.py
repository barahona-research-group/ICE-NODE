"""Hyperparameter optimization of EHR predictive models"""

import os
import argparse
import random
from pathlib import Path
from datetime import datetime, timedelta
import copy
from absl import logging
from tqdm import tqdm

import jax

import optuna
from optuna.storages import RDBStorage
from optuna.pruners import HyperbandPruner, PatientPruner
from optuna.samplers import TPESampler
from optuna.integration import MLflowCallback
from sqlalchemy.pool import NullPool
import mlflow

from .metrics import evaluation_table
from .utils import (write_config)
from .abstract_model import AbstractModel


class ResourceTimeout(Exception):
    """Raised when a trial is anticipated to be exceeding the timelimit of the compute."""


class StudyHalted(Exception):
    """Raised when a trial is spawned from a retired study."""


def mlflow_callback_noexcept(callback):
    """MLFlow callback with supressed exceptions."""

    def apply(study, trial):
        """wrapping."""
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

    parser.add_argument(
        '-d',
        '--data-tag',
        required=True,
        help='Data identifier tag (m3 for MIMIC-III or m4 for MIMIC-IV')

    parser.add_argument(
        '-e',
        '--emb',
        required=True,
        help=
        'Embedding method to use (matrix|orthogonal_gram|glove_gram|semi_frozen_gram|frozen_gram|tunable_gram)'
    )

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
        'data_tag': args.data_tag,
        'emb': args.emb,
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


def objective(model_cls: AbstractModel, emb: str, pretrained_components,
              patient_interface, train_ids, test_ids, valid_ids, rng, job_id,
              output_dir, trial_stop_time: datetime, frozen: bool,
              trial: optuna.Trial):
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
    config = model_cls.sample_experiment_config(
        trial, emb_kind=emb, pretrained_components=pretrained_components)

    try:
        mlflow.log_params(trial.params)
    except Exception as e:
        logging.warning(f'Supressed error when logging config sample: {e}')

    write_config(config, os.path.join(trial_dir, 'config.json'))

    logging.info(f'Trial {trial.number} HPs: {trial.params}')

    model: AbstractModel = model_cls.create_model(config, patient_interface,
                                                  train_ids,
                                                  pretrained_components)

    code_partitions = model.code_partitions(patient_interface, train_ids)

    m_state = model.init(config)
    logging.info('[DONE] Sampling & Initializing Models')

    trial.set_user_attr('parameters_size', model.parameters_size(m_state))
    mlflow_set_tag('parameters_size', model.parameters_size(m_state), frozen)
    batch_size = config['training']['batch_size']
    batch_size = min(batch_size, len(train_ids))

    epochs = config['training']['epochs']
    iters = round(epochs * len(train_ids) / batch_size)
    trial.set_user_attr('steps', iters)
    mlflow_set_tag('steps', iters, frozen)

    best_score = 0.0
    for i in tqdm(range(iters)):
        eval_step = round((i + 1) * 100 / iters)
        last_step = round(i * 100 / iters)

        if datetime.now() > trial_stop_time:
            trial.set_user_attr('timeout', 1)
            mlflow_set_tag('timeout', 1, frozen)
            break

        rng.shuffle(train_ids)
        train_batch = train_ids[:batch_size]

        m_state = model.step_optimizer(eval_step, m_state, train_batch)
        if model.hasnan(m_state):
            logging.warning('NaN detected')
            trial.set_user_attr('nan', 1)
            mlflow_set_tag('nan', 1, frozen)
            raise optuna.TrialPruned()
            # return float('nan')

        if eval_step == last_step and i < iters - 1:
            continue

        study_attrs = trial.study.user_attrs
        if study_attrs['halt']:
            trial.set_user_attr('halted', 1)
            raise StudyHalted('Study is halted')

        trial.set_user_attr("progress", eval_step)
        mlflow_set_tag("progress", eval_step, frozen)

        if frozen or i == iters - 1:
            raw_res = {
                'TRN': model.eval(m_state, train_batch),
                'VAL': model.eval(m_state, valid_ids),
                'TST': model.eval(m_state, test_ids)
            }
        else:
            raw_res = {
                'TRN': model.eval(m_state, train_batch),
                'VAL': model.eval(m_state, valid_ids)
            }

        eval_df, eval_flat = evaluation_table(raw_res, code_partitions)
        logging.info(eval_df)
        auc = eval_df.loc['MICRO-AUC', 'VAL']

        try:
            mlflow_log_metrics(eval_flat, eval_step, frozen)
        except Exception as e:
            logging.warning(f'Exception when logging metrics to mlflow: {e}')

        eval_df.to_csv(os.path.join(trial_dir,
                                    f'step{eval_step:04d}_eval.csv'))

        # Only dump parameters for frozen trials.
        if frozen or i == iters - 1 or auc > best_score:
            fname = os.path.join(trial_dir,
                                 f'step{eval_step:04d}_params.pickle')
            model.write_params(m_state, fname)
            best_score = auc

        trial.report(auc, eval_step)
        if study_attrs['enable_prune'] and trial.should_prune():
            raise optuna.TrialPruned()
    return auc


def run_trials(model_cls: AbstractModel, pretrained_components: str,
               study_name: str, optuna_store: str, mlflow_store: str,
               num_trials: int, mimic_processed_dir: str, data_tag: str,
               emb: str, output_dir: str, cpu: bool, job_id: str,
               trials_time_limit: int, training_time_limit: int):

    data_tag_fullname = {'M3': 'MIMIC-III', 'M4': 'MIMIC-IV'}

    termination_time = datetime.now() + timedelta(hours=trials_time_limit)
    logging.set_verbosity(logging.INFO)
    logging.info('[LOADING] patient interface')
    patient_interface = model_cls.create_patient_interface(mimic_processed_dir,
                                                           data_tag=data_tag)
    logging.info('[DONE] patient interface')

    storage = RDBStorage(url=optuna_store,
                         engine_kwargs={'poolclass': NullPool})
    study = optuna.create_study(study_name=study_name,
                                direction="maximize",
                                storage=storage,
                                load_if_exists=True,
                                sampler=TPESampler(),
                                pruner=PatientPruner(HyperbandPruner(),
                                                     patience=50))

    study.set_user_attr('metric', 'MICRO-AUC')
    study.set_user_attr('data', data_tag_fullname[data_tag])
    study.set_user_attr('embeddings', emb)

    study_attrs = study.user_attrs
    # A flag that we can control from the DB to halt training on all machines.
    # (False initiallly)
    if 'halt' not in study_attrs:
        study.set_user_attr('halt', False)

    # A flag that we can control from the DB to enable/disable pruning (True
    # initially)
    if 'enable_prune' not in study_attrs:
        study.set_user_attr('enable_prune', True)

    # A description text to be altered from the DB to describe the differences
    # between tags.
    if 'description' not in study_attrs:
        study.set_user_attr('description', "No description")

    if cpu:
        jax.config.update('jax_platform_name', 'cpu')
    else:
        jax.config.update('jax_platform_name', 'gpu')

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # splits = train:val:test = 0.7:.15:.15
    train_ids, valid_ids, test_ids = patient_interface.random_splits(
        split1=0.7, split2=0.85, random_seed=42)

    def objective_f(trial: optuna.Trial):
        study_attrs = study.user_attrs
        if study_attrs['halt']:
            trial.set_user_attr('halted', 1)
            raise StudyHalted('Study is halted')

        trial_stop_time = datetime.now() + timedelta(hours=training_time_limit)
        if trial_stop_time + timedelta(minutes=20) > termination_time:
            trial.set_user_attr('timeout', 1)
            raise ResourceTimeout('Time-limit exceeded, abort.')

        return objective(model_cls=model_cls,
                         emb=emb,
                         pretrained_components=pretrained_components,
                         patient_interface=patient_interface,
                         train_ids=train_ids,
                         test_ids=test_ids,
                         valid_ids=valid_ids,
                         rng=random.Random(42),
                         job_id=job_id,
                         output_dir=output_dir,
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
