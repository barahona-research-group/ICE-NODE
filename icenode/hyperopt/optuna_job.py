"""Hyperparameter optimization of EHR predictive models"""

import os
import argparse
import random
from pathlib import Path
from datetime import datetime, timedelta
import copy
from absl import logging

import jax

import optuna
from optuna.storages import RDBStorage
from optuna.pruners import HyperbandPruner, PatientPruner
from optuna.samplers import TPESampler
from optuna.integration import MLflowCallback
from sqlalchemy.pool import NullPool
import mlflow

from ..utils import (write_config)
from ..ehr_predictive.trainer import (MinibatchTrainReporter, MinibatchLogger,
                                      EvaluationDiskWriter, ParamsDiskWriter)
from ..ehr_predictive.abstract import (AbstractModel)


class ResourceTimeout(Exception):
    """Raised when a trial is anticipated to be exceeding the timelimit of the compute."""


class StudyHalted(Exception):
    """Raised when a trial is spawned from a retired study."""


class OptunaReporter(MinibatchTrainReporter):

    def __init__(self, trial):
        self.trial = trial

    def report_steps(self, steps):
        self.trial.set_user_attr('steps', steps)

    def report_timeout(self):
        self.trial.set_user_attr('timeout', 1)

    def report_progress(self, eval_step):
        self.trial.set_user_attr("progress", eval_step)
        study_attrs = self.trial.study.user_attrs

        if study_attrs['halt']:
            self.trial.set_user_attr('halted', 1)
            raise StudyHalted('Study is halted')

        if study_attrs['enable_prune'] and self.trial.should_prune():
            raise optuna.TrialPruned()

    def report_nan_detected(self):
        self.trial.set_user_attr('nan', 1)
        raise optuna.TrialPruned()

    def report_evaluation(self, eval_step, objective_v, *args):
        self.trial.report(objective_v, eval_step)


class MLFlowReporter(MinibatchTrainReporter):

    def report_steps(self, steps):
        mlflow.set_tag('steps', steps)

    def report_timeout(self):
        mlflow.set_tag('timeout', 1)

    def report_nan_detected(self):
        mlflow.set_tag('nan', 1)

    def report_progress(self, eval_step):
        mlflow.set_tag("progress", eval_step)

    def report_evaluation(self, eval_step, objective_v, evals_df,
                          flat_evals_df):
        try:
            mlflow.log_metrics(flat_evals_df, eval_step)
        except Exception as e:
            logging.warning(f'Exception when logging metrics to mlflow: {e}')


def mlflow_callback_noexcept(callback):
    """MLFlow callback with supressed exceptions."""

    def apply(study, trial):
        """wrapping."""
        try:
            return callback(study, trial)
        except Exception as e:
            logging.warning(f'MLFlow Exception supressed: {e}')

    return apply


def mlflow_set_tag(key, value, frozen):
    if not frozen:
        mlflow.set_tag(key, value)


def mlflow_log_metrics(eval_dict, step, frozen):
    if not frozen:
        mlflow.log_metrics(eval_dict, step=step)


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


def objective(model_cls: AbstractModel, emb: str, pretrained_components,
              patient_interface, splits, rng, job_id, output_dir,
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
    config = model_cls.sample_experiment_config(
        trial, emb_kind=emb, pretrained_components=pretrained_components)

    try:
        mlflow.log_params(trial.params)
    except Exception as e:
        logging.warning(f'Supressed error when logging config sample: {e}')

    write_config(config, os.path.join(trial_dir, 'config.json'))

    logging.info(f'Trial {trial.number} HPs: {trial.params}')

    model = model_cls.create_model(config, patient_interface, splits[0],
                                   pretrained_components)

    code_frequency_groups = model.code_partitions(patient_interface, splits[0])

    m_state = model.init(config)
    logging.info('[DONE] Sampling & Initializing Models')

    trial.set_user_attr('parameters_size', model.parameters_size(m_state))
    mlflow_set_tag('parameters_size', model.parameters_size(m_state), frozen)

    reporters = [
        MinibatchLogger(),
        EvaluationDiskWriter(trial_dir=trial_dir),
        ParamsDiskWriter(trial_dir=trial_dir)
    ]

    if frozen == False:
        reporters.append(MLFlowReporter())

    # Optuna reporter added last because it is the only one that may raise
    # Exceptions.
    reporters.append(OptunaReporter(trial=trial))

    return model.get_trainer()(model=model,
                               m_state=m_state,
                               config=config,
                               splits=splits,
                               rng=rng,
                               code_frequency_groups=code_frequency_groups,
                               trial_terminate_time=trial_stop_time,
                               reporters=reporters)['objective']


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
    splits = patient_interface.random_splits(split1=0.7,
                                             split2=0.85,
                                             random_seed=42)

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
                         splits=splits,
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
