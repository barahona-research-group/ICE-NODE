"""Hyperparameter optimization of EHR predictive models."""

import os
from pathlib import Path
from datetime import datetime, timedelta
import copy
from absl import logging

import optuna
from optuna.storages import RDBStorage
from optuna.pruners import HyperbandPruner, PatientPruner
from optuna.samplers import TPESampler
from optuna.integration import MLflowCallback
from sqlalchemy.pool import NullPool
import mlflow

from .cmd_args import get_cmd_parser

from . import ehr
from . import ml
from . import embeddings as E

cli_args = [
    '--model', '--dataset', '--emb', '--output-dir', '--num-trials',
    '--trials-time-limit', '--training-time-limit', '--optuna-store',
    '--mlflow-store', '--study-tag', '--job-id', '--dx-scheme',
    '--pr-scheme', '--dx-outcome'
]


class ResourceTimeout(Exception):
    """Raised when a trial is anticipated to be exceeding the timelimit of the compute."""


class StudyHalted(Exception):
    """Raised when a trial is spawned from a retired study."""


class OptunaReporter(ml.AbstractReporter):

    def __init__(self, trial):
        self.trial = trial

    def report_params_size(self, size):
        self.trial.set_user_attr('parameters_size', size)

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


class MLFlowReporter(ml.AbstractReporter):

    def report_params_size(self, size):
        mlflow.set_tag('parameters_size', size)

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


def objective(model: str, emb: str, subject_interface, job_id, study_dir,
              trial_terminate_time: datetime, frozen: bool,
              trial: optuna.Trial):
    trial.set_user_attr('job_id', job_id)
    mlflow_set_tag('job_id', job_id, frozen)
    mlflow_set_tag('trial_number', trial.number, frozen)

    if frozen:
        trial_dir = os.path.join(study_dir, f'frozen_trial_{trial.number:03d}')
    else:
        trial_dir = os.path.join(study_dir, f'trial_{trial.number:03d}')

    Path(trial_dir).mkdir(parents=True, exist_ok=True)
    trial.set_user_attr('dir', trial_dir)

    logging.info('[LOADING] Sampling Hyperparameters')
    m_cls = ml.model_cls[model]
    config = m_cls.sample_experiment_config(trial, emb_kind=emb)
    logging.info('[DONE] Sampling Hyperparameters')

    logging.info(f'Trial {trial.number} HPs: {trial.params}')
    try:
        mlflow.log_params(trial.params)
    except Exception as e:
        logging.warning(f'Supressed error when logging config sample: {e}')

    reporters = [
        ml.MinibatchLogger(),
        ml.EvaluationDiskWriter(output_dir=trial_dir),
        ml.ParamsDiskWriter(output_dir=trial_dir),
        ml.ConfigDiskWriter(output_dir=trial_dir)
    ]
    if frozen == False:
        reporters.append(MLFlowReporter())

    # Optuna reporter added last because it is the only one that may raise
    # Exceptions.
    reporters.append(OptunaReporter(trial=trial))

    # splits = train:val:test = 0.7:.15:.15
    splits = subject_interface.random_splits(split1=0.7,
                                             split2=0.85,
                                             random_seed=42)

    return ml.train_with_config(model=model,
                                config=config,
                                subject_interface=subject_interface,
                                splits=splits,
                                rng_seed=42,
                                trial_terminate_time=trial_terminate_time,
                                reporters=reporters)


if __name__ == '__main__':
    args = get_cmd_parser(cli_args).parse_args()
    study_name = f'{args.study_tag}{args.dataset}_{args.model}_{E.short_tag[args.emb]}'
    study_dir = os.path.join(args.output_dir, study_name)

    terminate_time = datetime.now() + timedelta(hours=args.trials_time_limit)
    logging.set_verbosity(logging.INFO)
    logging.info('[LOADING] patient interface')
    dataset = ehr.datasets[args.dataset]
    code_scheme = {
        'dx': args.dx_scheme,
        'dx_outcome': args.dx_outcome,
        'pr': args.pr_scheme
    }
    subject_interface = ehr.Subject_JAX.from_dataset(dataset, code_scheme)
    logging.info('[DONE] patient interface')

    storage = RDBStorage(url=args.optuna_store,
                         engine_kwargs={'poolclass': NullPool})
    study = optuna.create_study(study_name=study_name,
                                direction="maximize",
                                storage=storage,
                                load_if_exists=True,
                                sampler=TPESampler(),
                                pruner=PatientPruner(HyperbandPruner(),
                                                     patience=50))

    study.set_user_attr('metric', 'MICRO-AUC')
    study.set_user_attr('data', dataset.name)
    study.set_user_attr('embeddings', args.emb)

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

    def objective_f(trial: optuna.Trial):
        study_attrs = study.user_attrs
        if study_attrs['halt']:
            trial.set_user_attr('halted', 1)
            raise StudyHalted('Study is halted')

        trial_terminate_time = datetime.now() + timedelta(
            hours=args.training_time_limit)
        if trial_terminate_time + timedelta(minutes=20) > terminate_time:
            trial.set_user_attr('timeout', 1)
            raise ResourceTimeout('Time-limit exceeded, abort.')

        return objective(model=args.model,
                         emb=args.emb,
                         subject_interface=subject_interface,
                         job_id=args.job_id,
                         study_dir=study_dir,
                         trial_terminate_time=trial_terminate_time,
                         trial=trial,
                         frozen=args.num_trials <= 0)

    if args.num_trials > 0:
        mlflow_uri = f"{args.mlflow_store}_bystudy/{study_name}"
        mlflc = MLflowCallback(tracking_uri=mlflow_uri, metric_name="VAL-AUC")
        study.optimize(mlflc.track_in_mlflow()(objective_f),
                       n_trials=args.num_trials,
                       callbacks=[mlflow_callback_noexcept(mlflc)],
                       catch=(RuntimeError, ))
    else:
        number = -args.num_trials
        trials = study.trials
        trial_number_found = False
        for trial in trials:
            if trial.number == number:
                trial = copy.deepcopy(trial)
                objective_f(trial)
                trial_number_found = True
        if trial_number_found == False:
            raise RuntimeError(f'Trial number {number} not found')
