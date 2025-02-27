import logging
import os
import random
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple

import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .experiment import Experiment
from ..base import Config, Module
from ..db.models import (Experiment as ExperimentModel, EvaluationRun as
EvaluationRunModel, EvaluationStatus as
                         EvaluationStatusModel, Results as ResultsModel, Metric
                         as MetricModel, get_or_create, create_tables)
from ..ehr import TVxEHR
from ..metric.metrics import (MetricsCollection)
from ..utils import (load_config, zip_members)


def is_x64_enabled():
    x = jrandom.normal(jrandom.PRNGKey(0), (20,), dtype=jnp.float64)
    return x.dtype == jnp.float64


class EvaluationConfig(Config):
    metrics: List[Dict[str, Any]] = tuple()
    experiments_dir: str = 'experiments'
    frequency: int = 100
    db: str = 'evaluation.db'
    max_duration: int = 24 * 3  # in hours


# def no_config_metrics

class Evaluation(Module):
    config: EvaluationConfig

    @property
    def db_url(self) -> str:
        expr_abs_path = os.path.abspath(self.config.experiments_dir)
        return f'sqlite+pysqlite:///{expr_abs_path}/{self.config.db}'

    def load_metrics(self) -> MetricsCollection:
        return MetricsCollection(metrics=tuple(Module.import_module(config=config) for config in self.config.metrics))

    def filter_params(self, params_list: List[str]) -> List[str]:
        numbers = {}

        # Extract Dict[num, fname] from params_list
        for fname in params_list:
            num = re.findall(r'\d+', fname)
            assert len(num) == 1
            num = int(num[0])
            numbers[num] = fname

        # Filter using frequency
        numbers = {
            k // self.config.frequency: v
            for k, v in sorted(numbers.items())
        }

        # Return filtered list
        return [v for k, v in sorted(numbers.items())]

    @property
    def params_list(self) -> Dict[str, List[str]]:
        return {
            exp: zip_members(os.path.join(d, 'params.zip'))
            for exp, d in self.experiment_dir.items()
        }

    @property
    def experiment_dir(self) -> Dict[str, str]:
        experiments_dirs = [
            os.path.join(self.config.experiments_dir, d)
            for d in os.listdir(self.config.experiments_dir)
            if os.path.isdir(os.path.join(self.config.experiments_dir, d))
               and os.path.exists(
                os.path.join(self.config.experiments_dir, d, 'config.json'))
               and os.path.exists(
                os.path.join(self.config.experiments_dir, d, 'params.zip'))
        ]

        return {os.path.basename(d): d for d in experiments_dirs}

    @property
    def experiment_config(self) -> Dict[str, Config]:
        return {
            exp: Config.from_dict(load_config(os.path.join(d, 'config.json')))
            for exp, d in self.experiment_dir.items()
        }

    def generate_experiment_params_pairs(self) -> List[Tuple[str, str]]:
        exp_params = {
            exp: self.filter_params(params)
            for exp, params in self.params_list.items()
        }

        pairs = []
        while any(len(params) > 0 for params in exp_params.values()):
            shuffled_exps = list(exp_params.keys())
            random.shuffle(shuffled_exps)
            for exp in shuffled_exps:
                if len(exp_params[exp]) > 0:
                    pairs.append((exp, exp_params[exp].pop()))

        return pairs

    def save_metrics(self, engine: Engine, exp: str, snapshot: str,
                     metrics: Dict[str, float]):

        # Add metrics if they don't exist
        with Session(engine) as session, session.begin():
            for metric_name in metrics.keys():
                get_or_create(engine, MetricModel, name=metric_name)

        with Session(engine) as session, session.begin():
            evaluation = session.query(EvaluationRunModel).filter(
                EvaluationRunModel.experiment.has(name=exp),
                EvaluationRunModel.snapshot == snapshot).one()
            evaluation_id = evaluation.id
            metric_id = {
                metric.name: metric.id
                for metric in session.query(MetricModel).filter(
                    MetricModel.name.in_(metrics.keys())).all()
            }
            for metric_name, metric_value in metrics.items():
                result = ResultsModel(evaluation_id=evaluation_id,
                                      metric_id=metric_id[metric_name],
                                      value=metric_value)
                session.add(result)

    def get_experiment(self, exp: str) -> Experiment:
        return Experiment(config=self.experiment_config[exp])

    def evaluate(self, exp: str, snapshot: str, tvx_ehr: TVxEHR) -> Dict[str, float]:
        experiment = self.get_experiment(exp)
        metrics = self.load_metrics()
        model = experiment.load_model(tvx_ehr, 42)
        model = model.load_params_from_archive(os.path.join(self.experiment_dir[exp], 'params.zip'), snapshot)
        subjects_list = tvx_ehr.subject_ids # list of subject_ids
        # randomly select at most 1000 subjects.
        subjects_list = random.sample(subjects_list, min(600, len(subjects_list)))
        predictions = model.batch_predict(tvx_ehr.device_batch(subjects_list))
        return metrics(predictions).as_df(snapshot).iloc[0].to_dict()

    def run_evaluation(self, engine: Engine, exp: str, snapshot: str, tvx_ehr: TVxEHR):
        # Look for a RUNNING job, if exists and took too long, delete it.

        with Session(engine) as session, session.begin():
            running_eval = session.query(EvaluationRunModel).filter(EvaluationRunModel.experiment.has(name=exp),
                                                                    EvaluationRunModel.snapshot == snapshot,
                                                                    EvaluationRunModel.status.has(
                                                                        name='RUNNING')).one_or_none()
            if running_eval is not None:
                if running_eval.status.name == 'FINISHED':
                    logging.info(f'Evaluation {exp} {snapshot} already finished.')
                    return
                if (datetime.now() - running_eval.created_at).total_seconds() / 3600 > self.config.max_duration:
                    logging.info(f'Evaluation {exp} {snapshot} took too long. Restart.')
                    session.delete(running_eval)
                else:
                    logging.info(f'Evaluation {exp} {snapshot} already running.')
                    return

        with Session(engine) as session, session.begin():
            running_status_model = get_or_create(engine, EvaluationStatusModel, name='RUNNING')
            experiment_model = get_or_create(engine, ExperimentModel, name=exp)
            new_eval = EvaluationRunModel(experiment=experiment_model, snapshot=snapshot, status=running_status_model)
            session.add(new_eval)

        self.save_metrics(engine, exp, snapshot, self.evaluate(exp, snapshot, tvx_ehr))
        with Session(engine) as session, session.begin():
            finished_status = get_or_create(engine, EvaluationStatusModel, name='FINISHED')
            new_eval = session.query(EvaluationRunModel).filter(EvaluationRunModel.experiment.has(name=exp),
                                                                EvaluationRunModel.snapshot == snapshot).one()
            new_eval.status = finished_status

    def start(self, tvx_ehr_path: str):
        logging.info('Database URL: %s', self.db_url)
        engine = create_engine(self.db_url)
        create_tables(engine)
        tvx = TVxEHR.load(tvx_ehr_path)

        for exp, snapshot in self.generate_experiment_params_pairs():
            try:
                eqx.clear_caches()
                jax.clear_caches()
                jax.clear_backends()
                logging.info(f'Running {exp}, {snapshot}')
                self.run_evaluation(engine, exp=exp, snapshot=snapshot, tvx_ehr=tvx)
            except IntegrityError as e:
                logging.warning(f'Possible: evaluation already exists: {exp}, {snapshot}')
                logging.warning(e)
