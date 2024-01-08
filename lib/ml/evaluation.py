from typing import Optional, List, Union, Dict, Any, Tuple
import os
import random
from pathlib import Path
import re
import logging
from datetime import datetime
import time

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from ..metric import (MetricsCollection, Metric)
from ..utils import (translate_path, write_config, load_config, zip_members)

from ..base import Config, Module
from ..db.models import (Base, Experiment as ExperimentModel, EvaluationRun as
                         EvaluationRunModel, EvaluationStatus as
                         EvaluationStatusModel, Results as ResultsModel, Metric
                         as MetricModel, get_or_create, create_tables)

from .trainer import (Trainer, InTrainer, TrainerConfig, ReportingConfig,
                      TrainerReporting, WarmupConfig)
from .experiment import InpatientExperiment


class EvaluationConfig(Config):
    metrics: List[Dict[str, Any]] = tuple()
    experiments_dir: str = 'experiments'
    frequency: int = 100
    db: str = 'sqlite:///evaluation.db'
    max_duration: int = 24 * 3  # in hours


class Evaluation(Module):

    def __init__(self, config: EvaluationConfig, **kwargs):
        if isinstance(config, dict):
            config = Config.from_dict(config)
        super().__init__(config=config, **kwargs)

    def load_metrics(self, interface, splits):
        external_kwargs = {'patients': interface, 'train_split': splits[0]}

        metrics = []
        for config in self.config.metrics:
            metrics.append(
                Module.import_module(config=config, **external_kwargs))

        return MetricsCollection(metrics)

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
        l = [
            os.path.join(self.config.experiments_dir, d)
            for d in os.listdir(self.config.experiments_dir)
            if os.path.isdir(os.path.join(self.config.experiments_dir, d))
            and os.path.exists(
                os.path.join(self.config.experiments_dir, d, 'config.json'))
        ]
        return {os.path.basename(d): d for d in l}

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

    def run_evaluation(self, engine: Engine, exp: str, snapshot: str):
        with Session(engine) as session, session.begin():
            running_status = get_or_create(engine,
                                           EvaluationStatusModel,
                                           name='RUNNING')
            experiment = get_or_create(engine, ExperimentModel, name=exp)

            running_eval = session.query(EvaluationRunModel).filter(
                EvaluationRunModel.experiment.has(name=exp),
                EvaluationRunModel.snapshot == snapshot,
                EvaluationRunModel.status.has(name='RUNNING')).one_or_none()
            running_hours = lambda: (datetime.now() - running_eval.created_at
                                     ).total_seconds() / 3600
            if running_eval is not None and running_hours(
            ) > self.config.max_duration:
                running_hours = (datetime.now() - running_eval.created_at
                                 ).total_seconds() / 3600
                if running_hours > self.config.max_duration:
                    logging.info(
                        f'Evaluation {exp} {snapshot} took too long. Restart.')
            else:
                new_eval = EvaluationRunModel(experiment=experiment,
                                              snapshot=snapshot,
                                              status=running_status)
                session.add(new_eval)

        experiment = InpatientExperiment(config=self.experiment_config[exp])
        interface = experiment.load_interface()
        splits = experiment.load_splits(interface.dataset)
        metrics = self.load_metrics(interface, splits)
        model = experiment.load_model(interface)
        model = model.load_params_from_archive(
            os.path.join(self.experiment_dir[exp], 'params.zip'), snapshot)

        _, val_split, _ = splits
        predictions = model.batch_predict(interface.device_batch(val_split))
        results = metrics.to_df(snapshot, predictions).iloc[0].to_dict()
        self.save_metrics(engine, exp, snapshot, results)

        with Session(engine) as session, session.begin():
            finished_status = get_or_create(engine,
                                            EvaluationStatusModel,
                                            name='FINISHED')
            new_eval = session.query(EvaluationRunModel).filter(
                EvaluationRunModel.experiment.has(name=exp),
                EvaluationRunModel.snapshot == snapshot).one()
            new_eval.status = finished_status

    def start(self):
        engine = create_engine(self.config.db)
        create_tables(engine)
        for exp, snapshot in self.generate_experiment_params_pairs():
            try:
                self.run_evaluation(engine, exp=exp, snapshot=snapshot)
            except IntegrityError as e:

                logging.warning(
                    f'Evaluation already exists: {exp}, {snapshot}')
