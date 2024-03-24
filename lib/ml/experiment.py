from typing import Optional, List, Dict, Any

import jax.random as jrandom

from .model import (InpatientModel, ModelConfig)
from .trainer import (TrainerConfig, ReportingConfig,
                      TrainerReporting)
from ..base import Config, Module
from ..ehr import (TVxEHR, DatasetConfig,
                   TVxEHRConfig)


class SplitConfig(Config):
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1
    balanced: str = 'admissions'


class ExperimentConfig(Config):
    dataset: DatasetConfig
    interface: TVxEHRConfig
    split: SplitConfig
    trainer: TrainerConfig
    trainer_classname: str
    model: ModelConfig
    model_classname: str
    metrics: List[Dict[str, Any]]
    reporting: ReportingConfig
    n_evals: int
    model_snapshot_frequency: int
    continue_training: bool
    warmup: Optional[TrainerConfig] = None
    reg_hyperparams: Optional[Config] = None


class Experiment(Module):
    prng_seed: int = 42
    num_workers: int = 8

    def __init__(self,
                 config: ExperimentConfig,
                 prng_seed: int = 42,
                 **kwargs):
        self.prng_seed = prng_seed
        if isinstance(config, dict):
            config = Config.from_dict(config)
        super().__init__(config=config, **kwargs)

    @staticmethod
    def inpatient(config):
        model_class = Module._class_registry[config.model_classname]
        return issubclass(model_class, InpatientModel)

    def load_splits(self, dataset):
        p_splits = [
            self.config.split.train,
            self.config.split.train + self.config.split.val
        ]
        return dataset.random_splits(p_splits,
                                     random_seed=self.prng_seed,
                                     balance=self.config.split.balanced)

    def load_dataset(self, dataset_config):
        return load_dataset(config=dataset_config)

    def load_interface(self):
        return TVxEHR.try_load_cached(self.config.interface,
                                      dataset_config=self.config.dataset,
                                      dataset_generator=self.load_dataset,
                                      num_workers=self.num_workers)

    def load_model(self, interface):
        key = jrandom.PRNGKey(self.prng_seed)
        model_class = Module._class_registry[self.config.model_classname]
        return model_class(
            self.config.model,
            schemes=interface.schemes,
            demographic_vector_config=self.config.interface.demographic,
            key=key)

    def load_metrics(self, interface, splits):
        external_kwargs = {'patients': interface, 'train_split': splits[0]}

        metrics = []
        for config in self.config.metrics:
            metrics.append(
                Module.import_module(config=config, **external_kwargs))

        return metrics

    def load_trainer(self):
        trainer_class = Module._class_registry[self.config.trainer_classname]
        return trainer_class(self.config.trainer,
                             reg_hyperparams=self.config.reg_hyperparams)

    def load_reporting(self, interface, splits):
        metrics = self.load_metrics(interface, splits)
        return TrainerReporting(self.config.reporting, metrics)

    def run(self):
        # Load interface
        interface = self.load_interface()

        splits = self.load_splits(interface.dataset)

        # Load model
        model = self.load_model(interface)

        # Load trainer
        trainer = self.load_trainer()

        reporting = self.load_reporting(interface, splits)

        return trainer(
            model=model,
            patients=interface,
            splits=splits,
            reporting=reporting,
            n_evals=self.config.n_evals,
            model_snapshot_frequency=self.config.model_snapshot_frequency,
            continue_training=self.config.continue_training,
            exported_config=self.config.to_dict(),
            prng_seed=self.prng_seed)


class InpatientExperiment(Experiment):

    def load_dataset(self, dataset_config):
        dataset = load_dataset(config=dataset_config)
        # Use training-split for fitting the outlier_remover and the scalers.

        splits = self.load_splits(dataset)
        # Outlier removal
        outlier_remover = dataset.fit_outlier_remover(splits[0])
        dataset = dataset.remove_outliers(outlier_remover)

        # Scale
        scalers = dataset.fit_scalers(splits[0])
        return dataset.apply_scalers(scalers)

    def load_model(self, interface):
        key = jrandom.PRNGKey(self.prng_seed)
        model_class = Module._class_registry[self.config.model_classname]
        return model_class(
            self.config.model,
            schemes=interface.schemes,
            demographic_vector_config=self.config.interface.demographic,
            leading_observable_config=self.config.interface.leading_observable,
            key=key)
