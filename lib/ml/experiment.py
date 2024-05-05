from typing import Optional, List, Dict, Any, Type

from .embeddings import AdmissionEmbeddingsConfig
from .in_models import InpatientModelConfig
from .model import (InpatientModel)
from .trainer import (TrainerConfig, ReportingConfig,
                      TrainerReporting, LossMixer, Trainer)
from ..base import Config, Module
from ..ehr import (TVxEHR)


class ExperimentConfig(Config):
    trainer: TrainerConfig
    trainer_classname: str
    model: InpatientModelConfig
    embeddings: AdmissionEmbeddingsConfig
    model_classname: str
    metrics: List[Dict[str, Any]]
    reporting: ReportingConfig
    n_evals: int
    model_snapshot_frequency: int
    continue_training: bool
    warmup: Optional[TrainerConfig] = None
    loss_mixer: Optional[LossMixer] = None


class Experiment(Module):
    config: ExperimentConfig
    prng_seed: int = 42
    num_workers: int = 8

    def load_model(self, interface: TVxEHR) -> InpatientModel:
        model_class: Type[InpatientModel] = Module._class_registry[self.config.model_classname]
        return model_class.from_tvx_ehr(interface,
                                        model_config=self.config.model,
                                        embeddings_config=self.config.embeddings,
                                        seed=self.prng_seed)

    def load_metrics(self, interface, splits):
        external_kwargs = {'patients': interface, 'train_split': splits[0]}

        metrics = []
        for config in self.config.metrics:
            metrics.append(
                Module.import_module(config=config, **external_kwargs))

        return metrics

    def load_trainer(self) -> Trainer:
        trainer_class: Type[Trainer] = Module._class_registry[self.config.trainer_classname]
        return trainer_class(config=self.config.trainer,
                             loss_mixer=self.config.loss_mixer)

    def load_reporting(self, interface, splits):
        metrics = self.load_metrics(interface, splits)
        return TrainerReporting(self.config.reporting, metrics)

    def run(self):
        # Load interface
        interface = self.load_interface()

        # Load model
        model = self.load_model(interface)

        # Load trainer
        trainer = self.load_trainer()

        reporting = self.load_reporting(interface, splits)

        return trainer(
            model=model,
            patients=interface,
            splits=None,
            reporting=reporting,
            n_evals=self.config.n_evals,
            model_snapshot_frequency=self.config.model_snapshot_frequency,
            continue_training=self.config.continue_training,
            exported_config=self.config.to_dict(),
            prng_seed=self.prng_seed)
