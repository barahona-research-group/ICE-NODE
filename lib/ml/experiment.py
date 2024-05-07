from typing import Optional, List, Dict, Any, Type, Tuple

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
    reporting: ReportingConfig
    model_snapshot_frequency: int
    continue_training: bool
    loss_mixer: LossMixer
    warmup: Optional[TrainerConfig] = None


class Experiment(Module):
    config: ExperimentConfig
    tvx_ehr_path: str
    prng_seed: int = 42

    def load_model(self, interface: TVxEHR) -> InpatientModel:
        model_class: Type[InpatientModel] = Module.module_class(self.config.model_classname)
        return model_class.from_tvx_ehr(interface,
                                        config=self.config.model,
                                        embeddings_config=self.config.embeddings,
                                        seed=self.prng_seed)

    def load_trainer(self) -> Trainer:
        trainer_class: Type[Trainer] = Module.module_class(self.config.trainer_classname)
        return trainer_class(config=self.config.trainer, loss_mixer=self.config.loss_mixer)

    @staticmethod
    def train_split(tvx_ehr: TVxEHR) -> Tuple[str, ...]:
        if tvx_ehr.splits is None:
            return tuple(tvx_ehr.subject_ids)
        return tvx_ehr.splits[0]

    @staticmethod
    def eval_split(tvx_ehr: TVxEHR) -> Tuple[str, ...]:
        if tvx_ehr.splits is None:
            return tuple()
        return tvx_ehr.splits[1]

    def run(self):
        # Load interface
        tvx_ehr = TVxEHR.load(self.tvx_ehr_path)

        # Load model
        model = self.load_model(tvx_ehr)

        # Load trainer
        trainer = self.load_trainer()

        return trainer(
            model=model,
            patients=tvx_ehr,
            train_split=self.train_split(tvx_ehr),
            val_split=self.eval_split(tvx_ehr),
            reporting=TrainerReporting(config=self.config.reporting),
            n_evals=0,
            model_snapshot_frequency=self.config.model_snapshot_frequency,
            continue_training=self.config.continue_training,
            exported_config=self.config.to_dict(),
            prng_seed=self.prng_seed)
