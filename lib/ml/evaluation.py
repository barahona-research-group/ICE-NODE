from typing import Optional, List, Union, Dict, Any, Tuple
import os
import random
from pathlib import Path
import re
from ..metric import (MetricsCollection, Metric)
from ..utils import (translate_path, write_config, load_config, zip_members)

from ..base import Config, Module
from .trainer import (Trainer, InTrainer, TrainerConfig, ReportingConfig,
                      TrainerReporting, WarmupConfig)
from .experiment import InpatientExperiment


class EvaluationConfig(Config):
    experiments_dir: str
    frequency: int
    db: str
    metrics: List[Dict[str, Any]]


class Evaluation(Module):

    def __init__(self, config: EvaluationConfig, **kwargs):
        if isinstance(config, dict):
            config = Config.from_dict(config)
        super().__init__(config=config, **kwargs)

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
        while all(len(params) > 0 for params in exp_params.values()):
            shuffled_exps = list(exp_params.keys())
            random.shuffle(shuffled_exps)
            for exp in shuffled_exps:
                if len(exp_params[exp]) > 0:
                    pairs.append((exp, exp_params[exp].pop()))

        return pairs

    def save_config(self):
        write_config(self.config.to_dict(),
                     os.path.join(self.config.experiments_dir, 'config.json'))
