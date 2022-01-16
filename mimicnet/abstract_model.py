from typing import Dict, List, Any
from enum import Enum, Flag, auto
import optuna
from .gram import DAGGRAM


class LossMixingFlag(Flag):
    NONE = 0
    DIAG = auto()
    NUM = auto()
    ODE = auto()
    DYN = auto()

    @staticmethod
    def has(flag, attr):
        return (flag & attr).value != 0


class AbstractModel:
    def __call__(self, params: Any, subjects_batch: List[int], **kwargs):
        raise Exception('Should be overriden')

    def detailed_loss(self, loss_mixing, params, res):
        raise Exception('Should be overriden')

    def eval_stats(self, res):
        raise Exception('Should be overriden')

    def eval(self, loss_mixing: Dict[str, float], params: Any,
             batch: List[int]) -> Dict[str, float]:
        res = self(params, batch, count_nfe=True)

        return {
            'loss': self.detailed_loss(loss_mixing, params, res),
            'stats': self.eval_stats(res),
            'diag_detectability': res['diag_detectability']
        }

    def loss(self, loss_mixing: Dict[str, float], params: Any,
             batch: List[int]) -> float:
        res = self(params, batch, count_nfe=False)
        return self.detailed_loss(loss_mixing, params, res)['loss']

    @classmethod
    def create_model(cls, config, patient_interface, train_ids):
        raise Exception('Should be overriden')

    @staticmethod
    def _sample_training_config(trial: optuna.Trial, epochs):
        l_mixing = {
            'L_l1': trial.suggest_float('l1', 1e-7, 1e-1, log=True),
            'L_l2': trial.suggest_float('l2', 1e-6, 1e-1, log=True),
        }

        return {
            'epochs': epochs,
            'batch_size': trial.suggest_int('B', 2, 22, 5),
            # UNDO/TODO
            'optimizer': 'adam',
            # 'optimizer': trial.suggest_categorical('opt', ['adam', 'sgd']),
            'lr': trial.suggest_float('lr', 1e-6, 1e-2, log=True),
            'loss_mixing': l_mixing
        }

    @staticmethod
    def sample_gram_config(trial: optuna.Trial):
        return {'diag': DAGGRAM.sample_model_config('dx', trial)}

    @staticmethod
    def sample_training_config(trial: optuna.Trial):
        raise Exception('Should be overriden')

    @staticmethod
    def sample_model_config(trial: optuna.Trial):
        return {'state_size': trial.suggest_int('s', 100, 350, 50)}
