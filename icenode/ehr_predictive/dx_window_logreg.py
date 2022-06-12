"""Logistic Regression EHR predictive model based on diagnostic codes in
previous visits."""

from typing import Any, List, Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
import optuna

from ..ehr_model.jax_interface import (DxInterface_JAX,
                                       DxWindowedInterface_JAX)
from ..ehr_predictive.abstract import AbstractModel
from ..utils import Unsupported

from .risk import BatchPredictedRisks
from .trainer import batch_trainer


class WindowLogReg(AbstractModel):

    def __init__(
            self,
            subject_interface: DxInterface_JAX,
            reg_c: float,  # [0, \inf]
            reg_l1_ratio: float,  #[0, 1]
            balanced: bool):
        self.dx_interface = DxWindowedInterface_JAX(subject_interface)
        self.model_config = {
            'penalty': 'elasticnet',
            'multi_class': 'multinomial',
            'solver': 'saga',
            'C': reg_c,
            'l1_ratio': reg_l1_ratio,
            'class_weight': 'balanced' if balanced else None
        }

    def eval(self, opt_obj: Any, batch: List[int]) -> Dict[str, float]:
        params = self.get_params(opt_obj)
        res = self(params, batch)
        return {
            'loss': {},
            'stats': {},
            'risk_prediction': res['risk_prediction']
        }

    @classmethod
    def step_optimizer(self, eval_step, m_state: Any,
                       subjects_batch: List[int]):
        X, y = m_state.dx_interface.tabular_features(subjects_batch)
        m_state = m_state.fit(X, y)
        return m_state

    def __call__(self, params: Any, subjects_batch: List[int], **kwargs):
        model = self.init_with_params(self.model_config, params)
        risk_prediction = BatchPredictedRisks()
        for subj_id in subjects_batch:
            adms = self.dx_interface.dx_interface.subjects[subj_id]
            features = self.dx_interface.dx_win_features[subj_id]

            X = np.vstack([feats.dx_ccs_features for feats in features[1:]])
            y = np.vstack([adm.dx_flatccs_jax for adm in adms[1:]])
            risk = model.decision_function(X)

            for i, (r, gt) in enumerate(zip(risk, y)):
                risk_prediction.add(subject_id=subj_id,
                                    admission_id=adms[i + 1].admission_id,
                                    index=i + 1,
                                    prediction=r,
                                    ground_truth=gt)

        return {'risk_prediction': risk_prediction}

    def detailed_loss(self, loss_mixing, params, res):
        raise Unsupported("Unsupported.")

    def eval_stats(self, res):
        return {}

    def loss(self, loss_mixing: Dict[str, float], params: Any,
             batch: List[int], **kwargs) -> float:
        raise Unsupported("Unsupported.")

    def init_params(self, prng_seed: int = 0):
        self.model_config['random_state'] = prng_seed
        m_state = LogisticRegression(**self.model_config)
        return m_state.get_params()

    @staticmethod
    def optimizer_class(label: str):
        raise Unsupported("Unsupported.")

    @staticmethod
    def lr_schedule(lr, decay_rate):
        raise Unsupported("Unsupported.")

    @classmethod
    def init_optimizer(cls, config, params):
        raise Unsupported("Unsupported.")

    def init_with_params(self, config: Dict[str, Any], params: Any):
        m_state = LogisticRegression(**self.model_config)
        m_state.set_params(**params)
        return m_state

    @classmethod
    def get_params(cls, m_state):
        return m_state.get_params()

    @classmethod
    def sample_training_config(cls, trial: optuna.Trial):
        raise Unsupported("Unsupported.")

    @classmethod
    def sample_model_config(cls, trial: optuna.Trial):
        return {
            'reg_c': trial.suggest_loguniform('C', 1e-3, 1e3),
            'reg_l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
            'balanced': trial.suggest_categorical('weight', True, False)
        }

    @classmethod
    def sample_experiment_config(cls, trial: optuna.Trial, emb_kind: str,
                                 pretrained_components: str):
        return {
            'model': cls.sample_model_config(trial),
        }

    @staticmethod
    def get_trainer():
        return batch_trainer
