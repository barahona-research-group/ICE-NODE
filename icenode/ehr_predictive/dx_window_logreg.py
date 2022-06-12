"""Logistic Regression EHR predictive model based on diagnostic codes in
previous visits."""

from typing import Any, List, Dict
from absl import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
import optuna

from ..ehr_model.jax_interface import (DxInterface_JAX,
                                       DxWindowedInterface_JAX)
from ..ehr_predictive.abstract import AbstractModel
from ..utils import Unsupported

from .risk import BatchPredictedRisks
from .trainer import sklearn_trainer


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
            'solver': 'saga',
            'C': reg_c,
            'l1_ratio': reg_l1_ratio,
            'class_weight': 'balanced' if balanced else None
        }
        self.supported_labels = None
        self.n_labels = None

    def eval(self, opt_obj: Any, batch: List[int]) -> Dict[str, float]:
        skmodel = self.get_params(opt_obj)
        res = self(skmodel, batch)
        return {
            'loss': {},
            'stats': {},
            'risk_prediction': res['risk_prediction']
        }

    @classmethod
    def step_optimizer(cls, eval_step, m_state: Any,
                       subjects_batch: List[int]):
        model, skmodel = m_state
        X, y = model.dx_interface.tabular_features(subjects_batch)
        # Training with LR requires at least two classes in the training.
        y_mask = (y.sum(axis=0) != 0).squeeze()
        skmodel = skmodel.fit(X, y[:, y_mask])
        model.n_labels = y.shape[1]
        model.supported_labels = y_mask
        return (model, skmodel)

    def __call__(self, skmodel: Any, subjects_batch: List[int], **kwargs):
        model, skmodel = self.init_with_params(self.model_config, skmodel)

        y_mask = model.supported_labels

        risk_prediction = BatchPredictedRisks()
        for subj_id in subjects_batch:
            adms = model.dx_interface.dx_interface.subjects[subj_id]
            features = model.dx_interface.dx_win_features[subj_id]

            X = np.vstack([feats.dx_ccs_features for feats in features[1:]])
            y = np.vstack([adm.dx_flatccs_codes for adm in adms[1:]])

            risk = np.zeros_like(y, dtype=float)
            # .predict_proba function now returns a list of arrays
            # where the length of the list is n_outputs,
            # and each array is (n_samples, n_classes)
            # for that particular output.
            risk[:, y_mask] = np.vstack(
                [o[:, 0] for o in skmodel.predict_proba(X)]).T

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
        skmodel = MultiOutputClassifier(
            LogisticRegression(**self.model_config))
        return skmodel

    @staticmethod
    def optimizer_class(label: str):
        raise Unsupported("Unsupported.")

    @staticmethod
    def lr_schedule(lr, decay_rate):
        raise Unsupported("Unsupported.")

    @classmethod
    def init_optimizer(cls, config, params):
        raise Unsupported("Unsupported.")

    def init_with_params(self, config: Dict[str, Any], skmodel: Any):
        return (self, skmodel)

    @classmethod
    def get_params(cls, m_state):
        model, skmodel = m_state
        return skmodel

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

    @classmethod
    def create_model(cls, config, patient_interface, train_ids,
                     pretrained_components):
        return cls(subject_interface=patient_interface, **config['model'])

    @staticmethod
    def get_trainer():
        return sklearn_trainer
