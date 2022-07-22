"""Logistic Regression EHR predictive model based on diagnostic codes in
previous visits."""

from typing import Any, List, Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
import jax
import jax.numpy as jnp
from jaxopt import ProximalGradient
from jaxopt.prox import prox_elastic_net, prox_none
import optuna

from ..ehr_model.jax_interface import (Subject_JAX, WindowedInterface_JAX)
from ..ehr_predictive.abstract import AbstractModel
from ..metric.common_metrics import (softmax_logits_bce,
                                     softmax_logits_weighted_bce,
                                     softmax_logits_balanced_focal_bce)
from ..utils import Unsupported

from .risk import BatchPredictedRisks
from .trainer import onestep_trainer


@jax.jit
def multinomial(logits):
    return jax.nn.softmax(logits)


@jax.jit
def predict_multinomial(p, X):
    return multinomial(X @ p['W'] + p['b'])


@jax.jit
def logreg_loss_multinomial(p, X, y):
    logits = X @ p['W'] + p['b']
    return softmax_logits_bce(y, logits)


@jax.jit
def logreg_loss_multinomial_balanced_focal(p, X, y):
    logits = X @ p['W'] + p['b']
    return softmax_logits_balanced_focal_bce(y, logits)


@jax.jit
def logreg_loss_multinomial_balanced(p, X, y):
    logits = X @ p['W'] + p['b']
    weights = y.shape[0] / (y.sum(axis=0) + 1e-10)
    return softmax_logits_weighted_bce(y, logits, weights)


logreg_loss_multinomial_mode = {
    'none': logreg_loss_multinomial,
    'focal': logreg_loss_multinomial_balanced_focal,
    'balanced': logreg_loss_multinomial_balanced
}


@jax.jit
def prox_elastic_net_with_intercept(p, hyperparams, scaling):
    return {
        'W': prox_elastic_net(p['W'], hyperparams, scaling),
        'b': prox_none(p['b'])
    }


class WindowLogReg(AbstractModel):

    def __init__(
            self,
            subject_interface: Subject_JAX,
            alpha: float,  # [0, \inf]
            beta: float,  #[0, \inf]
            class_weight: str):
        self.interface = WindowedInterface_JAX(subject_interface)

        self.model_config = {
            'fun': logreg_loss_multinomial_mode[class_weight],
            'maxiter': 20000,
            'prox': prox_elastic_net_with_intercept,
            'hyperparams_prox': self.alpha_beta_config(alpha, beta),
        }

    @staticmethod
    def alpha_beta_config(alpha, beta):
        # alpha is for L2-norm, beta is for L1-norm
        return (beta, alpha / (beta + jnp.finfo('float').eps))

    def eval(self, opt_obj: Any, batch: List[int]) -> Dict[str, float]:
        params = self.get_params(opt_obj)
        res = self(params, batch)
        return {
            'loss': {
                'loss': res['loss']
            },
            'stats': {},
            'risk_prediction': res['risk_prediction']
        }

    @classmethod
    def step_optimizer(cls, eval_step, m_state: Any,
                       subjects_batch: List[int]):
        model, params = m_state
        X, y = model.dx_interface.tabular_features(subjects_batch)
        pg = ProximalGradient(fun=model.model_config['fun'],
                              prox=model.model_config['prox'],
                              maxiter=model.model_config['maxiter'],
                              jit=True)
        params = pg.run(
            params,
            hyperparams_prox=model.model_config['hyperparams_prox'],
            X=X,
            y=y).params
        return (model, params)

    def __call__(self, params: Any, subjects_batch: List[int], **kwargs):
        model, params = self.init_with_params(self.model_config, params)

        risk_prediction = BatchPredictedRisks()
        for subj_id in subjects_batch:
            adms = model.interface.interface.subjects[subj_id]
            features = model.interface.win_features[subj_id]

            X = np.vstack([feats.dx_features for feats in features[1:]])
            y = np.vstack([adm.dx_outcome for adm in adms[1:]])
            risk = predict_multinomial(params, X)

            for i, (r, gt) in enumerate(zip(risk, y)):
                risk_prediction.add(subject_id=subj_id,
                                    admission_id=adms[i + 1].admission_id,
                                    index=i + 1,
                                    prediction=r,
                                    ground_truth=gt)

        X, y = model.interface.tabular_features(subjects_batch)
        loss = self.model_config['fun'](params, X, y)

        return {'risk_prediction': risk_prediction, 'loss': loss}

    def detailed_loss(self, loss_mixing, params, res):
        raise Unsupported("Unsupported.")

    def eval_stats(self, res):
        return {}

    def loss(self, loss_mixing: Dict[str, float], params: Any,
             batch: List[int], **kwargs) -> float:
        raise Unsupported("Unsupported.")

    def init_params(self, prng_seed: int = 0):
        n_features = self.interface.n_features()
        n_targets = self.interface.n_targets()
        W = 1e-5 * jnp.ones((n_features, n_targets), dtype=float)
        b = jnp.ones(n_targets, dtype=float)
        return {'W': W, 'b': b}

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
        return (self, params)

    @classmethod
    def get_params(cls, m_state):
        model, params = m_state
        return params

    @classmethod
    def sample_training_config(cls, trial: optuna.Trial):
        raise Unsupported("Unsupported.")

    @classmethod
    def sample_model_config(cls, trial: optuna.Trial):
        return {
            'alpha':
            trial.suggest_loguniform('alpha', 1e-6, 1e3),
            'beta':
            trial.suggest_loguniform('beta', 1e-6, 1e3),
            'class_weight':
            trial.suggest_categorical('class_weight',
                                      ['none', 'balanced', 'focal'])
        }

    @classmethod
    def sample_experiment_config(cls, trial: optuna.Trial, emb_kind: str):
        return {
            'model': cls.sample_model_config(trial),
        }

    @classmethod
    def create_model(cls, config, subject_interface, train_ids):
        return cls(subject_interface=subject_interface, **config['model'])

    @staticmethod
    def get_trainer():
        return onestep_trainer


class WindowLogReg_Sklearn(WindowLogReg):

    def __init__(
            self,
            subject_interface: Subject_JAX,
            alpha: float,  # [0, \inf]
            beta: float,  #[0, \inf]
            class_weight: str):
        if class_weight != 'balanced':
            class_weight = None

        self.dx_interface = WindowedInterface_JAX(subject_interface)
        self.model_config = {
            'penalty': 'elasticnet',
            'solver': 'saga',
            'class_weight': class_weight,
            'max_iter': 2000,
            **self.alpha_beta_config(alpha, beta)
        }
        self.supported_labels = None
        self.n_labels = None

    @staticmethod
    def alpha_beta_config(alpha, beta):
        # alpha is for L2-norm, beta is for L1-norm
        C = 1 / (alpha + beta)
        return {'C': C, 'l1_ratio': beta * C}

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
            adms = model.interface.interface.subjects[subj_id]
            features = model.interface.win_features[subj_id]

            X = np.vstack([feats.dx_features for feats in features[1:]])
            y = np.vstack([adm.dx_outcome for adm in adms[1:]])

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

        return {'risk_prediction': risk_prediction, 'loss': 0}

    def init_params(self, prng_seed: int = 0):
        self.model_config['random_state'] = prng_seed
        skmodel = MultiOutputClassifier(
            LogisticRegression(**self.model_config))
        return skmodel
