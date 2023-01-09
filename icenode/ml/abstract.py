"""Abstract class for predictive EHR models."""

from __future__ import annotations
from typing import Dict, List, Any, TYPE_CHECKING
from abc import ABC, abstractmethod, ABCMeta

from functools import partial
from absl import logging
import jax
import equinox as eqx
import optax

if TYPE_CHECKING:
    import optuna

from .. import utils
from .. import embeddings as E
from .. import ehr
from .. import metric
from .trainer import minibatch_trainer


class AbstractModel(eqx.Module, metaclass=ABCMeta):
    model_cls = {}

    @classmethod
    def register_model(cls, label):
        cls.model_cls[label] = cls

    @abstractmethod
    def __call__(self, params: Any, subjects_batch: List[int], **kwargs):
        pass

    def subject_embeddings(self, opt_obj: Any, batch: List[int]):
        params = self.get_params(opt_obj)
        out = self(params, batch, return_embeddings=True)
        return {
            i: out['risk_prediction'].get_subject_embeddings(i)
            for i in batch
        }

    @abstractmethod
    def detailed_loss(self, loss_mixing, params, res):
        pass

    @abstractmethod
    def eval_stats(self, res):
        pass

    def eval(self, opt_obj: Any, batch: List[int]) -> Dict[str, float]:
        loss_mixing = opt_obj[-1]
        params = self.get_params(opt_obj)
        res = self(params, batch)

        return {
            'loss': self.detailed_loss(loss_mixing, params, res),
            'stats': self.eval_stats(res),
            'risk_prediction': res['risk_prediction']
        }

    def admissions_auc_scores(self, model_state: Any, batch: List[int]):
        params = self.get_params(model_state)
        res = self(params, batch)
        return metric.admissions_auc_scores(res['risk_prediction'])

    def codes_auc_scores(self, model_state: Any, batch: List[int]):
        params = self.get_params(model_state)
        res = self(params, batch)
        return metric.codes_auc_scores(res['risk_prediction'])

    def loss(self, loss_mixing: Dict[str, float], params: Any,
             batch: List[int], **kwargs) -> float:
        res = self(params, batch, **kwargs)
        return self.detailed_loss(loss_mixing, params, res)['loss']

    @staticmethod
    def optimizer_class(label: str):
        if label == 'adam':
            return optax.adam
        if label == 'sgd':
            return optax.sgd
        if label == 'adamax':
            return optax.adamax

    @staticmethod
    def lr_schedule(lr, decay_rate):
        if decay_rate is None:
            return lr
        return optax.exponential_decay(lr,
                                       decay_steps=50,
                                       decay_rate=decay_rate)

    @classmethod
    def init_optimizer(cls, config, params):
        c = config['training']
        lr = cls.lr_schedule(c['lr'], c['decay_rate'])
        opt_cls = cls.optimizer_class(c['optimizer'])
        opt_init, opt_update, get_params = opt_cls(step_size=lr)
        opt_state = opt_init(params)
        return opt_state, opt_update, get_params

    @classmethod
    def step_optimizer(cls, step, model_state, batch):
        opt_state, opt_update, get_params, loss_, loss_mixing = model_state
        params = get_params(opt_state)
        grads = jax.grad(loss_)(params, batch)
        opt_state = opt_update(step, grads, opt_state)
        return opt_state, opt_update, get_params, loss_, loss_mixing

    def init_with_params(self, config: Dict[str, Any], params: Any):
        opt_state, opt_update, get_params = self.init_optimizer(config, params)
        loss_mixing = config['training']['loss_mixing']
        loss_ = partial(self.loss, loss_mixing)
        return opt_state, opt_update, get_params, loss_, loss_mixing

    def init(self, config: Dict[str, Any], prng_seed: int = 0):
        params = self.init_params(prng_seed)
        return self.init_with_params(config, params)

    @classmethod
    def parameters_size(cls, opt_object):
        params = cls.get_params(opt_object)
        return utils.parameters_size(params)

    @classmethod
    def create_embedding(cls, emb_config, emb_kind, subject_interface,
                         train_ids):
        if emb_kind == 'matrix':
            return E.MatrixEmbeddings(input_dim=subject_interface.dx_dim,
                                      **emb_config)

        if emb_kind == 'gram':
            return E.GRAM(category='dx',
                          subject_interface=subject_interface,
                          train_ids=train_ids,
                          **emb_config)
        else:
            raise RuntimeError(f'Unrecognized Embedding kind {emb_kind}')

    @staticmethod
    def dx_outcome_partitions(subject_interface: ehr.Subject_JAX,
                              train_ids: List[int]):
        return subject_interface.dx_outcome_by_percentiles(20, train_ids)

    @classmethod
    def select_loss(cls, loss_label: str, subject_interface: ehr.Subject_JAX,
                    train_ids: List[int], dx_scheme: str):
        if loss_label == 'balanced_focal':
            return lambda t, p: metric.balanced_focal_bce(
                t, p, gamma=2, beta=0.999)
        elif loss_label == 'softmax_logits_bce':
            return metric.softmax_logits_bce
        elif loss_label == 'bce':
            return metric.bce
        elif loss_label == 'balanced_bce':
            codes_dist = subject_interface.dx_code_frequency_vec(
                train_ids, dx_scheme)
            weights = codes_dist.sum() / (codes_dist + 1e-1) * len(codes_dist)
            return lambda t, logits: metric.weighted_bce(t, logits, weights)
        else:
            raise ValueError(f'Unrecognized dx_loss: {loss_label}')

    @classmethod
    def sample_training_config(cls, trial: optuna.Trial):
        l_mixing = {
            'L_l1': 0,  #trial.suggest_float('l1', 1e-8, 5e-3, log=True),
            'L_l2': 0  # trial.suggest_float('l2', 1e-8, 5e-3, log=True),
        }

        return {
            'epochs': 10,
            'batch_size': trial.suggest_int('B', 2, 27, 5),
            'optimizer': 'adam',
            'lr': trial.suggest_float('lr', 5e-5, 5e-3, log=True),
            'decay_rate': None,
            'loss_mixing': l_mixing
        }

    @classmethod
    def sample_embeddings_config(cls, trial: optuna.Trial, emb_kind: str):
        if emb_kind == 'matrix':
            emb_config = E.MatrixEmbeddings.sample_model_config('dx', trial)
        elif emb_kind == 'gram':
            emb_config = E.GRAM.sample_model_config('dx', trial)
        else:
            raise RuntimeError(f'Unrecognized Embedding kind {emb_kind}')

        return {'dx': emb_config, 'kind': emb_kind}

    @classmethod
    def sample_model_config(cls, trial: optuna.Trial):
        return {'state_size': trial.suggest_int('s', 100, 350, 50)}

    @classmethod
    def sample_experiment_config(cls, trial: optuna.Trial, emb_kind: str):
        return {
            'emb': cls.sample_embeddings_config(trial, emb_kind),
            'model': cls.sample_model_config(trial),
            'training': cls.sample_training_config(trial)
        }

    @staticmethod
    def get_trainer():
        return minibatch_trainer
