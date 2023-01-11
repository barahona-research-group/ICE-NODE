import os
import copy
import random
from typing import List, Any, Dict, Type, Tuple
from datetime import datetime
from abc import ABC, abstractmethod, ABCMeta

from absl import logging
from tqdm import tqdm
import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import optuna
import optax

from .abstract_model import AbstractModel

from .reporters import AbstractReporter
from ..ehr import Subject_JAX
from ..metric import BatchPredictedRisks, admissions_auc_scores, codes_auc_scores
from ..utils import params_size, tree_hasnan

opts = {'sgd': optax.sgd, 'adam': optax.adam, 'fromage': optax.fromage}


class AbstractTrainer(eqx.Module):

    opt: Any = eqx.static_field()
    reg_hyperparams: Dict[str, float]
    model_cls: Type[AbstractModel]
    epochs: int
    batch_size: int
    lr: float

    trainer_registry = {}

    @classmethod
    def register_trainer(cls, model_cls):
        cls.trainer_registry[model_cls] = cls

    @classmethod
    def get_trainer(cls, model_cls):
        return cls.trainer_registry[model_cls]

    @classmethod
    def dx_loss(y: jnp.ndarray, dx_logits: jnp.ndarray):
        return -jnp.sum(y * jax.nn.log_softmax(dx_logits) +
                        (1 - y) * jnp.log(1 - jax.nn.softmax(dx_logits)))

    def unreg_loss(self,
                   model: AbstractModel,
                   subject_interface: Subject_JAX,
                   batch: List[int],
                   args: Dict[str, Any] = dict()):
        res = model(subject_interface, batch, args)
        l = res['predictions'].prediction_loss(self.dx_loss)
        return l, ({'dx_loss': l}, res['predictions'])

    def reg_loss(self,
                 model: AbstractModel,
                 subject_interface: Subject_JAX,
                 batch: List[int],
                 args: Dict[str, Any] = dict()):
        res = model(subject_interface, batch, args)
        l = res['predictions'].prediction_loss(self.dx_loss)
        l1_loss = model.l1()
        l2_loss = model.l2()
        l1_alpha = self.reg_hyperparams['L_l1']
        l2_alpha = self.reg_hyperparams['L_l2']

        loss = l + (l1_alpha * l1_loss) + (l2_alpha * l2_loss)

        return loss, ({
            'dx_loss': l,
            'loss': loss,
            'l1_loss': l1_loss,
            'l2_loss': l2_loss,
        }, res['predictions'])

    def loss(self,
             model: AbstractModel,
             subject_interface: Subject_JAX,
             batch: List[int],
             args: Dict[str, Any] = dict()):
        if self.reg_hyperparams is None:
            return self.unreg_loss(model, subject_interface, batch, args)
        else:
            return self.reg_loss(model, subject_interface, batch, args)

    def eval(self,
             model: AbstractModel,
             subject_interface: Subject_JAX,
             batch: List[int],
             args=dict()) -> Dict[str, float]:
        args['eval_only'] = True
        _, loss, preds = self.unreg_loss(model, subject_interface, batch, args)

        return {
            'loss': loss,
            'adm_scores': admissions_auc_scores(preds),
            'code_scores': codes_auc_scores(preds),
            'predictions': preds
        }

    @staticmethod
    def lr_schedule(lr, decay_rate):
        if decay_rate is None:
            return lr
        return optax.exponential_decay(lr,
                                       transition_steps=50,
                                       decay_rate=decay_rate)

    def init_opt(self, model):
        opt = opts[self.opt](self.lr)
        return opt, opt.init(eqx.filter(model, eqx.is_inexact_array))

    def step_optimizer(self, opt_state: Any, model: AbstractModel,
                       subject_interface: Subject_JAX, batch: Tuple[int],
                       key: "jax.random.PRNGKey"):
        opt, opt_state = opt_state
        grad_f = eqx.filter_grad(self.loss, has_aux=True)
        grads, aux = grad_f(model, batch, key)
        updates, opt_state = opt.update(grads, opt_state)
        new_model = eqx.apply_updates(model, updates)

        return (opt, opt_state), new_model, aux

    @classmethod
    def sample_reg_hyperparams(cls, trial: optuna.Trial):
        return {
            'L_l1': trial.suggest_float('l1', 1e-8, 5e-3, log=True),
            'L_l2': trial.suggest_float('l2', 1e-8, 5e-3, log=True)
        }

    @classmethod
    def sample_training_config(cls, trial: optuna.Trial):

        return {
            'epochs': 10,
            'batch_size': trial.suggest_int('B', 2, 27, 5),
            'optimizer': 'adam',
            'lr': trial.suggest_float('lr', 5e-5, 5e-3, log=True),
            'decay_rate': None,
            'reg_hyperparams': cls.sample_reg_hyperparams(trial)
        }

    def __call__(self,
                 model,
                 subject_interface: Subject_JAX,
                 splits: Tuple[List[int], ...],
                 prng_seed: int = 0,
                 code_frequency_groups=None,
                 trial_terminate_time=datetime.max,
                 reporters: List[AbstractReporter] = []):
        train_ids, valid_ids, test_ids = splits
        # Because shuffling is done in-place.
        train_ids = copy.deepcopy(train_ids)
        batch_size = min(self.batch_size, len(train_ids))
        iters = round(self.epochs * len(train_ids) / batch_size)
        opt_state = self.init_opt(model)

        for r in reporters:
            r.report_config()
            r.report_params_size(params_size(model))
            r.report_steps(iters)

        auc = 0.0
        best_score = 0.0
        for i in tqdm(range(iters)):
            eval_step = round((i + 1) * 100 / iters)
            last_step = round(i * 100 / iters)

            if datetime.now() > trial_terminate_time:
                [r.report_timeout() for r in reporters]
                break

            (key, k2) = jrandom.split(key, 2)
            train_ids = jrandom.shuffle(key, jnp.array(train_ids)).tolist()
            train_batch = train_ids[:batch_size]

            try:
                opt_state, model, aux = self.step_optimizer(
                    opt_state, model, subject_interface, train_batch, k2)

            except RuntimeError as e:
                [
                    r.report_nan_detected('Possible ODE failure')
                    for r in reporters
                ]
                break

            if tree_hasnan(model):
                [r.report_nan_detected() for r in reporters]
                break

            if eval_step == last_step and i < iters - 1:
                continue

            [r.report_progress(eval_step) for r in reporters]

            if i == iters - 1:
                raw_res = {
                    'TRN': self.eval(model, subject_interface, train_batch),
                    'VAL': self.eval(model, subject_interface, valid_ids),
                    'TST': self.eval(model, subject_interface, test_ids)
                }
            else:
                raw_res = {
                    'TRN': self.eval(model, subject_interface, train_batch),
                    'VAL': self.eval(model, subject_interface, valid_ids)
                }

            eval_df, eval_flat = metric.evaluation_table(
                raw_res, code_frequency_groups)

            auc = eval_df.loc['MICRO-AUC', 'VAL']

            for r in reporters:
                r.report_evaluation(eval_step, auc, eval_df, eval_flat)
                r.report_params(eval_step,
                                model,
                                m_state,
                                last_iter=i == iters - 1,
                                current_best=auc > best_score)

            if auc > best_score:
                best_score = auc

        return {'objective': auc, 'model': (model, m_state)}


def onestep_trainer(model,
                    m_state,
                    config,
                    splits,
                    code_frequency_groups=None,
                    reporters: List[AbstractReporter] = [],
                    **kwargs):

    train_ids, valid_ids, test_ids = splits

    m_state = model.step_optimizer(100, m_state, train_ids)

    for r in reporters:
        r.report_config(config)
        r.report_params_size(model.parameters_size(m_state))
        r.report_steps(100)
        r.report_progress(100)

    raw_res = {
        'TRN': model.eval(m_state, train_ids),
        'VAL': model.eval(m_state, valid_ids),
        'TST': model.eval(m_state, test_ids)
    }

    eval_df, eval_flat = metric.evaluation_table(raw_res,
                                                 code_frequency_groups)

    auc = eval_df.loc['MICRO-AUC', 'VAL']

    for r in reporters:
        r.report_evaluation(100, auc, eval_df, eval_flat)
        r.report_params(100, model, m_state, last_iter=True, current_best=True)

    return {'objective': auc, 'model': (model, m_state)}
