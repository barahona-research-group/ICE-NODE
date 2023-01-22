import copy
from typing import List, Any, Dict, Type, Tuple, Union, Optional
from datetime import datetime
from abc import ABC, abstractmethod, ABCMeta

import pandas as pd
from tqdm import tqdm
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import optuna
import optax

from ..ehr import Subject_JAX, BatchPredictedRisks
from .. import metric as M
from ..utils import params_size, tree_hasnan
from .abstract_model import AbstractModel

opts = {'sgd': optax.sgd, 'adam': optax.adam, 'fromage': optax.fromage}

class_weighting_dict = {
    'none':
    M.softmax_logits_bce,
    'balanced':
    M.softmax_logits_balanced_focal_bce,
    'focal':
    lambda y, logits, mask: M.softmax_logits_weighted_bce(
        y, logits, mask, y.shape[0] / (y.sum(axis=0) + 1e-10))
}


class MetricsHistory:
    metrics: M.MetricsCollection

    def __init__(self, metrics):
        self.metrics = metrics
        self._train_df = None
        self._val_df = None
        self._test_df = None

    def train_df(self):
        return self._train_df

    def validation_df(self):
        return self._val_df

    def test_df(self):
        return self._test_df

    def append_train_iteration(
            self,
            predictions: BatchPredictedRisks,
            other_estimated_metrics: Optional[Dict[str, float]] = None):
        niters = 1 if self._train_df is None else len(self._train_df) + 1
        row_df = self.metrics.to_df(niters, predictions,
                                    other_estimated_metrics)
        self._train_df = pd.concat([self._train_df, row_df])

    def append_validation_iteration(
            self,
            predictions: BatchPredictedRisks,
            other_estimated_metrics: Optional[Dict[str, float]] = None):
        niters = 1 if self._val_df is None else len(self._val_df) + 1
        row_df = self.metrics.to_df(niters, predictions,
                                    other_estimated_metrics)
        self._val_df = pd.concat([self._val_df, row_df])

    def append_test_iteration(
            self,
            predictions: BatchPredictedRisks,
            other_estimated_metrics: Optional[Dict[str, float]] = None):
        niters = 1 if self._test_df is None else len(self._test_df) + 1
        row_df = self.metrics.to_df(niters, predictions,
                                    other_estimated_metrics)
        self._test_df = pd.concat([self._test_df, row_df])


class Trainer(eqx.Module):

    opt: str
    reg_hyperparams: Dict[str, float]
    epochs: int
    batch_size: int
    lr: Union[float, Tuple[float, float]]
    decay_rate: Optional[Union[float, Tuple[float, float]]]
    class_weighting: str

    def __init__(self,
                 opt,
                 reg_hyperparams,
                 epochs,
                 batch_size,
                 lr,
                 decay_rate=None,
                 class_weighting='none',
                 **kwargs):
        self.opt = opt
        self.reg_hyperparams = reg_hyperparams
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.decay_rate = decay_rate
        self.class_weighting = class_weighting

    @staticmethod
    def lr_schedule(lr, decay_rate):
        if decay_rate is None:
            return lr
        return optax.exponential_decay(lr,
                                       transition_steps=50,
                                       decay_rate=decay_rate)

    def dx_loss(self):
        return class_weighting_dict[self.class_weighting]

    def unreg_loss(self,
                   model: AbstractModel,
                   subject_interface: Subject_JAX,
                   batch: List[int],
                   args: Dict[str, Any] = dict()):
        res = model(subject_interface, batch, args)
        l = res['predictions'].prediction_loss(self.dx_loss())
        return l, ({'dx_loss': l}, res['predictions'])

    def reg_loss(self,
                 model: AbstractModel,
                 subject_interface: Subject_JAX,
                 batch: List[int],
                 args: Dict[str, Any] = dict()):
        res = model(subject_interface, batch, args)
        l = res['predictions'].prediction_loss(self.dx_loss())
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
        _, (loss, preds) = self.unreg_loss(model, subject_interface, batch,
                                           args)

        return loss, preds

    def init_opt(self, model):
        opt = opts[self.opt](self.lr_schedule(self.lr, self.decay_rate))
        return opt, opt.init(eqx.filter(model, eqx.is_inexact_array))

    def _post_update_params(self, model: AbstractModel):
        return model

    def step_optimizer(self, opt_state: Any, model: AbstractModel,
                       subject_interface: Subject_JAX, batch: Tuple[int]):
        opt, opt_state = opt_state
        grad_f = eqx.filter_grad(self.loss, has_aux=True)
        grads, aux = grad_f(model, subject_interface, batch)
        updates, opt_state = opt.update(grads, opt_state)
        new_model = eqx.apply_updates(model, updates)
        new_model = self._post_update_params(new_model)
        return (opt, opt_state), new_model, aux

    @classmethod
    def sample_opt(cls, trial: optuna.Trial):
        return {
            'lr': trial.suggest_categorical('lr', [2e-3, 5e-3]),
            'opt':
            'adam'  #trial.suggest_categorical('opt', ['adam', 'sgd', 'fromage'])
        }

    def __call__(self,
                 model: AbstractModel,
                 subject_interface: Subject_JAX,
                 splits: Tuple[List[int], ...],
                 history: MetricsHistory,
                 prng_seed: int = 0,
                 trial_terminate_time=datetime.max,
                 reporters: List["lib.ml.AbstractReporter"] = []):
        train_ids, valid_ids, test_ids = splits
        # Because shuffling is done in-place.
        train_ids = copy.deepcopy(train_ids)
        batch_size = min(self.batch_size, len(train_ids))
        iters = round(self.epochs * len(train_ids) / batch_size)
        opt_state = self.init_opt(model)
        key = jrandom.PRNGKey(prng_seed)

        for r in reporters:
            r.report_config()
            r.report_params_size(params_size(model))
            r.report_steps(iters)

        for i in tqdm(range(iters)):
            eval_step = round((i + 1) * 100 / iters)
            last_step = round(i * 100 / iters)

            if datetime.now() > trial_terminate_time:
                [r.report_timeout() for r in reporters]
                break

            (key, ) = jrandom.split(key, 1)
            train_ids = jrandom.permutation(key,
                                            jnp.array(train_ids),
                                            independent=True).tolist()
            train_batch = train_ids[:batch_size]

            try:
                opt_state, model, _ = self.step_optimizer(
                    opt_state, model, subject_interface, train_batch)

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

            trn_loss, trn_preds = self.eval(model, subject_interface,
                                            train_batch)
            history.append_train_iteration(trn_preds, trn_loss)
            val_loss, val_preds = self.eval(model, subject_interface,
                                            valid_ids)
            history.append_validation_iteration(val_preds, val_loss)

            if i == iters - 1:
                tst_loss, tst_preds = self.eval(model, subject_interface,
                                                test_ids)

                history.append_test_iteration(tst_preds, tst_loss)

            for r in reporters:
                r.report_evaluation(history)
                r.report_params(eval_step, model)

        return {'history': history, 'model': model}


class Trainer2LR(Trainer):

    def init_opt(self, model: AbstractModel):
        decay_rate = self.decay_rate
        if not (isinstance(decay_rate, list) or isinstance(decay_rate, tuple)):
            decay_rate = (decay_rate, decay_rate)

        opt1 = opts[self.opt](self.lr_schedule(self.lr[0], decay_rate[0]))
        opt2 = opts[self.opt](self.lr_schedule(self.lr[1], decay_rate[1]))
        m1, m2 = model.emb_dyn_partition(model)
        m1 = eqx.filter(m1, eqx.is_inexact_array)
        m2 = eqx.filter(m2, eqx.is_inexact_array)
        return (opt1, opt2), (opt1.init(m1), opt2.init(m2))

    def step_optimizer(self, opt_state: Any, model: AbstractModel,
                       subject_interface: Subject_JAX, batch: Tuple[int]):
        (opt1, opt2), (opt1_s, opt2_s) = opt_state
        grad_f = eqx.filter_grad(self.loss, has_aux=True)
        grads, aux = grad_f(model, subject_interface, batch)
        g1, g2 = model.emb_dyn_partition(grads)

        updates1, opt1_s = opt1.update(g1, opt1_s)
        updates2, opt2_s = opt2.update(g2, opt2_s)

        updates = model.emb_dyn_merge(updates1, updates2)

        new_model = eqx.apply_updates(model, updates)

        return ((opt1, opt2), (opt1_s, opt2_s)), new_model, aux


def sample_training_config(cls, trial: optuna.Trial, model: AbstractModel):

    return {
        'epochs': 10,
        'batch_size': trial.suggest_int('B', 2, 27, 5),
        'opt': 'adam',
        'lr': trial.suggest_float('lr', 5e-5, 5e-3, log=True),
        'decay_rate': None,
        'class_weighting': 'none',
        #trial.suggest_categorical('class_weight',
        #                         ['none', 'balanced', 'focal']),
        'reg_hyperparams': model.sample_reg_hyperparams(trial)
    }


class LassoNetTrainer(Trainer):

    def loss(self,
             model: AbstractModel,
             subject_interface: Subject_JAX,
             batch: List[int],
             args: Dict[str, Any] = dict()):
        return self.unreg_loss(model, subject_interface, batch, args)

    def _post_update_params(self, model):
        if self.reg_hyperparams:
            return model.prox_map()(model, self.reg_hyperparams)
        else:
            return model


class ODETrainer(Trainer):
    tay_reg: int = 3

    @classmethod
    def odeint_time(cls, predictions: M.BatchPredictedRisks):
        int_time = 0
        for subj_id, preds in predictions.items():
            adms = [preds[idx].admission for idx in sorted(preds)]
            # Integration time from the first discharge (adm[0].(length of
            # stay)) to last discarge (adm[-1].time + adm[-1].(length of stay)
            int_time += adms[-1].admission_time + adms[-1].los - adms[0].los
        return int_time

    def dx_loss(self):
        return M.balanced_focal_bce

    def reg_loss(self,
                 model: AbstractModel,
                 subject_interface: Subject_JAX,
                 batch: List[int],
                 args: Dict[str, float] = dict()):
        args['tay_reg'] = self.tay_reg
        res = model(subject_interface, batch, args)
        preds = res['predictions']
        l = preds.prediction_loss(self.dx_loss())

        integration_weeks = self.odeint_time(preds) / 7
        l1_loss = model.l1()
        l2_loss = model.l2()
        dyn_loss = res['dyn_loss'] / integration_weeks
        l1_alpha = self.reg_hyperparams['L_l1']
        l2_alpha = self.reg_hyperparams['L_l2']
        dyn_alpha = self.reg_hyperparams['L_dyn']

        loss = l + (l1_alpha * l1_loss) + (l2_alpha * l2_loss) + (dyn_alpha *
                                                                  dyn_loss)

        return loss, ({
            'dx_loss': l,
            'loss': loss,
            'l1_loss': l1_loss,
            'l2_loss': l2_loss,
        }, preds)

    @classmethod
    def sample_reg_hyperparams(cls, trial: optuna.Trial):
        return {
            'L_l1': 0,  #trial.suggest_float('l1', 1e-8, 5e-3, log=True),
            'L_l2': 0,  # trial.suggest_float('l2', 1e-8, 5e-3, log=True),
            'L_dyn': 1e3  # trial.suggest_float('L_dyn', 1e-6, 1, log=True)
        }


class ODETrainer2LR(ODETrainer, Trainer2LR):
    pass
