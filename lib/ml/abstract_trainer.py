from typing import List, Any, Dict, Type, Tuple, Union, Optional
from datetime import datetime
from abc import ABC, abstractmethod, ABCMeta

import pandas as pd
from tqdm import tqdm
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
import jax.example_libraries.optimizers as jopt
import jax.tree_util as jtu
import equinox as eqx
import optuna

from ..ehr import Subject_JAX, BatchPredictedRisks
from .. import metric as M
from ..utils import params_size, tree_hasnan
from .abstract_model import AbstractModel
from .reporters import AbstractReporter

opts = {'sgd': jopt.sgd, 'adam': jopt.adam}

class_weighting_dict = {
    'none': M.softmax_logits_bce,
    'weighted': M.softmax_logits_weighted_bce,
    'focal': M.softmax_logits_balanced_focal_bce
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
        return jopt.exponential_decay(lr,
                                      decay_steps=50,
                                      decay_rate=decay_rate)

    def dx_loss(self):
        return class_weighting_dict[self.class_weighting]

    def unreg_loss(self,
                   model: AbstractModel,
                   subject_interface: Subject_JAX,
                   batch: List[int],
                   args: Dict[str, Any] = dict()):
        res = model(subject_interface, batch, args)

        l = res['predictions'].prediction_loss(self.dx_loss(),
                                               model.outcome_mixer())
        return l, ({'dx_loss': l}, res['predictions'])

    def reg_loss(self,
                 model: AbstractModel,
                 subject_interface: Subject_JAX,
                 batch: List[int],
                 args: Dict[str, Any] = dict()):
        res = model(subject_interface, batch, args)
        l = res['predictions'].prediction_loss(self.dx_loss(),
                                               model.outcome_mixer())
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
        opt_init, opt_update, get_params = opts[self.opt](self.lr_schedule(
            self.lr, self.decay_rate))
        opt_state = opt_init(eqx.filter(model, eqx.is_inexact_array))
        return (opt_update, get_params), opt_state

    def init_from_optstate(self, optstate):
        _, opt_update, get_params = opts[self.opt](self.lr_schedule(
            self.lr, self.decay_rate))

        return (opt_update, get_params), optstate

    def optstate_object(self, optstate):
        _, optstate = optstate
        return optstate

    def update_model(self, model, new_params):

        def _replace(new, old):
            if new is None:
                return old
            else:
                return new

        def _is_none(x):
            x is None

        return jtu.tree_map(_replace, new_params, model, is_leaf=_is_none)

    def _post_update_params(self, model: AbstractModel):
        return model

    def step_optimizer(self, step: int, opt_state: Any, model: AbstractModel,
                       subject_interface: Subject_JAX, batch: Tuple[int]):
        (opt_update, get_params), opt_state = opt_state
        grad_f = eqx.filter_grad(self.loss, has_aux=True)
        grads, aux = grad_f(model, subject_interface, batch)
        opt_state = opt_update(step, grads, opt_state)
        new_model = self.update_model(model, get_params(opt_state))
        new_model = self._post_update_params(new_model)
        opt_state = (opt_update, get_params), opt_state
        return opt_state, new_model, aux

    @classmethod
    def sample_opt(cls, trial: optuna.Trial):
        return {
            'lr': trial.suggest_categorical('lr', [2e-3, 5e-3]),
            'opt':
            'adam'  #trial.suggest_categorical('opt', ['adam', 'sgd', 'fromage'])
        }

    def continue_training(self, model, reporters: List[AbstractReporter]):
        for r in reporters:
            last_eval_step = r.last_eval_step()
            if last_eval_step is not None:
                m, optstate = r.trained_model(model, last_eval_step)
                opt_state = self.init_from_optstate(optstate)
                return last_eval_step, (m, optstate)

        raise RuntimeError(f'No history to continue training from.')

    def __call__(self,
                 model: AbstractModel,
                 subject_interface: Subject_JAX,
                 splits: Tuple[List[int], ...],
                 history: MetricsHistory,
                 continue_training: bool = False,
                 prng_seed: int = 0,
                 trial_terminate_time=datetime.max,
                 reporters: List[AbstractReporter] = []):
        train_ids, valid_ids, test_ids = splits
        batch_size = min(self.batch_size, len(train_ids))
        iters = round(self.epochs * len(train_ids) / batch_size)
        opt_state = self.init_opt(model)
        key = jrandom.PRNGKey(prng_seed)

        train_index = jnp.arange(len(train_ids))
        for r in reporters:
            r.report_config()
            r.report_params_size(params_size(model))
            r.report_steps(iters)

        eval_steps = sorted(set(np.linspace(0, iters - 1, 100).astype(int)))

        if continue_training:
            cont_idx, (cont_m,
                       cont_opt) = self.continue_training(model, reporters)

        for i in tqdm(range(iters)):
            (key, ) = jrandom.split(key, 1)

            if datetime.now() > trial_terminate_time:
                [r.report_timeout() for r in reporters]
                break

            if continue_training:
                j = eval_steps[cont_idx]
                if i <= j:
                    continue
                elif i - 1 == j:
                    model = cont_m
                    opt_state = cont_opt

            shuffled_idx = jrandom.permutation(key, train_index)
            train_batch = [train_ids[i] for i in shuffled_idx[:batch_size]]

            try:
                opt_state, model, _ = self.step_optimizer(
                    i, opt_state, model, subject_interface, train_batch)

            except RuntimeError as e:
                [
                    r.report_nan_detected('Possible ODE failure')
                    for r in reporters
                ]
                break

            if tree_hasnan(model):
                [r.report_nan_detected() for r in reporters]
                break

            if i not in eval_steps:
                continue

            [r.report_progress(eval_steps.index(i)) for r in reporters]

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
                r.report_params(eval_steps.index(i), model,
                                self.optstate_object(opt_state))

        return {'history': history, 'model': model}


class Trainer2LR(Trainer):

    def init_opt(self, model: AbstractModel):
        decay_rate = self.decay_rate
        if not (isinstance(decay_rate, list) or isinstance(decay_rate, tuple)):
            decay_rate = (decay_rate, decay_rate)

        opt1_i, opt1_u, opt1_p = opts[self.opt](self.lr_schedule(
            self.lr[0], decay_rate[0]))
        opt2_i, opt2_u, opt2_p = opts[self.opt](self.lr_schedule(
            self.lr[1], decay_rate[1]))
        m1, m2 = model.emb_dyn_partition(model)
        m1 = eqx.filter(m1, eqx.is_inexact_array)
        m2 = eqx.filter(m2, eqx.is_inexact_array)
        opt1_s = opt1_i(m1)
        opt2_s = opt2_i(m2)
        opt1 = opt1_u, opt1_p
        opt2 = opt2_u, opt2_p
        return (opt1, opt2), (opt1_s, opt2_s)

    def step_optimizer(self, step: int, opt_state: Any, model: AbstractModel,
                       subject_interface: Subject_JAX, batch: Tuple[int]):
        (opt1, opt2), (opt1_s, opt2_s) = opt_state
        opt1_u, opt1_p = opt1
        opt2_u, opt2_p = opt2

        grad_f = eqx.filter_grad(self.loss, has_aux=True)
        grads, aux = grad_f(model, subject_interface, batch)
        g1, g2 = model.emb_dyn_partition(grads)

        opt1_s = opt1_u(step, g1, opt1_s)
        opt2_s = opt2_u(step, g2, opt2_s)

        new_params = model.emb_dyn_merge(opt1_p(opt1_s), opt2_p(opt2_s))
        print(new_params)
        print('-----')
        print(model)
        print('-----')
        new_model = self.update_model(model, new_params)
        print(new_model)

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
        l = preds.prediction_loss(self.dx_loss(), model.outcome_mixer())

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
