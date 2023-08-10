from typing import List, Any, Dict, Type, Tuple, Union, Optional
from datetime import datetime

import pandas as pd
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
import jax.example_libraries.optimizers as jopt
import jax.tree_util as jtu
import equinox as eqx
import optuna

from ..ehr import Predictions, Patients
from .. import metric as M
from ..utils import params_size, tree_hasnan, tqdm_constructor
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
            predictions: Predictions,
            other_estimated_metrics: Optional[Dict[str, float]] = None):
        niters = 1 if self._train_df is None else len(self._train_df) + 1
        row_df = self.metrics.to_df(niters, predictions,
                                    other_estimated_metrics)
        self._train_df = pd.concat([self._train_df, row_df])

    def append_validation_iteration(
            self,
            predictions: Predictions,
            other_estimated_metrics: Optional[Dict[str, float]] = None):
        niters = 1 if self._val_df is None else len(self._val_df) + 1
        row_df = self.metrics.to_df(niters, predictions,
                                    other_estimated_metrics)
        self._val_df = pd.concat([self._val_df, row_df])

    def append_test_iteration(
            self,
            predictions: Predictions,
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
    counts_ignore_first_admission: bool
    lr: Union[float, Tuple[float, float]]
    decay_rate: Optional[Union[float, Tuple[float, float]]]
    class_weighting: str

    def __init__(self,
                 opt,
                 reg_hyperparams,
                 epochs,
                 batch_size,
                 lr,
                 counts_ignore_first_admission=False,
                 decay_rate=None,
                 class_weighting='focal',
                 **kwargs):
        self.opt = opt
        self.reg_hyperparams = reg_hyperparams
        self.epochs = epochs
        self.batch_size = batch_size
        self.counts_ignore_first_admission = counts_ignore_first_admission
        self.lr = lr
        self.decay_rate = decay_rate
        self.class_weighting = class_weighting

    @staticmethod
    def lr_schedule(lr, decay_rate, iters):
        if decay_rate is None:
            return lr
        return jopt.exponential_decay(lr,
                                      decay_steps=iters // 2,
                                      decay_rate=decay_rate)

    def dx_loss(self):
        return class_weighting_dict[self.class_weighting]

    def unreg_loss(self, model: AbstractModel, patients: Patients):
        predictions = model.batch_predict(patients, leave_pbar=False)
        l = predictions.prediction_loss(dx_loss=self.dx_loss())
        return l['dx_loss'], (l, predictions)

    def reg_loss(self, model: AbstractModel, patients: Patients):
        predictions = model.batch_predict(patients, leave_pbar=False)
        l = predictions.prediction_loss(dx_loss=self.dx_loss())['dx_loss']
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
        }, predictions)

    def loss(self, model: AbstractModel, patients: Patients):
        if self.reg_hyperparams is None:
            return self.unreg_loss(model, patients)
        else:
            return self.reg_loss(model, patients)

    def eval(self, model: AbstractModel,
             patients: Patients) -> Dict[str, float]:
        _, (loss, preds) = self.unreg_loss(model, patients)
        return loss, preds

    def init_opt(self, model, iters):
        opt_init, opt_update, get_params = opts[self.opt](self.lr_schedule(
            self.lr, self.decay_rate, iters))
        opt_state = opt_init(eqx.filter(model, eqx.is_inexact_array))
        return (opt_update, get_params), opt_state

    def init_from_loaded_optstate(self, optstate, model, iters):
        opt, _ = self.init_opt(model, iters=iters)
        optstate = jopt.pack_optimizer_state(optstate)
        return opt, optstate

    def serializable_optstate(self, optstate):
        _, optstate = optstate
        return jopt.unpack_optimizer_state(optstate)

    def update_model(self, model, new_params):

        def _replace(new, old):
            if new is None:
                return old
            else:
                return new

        def _is_none(x):
            return x is None

        return jtu.tree_map(_replace, new_params, model, is_leaf=_is_none)

    def _post_update_params(self, model: AbstractModel):
        return model

    def step_optimizer(self, step: int, opt_state: Any, model: AbstractModel,
                       patients: Patients):
        (opt_update, get_params), opt_state = opt_state
        grad_f = eqx.filter_grad(self.loss, has_aux=True)
        grads, aux = grad_f(model, patients)
        opt_state = opt_update(step, grads, opt_state)
        new_model = self.update_model(model, get_params(opt_state))
        new_model = self._post_update_params(new_model)
        opt_state = (opt_update, get_params), opt_state
        return opt_state, new_model, aux

    @classmethod
    def sample_opt(cls, trial: optuna.Trial):
        return {
            'lr': trial.suggest_categorical('lr', [2e-3, 5e-3]),
            'opt': 'adam'
        }

    def continue_training(self, model, reporters: List[AbstractReporter],
                          iters: int):
        for r in reporters:
            last_eval_step = r.last_eval_step()
            if last_eval_step is not None:
                m, optstate = r.trained_model(model, last_eval_step)
                optstate = self.init_from_loaded_optstate(optstate,
                                                          model,
                                                          iters=iters)
                return last_eval_step, (m, optstate)

        raise RuntimeError(f'No history to continue training from.')

    def __call__(self,
                 model: AbstractModel,
                 patients: Patients,
                 splits: Tuple[List[int], ...],
                 history: MetricsHistory,
                 n_evals=100,
                 continue_training: bool = False,
                 prng_seed: int = 0,
                 trial_terminate_time=datetime.max,
                 reporters: List[AbstractReporter] = []):
        train_ids, valid_ids, test_ids = splits
        n_train_admissions = patients.n_admissions(
            train_ids,
            ignore_first_admission=self.counts_ignore_first_admission)

        batch_size = min(self.batch_size, n_train_admissions)
        iters = round(self.epochs * n_train_admissions / batch_size)
        opt_state = self.init_opt(model, iters=iters)
        key = jrandom.PRNGKey(prng_seed)

        for r in reporters:
            r.report_config()
            r.report_params_size(params_size(model))
            r.report_steps(iters)

        eval_steps = sorted(set(
            np.linspace(0, iters - 1, n_evals).astype(int)))

        if continue_training:
            cont_idx, (cont_m, cont_opt) = self.continue_training(model,
                                                                  reporters,
                                                                  iters=iters)
        val_batch = patients.device_batch(valid_ids)
        step = 0
        for _ in tqdm_constructor(range(self.epochs), leave=True,
                                  unit='Epoch'):
            (key, ) = jrandom.split(key, 1)
            train_ids = jrandom.permutation(key, jnp.array(train_ids))
            batch_gen = patients.batch_gen(
                train_ids.tolist(),
                batch_n_admissions=batch_size,
                ignore_first_admission=self.counts_ignore_first_admission)
            n_batches = n_train_admissions // batch_size
            batch_gen = tqdm_constructor(batch_gen,
                                         leave=False,
                                         total=n_batches,
                                         unit='Batch')
            for batch in batch_gen:
                if datetime.now() > trial_terminate_time:
                    [r.report_timeout() for r in reporters]
                    break
                step += 1
                if continue_training:
                    j = eval_steps[cont_idx]
                    if step <= j:
                        continue
                    elif step - 1 == j:
                        model = cont_m
                        opt_state = cont_opt

                try:
                    opt_state, model, _ = self.step_optimizer(
                        step, opt_state, model, batch)

                except RuntimeError as e:
                    [
                        r.report_nan_detected(f'Possible ODE failure: {e}')
                        for r in reporters
                    ]
                    return {'history': history, 'model': model}

                if tree_hasnan(model):
                    [r.report_nan_detected() for r in reporters]
                    return {'history': history, 'model': model}

                if step not in eval_steps:
                    continue

                [r.report_progress(eval_steps.index(step)) for r in reporters]

                trn_loss, trn_preds = self.eval(model, batch)
                history.append_train_iteration(trn_preds, trn_loss)
                val_loss, val_preds = self.eval(model, val_batch)
                history.append_validation_iteration(val_preds, val_loss)

                if step == iters - 1:
                    test_batch = patients.device_batch(test_ids)
                    tst_loss, tst_preds = self.eval(model, test_batch)

                    history.append_test_iteration(tst_preds, tst_loss)

                for r in reporters:
                    r.report_evaluation(history)
                    r.report_params(eval_steps.index(step), model,
                                    self.serializable_optstate(opt_state))
            batch_gen.close()

        return {'history': history, 'model': model}



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

    def loss(self, model: AbstractModel, patients: Patients):
        return self.unreg_loss(model, patients)

    def _post_update_params(self, model):
        if self.reg_hyperparams:
            return model.prox_map()(model, self.reg_hyperparams)
        else:
            return model


class ODETrainer(Trainer):
    tay_reg: int = 3

    def dx_loss(self):
        return M.balanced_focal_bce


class InTrainer(Trainer):

    def dx_loss(self):
        return M.balanced_focal_bce

    def obs_loss(self):
        return M.masked_l2

    def reg_loss(self, model: AbstractModel, patients: Patients):
        return self.unreg_loss(model, patients)

    def unreg_loss(self, model: AbstractModel, patients: Patients):
        preds = model.batch_predict(patients, leave_pbar=False)
        l = preds.prediction_loss(dx_loss=self.dx_loss(),
                                  obs_loss=self.obs_loss())
        loss = l['dx_loss'] + l['obs_loss']
        return loss, ({
            'dx_loss': l['dx_loss'],
            'obs_loss': l['obs_loss'],
            'loss': loss
        }, preds)


