import copy
from typing import List, Any, Dict, Type, Tuple, Union
from datetime import datetime
from abc import ABC, abstractmethod, ABCMeta

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
from .. import metric as M
from ..utils import params_size, tree_hasnan

opts = {'sgd': optax.sgd, 'adam': optax.adam, 'fromage': optax.fromage}


class AbstractTrainer(eqx.Module):

    opt: str = eqx.static_field()
    reg_hyperparams: Dict[str, float]
    epochs: int
    batch_size: int
    lr: Union[float, Tuple[float, float]]
    decay_rate: Union[float, Tuple[float, float]] = None

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

        return loss, preds

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

    def evaluations(self,
                    split_predictions: Dict[str, M.BatchPredictedRisks],
                    split_loss: Dict[str, Dict[str, float]],
                    code_frequency_groups=None,
                    top_k_list=[20]):
        evals = {split: {} for split in split_predictions}

        for split, loss in split_loss.items():
            for key, val in loss.items():
                evals[split][key] = val

        for split, preds in split_predictions.items():

            # General confusion matrix statistics (after rounding risk-vals).
            cm = M.compute_confusion_matrix(preds)
            cm_scores = M.confusion_matrix_scores(cm)
            for stat, val in cm_scores.items():
                evals[split][stat] = val

            for stat, val in M.auc_scores(preds).items():
                evals[split][stat] = val

            if code_frequency_groups is not None:
                det_topk_scores = M.top_k_detectability_scores(
                    code_frequency_groups, preds, top_k_list)
                for k in top_k_list:
                    for stat, val in det_topk_scores[k].items():
                        evals[split][stat] = val

            evals[split] = {
                rowname: float(val)
                for rowname, val in evals[split].items()
            }

        flat_evals = {}
        for split in evals:
            flat_evals.update(
                {f'{split}_{key}': val
                 for key, val in evals[split].items()})

        for split, preds in split_predictions.items():
            _, code_auc_dict = M.code_auc_scores(preds, split)
            _, code_occ1_auc_dict = M.first_occurrence_auc_scores(preds, split)

            flat_evals.update(code_auc_dict)
            flat_evals.update(code_occ1_auc_dict)

        return flat_evals

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
            'opt': 'adam',
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
        key = jrandom.PRNGKey(prng_seed)

        for r in reporters:
            r.report_config()
            r.report_params_size(params_size(model))
            r.report_steps(iters)

        auc = 0.0
        best_score = 0.0
        history = M.MetricsHistory()

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
                opt_state, model, _ = self.step_optimizer(
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

            trn_loss, trn_preds = self.eval(model, subject_interface,
                                            train_batch)
            val_loss, val_preds = self.eval(model, subject_interface,
                                            valid_ids)
            split_preds = {'TRN': trn_preds, 'VAL': val_preds}
            split_loss = {'TRN': trn_loss, 'VAL': val_loss}
            if i == iters - 1:
                tst_loss, tst_preds = self.eval(model, subject_interface,
                                                test_ids)
                split_preds['TST'] = tst_preds
                split_loss['TST'] = tst_loss

            evals_dict = self.evaluations(split_preds, split_loss,
                                          code_frequency_groups)
            history.append_iteration(evals_dict)

            auc = evals_dict['VAL_MICRO-AUC']

            for r in reporters:

                r.report_evaluation(eval_step=eval_step,
                                    objective_v=auc,
                                    evals_df=history.to_df())

                if auc > best_score:
                    r.report_params(eval_step, model)

            if auc > best_score:
                best_score = auc

        return {'objective': auc, 'model': model}


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
