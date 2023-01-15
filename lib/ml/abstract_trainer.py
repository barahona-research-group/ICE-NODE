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

from .reporters import AbstractReporter
from ..ehr import Subject_JAX
from .. import metric as M
from ..utils import params_size, tree_hasnan

opts = {'sgd': optax.sgd, 'adam': optax.adam, 'fromage': optax.fromage}


def _loss_multinomial(logits, y):
    return M.softmax_logits_bce(y, logits)


def _loss_multinomial_balanced_focal(logits, y):
    return M.softmax_logits_balanced_focal_bce(y, logits)


def _loss_multinomial_balanced(logits, y):
    weights = y.shape[0] / (y.sum(axis=0) + 1e-10)
    return M.softmax_logits_weighted_bce(y, logits, weights)


class_weighting_dict = {
    'none': _loss_multinomial,
    'balanced': _loss_multinomial_balanced,
    'focal': _loss_multinomial_balanced_focal
}


class Trainer(eqx.Module):

    opt: str = eqx.static_field()
    reg_hyperparams: Dict[str, float]
    epochs: int
    batch_size: int
    lr: Union[float, Tuple[float, float]]
    class_weighting: str = 'none'

    def dx_loss(self):
        return class_weighting_dict[self.class_weighting]

    def unreg_loss(self,
                   model: "..ml.abstract_model.AbstractModel",
                   subject_interface: Subject_JAX,
                   batch: List[int],
                   args: Dict[str, Any] = dict()):
        res = model(subject_interface, batch, args)
        l = res['predictions'].prediction_loss(self.dx_loss())
        return l, ({'dx_loss': l}, res['predictions'])

    def reg_loss(self,
                 model: "..ml.AbstractModel",
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
             model: "..ml.abstract_model.AbstractModel",
             subject_interface: Subject_JAX,
             batch: List[int],
             args: Dict[str, Any] = dict()):
        if self.reg_hyperparams is None:
            return self.unreg_loss(model, subject_interface, batch, args)
        else:
            return self.reg_loss(model, subject_interface, batch, args)

    def eval(self,
             model: "..ml.abstract_model.AbstractModel",
             subject_interface: Subject_JAX,
             batch: List[int],
             args=dict()) -> Dict[str, float]:
        args['eval_only'] = True
        _, loss, preds = self.unreg_loss(model, subject_interface, batch, args)

        return loss, preds

    def init_opt(self, model):
        opt = opts[self.opt](self.lr)
        return opt, opt.init(eqx.filter(model, eqx.is_inexact_array))

    def _post_update_params(self, model: "..ml.AbstractModel"):
        return model

    def step_optimizer(self, opt_state: Any, model: "..ml.AbstractModel",
                       subject_interface: Subject_JAX, batch: Tuple[int],
                       key: "jax.random.PRNGKey"):
        opt, opt_state = opt_state
        grad_f = eqx.filter_grad(self.loss, has_aux=True)
        grads, aux = grad_f(model, subject_interface, batch, key)
        updates, opt_state = opt.update(grads, opt_state)
        new_model = eqx.apply_updates(model, updates)
        new_model = self._post_update_params(new_model)
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
    def sample_opt(cls, trial: optuna.Trial):
        return {
            'lr': trial.suggest_categorical('lr', [2e-3, 5e-3]),
            'opt':
            'adam'  #trial.suggest_categorical('opt', ['adam', 'sgd', 'fromage'])
        }


class Trainer2LR(Trainer):

    def init_opt(self, model: "..ml.abstract_model.AbstractModel"):
        opt1 = opts[self.opt](self.lr[0])
        opt2 = opts[self.opt](self.lr[1])
        m1, m2 = model.emb_dyn_partition(model)
        jax.debug.print('hiii init_opt')
        jax.debug.breakpoint()
        return (opt1, opt2), (opt1.init(m1), opt2.init(m2))

    def step_optimizer(self, opt_state: Any,
                       model: "..ml.abstract_model.AbstractModel",
                       subject_interface: Subject_JAX, batch: Tuple[int],
                       key: "jax.random.PRNGKey"):
        (opt1, opt2), (opt1_s, opt2_s) = opt_state
        grad_f = eqx.filter_grad(self.loss, has_aux=True)
        grads, aux = grad_f(model, subject_interface, batch, key)
        g1, g2 = model.emb_dyn_partition(grads)

        updates1, opt1_s = opt1.update(g1, opt1_s)
        updates2, opt2_s = opt2.update(g2, opt2_s)
        updates = model.emb_dyn_merge(updates1, updates2)

        new_model = eqx.apply_updates(model, updates)

        return ((opt1, opt2), (opt1_s, opt2_s)), new_model, aux


def lr_schedule(lr, decay_rate):
    if decay_rate is None:
        return lr
    return optax.exponential_decay(lr,
                                   transition_steps=50,
                                   decay_rate=decay_rate)


def trainer_from_conf(conf):
    lr = conf['lr']
    dr = conf.get('decay_rate', None)

    trainer_cls = Trainer
    if isinstance(lr, list):
        assert len(lr) == 2, "Shoule provide max of two learning rates."
        trainer_cls = Trainer2LR
        if not isinstance(dr, list):
            dr = (dr, dr)
        lr = tuple(map(lr_schedule, lr, dr))
    else:
        lr = lr_schedule(lr, dr)

    return trainer_cls(opt=conf['opt'],
                       reg_hyperparams=conf['reg_hyperparams'],
                       epochs=conf['epochs'],
                       batch_size=conf['batch_size'],
                       lr=lr)

    def __call__(self,
                 model: "..ml.AbstractModel",
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


def sample_training_config(cls, trial: optuna.Trial,
                           model: '..ml.AbstractModel'):

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

    def __call__(self,
                 model: "..ml.AbstractModel",
                 subject_interface: Subject_JAX,
                 splits: Tuple[List[int], ...],
                 prng_seed: int = 0,
                 code_frequency_groups=None,
                 trial_terminate_time=datetime.max,
                 reporters: List[AbstractReporter] = []):
        pass

    def loss(self,
             model: "..ml.abstract_model.AbstractModel",
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
            adms = [preds[idx] for idx in sorted(preds)]
            # Integration time from the first discharge (adm[0].(length of
            # stay)) to last discarge (adm[-1].time + adm[-1].(length of stay)
            int_time += adms[-1].time + adms[-1].los - adms[0].los
        return int_time

    def dx_loss(self):
        return M.balanced_focal_bce

    def reg_loss(self, model: '..ml.AbstractModel',
                 subject_interface: Subject_JAX, batch: List[int]):
        args = dict(tay_reg=self.tay_reg)
        res = model(subject_interface, batch, args)
        preds = res['predictions']
        l = preds.prediction_loss(self.dx_loss)

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
