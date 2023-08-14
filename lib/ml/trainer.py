from __future__ import annotations
from typing import List, Any, Dict, Tuple, Union, Optional, Callable
from datetime import datetime

import os
import re
import contextlib
from pathlib import Path
import pickle
import logging
from abc import abstractmethod, ABCMeta
import pandas as pd
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
import jax.example_libraries.optimizers as jopt
import jax.tree_util as jtu
import equinox as eqx
from blinker import signal
import optuna

from ..ehr import Predictions, Patients
from ..metric import (MetricsCollection, Metric, binary_loss, numeric_loss)
from ..utils import (params_size, tree_hasnan, tqdm_constructor, write_config,
                     append_params_to_zip, zip_members, translate_path)
from .model import AbstractModel

_opts = {'sgd': jopt.sgd, 'adam': jopt.adam}


class ResourceTimeout(Exception):
    """Raised when a trial is anticipated to
    be exceeding the timelimit of the compute."""
    pass


class StudyHalted(Exception):
    """Raised when a trial is spawned from a retired study."""
    pass


class TrainingHistory:

    def __init__(self, metrics: MetricsCollection):
        self.metrics = metrics
        self._train_df = None
        self._val_df = None
        self._test_df = None

    @property
    def train_df(self):
        return self._train_df

    @property
    def val_df(self):
        return self._val_df

    @property
    def test_df(self):
        return self._test_df

    def append_train_preds(self, res: Predictions):
        niters = 1 if self._train_df is None else len(self._train_df) + 1
        row_df = self.metrics.to_df(niters, res)
        self._train_df = pd.concat([self._train_df, row_df])

    def append_val_preds(self, res: Predictions):
        niters = 1 if self._val_df is None else len(self._val_df) + 1
        row_df = self.metrics.to_df(niters, res)
        self._val_df = pd.concat([self._val_df, row_df])

    def append_test_preds(self, res: Predictions):
        niters = 1 if self._test_df is None else len(self._test_df) + 1
        row_df = self.metrics.to_df(niters, res)
        self._test_df = pd.concat([self._test_df, row_df])


class TrainerSignals:
    start_training = signal('start_training')
    end = signal('end_training')
    start_evaluation = signal('start_evaluation')
    train_evaluation = signal('train_evaluation')
    val_evaluation = signal('val_evaluation')
    test_evaluation = signal('test_evaluation')
    end_evaluation = signal('end_evaluation')
    continue_training = signal('continue_training')
    timeout = signal('timeout')
    nan_detected = signal('nan_detected')


class AbstractReporter(metaclass=ABCMeta):

    @abstractmethod
    def signal_slot_pairs(self, trainer_signals: TrainerSignals):
        pass


class ConsoleReporter(AbstractReporter):

    def log_config(self, sender, **kw):
        logging.info(f'HPs: {sender.config}')

    def log_nan_detected(self, sender, **kwargs):
        logging.warning(kwargs['msg'] or 'NaN detected')

    def log_evaluation(self, sender, **kwargs):
        h = kwargs['history']
        for name, df in zip(('train', 'val', 'tst'),
                            (h.train_df, h.val_df, h.test_df)):
            logging.info(name + str(df))

    def signal_slot_pairs(self, trainer_signals: TrainerSignals):
        return [(trainer_signals.start_training, self.log_config),
                (trainer_signals.nan_detected, self.log_nan_detected),
                (trainer_signals.end_evaluation, self.log_evaluation)]


class EvaluationDiskWriter(AbstractReporter):

    def __init__(self, output_dir):
        self.output_dir = output_dir

    def report_evaluation(self, sender, **kwargs):
        h = kwargs['history']

        for name, df in zip(('train', 'val', 'tst'),
                            (h.train_df, h.val_df, h.test_df)):
            if df is None:
                continue
            fname = f'{name}_evals.csv.gz'
            fpath = os.path.join(self.output_dir, fname)
            df.to_csv(fpath,
                      compression="gzip",
                      mode='a',
                      header=not os.path.exists(fpath))

    def signal_slot_pairs(self, trainer_signals: TrainerSignals):
        return [(trainer_signals.end_evaluation, self.report_evaluation)]


class ParamsDiskWriter(AbstractReporter):

    def __init__(self, output_dir):
        self.output_dir = output_dir

    def report_params_optimizer(self, sender, **kwargs):
        model = kwargs['model']
        optimizer = kwargs['optimizer']
        step = kwargs['step']

        tarname = os.path.join(self.output_dir, 'params.zip')
        name = f'step{step:04d}.eqx'
        append_params_to_zip(model, name, tarname)
        optimizer.save(os.path.join(self.output_dir, 'optstate.pkl'))

    @staticmethod
    def last_eval_step(output_dir):
        tarname = os.path.join(output_dir, 'params.zip')
        filenames = zip_members(tarname)
        numbers = []
        for fname in filenames:
            numbers.extend(map(int, re.findall(r'\d+', fname)))
        return sorted(numbers)[-1]

    def load_trained_model(self, model, step):
        tarfname = os.path.join(self.output_dir, 'params.zip')
        return model.load_params_from_archive(tarfname, f'step{step:04d}.eqx')

    def continue_training(self, sender, **kwargs):
        messenger = kwargs['messenger']
        last_eval_step = self.last_eval_step(self.output_dir)
        if last_eval_step is not None:
            model = messenger['model']
            optimizer = messenger['optimizer']

            model = self.load_trained_model(model, last_eval_step)
            optimizer = optimizer.load_optimizer(
                os.path.join(self.output_dir, 'optstate.pkl'))
            messenger['model'] = model
            messenger['optimizer'] = optimizer
            messenger['step'] = last_eval_step

    def signal_slot_pairs(self, trainer_signals: TrainerSignals):
        return [(trainer_signals.end_evaluation, self.report_params_optimizer),
                (trainer_signals.continue_training, self.continue_training)]


class ConfigDiskWriter(AbstractReporter):

    def __init__(self, output_dir):
        self.output_dir = output_dir

    def report_config(self, sender, **kwargs):
        config = kwargs['config']
        write_config(config, os.path.join(self.output_dir, 'config.json'))

    def signal_slot_pairs(self, trainer_signals: TrainerSignals):
        return [(trainer_signals.start_training, self.report_config)]


class OptunaReporter(AbstractReporter):

    def __init__(self, trial, objective):
        self.trial = trial

        if callable(objective):
            self.objective = objective
        elif isinstance(objective, list):
            self.objective = None
        else:
            raise RuntimeError('Unexpected objective passed')

    def report_params_size(self, sender, **kwargs):
        params_size = kwargs['params_size']
        self.trial.set_user_attr('parameters_size', params_size)

    def report_steps(self, sender, **kwargs):
        n_iters = kwargs['n_iters']
        self.trial.set_user_attr('steps', n_iters)

    def report_timeout(self, sender, **kwargs):
        self.trial.set_user_attr('timeout', 1)

    def report_evaluation(self, sender, **kwargs):
        history = kwargs['history']
        eval_step = len(history.val_df)

        self.trial.set_user_attr("progress", eval_step)
        study_attrs = self.trial.study.user_attrs

        if study_attrs['halt']:
            self.trial.set_user_attr('halted', 1)
            raise StudyHalted('Study is halted')

        if study_attrs['enable_prune'] and self.trial.should_prune():
            raise optuna.TrialPruned()

        if self.objective:
            value = self.objective(history.val_df)
            self.trial.report(value, len(history.val_df))

    def report_nan_detected(self, sender, **kwargs):
        self.trial.set_user_attr('nan', 1)

    def signal_slot_pairs(self, trainer_signals: TrainerSignals):
        return [(trainer_signals.start_training, self.report_params_size),
                (trainer_signals.start_training, self.report_steps),
                (trainer_signals.timeout, self.report_timeout),
                (trainer_signals.nan_detected, self.report_nan_detected),
                (trainer_signals.end_evaluation, self.report_evaluation)]


class OptimizerConfig(eqx.Module):
    opt: str
    lr: Union[float, Tuple[float, float]]
    decay_rate: Optional[Union[float, Tuple[float, float]]] = None
    reverse_schedule: bool = False


class Optimizer(eqx.Module):
    config: OptimizerConfig
    iters: int
    opt_update: Callable
    get_params: Callable
    optstate: Any

    def __init__(self,
                 config: OptimizerConfig,
                 model=None,
                 iters=None,
                 optstate=None):
        self.config = config
        self.iters = iters
        lr = self.lr_schedule(config, iters)

        opt_init, self.opt_update, self.get_params = _opts[config.opt](lr)

        if optstate is None and model is not None:
            self.optstate = opt_init(eqx.filter(model, eqx.is_inexact_array))
        elif optstate is not None:
            self.optstate = optstate
        else:
            raise ValueError('Either optstate or model must be provided')

    def step(self, step, grads):
        optstate = self.opt_update(step, grads, self.optstate)
        return eqx.tree_at(lambda x: x.optstate, self, optstate)

    def __call__(self, model):
        new_params = self.get_params(self.optstate)

        def _replace(new, old):
            if new is None:
                return old
            else:
                return new

        def _is_none(x):
            return x is None

        return jtu.tree_map(_replace, new_params, model, is_leaf=_is_none)

    def load(self, filename):
        with open(filename, "rb") as optstate_file:
            optstate = pickle.load(optstate_file)

        optstate = jopt.pack_optimizer_state(optstate)
        return Optimizer(config=self.config,
                         optstate=optstate,
                         iters=self.iters)

    def save(self, filename):
        optstate = jopt.unpack_optimizer_state(self.optstate)
        with open(filename, "wb") as optstate_file:
            pickle.dump(optstate, optstate_file)

    @staticmethod
    def lr_schedule(config, iters=None, reverse=False):
        if config.decay_rate is None or iters is None:
            return config.lr

        schedule = jopt.exponential_decay(config.lr,
                                          decay_steps=iters // 2,
                                          decay_rate=config.decay_rate)
        if reverse:
            return lambda i: schedule(iters - i)

        return schedule

    @classmethod
    def sample_opt(cls, trial: optuna.Trial):
        return {
            'lr': trial.suggest_categorical('lr', [2e-3, 5e-3]),
            'opt': 'adam'
        }


class TrainerReporting(eqx.Module):
    reporters: List[AbstractReporter]
    metrics: MetricsCollection

    def __init__(self,
                 output_dir: Optional[str] = None,
                 console: bool = True,
                 metrics: Optional[List[Metric]] = None,
                 parameter_snapshots: bool = False,
                 config_json: bool = False,
                 optuna_trial: Optional[optuna.Trial] = None,
                 optuna_objective: Optional[Callable] = None):
        reporters = []
        if config_json or metrics is not None or parameter_snapshots:
            assert output_dir is not None, 'output_dir must be provided'
            output_dir = translate_path(output_dir)
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        if console:
            reporters.append(ConsoleReporter())

        if parameter_snapshots:
            reporters.append(ParamsDiskWriter(output_dir))

        if metrics is not None and len(metrics) > 0:
            reporters.append(EvaluationDiskWriter(output_dir))

        if config_json:
            reporters.append(ConfigDiskWriter(output_dir))

        if optuna_trial is not None:
            assert optuna_objective is not None, ('optuna_objective must '
                                                  'be provided')
            reporters.append(OptunaReporter(optuna_trial, optuna_objective))

        self.reporters = reporters
        self.metrics = MetricsCollection(metrics or [])

    def new_training_history(self):
        return TrainingHistory(self.metrics)

    @property
    def supports_continue_training(self):
        return any([isinstance(r, ParamsDiskWriter) for r in self.reporters])

    def connections(self, trainer_signals: TrainerSignals):
        return [
            signal.connected_to(slot) for r in self.reporters
            for signal, slot in r.signal_slot_pairs(trainer_signals)
        ]


class WarmupConfig(eqx.Module):
    epochs: int
    batch_size: int
    optimizer_config: OptimizerConfig

    def __init__(self,
                 epochs: int,
                 batch_size: int,
                 opt: str,
                 lr: float,
                 decay_rate: Optional[float] = None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer_config = OptimizerConfig(opt=opt,
                                                lr=lr,
                                                decay_rate=decay_rate,
                                                reverse_schedule=True)


class Trainer(eqx.Module):
    optimizer_config: OptimizerConfig
    reg_hyperparams: Dict[str, float]
    epochs: int
    batch_size: int
    counts_ignore_first_admission: bool
    dx_loss: Callable
    obs_loss: Callable
    kwargs: Dict[str, Any]

    def __init__(self,
                 optimizer_config: OptimizerConfig,
                 reg_hyperparams,
                 epochs,
                 batch_size,
                 counts_ignore_first_admission=False,
                 dx_loss='balanced_focal_bce',
                 obs_loss='mse',
                 **kwargs):
        self.optimizer_config = optimizer_config
        self.reg_hyperparams = reg_hyperparams
        self.epochs = epochs
        self.batch_size = batch_size
        self.counts_ignore_first_admission = counts_ignore_first_admission
        self.dx_loss = binary_loss[dx_loss]
        self.obs_loss = numeric_loss[obs_loss]
        kwargs = kwargs or {}
        kwargs.update({'dx_loss': dx_loss, 'obs_loss': obs_loss})
        self.kwargs = kwargs

    @property
    def config(self):
        return {
            'opt_config': self.optimizer_config.__dict__,
            'reg_hyperparams': self.reg_hyperparams,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'counts_ignore_first_admission':
            self.counts_ignore_first_admission,
        }

    def unreg_loss(self, model: AbstractModel, patients: Patients):
        predictions = model.batch_predict(patients, leave_pbar=False)
        return predictions.prediction_dx_loss(dx_loss=self.dx_loss)

    def reg_loss(self, model: AbstractModel, patients: Patients):
        predictions = model.batch_predict(patients, leave_pbar=False)
        l = predictions.prediction_dx_loss(dx_loss=self.dx_loss)
        l1_loss = model.l1()
        l2_loss = model.l2()
        l1_alpha = self.reg_hyperparams['L_l1']
        l2_alpha = self.reg_hyperparams['L_l2']

        loss = l + (l1_alpha * l1_loss) + (l2_alpha * l2_loss)

        return loss

    def loss(self, model: AbstractModel, patients: Patients):
        if self.reg_hyperparams is None:
            return self.unreg_loss(model, patients)
        else:
            return self.reg_loss(model, patients)

    def _post_update_params(self, model: AbstractModel):
        return model

    def step_optimizer(self, step: int, optimizer: Optimizer,
                       model: AbstractModel, patients: Patients):
        grad_f = eqx.filter_value_and_grad(self.loss)
        value, grads = grad_f(model, patients)
        optimizer = optimizer.step(step, grads)
        new_model = optimizer(model)
        new_model = self._post_update_params(new_model)
        return optimizer, new_model, value

    def __call__(self,
                 model: AbstractModel,
                 patients: Patients,
                 splits: Tuple[List[int], ...],
                 n_evals=100,
                 reporting: TrainerReporting = TrainerReporting(),
                 warmup_config: Optional[WarmupConfig] = None,
                 continue_training: bool = False,
                 prng_seed: int = 0,
                 trial_terminate_time=datetime.max):
        if continue_training:
            assert reporting.supports_continue_training, (
                'TrainerReporting must support continue_training. '
                'Make sure to include a ParamsDiskWriter '
                'in the reporters list.')
            assert warmup_config is None, (
                'warmup_config cannot be provided when '
                'continue_training is True')

        if warmup_config is not None:
            model = self._warmup(model=model,
                                 patients=patients,
                                 splits=splits,
                                 prng_seed=prng_seed,
                                 trial_terminate_time=trial_terminate_time,
                                 history=reporting.new_training_history(),
                                 signals=TrainerSignals(),
                                 warmup_config=warmup_config)

        with contextlib.ExitStack() as stack:
            signals = TrainerSignals()
            for c in reporting.connections(signals):
                stack.enter_context(c)

            return self._train(model=model,
                               patients=patients,
                               splits=splits,
                               n_evals=n_evals,
                               continue_training=continue_training,
                               prng_seed=prng_seed,
                               trial_terminate_time=trial_terminate_time,
                               history=reporting.new_training_history(),
                               signals=signals)

    def _warmup(self, model: AbstractModel, patients: Patients,
                splits: Tuple[List[int], ...], prng_seed, trial_terminate_time,
                history: TrainingHistory, signals: TrainerSignals,
                warmup_config: WarmupConfig):
        trainer = Trainer(
            optimizer_config=warmup_config.optimizer_config,
            reg_hyperparams=self.reg_hyperparams,
            epochs=warmup_config.epochs,
            batch_size=warmup_config.batch_size,
            counts_ignore_first_admission=self.counts_ignore_first_admission,
            **self.kwargs)
        return trainer._train(model=model,
                              patients=patients,
                              splits=(splits[0], [], []),
                              n_evals=0,
                              continue_training=False,
                              prng_seed=prng_seed,
                              trial_terminate_time=trial_terminate_time,
                              history=history,
                              signals=signals)

    def _train(self, model: AbstractModel, patients: Patients,
               splits: Tuple[List[int], ...], n_evals, continue_training: bool,
               prng_seed, trial_terminate_time, history: TrainingHistory,
               signals: TrainerSignals):

        train_ids, valid_ids, test_ids = splits
        if self.epochs > 0 and self.epochs < 1:
            epochs = 1
            train_ids = train_ids[:int(self.epochs * len(train_ids))]
        else:
            epochs = self.epochs

        n_train_admissions = patients.n_admissions(
            train_ids,
            ignore_first_admission=self.counts_ignore_first_admission)

        batch_size = min(self.batch_size, n_train_admissions)
        iters = round(self.epochs * n_train_admissions / batch_size)
        optimizer = Optimizer(self.optimizer_config, iters=iters, model=model)
        key = jrandom.PRNGKey(prng_seed)

        signals.start_training.send(self,
                                    params_size=params_size(model),
                                    n_iters=iters,
                                    config=self.config)
        eval_steps = sorted(set(
            np.linspace(0, iters - 1, n_evals).astype(int)))

        first_step = 0
        if continue_training:
            messenger = {'model': model, 'optimizer': optimizer}
            signals.continue_training.send(self, messenger=messenger)
            model = messenger['model']
            optimizer = messenger['optimizer']
            first_step = messenger.get('last_eval_step', 0)

        val_batch = patients.device_batch(valid_ids)
        step = 0
        for _ in tqdm_constructor(range(epochs), leave=True, unit='Epoch'):
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
                    signals.timeout.send(self)
                    break

                step += 1
                if step <= first_step:
                    continue
                try:
                    optimizer, model, loss_val = self.step_optimizer(
                        step, optimizer, model, batch)
                    batch_gen.set_description(f'Loss: {loss_val:.5f}')

                except RuntimeError as e:
                    signals.nan_detected.send(self,
                                              msg=f'Possible ODE failure: {e}')
                    return model

                if tree_hasnan(model):
                    signals.nan_detected.send(self,
                                              msg='NaN detected in model')
                    return model

                if step not in eval_steps:
                    continue

                history.append_train_preds(
                    model.batch_predict(batch, leave_pbar=False))
                if len(valid_ids) > 0:
                    history.append_val_preds(
                        model.batch_predict(val_batch, leave_pbar=False))

                if step == iters - 1 and len(test_ids) > 0:
                    test_batch = patients.device_batch(test_ids)
                    history.append_test_preds(
                        model.batch_predict(test_batch, leave_pbar=False))

                signals.end_evaluation.send(self,
                                            step=step,
                                            model=model,
                                            optimizer=optimizer,
                                            history=history)
            batch_gen.close()

        return model


def sample_training_config(cls, trial: optuna.Trial, model: AbstractModel):

    return {
        'epochs': 10,
        'batch_size': trial.suggest_int('B', 2, 27, 5),
        'opt': 'adam',
        'lr': trial.suggest_float('lr', 5e-5, 5e-3, log=True),
        'decay_rate': None,
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


class InTrainer(Trainer):

    def unreg_loss(self, model: AbstractModel, patients: Patients):
        preds = model.batch_predict(patients, leave_pbar=False)
        dx_loss = preds.prediction_dx_loss(dx_loss=self.dx_loss)
        obs_loss = preds.prediction_obs_loss(obs_loss=self.obs_loss)
        loss = dx_loss + obs_loss
        return loss
