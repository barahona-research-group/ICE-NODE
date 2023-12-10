from __future__ import annotations
from typing import List, Any, Dict, Tuple, Union, Optional, Callable
from datetime import datetime

import os
import re
import random
import contextlib
from pathlib import Path
import pickle
import logging
from abc import abstractmethod, ABCMeta
import pandas as pd
import numpy as np
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
from ..base import Config, Module
from .model import AbstractModel, ModelRegularisation

_opts = {'sgd': jopt.sgd, 'adam': jopt.adam}

LRType = Union[float, Dict[str, float]]


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
        self._stats_df = None

    @property
    def train_df(self):
        return self._train_df

    @property
    def val_df(self):
        return self._val_df

    @property
    def test_df(self):
        return self._test_df

    @property
    def stats_df(self):
        return self._stats_df

    @staticmethod
    def _concat(df, row):
        if df is None:
            return row
        else:
            return pd.concat([df, row], axis=0)

    def append_train_preds(self, step: int, res: Predictions,
                           elapsed_time: float, eval_time: float):
        row_df = self.metrics.to_df(step, res)
        row_df['timenow'] = datetime.now()
        row_df['elapsed_time'] = elapsed_time
        row_df['eval_time'] = (datetime.now() - eval_time).total_seconds()
        self._train_df = self._concat(self._train_df, row_df)

    def append_val_preds(self, step: int, res: Predictions,
                         elapsed_time: float, eval_time: float):
        row_df = self.metrics.to_df(step, res)
        row_df['timenow'] = datetime.now()
        row_df['elapsed_time'] = elapsed_time
        row_df['eval_time'] = (datetime.now() - eval_time).total_seconds()
        self._val_df = self._concat(self._val_df, row_df)

    def append_test_preds(self, step: int, res: Predictions,
                          elapsed_time: float, eval_time: float):
        row_df = self.metrics.to_df(step, res)
        row_df['timenow'] = datetime.now()
        row_df['elapsed_time'] = elapsed_time
        row_df['eval_time'] = (datetime.now() - eval_time).total_seconds()
        self._test_df = self._concat(self._test_df, row_df)

    def append_stats(self, step: int, model: AbstractModel, loss):
        pathwise_stats = model.pathwise_params_stats()
        data = pathwise_stats | {'other_stats': {'step': step, 'loss': loss}}
        # https://stackoverflow.com/a/66383008
        reform = {
            (outer_key, inner_key): values
            for outer_key, inner_dict in data.items()
            for inner_key, values in inner_dict.items()
        }
        row_df = pd.DataFrame.from_dict(reform, 'index').transpose()
        row_df.columns = pd.MultiIndex.from_tuples(row_df.columns)
        row_df = row_df.set_index(pd.Index([step]).rename('step'))
        self._stats_df = self._concat(self._stats_df, row_df)


class TrainerSignals:
    new_training = signal('new_training')
    start_training = signal('start_training')
    end = signal('end_training')
    start_evaluation = signal('start_evaluation')
    train_evaluation = signal('train_evaluation')
    val_evaluation = signal('val_evaluation')
    test_evaluation = signal('test_evaluation')
    end_evaluation = signal('end_evaluation')
    exit_training = signal('exit_training')
    continue_training = signal('continue_training')
    timeout = signal('timeout')
    nan_detected = signal('nan_detected')
    model_updated = signal('model_updated')
    model_snapshot = signal('model_snapshot')

    def disconnect_all_receivers(self):
        for sig in self.__dict__.values():
            for recv in sig.receivers.values():
                sig.disconnect(recv)


class AbstractReporter(metaclass=ABCMeta):

    @abstractmethod
    def signal_slot_pairs(self, trainer_signals: TrainerSignals):
        pass


class ConsoleReporter(AbstractReporter):

    def log_config(self, sender, **kw):
        logging.info(f'HPs: {sender.config}')

    def log_nan_detected(self, sender, **kwargs):
        logging.warning(kwargs['msg'] or 'NaN detected')

    def signal_slot_pairs(self, trainer_signals: TrainerSignals):
        return [(trainer_signals.start_training, self.log_config),
                (trainer_signals.nan_detected, self.log_nan_detected)]


class EvaluationDiskWriter(AbstractReporter):

    def __init__(self, output_dir):
        self.output_dir = output_dir

    def report_evaluation(self, sender, **kwargs):
        h = kwargs['history']

        for name, df in zip(('train', 'val', 'tst'),
                            (h.train_df, h.val_df, h.test_df)):
            if df is None:
                continue
            df = df.copy()

            fname = f'{name}_evals.csv.gz'
            fpath = os.path.join(self.output_dir, fname)
            header_changed = False
            if os.path.exists(fpath):
                old_df = pd.read_csv(fpath, index_col=0)
                df = df.loc[~df.index.isin(old_df.index)]
                header_changed = not old_df.columns.equals(df.columns)

            df.to_csv(fpath,
                      compression="gzip",
                      mode='a',
                      header=header_changed or not os.path.exists(fpath))

    def clear_files(self, sender):
        for name in ('train', 'val', 'tst'):
            fname = f'{name}_evals.csv.gz'
            fpath = os.path.join(self.output_dir, fname)
            if os.path.exists(fpath):
                os.remove(fpath)

    def signal_slot_pairs(self, trainer_signals: TrainerSignals):
        return [(trainer_signals.end_evaluation, self.report_evaluation),
                (trainer_signals.new_training, self.clear_files)]


class ModelStatsDiskWriter(AbstractReporter):

    def __init__(self, output_dir):
        self.output_dir = output_dir

    def record_stats(self, sender, **kwargs):
        model = kwargs['model']
        step = kwargs['step']
        loss = kwargs['loss_val']
        h = kwargs['history']
        h.append_stats(step, model, loss)

    def report_stats(self, sender, **kwargs):
        h = kwargs['history']
        fname = 'stats.csv.gz'
        fpath = os.path.join(self.output_dir, fname)
        h.stats_df.to_csv(fpath, compression="gzip")

    def clear_files(self, sender):
        fname = 'stats.csv.gz'
        fpath = os.path.join(self.output_dir, fname)
        if os.path.exists(fpath):
            os.remove(fpath)

    def signal_slot_pairs(self, trainer_signals: TrainerSignals):
        return [(trainer_signals.model_updated, self.record_stats),
                (trainer_signals.exit_training, self.report_stats),
                (trainer_signals.new_training, self.clear_files)]


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

    def clear_files(self, sender):
        tarname = os.path.join(self.output_dir, 'params.zip')
        if os.path.exists(tarname):
            os.remove(tarname)
        optname = os.path.join(self.output_dir, 'optstate.pkl')
        if os.path.exists(optname):
            os.remove(optname)

    @staticmethod
    def last_eval_step(output_dir):
        tarname = os.path.join(output_dir, 'params.zip')
        try:
            filenames = zip_members(tarname)
        except FileNotFoundError:
            return None

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
        if last_eval_step is not None and last_eval_step > 0:
            model = messenger['model']
            optimizer = messenger['optimizer']

            model = self.load_trained_model(model, last_eval_step)
            optimizer = optimizer.load(
                os.path.join(self.output_dir, 'optstate.pkl'))
            messenger['model'] = model
            messenger['optimizer'] = optimizer
            messenger['step'] = last_eval_step

    def signal_slot_pairs(self, trainer_signals: TrainerSignals):
        return [(trainer_signals.end_evaluation, self.report_params_optimizer),
                (trainer_signals.model_snapshot, self.report_params_optimizer),
                (trainer_signals.continue_training, self.continue_training),
                (trainer_signals.new_training, self.clear_files)]


class ConfigDiskWriter(AbstractReporter):

    def __init__(self, output_dir):
        self.output_dir = output_dir

    def report_config(self, sender, **kwargs):
        config = kwargs['config']
        write_config(config, os.path.join(self.output_dir, 'config.json'))

    def clear_files(self, sender):
        config_name = os.path.join(self.output_dir, 'config.json')
        if os.path.exists(config_name):
            os.remove(config_name)

    def signal_slot_pairs(self, trainer_signals: TrainerSignals):
        return [(trainer_signals.start_training, self.report_config),
                (trainer_signals.new_training, self.clear_files)]


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


class OptimizerConfig(Config):
    opt: str = 'adam'
    lr: LRType = 1e-3
    decay_rate: Optional[LRType] = None
    reverse_schedule: bool = False


class Optimizer(Module):
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
        super().__init__(config=config)
        self.iters = iters
        lr = self.lr_schedule(config, iters)

        opt_init, self.opt_update, self.get_params = _opts[config.opt](lr)

        if optstate is None and model is not None:
            self.optstate = opt_init(model.params_list())
        elif optstate is not None:
            self.optstate = optstate
        else:
            raise ValueError('Either optstate or model must be provided')

    @classmethod
    def external_argnames(cls):
        return ['iters', 'optstate', 'model']

    def step(self, step, grads):
        grads = jtu.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))
        optstate = self.opt_update(step, grads, self.optstate)
        return eqx.tree_at(lambda x: x.optstate, self, optstate)

    def __call__(self, model):
        new_params = self.get_params(self.optstate)
        param_part, other_part = eqx.partition(model, eqx.is_inexact_array)
        _, pdef = jtu.tree_flatten(param_part)
        params_part = jtu.tree_unflatten(pdef, new_params)
        return eqx.combine(params_part, other_part)

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


class MultiLearningRateOptimizer(Optimizer):
    grads_filter: Dict[str, Any]

    def __init__(self,
                 config: OptimizerConfig,
                 model=None,
                 iters=None,
                 optstate=None,
                 grads_filter=None):
        Module.__init__(self, config=config)
        self.iters = iters
        lr = self.lr_schedule(config, iters)

        self.get_params = {}
        self.optstate = {}
        self.opt_update = {}

        opt = _opts[config.opt]
        if model is not None:
            if grads_filter is None:
                self.grads_filter = model.params_list_mask(config.lr)
            if optstate is None:
                params_list = model.params_list()
                for k, lr in self.lr_schedule(config, iters).items():
                    params = eqx.filter(params_list, self.grads_filter[k])
                    opt_init, self.opt_update[k], self.get_params[k] = opt(lr)
                    self.optstate[k] = opt_init(params)

        elif optstate is not None and grads_filter is not None:
            for k, lr in self.lr_schedule(config, iters).items():
                opt_init, self.opt_update[k], self.get_params[k] = opt(lr)
            self.optstate = optstate
            self.grads_filter = grads_filter
        else:
            raise ValueError(
                'Either (optstate AND grads_filter) or model must be provided')

    @classmethod
    def external_argnames(cls):
        return ['iters', 'optstate', 'model', 'grads_filter']

    def step(self, step, grads):
        grads = jtu.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))
        updated_state = {}
        for k, grads_filter in self.grads_filter.items():
            grad_k = eqx.filter(grads, grads_filter)
            updated_state[k] = self.opt_update[k](step, grad_k,
                                                  self.optstate[k])

        return eqx.tree_at(lambda x: x.optstate, self, updated_state)

    def __call__(self, model):

        param_part, other_part = eqx.partition(model, eqx.is_inexact_array)
        _, pdef = jtu.tree_flatten(param_part)
        new_params = []
        for k, get_params in self.get_params.items():
            new_params.append(get_params(self.optstate[k]))
        new_params = eqx.combine(*new_params)
        params_part = jtu.tree_unflatten(pdef, new_params)
        return eqx.combine(params_part, other_part)

    @staticmethod
    def lr_schedule(config, iters=None, reverse=False):
        assert isinstance(config.lr, dict), 'lr must be either float or dict'

        if config.decay_rate is None or iters is None:
            return config.lr

        def schedule_gen(key):
            lr = config.lr[key]
            if isinstance(config.decay_rate, dict):
                _decay_rate = config.decay_rate[key]
            else:
                _decay_rate = config.decay_rate

            schedule = jopt.exponential_decay(lr,
                                              decay_steps=iters // 2,
                                              decay_rate=_decay_rate)
            if reverse:
                return lambda i: schedule(iters - i)

            return schedule

        return {k: schedule_gen(k) for k in config.lr.keys()}

    def load(self, filename):
        state_filename = filename + '.st'
        filter_filname = filename + '.flt'

        with open(filter_filname, "rb") as filter_file:
            grads_filter = pickle.load(filter_file)

        optstate = {}
        for k in grads_filter.keys():
            _filename = state_filename + '.' + k
            with open(_filename, "rb") as optstate_file:
                optstate_k = pickle.load(optstate_file)
                optstate_k = jopt.pack_optimizer_state(optstate_k)
                optstate[k] = optstate_k

        return MultiLearningRateOptimizer(config=self.config,
                                          optstate=optstate,
                                          iters=self.iters,
                                          grads_filter=grads_filter)

    def save(self, filename):
        state_filename = filename + '.st'
        filter_filname = filename + '.flt'

        for k, optstate_k in self.optstate.items():
            _filename = state_filename + '.' + k
            optstate_k = jopt.unpack_optimizer_state(optstate_k)
            with open(_filename, "wb") as optstate_file:
                pickle.dump(optstate_k, optstate_file)

        with open(filter_filname, "wb") as filter_file:
            pickle.dump(self.grads_filter, filter_file)


def make_optimizer(config: OptimizerConfig, *args, **kwargs):
    if isinstance(config.lr, dict):
        return MultiLearningRateOptimizer(config, *args, **kwargs)
    else:
        return Optimizer(config, *args, **kwargs)


class ReportingConfig(Config):
    output_dir: Optional[str] = None
    console: bool = True
    parameter_snapshots: bool = False
    config_json: bool = False
    model_stats: bool = False


class TrainerReporting(Module):
    config: ReportingConfig
    _reporters: List[AbstractReporter]
    _metrics: MetricsCollection

    def __init__(
        self,
        config: ReportingConfig = ReportingConfig(),
        metrics: Optional[List[Metric]] = None,
        # optuna_trial: Optional[optuna.Trial] = None,
        # optuna_objective: Optional[Callable] = None
    ):
        super().__init__(config=config)

        reporters = []
        if (config.config_json or metrics is not None
                or config.parameter_snapshots or config.model_stats):
            assert config.output_dir is not None, 'output_dir must be provided'
            output_dir = translate_path(config.output_dir)
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        if config.console:
            reporters.append(ConsoleReporter())

        if config.parameter_snapshots:
            reporters.append(ParamsDiskWriter(output_dir))

        if metrics is not None and len(metrics) > 0:
            reporters.append(EvaluationDiskWriter(output_dir))

        if config.config_json:
            reporters.append(ConfigDiskWriter(output_dir))

        if config.model_stats:
            reporters.append(ModelStatsDiskWriter(output_dir))


#         if optuna_trial is not None:
#             assert optuna_objective is not None, ('optuna_objective must '
#                                                   'be provided')
#             reporters.append(OptunaReporter(optuna_trial, optuna_objective))

        self._reporters = reporters
        self._metrics = MetricsCollection(metrics or [])

    @classmethod
    def external_argnames(cls):
        return ['metrics']

    def new_training_history(self):
        return TrainingHistory(self._metrics)

    @property
    def supports_continue_training(self):
        return any([isinstance(r, ParamsDiskWriter) for r in self._reporters])

    def connections(self, trainer_signals: TrainerSignals):
        return [
            signal.connected_to(slot) for r in self._reporters
            for signal, slot in r.signal_slot_pairs(trainer_signals)
        ]


class WarmupConfig(Config):
    epochs: float = 0.1
    batch_size: int = 16
    optimizer: OptimizerConfig = OptimizerConfig(reverse_schedule=True)

    def __init__(self,
                 epochs: float = 0.1,
                 batch_size: int = 16,
                 optimizer=None,
                 opt: str = None,
                 lr: float = None,
                 decay_rate: Optional[float] = None):
        self.epochs = epochs
        self.batch_size = batch_size
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = OptimizerConfig(opt=opt,
                                             lr=lr,
                                             decay_rate=decay_rate,
                                             reverse_schedule=True)


class TrainerConfig(Config):
    optimizer: OptimizerConfig = OptimizerConfig()
    epochs: int = 100
    batch_size: int = 32
    dx_loss: str = 'balanced_focal_bce'
    obs_loss: str = 'mse'
    lead_loss: str = 'mse'


class Trainer(Module):
    config: TrainerConfig
    reg_hyperparams: ModelRegularisation
    _dx_loss: Callable
    _obs_loss: Callable
    _lead_loss: Callable

    def __init__(self,
                 config: TrainerConfig,
                 reg_hyperparams: ModelRegularisation = ModelRegularisation):
        super().__init__(config=config)
        if reg_hyperparams is None:
            reg_hyperparams = ModelRegularisation()
        self.config = config
        self.reg_hyperparams = reg_hyperparams
        self._dx_loss = binary_loss[config.dx_loss]
        self._obs_loss = numeric_loss[config.obs_loss]
        self._lead_loss = numeric_loss[config.lead_loss]

    @classmethod
    def external_argnames(cls):
        return ['reg_hyperparams']

    def batch_predict(self, model: AbstractModel, patients: Patients):
        return model.batch_predict(patients,
                                   leave_pbar=False,
                                   regularisation=self.reg_hyperparams)

    def loss_term(self, model: AbstractModel, predictions: Predictions):
        loss = predictions.prediction_dx_loss(dx_loss=self._dx_loss)
        hps = self.reg_hyperparams.as_dict()
        if hps.get('L_l1', 0.0) != 0.0:
            loss += model.l1() * hps['L_l1']
            hps.pop('L_l1')
        if hps.pop('L_l2', 0.0) != 0.0:
            loss += model.l2() * hps['L_l2']
            hps.pop('L_l2')
        if len(hps) > 0:
            loss += self.prediction_auxiliary_loss(predictions)
        return loss

    def prediction_auxiliary_loss(self, predictions: Predictions):
        return predictions.associative_regularisation(self.reg_hyperparams)

    def loss(self, model: AbstractModel, patients: Patients):
        preds = self.batch_predict(model, patients)
        return self.loss_term(model, preds)

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
                 model_snapshot_frequency: int = 0,
                 n_evals=100,
                 reporting: TrainerReporting = TrainerReporting(),
                 warmup_config: Optional[WarmupConfig] = None,
                 continue_training: bool = False,
                 prng_seed: int = 0,
                 trial_terminate_time=datetime.max,
                 exported_config: Optional[Dict[str, Any]] = None):
        if continue_training:
            assert reporting.supports_continue_training, (
                'TrainerReporting must support continue_training. '
                'Make sure to include a ParamsDiskWriter '
                'in the reporters list.')

        if not continue_training and warmup_config is not None:
            logging.info('Warming up...')
            model = self._warmup(model=model,
                                 patients=patients,
                                 splits=splits,
                                 prng_seed=prng_seed,
                                 trial_terminate_time=trial_terminate_time,
                                 history=reporting.new_training_history(),
                                 signals=TrainerSignals(),
                                 warmup_config=warmup_config)
            logging.info('[DONE] Warming up.')

        exported_config = exported_config or {}

        with contextlib.ExitStack() as stack:
            signals = TrainerSignals()
            # In case Context Managers failed to exit from a previous
            # run (e.g. due to interruption via Jupyter Notebook).
            signals.disconnect_all_receivers()
            for c in reporting.connections(signals):
                stack.enter_context(c)

            return self._train(
                model=model,
                patients=patients,
                splits=splits,
                model_snapshot_frequency=model_snapshot_frequency,
                n_evals=n_evals,
                continue_training=continue_training,
                prng_seed=prng_seed,
                trial_terminate_time=trial_terminate_time,
                history=reporting.new_training_history(),
                signals=signals,
                exported_config=exported_config)

    def _warmup(self, model: AbstractModel, patients: Patients,
                splits: Tuple[List[int], ...], prng_seed, trial_terminate_time,
                history: TrainingHistory, signals: TrainerSignals,
                warmup_config: WarmupConfig):

        conf = self.config.update(warmup_config)
        trainer = type(self)(conf, reg_hyperparams=self.reg_hyperparams)
        return trainer._train(model=model,
                              patients=patients,
                              splits=(splits[0], [], []),
                              n_evals=0,
                              continue_training=False,
                              prng_seed=prng_seed,
                              trial_terminate_time=trial_terminate_time,
                              history=history,
                              signals=signals)

    def _train(self,
               model: AbstractModel,
               patients: Patients,
               splits: Tuple[List[int], ...],
               model_snapshot_frequency: int,
               n_evals: int,
               continue_training: bool,
               prng_seed,
               trial_terminate_time,
               history: TrainingHistory,
               signals: TrainerSignals,
               exported_config: Any = {}):

        train_ids, valid_ids, test_ids = splits
        if self.config.epochs > 0 and self.config.epochs < 1:
            epochs = 1
            train_ids = train_ids[:int(self.config.epochs * len(train_ids)) +
                                  1]
        else:
            epochs = self.config.epochs

        n_train_admissions = patients.n_admissions(
            train_ids,
            ignore_first_admission=model.counts_ignore_first_admission)

        batch_size = min(self.config.batch_size, n_train_admissions)
        iters = round(self.config.epochs * n_train_admissions / batch_size)
        optimizer = make_optimizer(self.config.optimizer,
                                   iters=iters,
                                   model=model)
        pyrng = random.Random(prng_seed)
        eval_steps = sorted(set(
            np.linspace(0, iters - 1, n_evals).astype(int)))

        if model_snapshot_frequency > 0:
            snapshot_steps = sorted(
                set(
                    np.arange(model_snapshot_frequency, iters -
                              1, model_snapshot_frequency).astype(int)) -
                set(eval_steps))
        else:
            snapshot_steps = []

        first_step = 0
        if continue_training:
            messenger = {'model': model, 'optimizer': optimizer}
            signals.continue_training.send(self, messenger=messenger)
            model = messenger['model']
            optimizer = messenger['optimizer']
            first_step = messenger.get('step', 0)
            logging.info(f'Continuing training from step {first_step}')

        if first_step == 0:
            signals.new_training.send(self)

        signals.start_training.send(self,
                                    params_size=params_size(model),
                                    n_iters=iters,
                                    config=exported_config)
        val_batch = patients.device_batch(valid_ids)
        step = 0
        for _ in tqdm_constructor(range(epochs), leave=False, unit='Epoch'):
            pyrng.shuffle(train_ids)
            epoch_splits = patients.epoch_splits(
                train_ids,
                batch_n_admissions=batch_size,
                ignore_first_admission=model.counts_ignore_first_admission)
            split_gen = tqdm_constructor(epoch_splits,
                                         leave=False,
                                         unit='Batch')

            timenow = datetime.now()
            for batch_split in split_gen:
                if datetime.now() > trial_terminate_time:
                    signals.timeout.send(self)
                    signals.exit_training.send(self)
                    return model

                step += 1
                if step <= first_step:
                    continue
                try:
                    next_eval_step = min(s for s in eval_steps if s > step)
                    steps_until_eval = next_eval_step - step

                    batch = patients.device_batch(batch_split)
                    optimizer, model, loss_val = self.step_optimizer(
                        step, optimizer, model, batch)
                    split_gen.set_description(
                        f'Loss: {loss_val:.4E} | {steps_until_eval} steps until eval.'
                    )

                    signals.model_updated.send(self,
                                               model=model,
                                               history=history,
                                               step=step,
                                               loss_val=loss_val)

                except RuntimeError as e:
                    signals.nan_detected.send(self,
                                              msg=f'Possible ODE failure: {e}')
                    signals.exit_training.send(self, history=history)
                    return model

                if tree_hasnan(model):
                    signals.nan_detected.send(self,
                                              msg='NaN detected in model')
                    signals.exit_training.send(self, history=history)
                    return model

                if step in snapshot_steps:

                    old_desc = split_gen.desc
                    split_gen.set_description('Saving model...')
                    signals.model_snapshot.send(self,
                                                step=step,
                                                model=model,
                                                optimizer=optimizer)
                    split_gen.set_description_str(old_desc)

                if step not in eval_steps:
                    continue

                elapsed_time = (datetime.now() - timenow).total_seconds()
                timenow = datetime.now()

                split_gen.set_description('Evaluating (Train) (A)...')
                preds = model.batch_predict(batch, leave_pbar=False)
                split_gen.set_description('Evaluating (Train) (B)...')
                history.append_train_preds(step, preds, elapsed_time, timenow)

                if len(valid_ids) > 0:
                    timenow = datetime.now()
                    split_gen.set_description('Evaluating (Val) (A)...')
                    preds = model.batch_predict(val_batch, leave_pbar=False)
                    split_gen.set_description('Evaluating (Val) (B)...')
                    history.append_val_preds(step, preds, elapsed_time,
                                             timenow)
                signals.end_evaluation.send(self,
                                            step=step,
                                            model=model,
                                            optimizer=optimizer,
                                            history=history)
                timenow = datetime.now()
            split_gen.close()

        if len(test_ids) > 0 and first_step < step:
            test_batch = patients.device_batch(test_ids)
            timenow = datetime.now()
            preds = model.batch_predict(test_batch, leave_pbar=False)
            history.append_test_preds(step, preds, 0.0, timenow)
            signals.end_evaluation.send(self,
                                        step=step,
                                        model=model,
                                        optimizer=optimizer,
                                        history=history)

        signals.exit_training.send(self, history=history)
        return model


class LassoNetTrainer(Trainer):

    def _post_update_params(self, model):
        if self.reg_hyperparams:
            return model.prox_map()(model, self.reg_hyperparams)
        else:
            return model


class InTrainer(Trainer):

    def loss_term(self, model: AbstractModel, predictions: Predictions):
        loss = super().loss_term(model, predictions)
        obs_loss = predictions.prediction_obs_loss(obs_loss=self._obs_loss)
        lead_loss = predictions.prediction_lead_loss(lead_loss=self._lead_loss)
        return loss + obs_loss + lead_loss


class InSKELKoopmanTrainer(InTrainer):

    def prediction_auxiliary_loss(self, predictions: Predictions):
        preds = predictions.get_predictions()
        loss = sum((p.auxiliary_loss['L_rec'] for p in preds), [])
        alpha_rec = self.reg_hyperparams.L_rec
        return alpha_rec * sum(loss) / len(loss)
