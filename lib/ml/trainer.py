from __future__ import annotations

import contextlib
import logging
import os
import pickle
import random
import re
from abc import abstractmethod, ABCMeta
from dataclasses import field
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import List, Any, Dict, Tuple, Union, Optional, Callable

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
import optuna
import pandas as pd
from blinker import signal

from ._ademamix import ademamix
from .artefacts import AdmissionsPrediction
from .exp_ode_icnn import AutoKoopmanICNN
from .in_models import ICENODEStateTrajectory
from .koopman_modules import KoopmanOperator
from .model import AbstractModel
from ..base import Config, Module
from ..ehr import TVxEHR
from ..metric.loss import BinaryLossLiteral, NumericLossLiteral, ProbNumericLossLiteral
from ..metric.loss_wrap import (ProbObsPredictionLoss, ImputedProbObsPredictionLoss,
                                OutcomePredictionLoss, ObsPredictionLoss, LeadPredictionLoss, ImputationLoss)
from ..metric.metrics import (MetricsCollection, Metric, )
from ..utils import (params_size, tree_hasnan, tqdm_constructor, write_config,
                     append_params_to_zip, zip_members, translate_path)

_opts = {'sgd': optax.sgd,
         'adam': optax.adam,
         'novograd': optax.novograd,
         'lamb': optax.lamb,
         'ademamix': ademamix}

LRType = Union[float, Dict[str, float]]


class ResourceTimeout(Exception):
    """Raised when a trial is anticipated to
    be exceeding the timelimit of the compute."""
    pass


class StudyHalted(Exception):
    """Raised when a trial is spawned from a retired study."""
    pass


class TrainingHistory:

    def __init__(self, metrics: Tuple[Metric, ...]):
        self.metrics = MetricsCollection(metrics)
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

    def append_train_preds(self, step: int, res: AdmissionsPrediction,
                           elapsed_time: float, eval_time: float):
        row_df = self.metrics.to_df(step, res)
        row_df['timenow'] = datetime.now()
        row_df['elapsed_time'] = elapsed_time
        row_df['eval_time'] = (datetime.now() - eval_time).total_seconds()
        self._train_df = self._concat(self._train_df, row_df)

    def append_val_preds(self, step: int, res: AdmissionsPrediction,
                         elapsed_time: float, eval_time: float):
        row_df = self.metrics.to_df(step, res)
        row_df['timenow'] = datetime.now()
        row_df['elapsed_time'] = elapsed_time
        row_df['eval_time'] = (datetime.now() - eval_time).total_seconds()
        self._val_df = self._concat(self._val_df, row_df)

    def append_test_preds(self, step: int, res: AdmissionsPrediction,
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
        self.report_stats(sender, **kwargs)

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
        return [
            (trainer_signals.model_updated, self.record_stats),
            # (trainer_signals.model_snapshot, self.report_stats),
            (trainer_signals.new_training, self.clear_files)
        ]


class ParamsDiskWriter(AbstractReporter):

    def __init__(self, output_dir):
        self.output_dir = output_dir

    def report_params_optimizer(self, sender, **kwargs):
        model = kwargs['model']
        opt, opt_state = kwargs['optimizer']
        step = kwargs['step']

        tarname = os.path.join(self.output_dir, 'params.zip')
        name = f'step{step:04d}.eqx'
        append_params_to_zip(model, name, tarname)
        with open(os.path.join(self.output_dir, 'optstate.pkl'), 'wb') as f:
            pickle.dump(opt_state, f)

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
            opt, opt_state = messenger['optimizer']

            model = self.load_trained_model(model, last_eval_step)
            with open(os.path.join(self.output_dir, 'optstate.pkl'), 'rb') as f:
                opt_state = pickle.load(f)

            messenger['model'] = model
            messenger['optimizer'] = (opt, opt_state)
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


class ReportingConfig(Config):
    output_dir: Optional[str] = None
    console: bool = True
    parameter_snapshots: bool = False
    config_json: bool = False
    model_stats: bool = False


class TrainerReporting(Module):
    config: ReportingConfig = field(default_factory=ReportingConfig)
    metrics: Tuple[Metric, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if (self.config.config_json or len(self.metrics) > 0
                or self.config.parameter_snapshots or self.config.model_stats):
            assert self.config.output_dir is not None, 'output_dir must be provided'
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    @property
    def output_dir(self):
        return translate_path(self.config.output_dir)

    @cached_property
    def reporters(self) -> List[AbstractReporter]:
        reporters = []

        if self.config.console:
            reporters.append(ConsoleReporter())

        if self.config.parameter_snapshots:
            reporters.append(ParamsDiskWriter(self.output_dir))

        if self.metrics is not None and len(self.metrics) > 0:
            reporters.append(EvaluationDiskWriter(self.output_dir))

        if self.config.config_json:
            reporters.append(ConfigDiskWriter(self.output_dir))

        if self.config.model_stats:
            reporters.append(ModelStatsDiskWriter(self.output_dir))

        return reporters

    def new_training_history(self):
        return TrainingHistory(self.metrics or tuple())

    @property
    def supports_continue_training(self):
        return any([isinstance(r, ParamsDiskWriter) for r in self.reporters])

    def connections(self, trainer_signals: TrainerSignals):
        return [signal.connected_to(slot) for r in self.reporters for signal, slot in
                r.signal_slot_pairs(trainer_signals)]


class WarmupConfig(Config):
    epochs: float = 0.1
    batch_size: int = 16
    optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig())

    def __init__(self,
                 epochs: float = 0.1,
                 batch_size: int = 16,
                 optimizer=None,
                 opt: str = None,
                 lr: float = None):
        self.epochs = epochs
        self.batch_size = batch_size
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = OptimizerConfig(opt=opt, lr=lr)


class TrainerConfig(Config):
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    epochs: int = 100
    batch_size: int = 32
    outcome_loss: Optional[BinaryLossLiteral] = None
    obs_loss: Optional[NumericLossLiteral] = None
    imputed_obs_loss: Optional[NumericLossLiteral] = None
    lead_loss: Optional[NumericLossLiteral | BinaryLossLiteral] = None
    normalised_obs_loss: bool = False


class ProbTrainerConfig(TrainerConfig):
    obs_loss: Optional[ProbNumericLossLiteral] = None
    imputed_obs_loss: Optional[ProbNumericLossLiteral] = None


class LossMixer(Config):
    l1: float = 0.0
    l2: float = 0.0
    outcome: float = 1.0
    observables: float = 1.0
    imputed_observables: float = 1.0
    leading_observable: float = 1.0


class KoopmanLossMixer(LossMixer):
    l1: float = 0.0
    l2: float = 0.0
    outcome: float = 1.0
    observables: float = 1.0
    imputed_observables: float = 1.0
    leading_observable: float = 1.0
    reconstruction: float = 1.0


class Trainer(Module):
    config: TrainerConfig
    loss_mixer: LossMixer = field(default_factory=LossMixer)

    @cached_property
    def outcome_loss(self) -> OutcomePredictionLoss | Callable[[AdmissionsPrediction], float]:
        if self.config.outcome_loss is None:
            return lambda p: 0.0
        return OutcomePredictionLoss(loss_key=self.config.outcome_loss)

    @cached_property
    def obs_loss(self) -> ObsPredictionLoss | Callable[[AdmissionsPrediction], float]:
        if self.config.obs_loss is None:
            return lambda p: 0.0
        if self.config.normalised_obs_loss:
            obs_loss = ObsPredictionLoss(loss_key=self.config.obs_loss, per_column=True)  # noqa
            return lambda p: jnp.nanmean(obs_loss(p))  # noqa
        return ObsPredictionLoss(loss_key=self.config.obs_loss)  # noqa

    @cached_property
    def imputed_obs_loss(self) -> ImputationLoss | Callable[[AdmissionsPrediction], float]:
        if self.config.obs_loss is None:
            return lambda p: 0.0
        if self.config.normalised_obs_loss:
            obs_loss = ImputationLoss(loss_key=self.config.imputed_obs_loss, per_column=True)
            return lambda p: jnp.nanmean(obs_loss(p))  # noqa
        return ImputationLoss(loss_key=self.config.imputed_obs_loss)  # noqa

    @cached_property
    def lead_loss(self) -> LeadPredictionLoss | Callable[[AdmissionsPrediction], float]:
        if self.config.lead_loss is None:
            return lambda p: 0.0
        return LeadPredictionLoss(loss_key=self.config.lead_loss)

    def batch_predict(self, model: AbstractModel, patients: TVxEHR):
        return model.batch_predict(patients, leave_pbar=False)

    def loss_term(self, model: AbstractModel, predictions: AdmissionsPrediction):
        loss = (self.loss_mixer.outcome * self.outcome_loss(predictions) +
                self.loss_mixer.observables * self.obs_loss(predictions) +
                self.loss_mixer.imputed_observables * self.imputed_obs_loss(predictions) +
                self.loss_mixer.leading_observable * self.lead_loss(predictions))
        if self.loss_mixer.l1 != 0.0:
            loss += model.l1() * self.loss_mixer.l1
        if self.loss_mixer.l2 != 0.0:
            loss += model.l2() * self.loss_mixer.l2
        return loss

    def loss(self, model: AbstractModel, patients: TVxEHR):
        preds = self.batch_predict(model, patients)
        return self.loss_term(model, preds)

    def _post_update_params(self, model: AbstractModel):
        return model

    def step_optimizer(self, optimizer: Tuple[optax.GradientTransformationExtraArgs, optax.OptState],
                       model: AbstractModel, patients: TVxEHR):
        grad_f = eqx.filter_value_and_grad(self.loss)
        value, grads = grad_f(model, patients)
        grads = jtu.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))
        opt, opt_state = optimizer
        updates, opt_state = opt.update(grads, opt_state,
                                        params=jtu.tree_leaves(eqx.filter(model, eqx.is_inexact_array)),
                                        value=value, grad=grads)
        _, pdef = jtu.tree_flatten(eqx.filter(model, eqx.is_inexact_array))
        model = eqx.apply_updates(model, jtu.tree_unflatten(pdef, updates))
        model = self._post_update_params(model)
        return (opt, opt_state), model, value

    def __call__(self,
                 model: AbstractModel,
                 patients: TVxEHR,
                 model_snapshot_frequency: int = 0,
                 n_evals=100,
                 train_split: Optional[Tuple[str, ...]] = None,
                 val_split: Optional[Tuple[str, ...]] = None,
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

        if warmup_config is not None:
            if (not continue_training) or (ParamsDiskWriter.last_eval_step(reporting.output_dir) is None):
                logging.info('Warming up...')
                model = self._warmup(model=model,
                                     patients=patients,
                                     train_split=train_split,
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
                train_split=train_split,
                val_split=val_split,
                model_snapshot_frequency=model_snapshot_frequency,
                n_evals=n_evals,
                continue_training=continue_training,
                prng_seed=prng_seed,
                trial_terminate_time=trial_terminate_time,
                history=reporting.new_training_history(),
                signals=signals,
                exported_config=exported_config)

    def _warmup(self, model: AbstractModel, patients: TVxEHR,
                train_split: Tuple[str, ...], prng_seed, trial_terminate_time,
                history: TrainingHistory, signals: TrainerSignals,
                warmup_config: WarmupConfig):

        conf = self.config.update(warmup_config)
        trainer = type(self)(config=conf, loss_mixer=self.loss_mixer)

        return trainer._train(model=model,
                              patients=patients,
                              train_split=train_split,
                              n_evals=0,
                              model_snapshot_frequency=0,
                              continue_training=False,
                              prng_seed=prng_seed,
                              trial_terminate_time=trial_terminate_time,
                              history=history,
                              signals=signals,
                              exported_config=conf.to_dict(),
                              val_split=None)

    def _train(self,
               model: AbstractModel,
               patients: TVxEHR,
               train_split: Optional[Tuple[str, ...]],
               val_split: Optional[Tuple[str, ...]],
               model_snapshot_frequency: int,
               n_evals: int,
               continue_training: bool,
               prng_seed,
               trial_terminate_time,
               history: TrainingHistory,
               signals: TrainerSignals,
               exported_config: Dict[str, Any]):

        if exported_config is None:
            exported_config = {}
        if train_split is None:
            train_split = patients.subject_ids
        else:
            train_split = list(train_split)

        if val_split is None:
            val_split = []

        if self.config.epochs > 0 and self.config.epochs < 1:
            epochs = 1
            train_split = train_split[:int(self.config.epochs * len(train_split)) + 1]
        else:
            epochs = self.config.epochs

        n_train_admissions = patients.n_admissions(train_split, ignore_first_admission=model.discount_first_admission)

        batch_size = min(self.config.batch_size, n_train_admissions)
        iters = round(self.config.epochs * n_train_admissions / batch_size)
        opt = _opts[self.config.optimizer.opt](self.config.optimizer.lr)
        opt_state = opt.init(jtu.tree_leaves(eqx.filter(model, eqx.is_inexact_array)))
        optimizer = (opt, opt_state)

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
        val_batch = patients.device_batch(val_split)
        step = 0
        for _ in tqdm_constructor(range(epochs), leave=False, unit='Epoch'):
            pyrng.shuffle(train_split)
            epoch_splits = patients.epoch_splits(train_split, batch_n_admissions=batch_size,
                                                 discount_first_admission=model.discount_first_admission)
            split_gen = tqdm_constructor(epoch_splits, leave=False, unit='Batch')

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
                    next_eval_step = min((s for s in eval_steps if s > step),
                                         default=0)
                    steps_until_eval = next_eval_step - step

                    batch = patients.device_batch(batch_split)
                    optimizer, model, loss_val = self.step_optimizer(optimizer, model, batch)
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
                    signals.model_snapshot.send(self,
                                                step=step,
                                                model=model,
                                                optimizer=optimizer)
                if step in eval_steps and len(val_split) > 0:
                    elapsed_time = (datetime.now() - timenow).total_seconds()
                    timenow = datetime.now()

                    split_gen.set_description('Evaluating (Train) (A)...')
                    preds = model.batch_predict(batch, leave_pbar=False)
                    split_gen.set_description('Evaluating (Train) (B)...')
                    history.append_train_preds(step, preds, elapsed_time, timenow)

                    if len(val_split) > 0:
                        timenow = datetime.now()
                        split_gen.set_description('Evaluating (Val) (A)...')
                        preds = model.batch_predict(val_batch,
                                                    leave_pbar=False)
                        split_gen.set_description('Evaluating (Val) (B)...')
                        history.append_val_preds(step, preds, elapsed_time, timenow)
                    signals.end_evaluation.send(self,
                                                step=step,
                                                model=model,
                                                optimizer=optimizer,
                                                history=history)
                    timenow = datetime.now()
            split_gen.close()
        signals.exit_training.send(self, history=history)
        return model


# class LassoNetTrainer(Trainer):
#
#     def _post_update_params(self, model):
#         if self.reg_hyperparams:
#             return model.prox_map()(model, self.reg_hyperparams)
#         else:
#             return model


class ProbTrainer(Trainer):
    config: ProbTrainerConfig

    @cached_property
    def obs_loss(self) -> ProbObsPredictionLoss:
        if self.config.prob_obs_loss is None:
            return lambda p: 0.0
        return ProbObsPredictionLoss(loss_key=self.config.obs_loss)

    @cached_property
    def imputed_obs_loss(self) -> ProbObsPredictionLoss:
        if self.config.imputed_obs_loss is None:
            return lambda p: 0.0
        return ImputedProbObsPredictionLoss(loss_key=self.config.imputed_obs_loss)

    def batch_predict(self, model: AbstractModel, patients: TVxEHR):
        return model.batch_predict(patients, leave_pbar=False)

    def loss_term(self, model: AbstractModel, predictions: AdmissionsPrediction):
        loss = (self.loss_mixer.outcome * self.outcome_loss(predictions) +
                self.loss_mixer.observables * self.obs_loss(predictions) +
                self.loss_mixer.imputed_observables * self.imputed_obs_loss(predictions) +
                self.loss_mixer.leading_observable * self.lead_loss(predictions))
        if self.loss_mixer.l1 != 0.0:
            loss += model.l1() * self.loss_mixer.l1
        if self.loss_mixer.l2 != 0.0:
            loss += model.l2() * self.loss_mixer.l2
        return loss


class KoopmanTrainer(Trainer):
    loss_mixer: KoopmanLossMixer = field(default_factory=KoopmanLossMixer)

    def reconstruction_loss(self, koopman_operator: KoopmanOperator, predictions: AdmissionsPrediction) -> jnp.ndarray:
        def trajectory_loss(trajectory: ICENODEStateTrajectory) -> jnp.ndarray:
            state = jnp.vstack((trajectory.imputed_state, trajectory.forecasted_state))
            return eqx.filter_vmap(koopman_operator.compute_phi_loss)(state)

        trajs_loss = [trajectory_loss(prediction.trajectory) for prediction in predictions if
                      prediction.trajectory is not None]

        if len(trajs_loss) > 0:
            return jnp.mean(jnp.hstack(trajs_loss))
        else:
            return 0.0

    def loss_term(self, model: AutoKoopmanICNN, predictions: AdmissionsPrediction):
        loss = (self.loss_mixer.outcome * self.outcome_loss(predictions) +
                self.loss_mixer.observables * self.obs_loss(predictions) +
                self.loss_mixer.observables * self.imputed_obs_loss(predictions) +
                self.loss_mixer.leading_observable * self.lead_loss(predictions) +
                self.loss_mixer.reconstruction * self.reconstruction_loss(model.f_dyn, predictions))
        if self.loss_mixer.l1 != 0.0:
            loss += model.l1() * self.loss_mixer.l1
        if self.loss_mixer.l2 != 0.0:
            loss += model.l2() * self.loss_mixer.l2
        return loss
