"""."""

import os
import re
import pickle

from absl import logging
import optuna

from .. import utils as U


class ResourceTimeout(Exception):
    """Raised when a trial is anticipated to be exceeding the timelimit of the compute."""


class StudyHalted(Exception):
    """Raised when a trial is spawned from a retired study."""


class AbstractReporter:
    """
    Different loggers and reporters:
        1. Optuna reporter
        2. MLFlow reporter
        3. logging
        4. evaluation disk writer
        5. parameters disk writer
    """

    def report_config(self):
        pass

    def report_params_size(self, size):
        pass

    def report_steps(self, steps):
        pass

    def report_progress(self, eval_step):
        pass

    def report_timeout(self):
        pass

    def report_nan_detected(self):
        pass

    def report_one_interation(self):
        pass

    def report_evaluation(self, history):
        pass

    def report_params(self, step, model, opt_state):
        pass

    def last_eval_step(self):
        return None

    def trained_model(self, step):
        return None


class MinibatchLogger(AbstractReporter):

    def __init__(self, config):
        self.config = config

    def report_config(self):
        logging.info(f'HPs: {self.config}')

    def report_nan_detected(self, msg=None):
        logging.warning(msg or 'NaN detected')

    def report_evaluation(self, history):
        h = history
        for name, df in zip(('train', 'val', 'tst'),
                            (h.train_df(), h.validation_df(), h.test_df())):
            logging.info(name + str(df))


class EvaluationDiskWriter(AbstractReporter):

    def __init__(self, output_dir, prefix=''):
        self.output_dir = output_dir
        self.prefix = prefix

    def report_evaluation(self, history: ".MetricsHistory"):
        h = history
        for name, df in zip(('train', 'val', 'tst'),
                            (h.train_df(), h.validation_df(), h.test_df())):
            if df is None:
                continue
            fname = f'{self.prefix}_{name}_evals.csv.gz'
            fpath = os.path.join(self.output_dir, fname)
            df.to_csv(fpath, compression="gzip")


class ParamsDiskWriter(AbstractReporter):

    def __init__(self, output_dir, prefix=''):
        self.output_dir = output_dir
        self.prefix = prefix

    def report_params(self, step, model, opt_state):
        tarname = os.path.join(self.output_dir, f'{self.prefix}params.zip')
        name = f'step{step:04d}.eqx'
        optstate_name = os.path.join(self.output_dir,
                                     f'{self.prefix}optstate.pkl')
        U.append_params_to_zip(model, name, tarname)

        with open(optstate_name, "wb") as optstate_file:
            pickle.dump(opt_state, optstate_file)

    def last_eval_step(self):
        tarname = os.path.join(self.output_dir, f'{self.prefix}params.zip')
        filenames = U.zip_members(tarname)
        numbers = []
        for fname in filenames:
            numbers.extend(map(int, re.findall(r'\d+', fname)))
        return sorted(numbers)[-1]

    def trained_model(self, model, step):
        tarfname = os.path.join(self.output_dir, f'{self.prefix}params.zip')
        membername = f'step{step:04d}.eqx'
        optstate_name = os.path.join(self.output_dir,
                                     f'{self.prefix}optstate.pkl')
        model = model.load_params_from_archive(tarfname, membername)
        with open(optstate_name, "rb") as optstate_file:
            opt = pickle.load(optstate_file)
        return model, opt


class ConfigDiskWriter(AbstractReporter):

    def __init__(self, output_dir, config, prefix=''):
        self.output_dir = output_dir
        self.config = config
        self.prefix = prefix

    def report_config(self):
        name = 'config.json' if self.prefix == '' else f'{self.prefix}_config.json'
        U.write_config(self.config, os.path.join(self.output_dir, name))


class OptunaReporter(AbstractReporter):

    def __init__(self, trial, objective):
        self.trial = trial

        if callable(objective):
            self.objective = objective
        elif isinstance(objective, list):
            self.objective = None
        else:
            raise RuntimeError('Unexpected objective passed')

    def report_params_size(self, size):
        self.trial.set_user_attr('parameters_size', size)

    def report_steps(self, steps):
        self.trial.set_user_attr('steps', steps)

    def report_timeout(self):
        self.trial.set_user_attr('timeout', 1)

    def report_progress(self, eval_step):
        self.trial.set_user_attr("progress", eval_step)
        study_attrs = self.trial.study.user_attrs

        if study_attrs['halt']:
            self.trial.set_user_attr('halted', 1)
            raise StudyHalted('Study is halted')

        if study_attrs['enable_prune'] and self.trial.should_prune():
            raise optuna.TrialPruned()

    def report_nan_detected(self):
        self.trial.set_user_attr('nan', 1)

    def report_evaluation(self, history):
        if self.objective:
            value = self.objective(history.validation_df())
            self.trial.report(value, len(history.validation_df()))
