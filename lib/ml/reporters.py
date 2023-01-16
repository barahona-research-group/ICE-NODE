"""."""

import os
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

    def report_evaluation(self, **kwargs):
        pass

    def report_plots(self, **kwargs):
        pass

    def report_params(self, eval_step, model):
        pass


class MinibatchLogger(AbstractReporter):

    def __init__(self, config):
        self.config = config

    def report_config(self):
        logging.info(f'HPs: {self.config}')

    def report_nan_detected(self, msg = None):
        logging.warning(msg or 'NaN detected')

    def report_evaluation(self, evals_df, **kwargs):
        logging.info(evals_df)


class Plotter(AbstractReporter):

    def __init__(self, output_dir, prefix=''):
        self.output_dir = output_dir
        self.prefix = prefix

class EvaluationDiskWriter(AbstractReporter):

    def __init__(self, output_dir, prefix=''):
        self.output_dir = output_dir
        self.prefix = prefix

    def report_evaluation(self, eval_step, evals_df, **kwargs):
        name = 'eval.csv.gz' if self.prefix == '' else f'{self.prefix}_eval.csv.gz'
        evals_df.to_csv(os.path.join(self.output_dir, name),
                        compression="gzip")


class ParamsDiskWriter(AbstractReporter):

    def __init__(self, output_dir, prefix=''):
        self.output_dir = output_dir
        self.prefix = prefix

    def report_params(self, eval_step, model):
        tarname = os.path.join(self.output_dir, f'{self.prefix}params.tar.bz2')
        name = f'step{eval_step:04d}.eqx'
        U.append_params_to_zip(model, name, tarname)


class ConfigDiskWriter(AbstractReporter):

    def __init__(self, output_dir, config, prefix=''):
        self.output_dir = output_dir
        self.config = config
        self.prefix = prefix

    def report_config(self):
        name = 'config.json' if self.prefix == '' else f'{self.prefix}_config.json'
        U.write_config(self.config, os.path.join(self.output_dir, name))


class OptunaReporter(AbstractReporter):

    def __init__(self, trial):
        self.trial = trial

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

    def report_evaluation(self, eval_step, objective_v, **kwargs):
        self.trial.report(objective_v, eval_step)