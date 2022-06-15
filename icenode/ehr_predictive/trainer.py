import os
import copy
import random
import argparse
from typing import List, Any, Dict, Tuple
from datetime import datetime
from pathlib import Path

from absl import logging
from tqdm import tqdm

from ..ehr_model.jax_interface import create_patient_interface
from ..metric.common_metrics import evaluation_table
from ..utils import write_config, load_config

from .dx_gram import GRAM
from .dx_icenode_2lr import ICENODE
from .dx_icenode_uniform2lr import ICENODE as ICENODE_UNIFORM
from .dx_retain import RETAIN
from .dx_window_logreg import WindowLogReg as WLR

model_cls = {
    'dx_gram': GRAM,
    'dx_icenode': ICENODE,
    'dx_icenode_uniform': ICENODE_UNIFORM,
    'dx_retain': RETAIN,
    'dx_window_logreg': WLR
}


class AbstractReporter:
    """
    Different loggers and reporters:
        1. Optuna reporter
        2. MLFlow reporter
        3. logging
        4. evaluation disk writer
        5. parameters disk writer
    """

    def report_config(self, config):
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

    def report_evaluation(self, eval_step, objective_v, evals_df,
                          flat_evals_df):
        pass

    def report_params(self, eval_step, model, state, last_iter, current_best):
        pass


class MinibatchLogger(AbstractReporter):

    def report_config(self, config):
        logging.info(f'HPs: {config}')

    def report_nan_detected(self):
        logging.warning('NaN detected')

    def report_evaluation(self, eval_step, objective_v, evals_df,
                          flat_evals_df):
        logging.info(evals_df)


class EvaluationDiskWriter(AbstractReporter):

    def __init__(self, output_dir):
        self.output_dir = output_dir

    def report_evaluation(self, eval_step, objective_v, evals_df,
                          flat_evals_df):
        evals_df.to_csv(
            os.path.join(self.output_dir, f'step{eval_step:04d}_eval.csv'))


class ParamsDiskWriter(AbstractReporter):

    def __init__(self, output_dir, write_every_iter=False):
        self.output_dir = output_dir
        self.write_every_iter = write_every_iter

    def report_params(self, eval_step, model, state, last_iter, current_best):
        if self.write_every_iter or last_iter or current_best:
            fname = os.path.join(self.output_dir,
                                 f'step{eval_step:04d}_params.pickle')
            model.write_params(state, fname)


class ConfigDiskWriter(AbstractReporter):

    def __init__(self, output_dir):
        self.output_dir = output_dir

    def report_config(self, config):
        write_config(config, os.path.join(self.output_dir, 'config.json'))


def minibatch_trainer(model,
                      m_state,
                      config,
                      splits,
                      rng_seed,
                      code_frequency_groups=None,
                      trial_terminate_time=datetime.max,
                      reporters: List[AbstractReporter] = []):
    rng = random.Random(rng_seed)
    train_ids, valid_ids, test_ids = splits
    # Because shuffling is done in-place.
    train_ids = copy.deepcopy(train_ids)

    batch_size = config['training']['batch_size']
    batch_size = min(batch_size, len(train_ids))

    epochs = config['training']['epochs']
    iters = round(epochs * len(train_ids) / batch_size)

    for r in reporters:
        r.report_config(config)
        r.report_params_size(model.parameters_size(m_state))
        r.report_steps(iters)

    auc = 0.0
    best_score = 0.0
    for i in tqdm(range(iters)):
        eval_step = round((i + 1) * 100 / iters)
        last_step = round(i * 100 / iters)

        if datetime.now() > trial_terminate_time:
            [r.report_timeout() for r in reporters]
            break

        rng.shuffle(train_ids)
        train_batch = train_ids[:batch_size]

        m_state = model.step_optimizer(eval_step, m_state, train_batch)
        if model.hasnan(m_state):
            [r.report_nan_detected() for r in reporters]

        if eval_step == last_step and i < iters - 1:
            continue

        [r.report_progress(eval_step) for r in reporters]

        if i == iters - 1:
            raw_res = {
                'TRN': model.eval(m_state, train_batch),
                'VAL': model.eval(m_state, valid_ids),
                'TST': model.eval(m_state, test_ids)
            }
        else:
            raw_res = {
                'TRN': model.eval(m_state, train_batch),
                'VAL': model.eval(m_state, valid_ids)
            }

        eval_df, eval_flat = evaluation_table(raw_res, code_frequency_groups)

        auc = eval_df.loc['MICRO-AUC', 'VAL']

        for r in reporters:
            r.report_evaluation(eval_step, auc, eval_df, eval_flat)
            r.report_params(eval_step,
                            model,
                            m_state,
                            last_iter=i == iters - 1,
                            current_best=auc > best_score)

        if auc > best_score:
            best_score = auc

    return {'objective': auc, 'model': (model, m_state)}


def sklearn_trainer(model,
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

    eval_df, eval_flat = evaluation_table(raw_res, code_frequency_groups)

    auc = eval_df.loc['MICRO-AUC', 'VAL']

    for r in reporters:
        r.report_evaluation(100, auc, eval_df, eval_flat)
        r.report_params(100, model, m_state, last_iter=True, current_best=True)

    return {'objective': auc, 'model': (model, m_state)}


def train_with_config(model: str,
                      config: Dict[str, Any],
                      subject_interface: str,
                      splits: Tuple[List[int]],
                      rng_seed: int,
                      output_dir: str,
                      trial_terminate_time=datetime.max,
                      reporters: List[AbstractReporter] = []):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logging.info('[LOADING] Initialize models')
    model = model_cls[model].create_model(config, subject_interface, splits[0])
    m_state = model.init(config, rng_seed)
    logging.info('[DONE] Initialize models')

    code_frequency_groups = model.code_partitions(subject_interface, splits[0])
    return model.get_trainer()(model=model,
                               m_state=m_state,
                               config=config,
                               splits=splits,
                               rng_seed=rng_seed,
                               code_frequency_groups=code_frequency_groups,
                               trial_terminate_time=trial_terminate_time,
                               reporters=reporters)['objective']


def capture_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, help='Model label')
    parser.add_argument('-c',
                        '--config',
                        required=True,
                        help='Path to config JSON file')

    parser.add_argument('-i',
                        '--mimic-processed-dir',
                        required=True,
                        help='Absolute path to MIMIC-III processed tables')

    parser.add_argument(
        '-e',
        '--emb',
        required=True,
        help=
        'Embedding method to use (matrix|orthogonal_gram|glove_gram|semi_frozen_gram|frozen_gram|tunable_gram|NA)'
    )

    parser.add_argument('-o',
                        '--output-dir',
                        required=True,
                        help='Aboslute path to log intermediate results')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = capture_args()
    logging.set_verbosity(logging.INFO)
    logging.info('[LOADING] patient interface')
    subject_interface = create_patient_interface(args.mimic_processed_dir)
    logging.info('[DONE] patient interface')

    # splits = train:val:test = 0.7:.15:.15
    splits = subject_interface.random_splits(split1=0.7,
                                             split2=0.85,
                                             random_seed=42)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)

    reporters = [
        MinibatchLogger(),
        EvaluationDiskWriter(output_dir=args.output_dir),
        ParamsDiskWriter(output_dir=args.output_dir),
        ConfigDiskWriter(output_dir=args.output_dir)
    ]
    train_with_config(model=args.model,
                      config=config,
                      subject_interface=subject_interface,
                      splits=splits,
                      rng_seed=42,
                      output_dir=args.output_dir,
                      reporters=reporters)
