"""."""

import argparse
from typing import List, Any, Dict, Tuple
from datetime import datetime
from pathlib import Path

from absl import logging

from ..ehr_model.jax_interface import create_patient_interface
from ..utils import load_config

from .trainer import (AbstractReporter, MinibatchLogger, ConfigDiskWriter,
                      ParamsDiskWriter, EvaluationDiskWriter)
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
        help='Embedding method to use: matrix, orthogonal_gram, glove_gram, \
        semi_frozen_gram, frozen_gram, tunable_gram, NA.')

    parser.add_argument('-o',
                        '--output-dir',
                        required=True,
                        help='Aboslute path to log intermediate results')

    return parser.parse_args()


if __name__ == '__main__':
    args = capture_args()
    logging.set_verbosity(logging.INFO)
    logging.info('[LOADING] patient interface')
    _subject_interface = create_patient_interface(args.mimic_processed_dir)
    logging.info('[DONE] patient interface')

    # splits = train:val:test = 0.7:.15:.15
    _splits = _subject_interface.random_splits(split1=0.7,
                                               split2=0.85,
                                               random_seed=42)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    _reporters = [
        MinibatchLogger(),
        EvaluationDiskWriter(output_dir=args.output_dir),
        ParamsDiskWriter(output_dir=args.output_dir),
        ConfigDiskWriter(output_dir=args.output_dir)
    ]
    train_with_config(model=args.model,
                      config=load_config(args.config),
                      subject_interface=_subject_interface,
                      splits=_splits,
                      rng_seed=42,
                      output_dir=args.output_dir,
                      reporters=_reporters)
