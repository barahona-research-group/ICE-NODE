"""."""
import os
from typing import List, Any, Dict, Tuple
from datetime import datetime
from pathlib import Path

from absl import logging

from ...cli.cmd_args import get_cmd_parser

from ..utils import load_config

from ..ehr_model.dataset import datasets
from ..ehr_model.jax_interface import Subject_JAX

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
short_tags = {
    'matrix': 'M',
    'orthogonal_gram': 'O',
    'glove_gram': 'G',
    'semi_frozen_gram': 'S',
    'frozen_gram': 'F',
    'tuneble_gram': 'T',
    'NA': ''
}


def train_with_config(model: str,
                      config: Dict[str, Any],
                      subject_interface: str,
                      splits: Tuple[List[int]],
                      rng_seed: int,
                      trial_terminate_time=datetime.max,
                      reporters: List[AbstractReporter] = []):

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


if __name__ == '__main__':
    args = get_cmd_parser([
        '--model', '--config', '--config-tag', '--study-tag', '--dataset',
        '--output-dir'
    ]).parse_args()
    logging.set_verbosity(logging.INFO)
    logging.info('[LOADING] patient interface')
    dataset = datasets[args.dataset]
    subject_interface = Subject_JAX.from_dataset(dataset)
    logging.info('[DONE] patient interface')

    # splits = train:val:test = 0.7:.15:.15
    splits = subject_interface.random_splits(split1=0.7,
                                             split2=0.85,
                                             random_seed=42)

    expt_dir = f'{args.model}_expt'
    if args.config_tag:
        expt_dir = f'T{args.config_tag}_{expt_dir}'
    if args.study_tag:
        expt_dir = f'S{args.study_tag}_{expt_dir}'

    expt_dir = os.path.join(args.output_dir, expt_dir)
    Path(expt_dir).mkdir(parents=True, exist_ok=True)

    _reporters = [
        MinibatchLogger(),
        EvaluationDiskWriter(output_dir=expt_dir),
        ParamsDiskWriter(output_dir=expt_dir),
        ConfigDiskWriter(output_dir=expt_dir)
    ]
    train_with_config(model=args.model,
                      config=load_config(args.config),
                      subject_interface=subject_interface,
                      splits=splits,
                      rng_seed=42,
                      reporters=_reporters)
