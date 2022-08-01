"""."""
import os
from typing import List, Any, Dict, Tuple
from datetime import datetime
from pathlib import Path

from absl import logging

from cli.cmd_args import get_cmd_parser

from .. import utils
from .. import ehr
from .. import ml

cli_args = [
    '--model', '--config', '--config-tag', '--study-tag', '--dataset',
    '--output-dir'
]


def train_with_config(model: str,
                      config: Dict[str, Any],
                      subject_interface: str,
                      splits: Tuple[List[int]],
                      rng_seed: int,
                      trial_terminate_time=datetime.max,
                      reporters: List[ml.AbstractReporter] = []):

    logging.info('[LOADING] Initialize models')
    model_cls = ml.AbstractModel.model_cls[model]
    model = model_cls.create_model(config, subject_interface, splits[0])
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
    args = get_cmd_parser(cli_args).parse_args()
    config = utils.load_config(args.config)
    logging.set_verbosity(logging.INFO)
    logging.info('[LOADING] patient interface')
    dataset = ehr.datasets[args.dataset]
    subject_interface = ehr.Subject_JAX.from_dataset(dataset,
                                                     config['code_scheme'])
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

    output_dir = utils.translate_path(args.output_dir)
    expt_dir = os.path.join(output_dir, expt_dir)
    Path(expt_dir).mkdir(parents=True, exist_ok=True)

    _reporters = [
        ml.MinibatchLogger(),
        ml.EvaluationDiskWriter(output_dir=expt_dir),
        ml.ParamsDiskWriter(output_dir=expt_dir),
        ml.ConfigDiskWriter(output_dir=expt_dir)
    ]
    train_with_config(model=args.model,
                      config=config,
                      subject_interface=subject_interface,
                      splits=splits,
                      rng_seed=42,
                      reporters=_reporters)
