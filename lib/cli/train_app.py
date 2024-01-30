"""."""
import os
from typing import List, Any, Dict, Tuple
from datetime import datetime
from pathlib import Path

from absl import logging
import jax
import jax.random as jrandom

from .cmd_args import get_cmd_parser
from ..ehr.coding_scheme import DxCCS, DxFlatCCS, DxICD9, DxICD10
from ..ehr import Subject_JAX, OutcomeExtractor, SurvivalOutcomeExtractor
from ..ehr import StaticInfoFlags
from .. import utils
from .. import ehr
from .. import ml
from ..metric import (CodeAUC, UntilFirstCodeAUC, AdmissionAUC,
                      CodeGroupTopAlarmAccuracy, LossMetric, MetricsCollection)

cli_args = [
    '--model', '--config', '--config-tag', '--study-tag', '--dataset',
    '--output-dir', '--device', '--prediction-task', '--dx_discharge-scheme',
    '--dx_discharge-outcome', '--static-features'
]

dx_scheme = {
    'dx_ccs': DxCCS,
    'dx_flat_ccs': DxFlatCCS,
    'dx_icd9': DxICD9,
    'dx_icd10': DxICD10
}

task = {'first': SurvivalOutcomeExtractor, 'all': OutcomeExtractor}

model_cls = {
    'AICE': ml.AICE,
    'ICE-NODE': ml.ICENODE,
    'ICE-NODE_UNIFORM': ml.ICENODE_UNIFORM,
    'GRU': ml.GRU,
    'RETAIN': ml.RETAIN,
    'LogReg': ml.WindowLogReg,
    'NJODE': ml.NJODE
}


def train_with_config(model: str,
                      config: Dict[str, Any],
                      subject_interface: str,
                      splits: Tuple[List[int]],
                      rng_seed: int,
                      trial_terminate_time=datetime.max,
                      reporters: List[ml.AbstractReporter] = []):

    logging.info('[LOADING] Initialize models')
    key = jrandom.PRNGKey(0)
    model = model_cls[model].from_config(config, interface, splits[0], key)
    trainer_cls = getattr(ml, config["training"]["classname"])
    trainer = trainer_cls(**config["training"])
    logging.info('[DONE] Initialize models')
    # pecentile_range=20 will partition the codes into five gruops, where each group contains
    # codes that overall constitutes 20% of the codes in all visits of specified 'subjects' list.
    code_freq_partitions = interface.outcome_by_percentiles(
        percentile_range=20, subjects=splits[0])

    # Evaluate for different k values
    top_k_list = [3, 5, 10, 15, 20]

    metrics = [
        CodeAUC(interface),
        UntilFirstCodeAUC(interface),
        AdmissionAUC(interface),
        LossMetric(interface),
        CodeGroupTopAlarmAccuracy(interface,
                                  top_k_list=top_k_list,
                                  code_groups=code_freq_partitions)
    ]

    metrics = MetricsCollection(metrics)
    history = ml.MetricsHistory(metrics)

    return trainer(model,
                   interface,
                   splits,
                   history=history,
                   reporters=reporters,
                   prng_seed=42)


if __name__ == '__main__':
    args = get_cmd_parser(cli_args).parse_args()
    jax.config.update('jax_platform_name', args.device)

    config = utils.load_config(args.config)
    logging.set_verbosity(logging.INFO)
    logging.info('[LOADING] patient interface')
    dataset = ehr.dataset.load_dataset(args.dataset)
    scheme = {
        'dx_discharge': dx_scheme[args.dx_scheme](),
        'outcome': task[args.prediction_task](args.dx_outcome)
    }
    static_features_dict = {}
    if args.static_features:
        static_features_dict = {
            f: True
            for f in args.static_features.split(',')
        }
    static_features = StaticInfoFlags(**static_features_dict)
    interface = Subject_JAX.from_dataset(dataset,
                                         code_scheme=scheme,
                                         static_info_flags=static_features,
                                         data_max_size_gb=5)
    logging.info('[DONE] patient interface')

    # splits = train:val:test = 0.7:.15:.15
    splits = interface.random_splits(split1=0.7, split2=0.85, random_seed=42)

    expt_dir = f'{args.model}'
    if args.config_tag:
        expt_dir = f'T{args.config_tag}_{expt_dir}'
    if args.study_tag:
        expt_dir = f'S{args.study_tag}_{expt_dir}'

    output_dir = utils.translate_path(args.output_dir)
    expt_dir = os.path.join(output_dir, expt_dir)
    Path(expt_dir).mkdir(parents=True, exist_ok=True)

    _reporters = [
        ml.MinibatchLogger(config),
        ml.EvaluationDiskWriter(output_dir=expt_dir),
        ml.ParamsDiskWriter(output_dir=expt_dir),
        ml.ConfigDiskWriter(output_dir=expt_dir, config=config)
    ]
    train_with_config(model=args.model,
                      config=config,
                      subject_interface=interface,
                      splits=splits,
                      rng_seed=42,
                      reporters=_reporters)
