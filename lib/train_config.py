import os
import argparse
import random
from pathlib import Path
from typing import Iterable, Dict, Any
from absl import logging
from tqdm import tqdm
import jax

from .metrics import evaluation_table
from .utils import (load_config, write_config)
from .abstract_model import AbstractModel

from .train_gram import GRAM
from .train_retain import RETAIN
from .train_icenode_2lr import ICENODE as ICENODE_2LR
from .train_icenode_tl import ICENODE as ICENODE_TL
from .train_icenode_uniform2lr import ICENODE as ICENODE_UNIFORM2LR
from .train_icenode_dtw2lr import ICENODE as ICENODE_DTW2LR
from .train_icenode_dtw import ICENODE as ICENODE_DTW


def run(model_cls: AbstractModel, config, patient_interface, tag: str,
        train_ids, test_ids, valid_ids, rng_seed, output_dir):

    prng_key = jax.random.PRNGKey(rng_seed)
    rng = random.Random(rng_seed)

    experiment_dir = os.path.join(output_dir, f'config_exp_{tag}')

    Path(experiment_dir).mkdir(parents=True, exist_ok=True)

    logging.info('[LOADING] Sampling & Initializing Models')

    write_config(config, os.path.join(experiment_dir, 'config.json'))

    logging.info(f'Tag {tag} HPs: {config}')

    model = model_cls.create_model(config, patient_interface, train_ids, None)

    code_partitions = model.code_partitions(patient_interface, train_ids)

    m_state = model.init(config)
    logging.info('[DONE] Sampling & Initializing Models')

    batch_size = config['training']['batch_size']
    batch_size = min(batch_size, len(train_ids))
    epochs = config['training']['epochs']
    iters = round(epochs * len(train_ids) / batch_size)
    for i in tqdm(range(iters)):
        eval_step = round((i + 1) * 100 / iters)
        last_step = round(i * 100 / iters)

        rng.shuffle(train_ids)
        train_batch = train_ids[:batch_size]

        m_state = model.step_optimizer(eval_step, m_state, train_batch)
        if model.hasnan(m_state):
            raise ValueError('NaN params')

        if eval_step == last_step and i < iters - 1:
            continue

        raw_res = {
            'TRN': model.eval(m_state, train_batch),
            'VAL': model.eval(m_state, valid_ids),
            'TST': model.eval(m_state, test_ids)
        }

        eval_df, _ = evaluation_table(raw_res, code_partitions)
        eval_df.to_csv(
            os.path.join(experiment_dir, f'step{eval_step:04d}_eval.csv'))

        fname = os.path.join(experiment_dir,
                             f'step{eval_step:04d}_params.pickle')
        model.write_params(m_state, fname)
        logging.info(eval_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, help='Model label')

    parser.add_argument('-c',
                        '--config',
                        required=True,
                        help='Model configuration JSON file')

    parser.add_argument('-i',
                        '--mimic-processed-dir',
                        required=True,
                        help='Absolute path to MIMIC-III processed tables')

    parser.add_argument('-t',
                        '--config-tag',
                        required=True,
                        help='Experiment tag')
    parser.add_argument('-d', '--data-tag', required=True, help='Data tag')

    parser.add_argument('-o',
                        '--output-dir',
                        required=True,
                        help='Aboslute path to log intermediate results')

    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    data_tag_fullname = {'M3': 'MIMIC-III', 'M4': 'MIMIC-IV'}
    model_class = {
        'gram': GRAM,
        'retain': RETAIN,
        'icenode_2lr': ICENODE_2LR,
        'icenode_tl': ICENODE_TL,
        'icenode_dtw': ICENODE_DTW,
        'icenode_dtw2lr': ICENODE_DTW2LR,
        'icenode_uniform2lr': ICENODE_UNIFORM2LR
    }
    model_cls = model_class[args.model]

    logging.set_verbosity(logging.INFO)
    logging.info('[LOADING] patient interface')
    patient_interface = model_cls.create_patient_interface(
        args.mimic_processed_dir, data_tag=args.data_tag)
    logging.info('[DONE] patient interface')

    if args.cpu:
        jax.config.update('jax_platform_name', 'cpu')
    else:
        jax.config.update('jax_platform_name', 'gpu')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # splits = train:val:test = 0.7:.15:.15
    train_ids, valid_ids, test_ids = patient_interface.random_splits(
        split1=0.7, split2=0.85, random_seed=42)

    config = load_config(args.config)
    run(model_cls,
        config=config,
        patient_interface=patient_interface,
        tag=args.config_tag,
        train_ids=train_ids,
        valid_ids=valid_ids,
        test_ids=test_ids,
        rng_seed=42,
        output_dir=args.output_dir)
