import os
import argparse
import random
from pathlib import Path
from typing import Iterable, Dict, Any, Optional, Tuple
from functools import partial
from absl import logging
from tqdm import tqdm

import jax
from jax.experimental import optimizers

from .metrics import evaluation_table
from .utils import (load_config, tree_hasnan, write_config, write_params)
from .abstract_model import AbstractModel

from .train_gram import GRAM
from .train_retain import RETAIN
from .train_icenode_2lr import ICENODE as ICENODE_2LR
from .train_icenode_tl import ICENODE as ICENODE_TL


def run(model_cls: AbstractModel, config, patient_interface, tag: str,
        train_ids, test_ids, valid_ids, prng_key, output_dir):

    experiment_dir = os.path.join(output_dir, f'config_exp_{tag}')

    Path(experiment_dir).mkdir(parents=True, exist_ok=True)

    logging.info('[LOADING] Sampling & Initializing Models')

    write_config(config, os.path.join(experiment_dir, 'config.json'))

    logging.info(f'Tag {tag} HPs: {config}')

    model = model_cls.create_model(config, patient_interface, train_ids, None)

    code_partitions = model.code_partitions(patient_interface, train_ids)

    params = model.init_params(prng_key)
    logging.info('[DONE] Sampling & Initializing Models')

    loss_mixing = config['training']['loss_mixing']
    lr = config['training']['lr']
    if config['training']['optimizer'] == 'adam':
        optimizer = optimizers.adam
    else:
        optimizer = optimizers.sgd

    opt_init, opt_update, get_params = optimizer(step_size=lr)
    opt_state = opt_init(params)

    loss_ = partial(model.loss, loss_mixing)
    eval_ = partial(model.eval, loss_mixing)

    def update(
            step: int, batch: Iterable[int],
            opt_state: optimizers.OptimizerState) -> optimizers.OptimizerState:
        params = get_params(opt_state)
        grads = jax.grad(loss_)(params, batch)
        return opt_update(step, grads, opt_state)

    batch_size = config['training']['batch_size']
    batch_size = min(batch_size, len(train_ids))

    epochs = config['training']['epochs']
    iters = round(epochs * len(train_ids) / batch_size)
    for i in tqdm(range(iters)):
        rng.shuffle(train_ids)
        train_batch = train_ids[:batch_size]

        opt_state = update(i, train_batch, opt_state)
        if tree_hasnan(get_params(opt_state)):
            raise ValueError('NaN params')

        eval_step = round((i + 1) * 100 / iters)

        last_step = round(i * 100 / iters)

        if eval_step == last_step:
            continue

        params = get_params(opt_state)

        raw_res = {
            'TRN': eval_(params, train_batch),
            'VAL': eval_(params, valid_ids),
            'TST': eval_(params, test_ids)
        }

        eval_df, _ = evaluation_table(raw_res, code_partitions)
        eval_df.to_csv(
            os.path.join(experiment_dir, f'step{eval_step:04d}_eval.csv'))

        fname = os.path.join(experiment_dir,
                             f'step{eval_step:04d}_params.pickle')
        write_params(get_params(opt_state), fname)
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
        'icenode_tl': ICENODE_TL
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

    rng = random.Random(42)

    subjects_id = list(patient_interface.subjects.keys())
    rng.shuffle(subjects_id)

    # splits = train:val:test = 0.7:.15:.15
    splits = int(.7 * len(subjects_id)), int(.85 * len(subjects_id))

    train_ids = subjects_id[:splits[0]]
    valid_ids = subjects_id[splits[0]:splits[1]]
    test_ids = subjects_id[splits[1]:]

    prng_key = jax.random.PRNGKey(rng.randint(0, 100))

    config = load_config(args.config)
    run(model_cls,
        config=config,
        patient_interface=patient_interface,
        tag=args.config_tag,
        train_ids=train_ids,
        valid_ids=valid_ids,
        test_ids=test_ids,
        prng_key=prng_key,
        output_dir=args.output_dir)
