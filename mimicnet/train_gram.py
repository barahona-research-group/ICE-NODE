import pickle
from datetime import datetime
from functools import partial
from typing import (Any, Dict, Iterable, List, Optional)

from absl import logging
import pandas as pd
import haiku as hk
import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.tree_util import tree_map

from tqdm import tqdm

from .jax_interface import SubjectDiagSequenceJAXInterface
from .gram import DAGGRAM
from .metrics import (l2_squared, l1_absolute, evaluation_table, EvalFlag)
from .concept import Subject
from .dag import CCSDAG
from .utils import (load_config, wrap_module, parameters_size)


@jax.jit
def diag_loss(y: jnp.ndarray, diag_logits: jnp.ndarray):
    return -jnp.sum(y * jax.nn.log_softmax(diag_logits) +
                    (1 - y) * jnp.log(1 - jax.nn.softmax(diag_logits)))


class GRAM:
    def __init__(self, subject_interface: SubjectDiagSequenceJAXInterface,
                 diag_gram: DAGGRAM, state_size: int):

        self.subject_interface = subject_interface
        self.diag_gram = diag_gram

        self.dimensions = {
            'diag_gram': diag_gram.basic_embeddings_dim,
            'diag_dag': len(subject_interface.diag_multi_ccs_idx),
            'diag_out': len(subject_interface.diag_single_ccs_idx),
            'state': state_size
        }

        gru_init, gru = hk.without_apply_rng(
            hk.transform(
                wrap_module(hk.GRU, hidden_size=state_size, name='gru')))
        self.gru = jax.jit(gru)

        out_init, out = hk.without_apply_rng(
            hk.transform(
                wrap_module(hk.Linear,
                            output_size=self.dimensions['diag_out'],
                            name='out')))
        self.out = jax.jit(out)

        self.initializers = {'gru': gru_init, 'out': out_init}

    def init_params(self, rng_key):
        state = jnp.zeros(self.dimensions['state'])
        diag_gram = jnp.zeros(self.dimensions['diag_gram'])

        return {
            "diag_gram": self.diag_gram.init_params(rng_key),
            "gru": self.initializers['gru'](rng_key, diag_gram, state),
            "out": self.initializers['out'](rng_key, state)
        }

    def state_size(self):
        return self.dimensions['state']

    def diag_out_index(self) -> List[str]:
        index2code = {
            i: c
            for c, i in self.subject_interface.diag_single_ccs_idx.items()
        }
        return list(map(index2code.get, range(len(index2code))))

    def __call__(self, params: Any, subjects_batch: List[int]):

        G = self.diag_gram.compute_embedding_mat(params["diag_gram"])
        gram = partial(self.diag_gram.encode, G)
        diag_seqs = self.subject_interface.diag_sequences_batch(subjects_batch)

        loss = {}
        diag_detectability = {}
        state0 = jnp.zeros(self.dimensions['state'])
        for subject_id, _diag_seqs in diag_seqs.items():
            # Exclude last one for irrelevance
            hierarchical_diag = _diag_seqs['diag_multi_ccs_vec'][:-1]
            # Exclude first one, we need to predict them for a future step.
            flat_diag = _diag_seqs['diag_single_ccs_vec'][1:]
            gram_seqs = map(gram, hierarchical_diag)

            diag_detectability[subject_id] = {}
            loss[subject_id] = []
            state = state0
            for i, diag_gram in enumerate(gram_seqs):
                y_i = flat_diag[i]
                output, state = self.gru(params['gru'], diag_gram, state)
                logits = self.out(params['out'], output)
                diag_detectability[subject_id][i] = {
                    'diag_true': y_i,
                    'pre_logits': logits
                }
                loss[subject_id].append(diag_loss(y_i, logits))

        loss = [sum(l) / len(l) for l in loss.values()]

        return {
            'loss': sum(loss) / len(loss),
            'diag_detectability': diag_detectability
        }


def loss_fn(model: GRAM, loss_mixing: Dict[str, float],
            params: optimizers.Params, batch: List[int]) -> Dict[str, float]:
    res = model(params, batch)

    diag_loss_ = res['loss']
    l1_loss = l1_absolute(params)
    l2_loss = l2_squared(params)
    l1_alpha = loss_mixing['l1_reg']
    l2_alpha = loss_mixing['l2_reg']

    loss = diag_loss_ + (l1_alpha * l1_loss) + (l2_alpha * l2_loss)
    return loss


def eval_fn(model: GRAM, loss_mixing: Dict[str, float],
            params: optimizers.Params, batch: List[int]) -> Dict[str, float]:
    res = model(params, batch)

    diag_loss_ = res['loss']
    l1_loss = l1_absolute(params)
    l2_loss = l2_squared(params)
    l1_alpha = loss_mixing['l1_reg']
    l2_alpha = loss_mixing['l2_reg']

    loss = diag_loss_ + (l1_alpha * l1_loss) + (l2_alpha * l2_loss)
    return {
        'loss': {
            'diag_loss': diag_loss_,
            'loss': loss,
            'l1_loss': l1_loss,
            'l2_loss': l2_loss,
        },
        'diag_detectability': res['diag_detectability']
    }


def train_ehr(
        subject_interface: SubjectDiagSequenceJAXInterface,
        diag_gram: DAGGRAM,
        rng: Any,
        # Model configurations
        model_config: Dict[str, Any],
        # Training configurations
        train_validation_split: float,
        batch_size: int,
        epochs: int,
        lr: float,
        loss_mixing: Dict[str, float],
        eval_freq: int,
        save_freq: int,
        output_dir: str):

    prng_key = jax.random.PRNGKey(rng.randint(0, 100))

    model = GRAM(subject_interface=subject_interface,
                 diag_gram=diag_gram,
                 **model_config)

    loss = partial(loss_fn, model, loss_mixing)
    eval_ = partial(eval_fn, model, loss_mixing)

    params = model.init_params(prng_key)
    logging.info(f'#params: {parameters_size(params)}')
    logging.debug(f'shape(params): {tree_map(jnp.shape, params)}')

    opt_init, opt_update, get_params = optimizers.adam(step_size=lr)

    def update(
            step: int, batch: Iterable[int],
            opt_state: optimizers.OptimizerState) -> optimizers.OptimizerState:
        params = get_params(opt_state)
        grads = jax.grad(loss)(params, batch)
        return opt_update(step, grads, opt_state)

    opt_state = opt_init(params)

    subjects_id = list(subject_interface.subjects.keys())
    rng.shuffle(subjects_id)

    train_ids = subjects_id[:int(train_validation_split * len(subjects_id))]
    valid_ids = subjects_id[int(train_validation_split * len(subjects_id)):]
    batch_size = min(batch_size, len(train_ids))
    val_batch_size = min(batch_size, len(valid_ids))

    codes_by_percentiles = subject_interface.diag_single_ccs_by_percentiles(
        20, train_ids)

    res_val = {}
    res_trn = {}
    if save_freq is None:
        save_freq = eval_freq

    iters = int(epochs * len(train_ids) / batch_size)
    val_pbar = tqdm(total=iters)

    for step in range(iters):
        rng.shuffle(train_ids)
        train_batch = train_ids[:batch_size]
        val_pbar.update(1)

        try:
            opt_state = update(step, train_batch, opt_state)
        except (ValueError, FloatingPointError) as e:
            from traceback import format_exception
            tb_str = ''.join(format_exception(None, e, e.__traceback__))
            logging.warning(f'ValueError exception raised: {tb_str}')
            break

        if step % save_freq == 0 and step > 0:
            with open(f'{output_dir}/step{step:03d}.pickle', 'wb') as f:
                pickle.dump(get_params(opt_state), f)

        if step % eval_freq != 0: continue

        rng.shuffle(valid_ids)
        valid_batch = valid_ids  #[:val_batch_size]
        params = get_params(opt_state)

        trn_res = eval_(params, train_batch)
        res_trn[step] = trn_res

        val_res = eval_(params, valid_batch)
        res_val[step] = val_res

        eval_table = evaluation_table(trn_res, val_res, EvalFlag.CM,
                                      codes_by_percentiles)
        logging.info('\n' + str(eval_table))

    return {
        'res_val': res_val,
        'res_trn': res_trn,
        'model_params': get_params(opt_state),
        'model': model,
        'trn_ids': train_ids,
        'val_ids': valid_ids
    }


def create_patient_interface(processed_mimic_tables_dir: str):
    static_df = pd.read_csv(f'{processed_mimic_tables_dir}/static_df.csv.gz')
    adm_df = pd.read_csv(f'{processed_mimic_tables_dir}/adm_df.csv.gz')
    diag_df = pd.read_csv(f'{processed_mimic_tables_dir}/diag_df.csv.gz',
                          dtype={'ICD9_CODE': str})
    proc_df = pd.read_csv(f'{processed_mimic_tables_dir}/proc_df.csv.gz',
                          dtype={'ICD9_CODE': str})
    # Cast columns of dates to datetime64
    static_df['DOB'] = pd.to_datetime(
        static_df.DOB, infer_datetime_format=True).dt.normalize()
    adm_df['ADMITTIME'] = pd.to_datetime(
        adm_df.ADMITTIME, infer_datetime_format=True).dt.normalize()
    adm_df['DISCHTIME'] = pd.to_datetime(
        adm_df.DISCHTIME, infer_datetime_format=True).dt.normalize()

    patients = Subject.to_list(static_df, adm_df, diag_df, proc_df, None)

    # CCS Knowledge Graph
    k_graph = CCSDAG()

    return SubjectDiagSequenceJAXInterface(patients, set(), k_graph)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)

    from pathlib import Path
    import random
    import argparse
    from .glove import glove_representation

    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--mimic-processed-dir',
                        required=True,
                        help='Absolute path to MIMIC-III processed tables')
    parser.add_argument('-o',
                        '--output-dir',
                        required=True,
                        help='Aboslute path to log intermediate results')

    parser.add_argument(
        '-c',
        '--config',
        required=True,
        help='Absolute path to JSON file of experiment configurations')

    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    mimic_processed_dir = args.mimic_processed_dir
    output_dir = args.output_dir
    cpu = args.cpu
    config_file = args.config

    if cpu:
        jax.config.update('jax_platform_name', 'cpu')
    else:
        jax.config.update('jax_platform_name', 'gpu')

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logging.info('[LOADING] Patients JAX Interface.')
    patient_interface = create_patient_interface(mimic_processed_dir)
    logging.info('[DONE] Patients JAX Interface')

    config = load_config(config_file)

    diag_glove, _ = glove_representation(
        diag_idx=patient_interface.diag_multi_ccs_idx,
        proc_idx=patient_interface.proc_multi_ccs_idx,
        ccs_dag=patient_interface.dag,
        subjects=patient_interface.subjects.values(),
        **config['glove_config'])

    diag_gram = DAGGRAM(
        ccs_dag=patient_interface.dag,
        code2index=patient_interface.diag_multi_ccs_idx,
        basic_embeddings=diag_glove,
        ancestors_mat=patient_interface.diag_multi_ccs_ancestors_mat,
        **config['gram_config']['diag'])

    res = train_ehr(subject_interface=patient_interface,
                    diag_gram=diag_gram,
                    rng=random.Random(42),
                    model_config=config['model'],
                    **config['training'],
                    output_dir=output_dir,
                    train_validation_split=0.8,
                    eval_freq=10,
                    save_freq=100)
