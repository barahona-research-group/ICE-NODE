import logging
import pickle
from collections import defaultdict
from datetime import datetime
from functools import partial
from typing import (AbstractSet, Any, Callable, Dict, Iterable, List, Mapping,
                    Optional, Tuple, Union, Set)

import numpy as onp
import pandas as pd
import haiku as hk
import jax
import jax.numpy as jnp
from jax import lax
from jax.profiler import annotate_function
from jax.experimental import optimizers
from jax.nn import softplus, sigmoid, leaky_relu

from tqdm import tqdm
from jax.tree_util import tree_flatten, tree_map

from .jax_interface import SubjectDiagSequenceJAXInterface
from .gram import DAGGRAM
from .metrics import (l2_squared, l1_absolute, confusion_matrix,
                      confusion_matrix_scores)

ode_logger = logging.getLogger("ode")
debug_flags = {'nan_debug': True, 'shape_debug': True}


def tree_lognan(t):
    return jax.tree_map(lambda x: jnp.any(jnp.isnan(x)).item(), t)


@jax.jit
def diag_loss(y: jnp.ndarray, diag_logits: jnp.ndarray):
    return -jnp.sum(y * jax.nn.log_softmax(diag_logits) +
                    (1 - y) * jnp.log(1 - jax.nn.softmax(diag_logits)))


def code_detectability(top_k: int, true_diag: jnp.ndarray,
                       predicted_diag: jnp.ndarray):
    ground_truth = jnp.argwhere(true_diag).squeeze()
    if ground_truth.ndim > 0:
        ground_truth = set(onp.array(ground_truth))
    else:
        ground_truth = {ground_truth.item()}

    predictions = set(onp.array(jnp.argsort(predicted_diag)[-top_k:]))
    detections = []
    for code_i in ground_truth:
        detected = 0
        if code_i in predictions:
            detected = 1
        detections.append((code_i, detected))

    return detections


def code_detectability_df(top_k: int, res):
    df_list = []
    for subject_id, _res in res.items():
        for point_n, __res in _res.items():
            for code_i, detected in code_detectability(top_k,
                                                       __res['diag_true'],
                                                       __res['logits']):
                df_list.append((subject_id, point_n, code_i, detected, top_k))

    return pd.DataFrame(
        df_list,
        columns=['subject_id', 'point_n', 'code', 'detected', 'top_k'])


def code_detectability_by_percentiles(codes_by_percentiles, detections_df):
    rate = {}
    for i, codes in enumerate(codes_by_percentiles):
        codes_detections_df = detections_df[detections_df.code.isin(codes)]
        detection_rate = codes_detections_df.detected.mean()
        C = len(codes)
        N = len(codes_detections_df)
        rate[f'P{i}(N={N} C={len(codes)})'] = detection_rate
    return rate


def parameters_size(pytree):
    leaves, _ = tree_flatten(pytree)
    return sum(jnp.size(x) for x in leaves)


def tree_hasnan(t):
    return any(map(lambda x: jnp.any(jnp.isnan(x)), jax.tree_leaves(t)))


def array_hasnan(arr):
    return jnp.any(jnp.isnan(arr) | jnp.isinf(arr))


def wrap_module(module, *module_args, **module_kwargs):
    """
    Wrap the module in a function to be transformed.
    """
    def wrap(*args, **kwargs):
        """
        Wrapping of module.
        """
        model = module(*module_args, **module_kwargs)
        return model(*args, **kwargs)

    return wrap


class PatientGRAMInterface:
    def __init__(self, subject_interface: SubjectDiagSequenceJAXInterface,
                 diag_gram: DAGGRAM, state_size: int,
                 diag_loss: Callable[[jnp.ndarray, jnp.ndarray], float]):

        self.subject_interface = subject_interface
        self.diag_gram = diag_gram
        self.diag_loss = diag_loss

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
            "gru": self.gru(diag_gram, state),
            "out": self.out(state)
        }

    def state_size(self):
        return self.dimensions['state']

    def diag_out_index(self) -> List[str]:
        index2code = {
            i: c
            for c, i in self.subject_interface.diag_single_ccs_idx.items()
        }
        return list(map(index2code.get, range(len(index2code))))

    def __diag_loss(self, diag_true: Dict[int, jnp.ndarray],
                    diag_predicted: Dict[int, jnp.ndarray]):
        loss = {
            i: self.diag_loss(diag_true[i], diag_predicted[i])
            for i in diag_predicted.keys()
        }
        if loss:
            return sum(loss.values()) / len(loss)
        else:
            return 0.0

    def __call__(self, params: Any, subjects_batch: List[int]):

        G = self.diag_gram.compute_embedding_mat(params["diag_gram"])
        gram = partial(self.diag_gram.encode, G)
        diag_seqs = self.subject_interface.diag_sequences_batch(subjects_batch)
        state0 = self.gru.initial_state()

        logits = {}
        loss = {}
        detectability = {}
        for subject_id, _diag_seqs in diag_seqs.items():
            # Exclude last one for irrelevance
            hierarchical_diag = _diag_seqs['diag_multi_ccs_vec'][:-1]
            # Exclude first one, we need to predict them for a future step.
            flat_diag = _diag_seqs['diag_single_ccs_vec'][1:]
            gram_seqs = map(diag_seqs, hierarchical_diag)
            state = state0

            detectability[subject_id] = {}
            loss[subject_id] = []
            for i, diag_gram in enumerate(gram_seqs):
                y_i = flat_diag[i]
                output, state = self.gru(diag_gram, state)
                logits = self.out(output)
                detectability[subject_id][i] = {
                    'diag_true': y_i,
                    'logits': logits
                }
                loss[subject_id].append(diag_loss(y_i, logits))

        loss = [sum(l) / len(l) for l in loss.values()]

        return {
            'loss': sum(loss) / len(loss),
            'diag_detectability_df': code_detectability_df(20, detectability)
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
        save_freq: Optional[int],
        save_params_prefix: Optional[str]):

    prng_key = jax.random.PRNGKey(rng.randint(0, 100))

    ode_model = PatientGRAMInterface(subject_interface=subject_interface,
                                     diag_gram=diag_gram,
                                     **model_config)

    params = ode_model.init_params(prng_key)
    ode_logger.info(f'#params: {parameters_size(params)}')
    ode_logger.debug(f'shape(params): {tree_map(jnp.shape, params)}')

    opt_init, opt_update, get_params = optimizers.adam(step_size=lr)

    def loss_fn(params: optimizers.Params,
                batch: List[int]) -> Dict[str, float]:
        res = ode_model(params, batch)

        diag_loss = res['loss']
        l1_loss = l1_absolute(params)
        l2_loss = l2_squared(params)
        l1_alpha = loss_mixing['l1_reg']
        l2_alpha = loss_mixing['l2_reg']

        loss = diag_loss + (l1_alpha * l1_loss) + (l2_alpha * l2_loss)
        return loss, {
            'loss': {
                'diag_loss': diag_loss,
                'loss': loss,
                'l1_loss': l1_loss,
                'l2_loss': l2_loss,
            },
            'diag_detectability_df': res['diag_detectability_df']
        }

    def update(
            step: int, batch: Iterable[int],
            opt_state: optimizers.OptimizerState) -> optimizers.OptimizerState:
        params = get_params(opt_state)
        """Single SGD update step."""
        if tree_hasnan(params):
            ode_logger.warning(tree_lognan(params))
            raise ValueError("Nan Params")

        grads, data = jax.grad(loss_fn, has_aux=True)(params, batch)
        if tree_hasnan(grads):
            ode_logger.warning(tree_lognan(grads))
            raise ValueError("Nan Grads")

        return opt_update(step, grads, opt_state), data

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

    if save_params_prefix is None:
        timestamp = datetime.now().strftime("%d-%b-%Y_%H-%M-%S")
        save_params_prefix = f'GRU_ODE_Bayes_B{batch_size}_{timestamp}'

    iters = int(epochs * len(train_ids) / batch_size)
    val_pbar = tqdm(total=iters)

    for step in range(iters):
        rng.shuffle(train_ids)
        train_batch = train_ids[:batch_size]
        val_pbar.update(1)

        try:
            opt_state, _ = update(step, train_batch, opt_state)
        except (ValueError, FloatingPointError) as e:
            from traceback import format_exception
            tb_str = ''.join(format_exception(None, e, e.__traceback__))
            ode_logger.warning(f'ValueError exception raised: {tb_str}')
            break

        if step % eval_freq == 0:
            rng.shuffle(valid_ids)
            valid_batch = valid_ids  #[:val_batch_size]
            params = get_params(opt_state)

            trn_loss, trn_res = loss_fn(params, train_batch)
            res_trn[step] = trn_res

            val_loss, val_res = loss_fn(params, valid_batch)
            res_val[step] = val_res

            losses = pd.DataFrame(index=trn_res['loss'].keys(),
                                  data={
                                      'Training': trn_res['loss'].values(),
                                      'Validation': val_res['loss'].values()
                                  })
            stats = pd.DataFrame(index=trn_res['stats'].keys(),
                                 data={
                                     'Training': trn_res['stats'].values(),
                                     'Valdation': val_res['stats'].values()
                                 })

            detections_trn = code_detectability_by_percentiles(
                codes_by_percentiles, trn_res['diag_detectability_df'])
            detections_val = code_detectability_by_percentiles(
                codes_by_percentiles, val_res['diag_detectability_df'])
            detections_trn_df = pd.DataFrame(
                index=detections_trn['pre'].keys(),
                data={
                    'Trn(pre)': detections_trn['pre'].values(),
                    'Trn(post)': detections_trn['post'].values()
                })

            detections_val_df = pd.DataFrame(
                index=detections_val['pre'].keys(),
                data={
                    'Val(pre)': detections_val['pre'].values(),
                    'Val(post)': detections_val['post'].values()
                })

            ode_logger.info('\n' + str(losses))
            ode_logger.info('\n' + str(stats))
            ode_logger.info('\n' + str(detections_trn_df))
            ode_logger.info('\n' + str(detections_val_df))

        if step % save_freq == 0 and step > 0:
            with open(f'{save_params_prefix}_step{step:03d}.pickle',
                      'wb') as f:
                pickle.dump(get_params(opt_state), f)

    return {
        'res_val': res_val,
        'res_trn': res_trn,
        'model_params': get_params(opt_state),
        'ode_model': ode_model,
        'trn_ids': train_ids,
        'val_ids': valid_ids
    }
