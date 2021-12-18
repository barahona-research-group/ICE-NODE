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

from .concept import Subject
from .dag import CCSDAG
from .jax_interface import (SubjectJAXInterface, create_patient_interface,
                            Ignore)
from .gram import DAGGRAM
from .models import (MLPDynamics, ResDynamics, GRUDynamics, TaylorAugmented,
                     NeuralODE, DiagnosesUpdate, StateDiagnosesDecoder,
                     StateInitializer)
from .metrics import (jit_sigmoid, bce, balanced_focal_bce, l2_squared,
                      l1_absolute, confusion_matrix, confusion_matrix_scores,
                      top_k_detectability_scores, top_k_detectability_df,
                      roc_df, auc_scores)

ode_logger = logging.getLogger("ode")
debug_flags = {'nan_debug': True, 'shape_debug': True}


def parameters_size(pytree):
    leaves, _ = tree_flatten(pytree)
    return sum(jnp.size(x) for x in leaves)


def tree_hasnan(t):
    return any(map(lambda x: jnp.any(jnp.isnan(x)), jax.tree_leaves(t)))


def tree_lognan(t):
    return jax.tree_map(lambda x: jnp.any(jnp.isnan(x)).item(), t)


def array_hasnan(arr):
    return jnp.any(jnp.isnan(arr) | jnp.isinf(arr))


def arrays_lognan_lazy(arrs_dict_fn, params_fn):
    if debug_flags['nan_debug']:
        arrays_lognan(arrs_dict_fn(), params_fn())


def arrays_lognan(arrs_dict, params=None):
    if debug_flags['nan_debug']:
        invalid_arrays = list(
            filter(lambda a: array_hasnan(a[1]), arrs_dict.items()))
        if len(invalid_arrays) > 0:
            to_np = lambda m: onp.array(lax.stop_gradient(m))
            arrs_dict = tree_map(to_np, arrs_dict)
            norm = lambda m: jnp.sqrt(jnp.sum(m**2))
            arrs_msg = ', '.join(
                map(
                    lambda t:
                    f'{t[0]}: {onp.argwhere(onp.isnan(arrs_dict[t[0]])).tolist()}',
                    invalid_arrays))
            msg = f"{len(invalid_arrays)} Invalid arrays: {arrs_msg}"
            ode_logger.warning('\n===Invalid Arrays Predicate===\n' + msg)
            ode_logger.warning('\n===Arrays Norm===')
            arrs_norm = tree_map(norm, arrs_dict)
            ode_logger.warning('\n' + str(arrs_norm) + '\n')
            timestamp = datetime.now().strftime("%d-%b-%Y_%H-%M-%S")
            arrs_fname = f'arrays_lognan_{timestamp}.npy'
            onp.save(arrs_fname, arrs_dict)
            ode_logger.warning(f'saved arrays to: {arrs_fname}')

            if params is not None:
                params = tree_map(to_np, params)
                params_fname = f'params_lognan_{timestamp}.npy'
                onp.save(params_fname, params)
                ode_logger.warning(f'saved params to :{params_fname}')
                norms = tree_map(norm, params)
                ode_logger.warning(f'\n===Parameters Norm===\n' + str(norms) +
                                   '\n')
            raise ValueError("NaN found")


def pad_list(l, size, pad_val=0):
    while len(l) < size:
        l.append(jnp.zeros_like(l[-1]) + pad_val)
    return l


def pad_mat(m, nrows, pad_val=0):
    if m.shape[0] < nrows:
        pad = jnp.zeros(
            (nrows - m.shape[0], m.shape[1]), dtype=m.dtype) + pad_val
        return jnp.vstack((m, pad))
    else:
        return m


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


class PatientGRUODEBayesInterface:
    def __init__(self, subject_interface: SubjectJAXInterface,
                 diag_gram: DAGGRAM, proc_gram: DAGGRAM, ode_dyn: str,
                 ode_depth: int, tay_reg: Optional[int], state_size: int,
                 init_depth: bool, bias: bool,
                 diag_loss: Callable[[jnp.ndarray, jnp.ndarray], float],
                 max_odeint_days: int, **init_kwargs):

        self.subject_interface = subject_interface
        self.diag_gram = diag_gram
        self.proc_gram = proc_gram
        self.tay_reg = tay_reg
        self.diag_loss = diag_loss
        self.max_odeint_days = max_odeint_days

        self.dimensions = {
            'age': 1,
            'static': len(subject_interface.static_idx),
            'proc_gram': proc_gram.basic_embeddings_dim,
            'diag_gram': diag_gram.basic_embeddings_dim,
            'proc_dag': len(subject_interface.proc_multi_ccs_idx),
            'diag_dag': len(subject_interface.diag_multi_ccs_idx),
            'diag_out': len(subject_interface.diag_single_ccs_idx),
            'state': state_size,
            'ode_depth': ode_depth,
            'init_depth': init_depth
        }

        self.ode_control_passes = ['age', 'static', 'proc_gram']
        self.state_init_passes = ['age', 'static', 'diag_gram']

        self.dimensions.update({
            'ode_control':
            sum(map(self.dimensions.get, self.ode_control_passes)),
            'state_init':
            sum(map(self.dimensions.get, self.state_init_passes))
        })
        """
        Constructs the GRU-ODE-Bayes model with the given dimensions.
        """
        init_kwargs = {
            "with_bias": bias,
            "w_init": hk.initializers.RandomNormal(mean=0, stddev=1e-3),
            "b_init": jnp.zeros
        }
        if ode_dyn == 'gru':
            ode_dyn_cls = GRUDynamics
        elif ode_dyn == 'res':
            ode_dyn_cls = ResDynamics
        elif ode_dyn == 'mlp':
            ode_dyn_cls = MLPDynamics
        else:
            raise RuntimeError(f"Unrecognized dynamics class: {ode_dyn}")

        n_ode_init, n_ode = hk.without_apply_rng(
            hk.transform(
                wrap_module(NeuralODE,
                            ode_dyn_cls=ode_dyn_cls,
                            state_size=state_size,
                            depth=ode_depth,
                            with_quad_augmentation=False,
                            name='n_ode',
                            tay_reg=tay_reg,
                            **init_kwargs)))
        self.n_ode = jax.jit(n_ode, static_argnums=(1, ))

        diag_update_init, diag_update = hk.without_apply_rng(
            hk.transform(
                wrap_module(DiagnosesUpdate,
                            state_size=state_size,
                            with_quad_augmentation=False,
                            name='diagnoses_update',
                            **init_kwargs)))
        self.diag_update = jax.jit(diag_update)

        f_dec_init, f_dec = hk.without_apply_rng(
            hk.transform(
                wrap_module(StateDiagnosesDecoder,
                            hidden_size=self.dimensions['diag_gram'],
                            gram_size=self.dimensions['diag_gram'],
                            output_size=self.dimensions['diag_out'],
                            name='f_dec')))
        self.f_dec = jax.jit(f_dec)

        f_state_init_init, f_state_init = hk.without_apply_rng(
            hk.transform(
                wrap_module(StateInitializer,
                            hidden_size=self.dimensions['diag_gram'],
                            state_size=state_size,
                            depth=init_depth,
                            name='f_init')))
        self.f_state_init = jax.jit(f_state_init)

        self.initializers = {
            'ode_dyn': n_ode_init,
            'diag_update': diag_update_init,
            'f_dec': f_dec_init,
            'f_state_init': f_state_init_init
        }

    def init_params(self, rng_key):
        init_data = self.__initialization_data()
        return {
            "diag_gram": self.diag_gram.init_params(rng_key),
            "proc_gram": self.proc_gram.init_params(rng_key),
            **{
                label: init(rng_key, *init_data[label])
                for label, init in self.initializers.items()
            }
        }

    def state_size(self):
        return self.dimensions['state']

    def diag_out_index(self) -> List[str]:
        index2code = {
            i: c
            for c, i in self.subject_interface.diag_single_ccs_idx.items()
        }
        return list(map(index2code.get, range(len(index2code))))

    def __initialization_data(self):
        """
        Creates data for initializing each of the
        modules based on the shapes of init_data.
        """
        diag_gram = jnp.zeros(self.dimensions['diag_gram'])
        state = jnp.zeros(self.dimensions['state'])
        c = jnp.zeros(self.dimensions['ode_control'])
        i = jnp.zeros(self.dimensions['state_init'])
        return {
            "ode_dyn": [state, 0.1, c],
            "diag_update": [state, diag_gram],
            "f_dec": [state],
            "f_state_init": [i]
        }

    def __extract_nth_points(self, params: Any, subjects_batch: List[int],
                             n: int) -> Dict[str, Dict[int, jnp.ndarray]]:

        diag_G, proc_G = self.__generate_embedding_mats(params)

        points = self.subject_interface.nth_points_batch(n, subjects_batch)
        if len(points) == 0:
            return None

        diag_gram = {
            i: self.diag_gram.encode(diag_G, v['diag_multi_ccs_vec'])
            for i, v in points.items() if v['diag_multi_ccs_vec'] is not None
        }
        proc_gram = {
            i: self.proc_gram.encode(proc_G, v['proc_multi_ccs_vec'])
            for i, v in points.items() if v['proc_multi_ccs_vec'] is not None
        }
        diag_out = {
            i: v['diag_single_ccs_vec']
            for i, v in points.items() if v['diag_single_ccs_vec'] is not None
        }
        days_ahead = {i: v['days_ahead'] for i, v in points.items()}
        age = {i: v['age'] for i, v in points.items()}

        def _ode_control(subject_id):
            zero_proc_gram = jnp.zeros(self.dimensions['proc_gram'])
            d = {
                'proc_gram': proc_gram.get(subject_id, zero_proc_gram),
                'age': points[subject_id]['age'],
                'static': self.subject_interface.subject_static(subject_id)
            }
            return jnp.hstack(map(d.get, self.ode_control_passes))

        ode_control = {i: _ode_control(i) for i in points.keys()}

        return {
            'age': age,
            'days_ahead': days_ahead,
            'diag_gram': diag_gram,
            'ode_control': ode_control,
            'diag_out': diag_out
        }

    def __odeint(self, params, count_nfe, h, t, c):

        h_r_nfe = {
            i: self.n_ode(params['ode_dyn'], count_nfe, h[i], t[i], c[i])
            for i in h.keys()
        }

        nfe = sum(n for h, r, n in h_r_nfe.values())
        r1 = jnp.sum(sum(r for (h, r, n) in h_r_nfe.values()))
        h1 = {i: h for i, (h, r, n) in h_r_nfe.items()}
        return h1, r1, nfe

    def __diag_update(self, params: Any, state: Dict[int, jnp.ndarray],
                      diag_gram_error: jnp.ndarray) -> jnp.ndarray:
        updated_state = {
            i: self.diag_update(params['diag_update'], state[i], gram_error)
            for i, gram_error in diag_gram_error.items()
        }

        return updated_state

    def __state_decode(
        self, params: Any, state: Dict[int, jnp.ndarray], selection: Set[int]
    ) -> Tuple[Dict[int, jnp.ndarray], Dict[int, jnp.ndarray]]:
        gram_out = {
            i: self.f_dec(params['f_dec'], state[i])
            for i in selection
        }
        gram = {i: g for i, (g, _) in gram_out.items()}
        out = {i: o for i, (_, o) in gram_out.items()}

        return gram, out

    def __generate_embedding_mats(self, params):
        diag = self.diag_gram.compute_embedding_mat(params["diag_gram"])
        proc = self.proc_gram.compute_embedding_mat(params["proc_gram"])
        return diag, proc

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

    def __gram_error(self, gram_true, gram_predicted):
        error_gram = {
            i: gram_true[i] - jit_sigmoid(gram_predicted[i])
            for i in gram_predicted.keys()
        }
        return error_gram

    def __confusion_matrix(self, diag_true, diag_predicted):
        cm = {
            i: confusion_matrix(diag_true[i], jit_sigmoid(diag_predicted[i]))
            for i in diag_predicted.keys()
        }
        if cm:
            return sum(cm.values())
        else:
            return None

    def __initial_state(self, params, diag_gram, age, subjects: Iterable[int],
                        days_ahead: Dict[int, int]):
        def _state_init(subject_id):
            d = {
                'diag_gram': diag_gram[subject_id],
                'age': age[subject_id],
                'static': self.subject_interface.subject_static(subject_id)
            }
            state_input = jnp.hstack(map(d.get, self.state_init_passes))
            return {
                'value': self.f_state_init(params['f_state_init'],
                                           state_input),
                'days': days_ahead[subject_id]
            }

        return {i: _state_init(i) for i in (set(diag_gram) & set(subjects))}

    def __call__(self,
                 params: Any,
                 subjects_batch: List[int],
                 return_path: bool = False,
                 count_nfe: bool = False):

        nth_points = partial(self.__extract_nth_points, params, subjects_batch)
        nn_odeint = partial(self.__odeint, params, count_nfe)
        nn_diag_update = partial(self.__diag_update, params)
        nn_state_decode = partial(self.__state_decode, params)

        subject_state = dict()
        dyn_loss = []
        nfe = []

        prejump_diag_loss = []
        postjump_diag_loss = []
        diag_weights = []

        diag_detectability = []
        diag_roc = []
        diag_cm = []  # Confusion matrix
        all_points_count = 0
        predictable_count = 0
        odeint_weeks = 0.0

        for n in self.subject_interface.n_support:
            points_n = nth_points(n)

            if points_n is None:
                continue
            all_points_count += len(points_n)

            days_ahead = points_n['days_ahead']
            age = points_n['age']
            diag_gram = points_n['diag_gram']
            diag_out = points_n['diag_out']
            delta_weeks = {
                i: (days_ahead[i] - subject_state[i]['days']) / 7.0
                for i in (set(subject_state) & set(days_ahead))
                if days_ahead[i] -
                subject_state[i]['days'] <= self.max_odeint_days
            }

            state_i = {i: subject_state[i]['value'] for i in delta_weeks}

            # From points returned at index n:
            # - Consider for odeint:
            #   1. subjects that already have a previous state, and
            #   2. days difference doesn't exceed maximum days.
            # - For returned subjects with days difference exceeding the
            # threshold, reset their previous state.
            # - Initialize new states for subjects that have diagnosis codes
            #   that has not been previously initialized or has been reset.

            # Reset subjects state with long gaps
            reset_subjects = set(days_ahead) - set(delta_weeks)
            map(lambda k: subject_state.pop(k, None), reset_subjects)

            state_0 = self.__initial_state(params, diag_gram, age,
                                           reset_subjects, days_ahead)
            subject_state.update(state_0)
            # This intersection ensures only prediction for:
            # 1. cases that are integrable (i.e. with previous state), and
            # 2. cases that have diagnosis at index n.
            predictable_cases = set(delta_weeks).intersection(diag_out.keys())
            predictable_count += len(predictable_cases)

            # No. of "Predictable" diagnostic points
            diag_weights.append(len(predictable_cases))
            '''
            Idea: scale days_forward to weeks_forward.
            This can:
                1. Improve the numerical stability and accuracy of numerical integration.
                2. Force the ode_dyn model to learn weekly dynamics, which is a suitable time scale for cancer development.
            '''
            odeint_weeks += sum(delta_weeks.values())
            ################## ODEINT #####################
            state_j, _dyn_loss, _nfe = nn_odeint(state_i, delta_weeks,
                                                 points_n['ode_control'])
            dyn_loss.append(_dyn_loss)
            nfe.append(_nfe)
            ########## PRE-JUMP DAG LOSS #########################
            pre_diag_gram, pre_diag_out = nn_state_decode(
                state_j, predictable_cases)
            pre_diag_loss = self.__diag_loss(diag_out, pre_diag_out)
            pre_diag_gram_error = self.__gram_error(diag_gram, pre_diag_gram)
            confusion_mat = self.__confusion_matrix(diag_out, pre_diag_out)
            prejump_diag_loss.append(pre_diag_loss)
            if confusion_mat is not None:
                diag_cm.append(confusion_mat)
            ############## GRU BAYES ####################
            # Using GRUObservationCell to update h.
            state_j_updated = nn_diag_update(state_j, pre_diag_gram_error)
            state_j.update(state_j_updated)
            # Update the states:
            for subject_id, new_state in state_j.items():
                subject_state[subject_id] = {
                    'days': days_ahead[subject_id],
                    'value': new_state
                }

            ############### POST-JUNP DAG LOSS ########################
            _, post_diag_out = nn_state_decode(state_j_updated,
                                               predictable_cases)

            post_diag_loss = self.__diag_loss(diag_out, post_diag_out)
            postjump_diag_loss.append(post_diag_loss)

            diag_detectability.append(
                top_k_detectability_df(20, diag_out, pre_diag_out,
                                       post_diag_out, n))
            diag_roc.append(roc_df(diag_out, pre_diag_out, post_diag_out, n))

        prejump_diag_loss = jnp.average(prejump_diag_loss,
                                        weights=diag_weights)
        postjump_diag_loss = jnp.average(postjump_diag_loss,
                                         weights=diag_weights)

        if diag_cm:
            confusion_mat = sum(cm for cm in diag_cm if cm is not None)
        else:
            confusion_mat = jnp.zeros((2, 2))

        ret = {
            'prejump_diag_loss': prejump_diag_loss,
            'postjump_diag_loss': postjump_diag_loss,
            'dyn_loss': jnp.sum(sum(dyn_loss)),
            'scores': confusion_matrix_scores(confusion_mat),
            'odeint_weeks': odeint_weeks,
            'all_points_count': all_points_count,
            'predictable_count': predictable_count,
            'nfe': sum(nfe),
            'diag_detectability_df': pd.concat(diag_detectability),
            'diag_roc_df': pd.concat(diag_roc)
        }

        return ret


def loss_fn_detail(ode_model: PatientGRUODEBayesInterface,
                   loss_mixing: Dict[str, float], params: optimizers.Params,
                   batch: List[int]) -> Dict[str, float]:
    res = ode_model(params, batch, count_nfe=True)
    prejump_diag_loss = res['prejump_diag_loss']
    postjump_diag_loss = res['postjump_diag_loss']
    l1_loss = l1_absolute(params)
    l2_loss = l2_squared(params)
    dyn_loss = res['dyn_loss']
    diag_alpha = loss_mixing['diag_alpha']
    l1_alpha = loss_mixing['l1_reg']
    l2_alpha = loss_mixing['l2_reg']
    dyn_alpha = loss_mixing['dyn_reg'] / (res['odeint_weeks'])

    diag_loss = (
        1 - diag_alpha) * prejump_diag_loss + diag_alpha * postjump_diag_loss
    loss = diag_loss + (l1_alpha * l1_loss) + (l2_alpha * l2_loss) + (
        dyn_alpha * dyn_loss)
    nfe = res['nfe']
    return {
        'loss': {
            'prejump_diag_loss': prejump_diag_loss,
            'postjump_diag_loss': postjump_diag_loss,
            'diag_loss': diag_loss,
            'loss': loss,
            'l1_loss': l1_loss,
            'l2_loss': l2_loss,
            'dyn_loss': dyn_loss,
            'dyn_loss/week': dyn_loss / res['odeint_weeks']
        },
        'stats': {
            **{name: score.item()
               for name, score in res['scores'].items()}, 'all_points_count':
            res['all_points_count'],
            'predictable_count': res['predictable_count'],
            'nfe/week': nfe / res['odeint_weeks'],
            'nfex1000': nfe / 1000
        },
        'diag_detectability_df': res['diag_detectability_df'],
        'diag_roc_df': res['diag_roc_df']
    }


def loss_fn(ode_model: PatientGRUODEBayesInterface, loss_mixing: Dict[str,
                                                                      float],
            params: optimizers.Params, batch: List[int]) -> float:
    res = ode_model(params, batch, count_nfe=False)

    prejump_diag_loss = res['prejump_diag_loss']
    postjump_diag_loss = res['postjump_diag_loss']
    l1_loss = l1_absolute(params)
    l2_loss = l2_squared(params)
    dyn_loss = res['dyn_loss'] / (res['odeint_weeks'])
    diag_alpha = loss_mixing['diag_alpha']
    l1_alpha = loss_mixing['l1_reg']
    l2_alpha = loss_mixing['l2_reg']
    dyn_alpha = loss_mixing['dyn_reg']

    diag_loss = (
        1 - diag_alpha) * prejump_diag_loss + diag_alpha * postjump_diag_loss
    loss = diag_loss + (l1_alpha * l1_loss) + (l2_alpha * l2_loss) + (
        dyn_alpha * dyn_loss)

    return loss


def train_ehr(
        subject_interface: SubjectJAXInterface,
        diag_gram: DAGGRAM,
        proc_gram: DAGGRAM,
        rng: Any,
        # Model configurations
        model_config: Dict[str, Any],
        # Training configurations
        train_validation_split: float,
        batch_size: int,
        epochs: int,
        lr: float,
        diag_loss: str,
        tay_reg: Optional[int],
        loss_mixing: Dict[str, float],
        eval_freq: int,
        save_freq: Optional[int],
        save_params_prefix: Optional[str],
        # Debugging
        verbose_debug=False,
        nan_debug=False,
        shape_debug=False,
        memory_profile=False,
        **init_kwargs):

    diag_loss_function = {
        'balanced_focal':
        lambda t, p: balanced_focal_bce(t, p, gamma=2, beta=0.999),
        'bce':
        bce
    }

    if verbose_debug:
        ode_logger.setLevel(logging.DEBUG)
    else:
        ode_logger.setLevel(logging.INFO)
    debug_flags['nan_debug'] = nan_debug
    debug_flags['shape_debug'] = shape_debug

    prng_key = jax.random.PRNGKey(rng.randint(0, 100))

    ode_model = PatientGRUODEBayesInterface(
        subject_interface=subject_interface,
        diag_gram=diag_gram,
        proc_gram=proc_gram,
        **model_config,
        tay_reg=tay_reg,
        diag_loss=diag_loss_function[diag_loss])

    params = ode_model.init_params(prng_key)
    ode_logger.info(f'#params: {parameters_size(params)}')
    ode_logger.debug(f'shape(params): {tree_map(jnp.shape, params)}')

    opt_init, opt_update, get_params = optimizers.adam(step_size=lr)

    def update(
            step: int, batch: Iterable[int],
            opt_state: optimizers.OptimizerState) -> optimizers.OptimizerState:
        params = get_params(opt_state)
        """Single SGD update step."""
        if nan_debug:
            if tree_hasnan(params):
                ode_logger.warning(tree_lognan(params))
                raise ValueError("Nan Params")

        grads = jax.grad(loss_fn)(ode_model, loss_mixing, params, batch)
        if nan_debug:
            if tree_hasnan(grads):
                ode_logger.warning(tree_lognan(grads))
                raise ValueError("Nan Grads")

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
            opt_state = update(step, train_batch, opt_state)
        except (ValueError, FloatingPointError) as e:
            from traceback import format_exception
            tb_str = ''.join(format_exception(None, e, e.__traceback__))
            ode_logger.warning(f'ValueError exception raised: {tb_str}')
            break

        if step % save_freq == 0 and step > 0:
            with open(f'{save_params_prefix}_step{step:03d}.pickle',
                      'wb') as f:
                pickle.dump(get_params(opt_state), f)

        if step % eval_freq != 0:
            continue

        rng.shuffle(valid_ids)
        valid_batch = valid_ids  #[:val_batch_size]
        params = get_params(opt_state)
        trn_res = loss_fn_detail(ode_model, loss_mixing, params, train_batch)
        res_trn[step] = trn_res

        val_res = loss_fn_detail(ode_model, loss_mixing, params, valid_batch)
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

        detections_trn = top_k_detectability_scores(
            codes_by_percentiles, trn_res['diag_detectability_df'])
        detections_val = top_k_detectability_scores(
            codes_by_percentiles, val_res['diag_detectability_df'])
        detections_trn_df = pd.DataFrame(index=detections_trn['pre'].keys(),
                                         data={
                                             'Trn(pre)':
                                             detections_trn['pre'].values(),
                                             'Trn(post)':
                                             detections_trn['post'].values()
                                         })

        detections_val_df = pd.DataFrame(index=detections_val['pre'].keys(),
                                         data={
                                             'Val(pre)':
                                             detections_val['pre'].values(),
                                             'Val(post)':
                                             detections_val['post'].values()
                                         })

        auc_trn = auc_scores(trn_res['diag_roc_df'])
        auc_val = auc_scores(val_res['diag_roc_df'])

        auc_df = pd.DataFrame(index=auc_trn.keys(),
                              data={
                                  'Trn(AUC)': auc_trn.values(),
                                  'Val(AUC)': auc_val.values()
                              })

        ode_logger.info('\n' + str(losses))
        ode_logger.info('\n' + str(stats))
        ode_logger.info('\n' + str(detections_trn_df))
        ode_logger.info('\n' + str(detections_val_df))
        ode_logger.info('\n' + str(auc_df))

    return {
        'res_val': res_val,
        'res_trn': res_trn,
        'model_params': get_params(opt_state),
        'ode_model': ode_model,
        'trn_ids': train_ids,
        'val_ids': valid_ids
    }


if __name__ == '__main__':
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
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    mimic_processed_dir = args.mimic_processed_dir
    output_dir = args.output_dir
    cpu = args.cpu

    if cpu:
        jax.config.update('jax_platform_name', 'cpu')
    else:
        jax.config.update('jax_platform_name', 'gpu')

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logging.info('[LOADING] Patients JAX Interface.')
    patient_interface = create_patient_interface(mimic_processed_dir,
                                                 Ignore.TESTS)
    logging.info('[DONE] Patients JAX Interface')

    rng = random.Random(42)
    subjects_id = list(patient_interface.subjects.keys())
    rng.shuffle(subjects_id)

    config = {
        'glove_config': {
            'diag_idx': patient_interface.diag_multi_ccs_idx,
            'proc_idx': patient_interface.proc_multi_ccs_idx,
            'ccs_dag': patient_interface.dag,
            'subjects': patient_interface.subjects.values(),
            'diag_vector_size': 100,
            'proc_vector_size': 50,
            'iterations': 30,
            'window_size_days': 2 * 365
        },
        'gram_config': {
            'diag': {
                'ccs_dag': patient_interface.dag,
                'code2index': patient_interface.diag_multi_ccs_idx,
                'attention_method': 'tanh',
                'attention_dim': 100,
                'ancestors_mat':
                patient_interface.diag_multi_ccs_ancestors_mat,
            },
            'proc': {
                'ccs_dag': patient_interface.dag,
                'code2index': patient_interface.proc_multi_ccs_idx,
                'attention_method': 'tanh',
                'attention_dim': 50,
                'ancestors_mat':
                patient_interface.proc_multi_ccs_ancestors_mat,
            }
        },
        'model': {
            'ode_dyn': 'gru',
            'state_size': 100,
            'init_depth': 1,
            'bias': True,
            'max_odeint_days': 8 * 7
        },
        'training': {
            'batch_size': 5,
            'epochs': 1,
            'lr': 1e-5,
            'diag_loss': 'balanced_focal',
            'tay_reg':
            3,  # Order of regularized derivative of the dynamics function (None for disable).
            'loss_mixing': {
                'diag_alpha': 1e-4,
                'ode_alpha': 1e-5,
                'l1_reg': 1e-7,
                'l2_reg': 1e-6,
                'dyn_reg': 1e-3
            }
        }
    }
    diag_glove, proc_glove = glove_representation(**config['glove_config'])
    diag_gram = DAGGRAM(**config['gram_config']['diag'],
                        basic_embeddings=diag_glove)
    proc_gram = DAGGRAM(**config['gram_config']['proc'],
                        basic_embeddings=proc_glove)

    res = train_ehr(subject_interface=patient_interface,
                    diag_gram=diag_gram,
                    proc_gram=proc_gram,
                    rng=random.Random(42),
                    model_config=config['model'],
                    **config['training'],
                    verbose_debug=False,
                    shape_debug=False,
                    nan_debug=False,
                    memory_profile=False)
