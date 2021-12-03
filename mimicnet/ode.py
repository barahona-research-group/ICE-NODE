import logging
import math
import pickle
from collections import defaultdict
from datetime import datetime
from enum import Enum, auto, unique
from functools import partial
from typing import (AbstractSet, Any, Callable, Dict, Iterable, List, Mapping,
                    Optional, Tuple, Union)

import numpy as onp
import pandas as pd
import haiku as hk
import jax
import jax.numpy as jnp
from jax import lax
from jax.profiler import annotate_function
from jax.experimental import optimizers
from jax.experimental.jet import jet
from jax.experimental.ode import odeint
from jax.nn import softplus, sigmoid, leaky_relu

from tqdm import tqdm
from jax.tree_util import tree_flatten, tree_map

from .jax_interface import SubjectJAXInterface
from .gram import DAGGRAM
from .odeint_nfe import odeint as odeint_nfe

ode_logger = logging.getLogger("ode")
debug_flags = {'nan_debug': True, 'shape_debug': True}


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


@jax.jit
def jit_sigmoid(x):
    return sigmoid(x)


@partial(annotate_function, name="bce")
@jax.jit
def bce(y: jnp.ndarray, logits: jnp.ndarray):
    return jnp.mean(y * softplus(-logits) + (1 - y) * softplus(logits))


# The following loss function employs two concepts:
# A) Effective number of sample, to mitigate class imbalance:
# Paper: Class-Balanced Loss Based on Effective Number of Samples (Cui et al)
# B) Focal loss, to underweight the easy to classify samples:
# Paper: Focal Loss for Dense Object Detection (Lin et al)
@partial(annotate_function, name="balanced_focal_bce")
@jax.jit
def balanced_focal_bce(y: jnp.ndarray,
                       logits: jnp.ndarray,
                       gamma=2,
                       beta=0.999):
    n1 = jnp.sum(y)
    n0 = jnp.size(y) - n1
    # Effective number of samples.
    e1 = (1 - beta**n1) / (1 - beta) + 1e-1
    e0 = (1 - beta**n0) / (1 - beta) + 1e-1

    # Focal weighting
    p = sigmoid(logits)
    w1 = jnp.power(1 - p, gamma)
    w0 = jnp.power(p, gamma)
    # Note: softplus(-logits) = -log(sigmoid(logits)) = -log(p)
    # Note: softplut(logits) = -log(1 - sigmoid(logits)) = -log(1-p)
    return jnp.mean(y * (w1 / e1) * softplus(-logits) + (1 - y) *
                    (w0 / e0) * softplus(logits))


@partial(annotate_function, name="l2_loss")
@jax.jit
def l2_squared(pytree):
    leaves, _ = tree_flatten(pytree)
    return sum(jnp.vdot(x, x) for x in leaves)


@partial(annotate_function, name="l1_loss")
@jax.jit
def l1_absolute(pytree):
    leaves, _ = tree_flatten(pytree)
    return sum(jnp.sum(jnp.fabs(x)) for x in leaves)


def parameters_size(pytree):
    leaves, _ = tree_flatten(pytree)
    return sum(jnp.size(x) for x in leaves)


# MIT License

# Copyright (c) 2020 Jacob Kelly and Jesse Bettencourt

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.


# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
def sol_recursive(order: int, f: Callable[[jnp.ndarray, jnp.ndarray],
                                          jnp.ndarray], z: jnp.ndarray,
                  t: float):
    """
    Recursively compute higher order derivatives of dynamics of ODE.
    """
    if order < 2:
        return f(z, t), jnp.zeros_like(z)

    z_shape = z.shape
    z_t = jnp.concatenate((jnp.ravel(z), jnp.array([t])))

    def g(z_t):
        """
        Closure to expand z.
        """
        z, t = jnp.reshape(z_t[:-1], z_shape), z_t[-1]
        dz = jnp.ravel(f(z, t))
        dt = jnp.array([1.])
        dz_t = jnp.concatenate((dz, dt))
        return dz_t

    (y0, [*yns]) = jet(g, (z_t, ), ((jnp.ones_like(z_t), ), ))
    for _ in range(order - 1):
        (y0, [*yns]) = jet(g, (z_t, ), ((y0, *yns), ))

    return (jnp.reshape(y0[:-1], z_shape), jnp.reshape(yns[-2][:-1], z_shape))


def augment_dynamics(reg, dynamics):
    """
    Closure to augment dynamics.
    """
    def reg_dynamics(y, t):
        """
        Dynamics of regularization.
        """
        y0, r = sol_recursive(reg, lambda _y, _t: dynamics(_y, _t), y, t)
        return y0, jnp.mean(r**2)

    def aug_dynamics(yr, t):
        """
        Dynamics augmented with regularization.
        """
        y, r = yr
        dydt, drdt = reg_dynamics(y, t)
        return dydt, drdt

    return aug_dynamics


# GRU-ODE: Neural Negative Feedback ODE with Bayesian jumps
'''
Note(Asem): Haiku modules don't define the input dimensions
Input dimensions are defined during initialization step afterwards.
'''

# IDEA: research on mixed loss functions.
# Instead of using hyperparamters for mixing loss functions,
# Each training iteration optimizes one of the loss functions on a rolling
# basis.


class MLPDynamics(hk.Module):
    """
    Dynamics for ODE as an MLP.
    """
    def __init__(self,
                 state_size: int,
                 with_bias: bool,
                 w_init: hk.initializers.Initializer,
                 b_init: hk.initializers.Initializer,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.lin1 = hk.Linear(state_size,
                              with_bias=with_bias,
                              w_init=w_init,
                              b_init=b_init)
        self.lin2 = hk.Linear(state_size,
                              with_bias=with_bias,
                              w_init=w_init,
                              b_init=b_init)

    def __call__(self, h, t, c):
        out = sigmoid(h)
        out = jnp.hstack((c, out, t))
        out = self.lin1(out)

        out = sigmoid(out)
        out = jnp.hstack((c, out, t))
        out = self.lin2(out)

        return out


class GRUDynamics(hk.Module):
    """
    Modified GRU unit to deal with latent state ``h(t)``.
    """
    def __init__(self,
                 state_size: int,
                 with_bias: bool,
                 w_init: hk.initializers.Initializer,
                 b_init: hk.initializers.Initializer,
                 name: Optional[str] = None,
                 **init_kwargs):
        super().__init__(name=name)

        self.__hc_r = hk.Sequential([
            hk.Linear(state_size,
                      with_bias=with_bias,
                      w_init=w_init,
                      b_init=b_init,
                      name='hc_r'), sigmoid
        ])

        self.__hc_z = hk.Sequential([
            hk.Linear(state_size,
                      with_bias=with_bias,
                      w_init=w_init,
                      b_init=b_init,
                      name='hc_z'), sigmoid
        ])

        self.__rhc_g = hk.Sequential([
            hk.Linear(state_size,
                      with_bias=with_bias,
                      w_init=w_init,
                      b_init=b_init,
                      name='rhc_g'), jnp.tanh
        ])

    def __call__(self, h: jnp.ndarray, t: float,
                 c: jnp.ndarray) -> jnp.ndarray:
        """
        Returns a change due to one step of using GRU-ODE for all h.
        Args:
            h: hidden state (current) of the observable variables.
            c: control variables.
        Returns:
            dh/dt
        """
        htc = jnp.hstack([h, c])
        r = self.__hc_r(htc)
        z = self.__hc_z(htc)
        rhtc = jnp.hstack([r * h, c])
        g = self.__rhc_g(rhtc)

        return (1 - z) * (g - h)


class NeuralODE(hk.Module):
    def __init__(self,
                 ode_dyn: str,
                 state_size: int,
                 name: Optional[str] = None,
                 **init_kwargs):
        super().__init__(name=name)

        if ode_dyn == 'gru':
            ode_dyn_cls = GRUDynamics
        elif ode_dyn == 'mlp':
            ode_dyn_cls = MLPDynamics
        else:
            raise RuntimeError(f"Unrecognized dynamics class: {ode_dyn}")

        self.ode_dyn = ode_dyn_cls(state_size=state_size,
                                   name='ode_dyn',
                                   **init_kwargs)
        self.tay_reg = None

    def __call__(self, h, t, c):
        if hk.running_init():
            h = self.ode_dyn(h, t, c)
            return jnp.split(h, 2)[1].squeeze()

        h = odeint(self.ode_dyn, h, jnp.array([0.0, t]), c)
        h1 = jnp.split(h, 2)[1].squeeze()
        return h1


class NeuralODETayMode(NeuralODE):
    def __init__(self,
                 ode_dyn: str,
                 state_size: int,
                 tay_reg: int,
                 name: Optional[str] = None,
                 **init_kwargs):
        super().__init__(ode_dyn, state_size, name, **init_kwargs)
        self.tay_reg = tay_reg

    def __call__(self, h, t, c):
        if hk.running_init():
            h = self.ode_dyn(h, t, c)
            return jnp.split(h, 2)[1].squeeze(), jnp.zeros(1)

        h, r = odeint(
            augment_dynamics(self.tay_reg,
                             lambda _h, _t: self.ode_dyn(_h, _t, c)),
            (h, jnp.zeros(1)), jnp.array([0.0, t]))
        h1 = jnp.split(h, 2)[1].squeeze()
        r1 = jnp.split(r, 2)[1].squeeze()
        return h1, r1


class NeuralODETayModeNFE(NeuralODE):
    def __init__(self,
                 ode_dyn: str,
                 state_size: int,
                 tay_reg: int,
                 name: Optional[str] = None,
                 **init_kwargs):
        super().__init__(ode_dyn, state_size, name, **init_kwargs)
        self.tay_reg = tay_reg

    def __call__(self, h, t, c):
        if hk.running_init():
            h = self.ode_dyn(h, t, c)
            return jnp.split(h, 2)[1].squeeze(), jnp.zeros(1), 0

        (h, r), nfe = odeint_nfe(
            augment_dynamics(self.tay_reg,
                             lambda _h, _t: self.ode_dyn(_h, _t, c)),
            (h, jnp.zeros(1)), jnp.array([0.0, t]))

        h1 = jnp.split(h, 2)[1].squeeze()
        r1 = jnp.split(r, 2)[1].squeeze()
        return h1, r1, nfe


class NeuralODENFE(NeuralODE):
    def __init__(self,
                 ode_dyn: str,
                 state_size: int,
                 name: Optional[str] = None,
                 **init_kwargs):
        super().__init__(ode_dyn, state_size, name, **init_kwargs)

    def __call__(self, h, t, c):
        if hk.running_init():
            h = self.ode_dyn(h, t, c)
            return jnp.split(h, 2)[1].squeeze(), 0

        h, nfe = odeint_nfe(lambda _h, _t: self.ode_dyn(_h, _t, c), h,
                            jnp.array([0.0, t]))
        h1 = jnp.split(h, 2)[1].squeeze()
        return h1, nfe


class NumericObsModel(hk.Module):
    """
    The mapping from hidden h to the distribution parameters of Y(t).
    The dimension of the transformation is |h|->|obs_hidden|->|2D|.
    """
    def __init__(self,
                 numeric_size: int,
                 numeric_hidden_size: int,
                 name: Optional[str] = None,
                 **init_kwargs):
        super().__init__(name=name)
        self.__numeric_size = numeric_size
        self.__numeric_hidden_size = numeric_hidden_size
        self.__lin_mean = hk.Linear(numeric_size)
        self.__lin_logvar = hk.Linear(numeric_size)

    def __call__(self, h: jnp.ndarray) -> jnp.ndarray:
        out = hk.Linear(self.__numeric_hidden_size)(h)
        out = leaky_relu(out, negative_slope=2e-1)

        out_logvar = jnp.tanh(self.__lin_logvar(out))
        out_mean = jnp.tanh(self.__lin_mean(out))
        return out_mean, out_logvar


class GRUBayes(hk.Module):
    """Implements discrete update based on the received observations."""
    def __init__(self,
                 state_size: int,
                 name: Optional[str] = None,
                 **init_kwargs):
        super().__init__(name=name)
        self.__prep = hk.Sequential([
            hk.Linear(state_size, with_bias=True, name=f'{name}_prep1'),
            lambda o: leaky_relu(o, negative_slope=2e-1),
            hk.Linear(state_size, with_bias=True,
                      name=f'{name}_prep2'), jnp.tanh
        ])

        self.__gru_d = hk.GRU(state_size)

    def __call__(self, state: jnp.ndarray, error_numeric: jnp.ndarray,
                 numeric_mask: jnp.ndarray,
                 error_gram: jnp.ndarray) -> jnp.ndarray:

        error_numeric = error_numeric * numeric_mask
        gru_input = self.__prep(jnp.hstack((error_numeric, error_gram)))
        _, updated_state = self.__gru_d(gru_input, state)
        return updated_state


class StateDecoder(hk.Module):
    def __init__(self,
                 hidden_size: int,
                 gram_size: int,
                 output_size: int,
                 name: Optional[str] = None,
                 **init_kwargs):
        super().__init__(name=name)
        self.__lin_h = hk.Linear(hidden_size // 2, name='lin_h_hidden')
        self.__lin_num1 = hk.Linear(hidden_size // 2, name='lin_num_hidden1')
        self.__lin_num2 = hk.Linear(hidden_size, name='lin_num_hidden2')

        self.__lin_gram = hk.Linear(gram_size, name='lin_gram')
        self.__lin_out = hk.Linear(output_size, name='lin_out')

    def __call__(self, h: jnp.ndarray, mean: jnp.ndarray):
        out_h = jnp.tanh(self.__lin_h(h))

        out_n = leaky_relu(self.__lin_num1(mean), negative_slope=2e-1)
        out_n = jnp.tanh(self.__lin_num2(out_n))

        dec_in = jnp.hstack((out_h, out_n))
        dec_gram = self.__lin_gram(dec_in)
        logits = self.__lin_out(leaky_relu(dec_gram, negative_slope=2e-1))
        return dec_gram, logits


@partial(annotate_function, name="numeric_error")
@jax.jit
def numeric_error(mean_true: jnp.ndarray, mean_predicted: jnp.ndarray,
                  logvar: jnp.ndarray) -> jnp.ndarray:
    sigma = jnp.exp(0.5 * logvar)
    return (mean_true - mean_predicted) / sigma


@partial(annotate_function, name="lognormal_loss")
@jax.jit
def lognormal_loss(mask: jnp.ndarray, error: jnp.ndarray,
                   logvar: jnp.ndarray) -> float:
    log_lik_c = jnp.log(jnp.sqrt(2 * jnp.pi))
    return 0.5 * ((jnp.power(error, 2) + logvar + 2 * log_lik_c) *
                  mask).sum() / (mask.sum() + 1e-10)


def gaussian_KL(mu_1: jnp.ndarray, mu_2: jnp.ndarray, sigma_1: jnp.ndarray,
                sigma_2: float) -> jnp.ndarray:
    return (jnp.log(sigma_2) - jnp.log(sigma_1) +
            (jnp.power(sigma_1, 2) + jnp.power(
                (mu_1 - mu_2), 2)) / (2 * sigma_2**2) - 0.5)


@partial(annotate_function, name="kl_loss")
@jax.jit
def compute_KL_loss(mean_true: jnp.ndarray,
                    mask: jnp.ndarray,
                    mean_predicted: jnp.ndarray,
                    logvar_predicted: jnp.ndarray,
                    obs_noise_std: float = 1e-1) -> float:
    std = jnp.exp(0.5 * logvar_predicted)
    return (gaussian_KL(mu_1=mean_predicted,
                        mu_2=mean_true,
                        sigma_1=std,
                        sigma_2=obs_noise_std) * mask).sum() / (jnp.sum(mask) +
                                                                1e-10)


@jax.jit
def confusion_matrix(y_true: jnp.ndarray, y_hat: jnp.ndarray):
    y_hat = (jnp.round(y_hat) == 1)
    y_true = (y_true == 1)

    tp = jnp.sum(y_true & y_hat)
    tn = jnp.sum((~y_true) & (~y_hat))
    fp = jnp.sum((~y_true) & y_hat)
    fn = jnp.sum(y_true & (~y_hat))

    return jnp.array([[tp, fn], [fp, tn]], dtype=int)


def confusion_matrix_scores(cm: jnp.ndarray):
    cm = cm / cm.sum()
    tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    p = tp + fn
    n = tn + fp
    return {
        'accuracy': (tp + tn) / (p + n),
        'recall': tp / p,
        'npv': tn / (tn + fn),
        'specificity': tn / n,
        'precision': tp / (tp + fp),
        'f1-score': 2 * tp / (2 * tp + fp + fn),
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }


def code_detectability(top_k: int, true_diag: jnp.ndarray,
                       prejump_predicted_diag: jnp.ndarray,
                       postjump_predicted_diag: jnp.ndarray):
    ground_truth = jnp.argwhere(true_diag).squeeze()
    if ground_truth.ndim > 0:
        ground_truth = set(ground_truth)
    else:
        ground_truth = {ground_truth.item()}

    prejump_predictions = set(jnp.argsort(prejump_predicted_diag)[-top_k:])
    postjump_predictions = set(jnp.argsort(postjump_predicted_diag)[-top_k:])
    detections = []
    for code_i in ground_truth:
        pre_detected, post_detected = 0, 0
        if code_i in prejump_predictions:
            pre_detected = 1
        if code_i in postjump_predictions:
            post_detected = 1
        detections.append((code_i, pre_detected, post_detected))

    return detections


def code_detectability_df(top_k: int, true_diag: Dict[int, jnp.ndarray],
                          prejump_predicted_diag: Dict[int, jnp.ndarray],
                          postjump_predicted_diag: Dict[int, jnp.ndarray],
                          point_n: int):
    detections = {
        i: code_detectability(top_k, true_diag[i], prejump_predicted_diag[i],
                              postjump_predicted_diag[i])
        for i in true_diag.keys()
    }
    df_list = []

    for subject_id, _detections in detections.items():
        for code_i, pre_detected, post_detected in _detections:
            df_list.append((subject_id, point_n, code_i, pre_detected,
                            post_detected, top_k))

    if df_list:
        return pd.DataFrame(df_list,
                            columns=[
                                'subject_id', 'point_n', 'code',
                                'pre_detected', 'post_detected', 'top_k'
                            ])
    else:
        return None


def code_detectability_by_percentiles(codes_by_percentiles, detections_df):
    rate = {'pre': {}, 'post': {}}
    for i, codes in enumerate(codes_by_percentiles):
        codes_detections_df = detections_df[detections_df.code.isin(codes)]
        detection_rate_pre = codes_detections_df.pre_detected.mean()
        detection_rate_post = codes_detections_df.post_detected.mean()
        C = len(codes)
        N = len(codes_detections_df)
        rate['pre'][f'P{i}(N={N} C={len(codes)})'] = detection_rate_pre
        rate['post'][f'P{i}(N={N} C={len(codes)})'] = detection_rate_post
    return rate


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


Tf_num = Callable[[jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]
Todeint = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
Tgru_bayes = Callable[
    [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    jnp.ndarray]
Tf_state_decode = Callable[[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray,
                                                             jnp.ndarray]]


class PatientGRUODEBayesInterface:
    def __init__(self, subject_interface: SubjectJAXInterface,
                 diag_gram: DAGGRAM, proc_gram: DAGGRAM, ode_dyn: str,
                 tay_reg: Optional[int], state_size: int,
                 numeric_hidden_size: int, bias: bool,
                 diag_loss: Callable[[jnp.ndarray, jnp.ndarray], float]):

        self.subject_interface = subject_interface
        self.diag_gram = diag_gram
        self.proc_gram = proc_gram
        self.tay_reg = tay_reg
        self.diag_loss = diag_loss

        self.dimensions = {
            'age': 1,
            'static': len(subject_interface.static_idx),
            'numeric': len(subject_interface.test_idx),
            'proc_gram': proc_gram.basic_embeddings_dim,
            'diag_gram': diag_gram.basic_embeddings_dim,
            'proc_dag': len(subject_interface.proc_multi_ccs_idx),
            'diag_dag': len(subject_interface.diag_multi_ccs_idx),
            'diag_out': len(subject_interface.diag_single_ccs_idx),
            'state': state_size,
            'numeric_hidden': numeric_hidden_size
        }

        self.ode_control_passes = ['age', 'static', 'proc_gram']

        self.dimensions.update({
            'ode_control':
            sum(map(self.dimensions.get, ['age', 'static', 'proc_gram']))
        })
        """
        Constructs the GRU-ODE-Bayes model with the given dimensions.
        """
        init_kwargs = {
            "with_bias": bias,
            "w_init": hk.initializers.RandomNormal(mean=0, stddev=0.01),
            "b_init": jnp.zeros
        }

        n_ode = {
            'default':
            hk.without_apply_rng(
                hk.transform(
                    wrap_module(NeuralODE,
                                ode_dyn=ode_dyn,
                                state_size=state_size,
                                name='n_ode',
                                **init_kwargs))),
            'nfe':
            hk.without_apply_rng(
                hk.transform(
                    wrap_module(NeuralODENFE,
                                ode_dyn=ode_dyn,
                                state_size=state_size,
                                name='n_ode',
                                **init_kwargs))),
            'tay':
            hk.without_apply_rng(
                hk.transform(
                    wrap_module(NeuralODETayMode,
                                ode_dyn=ode_dyn,
                                state_size=state_size,
                                name='n_ode',
                                tay_reg=tay_reg,
                                **init_kwargs))),
            'tay_nfe':
            hk.without_apply_rng(
                hk.transform(
                    wrap_module(NeuralODENFE,
                                ode_dyn=ode_dyn,
                                state_size=state_size,
                                name='n_ode',
                                tay_reg=tay_reg,
                                **init_kwargs))),
        }

        n_ode_init = n_ode['default'][0]
        self.n_ode = {l: jax.jit(f) for l, (_, f) in n_ode.items()}

        gru_bayes_init, gru_bayes = hk.without_apply_rng(
            hk.transform(
                wrap_module(GRUBayes,
                            state_size=state_size,
                            name='gru_bayes',
                            **init_kwargs)))
        self.gru_bayes = jax.jit(gru_bayes)

        f_num_init, f_num = hk.without_apply_rng(
            hk.transform(
                wrap_module(NumericObsModel,
                            numeric_size=self.dimensions['numeric'],
                            numeric_hidden_size=numeric_hidden_size,
                            name='f_numeric',
                            **init_kwargs)))

        self.f_num = jax.jit(f_num)

        f_dec_init, f_dec = hk.without_apply_rng(
            hk.transform(
                wrap_module(StateDecoder,
                            hidden_size=self.dimensions['diag_gram'],
                            gram_size=self.dimensions['diag_gram'],
                            output_size=self.dimensions['diag_out'],
                            name='f_dec')))
        self.f_dec = jax.jit(f_dec)

        self.initializers = {
            'ode_dyn': n_ode_init,
            'gru_bayes': gru_bayes_init,
            'f_num': f_num_init,
            'f_dec': f_dec_init
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

    def numeric_index(self) -> List[str]:
        index2test = {i: t for t, i in self.subject_interface.test_idx.items()}
        return list(map(index2test.get, range(len(index2test))))

    def __initialization_data(self):
        """
        Creates data for initializing each of the
        modules based on the shapes of init_data.
        """
        numeric = jnp.zeros(self.dimensions['numeric'])
        diag_gram = jnp.zeros(self.dimensions['diag_gram'])
        state = jnp.zeros(self.dimensions['state'])
        c = jnp.zeros(self.dimensions['ode_control'])
        return {
            "ode_dyn": [state, 0.1, c],
            "gru_bayes": [state, numeric, numeric, diag_gram],
            "f_num": [state],
            "f_dec": [state, numeric]
        }

    def __extract_nth_points(
            self, diag_emb_mat: jnp.ndarray, proc_emb_mat: jnp.ndarray, n: int,
            subjects_batch: List[int],
            progress_callback: Any) -> Dict[str, Dict[int, jnp.ndarray]]:

        progress_callback('info retrieval')
        points = self.subject_interface.nth_points_batch(n, subjects_batch)
        if len(points) == 0:
            return None

        progress_callback('GRAM')
        diag_gram = {
            i: self.diag_gram.encode(diag_emb_mat, v['diag_multi_ccs_vec'])
            for i, v in points.items() if v['diag_multi_ccs_vec'] is not None
        }
        proc_gram = {
            i: self.proc_gram.encode(proc_emb_mat, v['proc_multi_ccs_vec'])
            for i, v in points.items() if v['proc_multi_ccs_vec'] is not None
        }
        diag_out = {
            i: v['diag_single_ccs_vec']
            for i, v in points.items() if v['diag_single_ccs_vec'] is not None
        }
        days_ahead = {i: v['days_ahead'] for i, v in points.items()}
        numeric = {i: v['tests'][0] for i, v in points.items()}
        mask = {i: v['tests'][1] for i, v in points.items()}

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
            'days_ahead': days_ahead,
            'diag_gram': diag_gram,
            'numeric': numeric,
            'mask': mask,
            'ode_control': ode_control,
            'diag_out': diag_out
        }

    def __f_num(
        self, params: Any, state: Dict[str, jnp.ndarray]
    ) -> Tuple[Dict[int, jnp.ndarray], Dict[int, jnp.ndarray]]:
        mean_logvar = {
            i: self.f_num(params['f_num'], h)
            for i, h in state.items()
        }
        mean = {i: m for i, (m, _) in mean_logvar.items()}
        logvar = {i: lv for i, (_, lv) in mean_logvar.items()}
        return mean, logvar

    def __odeint(self, params, h, t, c):
        return {
            i: self.n_ode['default'](params, h[i], t[i], c[i])
            for i in h.keys()
        }

    def __odeint_tay(self, params, h, t, c):
        h_r = {
            i: self.n_ode['tay'](params, h[i], t[i], c[i])
            for i in h.keys()
        }

        h1 = {i: h for i, (h, _) in h_r.items()}
        r1 = {i: r for i, (_, r) in h_r.items()}
        return h1, r1

    def __odeint_tay_nfe(self, params, h, t, c):
        h_r_nfe = {
            i: self.n_ode['tay_nfe'](params, h[i], t[i], c[i])
            for i in h.keys()
        }
        nfe = sum(n for h, r, n in h_r_nfe.values())
        h1 = {i: h for i, (h, r, n) in h_r_nfe.items()}
        r1 = {i: r for i, (h, r, n) in h_r_nfe.items()}
        return h1, r1, nfe

    def __odeint_nfe(self, params, h, t, c):
        h_nfe = {
            i: self.n_ode['nfe'](params, h[i], t[i], c[i])
            for i in h.keys()
        }
        nfe = sum(n for _, n in h_nfe.values())
        h1 = {i: h for i, (h, _) in h_nfe.items()}
        return h1, nfe

    def __odeint_wrapper(self, store_nfe_fn: Callable[[int], None],
                         store_reg_fn: Callable[[int], None], params: Any,
                         count_nfe) -> Todeint:
        def with_dynamic_regulation_nfe(h, t, c):
            h1, r, nfe = self.__odeint_tay_nfe(params['ode_dyn'], h, t, c)
            store_reg_fn(sum(r.values()))
            store_nfe_fn(nfe)
            return h1

        def without_dynamic_regulation_nfe(h, t, c):
            h1, nfe = self.__odeint_nfe(params['ode_dyn'], h, t, c)
            store_reg_fn(nfe)
            return h1

        def with_dynamic_regulation(h, t, c):
            h1, r = self.__odeint_tay_nfe(params['ode_dyn'], h, t, c)
            store_reg_fn(sum(r.values()))
            return h1

        def without_dynamic_regulation(h, t, c):
            h1 = self.__odeint(params['ode_dyn'], h, t, c)
            return h1

        if self.tay_reg is not None and self.tay_reg >= 2:
            if count_nfe:
                return with_dynamic_regulation_nfe
            else:
                return with_dynamic_regulation
        elif count_nfe:
            return without_dynamic_regulation_nfe
        else:
            return without_dynamic_regulation

    @partial(annotate_function, name="gru_bayes_batch")
    def __gru_bayes(self, params: Any, state: Dict[int, jnp.ndarray],
                    numeric_error: Dict[int, jnp.ndarray],
                    numeric_mask: Dict[int, jnp.ndarray],
                    diag_gram_error: jnp.ndarray) -> jnp.ndarray:
        zero_gram_error = jnp.zeros(self.dimensions['diag_gram'])
        updated_state = {
            i: self.gru_bayes(params['gru_bayes'], h, numeric_error[i],
                              numeric_mask[i],
                              diag_gram_error.get(i, zero_gram_error))
            for i, h in state.items()
        }

        return updated_state

    @partial(annotate_function, name="decode_state_batch")
    def __state_decode(
        self,
        params: Any,
        state: Dict[int, jnp.ndarray],
        numeric_mean: Dict[int, jnp.ndarray],
        subject_batch: Optional[Iterable[int]] = None
    ) -> Tuple[Dict[int, jnp.ndarray], Dict[int, jnp.ndarray]]:
        if subject_batch is None:
            subject_batch = state.keys()

        gram_out = {
            i: self.f_dec(params['f_dec'], state[i], numeric_mean[i])
            for i in subject_batch
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
            for i in diag_true.keys()
        }
        if loss:
            return sum(loss.values())
        else:
            return 0.0

    def __numeric_error(self, mean_true, mean_predicted, logvar_predicted):

        error_num = {
            i: numeric_error(mean, mean_predicted[i], logvar_predicted[i])
            for i, mean in mean_true.items()
        }
        return error_num

    def __lognormal_loss(self, mask: Dict[int, jnp.ndarray],
                         normal_error: Dict[int, jnp.ndarray],
                         logvar: Dict[int, jnp.ndarray]):
        loss = {
            i: lognormal_loss(mask[i], error, logvar[i])
            for i, error in normal_error.items()
        }

        return sum(loss.values())

    def __kl_loss(self, mean_true: Dict[int, jnp.ndarray],
                  mask: Dict[int,
                             jnp.ndarray], mean_predicted: Dict[int,
                                                                jnp.ndarray],
                  logvar_predicted: Dict[int, jnp.ndarray]):
        loss = {
            i: compute_KL_loss(mean, mask[i], mean_predicted[i],
                               logvar_predicted[i])
            for i, mean in mean_true.items()
        }

        return sum(loss.values())

    def __gram_error(self, gram_true, gram_predicted):
        error_gram = {
            i: gram_true[i] - jit_sigmoid(gram_predicted[i])
            for i in gram_true.keys()
        }
        return error_gram

    def __confusion_matrix(self, diag_true, diag_predicted):
        cm = {
            i: confusion_matrix(diag_true[i], jit_sigmoid(diag_predicted[i]))
            for i in diag_true.keys()
        }
        if cm:
            return sum(cm.values())
        else:
            return None

    def __call__(self,
                 params: Any,
                 subjects_batch: List[int],
                 return_path: bool = False,
                 count_nfe: bool = False,
                 iteration_text_callback: Any = None):

        if iteration_text_callback is None:
            iteration_text_callback = lambda _: _

        iteration_text_callback('compute embedding mats')
        diag_emb_mat, proc_emb_mat = self.__generate_embedding_mats(params)

        nth_points_fn = lambda n: self.__extract_nth_points(
            diag_emb_mat, proc_emb_mat, n, subjects_batch,
            iteration_text_callback)  # (n)

        ode_logger.debug(f'subjects: {subjects_batch}')
        nn_f_num: Tf_num = partial(self.__f_num, params)
        dyn_reg = []
        nfe = []
        nn_odeint: Todeint = self.__odeint_wrapper(nfe.append, dyn_reg.append,
                                                   params, count_nfe)
        enc = self.diag_gram.encode

        # Input: (state, numeric_error, numeric_mask, diag_gram_error)
        nn_gru_bayes: Tgru_bayes = partial(self.__gru_bayes, params)
        # (state, numeric)
        nn_state_decode: Tf_state_decode = partial(self.__state_decode, params)

        def initial_state():
            # Possibly not all batch members populated in some dictionaries dictionary.
            # Only subjects that have diagnosis code at the first time point
            # will have corresponding items in:
            # 1. diag_gram_true
            # 2. diag_gram_predicted
            # 3. error_gram

            # (A) Retrieve the first time point for each subject.
            points_0 = nth_points_fn(0)
            diag_gram_true = points_0['diag_gram']
            mean_true = points_0['numeric']
            mask = points_0['mask']

            # (B) Initialize a zero-state before the first time point.
            h0 = jnp.zeros(self.dimensions['state'])
            h0 = {i: h0 for i in subjects_batch}

            # (C) Generate numerical predictions based in the zero-state
            mean_predicted, logvar_predicted = nn_f_num(h0)

            # (D) Use the predicted numerical predictions and the zero state
            # to generate gram predictions.
            diag_gram_predicted, _ = nn_state_decode(h0, mean_predicted)

            # (E) Compute errors between the predicted numerics and the
            # predicted GRAM values.
            error_num = self.__numeric_error(mean_true, mean_predicted,
                                             logvar_predicted)
            error_diag_gram = self.__gram_error(diag_gram_true,
                                                diag_gram_predicted)

            # (F) Update the state given the errors above
            state = nn_gru_bayes(h0, error_num, mask, error_diag_gram)
            return state

        subject_last_state: Dict[int, jnp.ndarray] = initial_state()
        days_fwd = {i: 0 for i in subjects_batch}

        if return_path:
            path: Dict[str, Dict[int, List[Any]]] = {
                'time': defaultdict(list),
                'mean': defaultdict(list),
                'logvar': defaultdict(list),
                'diag_out': defaultdict(list),
                'state': defaultdict(list)
            }

            _mean, _logvar = nn_f_num(subject_last_state)
            _, _diag_logits_out = nn_state_decode(subject_last_state, _mean)
            _diag_out = tree_map(jit_sigmoid, _diag_logits_out)

            for subject_id in subjects_batch:
                path['time'][subject_id].append(0)
                path['mean'][subject_id].append(_mean[subject_id])
                path['logvar'][subject_id].append(_logvar[subject_id])
                path['diag_out'][subject_id].append(_diag_out[subject_id])
                path['state'][subject_id].append(
                    subject_last_state[subject_id])

        prejump_diag_loss = []
        postjump_diag_loss = []
        prejump_num_loss = []
        postjump_num_loss = []
        diag_detectability = []
        num_weights = []
        diag_weights = []
        diag_cm = []  # Confusion matrix
        points_count = 0
        odeint_weeks = 0.0

        for n in self.subject_interface.n_support[1:]:
            iter_prefix = f'point #{n}/{self.subject_interface.n_support[-1]}'
            iteration_text_callback(iter_prefix)
            points_n = nth_points_fn(n)

            if points_n is None:
                continue

            delta_days = {
                i: days_ahead - days_fwd[i]
                for i, days_ahead in points_n['days_ahead'].items()
            }

            days_fwd.update(points_n['days_ahead'])

            numeric = points_n['numeric']
            mask = points_n['mask']
            h0 = {i: subject_last_state[i] for i in numeric.keys()}
            diag_gram = points_n['diag_gram']
            diag_out = points_n['diag_out']

            points_count += len(numeric)
            # No. of tests
            num_weights.append(sum(sum(mask.values())) + 1e-10)
            # No. of diagnostic points
            diag_weights.append(len(diag_gram))
            '''
            Idea: scale days_forward to weeks_forward.
            This can:
                1. Improve the numerical stability and accuracy of numerical integration.
                2. Force the ode_dyn model to learn weekly dynamics, which is a suitable time scale for cancer development.
            '''
            delta_weeks = {i: days / 7.0 for i, days in delta_days.items()}
            odeint_weeks += sum(delta_weeks.values())
            ################## ODEINT #####################
            iteration_text_callback(iter_prefix + ' - odeint')
            h1 = nn_odeint(h0, delta_days, points_n['ode_control'])
            iteration_text_callback(iter_prefix)

            ########## PRE-JUMP NUM LOSS ########################
            iteration_text_callback(iter_prefix + ' - pre_num_loss')
            pre_mean, pre_logvar = nn_f_num(h1)
            error = self.__numeric_error(numeric, pre_mean, pre_logvar)

            prejump_num_loss.append(
                self.__lognormal_loss(mask, error, pre_logvar))

            ########## PRE-JUMP DAG LOSS #########################
            iteration_text_callback(iter_prefix + ' - pre_dag_loss')
            pre_diag_gram, pre_diag_out = nn_state_decode(
                h1, pre_mean, diag_out.keys())
            prejump_diag_loss.append(self.__diag_loss(diag_out, pre_diag_out))

            pre_diag_gram_error = self.__gram_error(diag_gram, pre_diag_gram)

            ############## GRU BAYES ####################
            iteration_text_callback(iter_prefix + ' - gru_bayes')
            # Using GRUObservationCell to update h.

            h2 = nn_gru_bayes(h1, error, mask, pre_diag_gram_error)

            ################ POST-JUMP NUM LOSS ####################
            iteration_text_callback(iter_prefix + ' - num_post_loss')

            post_mean, post_logvar = nn_f_num(h2)
            postjump_num_loss.append(
                self.__kl_loss(numeric, mask, post_mean, post_logvar))

            ############### POST-JUNP DAG LOSS ########################
            iteration_text_callback(iter_prefix + ' - dag_post_loss')
            _, post_diag_out = nn_state_decode(h2, post_mean, diag_out.keys())
            postjump_diag_loss.append(self.__diag_loss(diag_out,
                                                       post_diag_out))

            diag_cm.append(self.__confusion_matrix(diag_out, post_diag_out))

            diag_detectability.append(
                code_detectability_df(20, diag_out, pre_diag_out,
                                      post_diag_out, n))
            # Update person_last_state
            for subject_id in h0.keys():
                subject_last_state[subject_id] = h2[subject_id]

                if return_path:
                    path['time'][subject_id].append(days_fwd[subject_id])
                    path['mean'][subject_id].append(post_mean[subject_id])
                    path['logvar'][subject_id].append(post_logvar[subject_id])
                    path['diag'][subject_id].append(
                        jit_sigmoid(post_diag_out[subject_id]))
                    path['state'][subject_id].append(h2[subject_id])

        prejump_num_loss = jnp.average(prejump_num_loss, weights=num_weights)
        postjump_num_loss = jnp.average(postjump_num_loss, weights=num_weights)
        prejump_diag_loss = jnp.average(prejump_diag_loss,
                                        weights=diag_weights)
        postjump_diag_loss = jnp.average(postjump_diag_loss,
                                         weights=diag_weights)

        confusion_mat = sum(cm for cm in diag_cm if cm is not None)

        ret = {
            'prejump_num_loss': prejump_num_loss,
            'postjump_num_loss': postjump_num_loss,
            'prejump_diag_loss': prejump_diag_loss,
            'postjump_diag_loss': postjump_diag_loss,
            'dyn_reg': jnp.sum(sum(dyn_reg)),
            'scores': confusion_matrix_scores(confusion_mat),
            'odeint_weeks': odeint_weeks,
            'points_count': points_count,
            'nfe': sum(nfe),
            'diag_detectability_df': pd.concat(diag_detectability)
        }
        if return_path:
            ret['path'] = {
                'data': path,
                'dag_out_index': self.diag_out_index(),
                'numeric_index': self.numeric_index()
            }

        return ret


def train_ehr(
        subject_interface: SubjectJAXInterface,
        diag_gram: DAGGRAM,
        proc_gram: DAGGRAM,
        rng: Any,
        # Model configurations
        ode_dyn: str,
        state_size: int,
        numeric_hidden_size: int,
        bias: bool,
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
        ode_dyn=ode_dyn,
        tay_reg=tay_reg,
        state_size=state_size,
        numeric_hidden_size=numeric_hidden_size,
        bias=bias,
        diag_loss=diag_loss_function[diag_loss])

    jax.profiler.save_device_memory_profile("before_params_init.prof")
    params = ode_model.init_params(prng_key)
    jax.profiler.save_device_memory_profile("after_params_init.prof")

    ode_logger.info(f'#params: {parameters_size(params)}')
    ode_logger.debug(f'shape(params): {tree_map(jnp.shape, params)}')
    opt_init, opt_update, get_params = optimizers.adam(step_size=lr)

    def loss_fn_detail(params: optimizers.Params, batch: List[int],
                       iteration_text_callback: Any) -> Dict[str, float]:
        res = ode_model(params,
                        batch,
                        count_nfe=True,
                        iteration_text_callback=iteration_text_callback)

        prejump_num_loss = res['prejump_num_loss'].item()
        postjump_num_loss = res['postjump_num_loss'].item()
        prejump_diag_loss = res['prejump_diag_loss'].item()
        postjump_diag_loss = res['postjump_diag_loss'].item()
        l1_loss = l1_absolute(params).item()
        l2_loss = l2_squared(params).item()
        dyn_loss = res['dyn_reg']
        num_alpha = loss_mixing['num_alpha']
        diag_alpha = loss_mixing['diag_alpha']
        ode_alpha = loss_mixing['ode_alpha']
        l1_alpha = loss_mixing['l1_reg'] / (res['points_count'])
        l2_alpha = loss_mixing['l2_reg'] / (2 * res['points_count'])
        dyn_alpha = loss_mixing['dyn_reg'] / (res['points_count'])

        num_loss = (
            1 - num_alpha) * prejump_num_loss + num_alpha * postjump_num_loss
        diag_loss = (1 - diag_alpha
                     ) * prejump_diag_loss + diag_alpha * postjump_diag_loss
        ode_loss = (1 - ode_alpha) * diag_loss + ode_alpha * num_loss
        loss = ode_loss + (l1_alpha * l1_loss) + (l2_alpha * l2_loss) + (
            dyn_alpha * dyn_loss)
        nfe = res['nfe']
        return {
            'loss': {
                'prejump_num_loss': prejump_num_loss,
                'postjump_num_loss': postjump_num_loss,
                'prejump_diag_loss': prejump_diag_loss,
                'postjump_diag_loss': postjump_diag_loss,
                'num_loss': num_loss,
                'diag_loss': diag_loss,
                'ode_loss': ode_loss,
                'l1_loss': l1_loss,
                'l1_loss_per_point': l1_loss / res['points_count'],
                'l2_loss': l2_loss,
                'l2_loss_per_point': l2_loss / res['points_count'],
                'dyn_loss': dyn_loss,
                'dyn_loss_per_week': dyn_loss / res['odeint_weeks'],
                'loss': loss
            },
            'stats': {
                **{
                    name: score.item()
                    for name, score in res['scores'].items()
                }, 'points_count': res['points_count'],
                'odeint_weeks_per_point':
                res['odeint_weeks'] / res['points_count'],
                'nfe_per_point': nfe / res['points_count'],
                'nfe_per_week': nfe / res['odeint_weeks'],
                'nfex1000': nfe / 1000
            },
            'diag_detectability_df': res['diag_detectability_df']
        }

    def loss_fn(params: optimizers.Params, batch: List[int],
                iteration_text_callback: Any) -> float:
        res = ode_model(params,
                        batch,
                        iteration_text_callback=iteration_text_callback,
                        count_nfe=False)
        iteration_text_callback('')
        prejump_num_loss = res['prejump_num_loss']
        postjump_num_loss = res['postjump_num_loss']
        prejump_diag_loss = res['prejump_diag_loss']
        postjump_diag_loss = res['postjump_diag_loss']
        l1_loss = l1_absolute(params)
        l2_loss = l2_squared(params)
        dyn_loss = res['dyn_reg']
        num_alpha = loss_mixing['num_alpha']
        diag_alpha = loss_mixing['diag_alpha']
        ode_alpha = loss_mixing['ode_alpha']
        l1_alpha = loss_mixing['l1_reg'] / (res['points_count'])
        l2_alpha = loss_mixing['l2_reg'] / (2 * res['points_count'])
        dyn_alpha = loss_mixing['dyn_reg']

        num_loss = (
            1 - num_alpha) * prejump_num_loss + num_alpha * postjump_num_loss
        diag_loss = (1 - diag_alpha
                     ) * prejump_diag_loss + diag_alpha * postjump_diag_loss
        ode_loss = (1 - ode_alpha) * diag_loss + ode_alpha * num_loss

        loss = ode_loss + (l1_alpha * l1_loss) + (l2_alpha * l2_loss) + (
            dyn_alpha * dyn_loss)

        return loss

    def update(step: int, batch: Iterable[int],
               opt_state: optimizers.OptimizerState,
               iteration_text_callback: Any) -> optimizers.OptimizerState:
        params = get_params(opt_state)
        """Single SGD update step."""
        if nan_debug:
            if tree_hasnan(params):
                ode_logger.warning(tree_lognan(params))
                raise ValueError("Nan Params")

        grads = jax.grad(loss_fn)(params, batch, iteration_text_callback)
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

    def update_batch_desc(text):
        val_pbar.set_description(text)

    for step in range(iters):
        rng.shuffle(train_ids)
        train_batch = train_ids[:batch_size]

        val_pbar.update(1)

        try:
            opt_state = update(step, train_batch, opt_state, update_batch_desc)
        except ValueError as e:
            from traceback import format_exception
            tb_str = ''.join(format_exception(None, e, e.__traceback__))
            ode_logger.warning(f'ValueError exception raised: {tb_str}')
            break

        update_batch_desc('')

        if memory_profile and step == 0:
            jax.profiler.save_device_memory_profile("after_first_batch.prof")

        if step % eval_freq == 0:
            rng.shuffle(valid_ids)
            valid_batch = valid_ids  #[:val_batch_size]
            params = get_params(opt_state)
            trn_res = loss_fn_detail(params, train_batch, update_batch_desc)
            val_res = loss_fn_detail(params, valid_batch, update_batch_desc)
            res_trn[step] = trn_res
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
