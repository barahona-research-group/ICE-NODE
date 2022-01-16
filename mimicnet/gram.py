#################################################################
# Implementation in Theano originally provided by
# Edward Choi (mp2893@gatech.edu)
# The JAX version of GRAM is written by Asem Alaa (asem.a.abdelaziz@gmail.com)
# For bug report, please contact author using the email address
#################################################################
from __future__ import annotations

from functools import partial
from typing import (Any, Dict, Iterable, Optional, Tuple)

import numpy as onp
import jax
from jax import lax
import jax.numpy as jnp

import haiku as hk
import optuna

from .utils import wrap_module


def unnormalized_softmax(x, axis=-1):
    return jnp.exp(x - lax.stop_gradient(x.max(axis, keepdims=True)))


class DAGAttention(hk.Module):
    def __init__(self,
                 attention_dim,
                 name: Optional[str] = None,
                 **init_kwargs):
        super().__init__(name=name)
        self.attention_dim = attention_dim
        self.linear = hk.Linear(attention_dim, with_bias=True)
        self.weight = hk.Linear(1, with_bias=False)

    def __call__(self, ei: jnp.ndarray, ej: jnp.ndarray) -> jnp.ndarray:
        ei_ej = jnp.hstack((ei, ej))
        return self.weight(jnp.tanh(self.linear(ei_ej))).squeeze()


class DAGL2Attention(hk.Module):
    """
    The Lipschitz Constant of Self-Attention:
    https://arxiv.org/abs/2006.04710
    """
    def __init__(self,
                 attention_dim,
                 name: Optional[str] = None,
                 **init_kwargs):
        super().__init__(name=name)
        self.attention_dim = attention_dim
        self.linear = hk.Linear(attention_dim, with_bias=False)

    def __call__(self, ei: jnp.ndarray, ej: jnp.ndarray) -> jnp.ndarray:
        den = jnp.sqrt(jnp.size(ei))
        diff = self.linear(ei - ej)
        return -jnp.dot(diff, diff) / den


class DAGGRAM:
    def __init__(self,
                 attention_dim: int,
                 attention_method: str,
                 code2index: Dict[str, int],
                 ancestors_mat: jnp.ndarray,
                 basic_embeddings: Dict[str, jnp.ndarray],
                 frozen_params: Optional[Any] = None,
                 name: Optional[str] = None,
                 **init_kwargs):
        if attention_method == 'tanh':
            attention_cls = DAGAttention
        elif attention_method == 'l2':
            attention_cls = DAGL2Attention
        else:
            raise RuntimeError('Unrecognized attention method.'
                               f'Got {attention_method}. Expected either of'
                               f'{", ".join(["tanh", "l2"])}.')
        index2code = {i: c for c, i in code2index.items()}
        codes_ordered = map(index2code.get, range(len(index2code)))

        self.code2index = code2index
        self.A = self._augment_ancestors_mat(ancestors_mat)

        self.initial_E = jnp.vstack(map(basic_embeddings.get, codes_ordered))
        self.basic_embeddings_dim = self.initial_E.shape[1]

        self.init_att, fwd_att = hk.without_apply_rng(
            hk.transform(
                wrap_module(attention_cls,
                            attention_dim=attention_dim,
                            name=f"{name}_DAG_Attention")))
        self.fwd_att = jax.jit(fwd_att)
        self.frozen_params = frozen_params

    @staticmethod
    def _augment_ancestors_mat(ancestors_mat: jnp.ndarray):
        '''Include the code itself in the set of its Ancestors'''
        A = onp.array(ancestors_mat)
        onp.fill_diagonal(A, 1)

        return [jnp.nonzero(ancestors_v) for ancestors_v in A]

    def init_params(self, rng_key):
        if self.frozen_params:
            return None

        e = self.initial_E[0, :]
        return self.initial_E, self.init_att(rng_key, e, e)

    @partial(jax.jit, static_argnums=(0, ))
    def _self_attention(self, params: Any, E: jnp.ndarray, e_i: jnp.ndarray,
                        ancestors_mask: Tuple[jnp.ndarray]):
        if self.frozen_params:
            params = self.frozen_params

        E = E[ancestors_mask]
        A_att = jax.vmap(partial(self.fwd_att, params, e_i))(E)
        return jnp.average(E, axis=0, weights=unnormalized_softmax(A_att))

    def compute_embedding_mat(self, params):
        if self.frozen_params:
            params = self.frozen_params

        E, att_params = params
        G = map(partial(self._self_attention, att_params, E), E, self.A)
        return jnp.vstack(list(G))

    @partial(jax.jit, static_argnums=(0, ))
    def encode(self, G: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.tanh(jnp.matmul(x, G))

    @staticmethod
    def sample_model_config(prefix: str, trial: optuna.Trial):
        return {
            'attention_method':
            trial.suggest_categorical(f'{prefix}_att_f', ['tanh', 'l2']),
            'attention_dim':
            trial.suggest_int(f'{prefix}_att_d', 50, 250, 50),
        }
