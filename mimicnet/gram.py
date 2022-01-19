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
                 embeddings_dim: int,
                 initial_params: Optional[Any],
                 mode: str,
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
        self.embeddings_dim = embeddings_dim

        self.init_att, fwd_att = hk.without_apply_rng(
            hk.transform(
                wrap_module(attention_cls,
                            attention_dim=attention_dim,
                            name=f"{name}_DAG_Attention")))
        self.fwd_att = jax.jit(fwd_att)
        self.mode = mode
        if mode == 'frozen_params':
            self.frozen_params = initial_params
            self.G = self.compute_embedding_mat(initial_params)
        elif mode == 'tunable_params':
            self.tunable_params = initial_params
        elif mode == 'semi_frozen':
            self.E, self.att_params = initial_params
        elif mode == 'initial_embeddings':
            self.initial_E = jnp.vstack(map(initial_params.get, codes_ordered))
        else:
            raise RuntimeError(f'Unrecognized mode: {mode}')

    @staticmethod
    def _augment_ancestors_mat(ancestors_mat: jnp.ndarray):
        '''Include the code itself in the set of its Ancestors'''
        A = onp.array(ancestors_mat)
        onp.fill_diagonal(A, 1)

        return [jnp.nonzero(ancestors_v) for ancestors_v in A]

    def init_params(self, rng_key):
        if self.mode == 'frozen_params':
            return None
        elif self.mode == 'tunable_params':
            return self.tunable_params
        elif self.mode == 'semi_frozen':
            return self.att_params
        elif self.mode == 'initial_embeddings':
            e = self.initial_E[0, :]
            return self.initial_E, self.init_att(rng_key, e, e)
        else:
            raise RuntimeError(f'Unrecognized mode: {self.mode}')

    @partial(jax.jit, static_argnums=(0, ))
    def _self_attention(self, params: Any, E: jnp.ndarray, e_i: jnp.ndarray,
                        ancestors_mask: Tuple[jnp.ndarray]):
        E = E[ancestors_mask]
        A_att = jax.vmap(partial(self.fwd_att, params, e_i))(E)
        return jnp.average(E, axis=0, weights=unnormalized_softmax(A_att))

    def compute_embedding_mat(self, params):
        if self.mode == 'frozen_params' and params is None:
            return self.G
        elif self.mode == 'semi_frozen':
            E, att_params = self.E, params
        else:
            E, att_params = params

        G = map(partial(self._self_attention, att_params, E), E, self.A)
        return jnp.vstack(list(G))

    @partial(jax.jit, static_argnums=(0, ))
    def encode(self, G: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.tanh(jnp.matmul(x, G))

    @staticmethod
    def sample_model_config(prefix: str, trial: optuna.Trial):
        return {
            'embeddings_dim':
            trial.suggest_int('dx', 50, 250, 50),
            'attention_method':
            trial.suggest_categorical(f'{prefix}_att_f', ['tanh', 'l2']),
            'attention_dim':
            trial.suggest_int(f'{prefix}_att_d', 50, 250, 50),
        }
