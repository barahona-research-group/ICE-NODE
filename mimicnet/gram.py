#################################################################
# Implementation in Theano originally provided by
# Edward Choi (mp2893@gatech.edu)
# The JAX version of GRAM is written by Asem Alaa (asem.a.abdelaziz@gmail.com)
# For bug report, please contact author using the email address
#################################################################
from __future__ import annotations

import abc
from concurrent import futures
import logging
import math
from collections import defaultdict, namedtuple
from enum import Enum, auto, unique
from functools import partial, reduce
from typing import (AbstractSet, Set, Any, Dict, Iterable, List, Mapping,
                    Optional, Tuple, Union, Callable)

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import lax, vmap
from jax.profiler import annotate_function
from jax.nn import softmax
from jax.tree_util import tree_map
from .concept import Subject
from .jax_interface import SubjectJAXInterface
from .dag import CCSDAG


def unnormalized_softmax(x, axis=-1):
    return jnp.exp(x - lax.stop_gradient(x.max(axis, keepdims=True)))


# lax.cond is so problematic with memory!!!


class HashableArrayWrapper:
    def __init__(self, val):
        assert isinstance(val, jnp.ndarray), f"Not supported :{type(val)}."
        self.val = val

    def sync(self):
        val = self.val.block_until_ready()
        return self

    def __hash__(self):
        return int(jnp.sum(self.val))

    def __eq__(self, other):
        return (isinstance(other, HashableArrayWrapper)
                and jnp.all(jnp.equal(self.val, other.val)))


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


class DAGGRAM:
    def __init__(self,
                 attention_dim: int,
                 attention_method: str,
                 code2index: Dict[str, int],
                 ancestors_mat: jnp.ndarray,
                 basic_embeddings: Mapping[str, jnp.ndarray],
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
        self.ancestors_mat = ancestors_mat
        self.basic_embeddings_mat = jnp.vstack(
            map(basic_embeddings.get, codes_ordered))
        self.basic_embeddings_dim = self.basic_embeddings_mat.shape[1]
        self.init_att, self.fwd_att = hk.without_apply_rng(
            hk.transform(
                wrap_module(attention_cls,
                            attention_dim=attention_dim,
                            name=f"{name}_DAG_Attention")))

    def init_params(self, rng_key):
        e = self.basic_embeddings_mat[0, :]
        return self.basic_embeddings_mat, self.init_att(rng_key, e, e)

    @staticmethod
    def __self_attention(f: Callable[[jnp.ndarray, jnp.ndarray],
                                     jnp.ndarray], params: Any, E: jnp.ndarray,
                         e_i: jnp.ndarray, ancestors_vec: jnp.ndarray):
        # |S|: number of codes.
        # m: size of embedding vector

        # (|S|, )
        comp_v = vmap(partial(f, params, e_i))(E) * ancestors_vec
        # (m, )
        return jnp.average(E, axis=0, weights=unnormalized_softmax(comp_v))

    @partial(annotate_function, name="compute_embedding_mat")
    @partial(jax.jit, static_argnums=(0, ))
    def compute_embedding_mat(self, params):
        self_attend = DAGGRAM.__self_attention
        E, att_params = params
        A = self.ancestors_mat
        return vmap(partial(self_attend, self.fwd_att, att_params, E))(E, A)

    @partial(annotate_function, name="gram_encode")
    @partial(jax.jit, static_argnums=(0, ))
    def encode(self, embedding_mat: jnp.ndarray,
               dag_vec: jnp.ndarray) -> List[jnp.ndarray]:
        return jnp.matmul(dag_vec, embedding_mat)
