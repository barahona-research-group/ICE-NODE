"""
Implementation of GRAM embedding technique with other variants.

The implementation in Theano originally provided by
Edward Choi (mp2893@gatech.edu),
while the JAX version of GRAM here is written
by Asem Alaa (asem.a.abdelaziz@gmail.com)
"""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, Iterable, Optional, Tuple, TYPE_CHECKING

import numpy as onp
import jax
from jax import lax
import jax.numpy as jnp

import haiku as hk

if TYPE_CHECKING:
    import optuna

from ..utils import wrap_module
from .. import ehr
from .glove import glove_representation


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


class AbstractEmbeddingsLayer:

    embedding_cls = {}
    short_tag = {}

    def __init__(self, embeddings_dim):
        self.embeddings_dim = embeddings_dim

    def init_params(self, prng_key):
        raise RuntimeError('Should be overriden')

    def compute_embeddings_mat(self, params):
        raise RuntimeError('Should be overriden')

    def encode(self, params, x: jnp.ndarray) -> jnp.ndarray:
        raise RuntimeError('Should be overriden')

    @staticmethod
    def sample_model_config(prefix: str, trial: optuna.Trial):
        raise RuntimeError('Should be overriden')

    @classmethod
    def register_embedding(cls, label, short):
        cls.embedding_cls[label] = cls
        cls.short_tag[label] = short


class AbstractGRAM(AbstractEmbeddingsLayer):

    def __init__(self,
                 attention_dim: int,
                 attention_method: str,
                 ancestors_mat: jnp.ndarray,
                 embeddings_dim: int,
                 name: Optional[str] = None,
                 **init_kwargs):
        super().__init__(embeddings_dim=embeddings_dim)

        if attention_method == 'tanh':
            attention_cls = DAGAttention
        elif attention_method == 'l2':
            attention_cls = DAGL2Attention
        else:
            raise RuntimeError('Unrecognized attention method.'
                               f'Got {attention_method}. Expected either of'
                               f'{", ".join(["tanh", "l2"])}.')

        self.code_ancestry = self._code_ancestry_mat(ancestors_mat)
        self.index = jnp.arange(len(ancestors_mat))

        self.init_att, fwd_att = hk.without_apply_rng(
            hk.transform(
                wrap_module(attention_cls,
                            attention_dim=attention_dim,
                            name=f"{name}_DAG_Attention")))
        self.fwd_att = jax.jit(fwd_att)

    @staticmethod
    def _code_ancestry_mat(ancestors_mat: jnp.ndarray):
        """
        This function returns an array, where each row correspond to one code.
        Each row includes the ancestor indices of the corresponding code,
        in addition to the index of the code itself.
        As codes have different number of ancestors, each row has length of
        (self.max_ancestors + 1), so if the branch elements doesn't fill the
        entire row, the row is padded with the code index repetitions.
        """
        max_ancestors = ancestors_mat.sum(axis=1).max()
        code_ancestry = []
        for i, ancestors_i in enumerate(ancestors_mat):
            ancestors_i = onp.nonzero(ancestors_i)[0]
            fill_size = max_ancestors + 1 - len(ancestors_i)
            fill = onp.zeros(fill_size, dtype=int) + i
            code_ancestry.append(onp.hstack((ancestors_i, fill)))

        return jnp.vstack(code_ancestry)

    def init_params(self, rng_key):
        raise RuntimeError(f'Should be overriden')

    @partial(jax.jit, static_argnums=(0, ))
    def _self_attention(self, params: Any, E: jnp.ndarray,
                        ancestry: jnp.ndarray, e_i: jnp.ndarray):
        # E: basic embeddings of ancestors
        E = E.at[ancestry].get()
        A_att = jax.vmap(partial(self.fwd_att, params, e_i))(E)
        return jnp.average(E, axis=0, weights=unnormalized_softmax(A_att))

    @partial(jax.jit, static_argnums=(0, ))
    def _compute_embeddings_mat(self, params):
        E, att_params = params
        return jax.vmap(partial(self._self_attention, att_params,
                                E))(self.code_ancestry, E)

    # This can be overriden.
    def compute_embeddings_mat(self, params):
        return self._compute_embeddings_mat(params)

    @partial(jax.jit, static_argnums=(0, ))
    def encode(self, G: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.tanh(jnp.matmul(x, G))

    @staticmethod
    def sample_model_config(prefix: str, trial: optuna.Trial):
        raise RuntimeError('Should be overriden')


class GRAM(AbstractGRAM):

    def __init__(self,
                 category: str,
                 subject_interface: ehr.Subject_JAX,
                 train_ids: Iterable[int],
                 glove_config: Dict[str, int],
                 attention_dim: int,
                 attention_method: str,
                 embeddings_dim: int,
                 name: Optional[str] = None,
                 **init_kwargs):

        if category == 'dx':
            ancestors_mat = subject_interface.dx_make_ancestors_mat()
        else:
            ancestors_mat = subject_interface.pr_make_ancestors_mat()
        super().__init__(attention_dim=attention_dim,
                         attention_method=attention_method,
                         ancestors_mat=ancestors_mat,
                         embeddings_dim=embeddings_dim,
                         name=name)

        self.subject_interface = subject_interface
        self.train_ids = train_ids
        self.category = category
        self.glove_config = glove_config

    def init_params(self, rng_key):
        glove_E = glove_representation(
            category=self.category,
            subject_interface=self.subject_interface,
            train_ids=self.train_ids,
            vector_size=self.embeddings_dim,
            **self.glove_config)

        if self.category == 'dx':
            t_scheme = ehr.code_scheme[self.subject_interface.dx_scheme]
        else:
            t_scheme = ehr.code_scheme[self.subject_interface.pr_scheme]

        index2code = {i: c for c, i in t_scheme.dag_index.items()}
        codes_ordered = list(index2code[i] for i in range(len(index2code)))
        initial_E = jnp.vstack([glove_E[c] for c in codes_ordered])

        e = initial_E[0, :]
        return initial_E, self.init_att(rng_key, e, e)

    @staticmethod
    def sample_model_config(prefix: str, trial: optuna.Trial):
        return {
            'glove_config': {
                'iterations': 30,
                'window_size_days': 2 * 365
            },
            'embeddings_dim':
            trial.suggest_int(f'{prefix}_k', 30, 300, 30),
            'attention_method':
            trial.suggest_categorical(f'{prefix}_att_f', ['tanh', 'l2']),
            'attention_dim':
            trial.suggest_int(f'{prefix}_att_d', 30, 300, 30),
        }


class CachedEmbeddingsMatrix(dict):

    def __init__(self, params, code_ancestry, att_f):
        self.E, self.att_params = params
        self.code_ancestry = code_ancestry
        self.att_f = att_f

    def multiply(self, x: jnp.ndarray):
        index = onp.nonzero(x)[0]
        if len(index) == 0:
            return jnp.zeros_like(self.E[0])
        return sum(self[i.item()] for i in index)

    def __getitem__(self, idx):
        if idx in self:
            return super().__getitem__(idx)
        else:
            gi = self.att_f(self.att_params, self.E, self.code_ancestry[idx],
                            self.E[idx])
            super().__setitem__(idx, gi)
            return gi

    def get(self, k, default=None):
        if k in self:
            return self.__getitem__(k)
        return default


class CachedGRAM(GRAM):

    def compute_embeddings_mat(self, params):
        return CachedEmbeddingsMatrix(params,
                                      code_ancestry=self.code_ancestry,
                                      att_f=self._self_attention)

    def encode(self, G: CachedEmbeddingsMatrix, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.tanh(G.multiply(x))


class MatrixEmbeddings(AbstractEmbeddingsLayer):

    def __init__(self,
                 embeddings_dim: int,
                 input_dim: int,
                 name: Optional[str] = None):
        super().__init__(embeddings_dim=embeddings_dim)

        self.init_emb, fwd_emb = hk.without_apply_rng(
            hk.transform(
                wrap_module(hk.Linear, output_size=embeddings_dim, name=name)))

        self.fwd_emb = jax.jit(fwd_emb)
        self.input_dim = input_dim

    def init_params(self, prng_key):
        return self.init_emb(prng_key, jnp.zeros((self.input_dim, )))

    def compute_embeddings_mat(self, params):
        return params

    @partial(jax.jit, static_argnums=(0, ))
    def encode(self, params, x: jnp.ndarray) -> jnp.ndarray:
        return self.fwd_emb(params, x)

    @staticmethod
    def sample_model_config(prefix: str, trial: optuna.Trial):
        return {
            'embeddings_dim': trial.suggest_int(f'{prefix}_k', 30, 300, 30)
        }


MatrixEmbeddings.register_embedding('matrix', 'M')
GRAM.register_embedding('gram', 'G')
