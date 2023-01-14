"""
Implementation of GRAM embedding technique with other variants.

The implementation in Theano originally provided by
Edward Choi (mp2893@gatech.edu),
while the JAX version of GRAM here is written
by Asem Alaa (asem.a.abdelaziz@gmail.com)
"""

from __future__ import annotations

import importlib
from functools import partial
from typing import Any, Dict, Iterable, Optional, Tuple, TYPE_CHECKING, Callable, List, Union
from abc import ABC, abstractmethod, ABCMeta

import numpy as onp
import jax
import jax.random as jrandom
from jax import lax
import jax.numpy as jnp

import equinox as eqx

if TYPE_CHECKING:
    import optuna

from .glove import train_glove
from ..ehr import Subject_JAX


def unnormalized_softmax(x, axis=-1):
    return jnp.exp(x - lax.stop_gradient(x.max(axis, keepdims=True)))


class DAGAttention(eqx.Module):

    project: Callable
    weighted_sum: Callable

    def __init__(self, embeddings_size: int, attention_size: int,
                 key: "jax.random.PRNGKey"):
        key1, key2 = jrandom.split(key, 2)
        self.project = eqx.nn.Linear(embeddings_size * 2,
                                     attention_size,
                                     use_bias=True,
                                     key=key1)
        self.weighted_sum = eqx.nn.Linear(attention_size,
                                          1,
                                          use_bias=False,
                                          key=key2)

    def __call__(self, ei: jnp.ndarray, ej: jnp.ndarray) -> jnp.ndarray:
        ei_ej = jnp.hstack((ei, ej))
        return self.weighted_sum(jnp.tanh(self.project(ei_ej))).squeeze()


class DAGL2Attention(eqx.Module):
    """
    The Lipschitz Constant of Self-Attention:
    https://arxiv.org/abs/2006.04710
    """

    project: Callable

    def __init__(self, embeddings_size, attention_size,
                 key: "jax.random.PRNGKey"):
        self.project = eqx.nn.Linear(embeddings_size,
                                     attention_size,
                                     use_bias=False,
                                     key=key)

    def __call__(self, ei: jnp.ndarray, ej: jnp.ndarray) -> jnp.ndarray:
        den = jnp.sqrt(jnp.size(ei))
        diff = self.project(ei - ej)
        return -jnp.dot(diff, diff) / den


class AbstractEmbeddingsLayer(eqx.Module, metaclass=ABCMeta):
    embeddings_size: int
    input_size: int

    embedding_cls = {}
    short_tag = {}

    @abstractmethod
    def compute_embeddings_mat(self):
        pass

    @abstractmethod
    def encode(self, x: jnp.ndarray) -> jnp.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def sample_model_config(prefix: str, trial: optuna.Trial):
        pass

    @classmethod
    def register_embedding(cls, label, short):
        cls.embedding_cls[label] = cls
        cls.short_tag[label] = short


class GRAM(AbstractEmbeddingsLayer):
    f_att: Callable
    basic_embeddings: jnp.ndarray

    code_ancestry: jnp.ndarray = eqx.static_field()
    code_ancestry_mask: jnp.ndarray = eqx.static_field()
    index: jnp.ndarray = eqx.static_field()

    def __init__(self, basic_embeddings: jnp.ndarray, attention_size: int,
                 attention_method: str, ancestors_mat: jnp.ndarray,
                 key: "jax.random.PRNGKey"):
        super().__init__(input_size=len(basic_embeddings),
                         embeddings_size=basic_embeddings.shape[1])
        self.basic_embeddings = jnp.array(basic_embeddings)

        if attention_method == 'tanh':
            attention_cls = DAGAttention
        elif attention_method == 'l2':
            attention_cls = DAGL2Attention
        else:
            raise RuntimeError('Unrecognized attention method.'
                               f'Got {attention_method}. Expected either of'
                               f'{", ".join(["tanh", "l2"])}.')

        self.code_ancestry, self.code_ancestry_mask = self._code_ancestry_mat(
            ancestors_mat)
        self.index = jnp.arange(len(ancestors_mat))
        self.f_att = attention_cls(embeddings_size=basic_embeddings.shape[1],
                                   attention_size=attention_size,
                                   key=key)

    @staticmethod
    def _code_ancestry_mat(ancestors_mat: jnp.ndarray):
        """
        Args:
            ancestors_mat: Adjecency matrix.

        This function returns an array, where each row correspond to one code.
        Each row includes the ancestor indices of the corresponding code,
        in addition to the index of the code itself.
        As codes have different number of ancestors, each row has length of
        (self.max_ancestors + 1), so if the branch elements doesn't fill the
        entire row, the row is padded with the code index repetitions.
        """
        # Work on a copy, and fill diagonal (nodes themselves are retrieved
        # when their ancestors are queried).

        ancestors_mat = onp.array(ancestors_mat.copy())
        onp.fill_diagonal(ancestors_mat, 1)

        max_ancestors = ancestors_mat.sum(axis=1).max()
        code_ancestry = []
        code_ancestry_mask = []
        default_mask = onp.ones(max_ancestors)

        for i, ancestors_i in enumerate(ancestors_mat):

            # indices of ancestors, including i itself.
            ancestors_i = onp.nonzero(ancestors_i)[0]
            mask = default_mask.copy()
            mask[len(ancestors_i):] = 0.0
            code_ancestry_mask.append(mask)

            residual_size = max_ancestors - len(ancestors_i)
            assert residual_size >= 0, "Unexpected."

            if residual_size > 0:
                padding = onp.zeros(residual_size, dtype=int) + i
                code_ancestry.append(onp.hstack((ancestors_i, padding)))
            else:
                code_ancestry.append(ancestors_i)

        return jnp.vstack(code_ancestry), jnp.vstack(code_ancestry_mask)

    @eqx.filter_jit
    def ancestry_attention(self, ancestry: jnp.ndarray,
                           ancestry_mask: jnp.ndarray, e_i: jnp.ndarray):
        # E_i: basic embeddings of i-th node ancestors
        E_i = self.basic_embeddings.at[ancestry].get()

        # Ancestry mask will zero-out the padded embeddings.
        A_att = jax.vmap(partial(self.f_att, e_i))(E_i) * ancestry_mask
        return jnp.average(E_i, axis=0, weights=unnormalized_softmax(A_att))

    @eqx.filter_jit
    def compute_embeddings_mat(self):
        return jax.vmap(self.ancestry_attention)(self.code_ancestry,
                                                 self.code_ancestry_mask,
                                                 self.basic_embeddings)

    @eqx.filter_jit
    def encode(self, G: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.tanh(jnp.matmul(x, G))

    @staticmethod
    def sample_model_config(prefix: str, trial: optuna.Trial):
        return {
            'glove_config': {
                'iterations': 30,
                'window_size_days': 2 * 365
            },
            'embeddings_size':
            trial.suggest_int(f'{prefix}_k', 30, 300, 30),
            'attention_method':
            trial.suggest_categorical(f'{prefix}_att_f', ['tanh', 'l2']),
            'attention_size':
            trial.suggest_int(f'{prefix}_att_d', 30, 300, 30),
        }


class CachedEmbeddingsMatrix(dict):
    gram: GRAM

    def __init__(self, gram):
        self.gram = gram

    def multiply(self, x: jnp.ndarray):
        index = onp.nonzero(x)[0]
        if len(index) == 0:
            return jnp.zeros_like(self.gram.basic_embeddings[0])
        return sum(self[i.item()] for i in index)

    def __getitem__(self, idx):
        if idx in self:
            return super().__getitem__(idx)
        else:
            gi = self.gram.ancestry_attention(
                self.gram.code_ancestry[idx],
                self.gram.code_ancestry_mask[idx],
                self.gram.basic_embeddings[idx])
            super().__setitem__(idx, gi)
            return gi

    def get(self, k, default=None):
        if k in self:
            return self.__getitem__(k)
        return default


class CachedGRAM(GRAM):

    def compute_embeddings_mat(self):
        return CachedEmbeddingsMatrix(self)

    def encode(self, G: CachedEmbeddingsMatrix, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.tanh(G.multiply(x))


class MatrixEmbeddings(AbstractEmbeddingsLayer):
    linear: Callable

    def __init__(self, embeddings_size: int, input_size: int,
                 key: "jax.random.PRNGKey"):
        super().__init__(embeddings_size=embeddings_size,
                         input_size=input_size)
        self.linear = eqx.nn.Linear(input_size,
                                    embeddings_size,
                                    use_bias=True,
                                    key=key)

    def compute_embeddings_mat(self, params):
        return self

    @eqx.filter_jit
    def encode(self, G: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        return self.linear(x)

    @staticmethod
    def sample_model_config(prefix: str, trial: optuna.Trial):
        return {
            'embeddings_size': trial.suggest_int(f'{prefix}_k', 30, 300, 30)
        }


class LogitsDecoder(eqx.Module):
    f_dec: Callable
    embeddings_size: int
    output_size: int

    def __init__(self, n_layers: int, embeddings_size: int, output_size: int,
                 key: "jax.random.PRNGKey"):

        def _act(index):
            if index < n_layers - 2:
                return jax.nn.leaky_relu
            else:
                return jnp.tanh

        layers = []
        keys = jrandom.split(key, n_layers)
        for i in range(n_layers):
            out_size = embeddings_size if i != n_layers - 1 else output_size
            layers.append(
                eqx.nn.Linear(embeddings_size,
                              out_size,
                              use_bias=True,
                              key=keys[i]))
            layers.append(eqx.nn.Lambda(_act(i)))

        self.f_dec = eqx.nn.Sequential(layers[:-1])

    def __call__(self, logits: jnp.ndarray):
        return self.f_dec(logits)


def create_embeddings_model(code_type: str,
                            emb_conf: Dict[str, Union[str, int, float]],
                            subject_interface: Subject_JAX,
                            train_ids: Optional[List[int]] = None):

    classname = emb_conf['classname']
    embeddings_size = emb_conf['embeddings_size']

    if classname == 'MatrixEmbeddings':
        if code_type == 'dx':
            input_size = subject_interface.dx_dim
        else:
            input_size = subject_interface.pr_dim

        return MatrixEmbeddings(embeddings_size=embeddings_size,
                                input_size=input_size,
                                key=jrandom.PRNGKey(0))

    if classname in ['GRAM', 'CachedGRAM']:
        emb_cls = globals()[classname]

        if code_type == 'dx':
            cooc_f = subject_interface.dx_augmented_coocurrence
            ancestors_mat = subject_interface.dx_make_ancestors_mat()
        else:
            cooc_f = subject_interface.pr_augmented_coocurrence
            ancestors_mat = subject_interface.pr_make_ancestors_mat()

        win_size = emb_conf['cooc_window_size_days']

        coocurrence_mat = cooc_f(train_ids, window_size_days=win_size)
        glove = train_glove(cooccurrences=coocurrence_mat,
                            embeddings_size=embeddings_size,
                            iterations=emb_conf['glove_iterations'])
        return emb_cls(basic_embeddings=glove,
                       attention_method=emb_conf['attention_method'],
                       attention_size=emb_conf['attention_size'],
                       ancestors_mat=ancestors_mat,
                       key=jrandom.PRNGKey(0))
    else:
        raise RuntimeError(f'Unrecognized Embedding class {classname}')


def embeddings_from_conf(conf: Dict[str, Union[str, int, float]],
                         subject_interface: Subject_JAX,
                         train_ids: Optional[List[int]] = None):
    models = {}
    if conf.get('dx'):
        models['dx_emb'] = create_embeddings_model('dx', conf['dx'],
                                                   subject_interface,
                                                   train_ids)
        TODO: DECODER ADD

    if conf.get('pr'):
        models['pr_emb'] = create_embeddings_model('pr', conf['pr'],
                                                   subject_interface,
                                                   train_ids)

    return models


MatrixEmbeddings.register_embedding('matrix', 'M')
GRAM.register_embedding('gram', 'G')
