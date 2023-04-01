"""
Implementation of GRAM embedding technique with other variants.

The implementation in Theano originally provided by
Edward Choi (mp2893@gatech.edu),
while the JAX version of GRAM here is written
by Asem Alaa (asem.a.abdelaziz@gmail.com)
"""

from __future__ import annotations

from functools import partial
from typing import Dict, Optional, TYPE_CHECKING, Callable, List, Union
from abc import ABC, abstractmethod, ABCMeta

import numpy as onp
import jax
import jax.random as jrandom
from jax import lax
import jax.numpy as jnp
import jax.tree_util as jtu

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

    def weights(self):
        has_weight = lambda leaf: hasattr(leaf, 'weight')
        # Valid for eqx.nn.MLP and ml.base_models.GRUDynamics
        return tuple(x.weight
                     for x in jtu.tree_leaves(self, is_leaf=has_weight)
                     if has_weight(x))

    def l1(self):
        return sum(jnp.abs(w).sum() for w in jtu.tree_leaves(self.weights()))

    def l2(self):
        return sum(
            jnp.square(w).sum() for w in jtu.tree_leaves(self.weights()))


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

    def compute_embeddings_mat(self, in_vec):
        G = [None] * self.input_size
        for i in jnp.nonzero(in_vec)[0]:
            G[i] = self.ancestry_attention(self.code_ancestry[i],
                                           self.code_ancestry_mask[i],
                                           self.basic_embeddings[i])
        return tuple(G)

    def encode(self, G: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.tanh(sum(G[i] for i in jnp.nonzero(x)[0]))

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

    def compute_embeddings_mat(self, in_vec):
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
    input_size: int
    output_size: int
    n_layers: int

    def __init__(self, n_layers: int, input_size: int, output_size: int,
                 key: "jax.random.PRNGKey"):
        self.f_dec = eqx.nn.MLP(in_size=input_size,
                                out_size=output_size,
                                width_size=input_size,
                                depth=n_layers - 1,
                                activation=jax.nn.leaky_relu,
                                key=key)
        self.n_layers = n_layers
        self.input_size = input_size
        self.output_size = output_size

    def __call__(self, logits: jnp.ndarray):
        return self.f_dec(logits)

    def weights(self):
        has_weight = lambda leaf: hasattr(leaf, 'weight')
        # Valid for eqx.nn.MLP and ml.base_models.GRUDynamics
        return tuple(x.weight
                     for x in jtu.tree_leaves(self, is_leaf=has_weight)
                     if has_weight(x))

    def l1(self):
        return sum(jnp.abs(w).sum() for w in jtu.tree_leaves(self.weights()))

    def l2(self):
        return sum(
            jnp.square(w).sum() for w in jtu.tree_leaves(self.weights()))


def create_embeddings_model(code_type: str, emb_conf: Dict[str, Union[str, int,
                                                                      float]],
                            subject_interface: Subject_JAX,
                            train_ids: List[int]):

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

    if classname == 'GRAM':
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
        return GRAM(basic_embeddings=glove,
                    attention_method=emb_conf['attention_method'],
                    attention_size=emb_conf['attention_size'],
                    ancestors_mat=ancestors_mat,
                    key=jrandom.PRNGKey(0))
    else:
        raise RuntimeError(f'Unrecognized Embedding class {classname}')


def embeddings_from_conf(conf: Dict[str, Union[str, int, float]],
                         subject_interface: Subject_JAX, train_ids: List[int],
                         decoder_input_size: int):
    models = {}
    if conf.get('dx'):
        models['dx_emb'] = create_embeddings_model('dx', conf['dx'],
                                                   subject_interface,
                                                   train_ids)
        dec_n_layers = conf['dx']['decoder_n_layers']
        dec_output_size = subject_interface.outcome_dim
        models['dx_dec'] = LogitsDecoder(n_layers=dec_n_layers,
                                         input_size=decoder_input_size,
                                         output_size=dec_output_size,
                                         key=jrandom.PRNGKey(0))

    if conf.get('pr'):
        models['pr_emb'] = create_embeddings_model('pr', conf['pr'],
                                                   subject_interface,
                                                   train_ids)

    return models
