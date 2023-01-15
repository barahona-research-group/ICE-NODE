"""Abstract class for predictive EHR models."""

from __future__ import annotations
from typing import List, Any, TYPE_CHECKING, Callable, Union, Tuple, Optional
from abc import ABC, abstractmethod, ABCMeta

import jax.numpy as jnp
import equinox as eqx

if TYPE_CHECKING:
    import optuna

from ..ehr import Subject_JAX
from .. import metric


class AbstractModel(eqx.Module, metaclass=ABCMeta):
    dx_emb: Callable
    dx_dec: Callable

    state_size: Union[int, Tuple[int, ...]]
    control_size: int = 0

    @abstractmethod
    def __call__(self, subject_interface: Subject_JAX,
                 subjects_batch: List[int], args):
        pass

    def subject_embeddings(self, subject_interface: Subject_JAX,
                           batch: List[int]):
        out = self(subject_interface, batch, dict(return_embeddings=True))
        return {
            i: out['risk_prediction'].get_subject_embeddings(i)
            for i in batch
        }

    @staticmethod
    def _emb_subtrees(pytree):
        return (pytree.dx_emb, pytree.dx_dec)

    @staticmethod
    def emb_dyn_partition(pytree):
        """
        Separate the dynamics parameters from the embedding parameters.
        Thanks to Patrick Kidger for the clever function of eqx.partition.
        """
        emb_nodes = AbstractModel._emb_subtrees(pytree)
        emb_predicate = lambda _t: any(_t is t for t in emb_nodes)

        emb_tree, dyn_tree = eqx.partition(pytree,
                                           emb_predicate,
                                           is_leaf=emb_predicate)
        return emb_tree, dyn_tree

    @staticmethod
    def emb_dyn_merge(emb_tree, dyn_tree):
        return eqx.combine(emb_tree, dyn_tree)

    @staticmethod
    def dx_outcome_partitions(subject_interface: Subject_JAX,
                              train_ids: List[int]):
        return subject_interface.dx_outcome_by_percentiles(20, train_ids)

    @classmethod
    def select_loss(cls, loss_label: str, subject_interface: Subject_JAX,
                    train_ids: List[int], dx_scheme: str):
        if loss_label == 'balanced_focal':
            return lambda t, p: metric.balanced_focal_bce(
                t, p, gamma=2, beta=0.999)
        elif loss_label == 'softmax_logits_bce':
            return metric.softmax_logits_bce
        elif loss_label == 'bce':
            return metric.bce
        elif loss_label == 'balanced_bce':
            codes_dist = subject_interface.dx_code_frequency_vec(
                train_ids, dx_scheme)
            weights = codes_dist.sum() / (codes_dist + 1e-1) * len(codes_dist)
            return lambda t, logits: metric.weighted_bce(t, logits, weights)
        else:
            raise ValueError(f'Unrecognized dx_loss: {loss_label}')

    @classmethod
    def sample_reg_hyperparams(cls, trial: optuna.Trial):
        return {
            'L_l1': 0,  #trial.suggest_float('l1', 1e-8, 5e-3, log=True),
            'L_l2': 0  # trial.suggest_float('l2', 1e-8, 5e-3, log=True),
        }

    @classmethod
    def sample_model_config(cls, trial: optuna.Trial):
        return {'state_size': trial.suggest_int('s', 100, 350, 50)}

    @abstractmethod
    def weights(self):
        pass

    def l1(self):
        l1 = sum(jnp.abs(w).sum() for w in self.weights)
        return l1 + self.dx_emb.l1() + self.dx_dec.l1()

    def l2(self):
        l2 = sum(jnp.square(w).sum() for w in self.weights)
        return l2 + self.dx_emb.l2() + self.dx_dec.l2()

class AbstractModelProxMap(AbstractModel, metaclass=ABCMeta):

    @staticmethod
    def prox_map(model):
        pass
