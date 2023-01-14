"""Abstract class for predictive EHR models."""

from __future__ import annotations
from typing import List, Any, TYPE_CHECKING, Callable, Union, Tuple, Optional
from abc import ABC, abstractmethod, ABCMeta

from absl import logging
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx

if TYPE_CHECKING:
    import optuna

from .. import utils
from ..embeddings import (CachedGRAM, MatrixEmbeddings, train_glove,
                          LogitsDecoder)
from ..ehr import Subject_JAX, OutcomeExtractor
from .. import metric


class AbstractModel(eqx.Module, metaclass=ABCMeta):
    model_cls = {}

    dx_emb: Callable
    dx_dec: Callable

    state_size: Union[int, Tuple[int, ...]]
    control_size: int

    @classmethod
    def register_model(cls, label):
        cls.model_cls[label] = cls

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

    @classmethod
    def create_embeddings_model(cls,
                                code_type: str,
                                emb_model: str,
                                subject_interface: Subject_JAX,
                                embeddings_size: int,
                                train_ids: Optional[List[int]] = None,
                                **emb_config):

        if emb_model == 'matrix':
            if code_type == 'dx':
                input_size = subject_interface.dx_dim
            else:
                input_size = subject_interface.pr_dim

            return MatrixEmbeddings(embeddings_size=embeddings_size,
                                    input_size=input_size,
                                    key=jrandom.PRNGKey(0))

        if emb_model == 'gram':
            if code_type == 'dx':
                cooc_f = subject_interface.dx_augmented_coocurrence
                ancestors_mat = subject_interface.dx_make_ancestors_mat()
            else:
                cooc_f = subject_interface.pr_augmented_coocurrence
                ancestors_mat = subject_interface.pr_make_ancestors_mat()

            win_size = emb_config['cooc_window_size_days']

            coocurrence_mat = cooc_f(train_ids, window_size_days=win_size)
            glove = train_glove(cooccurrences=coocurrence_mat,
                                embeddings_size=embeddings_size,
                                iterations=emb_config['glove_iterations'])
            return CachedGRAM(basic_embeddings=glove,
                              attention_method=emb_config['attention_method'],
                              attention_size=emb_config['attention_size'],
                              ancestors_mat=ancestors_mat,
                              key=jrandom.PRNGKey(0))
        else:
            raise RuntimeError(f'Unrecognized Embedding kind {emb_model}')

    @classmethod
    def create_dx_embeddings_decoder(cls, outcome_extractor: OutcomeExtractor,
                                     embeddings_size: int, n_layers: int):
        output_size = outcome_extractor.outcome_dim
        return LogitsDecoder(n_layers=n_layers,
                             embeddings_size=embeddings_size,
                             output_size=output_size,
                             key=jrandom.PRNGKey(0))

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
    def sample_embeddings_config(cls, code_type: str, emb_model: str,
                                 trial: optuna.Trial):
        if emb_model == 'matrix':
            emb_config = MatrixEmbeddings.sample_model_config(code_type, trial)
        elif emb_model == 'gram':
            emb_config = CachedGRAM.sample_model_config(code_type, trial)
        else:
            raise RuntimeError(f'Unrecognized Embedding kind {emb_model}')

        return {code_type: emb_config, 'emb_model': emb_model}

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
