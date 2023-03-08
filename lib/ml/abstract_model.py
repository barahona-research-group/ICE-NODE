"""Abstract class for predictive EHR models."""

from __future__ import annotations
from typing import List, Any, TYPE_CHECKING, Callable, Union, Tuple, Any, Dict, Optional
from abc import ABC, abstractmethod, ABCMeta
import zipfile

import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx

if TYPE_CHECKING:
    import optuna

from ..ehr import Subject_JAX, OutcomeExtractor, MixedOutcomeExtractor
from ..embeddings import embeddings_from_conf
from ..utils import translate_path


class AbstractModel(eqx.Module, metaclass=ABCMeta):
    dx_emb: Callable
    dx_dec: Callable
    outcome: OutcomeExtractor

    state_size: Union[int, Tuple[int, ...]]
    control_size: int = 0

    @abstractmethod
    def __call__(self, subject_interface: Subject_JAX,
                 subjects_batch: List[int], args):
        pass

    def outcome_mixer(self):
        if isinstance(self.outcome, MixedOutcomeExtractor):
            return self.outcome.mixer_params()
        else:
            return None

    @staticmethod
    def decoder_input_size(expt_config):
        return expt_config["model"]["state_size"]

    def subject_embeddings(self, subject_interface: Subject_JAX,
                           batch: List[int]):
        out = self(subject_interface, batch, dict(return_embeddings=True))
        return {i: out['predictions'].get_subject_embeddings(i) for i in batch}

    @staticmethod
    def _emb_subtrees(pytree):
        return (pytree.dx_emb, pytree.dx_dec)

    @staticmethod
    def emb_dyn_partition(pytree):
        """
        Separate the dynamics parameters from the embedding parameters.
        Thanks to Patrick Kidger for the clever function of eqx.partition.
        """
        emb_leaves = jtu.tree_leaves(AbstractModel._emb_subtrees(pytree))
        emb_predicate = lambda _t: any(_t is t for t in emb_leaves)
        emb_tree, dyn_tree = eqx.partition(pytree, emb_predicate)
        return emb_tree, dyn_tree

    @staticmethod
    def emb_dyn_merge(emb_tree, dyn_tree):
        return eqx.combine(emb_tree, dyn_tree)

    @staticmethod
    def dx_outcome_partitions(subject_interface: Subject_JAX,
                              train_ids: List[int]):
        return subject_interface.dx_outcome_by_percentiles(20, train_ids)

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
        l1 = sum(jnp.abs(w).sum() for w in jtu.tree_leaves(self.weights()))
        return l1 + self.dx_emb.l1() + self.dx_dec.l1()

    def l2(self):
        l2 = sum(jnp.square(w).sum() for w in jtu.tree_leaves(self.weights()))
        return l2 + self.dx_emb.l2() + self.dx_dec.l2()

    @classmethod
    def from_config(cls, conf: Dict[str, Any], subject_interface: Subject_JAX,
                    train_split: List[int], key: "jax.random.PRNGKey"):
        decoder_input_size = cls.decoder_input_size(conf)
        emb_models = embeddings_from_conf(conf["emb"], subject_interface,
                                          train_split, decoder_input_size)
        control_size = subject_interface.control_dim
        return cls(**emb_models,
                   **conf["model"],
                   outcome=subject_interface.outcome_extractor,
                   control_size=control_size,
                   key=key)

    def load_params(self, params_file):
        """
        Load the parameters in `params_file` filepath and return as PyTree Object.
        """
        with open(translate_path(params_file), 'rb') as file_rsc:
            return eqx.tree_deserialise_leaves(file_rsc, self)

    def write_params(self, params_file):
        """
        Store the parameters (PyTree object) into a new file
        given by `params_file`.
        """
        with open(translate_path(params_file), 'wb') as file_rsc:
            eqx.tree_serialise_leaves(file_rsc, self)

    def load_params_from_archive(self, zipfile_fname: str, params_fname: str):

        with zipfile.ZipFile(translate_path(zipfile_fname),
                             compression=zipfile.ZIP_STORED,
                             mode="r") as archive:
            with archive.open(params_fname, "r") as zip_member:
                return eqx.tree_deserialise_leaves(zip_member, self)


class AbstractModelProxMap(AbstractModel, metaclass=ABCMeta):

    @staticmethod
    def prox_map(model):
        pass
