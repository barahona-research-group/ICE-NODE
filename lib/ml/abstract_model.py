"""Abstract class for predictive EHR models."""

from __future__ import annotations
from typing import List, Any, TYPE_CHECKING, Callable, Union, Tuple, Any, Dict, Optional
from abc import ABC, abstractmethod, ABCMeta
import zipfile

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx

if TYPE_CHECKING:
    import optuna

from ..ehr import (Patients, Patient, OutcomeExtractor,
                   DemographicVectorConfig, MIMICDatasetScheme, Predictions,
                   Admission)
from ..embeddings import (PatientEmbedding, PatientEmbeddingDimensions,
                          EmbeddedAdmission)
from ..utils import translate_path, tqdm_constructor


class ModelDimensions(eqx.Module):
    emb: PatientEmbeddingDimensions = PatientEmbeddingDimensions()


class AbstractModel(eqx.Module, metaclass=ABCMeta):
    f_emb: Callable[[int], jnp.ndarray]
    f_dx_dec: Callable

    scheme: MIMICDatasetScheme = eqx.static_field()
    dims: ModelDimensions = eqx.static_field()
    demographic_vector_config: DemographicVectorConfig = eqx.static_field()

    @abstractmethod
    def __call__(self, x: Union[Patient, Admission],
                 embedded_x: Union[List[EmbeddedAdmission],
                                   EmbeddedAdmission]):
        pass

    @abstractmethod
    def batch_predict(self, patients: Patients, leave_pbar: bool = False):
        pass

    # def subject_embeddings(self, patients: Patients, batch: List[int]):
    #     out = self(patients, batch, dict(return_embeddings=True))
    #     return {i: out['predictions'].get_subject_embeddings(i) for i in batch}

    # @classmethod
    # def sample_reg_hyperparams(cls, trial: optuna.Trial):
    #     return {
    #         'L_l1': 0,  #trial.suggest_float('l1', 1e-8, 5e-3, log=True),
    #         'L_l2': 0  # trial.suggest_float('l2', 1e-8, 5e-3, log=True),
    #     }

    @classmethod
    def sample_model_config(cls, trial: optuna.Trial):
        return {}

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


class InpatientModel(AbstractModel):

    def batch_predict(self,
                      inpatients: Patients,
                      leave_pbar: bool = False) -> Predictions:
        total_int_days = inpatients.interval_days()

        inpatients_emb = {
            i: self.f_emb(subject)
            for i, subject in tqdm_constructor(inpatients.subjects.items(),
                                               desc="Embedding",
                                               unit='subject',
                                               leave=leave_pbar)
        }

        r_bar = '| {n:.2f}/{total:.2f} [{elapsed}<{remaining}, ' '{rate_fmt}{postfix}]'
        bar_format = '{l_bar}{bar}' + r_bar
        with tqdm_constructor(total=total_int_days,
                              bar_format=bar_format,
                              unit='odeint-days',
                              leave=leave_pbar) as pbar:
            results = Predictions()
            for i, subject_id in enumerate(inpatients.subjects.keys()):
                pbar.set_description(f"Subject {i+1}/{len(inpatients)}")
                inpatient = inpatients.subjects[subject_id]
                embedded_admissions = inpatients_emb[subject_id]
                for adm, adm_e in zip(inpatient.admissions,
                                      embedded_admissions):
                    results.add(subject_id=subject_id,
                                prediction=self(adm, adm_e))
                    pbar.update(adm.interval_days)
            return results.filter_nans()


class OutpatientModel(AbstractModel):

    def batch_predict(self,
                      inpatients: Patients,
                      leave_pbar: bool = False) -> Predictions:
        total_int_days = inpatients.d2d_interval_days()

        inpatients_emb = {
            i: self.f_emb(subject)
            for i, subject in tqdm_constructor(inpatients.subjects.items(),
                                               desc="Embedding",
                                               unit='subject',
                                               leave=leave_pbar)
        }

        r_bar = '| {n:.2f}/{total:.2f} [{elapsed}<{remaining}, ' '{rate_fmt}{postfix}]'
        bar_format = '{l_bar}{bar}' + r_bar
        with tqdm_constructor(total=total_int_days,
                              bar_format=bar_format,
                              unit='odeint-days',
                              leave=leave_pbar) as pbar:
            results = Predictions()
            for i, subject_id in enumerate(inpatients.subjects.keys()):
                pbar.set_description(f"Subject {i+1}/{len(inpatients)}")
                inpatient = inpatients.subjects[subject_id]
                embedded_admissions = inpatients_emb[subject_id]
                for pred in self(inpatient, embedded_admissions):
                    results.add(subject_id=subject_id, prediction=pred)
                pbar.update(inpatient.d2d_interval_days)
            return results.filter_nans()


#     @classmethod
#     def from_config(cls, conf: Dict[str, Any], patients: Patients,
#                     train_split: List[int], key: "jax.random.PRNGKey"):
#         decoder_input_size = cls.decoder_input_size(conf)
#         emb_models = embeddings_from_conf(conf["emb"], patients, train_split,
#                                           decoder_input_size)
#         control_size = patients.control_dim
#         return cls(**emb_models,
#                    **conf["model"],
#                    outcome=patients.outcome_extractor,
#                    control_size=control_size,
#                    key=key)

# def load_params(self, params_file):
#     """
#     Load the parameters in `params_file` filepath and return as PyTree Object.
#     """
#     with open(translate_path(params_file), 'rb') as file_rsc:
#         return eqx.tree_deserialise_leaves(file_rsc, self)

# def write_params(self, params_file):
#     """
#     Store the parameters (PyTree object) into a new file
#     given by `params_file`.
#     """
#     with open(translate_path(params_file), 'wb') as file_rsc:
#         eqx.tree_serialise_leaves(file_rsc, self)

# def load_params_from_archive(self, zipfile_fname: str, params_fname: str):

#     with zipfile.ZipFile(translate_path(zipfile_fname),
#                          compression=zipfile.ZIP_STORED,
#                          mode="r") as archive:
#         with archive.open(params_fname, "r") as zip_member:
#             return eqx.tree_deserialise_leaves(zip_member, self)
