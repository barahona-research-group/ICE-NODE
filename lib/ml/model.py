"""Abstract class for predictive EHR models."""

from __future__ import annotations
from typing import List, TYPE_CHECKING, Callable, Union, Tuple
from abc import abstractmethod, ABCMeta
import zipfile
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx

from ..utils import tqdm_constructor, translate_path
from ..ehr import (Patients, Patient, DemographicVectorConfig, DatasetScheme,
                   Predictions, Admission)
from .embeddings import (PatientEmbedding, PatientEmbeddingDimensions,
                         EmbeddedAdmission)

if TYPE_CHECKING:
    import optuna


class ModelDimensions(eqx.Module):
    emb: PatientEmbeddingDimensions = PatientEmbeddingDimensions()


class AbstractModel(eqx.Module, metaclass=ABCMeta):
    f_emb: PatientEmbedding
    f_dx_dec: Callable

    schemes: Tuple[DatasetScheme] = eqx.static_field()
    dims: ModelDimensions = eqx.static_field()
    demographic_vector_config: DemographicVectorConfig = eqx.static_field()

    @classmethod
    def _assert_demo_dim(cls, dims: ModelDimensions, scheme: DatasetScheme,
                        demographic_vector_config: DemographicVectorConfig):
        demo_vec_dim = scheme.demographic_vector_size(
            demographic_vector_config)
        assert ((demo_vec_dim == 0 and dims.emb.demo == 0) or
                (demo_vec_dim > 0 and dims.emb.demo > 0)), \
            f"Model dimensionality for demographic embedding size "\
            f"({dims.emb.demo}) and input demographic vector size "\
            f"({demo_vec_dim}) must both be zero or non-zero."

    @abstractmethod
    def __call__(self, x: Union[Patient, Admission],
                 embedded_x: Union[List[EmbeddedAdmission],
                                   EmbeddedAdmission]):
        pass

    @abstractmethod
    def batch_predict(self, patients: Patients, leave_pbar: bool = False):
        pass

    @property
    @abstractmethod
    def counts_ignore_first_admission(self):
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

    def load_params(self, params_file):
        """
        Load the parameters in `params_file`\
            filepath and return as PyTree Object.
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

    def pathwise_params(self):

        def label(x):
            if hasattr(x, 'key'):
                return x.key
            if hasattr(x, 'name'):
                return x.name
            if hasattr(x, 'idx'):
                return str(x.idx)
            return 'unknown'

        params_part, _ = eqx.partition(self, eqx.is_inexact_array)
        params_w_path, _ = jtu.tree_flatten_with_path(params_part)
        paths, params = zip(*params_w_path)
        dotted_paths = map(lambda path: '.'.join(map(label, path)), paths)
        return dict(zip(dotted_paths, params))

    @eqx.filter_jit
    def pathwise_params_stats(self):
        params = self.pathwise_params()
        return {
            k: {
                'mean': jnp.nanmean(v),
                'std': jnp.nanstd(v),
                'min': jnp.nanmin(v),
                'max': jnp.nanmax(v),
                'l1': jnp.abs(v).sum(),
                'l2': jnp.square(v).sum(),
                'nans': jnp.isnan(v).sum(),
                'size': jnp.size(v)
            }
            for k, v in params.items()
        }


class InpatientModel(AbstractModel):

    @property
    def counts_ignore_first_admission(self):
        return False

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
                              unit='longitudinal-days',
                              leave=leave_pbar) as pbar:
            results = Predictions()
            for i, subject_id in enumerate(inpatients.subjects.keys()):
                pbar.set_description(
                    f"Subject: {subject_id} ({i+1}/{len(inpatients)})")
                inpatient = inpatients.subjects[subject_id]
                embedded_admissions = inpatients_emb[subject_id]
                for adm, adm_e in zip(inpatient.admissions,
                                      embedded_admissions):
                    results.add(subject_id=subject_id,
                                prediction=self(adm, adm_e))
                    pbar.update(adm.interval_days)
            return results.filter_nans()


class OutpatientModel(AbstractModel):

    @property
    def counts_ignore_first_admission(self):
        return True

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
                              unit='longitudinal-days',
                              leave=leave_pbar) as pbar:
            results = Predictions()
            for i, subject_id in enumerate(inpatients.subjects.keys()):
                pbar.set_description(
                    f"Subject: {subject_id} ({i+1}/{len(inpatients)})")
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
