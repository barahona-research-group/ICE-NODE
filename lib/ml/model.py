"""Abstract class for predictive EHR models."""

from __future__ import annotations

import zipfile
from abc import abstractmethod
from typing import (Tuple, Self)

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu

from .artefacts import AdmissionsPrediction
from .embeddings import (EmbeddedAdmission, AdmissionEmbeddingsConfig)
from ..base import Config, Module, VxData
from ..ehr import (TVxEHR, Patient, Admission)
from ..utils import tqdm_constructor, translate_path


class ModelConfig(Config):
    pass


class Precomputes(VxData):
    pass


class AbstractModel(Module):
    config: ModelConfig = eqx.static_field()

    @property
    @abstractmethod
    def dyn_params_list(self):
        pass

    @property
    def params_list(self):
        return jtu.tree_leaves(eqx.filter(self, eqx.is_inexact_array))

    def params_list_mask(self, pytree):
        assert isinstance(pytree, dict) and 'dyn' in pytree and len(
            pytree) == 2, 'Expected a "dyn" label in Dict'
        (other_key,) = set(pytree.keys()) - {'dyn'}

        params_list = jtu.tree_leaves(eqx.filter(self, eqx.is_inexact_array))
        dyn_params = self.dyn_params_list
        is_dyn_param = lambda x: any(x is y for y in dyn_params)
        mask = {'dyn': [is_dyn_param(x) for x in params_list]}
        mask[other_key] = [not m for m in mask['dyn']]
        return mask

    def precomputes(self, inpatients: TVxEHR):
        return Precomputes()

    @abstractmethod
    def batch_predict(
            self,
            patients: TVxEHR,
            leave_pbar: bool = False
    ) -> AdmissionsPrediction:
        pass

    @property
    @abstractmethod
    def discount_first_admission(self):
        raise NotImplementedError

    @classmethod
    def sample_model_config(cls, trial):
        return {}

    def weights(self) -> Tuple[jnp.ndarray, ...]:

        def _weights(x):
            if not hasattr(x, '__dict__'):
                return tuple()
            return tuple(v for k, v in x.__dict__.items() if 'weight' in k)

        def _has_weight(leaf):
            return len(_weights(leaf)) > 0

        # Valid for eqx.nn.MLP and ml.base_models.GRUDynamics
        return sum((_weights(x)
                    for x in jtu.tree_leaves(self, is_leaf=_has_weight)
                    if _has_weight(x)), tuple())

    def l1(self) -> float:
        return sum(jnp.abs(w).sum() for w in jtu.tree_leaves(self.weights()))

    def l2(self) -> float:
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

    @classmethod
    def from_tvx_ehr(cls, tvx_ehr: TVxEHR, config: ModelConfig, embeddings_config: AdmissionEmbeddingsConfig,
                     seed: int = 0) -> Self:
        raise NotImplementedError


class InpatientModel(AbstractModel):

    @property
    def discount_first_admission(self):
        return False

    @abstractmethod
    def __call__(self,
                 x: Admission,
                 embedded_x: EmbeddedAdmission,
                 precomputes: Precomputes):
        pass

    def batch_predict(
            self,
            inpatients: TVxEHR,
            leave_pbar: bool = False
    ) -> AdmissionsPrediction:
        pass


class DischargeSummaryModel(AbstractModel):

    @property
    def discount_first_admission(self):
        return True

    @abstractmethod
    def __call__(self,
                 x: Patient,
                 embedded_x: Tuple[EmbeddedAdmission, ...],
                 precomputes: Precomputes):
        pass

    def batch_predict(
            self,
            inpatients: TVxEHR,
            leave_pbar: bool = False
    ) -> AdmissionsPrediction:
        total_int_days = inpatients.d2d_interval_days()
        precomputes = self.precomputes(inpatients)
        inpatients_emb = {
            i: tuple(self.f_emb(admission, inpatients.admission_demographics[admission.admission_id]) for admission in
                     subject.admissions)
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
            results = AdmissionsPrediction()
            for i, subject_id in enumerate(inpatients.subjects.keys()):
                pbar.set_description(
                    f"Subject: {subject_id} ({i + 1}/{len(inpatients)})")
                inpatient = inpatients.subjects[subject_id]
                embedded_admissions = inpatients_emb[subject_id]
                for pred in self(inpatient,
                                 embedded_admissions,
                                 precomputes=precomputes):
                    results = results.add(subject_id=subject_id, prediction=pred)
                pbar.update(inpatient.d2d_interval_days)
            return results.filter_nans()

    def patient_embeddings(self, patients: TVxEHR):
        pass
