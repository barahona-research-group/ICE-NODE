from types import NoneType
from typing import (Self)

import equinox as eqx

from lib.ml.model import AbstractModel
from .artefacts import AdmissionsPrediction, AdmissionPrediction
from .in_models import AdmissionTrajectoryPrediction
from ..base import np_module, Array
from ..ehr import (TVxEHR, Admission, InpatientObservables)
from ..utils import tqdm_constructor


class ZeroImputer(AbstractModel):
    def discount_first_admission(self):
        return False

    def dyn_params_list(self):
        return []

    def params_list(self):
        return []

    def _apply_imputation(self, x: Array, mask: Array) -> Array:
        return np_module(x).zeros_like(x)

    def __call__(self, admission: Admission) -> AdmissionPrediction:
        prediction = AdmissionTrajectoryPrediction(admission=admission)
        obs = admission.observables
        imputed_obs = InpatientObservables(time=obs.time, value=self._apply_imputation(obs.value, obs.mask),
                                           mask=obs.mask)
        return prediction.add(observables=imputed_obs)

    def batch_predict(
            self,
            patients: TVxEHR,
            leave_pbar: bool = False
    ) -> AdmissionsPrediction:
        results = AdmissionsPrediction()
        for subject_id in tqdm_constructor(patients.subjects.keys(), leave=leave_pbar):
            inpatient = patients.subjects[subject_id]
            for admission in inpatient.admissions:
                results = results.add(subject_id=subject_id, prediction=self(admission))
        return results

    @classmethod
    def from_tvx_ehr(cls, tvx_ehr: TVxEHR, config: NoneType = None,
                     embeddings_config: NoneType = None,
                     seed: NoneType = None) -> Self:
        assert config is None and embeddings_config is None and seed is None
        return cls(config=None)


class MeanImputer(ZeroImputer):
    mean_fit: Array = None

    def fit_mean(self, patients: TVxEHR) -> Self:
        obs_list = [admission.observables for subject in patients.subjects.values() for admission in subject.admissions]
        np = np_module(obs_list[0].value)
        mean_fit = np.nanmean(np.vstack([obs.value for obs in obs_list]), axis=0,
                              where=np.vstack([obs.mask for obs in obs_list]))
        mean_fit = np.nan_to_num(mean_fit, nan=0.0)
        return eqx.tree_at(lambda x: x.mean_fit, self, mean_fit, is_leaf=lambda x: x is None)

    def params_list(self):
        return [self.mean_fit]

    def _apply_imputation(self, x: Array, mask: Array) -> Array:
        return np_module(x).vstack([self.mean_fit for _ in range(len(x))]) if len(x) > 0 else x

    @classmethod
    def from_tvx_ehr(cls, tvx_ehr: TVxEHR, config: NoneType = None,
                     embeddings_config: NoneType = None,
                     seed: NoneType = None) -> Self:
        assert config is None and embeddings_config is None and seed is None
        return cls(config=None).fit_mean(tvx_ehr)


class RectilinearImputer(MeanImputer):
    """
    Imputes (predicts) the missing observation using the rectilinear online interpolation, i.e. prediction of
    an observation at any timestamp is inherited from the last observed value. If the first observation is missing,
    it is imputed with the mean value of the feature. If there is no valid mean (e.g. no observations for
    the feature in the training data), then substitute with 0.
    """

    def _apply_imputation(self, x: Array, mask: Array) -> Array:
        if len(x) == 0:
            return x
        last_obs = self.mean_fit
        imputed_x = []
        np = np_module(x)
        for i, (obs, m) in enumerate(zip(x, mask)):
            imputed_x.append(last_obs)
            last_obs = np.where(m, obs, last_obs)
        return np.vstack(imputed_x)
