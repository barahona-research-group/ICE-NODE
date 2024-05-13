import equinox as eqx
import numpy as np
import pytest

from lib.ehr import CodesVector, InpatientObservables, Admission
from lib.ml.artefacts import AdmissionPrediction, AdmissionsPrediction

LEADING_OBS_INDEX = 0

@pytest.fixture(params=[1, 5])
def outcome_size(request):
    return request.param


@pytest.fixture(params=[1, 5])
def obs_size(request):
    return request.param


@pytest.fixture(params=[1, 5], ids=['lead_times_1', 'lead_times_5'])
def lead_times(request):
    return request.param


def _gen_admission(admission_id: str, obs_size: int, n_timestamps: int, outcome_size: int,
                   lead_times: int) -> Admission:
    time = set(np.random.uniform(size=(n_timestamps,), low=0, high=10).astype(np.float64))
    time = np.array(sorted(time))

    outcome = np.random.binomial(n=1, p=0.5, size=(outcome_size,)).astype(bool)
    obs_mask = np.random.binomial(n=1, p=0.8, size=(n_timestamps, obs_size)).astype(bool)
    lead_mask = np.random.binomial(n=1, p=0.5, size=(n_timestamps, lead_times)).astype(bool)
    obs = np.random.normal(size=(n_timestamps, obs_size))
    obs[:, LEADING_OBS_INDEX] = np.random.binomial(n=1, p=0.5, size=(n_timestamps,))
    leading_obs = np.random.normal(size=(n_timestamps, lead_times))

    outcome = CodesVector(vec=outcome, scheme='test')
    obs = InpatientObservables(time=time, value=obs, mask=obs_mask)
    leading_obs = InpatientObservables(time=time, value=leading_obs, mask=lead_mask)

    return Admission(admission_id=admission_id,
                     admission_dates=None,
                     dx_codes=None,
                     dx_codes_history=None,
                     leading_observable=leading_obs,
                     observables=obs,
                     outcome=outcome,
                     interventions=None)


def _id_prediction(admission: Admission) -> AdmissionPrediction:
    outcome = eqx.tree_at(lambda x: x.vec, admission.outcome, admission.outcome.vec.astype(float))
    obs = eqx.tree_at(lambda x: x.value, admission.observables, admission.observables.value.astype(float))
    leading_obs = eqx.tree_at(lambda x: x.value, admission.leading_observable,
                              admission.leading_observable.value.astype(float))
    return AdmissionPrediction(admission=admission, observables=obs, outcome=outcome, leading_observable=leading_obs)


@pytest.fixture
def identical_predictions(outcome_size: int, obs_size: int, lead_times: int) -> AdmissionsPrediction:
    n_admissions = 10
    admissions_predictions = AdmissionsPrediction()
    for i in range(n_admissions):
        prediction = _id_prediction(_gen_admission(admission_id=str(i),
                                                   obs_size=obs_size,
                                                   n_timestamps=10,
                                                   outcome_size=outcome_size,
                                                   lead_times=lead_times))
        admissions_predictions = admissions_predictions.add(subject_id='test_subject',
                                                            prediction=prediction)
    return admissions_predictions
