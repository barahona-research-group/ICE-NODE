import random
from typing import List, Callable

import jax
import numpy as np
import numpy.random as nrand
import pandas as pd
import pytest

from ehr.conftest import leading_observables_extractor
from lib.ehr import CodingScheme, \
    DemographicVectorConfig, InpatientObservables
from lib.ehr.coding_scheme import NumericScheme, FrozenDict11, FrozenDict1N, CodesVector, ReducedCodeMapN1, \
    CodingSchemesManager, GroupingData
from lib.ehr.dataset import DatasetSchemeConfig
from lib.ehr.tvx_concepts import SegmentedAdmission, InpatientInterventions, AdmissionDates, Admission, \
    SegmentedInpatientInterventions, InpatientInput, \
    Patient, SegmentedPatient, StaticInfo

DATASET_SCOPE = "function"
MAX_STAY_DAYS = 356
jax.config.update('jax_platform_name', 'cpu')


def scheme(name: str, codes: List[str]) -> CodingScheme:
    return CodingScheme(name=name, codes=tuple(sorted(codes)),
                        desc=FrozenDict11.from_dict(dict(zip(codes, codes))))


BINARY_OBSERVATION_CODE_INDEX = 0
CATEGORICAL_OBSERVATION_CODE_INDEX = 1
ORDINAL_OBSERVATION_CODE_INDEX = 2
NUMERIC_OBSERVATION_CODE_INDEX = 3


@pytest.fixture(params=[('ethnicity1', ['E1', 'E2', 'E3'])])
def ethnicity_scheme(request) -> CodingScheme:
    return scheme(*request.param)


@pytest.fixture(params=[('gender1', ['M', 'F'])])
def gender_scheme(request) -> CodingScheme:
    return scheme(*request.param)


@pytest.fixture(params=[('dx1', ['Dx1', 'Dx2', 'Dx3', 'Dx4', 'Dx5'])])
def dx_scheme(request) -> CodingScheme:
    return scheme(*request.param)


@pytest.fixture(params=[('hosp_proc1', ['P1', 'P2', 'P3'])])
def hosp_proc_scheme(request) -> CodingScheme:
    return scheme(*request.param)


@pytest.fixture(params=[('icu_proc1', ['P1', 'P2', 'P3'])])
def icu_proc_scheme(request) -> CodingScheme:
    return scheme(*request.param)


@pytest.fixture(params=[('icu_inputs1', ['I1', 'I2', 'I3'])])
def icu_inputs_scheme(request) -> CodingScheme:
    return scheme(*request.param)


@pytest.fixture(params=[('observation1',
                         ('O1', 'O2', 'O3', 'O4', 'O5'),
                         ('B', 'C', 'O', 'N', 'N'))])
def observation_scheme(request) -> CodingScheme:
    name, codes, types = request.param
    return NumericScheme(name=name,
                         codes=tuple(sorted(codes)),
                         desc=FrozenDict11.from_dict(dict(zip(codes, codes))),
                         type_hint=FrozenDict11.from_dict(dict(zip(codes, types))))


@pytest.fixture
def outcome_extractor(dx_scheme: CodingScheme) -> Callable[[CodesVector], CodesVector]:
    name = f'{dx_scheme.name}_outcome'
    k = max(3, len(dx_scheme.codes) - 1)
    random.seed(0)
    excluded = random.sample(dx_scheme.codes, k=k)
    codes = set(dx_scheme.codes) - set(excluded)
    outcome = CodingScheme(name=name, codes=tuple(sorted(codes)), desc=FrozenDict11.from_dict(dict(zip(codes, codes))))

    def _extractor(dx_codes: CodesVector) -> CodesVector:
        codeset = {dx_scheme.index2code[i] for i in np.flatnonzero(dx_codes.vec)}
        codeset = codeset - set(excluded)
        return outcome.codeset2vec(codeset)

    return _extractor


@pytest.fixture
def dataset_scheme_config(ethnicity_scheme: CodingScheme,
                          gender_scheme: CodingScheme,
                          dx_scheme: CodingScheme,
                          icu_proc_scheme: CodingScheme,
                          icu_inputs_scheme: CodingScheme,
                          observation_scheme: CodingScheme,
                          hosp_proc_scheme: CodingScheme) -> DatasetSchemeConfig:
    return DatasetSchemeConfig(ethnicity=ethnicity_scheme.name,
                               gender=gender_scheme.name,
                               dx_discharge=dx_scheme.name,
                               icu_procedures=icu_proc_scheme.name,
                               icu_inputs=icu_inputs_scheme.name,
                               obs=observation_scheme.name,
                               hosp_procedures=hosp_proc_scheme.name)


@pytest.fixture(params=[('icu_inputs1_target', ['I1_target', 'I2_target', 'I3_target', 'I4_target', 'I5_target'])])
def icu_inputs_target_scheme(request) -> CodingScheme:
    return scheme(*request.param)


@pytest.fixture
def icu_inputs_aggregation(icu_inputs_target_scheme) -> FrozenDict11:
    return FrozenDict11.from_dict({c: 'w_sum' for c in icu_inputs_target_scheme.codes})


@pytest.fixture
def icu_inputs_mapping_data(icu_inputs_scheme, icu_inputs_target_scheme) -> FrozenDict1N:
    return FrozenDict1N.from_dict({c: {random.choice(icu_inputs_target_scheme.codes)} for c in icu_inputs_scheme.codes})


@pytest.fixture
def icu_inputs_map(icu_inputs_scheme, icu_inputs_target_scheme, icu_inputs_aggregation,
                   icu_inputs_mapping_data) -> ReducedCodeMapN1:
    m = ReducedCodeMapN1.from_data(icu_inputs_scheme.name,
                                   icu_inputs_target_scheme.name,
                                   icu_inputs_mapping_data, icu_inputs_aggregation)
    return m


@pytest.fixture
def icu_inputs_scheme_manager(icu_inputs_scheme: CodingScheme, icu_inputs_target_scheme: CodingScheme,
                              icu_inputs_map: ReducedCodeMapN1) -> CodingSchemesManager:
    manager = CodingSchemesManager().add_scheme(icu_inputs_scheme).add_scheme(icu_inputs_target_scheme)
    manager = manager.add_map(icu_inputs_map)
    return manager


@pytest.fixture
def icu_inputs_grouping_data(icu_inputs_scheme_manager: CodingSchemesManager,
                             icu_inputs_map: ReducedCodeMapN1) -> GroupingData:
    return icu_inputs_map.grouping_data(icu_inputs_scheme_manager)


LENGTH_OF_STAY = 10.0


def _singular_codevec(scheme: CodingScheme) -> CodesVector:
    return scheme.codeset2vec({random.choice(scheme.codes)})


@pytest.fixture
def gender(gender_scheme: CodingScheme) -> CodesVector:
    return _singular_codevec(gender_scheme)


@pytest.fixture
def ethnicity(ethnicity_scheme: CodingScheme) -> CodesVector:
    return _singular_codevec(ethnicity_scheme)


def date_of_birth() -> pd.Timestamp:
    return pd.to_datetime(pd.Timestamp('now') - pd.to_timedelta(nrand.randint(0, 100 * 365), unit='D'))


def demographic_vector_config() -> DemographicVectorConfig:
    flags = random.choices([True, False], k=3)
    return DemographicVectorConfig(*flags)


def _static_info(ethnicity: CodesVector, gender: CodesVector) -> StaticInfo:
    return StaticInfo(ethnicity=ethnicity, gender=gender,
                      date_of_birth=date_of_birth())


@pytest.fixture
def static_info(ethnicity: CodesVector, gender: CodesVector) -> StaticInfo:
    return _static_info(ethnicity, gender)


def _dx_codes(dx_scheme: CodingScheme):
    v = nrand.binomial(1, 0.5, size=len(dx_scheme)).astype(bool)
    return CodesVector(vec=v, scheme=dx_scheme.name)


@pytest.fixture
def dx_codes(dx_scheme: CodingScheme):
    return _dx_codes(dx_scheme)


def _dx_codes_history(dx_codes: CodesVector):
    v = nrand.binomial(1, 0.5, size=len(dx_codes)).astype(bool)
    return CodesVector(vec=v + dx_codes.vec, scheme=dx_codes.scheme)


@pytest.fixture
def dx_codes_history(dx_codes: CodesVector):
    return _dx_codes_history(dx_codes)


@pytest.fixture
def outcome(outcome_extractor: Callable[[CodesVector], CodesVector], dx_codes: CodesVector) -> CodesVector:
    return outcome_extractor(dx_codes)


def _inpatient_observables(observation_scheme: CodingScheme, n_timestamps: int):
    d = len(observation_scheme)
    timestamps_grid = np.linspace(0, LENGTH_OF_STAY, 1000, dtype=np.float64)
    t = np.array(sorted(nrand.choice(timestamps_grid, replace=False, size=n_timestamps)))
    v = nrand.randn(n_timestamps, d)
    mask = nrand.binomial(1, 0.5, size=(n_timestamps, d)).astype(bool)
    return InpatientObservables(t, v, mask)


@pytest.fixture(params=[0, 1, 501])
def inpatient_observables(observation_scheme: CodingScheme, request):
    n_timestamps = request.param
    return _inpatient_observables(observation_scheme, n_timestamps)


def inpatient_binary_input(n: int, p: int):
    starttime = np.array(
        sorted(nrand.choice(np.linspace(0, LENGTH_OF_STAY, max(1000, n)), replace=False, size=n)))
    endtime = starttime + nrand.uniform(0, LENGTH_OF_STAY - starttime, size=(n,))
    code_index = nrand.choice(p, size=n, replace=True)
    return InpatientInput(starttime=starttime, endtime=endtime, code_index=code_index)


def inpatient_rated_input(n: int, p: int):
    bin_input = inpatient_binary_input(n, p)
    return InpatientInput(starttime=bin_input.starttime, endtime=bin_input.endtime, code_index=bin_input.code_index,
                          rate=nrand.uniform(0, 1, size=(n,)))


def _icu_inputs(icu_inputs_scheme: CodingScheme, n_timestamps: int):
    return inpatient_rated_input(n_timestamps, len(icu_inputs_scheme))


@pytest.fixture(params=[0, 1, 501])
def icu_inputs(icu_inputs_scheme: CodingScheme, request):
    return _icu_inputs(icu_inputs_scheme, request.param)


def _proc(scheme: CodingScheme, n_timestamps: int):
    return inpatient_binary_input(n_timestamps, len(scheme))


@pytest.fixture(params=[0, 1, 501])
def icu_proc(icu_proc_scheme: CodingScheme, request):
    return _proc(icu_proc_scheme, request.param)


@pytest.fixture(params=[0, 1, 501])
def hosp_proc(hosp_proc_scheme: CodingScheme, request):
    return _proc(hosp_proc_scheme, n_timestamps=request.param)


def _inpatient_interventions(hosp_proc, icu_proc, icu_inputs):
    return InpatientInterventions(hosp_proc, icu_proc, icu_inputs)


@pytest.fixture(params=[0, 1, 2, -1])
def inpatient_interventions(hosp_proc, icu_proc, icu_inputs, request):
    whoisnull = request.param
    return _inpatient_interventions(None if whoisnull == 0 else hosp_proc,
                                    None if whoisnull == 1 else icu_proc,
                                    None if whoisnull == 2 else icu_inputs)


def _segmented_inpatient_interventions(inpatient_interventions: InpatientInterventions, hosp_proc_scheme,
                                       icu_proc_scheme,
                                       icu_inputs_scheme,
                                       maximum_padding: int = 1) -> SegmentedInpatientInterventions:
    assert all(isinstance(scheme, CodingScheme) for scheme in [hosp_proc_scheme, icu_proc_scheme, icu_inputs_scheme])
    return SegmentedInpatientInterventions.from_interventions(inpatient_interventions, LENGTH_OF_STAY,
                                                              hosp_procedures_size=len(hosp_proc_scheme),
                                                              icu_procedures_size=len(icu_proc_scheme),
                                                              icu_inputs_size=len(icu_inputs_scheme),
                                                              maximum_padding=maximum_padding)


@pytest.fixture
def segmented_inpatient_interventions(inpatient_interventions: InpatientInterventions, hosp_proc_scheme,
                                      icu_proc_scheme,
                                      icu_inputs_scheme) -> SegmentedInpatientInterventions:
    return _segmented_inpatient_interventions(inpatient_interventions,
                                              hosp_proc_scheme=hosp_proc_scheme,
                                              icu_proc_scheme=icu_proc_scheme,
                                              icu_inputs_scheme=icu_inputs_scheme,
                                              maximum_padding=1)


@pytest.fixture
def leading_observable(observation_scheme: NumericScheme,
                       inpatient_observables: InpatientObservables) -> InpatientObservables:
    return leading_observables_extractor(observation_scheme=observation_scheme)(inpatient_observables)


def _admission(admission_id: str, admission_date: pd.Timestamp,
               dx_codes: CodesVector,
               dx_codes_history: CodesVector, outcome: CodesVector, observables: InpatientObservables,
               interventions: InpatientInterventions, leading_observable: InpatientObservables):
    discharge_date = pd.to_datetime(admission_date + pd.to_timedelta(LENGTH_OF_STAY, unit='H'))

    return Admission(admission_id=admission_id, admission_dates=AdmissionDates(admission_date, discharge_date),
                     dx_codes=dx_codes,
                     dx_codes_history=dx_codes_history, outcome=outcome, observables=observables,
                     interventions=interventions, leading_observable=leading_observable)


@pytest.fixture
def admission(dx_codes: CodesVector, dx_codes_history: CodesVector,
              outcome: CodesVector, inpatient_observables: InpatientObservables,
              inpatient_interventions: InpatientInterventions,
              leading_observable: InpatientObservables) -> Admission:
    admission_id = 'test'
    return _admission(admission_id=admission_id, admission_date=pd.to_datetime('now'),
                      dx_codes=dx_codes, dx_codes_history=dx_codes_history, outcome=outcome,
                      observables=inpatient_observables, interventions=inpatient_interventions,
                      leading_observable=leading_observable)


@pytest.fixture
def segmented_admission(admission: Admission, icu_inputs_scheme: CodingScheme, icu_proc_scheme: CodingScheme,
                        hosp_proc_scheme: CodingScheme) -> SegmentedAdmission:
    return SegmentedAdmission.from_admission(admission=admission, maximum_padding=1,
                                             icu_inputs_size=len(icu_inputs_scheme),
                                             icu_procedures_size=len(icu_proc_scheme),
                                             hosp_procedures_size=len(hosp_proc_scheme))


@pytest.fixture
def segmented_patient(patient: Patient, icu_inputs_scheme: CodingScheme, icu_proc_scheme: CodingScheme,
                      hosp_proc_scheme: CodingScheme) -> SegmentedPatient:
    return SegmentedPatient.from_patient(patient=patient, maximum_padding=1,
                                         icu_inputs_size=len(icu_inputs_scheme),
                                         icu_procedures_size=len(icu_proc_scheme),
                                         hosp_procedures_size=len(hosp_proc_scheme))


def _admissions(n_admissions, dx_scheme: CodingScheme, observation_scheme: NumericScheme,
                icu_inputs_scheme: CodingScheme, icu_proc_scheme: CodingScheme,
                hosp_proc_scheme: CodingScheme, outcome_extractor: Callable[[CodesVector], CodesVector]) -> List[
    Admission]:
    admissions = []
    for i in range(n_admissions):
        dx_codes = _dx_codes(dx_scheme)
        obs = _inpatient_observables(observation_scheme, n_timestamps=nrand.randint(0, 100))
        lead = leading_observables_extractor(observation_scheme=observation_scheme)(obs)
        icu_proc = _proc(icu_proc_scheme, n_timestamps=nrand.randint(0, 50))
        hosp_proc = _proc(hosp_proc_scheme, n_timestamps=nrand.randint(0, 50))
        icu_inputs = _icu_inputs(icu_inputs_scheme, n_timestamps=nrand.randint(0, 50))

        admissions.append(_admission(admission_id=f'test_{i}', admission_date=pd.to_datetime('now'),
                                     dx_codes=dx_codes,
                                     dx_codes_history=_dx_codes_history(dx_codes),
                                     outcome=outcome_extractor(dx_codes),
                                     observables=obs,
                                     interventions=_inpatient_interventions(hosp_proc=hosp_proc, icu_proc=icu_proc,
                                                                            icu_inputs=icu_inputs),
                                     leading_observable=lead))
    return admissions


@pytest.fixture(params=[0, 1, 50])
def patient(request, static_info: StaticInfo,
            dx_scheme: CodingScheme,
            outcome_extractor: Callable[[CodesVector], CodesVector], observation_scheme: NumericScheme,
            icu_inputs_scheme: CodingScheme, icu_proc_scheme: CodingScheme,
            hosp_proc_scheme: CodingScheme) -> Patient:
    admissions = _admissions(n_admissions=request.param, dx_scheme=dx_scheme, observation_scheme=observation_scheme,
                             icu_inputs_scheme=icu_inputs_scheme, icu_proc_scheme=icu_proc_scheme,
                             outcome_extractor=outcome_extractor,
                             hosp_proc_scheme=hosp_proc_scheme)
    return Patient(subject_id='test', admissions=admissions, static_info=static_info)
