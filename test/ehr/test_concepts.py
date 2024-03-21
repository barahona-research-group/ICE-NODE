import random
from copy import deepcopy
from typing import List, Callable
from unittest import mock

import equinox as eqx
import numpy as np
import numpy.random as nrand
import pandas as pd
import pytest
import tables as tb

from lib.ehr import CodingScheme, CodesVector, OutcomeExtractor
from lib.ehr.coding_scheme import NumericalTypeHint, NumericScheme
from lib.ehr.tvx_concepts import (InpatientObservables, LeadingObservableExtractorConfig, LeadingObservableExtractor,
                                  InpatientInput, InpatientInterventions, SegmentedInpatientInterventions, Admission,
                                  SegmentedAdmission, DemographicVectorConfig, StaticInfo, Patient, AdmissionDates,
                                  SegmentedPatient, SegmentedInpatientObservables)
from test.ehr.conftest import BINARY_OBSERVATION_CODE_INDEX, CATEGORICAL_OBSERVATION_CODE_INDEX, \
    NUMERIC_OBSERVATION_CODE_INDEX, ORDINAL_OBSERVATION_CODE_INDEX

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


def _outcome(outcome_extractor_: OutcomeExtractor, dx_codes: CodesVector):
    return outcome_extractor_.map_vector(dx_codes)


@pytest.fixture
def outcome(dx_codes: CodesVector, outcome_extractor: OutcomeExtractor):
    return _outcome(outcome_extractor, dx_codes)


def _inpatient_observables(obs_scheme: CodingScheme, n_timestamps: int):
    d = len(obs_scheme)
    timestamps_grid = np.linspace(0, LENGTH_OF_STAY, 1000, dtype=np.float64)
    t = np.array(sorted(nrand.choice(timestamps_grid, replace=False, size=n_timestamps)))
    v = nrand.randn(n_timestamps, d)
    mask = nrand.binomial(1, 0.5, size=(n_timestamps, d)).astype(bool)
    return InpatientObservables(t, v, mask)


@pytest.fixture(params=[0, 1, 501])
def inpatient_observables(obs_scheme: CodingScheme, request):
    n_timestamps = request.param
    return _inpatient_observables(obs_scheme, n_timestamps)


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


def leading_observables_extractor(obs_scheme: NumericScheme, leading_hours: List[float] = (1.0,),
                                  entry_neglect_window: float = 0.0,
                                  recovery_window: float = 0.0,
                                  minimum_acquisitions: int = 0,
                                  code_index: int = BINARY_OBSERVATION_CODE_INDEX) -> LeadingObservableExtractor:
    config = LeadingObservableExtractorConfig(observable_code=obs_scheme.codes[code_index],
                                              scheme=obs_scheme.name,
                                              entry_neglect_window=entry_neglect_window,
                                              recovery_window=recovery_window,
                                              minimum_acquisitions=minimum_acquisitions,
                                              leading_hours=leading_hours)
    return LeadingObservableExtractor(config=config)


@pytest.fixture
def leading_observable(obs_scheme: NumericScheme, inpatient_observables: InpatientObservables) -> InpatientObservables:
    return leading_observables_extractor(obs_scheme=obs_scheme)(inpatient_observables)


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


def _admissions(n_admissions, dx_scheme: CodingScheme,
                outcome_extractor_: OutcomeExtractor, obs_scheme: NumericScheme,
                icu_inputs_scheme: CodingScheme, icu_proc_scheme: CodingScheme,
                hosp_proc_scheme: CodingScheme) -> List[Admission]:
    admissions = []
    for i in range(n_admissions):
        dx_codes = _dx_codes(dx_scheme)
        obs = _inpatient_observables(obs_scheme, n_timestamps=nrand.randint(0, 100))
        lead = leading_observables_extractor(obs_scheme=obs_scheme)(obs)
        icu_proc = _proc(icu_proc_scheme, n_timestamps=nrand.randint(0, 50))
        hosp_proc = _proc(hosp_proc_scheme, n_timestamps=nrand.randint(0, 50))
        icu_inputs = _icu_inputs(icu_inputs_scheme, n_timestamps=nrand.randint(0, 50))

        admissions.append(_admission(admission_id=f'test_{i}', admission_date=pd.to_datetime('now'),
                                     dx_codes=dx_codes,
                                     dx_codes_history=_dx_codes_history(dx_codes),
                                     outcome=_outcome(outcome_extractor_, dx_codes),
                                     observables=obs,
                                     interventions=_inpatient_interventions(hosp_proc=hosp_proc, icu_proc=icu_proc,
                                                                            icu_inputs=icu_inputs),
                                     leading_observable=lead))
    return admissions


@pytest.fixture(params=[0, 1, 50])
def patient(request, static_info: StaticInfo,
            dx_scheme: CodingScheme,
            outcome_extractor: OutcomeExtractor, obs_scheme: CodingScheme,
            icu_inputs_scheme: CodingScheme, icu_proc_scheme: CodingScheme,
            hosp_proc_scheme: CodingScheme) -> List[Patient]:
    admissions = _admissions(n_admissions=request.param, dx_scheme=dx_scheme,
                             outcome_extractor_=outcome_extractor, obs_scheme=obs_scheme,
                             icu_inputs_scheme=icu_inputs_scheme, icu_proc_scheme=icu_proc_scheme,
                             hosp_proc_scheme=hosp_proc_scheme)
    return Patient(subject_id='test', admissions=admissions, static_info=static_info)


@pytest.fixture
def hf5_writer_file(tmpdir) -> tb.File:
    h5f = tb.open_file(tmpdir.join('test.h5'), 'w')
    yield h5f
    h5f.close()


@pytest.fixture
def hf5_group(hf5_writer_file: tb.File) -> tb.Group:
    return hf5_writer_file.create_group('/', 'test')


class TestInpatientObservables:

    def test_empty(self, obs_scheme: str):
        obs = InpatientObservables.empty(len(obs_scheme))
        assert len(obs.time) == 0
        assert len(obs.value) == 0
        assert len(obs.mask) == 0
        assert all(a.shape == (0, len(obs_scheme)) for a in [obs.value, obs.mask])

    @pytest.mark.parametrize('time_valid_dtype', [np.float64])
    @pytest.mark.parametrize('mask_valid_dtype', [bool])
    @pytest.mark.parametrize('time_invalid_dtype', [np.int32, np.int64, np.float32, bool, str, object])
    @pytest.mark.parametrize('mask_invalid_dtype', [np.int32, np.int64, np.float32, str, object])
    def test_invalid_init(self, time_invalid_dtype, mask_invalid_dtype, time_valid_dtype, mask_valid_dtype):
        obs = InpatientObservables(time=np.array([1.0, 2.0], dtype=time_valid_dtype),
                                   value=np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
                                   mask=np.array([[True, False, True], [False, True, False]],
                                                 dtype=mask_valid_dtype))

        # Invalid value shape
        with pytest.raises(AssertionError):
            InpatientObservables(time=obs.time[:1], value=obs.value[:1].squeeze(), mask=obs.mask[:1])

        # Invalid mask shape
        with pytest.raises(AssertionError):
            InpatientObservables(time=obs.time[:1], value=obs.value[:1], mask=obs.mask[:1].squeeze())

        # inconsistent time and value shape
        with pytest.raises(AssertionError):
            InpatientObservables(time=obs.time, value=obs.value[:-1], mask=obs.mask)

        # inconsistent time and mask shape
        with pytest.raises(AssertionError):
            InpatientObservables(time=obs.time, value=obs.value, mask=obs.mask[:-1])

        with pytest.raises(AssertionError):
            InpatientObservables(time=np.array([1.0, 2.0], dtype=time_invalid_dtype),
                                 value=np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
                                 mask=np.array([[True, False, True], [False, True, False]],
                                               dtype=mask_valid_dtype))

        with pytest.raises(AssertionError):
            InpatientObservables(time=np.array([1.0, 2.0], dtype=time_valid_dtype),
                                 value=np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
                                 mask=np.array([[True, False, True], [False, True, False]],
                                               dtype=mask_invalid_dtype))

    def test_fixture_sorted(self, inpatient_observables: InpatientObservables):
        assert np.all(np.diff(inpatient_observables.time) >= 0)

    def test_len(self, inpatient_observables: InpatientObservables):
        assert len(inpatient_observables) == len(inpatient_observables.time)

    def test_equal(self, inpatient_observables: InpatientObservables):
        assert inpatient_observables.equals(deepcopy(inpatient_observables))
        if len(inpatient_observables) == 0:
            return
        time = inpatient_observables.time.copy()
        time[-1] = time[-1] + 1
        c1 = eqx.tree_at(lambda x: x.time, inpatient_observables,
                         time)
        assert not inpatient_observables.equals(c1)

        value = inpatient_observables.value.copy()
        value[-1, -1] = np.nan
        c2 = eqx.tree_at(lambda x: x.value, inpatient_observables,
                         value)
        assert not inpatient_observables.equals(c2)

        mask = inpatient_observables.mask.astype(int)
        c3 = eqx.tree_at(lambda x: x.mask, inpatient_observables,
                         mask)
        assert not inpatient_observables.equals(c3)

    def test_as_dataframe(self):
        pass

    def test_groupby_code(self):
        pass

    @pytest.fixture(params=[np.array([5.0]), np.array([0.5, 2.0]), np.array([1.0, 3.0, 5.0])])
    def sep(self, request):
        return request.param

    @pytest.fixture
    def segmented_inpatient_observables(self, inpatient_observables: InpatientObservables, sep):
        return SegmentedInpatientObservables.from_observables(inpatient_observables, sep)

    def test_segmentation_concat(self, inpatient_observables: InpatientObservables,
                                 segmented_inpatient_observables: SegmentedInpatientObservables, sep: np.array):
        seg = segmented_inpatient_observables._segments
        assert len(seg) == len(sep) + 1
        assert sum(len(s) for s in seg) == len(inpatient_observables)
        assert sum(s.time.size + s.value.size + s.mask.size for s in seg) == (inpatient_observables.time.size +
                                                                              inpatient_observables.value.size +
                                                                              inpatient_observables.mask.size)
        assert inpatient_observables.equals(InpatientObservables.concat(seg))
        assert all(seg[i].time.max() <= sep[i] for i in range(len(sep)) if len(seg[i]) > 0)
        assert all(seg[i + 1].time.min() >= sep[i] for i in range(len(sep)) if len(seg[i + 1]) > 0)

    def test_hf5_group_serialization(self, inpatient_observables: InpatientObservables, hf5_group: tb.Group):
        inpatient_observables.to_hdf_group(hf5_group)
        assert inpatient_observables.equals(InpatientObservables.from_hdf_group(hf5_group))

    def test_hd5_group_serialization_segmented(self, segmented_inpatient_observables: SegmentedInpatientObservables,
                                               hf5_group: tb.Group):
        segmented_inpatient_observables.to_hdf_group(hf5_group)
        assert segmented_inpatient_observables.equals(SegmentedInpatientObservables.from_hdf_group(hf5_group))

    @pytest.mark.parametrize("ntype", ['N', 'B', 'C', 'O'])
    def test_type_aggregator(self, ntype: NumericalTypeHint):
        aggregator = InpatientObservables.type_hint_aggregator()
        assert isinstance(aggregator, dict)
        assert isinstance(aggregator[ntype], Callable)
        assert np.isscalar(aggregator[ntype](np.array([1, 2, 3])))

    @pytest.mark.parametrize("mask", [np.array([1, 0, 1], dtype=bool)])
    @pytest.mark.parametrize("x,ntype,out", [(np.array([1.0, 2.0, 3.0]), 'N', 2.0),
                                             (np.array([1, 0, 1]), 'B', 1.0),
                                             (np.array([0, 1, 0]), 'B', 0.0),
                                             (np.array([0, 1, 0]), 'C', 0),
                                             (np.array([1, 0, 1]), 'C', 1),
                                             (np.array([0, 1, 0]), 'O', 0),
                                             (np.array([1, 0, 1]), 'O', 1),
                                             (np.array([1, 0, 0]), 'O', 1)])
    def test_time_binning_aggregate(self, x, mask, ntype, out):
        x = x.reshape(-1, 1)
        mask = mask.reshape(-1, 1)
        ntype = np.array([ntype])
        out = np.array([out])

        assert np.array_equal(InpatientObservables._time_binning_aggregate(x, mask, ntype), out)
        # if mask is all zeros, then the result is nan.
        assert np.isnan(InpatientObservables._time_binning_aggregate(x, np.zeros_like(mask, dtype=bool), ntype)).all()

        with pytest.raises(AssertionError):
            InpatientObservables._time_binning_aggregate(x, mask.squeeze(), ntype)

        with pytest.raises(AssertionError):
            InpatientObservables._time_binning_aggregate(x.squeeze(), mask, ntype)

    @pytest.mark.parametrize("hours", [1.0, 2.0, 3.0])
    def test_time_binning(self, hours: float, inpatient_observables: InpatientObservables,
                          obs_scheme: NumericScheme):
        if len(inpatient_observables) == 0:
            pytest.skip("No observations to test")

        binned = inpatient_observables.time_binning(hours, obs_scheme.type_array)
        assert np.all(binned.time % hours == 0.0)
        assert sorted(binned.time) == binned.time.tolist()

        # test causality. inject np.inf to the time array and ensure that
        # values in future time bins do not leak into the past ones.
        for i, ti in enumerate(inpatient_observables.time):
            val = inpatient_observables.value.copy()
            # NUMERIC_OBSERVATION_CODE_INDEX is aggregated with mean.
            val[i:, NUMERIC_OBSERVATION_CODE_INDEX] = np.inf
            obs = eqx.tree_at(lambda x: x.value, inpatient_observables, val)
            binned = obs.time_binning(hours, obs_scheme.type_array)
            assert np.all(binned.value[binned.time < ti] < np.inf)

            mask = inpatient_observables.mask.copy()
            if mask[i:, NUMERIC_OBSERVATION_CODE_INDEX].sum() > 0:
                assert np.any(np.isinf(binned.value[binned.time >= ti]))


class TestLeadingObservableExtractor:

    @pytest.mark.parametrize("leading_hours", [[1.0], [2.0], [1.0, 2.0], [1.0, 2.0, 3.0]])
    def test_len(self, obs_scheme: NumericScheme, leading_hours: List[float]):
        extractor = leading_observables_extractor(obs_scheme=obs_scheme,
                                                  leading_hours=leading_hours)
        assert len(extractor) == len(leading_hours)

    @pytest.mark.parametrize("leading_hours", [[1.0], [2.0], [1.0, 2.0], [1.0, 2.0, 3.0]])
    @pytest.mark.parametrize("entry_neglect_window", [0.0, 1.0, 2.0])
    @pytest.mark.parametrize("recovery_window", [0.0, 1.0, 2.0])
    @pytest.mark.parametrize("minimum_acquisitions", [0, 1, 2, 3])
    def test_init(self, obs_scheme: NumericScheme, leading_hours: List[float],
                  entry_neglect_window: float, recovery_window: float,
                  minimum_acquisitions: int):
        if len(leading_hours) < 2:
            pytest.skip("Not enough leading hours to test")

        with pytest.raises(AssertionError):
            # leading hours must be sorted
            leading_observables_extractor(obs_scheme=obs_scheme,
                                          leading_hours=list(reversed(leading_hours)),
                                          entry_neglect_window=entry_neglect_window,
                                          recovery_window=recovery_window,
                                          minimum_acquisitions=minimum_acquisitions,
                                          code_index=BINARY_OBSERVATION_CODE_INDEX)

        with pytest.raises(AssertionError):
            # categorical and numerical codes are not supported, yet.
            leading_observables_extractor(obs_scheme=obs_scheme,
                                          leading_hours=leading_hours,
                                          entry_neglect_window=entry_neglect_window,
                                          recovery_window=recovery_window,
                                          minimum_acquisitions=minimum_acquisitions,
                                          code_index=CATEGORICAL_OBSERVATION_CODE_INDEX)

        with pytest.raises(AssertionError):
            # categorical and numerical codes are not supported, yet.
            leading_observables_extractor(obs_scheme=obs_scheme,
                                          leading_hours=leading_hours,
                                          entry_neglect_window=entry_neglect_window,
                                          recovery_window=recovery_window,
                                          minimum_acquisitions=minimum_acquisitions,
                                          code_index=NUMERIC_OBSERVATION_CODE_INDEX)

    def test_index2code(self):
        pass

    def test_index2desc(self):
        pass

    def test_code2index(self):
        pass

    @pytest.mark.parametrize("leading_hours", [[1.0], [2.0], [1.0, 2.0], [1.0, 2.0, 3.0]])
    @pytest.mark.parametrize("code_index", [BINARY_OBSERVATION_CODE_INDEX, ORDINAL_OBSERVATION_CODE_INDEX])
    def test_empty(self, obs_scheme: NumericScheme, leading_hours: List[float], code_index: int):
        extractor = leading_observables_extractor(obs_scheme=obs_scheme,
                                                  leading_hours=leading_hours,
                                                  code_index=code_index)
        empty = extractor.empty()
        assert len(extractor) == len(leading_hours)
        assert empty.value.shape[1] == len(extractor)
        assert empty.mask.shape[1] == len(extractor)

        assert len(empty) == 0
        assert empty.time.size == 0
        assert empty.value.size == 0
        assert empty.mask.size == 0

    @pytest.mark.parametrize("x, y", [(np.array([1., 2., 3.]),
                                       np.array([[1., 2., 3.],
                                                 [2., 3., np.nan],
                                                 [3., np.nan, np.nan]]))])
    def test_nan_concat_leading_windows(self, x, y):
        assert np.array_equal(LeadingObservableExtractor._nan_concat_leading_windows(x), y,
                              equal_nan=True)

    @pytest.mark.parametrize("x, y", [(np.array([[1., 2., 3.],
                                                 [2., 3., np.nan],
                                                 [3., np.nan, np.nan]]), np.array([1., 1., 1.])),
                                      (np.array([[1., 2., -3.],
                                                 [0., 0., np.nan],
                                                 [np.nan, np.nan, np.nan]]), np.array([1., 0., np.nan]))
                                      ])
    def test_nan_agg_nonzero(self, x, y):
        assert np.array_equal(LeadingObservableExtractor._nan_agg_nonzero(x, axis=1), y,
                              equal_nan=True)

    @pytest.mark.parametrize("x, y", [(np.array([[1., 2., 3.],
                                                 [2., 3., np.nan],
                                                 [3., np.nan, np.nan]]), np.array([3., 3., 3.])),
                                      (np.array([[1., 2., -3.],
                                                 [0., 0., np.nan],
                                                 [np.nan, np.nan, np.nan]]), np.array([2., 0., np.nan]))
                                      ])
    def test_nan_agg_max(self, x, y):
        assert np.array_equal(LeadingObservableExtractor._nan_agg_max(x, axis=1), y,
                              equal_nan=True)

    @pytest.mark.parametrize("minimum_acquisitions", [0, 1, 2, 3])
    def test_filter_first_acquisitions(self, inpatient_observables: InpatientObservables,
                                       obs_scheme: NumericScheme,
                                       minimum_acquisitions: int):
        lead_extractor = leading_observables_extractor(obs_scheme=obs_scheme,
                                                       minimum_acquisitions=minimum_acquisitions)
        m = lead_extractor.filter_first_acquisitions(len(inpatient_observables), minimum_acquisitions)
        n_affected = min(minimum_acquisitions, len(inpatient_observables))
        assert (~m).sum() == n_affected
        assert (~m)[n_affected:].sum() == 0

    @pytest.mark.parametrize("entry_neglect_window", [0.0, 1.0, 2.0])
    def test_neutralize_entry_neglect_window(self, inpatient_observables: InpatientObservables,
                                             obs_scheme: NumericScheme,
                                             entry_neglect_window: int):
        lead_extractor = leading_observables_extractor(obs_scheme=obs_scheme,
                                                       entry_neglect_window=entry_neglect_window)
        t = inpatient_observables.time
        m = lead_extractor.filter_entry_neglect_window(t, entry_neglect_window)
        n_affected = np.sum(t <= entry_neglect_window)
        assert (~m).sum() == n_affected
        assert (~m)[n_affected:].sum() == 0

    @pytest.mark.parametrize("recovery_window", [0.0, 1.0, 2.0])
    def test_neutralize_recovery_window(self, inpatient_observables: InpatientObservables,
                                        obs_scheme: NumericScheme,
                                        recovery_window: float):
        if len(inpatient_observables) == 0:
            pytest.skip("No observations to test")

        lead_extractor = leading_observables_extractor(obs_scheme=obs_scheme,
                                                       entry_neglect_window=recovery_window)
        x = inpatient_observables.value[:, lead_extractor.config.code_index]
        t = inpatient_observables.time
        m = lead_extractor.filter_recovery_window(t, x, recovery_window)

        arg_x_recovery = np.flatnonzero((x[:-1] != 0) & (x[1:] == 0))
        assert np.isnan(x).sum() == 0
        for i, arg in enumerate(arg_x_recovery):
            last = len(t)
            if i < len(arg_x_recovery) - 1:
                last = arg_x_recovery[i + 1]

            ti = t[arg + 1:last] - t[arg]
            mi = m[arg + 1:last]
            n_affected = np.sum(ti <= recovery_window)

            assert (~mi).sum() == n_affected

            assert (~mi)[n_affected:].sum() == 0

    @pytest.mark.parametrize("minimum_acquisitions", [0, 100])
    @pytest.mark.parametrize("entry_neglect_window", [0.0, 1000.0])
    @pytest.mark.parametrize("recovery_window", [0.0, 10000.0])
    def test_mask_noisy_observations(self, inpatient_observables: InpatientObservables,
                                     obs_scheme: NumericScheme,
                                     minimum_acquisitions: int,
                                     entry_neglect_window: int,
                                     recovery_window: float):
        lead_extractor = leading_observables_extractor(obs_scheme=obs_scheme,
                                                       minimum_acquisitions=minimum_acquisitions,
                                                       entry_neglect_window=entry_neglect_window,
                                                       recovery_window=recovery_window)
        x = inpatient_observables.value[:, lead_extractor.config.code_index]
        t = inpatient_observables.time
        qual = f'{lead_extractor.__module__}.{type(lead_extractor).__qualname__}'
        with mock.patch(f'{qual}.filter_first_acquisitions') as mocker1:
            with mock.patch(f'{qual}.filter_entry_neglect_window') as mocker2:
                with mock.patch(f'{qual}.filter_recovery_window') as mocker3:
                    lead_extractor.mask_noisy_observations(t, x, minimum_acquisitions=minimum_acquisitions,
                                                           entry_neglect_window=entry_neglect_window,
                                                           recovery_window=recovery_window)
                    mocker1.assert_called_once_with(len(t), minimum_acquisitions)
                    mocker2.assert_called_once_with(t, entry_neglect_window)
                    mocker3.assert_called_once_with(t, x, recovery_window)

    @pytest.mark.parametrize("code_index", [BINARY_OBSERVATION_CODE_INDEX, ORDINAL_OBSERVATION_CODE_INDEX])
    def test_extract_leading_window(self, inpatient_observables: InpatientObservables,
                                    obs_scheme: NumericScheme, code_index: int):
        lead_extractor = leading_observables_extractor(obs_scheme=obs_scheme,
                                                       code_index=code_index)
        x = inpatient_observables.value[:, lead_extractor.config.code_index]
        m = inpatient_observables.mask[:, lead_extractor.config.code_index]
        t = inpatient_observables.time
        lead = lead_extractor(inpatient_observables)

        assert len(lead) == len(inpatient_observables)
        assert lead.value.shape[1] == len(lead_extractor)

        for iw, w in enumerate(lead_extractor.config.leading_hours):
            for i in range(len(t)):
                delta_t = t[i:] - t[i]
                indexer = (delta_t <= w) & m[i:]
                xi = np.where(indexer, x[i:], np.nan)
                yi = lead.value[i, iw]
                # if all are nan, then the lead is nan.
                if np.isnan(xi).all():
                    assert np.isnan(yi)
                elif code_index == BINARY_OBSERVATION_CODE_INDEX:
                    assert yi == xi[~np.isnan(xi)].any()
                elif code_index == ORDINAL_OBSERVATION_CODE_INDEX:
                    assert yi == xi[~np.isnan(xi)].max()
                else:
                    assert False

    def test_apply(self):
        pass


class TestInpatientInput:

    def test_init(self):
        pass

    def test_apply(self):
        # TODO: priority.
        pass

    @pytest.mark.parametrize("n", [0, 1, 100, 501])
    @pytest.mark.parametrize("p", [1, 100, 501])
    def test_hf5_group_serialization(self, n: int, p: int, hf5_group: tb.Group):
        input = inpatient_binary_input(n, p)
        input.to_hdf_group(hf5_group)
        loaded_input = InpatientInput.from_hdf_group(hf5_group)
        assert len(input) == len(loaded_input)
        assert input.equals(loaded_input)

    def test_empty(self):
        pass


class TestInpatientInterventions:

    def test_hf5_group_serialization(self, inpatient_interventions: InpatientInterventions, hf5_group: tb.Group):
        inpatient_interventions.to_hdf_group(hf5_group)
        assert inpatient_interventions.equals(InpatientInterventions.from_hdf_group(hf5_group))

    def test_timestamps(self, inpatient_interventions: InpatientInterventions):
        timestamps = inpatient_interventions.timestamps
        assert len(timestamps) == len(set(timestamps))
        assert all(isinstance(t, float) for t in timestamps)
        assert list(sorted(timestamps)) == timestamps
        ts = set()
        if inpatient_interventions.hosp_procedures is not None:
            ts.update(inpatient_interventions.hosp_procedures.starttime.tolist())
            ts.update(inpatient_interventions.hosp_procedures.endtime.tolist())
        if inpatient_interventions.icu_procedures is not None:
            ts.update(inpatient_interventions.icu_procedures.starttime.tolist())
            ts.update(inpatient_interventions.icu_procedures.endtime.tolist())
        if inpatient_interventions.icu_inputs is not None:
            ts.update(inpatient_interventions.icu_inputs.starttime.tolist())
            ts.update(inpatient_interventions.icu_inputs.endtime.tolist())
        assert ts == set(timestamps)


class TestSegmentedInpatientInterventions:

    def test_from_inpatient_interventions(self, inpatient_interventions, hosp_proc_scheme, icu_proc_scheme,
                                          icu_inputs_scheme):
        assert all(
            isinstance(scheme, CodingScheme) for scheme in (hosp_proc_scheme, icu_proc_scheme, icu_inputs_scheme))
        schemes = {"hosp_procedures": hosp_proc_scheme,
                   "icu_procedures": icu_proc_scheme,
                   "icu_inputs": icu_inputs_scheme}
        seg = SegmentedInpatientInterventions.from_interventions(inpatient_interventions, LENGTH_OF_STAY,
                                                                 hosp_procedures_size=len(hosp_proc_scheme),
                                                                 icu_procedures_size=len(icu_proc_scheme),
                                                                 icu_inputs_size=len(icu_inputs_scheme),
                                                                 maximum_padding=1)
        assert abs(len(seg.time) - len(inpatient_interventions.timestamps)) <= 2
        for k in ("hosp_procedures", "icu_procedures", "icu_inputs"):
            if getattr(inpatient_interventions, k) is not None:
                seg_intervention = getattr(seg, k)
                assert seg_intervention.shape == (len(seg.time) - 1, len(schemes[k]))

            else:
                assert getattr(seg, k) is None

    @pytest.mark.parametrize("array", [np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])])
    @pytest.mark.parametrize("value", [0.0, np.nan])
    @pytest.mark.parametrize("maximum_padding", [1, 2, 3, 5, 10, 100])
    def test_pad_array(self, array, value, maximum_padding):
        y = SegmentedInpatientInterventions.pad_array(array, value=value, maximum_padding=maximum_padding)
        assert len(y) >= len(array)
        assert len(y) <= len(array) + maximum_padding
        assert np.all(y[:len(array)] == array)
        if np.isnan(value):
            assert np.isnan(y[len(array):]).all()
        else:
            assert np.all(y[len(array):] == value)

        if len(array) == maximum_padding or maximum_padding == 1:
            assert len(y) == len(array)

    @pytest.mark.parametrize("test_target", ["hosp_procedures", "icu_procedures", "icu_inputs"])
    def test_segmentation(self, inpatient_interventions: InpatientInterventions, test_target: str,
                          hosp_proc_scheme, icu_proc_scheme, icu_inputs_scheme):
        inpatient_intervention = getattr(inpatient_interventions, test_target)
        if inpatient_intervention is None or inpatient_intervention.starttime.size == 0:
            pytest.skip("No interventions to test")

        scheme = {"hosp_procedures": hosp_proc_scheme,
                  "icu_procedures": icu_proc_scheme,
                  "icu_inputs": icu_inputs_scheme}[test_target]
        seg = SegmentedInpatientInterventions._segment(inpatient_intervention.starttime,
                                                       inpatient_intervention,
                                                       len(scheme))
        for i, t in enumerate(inpatient_intervention.starttime):
            assert np.array_equal(inpatient_intervention(t, len(scheme)), seg[i])

    def test_hf5_group_serialization(self, segmented_inpatient_interventions: SegmentedInpatientInterventions,
                                     hf5_group: tb.Group):
        segmented_inpatient_interventions.to_hdf_group(hf5_group)
        assert segmented_inpatient_interventions.equals(SegmentedInpatientInterventions.from_hdf_group(hf5_group))


class TestAdmission:

    def test_hf5_group_serialization(self, admission: Admission, hf5_group: tb.Group):
        admission.to_hdf_group(hf5_group)
        assert admission.equals(Admission.from_hdf_group(hf5_group))

    def test_interval_hours(self):
        pass

    def test_interval_days(self):
        pass

    def test_days_since(self):
        pass


class TestSegmentedAdmission:

    def test_from_admission(self, admission: Admission, segmented_admission: SegmentedAdmission):
        assert segmented_admission.admission_id == admission.admission_id
        assert segmented_admission.admission_dates == admission.admission_dates
        assert segmented_admission.dx_codes == admission.dx_codes
        assert segmented_admission.dx_codes_history == admission.dx_codes_history
        assert segmented_admission.outcome == admission.outcome
        if admission.observables is not None:
            assert InpatientObservables.concat(segmented_admission.observables._segments).equals(admission.observables)
        if admission.leading_observable is not None:
            assert InpatientObservables.concat(segmented_admission.leading_observable._segments).equals(
                admission.leading_observable)
        if admission.interventions is None:
            return
        interventions = admission.interventions
        seg_interventions = segmented_admission.interventions
        timestamps = admission.interventions.timestamps

        time = sorted(set([0.0] + timestamps + [LENGTH_OF_STAY]))
        for i, (start, end) in enumerate(zip(time[:-1], time[1:])):
            if interventions.hosp_procedures is not None:
                p = seg_interventions.hosp_procedures.shape[1]
                hosp_procedure = interventions.hosp_procedures(start, p)
                seg_hosp_procedure = seg_interventions.hosp_procedures[i]
                assert np.array_equal(seg_hosp_procedure, hosp_procedure)
            if interventions.icu_procedures is not None:
                p = seg_interventions.icu_procedures.shape[1]
                icu_procedure = interventions.icu_procedures(start, p)
                seg_icu_procedure = seg_interventions.icu_procedures[i]
                assert np.array_equal(seg_icu_procedure, icu_procedure)
            if interventions.icu_inputs is not None:
                p = seg_interventions.icu_inputs.shape[1]
                icu_input = interventions.icu_inputs(start, p)
                seg_icu_input = seg_interventions.icu_inputs[i]
                assert np.array_equal(seg_icu_input, icu_input)

            if admission.observables is not None:
                obs_time = admission.observables.time
                segment_time = segmented_admission.observables[i].time
                assert np.all((start <= segment_time) & (segment_time <= end))
                if i == len(time) - 2:
                    time_mask = (obs_time >= start) & (obs_time <= end)
                else:
                    time_mask = (obs_time >= start) & (obs_time < end)
                segment_val = segmented_admission.observables[i].value
                segment_mask = segmented_admission.observables[i].mask
                val = admission.observables.value[time_mask]
                mask = admission.observables.mask[time_mask]
                assert np.array_equal(segment_val, val)
                assert np.array_equal(segment_mask, mask)

    def test_hf5_group_serialization(self, segmented_admission: SegmentedAdmission, hf5_group: tb.Group):
        segmented_admission.to_hdf_group(hf5_group)
        assert segmented_admission.equals(SegmentedAdmission.from_hdf_group(hf5_group))


class TestStaticInfo:

    def test_hf5_group_serialization(self, static_info: StaticInfo, hf5_group: tb.Group):
        static_info.to_hdf_group(hf5_group)
        assert static_info.equals(StaticInfo.from_hdf_group(hf5_group))

    def test_constant_attributes(self):
        pass

    def test_constant_vec(self):
        pass

    def test_dynamic_attributes(self):
        pass


class TestPatient:

    def test_d2d_interval_days(self):
        pass

    def test_outcome_frequency(self):
        pass

    def test_hf5_group_serialization(self, patient: Patient, hf5_group: tb.Group):
        patient.to_hdf_group(hf5_group)
        assert patient.equals(Patient.from_hdf_group(hf5_group))


class TestSegmentedPatient:

    def test_d2d_interval_days(self):
        pass

    def test_outcome_frequency(self):
        pass

    def test_hf5_group_serialization(self, segmented_patient: SegmentedPatient, hf5_group: tb.Group):
        segmented_patient.to_hdf_group(hf5_group)
        assert segmented_patient.equals(Patient.from_hdf_group(hf5_group))
