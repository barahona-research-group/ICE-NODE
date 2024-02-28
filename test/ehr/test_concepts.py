from copy import deepcopy
from typing import List
from unittest import mock

import equinox as eqx
import numpy as np
import numpy.random as nrand
import pandas as pd
import pytest

from lib.ehr import CodingScheme, CodesVector, OutcomeExtractor
from lib.ehr.concepts import (InpatientObservables, LeadingObservableExtractorConfig, LeadingObservableExtractor,
                              InpatientInput, InpatientInterventions, SegmentedInpatientInterventions, Admission,
                              SegmentedAdmission)


@pytest.fixture
def dx_scheme_(dx_scheme: str) -> CodingScheme:
    return CodingScheme.from_name(dx_scheme)


@pytest.fixture
def outcome_extractor_(outcome_extractor: str) -> CodingScheme:
    return OutcomeExtractor.from_name(outcome_extractor)


@pytest.fixture
def obs_scheme(observation_scheme: str) -> CodingScheme:
    return CodingScheme.from_name(observation_scheme)


@pytest.fixture
def icu_inputs_scheme_(icu_inputs_scheme: str) -> CodingScheme:
    return CodingScheme.from_name(icu_inputs_scheme)


@pytest.fixture
def icu_proc_scheme_(icu_proc_scheme: str) -> CodingScheme:
    return CodingScheme.from_name(icu_proc_scheme)


@pytest.fixture
def hosp_proc_scheme_(hosp_proc_scheme: str) -> CodingScheme:
    return CodingScheme.from_name(hosp_proc_scheme)


LENGTH_OF_STAY = 10.0
BINARY_OBSERVATION_CODE_INDEX = 0


@pytest.fixture
def dx_codes(dx_scheme_: CodingScheme):
    v = nrand.binomial(1, 0.5, size=len(dx_scheme_)).astype(bool)
    return CodesVector(vec=v, scheme=dx_scheme_.name)


@pytest.fixture
def dx_codes_history(dx_codes: CodesVector):
    v = nrand.binomial(1, 0.5, size=len(dx_codes)).astype(bool)
    return CodesVector(vec=v + dx_codes.vec, scheme=dx_codes.scheme)


@pytest.fixture
def outcome(dx_codes: CodesVector, outcome_extractor_: OutcomeExtractor):
    return outcome_extractor_.mapcodevector(dx_codes)


@pytest.fixture(params=[0, 1, 501])
def inpatient_observables(obs_scheme: CodingScheme, request):
    d = len(obs_scheme)
    n = request.param
    t = np.array(sorted(nrand.choice(np.linspace(0, LENGTH_OF_STAY, 1000), replace=False, size=n)))
    v = np.zeros((n, d))
    i = BINARY_OBSERVATION_CODE_INDEX
    v[:, i: i + 1] = nrand.binomial(1, 0.5, size=n).astype(bool).reshape(-1, 1)
    v[:, i + 1:] = nrand.randn(n, d - 1)

    mask = nrand.binomial(1, 0.5, size=(n, d)).astype(bool)
    return InpatientObservables(t, v, mask)


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


@pytest.fixture(params=[0, 1, 501])
def icu_inputs(icu_inputs_scheme_: CodingScheme, request):
    return inpatient_rated_input(request.param, len(icu_inputs_scheme_))


@pytest.fixture(params=[0, 1, 501])
def icu_proc(icu_proc_scheme_: CodingScheme, request):
    return inpatient_binary_input(request.param, len(icu_proc_scheme_))


@pytest.fixture(params=[0, 1, 501])
def hosp_proc(hosp_proc_scheme_: CodingScheme, request):
    return inpatient_binary_input(request.param, len(hosp_proc_scheme_))


@pytest.fixture(params=[0, 1, 2, -1])
def inpatient_interventions(hosp_proc, icu_proc, icu_inputs, request):
    whoisnull = request.param
    return InpatientInterventions(None if whoisnull == 0 else hosp_proc,
                                  None if whoisnull == 1 else icu_proc,
                                  None if whoisnull == 2 else icu_inputs)


@pytest.fixture
def segmented_inpatient_interventions(inpatient_interventions: InpatientInterventions, hosp_proc_scheme_,
                                      icu_proc_scheme_,
                                      icu_inputs_scheme_) -> SegmentedInpatientInterventions:
    return SegmentedInpatientInterventions.from_interventions(inpatient_interventions, LENGTH_OF_STAY,
                                                              hosp_procedures_size=len(hosp_proc_scheme_),
                                                              icu_procedures_size=len(icu_proc_scheme_),
                                                              icu_inputs_size=len(icu_inputs_scheme_))


def leading_observables_extractor(observation_scheme: str, leading_hours: List[float] = (1.0,),
                                  entry_neglect_window: float = 0.0,
                                  recovery_window: float = 0.0,
                                  minimum_acquisitions: int = 0,
                                  aggregation: str = "any") -> LeadingObservableExtractor:
    config = LeadingObservableExtractorConfig(code_index=BINARY_OBSERVATION_CODE_INDEX,
                                              scheme=observation_scheme,
                                              entry_neglect_window=entry_neglect_window,
                                              recovery_window=recovery_window,
                                              minimum_acquisitions=minimum_acquisitions,
                                              aggregation=aggregation,
                                              leading_hours=leading_hours)
    return LeadingObservableExtractor(config=config)


@pytest.fixture
def leading_observable(observation_scheme: str, inpatient_observables: InpatientObservables) -> InpatientObservables:
    return leading_observables_extractor(observation_scheme=observation_scheme)(inpatient_observables)


@pytest.fixture
def admission(dx_codes: CodesVector, dx_codes_history: CodesVector,
              outcome: CodesVector, inpatient_observables: InpatientObservables,
              inpatient_interventions: InpatientInterventions,
              leading_observable: InpatientObservables) -> Admission:
    admission_id = 'test'
    admission_date = pd.to_datetime('now')
    discharge_date = pd.to_datetime(admission_date + pd.to_timedelta(LENGTH_OF_STAY, unit='H'))
    return Admission(admission_id=admission_id, admission_dates=(admission_date, discharge_date),
                     dx_codes=dx_codes, dx_codes_history=dx_codes_history, outcome=outcome,
                     observables=inpatient_observables, interventions=inpatient_interventions,
                     leading_observable=leading_observable)


@pytest.fixture(params=[1, 50])
def segmented_admission(admission: Admission, hosp_proc_scheme_, icu_proc_scheme_, icu_inputs_scheme_,
                        request) -> SegmentedAdmission:
    maximum_padding = request.param
    return SegmentedAdmission.from_admission(admission,
                                             hosp_procedures_size=len(hosp_proc_scheme_),
                                             icu_procedures_size=len(icu_proc_scheme_),
                                             icu_inputs_size=len(icu_inputs_scheme_),
                                             maximum_padding=maximum_padding)


class TestInpatientObservables:

    def test_empty(self, obs_scheme: str):
        obs = InpatientObservables.empty(len(obs_scheme))
        assert len(obs.time) == 0
        assert len(obs.value) == 0
        assert len(obs.mask) == 0
        assert all(a.shape == (0, len(obs_scheme)) for a in [obs.value, obs.mask])

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

    def test_dataframe_serialization(self, inpatient_observables: InpatientObservables):
        df = inpatient_observables.to_dataframe()
        assert len(df) == len(inpatient_observables)
        assert inpatient_observables.equals(InpatientObservables.from_dataframe(df))

    def test_as_dataframe(self):
        pass

    def test_groupby_code(self):
        pass

    @pytest.mark.parametrize("sep", [np.array([5.0]), np.array([0.5, 2.0]), np.array([1.0, 3.0, 5.0])])
    def test_segmentation_concat(self, inpatient_observables: InpatientObservables, sep: np.array):
        seg = inpatient_observables.segment(sep)
        assert len(seg) == len(sep) + 1
        assert sum(len(s) for s in seg) == len(inpatient_observables)
        assert sum(s.time.size + s.value.size + s.mask.size for s in seg) == (inpatient_observables.time.size +
                                                                              inpatient_observables.value.size +
                                                                              inpatient_observables.mask.size)
        assert inpatient_observables.equals(InpatientObservables.concat(seg))
        assert all(seg[i].time.max() <= sep[i] for i in range(len(sep)) if len(seg[i]) > 0)
        assert all(seg[i + 1].time.min() >= sep[i] for i in range(len(sep)) if len(seg[i + 1]) > 0)

    def test_time_binning(self):
        pass


class TestLeadingObservableExtractor:

    @pytest.mark.parametrize("leading_hours", [[1.0], [2.0], [1.0, 2.0], [1.0, 2.0, 3.0]])
    def test_len(self, observation_scheme: str, leading_hours: List[float]):
        extractor = leading_observables_extractor(observation_scheme=observation_scheme,
                                                  leading_hours=leading_hours)
        assert len(extractor) == len(leading_hours)

    @pytest.mark.parametrize("leading_hours", [[1.0], [2.0], [1.0, 2.0], [1.0, 2.0, 3.0]])
    @pytest.mark.parametrize("entry_neglect_window", [0.0, 1.0, 2.0])
    @pytest.mark.parametrize("recovery_window", [0.0, 1.0, 2.0])
    @pytest.mark.parametrize("minimum_acquisitions", [0, 1, 2, 3])
    @pytest.mark.parametrize("aggregation", ["any", "max"])
    def test_init(self, observation_scheme: str, leading_hours: List[float],
                  entry_neglect_window: float, recovery_window: float,
                  minimum_acquisitions: int, aggregation: str):
        if len(leading_hours) < 2:
            pytest.skip("Not enough leading hours to test")

        with pytest.raises(AssertionError):
            leading_observables_extractor(observation_scheme=observation_scheme,
                                          leading_hours=list(reversed(leading_hours)),
                                          entry_neglect_window=entry_neglect_window,
                                          recovery_window=recovery_window,
                                          minimum_acquisitions=minimum_acquisitions,
                                          aggregation=aggregation)

    def test_index2code(self):
        pass

    def test_index2desc(self):
        pass

    def test_code2index(self):
        pass

    @pytest.mark.parametrize("leading_hours", [[1.0], [2.0], [1.0, 2.0], [1.0, 2.0, 3.0]])
    def test_empty(self, observation_scheme: str, leading_hours: List[float]):
        extractor = leading_observables_extractor(observation_scheme=observation_scheme,
                                                  leading_hours=leading_hours)
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
                                       observation_scheme: str,
                                       minimum_acquisitions: int):
        lead_extractor = leading_observables_extractor(observation_scheme=observation_scheme,
                                                       minimum_acquisitions=minimum_acquisitions)
        m = lead_extractor.filter_first_acquisitions(len(inpatient_observables), minimum_acquisitions)
        n_affected = min(minimum_acquisitions, len(inpatient_observables))
        assert (~m).sum() == n_affected
        assert (~m)[n_affected:].sum() == 0

    @pytest.mark.parametrize("entry_neglect_window", [0.0, 1.0, 2.0])
    def test_neutralize_entry_neglect_window(self, inpatient_observables: InpatientObservables,
                                             observation_scheme: str,
                                             entry_neglect_window: int):
        lead_extractor = leading_observables_extractor(observation_scheme=observation_scheme,
                                                       entry_neglect_window=entry_neglect_window)
        t = inpatient_observables.time
        m = lead_extractor.filter_entry_neglect_window(t, entry_neglect_window)
        n_affected = np.sum(t <= entry_neglect_window)
        assert (~m).sum() == n_affected
        assert (~m)[n_affected:].sum() == 0

    @pytest.mark.parametrize("recovery_window", [0.0, 1.0, 2.0])
    def test_neutralize_recovery_window(self, inpatient_observables: InpatientObservables,
                                        observation_scheme: str,
                                        recovery_window: float):
        if len(inpatient_observables) == 0:
            pytest.skip("No observations to test")

        lead_extractor = leading_observables_extractor(observation_scheme=observation_scheme,
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
                                     observation_scheme: str,
                                     minimum_acquisitions: int,
                                     entry_neglect_window: int,
                                     recovery_window: float):
        lead_extractor = leading_observables_extractor(observation_scheme=observation_scheme,
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

    def test_extract_leading_window(self, inpatient_observables: InpatientObservables,
                                    observation_scheme: str):
        lead_extractor = leading_observables_extractor(observation_scheme=observation_scheme)
        x = inpatient_observables.value[:, lead_extractor.config.code_index]
        m = inpatient_observables.mask[:, lead_extractor.config.code_index]
        t = inpatient_observables.time
        lead = lead_extractor(inpatient_observables)

        assert len(lead) == len(inpatient_observables)
        assert lead.value.shape[1] == len(lead_extractor)

        for iw, w in enumerate(lead_extractor.config.leading_hours):
            for i in range(len(t)):
                delta_t = t[i:] - t[i]
                xi = np.where((delta_t <= w) & (m[i:]), x[i:], np.nan)
                yi = lead.value[i, iw]
                # if all are nan, then the lead is nan.
                if np.isnan(xi).all():
                    assert np.isnan(yi)
                else:
                    assert yi == xi[~np.isnan(xi)].any()

    def test_apply(self):
        pass


class TestMaskedPerceptron:

    def test_apply(self):
        pass


class TestMaskedSum:

    def test_apply(self):
        pass


class TestMaskedOr:

    def test_apply(self):
        pass


class TestAggregateRepresentation:

    def test_init(self):
        pass

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
    def test_dataframe_serialization(self, n: int, p: int):
        input = inpatient_binary_input(n, p)
        df = input.to_dataframe()
        assert len(df) == n
        assert input.equals(InpatientInput.from_dataframe(df))

    def test_empty(self):
        pass


class TestInpatientInterventions:

    def test_hdf_serialization(self, inpatient_interventions: InpatientInterventions, tmpdir):
        hdf_filename = f'{tmpdir}/test_inpatient_interventions_serialization.h5'
        inpatient_interventions.to_hdf(hdf_filename, key='test_inpatient_interventions')
        with pd.HDFStore(hdf_filename, 'r') as hdf:
            assert inpatient_interventions.equals(
                InpatientInterventions.from_hdf(hdf, key='test_inpatient_interventions'))

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

    def test_from_inpatient_interventions(self, inpatient_interventions, hosp_proc_scheme_, icu_proc_scheme_,
                                          icu_inputs_scheme_):
        schemes = {"hosp_procedures": hosp_proc_scheme_,
                   "icu_procedures": icu_proc_scheme_,
                   "icu_inputs": icu_inputs_scheme_}
        seg = SegmentedInpatientInterventions.from_interventions(inpatient_interventions, LENGTH_OF_STAY,
                                                                 hosp_procedures_size=len(hosp_proc_scheme_),
                                                                 icu_procedures_size=len(icu_proc_scheme_),
                                                                 icu_inputs_size=len(icu_inputs_scheme_))
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
                          hosp_proc_scheme_, icu_proc_scheme_, icu_inputs_scheme_):
        inpatient_intervention = getattr(inpatient_interventions, test_target)
        if inpatient_intervention is None or inpatient_intervention.starttime.size == 0:
            pytest.skip("No interventions to test")

        scheme = {"hosp_procedures": hosp_proc_scheme_,
                  "icu_procedures": icu_proc_scheme_,
                  "icu_inputs": icu_inputs_scheme_}[test_target]
        seg = SegmentedInpatientInterventions._segment(inpatient_intervention.starttime,
                                                       inpatient_intervention,
                                                       len(scheme))
        for i, t in enumerate(inpatient_intervention.starttime):
            assert np.array_equal(inpatient_intervention(t, len(scheme)), seg[i])

    def test_dataframe_serialization(self, segmented_inpatient_interventions: SegmentedInpatientInterventions):
        dfs = segmented_inpatient_interventions.to_dataframes()
        assert segmented_inpatient_interventions.equals(SegmentedInpatientInterventions.from_dataframes(dfs))

    def test_hdf_serialization(self, segmented_inpatient_interventions, tmpdir):
        hdf_filename = f'{tmpdir}/test_segmented_inpatient_interventions_serialization.h5'
        segmented_inpatient_interventions.to_hdf(hdf_filename, key='test_segmented_inpatient_interventions')
        with pd.HDFStore(hdf_filename, 'r') as hdf:
            assert segmented_inpatient_interventions.equals(
                SegmentedInpatientInterventions.from_hdf(hdf, key='test_segmented_inpatient_interventions'))


class TestAdmission:

    def test_serialization(self, admission: Admission, tmpdir: str):
        hdf_filename = f'{tmpdir}/test_admissions_serialization.h5'

        admission.to_hdf(hdf_filename, 'test_admissions')
        with pd.HDFStore(hdf_filename, 'r') as hdf:
            deserialized = Admission.from_hdf_store(hdf, 'test_admissions',
                                                    admission.admission_id)
        assert admission.equals(deserialized)

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
            assert InpatientObservables.concat(segmented_admission.observables).equals(admission.observables)
        if admission.leading_observable is not None:
            assert InpatientObservables.concat(segmented_admission.leading_observable).equals(
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
                assert np.all(start <= segmented_admission.observables[i].time <= end)
                time_mask = (admission.observables.time >= start) & (admission.observables.time <= end)
                segment_val = segmented_admission.observables[i].value
                segment_mask = segmented_admission.observables[i].mask
                val = admission.observables.value[time_mask]
                mask = admission.observables.mask[time_mask]
                assert np.array_equal(segment_val, val)
                assert np.array_equal(segment_mask, mask)

    def test_serialization(self, admission: Admission, tmpdir: str):
        hdf_filename = f'{tmpdir}/test_admissions_serialization.h5'

        admission.to_hdf(hdf_filename, 'test_admissions')
        with pd.HDFStore(hdf_filename, 'r') as hdf:
            deserialized = Admission.from_hdf_store(hdf, 'test_admissions',
                                                    admission.admission_id)
        assert admission.equals(deserialized)


class TestStaticInfo:

    def test_dataframe_serialization(self):
        pass

    def test_constant_attributes(self):
        pass

    def test_constant_vec(self):
        pass


class TestPatient:

    def test_d2d_interval_days(self):
        pass

    def test_outcome_frequency(self):
        pass

    def test_serialization(self):
        pass
