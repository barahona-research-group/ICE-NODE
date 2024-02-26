from copy import deepcopy
from typing import List
from unittest import mock

import equinox as eqx
import numpy as np
import numpy.random as nrand
import pytest

from lib.ehr import CodingScheme
from lib.ehr.concepts import (InpatientObservables, LeadingObservableExtractorConfig, LeadingObservableExtractor,
                              InpatientInput, InpatientInterventions)


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


FIXTURE_OBS_MAX_TIME = 10.0
BINARY_OBSERVATION_CODE_INDEX = 0


@pytest.fixture(params=[0, 1, 100, 501])
def inpatient_observables(obs_scheme: CodingScheme, request):
    d = len(obs_scheme)
    n = request.param
    t = np.array(sorted(nrand.choice(np.linspace(0, FIXTURE_OBS_MAX_TIME, 1000), replace=False, size=n)))
    v = np.zeros((n, d))
    i = BINARY_OBSERVATION_CODE_INDEX
    v[:, i: i + 1] = nrand.binomial(1, 0.5, size=n).astype(bool).reshape(-1, 1)
    v[:, i + 1:] = nrand.randn(n, d - 1)

    mask = nrand.binomial(1, 0.5, size=(n, d)).astype(bool)
    return InpatientObservables(t, v, mask)


def inpatient_binary_input(n: int, p: int):
    starttime = np.array(
        sorted(nrand.choice(np.linspace(0, FIXTURE_OBS_MAX_TIME, max(1000, n)), replace=False, size=n)))
    endtime = starttime + nrand.uniform(0, FIXTURE_OBS_MAX_TIME - starttime, size=(n,))
    code_index = nrand.choice(p, size=n, replace=True)
    return InpatientInput(starttime=starttime, endtime=endtime, code_index=code_index)


def inpatient_rated_input(n: int, p: int):
    bin_input = inpatient_binary_input(n, p)
    return eqx.tree_at(lambda x: x.rate, bin_input,
                       nrand.uniform(0, 1, size=(n,)))


@pytest.fixture(params=[0, 1, 501])
def icu_inputs(icu_inputs_scheme_: CodingScheme, request):
    return inpatient_binary_input(request.param, len(icu_inputs_scheme_))


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

    def test_dataframe_serialization(self, inpatient_interventions: InpatientInterventions):
        dfs = inpatient_interventions.to_dataframes()
        assert inpatient_interventions.equals(InpatientInterventions.from_dataframes(dfs))

    def test_timestamps(self):
        pass


class TestSegmentedInpatientInterventions:

    def test_from_inpatient_interventions(self):
        pass

    def test_pad_array(self):
        pass

    def test_segmentation(self):
        pass


class TestAdmission:

    def test_serialization(self):
        pass

    def test_interval_hours(self):
        pass

    def test_interval_days(self):
        pass

    def test_days_since(self):
        pass


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
