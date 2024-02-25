from copy import deepcopy
from typing import List

import equinox as eqx
import numpy as np
import numpy.random as nrand
import pytest

from lib.ehr import CodingScheme
from lib.ehr.concepts import (InpatientObservables, LeadingObservableExtractorConfig, LeadingObservableExtractor,
                              InpatientInput)


@pytest.fixture
def obs_scheme(observation_scheme: str) -> CodingScheme:
    return CodingScheme.from_name(observation_scheme)


@pytest.fixture(params=[0, 1, 100, 501])
def inpatient_observables(obs_scheme: CodingScheme, request):
    d = len(obs_scheme)
    n = request.param
    t = np.array(sorted(nrand.choice(np.linspace(0, 10, 1000), replace=False, size=n)))
    v = np.zeros((n, d))

    v[:, 0] = nrand.binomial(1, 0.5, size=d).astype(bool)
    v[:, 1:] = nrand.randn(n, d - 1)

    mask = nrand.binomial(1, 0.5, size=(n, d)).astype(bool)
    return InpatientObservables(t, v, mask)


def inpatient_binary_input(n: int, p: int):
    starttime = np.array(sorted(nrand.choice(np.linspace(0, 10, max(1000, n)), replace=False, size=n)))
    endtime = starttime + nrand.uniform(0, 10 - starttime, size=(n,))
    code_index = nrand.choice(p, size=n, replace=True)
    return InpatientInput(starttime=starttime, endtime=endtime, code_index=code_index)


def inpatient_rated_input(n: int, p: int):
    bin_input = inpatient_binary_input(n, p)
    return eqx.tree_at(lambda x: x.rate, bin_input,
                       nrand.uniform(0, 1, size=(n,)))


def leading_observables_extractor(observation_scheme: str, leading_hours: List[float],
                                  entry_neglect_window: float = 0.0,
                                  recovery_window: float = 0.0,
                                  minimum_acquisitions: int = 0,
                                  aggregation: str = "any") -> LeadingObservableExtractor:
    config = LeadingObservableExtractorConfig(code_index=0,
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


@pytest.mark.parametrize("leading_hours", [[1.0], [2.0], [1.0, 2.0], [1.0, 2.0, 3.0]])
@pytest.mark.parametrize("entry_neglect_window", [0.0, 1.0, 2.0])
@pytest.mark.parametrize("recovery_window", [0.0, 1.0, 2.0])
@pytest.mark.parametrize("minimum_acquisitions", [1, 2, 3])
@pytest.mark.parametrize("aggregation", ["any", "max"])
class TestLeadingObservableExtractor:

    def test_len(self, observation_scheme: str, leading_hours: List[float]):
        extractor = leading_observables_extractor(observation_scheme=observation_scheme,
                                                  leading_hours=leading_hours)
        assert len(extractor) == len(leading_hours)

    def test_init(self, observation_scheme: str, leading_hours: List[float],
                  entry_neglect_window: float, recovery_window: float,
                  minimum_acquisitions: int, aggregation: str):
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

    def test_nan_concat_leading_windows(self):
        pass

    def test_nan_agg_nonzero(self):
        pass

    def test_nan_agg_max(self):
        pass

    def test_neutralize_first_acquisitions(self):
        pass

    def test_neutralize_entry_neglect_window(self):
        pass

    def test_neutralize_recovery_window(self):
        pass

    def test_clean_values(self):
        pass

    def test_extract_leading_window(self):
        pass

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

    def test_dataframe_serialization(self):
        pass

    def test_empty(self):
        pass


class TestInpatientInterventions:

    def test_dataframe_serialization(self):
        pass

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
