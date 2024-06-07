from typing import Dict, List, Tuple

import pytest
from lib.metric.stat import VisitsAUC, ObsPredictionLossMetric, OutcomePredictionLossMetric, LeadPredictionLossMetric, \
    Metric, LeadingPredictionAccuracyConfig, LeadingAKIPredictionAccuracy
from lib.ml.artefacts import AdmissionsPrediction
import jax.numpy as jnp


class TestGlobalMetric:
    @pytest.fixture(params=[ObsPredictionLossMetric, OutcomePredictionLossMetric, LeadPredictionLossMetric, VisitsAUC])
    def metric(self, request) -> Metric:
        return request.param()

    @pytest.fixture
    def metric_out(self, metric: Metric, identical_predictions: AdmissionsPrediction) -> Tuple[Tuple[str,...], Tuple[float, ...]]:
        return metric(identical_predictions)

    def test_metric_out(self, metric: ObsPredictionLossMetric, metric_out: Tuple[Tuple[str,...], Tuple[float, ...]]):
        assert isinstance(metric_out[0], tuple)
        assert isinstance(metric_out[1], tuple)
        assert set(metric_out[0][:-1]) == set(metric.column_names())
        assert all(isinstance(v, (jnp.ndarray, float)) for v in metric_out[1])


class TestLeadingAKIPredictionAccuracy:
    # leading_hours: List[float]
    # entry_neglect_window: float
    # minimum_acquisitions: int  # minimum number of acquisitions to consider
    # recovery_window: float = 0.0
    @pytest.fixture(params=[[2.], [1., 2., 3.]], ids=['single_window', 'multiple_windows'])
    def config_leading_hours(self, request):
        return request.param

    @pytest.fixture(params=[0.0, 1.0], ids=['no_neglect', 'neglect_window'])
    def config_entry_neglect_window(self, request):
        return request.param

    @pytest.fixture(params=[0, 2], ids=['no_minimum_acquisitions', 'minimum_acquisitions'])
    def config_minimum_acquisitions(self, request):
        return request.param

    @pytest.fixture(params=[0.0, 1.0], ids=['no_recovery', 'recovery_window'])
    def config_recovery_window(self, request):
        return request.param

    @pytest.fixture
    def config(self, config_leading_hours: List[float], config_entry_neglect_window: float,
               config_minimum_acquisitions: int, config_recovery_window: float) -> LeadingPredictionAccuracyConfig:
        return LeadingPredictionAccuracyConfig(aki_binary_index=0, scheme='',
                                        observable_code='',
                                        leading_hours=config_leading_hours,
                                        entry_neglect_window=config_entry_neglect_window,
                                        minimum_acquisitions=config_minimum_acquisitions,
                                        recovery_window=config_recovery_window)


    @pytest.fixture
    def metric(self, config: LeadingPredictionAccuracyConfig) -> LeadingAKIPredictionAccuracy:
        return LeadingAKIPredictionAccuracy(config=config)

    @pytest.fixture
    def metric_out(self, metric: VisitsAUC, identical_predictions: AdmissionsPrediction) -> Dict[str, float]:
        return metric(identical_predictions)

    def test_leading_auc(self, metric: VisitsAUC, metric_out: Dict[str, float]):
        assert isinstance(metric_out, dict)
        estimand_set = set(metric.estimands())
        assert set(metric_out.keys()).issubset(estimand_set)
        assert all(isinstance(v, (int, float)) for v in metric_out.values())


