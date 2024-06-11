from typing import List

import jax.numpy as jnp
import pandas as pd
import pytest

from lib.base import Array
from lib.metric.metrics import VisitsAUC, ObsPredictionLossMetric, OutcomePredictionLossMetric, \
    LeadPredictionLossMetric, \
    LeadingPredictionAccuracyConfig, LeadingAKIPredictionAccuracy, MetricsOutput, AdmissionAUC, MetricsCollection, \
    MetricsCollectionOutput, PerColumnObsPredictionLoss
from lib.ml.artefacts import AdmissionsPrediction


@pytest.mark.parametrize('standard_metric',
                         [ObsPredictionLossMetric(), OutcomePredictionLossMetric(), LeadPredictionLossMetric(),
                          PerColumnObsPredictionLoss(),
                          VisitsAUC(),
                          AdmissionAUC()])
def test_metric_out(standard_metric: ObsPredictionLossMetric, identical_predictions: AdmissionsPrediction):
    metric_out = standard_metric(identical_predictions)
    assert isinstance(metric_out.values, tuple)
    assert isinstance(metric_out.estimands, tuple)
    assert metric_out.name == type(standard_metric).__name__
    assert metric_out.estimands == standard_metric.estimands
    assert len(metric_out.values) == len(standard_metric.estimands)
    assert isinstance(metric_out.as_df(), pd.DataFrame)
    assert all(isinstance(v, (Array, float)) for v in metric_out.values)


@pytest.fixture
def metrics_collection():
    return MetricsCollection(
        metrics=[ObsPredictionLossMetric(), OutcomePredictionLossMetric(), LeadPredictionLossMetric(),
                 VisitsAUC(), AdmissionAUC(), PerColumnObsPredictionLoss()])


def test_metrics_collection_output(metrics_collection: MetricsCollection, identical_predictions: AdmissionsPrediction):
    metrics_out = metrics_collection(identical_predictions)
    assert isinstance(metrics_out, MetricsCollectionOutput)
    assert all(isinstance(m, MetricsOutput) for m in metrics_out.metrics)
    assert isinstance(metrics_out.as_df(), pd.DataFrame)
    assert len(metrics_out.as_df()) == 1


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
    def metric(self, config: LeadingPredictionAccuracyConfig,
               identical_predictions: AdmissionsPrediction) -> LeadingAKIPredictionAccuracy:
        return LeadingAKIPredictionAccuracy(config=config)

    @pytest.fixture
    def metric_out(self, metric: VisitsAUC, identical_predictions: AdmissionsPrediction) -> MetricsOutput:
        return metric(identical_predictions)

    def test_leading_auc(self, metric: LeadingAKIPredictionAccuracy, metric_out: MetricsOutput):
        assert isinstance(metric_out.values, tuple)
        assert isinstance(metric_out.estimands, tuple)
        assert metric_out.name == type(metric).__name__
        assert metric_out.estimands == metric.estimands
        assert len(metric_out.values) == len(metric.estimands)
        assert isinstance(metric_out.as_df(), pd.DataFrame)
        assert all(isinstance(v, (jnp.ndarray, float, int)) for v in metric_out.values)
