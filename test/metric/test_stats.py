from typing import Dict

import pytest
from lib.metric.stat import VisitsAUC, ObsPredictionLossMetric, OutcomePredictionLossMetric, LeadPredictionLossMetric, \
    PredictionLoss, LossMetric
from lib.ml.artefacts import AdmissionsPrediction
import jax.numpy as jnp


class TestObsPredictionLossMetric:
    @pytest.fixture(params=[ObsPredictionLossMetric, OutcomePredictionLossMetric, LeadPredictionLossMetric])
    def metric(self, request) -> LossMetric:
        return request.param()

    @pytest.fixture
    def metric_out(self, metric: LossMetric, identical_predictions: AdmissionsPrediction) -> Dict[str, float]:
        return metric(identical_predictions)

    def test_obs_perdiction_loss_metric(self, metric: ObsPredictionLossMetric, metric_out: Dict[str, float]):
        assert isinstance(metric_out, dict)
        assert set(metric_out.keys()) == set(metric.config.loss_keys)
        assert all(isinstance(v, jnp.ndarray) for v in metric_out.values())

