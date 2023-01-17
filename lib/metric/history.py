from typing import Dict, Optional

import pandas as pd
from .risk import BatchPredictedRisks
from .stat import MetricsCollection


class MetricsHistory:
    metrics: MetricsCollection

    def __init__(self):
        self._df = None

    def to_df(self):
        return self._df

    def __len__(self):
        return len(self._df) if self._df is not None else 0

    def last_evals(self):
        return self._df.iloc[-1, :].to_dict()

    def append_iteration(
            self,
            predictions: BatchPredictedRisks,
            other_estimated_metrics: Optional[Dict[str, float]] = None):
        niters = 1 if self._df is None else len(self._df) + 1
        row_df = self.metrics.to_df(niters, predictions,
                                    other_estimated_metrics)
        self._df = pd.concat([self._df, row_df])

    def last_value(self, key):
        return self._df.iloc[-1, key]
