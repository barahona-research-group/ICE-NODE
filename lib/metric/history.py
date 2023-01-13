import pandas as pd


class MetricsHistory:

    def __init__(self):
        self._df = None

    def to_df(self):
        return self._df

    def __len__(self):
        return len(self._df) if self._df is not None else 0

    def last_evals(self):
        return self._df.iloc[-1, :].to_dict()

    def append_iteration(self, iter_dict):
        niters = 1 if self._df is None else len(self._df) + 1
        iter_df = pd.DataFrame(iter_dict, index=[niters], dtype=float)
        self._df = pd.concat([self._df, iter_df])
