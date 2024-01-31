from abc import abstractmethod
from typing import Callable, List, Dict

import equinox as eqx
import numpy as np
import pandas as pd

from lib.ehr import Dataset


class CodedValueProcessor(eqx.Module):
    code_column: Callable[[Dataset], str]
    value_column: Callable[[Dataset], str]
    table: Callable[[Dataset], pd.DataFrame]

    def fit(self, dataset: Dataset, admission_ids: List[str]) -> 'CodedValueProcessor':
        df = self.table(dataset)
        c_value = self.value_column(dataset)
        c_code = self.code_column(dataset)
        c_adm_id = dataset.config.tables.obs.admission_id_alias
        df = df[[c_code, c_value, c_adm_id]]
        df = df[df[c_adm_id].isin(admission_ids)]

        fitted = self
        for k, v in self._extract_stats(df, c_code, c_value).items():
            fitted = eqx.tree_at(lambda x: getattr(x, k), fitted, v)
        return fitted

    @abstractmethod
    def _extract_stats(self, df: pd.DataFrame, c_code: str, c_value: str) -> Dict[str, pd.Series]:
        pass

    @abstractmethod
    def __call__(self, dataset: Dataset) -> Dataset:
        pass


class CodedValueScaler(CodedValueProcessor):
    use_float16: bool = True

    @abstractmethod
    @property
    def original_dtype(self) -> np.dtype:
        pass

    @abstractmethod
    def unscale(self, array: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def unscale_code(self, array: np.ndarray, code_index: int) -> np.ndarray:
        pass


class ZScoreScaler(CodedValueScaler):
    mean: pd.Series
    std: pd.Series

    @property
    def original_dtype(self) -> np.dtype:
        return self.mean.dtype

    def __call__(self, dataset: Dataset) -> Dataset:
        table = self.table(dataset)
        c_value = self.value_column(dataset)
        c_code = self.code_column(dataset)

        mean = table[c_code].map(self.mean)
        std = table[c_code].map(self.std)
        table.loc[:, c_value] = (table[c_value] - mean) / std
        if self.use_float16:
            table = table.astype({c_value: np.float16})

        return eqx.tree_at(lambda x: self.table(dataset), dataset, table)

    def unscale(self, array: np.ndarray) -> np.ndarray:
        array = array.astype(self.original_dtype)
        index = np.arange(array.shape[-1])
        return array * self.std.loc[index].values + self.mean.loc[index].values

    def unscale_code(self, array: np.ndarray, code_index: int) -> np.ndarray:
        array = array.astype(self.original_dtype)
        return array * self.std.loc[code_index] + self.mean.loc[code_index]

    def _extract_stats(self, df: pd.DataFrame, c_code: str, c_value: str) -> Dict[str, pd.Series]:
        stat = df.groupby(c_code).apply(
            lambda x: pd.Series({
                'mu': x[c_value].mean(),
                'sigma': x[c_value].std()
            }))
        return dict(mean=stat['mu'], std=stat['sigma'])


class MaxScaler(CodedValueScaler):
    max_val: pd.Series

    @property
    def original_dtype(self) -> np.dtype:
        return self.max_val.dtype

    def __call__(self, dataset: Dataset) -> Dataset:
        df = self.table(dataset)
        c_value = self.value_column(dataset)
        c_code = self.code_column(dataset)

        max_val = df[c_code].map(self.max_val)
        df.loc[:, c_value] = (df[c_value] / max_val)
        if self.use_float16:
            df = df.astype({c_value: np.float16})
        return eqx.tree_at(self.table, dataset, df)

    def unscale(self, array: np.ndarray) -> np.ndarray:
        array = array.astype(self.original_dtype)
        if array.shape[-1] == len(self.max_val):
            index = np.arange(array.shape[-1])
            return array * self.max_val.loc[index].values
        index = self.max_val.index.values
        array = array.copy()
        if array.ndim == 1:
            array[index] *= self.max_val.values
        else:
            array[:, index] *= self.max_val.values
        return array

    def unscale_code(self, array: np.ndarray, code_index: int) -> np.ndarray:
        array = array.astype(self.original_dtype)
        return array * self.max_val.loc[code_index]

    def _extract_stats(self, df: pd.DataFrame, c_code: str, c_value: str) -> Dict[str, pd.Series]:
        stat = df.groupby(c_code).apply(
            lambda x: pd.Series({
                'min': x[c_value].min(),
                'max': x[c_value].max()
            }))
        return dict(min_val=stat['min'], max_val=stat['max'])


class AdaptiveScaler(CodedValueScaler):
    max_val: pd.Series
    min_val: pd.Series
    mean: pd.Series
    std: pd.Series

    @property
    def original_dtype(self) -> np.dtype:
        return self.max_val.dtype

    def __call__(self, dataset: Dataset) -> Dataset:
        df = self.table(dataset)
        c_value = self.value_column(dataset)
        c_code = self.code_column(dataset)

        min_val = df[c_code].map(self.min_val)
        max_val = df[c_code].map(self.max_val)
        mean = df[c_code].map(self.mean)
        std = df[c_code].map(self.std)

        minmax_scaled = (df[c_value] - min_val) / max_val
        z_scaled = ((df[c_value] - mean) / std)

        df.loc[:, c_value] = np.where(min_val >= 0.0, minmax_scaled, z_scaled)
        if self.use_float16:
            df = df.astype({c_value: np.float16})
        return eqx.tree_at(self.table, dataset, df)

    def unscale(self, array: np.ndarray) -> np.ndarray:
        array = array.astype(self.original_dtype)
        index = np.arange(array.shape[-1])
        mu = self.mean.loc[index].values
        sigma = self.std.loc[index].values
        min_val = self.min_val.loc[index].values
        max_val = self.max_val.loc[index].values
        z_unscaled = array * sigma + mu
        minmax_unscaled = array * max_val + min_val
        return np.where(min_val >= 0.0, minmax_unscaled, z_unscaled)

    def unscale_code(self, array: np.ndarray, code_index: str) -> np.ndarray:
        array = array.astype(self.original_dtype)
        mu = self.mean.loc[code_index]
        sigma = self.std.loc[code_index]
        min_val = self.min_val.loc[code_index]
        max_val = self.max_val.loc[code_index]
        z_unscaled = array * sigma + mu
        minmax_unscaled = array * max_val + min_val
        return np.where(min_val >= 0.0, minmax_unscaled, z_unscaled)

    def _extract_stats(self, df: pd.DataFrame, c_code: str, c_value: str) -> Dict[str, pd.Series]:
        stat = df.groupby(c_code).apply(
            lambda x: pd.Series({
                'mu': x[c_value].mean(),
                'sigma': x[c_value].std(),
                'min': x[c_value].min(),
                'max': x[c_value].max()
            }))
        return dict(mean=stat['mu'],
                    std=stat['sigma'],
                    min_val=stat['min'],
                    max_val=stat['max'])


class IQROutlierRemover(CodedValueProcessor):
    outlier_q1: float
    outlier_q2: float
    outlier_iqr_scale: float
    outlier_z1: float
    outlier_z2: float
    min_val: pd.Series
    max_val: pd.Series

    def __call__(self, dataset: Dataset) -> Dataset:
        table = self.table(dataset)
        c_value = self.value_column(dataset)
        c_code = self.code_column(dataset)

        min_val = table[c_code].map(self.min_val)
        max_val = table[c_code].map(self.max_val)
        table = table[table[c_value].between(min_val, max_val)]

        return eqx.tree_at(self.table, dataset, table)

    def _extract_stats(self, df: pd.DataFrame, c_code: str, c_value: str) -> Dict[str, pd.Series]:
        outlier_q = np.array([self.outlier_q1, self.outlier_q2])
        q = df.groupby(c_code).apply(lambda x: x[c_value].quantile(outlier_q))

        q.columns = ['q1', 'q2']
        q['iqr'] = q['q2'] - q['q1']
        q['out_q1'] = q['q1'] - self.outlier_iqr_scale * q['iqr']
        q['out_q2'] = q['q2'] + self.outlier_iqr_scale * q['iqr']

        stat = df.groupby(c_code).apply(
            lambda x: pd.Series({
                'mu': x[c_value].mean(),
                'sigma': x[c_value].std()
            }))

        stat['out_z1'] = stat['mu'] - self.outlier_z1 * stat['sigma']
        stat['out_z2'] = stat['mu'] + self.outlier_z2 * stat['sigma']
        return dict(min_val=np.minimum(q['out_q1'], stat['out_z1']),
                    max_val=np.maximum(q['out_q2'], stat['out_z2']))
