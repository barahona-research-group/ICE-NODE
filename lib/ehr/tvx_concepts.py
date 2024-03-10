"""Data Model for Subjects in MIMIC-III and MIMIC-IV"""

from __future__ import annotations

import functools
import statistics
from datetime import date
from functools import cached_property
from typing import (List, Tuple, Optional, Dict, Union, Callable, Type, Any, ClassVar)

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pandas as pd

from .coding_scheme import (CodesVector, CodingScheme, OutcomeExtractor, NumericalTypeHint, NumericScheme)
from ..base import Config, Data, Module

Array = Union[npt.NDArray[Union[np.float64, np.float32, bool, int]], jax.Array]


class InpatientObservables(Data):
    """
    Vectorized representation of inpatient observables.

    Attributes:
        time (Array): array of time values.
        value (Array): array of observable values.
        mask (Array): array indicating missing values.

    Methods:
        empty(size: int): creates an empty instance of InpatientObservables.
        __len__(): returns the length of the time array.
        as_dataframe(scheme: AbstractScheme, filter_missing_columns=False): converts the observables to a pandas DataFrame.
        groupby_code(index2code: Dict[int, str]): groups the observables by code.
        segment(t_sep: Array): splits the observables into segments based on time values.
        concat(observables: Union[InpatientObservables, List[InpatientObservables]]): concatenates multiple instances of InpatientObservables.
        time_binning(hours: float): bins the time-series into time-windows and averages the values in each window.
    """

    time: Array  # (time,)
    value: Array  # (time, size)
    mask: Array  # (time, size)

    def __post_init__(self):
        assert self.time.dtype == np.float64, f"Expected time to be of type float64, got {self.time.dtype}"
        assert self.mask.dtype == bool, f"Expected mask to be of type bool, got {self.mask.dtype}"

        assert self.value.ndim == 2, f"Expected value to be 2D, got {self.value.ndim}"
        assert self.mask.ndim == 2, f"Expected mask to be 2D, got {self.mask.ndim}"
        assert self.value.shape == self.mask.shape, f"Expected value.shape to be {self.mask.shape}, " \
                                                    f"got {self.value.shape}"
        assert self.time.ndim == 1, f"Expected time to be 1D, got {self.time.ndim}"

    @staticmethod
    def empty(size: int,
              time_dtype: Type | str = np.float64,
              value_dtype: Type | str = np.float16,
              mask_dtype: Type | str = bool) -> InpatientObservables:
        """
        Create an empty InpatientObservables object.

        Parameters:
        - size (int): the size of the InpatientObservables object.

        Returns:
        - InpatientObservables: an empty InpatientObservables object with zero time, value, and mask arrays.
        """
        return InpatientObservables(time=np.zeros(shape=0, dtype=time_dtype),
                                    value=np.zeros(shape=(0, size), dtype=value_dtype),
                                    mask=np.zeros(shape=(0, size), dtype=mask_dtype))

    def __len__(self):
        """
        Returns the length of the 'time' attribute.
        """
        return self.time.shape[0]

    def count(self) -> int:
        """
        Returns the number of non-missing values in the 'value' attribute.
        """
        return np.sum(self.mask)

    def equals(self, other: InpatientObservables) -> bool:
        """
        Compares two InpatientObservables objects for equality.

        Args:
            other (InpatientObservables): the other InpatientObservables object to compare.

        Returns:
            bool: whether the two InpatientObservables objects are equal.
        """
        return (np.array_equal(self.time, other.time, equal_nan=True) and
                np.array_equal(self.value, other.value, equal_nan=True) and
                np.array_equal(self.mask, other.mask, equal_nan=True) and
                self.time.dtype == other.time.dtype and
                self.value.dtype == other.value.dtype and
                self.mask.dtype == other.mask.dtype)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the observables to a pandas DataFrame.

        Returns:
            pd.DataFrame: a pandas DataFrame containing the time and observable values.
        """
        time = {'time': self.time}
        value = {f'val_{i}': self.value[:, i] for i in range(self.value.shape[1])}
        mask = {f'mask_{i}': self.mask[:, i] for i in range(self.mask.shape[1])}
        return pd.DataFrame({**time, **value, **mask})

    def to_hdf(self, path: str, key: str, meta_prefix: str = '') -> Dict[str, str]:
        df = self.to_dataframe()
        if len(df) == 0:
            return {f'{meta_prefix}_time_dtype': str(self.time.dtype),
                    f'{meta_prefix}_value_dtype': str(self.value.dtype),
                    f'{meta_prefix}_mask_dtype': str(self.mask.dtype),
                    f'{meta_prefix}_dim': self.value.shape[1]}
        df.to_hdf(path, key=key, format='table')
        return {}

    @staticmethod
    def empty_from_meta(meta: Dict[str, str], meta_prefix: str = '') -> Optional[InpatientObservables]:
        if not any(k.startswith(meta_prefix) for k in meta.keys()):
            return None
        return InpatientObservables.empty(size=int(meta[f'{meta_prefix}_dim']),
                                          time_dtype=meta[f'{meta_prefix}_time_dtype'],
                                          value_dtype=meta[f'{meta_prefix}_value_dtype'],
                                          mask_dtype=meta[f'{meta_prefix}_mask_dtype'])

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> InpatientObservables:
        """
        Converts a pandas DataFrame to an InpatientObservables object.

        Args:
            df (pd.DataFrame): the pandas DataFrame to convert.

        Returns:
            InpatientObservables: the resulting InpatientObservables object.
        """
        time = df['time'].values
        p = (len(df.columns) - 1) // 2
        value = df[[f'val_{i}' for i in range(p)]].values
        mask = df[[f'mask_{i}' for i in range(p)]].values
        return InpatientObservables(time, value, mask)

    def as_dataframe(self,
                     scheme: CodingScheme,
                     filter_missing_columns: bool = False) -> pd.DataFrame:
        """
        Converts the observables to a pandas DataFrame.

        Args:
            scheme (AbstractScheme): the coding scheme used to convert the codes to descriptions.
            filter_missing_columns (bool): whether to filter out columns that contain only missing values.

        Returns:
            pd.DataFrame: a pandas DataFrame containing the time and observable values.
        """
        cols = ['time'] + [scheme.index2desc[i] for i in range(len(scheme))]
        value = jnp.where(self.mask, self.value, np.nan)
        tuples = np.hstack([self.time.reshape(-1, 1), value]).tolist()
        df = pd.DataFrame(tuples, columns=cols)
        if filter_missing_columns:
            return df.dropna(axis=1, how='all')
        else:
            return df

    def groupby_code(self, index2code: Dict[int, str]) -> Dict[str, InpatientObservables]:
        """
        Groups the data in the EHR object by the provided index-to-code mapping.

        Args:
            index2code (Dict[int, str]): a dictionary mapping column indices to codes.

        Returns:
            dict: a dictionary where the keys are the codes and the values are InpatientObservables objects.

        Raises:
            AssertionError: if the number of columns in the EHR object does not match the length of index2code.
        """

        assert len(index2code) == self.value.shape[1], \
            f'Expected {len(index2code)} columns, got {self.value.shape[1]}'
        if isinstance(self.time, jax.Array):
            _np = jnp
        else:
            _np = np

        dic = {}
        time = _np.array(self.time)
        value = _np.array(self.value)
        mask = _np.array(self.mask)

        for i, code in index2code.items():
            mask_i = mask[:, i]
            if not _np.any(mask_i):
                continue

            value_i = value[:, i][mask_i]
            time_i = time[mask_i]
            dic[code] = InpatientObservables(time=time_i,
                                             value=value_i,
                                             mask=_np.ones_like(value_i, dtype=bool))
        return dic

    def segment(self, t_sep: Array) -> List[InpatientObservables]:
        """
        Splits the InpatientObservables object into multiple segments based on the given time points.

        Args:
            t_sep (Array): array of time points used to split the InpatientObservables object.

        Returns:
            List[InpatientObservables]: list of segmented InpatientObservables objects.
        """
        if len(t_sep) == 0:
            return [self]

        split = np.searchsorted(self.time, t_sep)
        time = np.split(self.time, split)
        value = np.vsplit(self.value, split)
        mask = np.vsplit(self.mask, split)

        return [
            InpatientObservables(t, v, m)
            for t, v, m in zip(time, value, mask)
        ]

    @staticmethod
    def concat(observables: Union[InpatientObservables, List[InpatientObservables]]) -> InpatientObservables:
        """
        Concatenates a list of InpatientObservables into a single InpatientObservables object.

        Args:
            observables (Union[InpatientObservables, List[InpatientObservables]]): the list of InpatientObservables to concatenate.

        Returns:
            InpatientObservables: the concatenated InpatientObservables object.
        """
        if isinstance(observables, InpatientObservables):
            return observables

        if len(observables) == 0:
            return InpatientObservables.empty(0)
        if isinstance(observables[0].time, jax.Array):
            _np = jnp
        else:
            _np = np

        time = _np.hstack([o.time for o in observables])
        value = _np.vstack([o.value for o in observables])
        mask = _np.vstack([o.mask for o in observables])

        return InpatientObservables(time, value, mask)

    @staticmethod
    @functools.cache
    def type_hint_aggregator() -> Dict[NumericalTypeHint, Callable]:
        """
        Returns the type hint aggregator based on the type hints of the observables.

        Returns:
            Dict[NumericalTypeHint, Callable]: the type hint aggregator.
        """
        return {
            'B': lambda x: np.any(x) * 1.0,
            'O': lambda x: np.quantile(a=x, q=0.5, interpolation='higher'),
            # Most frequent value (mode) for categorical.
            'C': statistics.mode,
            'N': np.mean
        }

    @staticmethod
    def _time_binning_aggregate(x: Array, mask: npt.NDArray[bool],
                                type_hint: npt.NDArray[NumericalTypeHint]) -> Array:
        """
        Aggregates the values in a given array based on the type hint.

        Args:
            x (Array): The input array.
            type_hint (npt.NDArray[NumericalTypeHint]): The type hint.

        Returns:
            Array: The aggregated array.
        """
        type_hint_aggregator = InpatientObservables.type_hint_aggregator()
        assert x.ndim == 2 and mask.ndim == 2, f"Expected x, mask to be 2D, got ({x.ndim}, {mask.ndim})"
        assert x.shape == mask.shape, f"Expected x.shape to be {mask.shape}, got {x.shape}"
        assert x.shape[1] == len(type_hint), f"Expected x.shape[1] to be {len(type_hint)}, got {x.shape[1]}"
        return np.array([type_hint_aggregator[ti](xi[mi]) if mi.sum() > 0 else np.nan
                         for xi, mi, ti in zip(x.T, mask.T, type_hint)])

    def time_binning(self, hours: float, type_hint: npt.NDArray[NumericalTypeHint]) -> InpatientObservables:
        """
        Bin the time-series into time-windows of length `hours`.
        The values are aggregated in each window and assigned to the
        end of the window. Aggregation by default is `mean`, except for observables at binary_indices,
        where the aggregation is `any`.

        Args:
            hours (float): length of the time-windows in hours.
            type_hint (npt.NDArray[NumericalTypeHint]): type hints for the observables vector as
                (B) binary, (O) ordinal, (C) categorical, or (N) numerical.
        Returns:
            InpatientObservables: A new instance of InpatientObservables
                with the time-series binned according to the specified
                time-windows.
        """

        if len(self) == 0:
            return self

        last_ts = (int(self.time[-1] / hours) + 1) * hours
        new_time = np.arange(0, last_ts + hours, hours) * 1.0
        values = []
        masks = []
        for ti, tf in zip(new_time[:-1], new_time[1:]):
            time_mask = (ti <= self.time) & (self.time < tf)
            value = self._time_binning_aggregate(self.value[time_mask], self.mask[time_mask], type_hint)
            mask = np.where(np.isnan(value), False, True)
            values.append(value)
            masks.append(mask)

        values = np.vstack(values)
        masks = np.vstack(masks)
        values = np.where(masks, values, 0.0)
        return InpatientObservables(time=new_time[1:], value=values, mask=masks)


class LeadingObservableExtractorConfig(Config):
    """
    Config for LeadingObservableExtractor.

    Attributes:
        observable_code (str): the observable code to extract the leading window from.
        leading_hours (List[float]): list of leading hours to extract. Must be sorted.
        recovery_window (float): time window in hours to mask out between a nonzero value and zero.
        entry_neglect_window (float): hours to mask out in the beginning.
        minimum_acquisitions (int): minimum number of acquisitions before extracting the leading observable.
        scheme (str): name of the observation coding scheme to use.
    """

    observable_code: str
    scheme: str

    leading_hours: List[float]
    entry_neglect_window: float
    minimum_acquisitions: int  # minimum number of acquisitions to consider
    recovery_window: float = 0.0

    def __post_init__(self):
        # `leading_hours` must be sorted.
        assert all(
            x <= y for x, y in zip(self.leading_hours[:-1], self.leading_hours[1:])
        ), f"leading_hours must be sorted"
        self.leading_hours = list(self.leading_hours)

        assert self.type_hint in ('B', 'O'), (
            f"LeadingObservableExtractor only supports binary and ordinal observables, "
            f"got {self.type_hint}. Categorical and Numeric types "
            "would require custom aggregation function specific to the observation of interest,"
            "e.g. the mode of categorical or the mean of numerical. In other cases, it could be more "
            "relevant to use max/min aggregation over numeric observables. Create a feature request "
            "if you need this feature.")

    @cached_property
    def code_index(self) -> int:
        """
        Returns the index of the observable code.

        Returns:
            int: the index of the observable code.
        """
        return self.scheme_object.index[self.observable_code]

    @cached_property
    def scheme_object(self) -> NumericScheme:
        """
        Returns the scheme object based on the scheme name.

        Returns:
            NumericScheme: the scheme object.
        """
        scheme = CodingScheme.from_name(self.scheme)
        assert isinstance(scheme, NumericScheme), f"scheme must be numeric"
        return scheme

    @property
    def type_hint(self) -> NumericalTypeHint:
        """
        Returns the type hint for the observable.

        Returns:
            NumericalTypeHint: the type hint for the observable.
        """
        return self.scheme_object.type_hint[self.observable_code]

    @cached_property
    def aggregation(self) -> str:
        """
        Returns the aggregation function based on the aggregation scheme.

        Returns:
            Callable: the aggregation function.
        """
        if self.type_hint == 'B':
            return 'any'
        elif self.type_hint == 'O':
            return 'max'
        else:
            assert False, f"unsupported type hint {self.type_hint}"


class LeadingObservableExtractor(Module):
    """
    Extracts leading observables from a given timestamp based on a specified configuration.

    Attributes:
        config (LeadingObservableExtractorConfig): the configuration for the extractor.
        scheme (AbstractScheme): the scheme used for indexing.
        index2code (Dict[int, str]): a dictionary mapping index to code.
        code2index (Dict[str, int]): a dictionary mapping code to index.
    """
    config: LeadingObservableExtractorConfig

    def __len__(self):
        """
        Returns the number of leading hours.

        Returns:
            int: the number of leading hours.
        """
        return len(self.config.leading_hours)

    @cached_property
    def index2code(self):
        """
        Returns the mapping of index to code.

        Returns:
            Dict[int, str]: the mapping of index to code.
        """
        desc = self.config.scheme_object.desc[self.config.observable_code]
        return dict(
            zip(range(len(self.config.leading_hours)),
                [f'{desc}_next_{h}hrs' for h in self.config.leading_hours]))

    @cached_property
    def index2desc(self):
        """
        Returns the mapping of index to description.

        Returns:
            Dict[int, str]: the mapping of index to description.
        """
        return self.index2code

    @cached_property
    def code2index(self):
        """
        Returns the mapping of code to index.

        Returns:
            Dict[str, int]: the mapping of code to index.
        """
        return {v: k for k, v in self.index2code.items()}

    def empty(self):
        """
        Returns an empty InpatientObservables object.

        Returns:
            InpatientObservables: an empty InpatientObservables object.
        """
        return InpatientObservables.empty(len(self.config.leading_hours))

    @staticmethod
    def _nan_concat_leading_windows(x: Array) -> Array:
        """
        Generates sliding windows of the input array, padded with NaN values.

        Args:
            x (Array): the input array.

        Returns:
            Array: the sliding windows of the input array, padded with NaN values.
        """
        n = len(x)
        add_arr = np.full(n - 1, np.nan)
        x_ext = np.concatenate((add_arr, x[::-1]))
        strided = np.lib.stride_tricks.as_strided
        nrows = len(x_ext) - n + 1
        s = x_ext.strides[0]
        return strided(x_ext, shape=(nrows, n), strides=(s, s))[::-1, ::-1]

    @staticmethod
    def filter_first_acquisitions(m: int, n: int) -> Array:
        """
        Generates a mask ignoring the first acquisitions in the input array.

        Args:
            x (Array): the input array.
            n (int): number of acquisitions to neutralize.


        Returns:
            Array: the mask ignoring the first acquisitions in the input array.
        """
        return np.where(np.arange(m) < n, False, True).astype(bool)

    @staticmethod
    def filter_entry_neglect_window(t: Array, neglect_window: float) -> Array:
        """
        Generates a mask that ignores the observations in the beginning of the admission within `neglect_window`.
        The mask suppression is inclusive of the timestamps equal to `neglect_window`.

        Args:
            t (Array): the time array.
            neglect_window (float): number of hours to neglect in the beginning.

        Returns:
            Array: the neutralized input array.
        """
        return np.where(t <= neglect_window, False, True).astype(bool)

    @staticmethod
    def filter_recovery_window(t: Array, x: Array, recovery_window: float) -> Array:
        """
        Generates a mask the ignores the observations within the recovery window.

        Args:
            t (Array): the time array.
            x (Array): the input array.
            recovery_window (float): number of hours to neglect in the beginning.

        Returns:
            Array: the neutralized input array.
        """
        mask = np.ones_like(x).astype(bool)
        if len(t) == 0 or len(t) == 1:
            return mask
        x0 = x[0: -1]
        x1 = x[1:]
        next_recovery = (x0 != 0) & (~np.isnan(x0)) & (x1 == 0)

        for i in np.flatnonzero(next_recovery):
            mask[i + 1:] = np.where(t[i + 1:] - t[i] <= recovery_window, 0, 1)
        return mask

    @staticmethod
    def _nan_agg_nonzero(x, axis):
        """
        Aggregates the values in a given array along the specified axis, treating NaN values as zero.

        Args:
            x (Array): The input array.
            axis: The axis along which to aggregate.

        Returns:
            Array: The aggregated array.
        """
        all_nan = np.all(np.isnan(x), axis=axis) * 1.0
        replaced_nan = np.where(np.isnan(x), 0, x)
        return np.where(all_nan, np.nan, np.any(replaced_nan, axis=axis) * 1.0)

    @staticmethod
    def _nan_agg_max(x, axis):
        """
        Aggregates the values in a given array along the specified axis, treating NaN values as zero.

        Args:
            x (Array): The input array.
            axis: The axis along which to aggregate.

        Returns:
            Array: The aggregated array.
        """
        return np.nanmax(x, axis=axis)

    @classmethod
    def aggregation(cls, aggregation: str) -> Callable:
        """
        Returns the aggregation function based on the aggregation scheme.

        Returns:
            Callable: the aggregation function.
        """
        if aggregation == 'max':
            return cls._nan_agg_max
        elif aggregation == 'any':
            return cls._nan_agg_nonzero
        else:
            raise ValueError(f"Aggregation {aggregation} not supported")

    @classmethod
    def mask_noisy_observations(cls, t: Array, x: Array, entry_neglect_window: float, recovery_window: float,
                                minimum_acquisitions: int) -> Array:
        # neutralize the first acquisitions
        m = cls.filter_first_acquisitions(len(t), minimum_acquisitions)
        # neutralize the observations in the beginning within the entry neglect window.
        m &= cls.filter_entry_neglect_window(t, entry_neglect_window)
        # neutralize the observations within the recovery window.
        m &= cls.filter_recovery_window(t, x, recovery_window)
        return m

    @classmethod
    def extract_leading_window(cls, t: Array, x: Array, leading_hours: List[float],
                               aggregation: str) -> Array:
        aggregation = cls.aggregation(aggregation)
        # a time-window starting from timestamp_i for each row_i
        # if time = [t0, t1, t2, t3]
        # then t_leads = [[t0, t1, t2, t3],
        #                 [t1, t2, t3, nan],
        #                 [t2, t3, nan, nan],
        #                 [t3, nan, nan, nan]]
        t_leads = cls._nan_concat_leading_windows(t)

        # if time = [t0, t1, t2, t3]
        # then delta_leads = [[0, t1-t0, t2-t0, t3-t0],
        #                     [0, t2-t1, t3-t1, nan],
        #                     [0, t3-t2, nan, nan],
        #                     [0, nan, nan, nan]]
        delta_leads = t_leads - t_leads[:, 0, None]

        # a value-window starting from timestamp_i for each row_i
        # if value = [v0, v1, v2, v3]
        # then v_leads = [[v0, v1, v2, v3],
        #                 [v1, v2, v3, nan],
        #                 [v2, v3, nan, nan],
        #                 [v3, nan, nan, nan]]
        v_leads = cls._nan_concat_leading_windows(x)

        values = []
        for w in leading_hours:
            # select the rows where the time-window is less than t
            # if time = [t0, t1, t2, t3]
            # then mask = [[0 < t, t1 - t0 < t, t2 - t0 < t, t3 - t0 < t],
            #              [0 < t, t2 - t1 < t, t3 - t1 < t, nan < t],
            #              [0 < t, t3 - t2 < t, nan < t, nan < t]],
            #              [0 < t, nan < t, nan < t, nan < t]]
            mask = delta_leads < w
            v_lead = np.where(mask, v_leads, np.nan)
            values.append(aggregation(v_lead, axis=1).flatten())

        return np.stack(values, axis=1)

    def __call__(self, observables: InpatientObservables):
        """
        Makes a leading observable from the current timestamp.

        Args:
            observables (InpatientObservables): The input observables.

        Returns:
            InpatientObservables: The resulting leading observables.
        """
        if len(observables) == 0:
            return self.empty()

        time = observables.time
        value = observables.value[:, self.config.code_index]
        mask = observables.mask[:, self.config.code_index]
        mask &= self.mask_noisy_observations(time, value,
                                             entry_neglect_window=self.config.entry_neglect_window,
                                             recovery_window=self.config.recovery_window,
                                             minimum_acquisitions=self.config.minimum_acquisitions)
        value = np.where(mask, value, np.nan)
        value = self.extract_leading_window(time, value, self.config.leading_hours, self.config.aggregation)
        return InpatientObservables(time, value, mask=~np.isnan(value))


class InpatientInput(Data):
    """
    Represents inpatient input data.

    Attributes:
        code_index (Array): the code index array. Each index value correspond to a unique input code.
        starttime (Array): the start time array.
        endtime (Array): the end time array.
        rate (Optional[Array]): the rate array, if available. If not provided, it is assumed to be a vector of ones.
    """

    code_index: Array
    starttime: Array
    endtime: Array
    rate: Optional[Array] = None

    def __post_init__(self):
        if self.rate is None:
            self.rate = np.ones(len(self.code_index), dtype=bool)

    def to_dataframe(self):
        """
        Converts the input data to a pandas DataFrame.

        Returns:
            pd.DataFrame: a pandas DataFrame containing the input data.
        """
        return pd.DataFrame({
            'code_index': self.code_index,
            'rate': self.rate,
            'starttime': self.starttime,
            'endtime': self.endtime
        })

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> "InpatientInput":
        """
        Converts a pandas DataFrame to an InpatientInput object.

        Args:
            df (pd.DataFrame): the pandas DataFrame to convert.

        Returns:
            InpatientInput: the resulting InpatientInput object.
        """
        return InpatientInput(df['code_index'].values, df['starttime'].values, df['endtime'].values, df['rate'].values)

    def equals(self, other: "InpatientInput") -> bool:
        """
        Compares two InpatientInput objects for equality.

        Args:
            other (InpatientInput): the other InpatientInput object to compare.

        Returns:
            bool: whether the two InpatientInput objects are equal.
        """
        return (np.array_equal(self.code_index, other.code_index) and
                np.array_equal(self.starttime, other.starttime) and
                np.array_equal(self.endtime, other.endtime) and
                np.array_equal(self.rate, other.rate) and
                self.code_index.dtype == other.code_index.dtype and
                self.starttime.dtype == other.starttime.dtype and
                self.endtime.dtype == other.endtime.dtype and
                self.rate.dtype == other.rate.dtype)

    def __call__(self, t: float, input_size: int) -> Array:
        """
        Returns the vectorized input at time t.

        Args:
            t (float): the time at which to return the input.
            input_size (int): the size of the input vector.

        Returns:
            Array: the input at time t.
        """

        mask = (self.starttime <= t) & (t < self.endtime)
        index = self.code_index[mask]
        rate = self.rate[mask]
        vec_input = np.zeros(input_size, dtype=rate.dtype)
        vec_input[index] += rate
        return vec_input

    @classmethod
    def empty(cls,
              code_index_dtype=int,
              starttime_dtype=float,
              endtime_dtype=float,
              rate_dtype=float) -> InpatientInput:
        """
        Returns an empty InpatientInput object.

        Args:
            size (int): the size of the input.

        Returns:
            InpatientInput: an empty InpatientInput object.
        """
        zvec = np.zeros(0, dtype=bool)
        ii = cls(code_index=zvec.astype(code_index_dtype),
                 starttime=zvec.astype(starttime_dtype),
                 endtime=zvec.astype(endtime_dtype),
                 rate=zvec.astype(rate_dtype))
        return ii

    def to_hdf(self, path: str, key: str, meta_prefix: str = '') -> Dict[str, str]:
        df = self.to_dataframe()
        if len(df) == 0:
            return {f'{meta_prefix}_code_index_dtype': str(self.code_index.dtype),
                    f'{meta_prefix}_rate_dtype': str(self.rate.dtype),
                    f'{meta_prefix}_starttime_dtype': str(self.starttime.dtype),
                    f'{meta_prefix}_endtime_dtype': str(self.endtime.dtype)}
        df.to_hdf(path, key=key, format='table')
        return {}

    @staticmethod
    def empty_from_meta(meta: Dict[str, str], meta_prefix: str = '') -> Optional[InpatientInput]:
        if not any(k.startswith(meta_prefix) for k in meta.keys()):
            return None
        return InpatientInput.empty(code_index_dtype=meta[f'{meta_prefix}_code_index_dtype'],
                                    starttime_dtype=meta[f'{meta_prefix}_starttime_dtype'],
                                    endtime_dtype=meta[f'{meta_prefix}_endtime_dtype'],
                                    rate_dtype=meta[f'{meta_prefix}_rate_dtype'])


class InpatientInterventions(Data):
    # TODO: Add docstring.
    hosp_procedures: Optional[InpatientInput]
    icu_procedures: Optional[InpatientInput]
    icu_inputs: Optional[InpatientInput]

    def to_hdf(self, path: str, key: str) -> None:
        meta = {}
        for k in ('hosp_procedures', 'icu_procedures', 'icu_inputs'):
            ii = getattr(self, k)
            if ii is None:
                continue
            meta.update(ii.to_hdf(path, key=f'{key}/{k}', meta_prefix=f'{k}'))
        pd.DataFrame(meta, index=[0]).to_hdf(path, key=f'{key}/meta', format='table')

    @staticmethod
    def from_hdf(store: pd.HDFStore, key: str) -> InpatientInterventions:
        ii = {}
        meta = store[f'{key}/meta'].iloc[0].to_dict() if f'{key}/meta' in store else {}
        for k in ('hosp_procedures', 'icu_procedures', 'icu_inputs'):
            if f'{key}/{k}' in store:
                ii[k] = InpatientInput.from_dataframe(store[f'{key}/{k}'])
            elif any(kk.startswith(k) for kk in meta):
                ii[k] = InpatientInput.empty_from_meta(meta, meta_prefix=f'{k}')
            else:
                ii[k] = None
        return InpatientInterventions(**ii)

    @property
    def timestamps(self) -> List[float]:
        timestamps = []
        for k in self.__dict__.keys():
            ii = getattr(self, k, None)
            if isinstance(ii, InpatientInput):
                timestamps.extend(ii.starttime)
                timestamps.extend(ii.endtime)
        return list(sorted(set(timestamps)))

    def equals(self, other: "InpatientInterventions") -> bool:
        cond1 = (self.hosp_procedures is None and other.hosp_procedures is None) or (
            self.hosp_procedures.equals(other.hosp_procedures)
        )
        cond2 = (self.icu_procedures is None and other.icu_procedures is None) or (
            self.icu_procedures.equals(other.icu_procedures)
        )
        cond3 = (self.icu_inputs is None and other.icu_inputs is None) or (
            self.icu_inputs.equals(other.icu_inputs)
        )
        return cond1 and cond2 and cond3


class SegmentedInpatientInterventions(Data):
    time: Array
    hosp_procedures: Optional[Array]
    icu_procedures: Optional[Array]
    icu_inputs: Optional[Array]

    @classmethod
    def from_interventions(cls, inpatient_interventions: InpatientInterventions, terminal_time: float,
                           hosp_procedures_size: Optional[int] = None,
                           icu_procedures_size: Optional[int] = None,
                           icu_inputs_size: Optional[int] = None,
                           maximum_padding: int = 100) -> "SegmentedInpatientInterventions":

        timestamps = inpatient_interventions.timestamps
        assert terminal_time >= max(timestamps, default=0.0), (
            f"Terminal time {terminal_time} should be greater than the maximum timestamp {max(timestamps, default=0.0)}"
        )
        assert min(timestamps, default=0.0) >= 0.0, (
            f"Minimum timestamp {min(timestamps, default=0.0)} should be greater than or equal to 0.0"
        )

        time = np.unique(np.array(timestamps + [0.0, terminal_time], dtype=np.float64))
        time = cls.pad_array(time, value=np.nan, maximum_padding=maximum_padding)
        t0 = time[:-1]
        hosp_procedures = None
        icu_procedures = None
        icu_inputs = None
        if inpatient_interventions.hosp_procedures is not None and hosp_procedures_size is not None:
            hosp_procedures = cls._segment(t0, inpatient_interventions.hosp_procedures, hosp_procedures_size)
        if inpatient_interventions.icu_procedures is not None and icu_procedures_size is not None:
            icu_procedures = cls._segment(t0, inpatient_interventions.icu_procedures, icu_procedures_size)
        if inpatient_interventions.icu_inputs is not None and icu_inputs_size is not None:
            icu_inputs = cls._segment(t0, inpatient_interventions.icu_inputs, icu_inputs_size)
        return cls(time=time, hosp_procedures=hosp_procedures, icu_procedures=icu_procedures, icu_inputs=icu_inputs)

    @staticmethod
    def pad_array(array: Array,
                  maximum_padding: int = 100,
                  value: float = 0.0) -> Array:
        """
        Pad array to be a multiple of maximum_padding. This is to
        minimize the number of jit-compiling that is made for the same functions
        when the input shape changes.
        
        Args:
            array (Array): the array to be padded.
            maximum_padding (int, optional): the maximum padding. Defaults to 100.
            value (float, optional): the value to pad with. Defaults to 0.0.
            
        Returns:
            Array: the padded array."""

        n = len(array)
        n_pad = maximum_padding - (n % maximum_padding)
        if n_pad == maximum_padding:
            return array

        return np.pad(array, pad_width=(0, n_pad), mode='constant', constant_values=value)

    @staticmethod
    def _segment(t0_padded: Array, inpatient_input: InpatientInput, input_size: int) -> Array:
        """
        Generate segmented procedures and apply the specified procedure transformation.
        Returns:
            numpy.ndarray: the processed segments.
        """
        t = t0_padded[~np.isnan(t0_padded)]
        t_nan = t0_padded[np.isnan(t0_padded)]
        out = np.stack([inpatient_input(ti, input_size) for ti in t], axis=0)
        pad = np.zeros((len(t_nan), out[0].shape[0]), dtype=out.dtype)
        return np.vstack([out, pad])

    def to_dataframes(self) -> Dict[str, pd.DataFrame]:
        df1 = pd.DataFrame(self.hosp_procedures) if self.hosp_procedures is not None else None
        df2 = pd.DataFrame(self.icu_procedures) if self.icu_procedures is not None else None
        df3 = pd.DataFrame(self.icu_inputs) if self.icu_inputs is not None else None
        return {'hosp_procedures': df1, 'icu_procedures': df2, 'icu_inputs': df3,
                'time': pd.DataFrame(self.time)}

    def to_hdf(self, path: str, key: str) -> None:
        dataframes = self.to_dataframes()
        for k, v in dataframes.items():
            if v is not None:
                v.to_hdf(path, key=f'{key}/{k}', format='table')

    @staticmethod
    def from_hdf(store: pd.HDFStore, key: str) -> "SegmentedInpatientInterventions":
        segmented_interventions = {}
        for k in ('hosp_procedures', 'icu_procedures', 'icu_inputs', 'time'):
            if f'{key}/{k}' in store:
                segmented_interventions[k] = store[f'{key}/{k}']
        return SegmentedInpatientInterventions.from_dataframes(segmented_interventions)

    def equals(self, other: "SegmentedInpatientInterventions") -> bool:
        cond1 = (self.hosp_procedures is None and other.hosp_procedures is None) or (
            np.array_equal(self.hosp_procedures, other.hosp_procedures)
        )
        cond2 = (self.icu_procedures is None and other.icu_procedures is None) or (
            np.array_equal(self.icu_procedures, other.icu_procedures)
        )
        cond3 = (self.icu_inputs is None and other.icu_inputs is None) or (
            np.array_equal(self.icu_inputs, other.icu_inputs)
        )
        return cond1 and cond2 and cond3 and np.array_equal(self.time, other.time)

    @staticmethod
    def from_dataframes(dataframes: Dict[str, pd.DataFrame]) -> "SegmentedInpatientInterventions":
        df1 = dataframes.get('hosp_procedures')
        df2 = dataframes.get('icu_procedures')
        df3 = dataframes.get('icu_inputs')
        time = dataframes['time']
        hosp_procedures = df1.values if df1 is not None else None
        icu_procedures = df2.values if df2 is not None else None
        icu_inputs = df3.values if df3 is not None else None
        return SegmentedInpatientInterventions(time.values.flatten(), hosp_procedures, icu_procedures, icu_inputs)


class Admission(Data):
    """Admission data class representing a hospital admission.
    
    Attributes:
        admission_id: unique ID for the admission.
        admission_dates: start and end dates for the admission.
        dx_codes: diagnosis codes recorded during the admission.
        dx_codes_history: historical diagnosis codes prior to admission.
        outcome: outcome codes of interest derived from the diagnosis codes.
        observables: timeseries clinical observations data.
        interventions: timeseries clinical interventions data.  
        leading_observable: timeseries clinical leading observable (of interest). 
    """
    admission_id: str  # Unique ID for each admission
    admission_dates: Tuple[date, date]
    dx_codes: CodesVector
    dx_codes_history: CodesVector
    outcome: CodesVector
    observables: Optional[InpatientObservables]
    interventions: Optional[InpatientInterventions] = None
    leading_observable: Optional[InpatientObservables] = None

    def extract_leading_observable(self, leading_observable_extractor: LeadingObservableExtractor) -> Admission:
        """
        Extracts the leading observable from the admission data.

        Args:
            leading_observable_extractor (LeadingObservableExtractor): the leading observable extractor.

        Returns:
            Admission: the admission data with the leading observable extracted.
        """
        leading_observable = leading_observable_extractor(self.observables)
        return Admission(admission_id=self.admission_id,
                         admission_dates=self.admission_dates,
                         dx_codes=self.dx_codes,
                         dx_codes_history=self.dx_codes_history,
                         outcome=self.outcome,
                         observables=self.observables,
                         leading_observable=leading_observable,
                         interventions=self.interventions)

    def observables_time_binning(self, interval: float, obs_scheme: NumericScheme) -> Admission:
        """
        Bins the observables data into time intervals of the specified length.

        Args:
            interval (float): the length of the time intervals.
            obs_scheme (NumericScheme): the numeric scheme for the observables, which is used to guide to aggregation
                operations.
        Returns:
            Admission: the admission data with the observables data binned.

        """
        observables = self.observables.time_binning(interval, obs_scheme.type_array)
        return Admission(admission_id=self.admission_id,
                         admission_dates=self.admission_dates,
                         dx_codes=self.dx_codes,
                         dx_codes_history=self.dx_codes_history,
                         outcome=self.outcome,
                         observables=observables,
                         leading_observable=self.leading_observable,
                         interventions=self.interventions)

    @classmethod
    def _hdf_serialize_observables(cls, observables: Optional[InpatientObservables], path: str, key: str,
                                   meta_prefix: str = '') -> Dict[str, Any]:
        if observables is not None:
            return observables.to_hdf(path, key=key, meta_prefix=meta_prefix)
        return {}

    @classmethod
    def _hdf_deserialize_observables(cls, store: pd.HDFStore, key: str,
                                     meta: Dict[str, Any],
                                     meta_prefix: str = '') -> Optional[InpatientObservables]:
        if key in store:
            return InpatientObservables.from_dataframe(store[key])
        else:
            return InpatientObservables.empty_from_meta(meta=meta, meta_prefix=meta_prefix)

    @classmethod
    def _hdf_serialize_interventions(cls, interventions: Optional[InpatientInterventions], path: str, key: str) -> None:
        if interventions is not None:
            interventions.to_hdf(path, key=key)

    @classmethod
    def _hdf_deserialize_interventions(cls, store: pd.HDFStore, key: str) -> Optional[InpatientInterventions]:
        if key in store:
            return InpatientInterventions.from_hdf(store, key)
        return None

    def to_hdf(self, path: str, key: str) -> None:
        """
        Save the admission data to an HDF5 file.

        Args:
            path (str): the path to the HDF5 file.
            key (str): the key to use for the admission data.
        """
        meta = {'start': self.admission_dates[0],
                'end': self.admission_dates[1],
                'admission_id': self.admission_id,
                'dx_codes_scheme': self.dx_codes.scheme,
                'dx_codes_history_scheme': self.dx_codes_history.scheme,
                'outcome_scheme': self.outcome.scheme}
        pd.DataFrame(self.dx_codes.vec).to_hdf(path, key=f'{key}/dx_codes', format='table')
        pd.DataFrame(self.dx_codes_history.vec).to_hdf(path, key=f'{key}/dx_codes_history', format='table')
        pd.DataFrame(self.outcome.vec).to_hdf(path, key=f'{key}/outcome', format='table')
        self._hdf_serialize_interventions(self.interventions, path=path, key=f'{key}/interventions')
        meta.update(self._hdf_serialize_observables(self.observables, path=path, key=f'{key}/observables',
                                                    meta_prefix='obs'))
        meta.update(self._hdf_serialize_observables(self.leading_observable, path=path, key=f'{key}/leading_observable',
                                                    meta_prefix='lead_obs'))
        pd.DataFrame(meta, index=[0]).to_hdf(path, key=f'{key}/admission_meta', format='table')

    @staticmethod
    def from_hdf_store(hdf_store: pd.HDFStore, key: str) -> 'Admission':
        """
        Load the admission data from an HDF5 file.

        Args:
            hdf_store (pd.HDFStore): the HDF5 store.
            key (str): the key to use for the admission data.
        Returns:
            Admission: the admission data.
        """
        meta = hdf_store[f'{key}/admission_meta'].iloc[0].to_dict()

        dx_codes_scheme = CodingScheme.from_name(meta['dx_codes_scheme'])
        dx_codes_history_scheme = CodingScheme.from_name(meta['dx_codes_history_scheme'])
        outcome_scheme = OutcomeExtractor.from_name(meta['outcome_scheme'])

        dx_codes = dx_codes_scheme.wrap_vector(hdf_store[f'{key}/dx_codes'][0].values)
        dx_codes_history = dx_codes_history_scheme.wrap_vector(hdf_store[f'{key}/dx_codes_history'][0].values)
        outcome = outcome_scheme.wrap_vector(hdf_store[f'{key}/outcome'][0].values)

        observables = Admission._hdf_deserialize_observables(hdf_store, key=f'{key}/observables', meta=meta,
                                                             meta_prefix='obs')
        leading_observable = Admission._hdf_deserialize_observables(hdf_store,
                                                                    key=f'{key}/leading_observable',
                                                                    meta=meta, meta_prefix='lead_obs')
        interventions = Admission._hdf_deserialize_interventions(hdf_store,
                                                                 key=f'{key}/interventions')
        return Admission(admission_id=meta['admission_id'],
                         admission_dates=(meta['start'], meta['end']),
                         dx_codes=dx_codes,
                         dx_codes_history=dx_codes_history,
                         outcome=outcome,
                         observables=observables,
                         leading_observable=leading_observable,
                         interventions=interventions)

    @property
    def interval_hours(self) -> float:
        """
        Calculates the interval in hours between the admission start and end dates.

        Returns:
            float: the interval in hours.
        """
        return (self.admission_dates[1] - self.admission_dates[0]).total_seconds() / 3600

    @property
    def interval_days(self) -> float:
        """
        Calculates the interval in days based on the interval in hours.

        Returns:
            float: the interval in days.
        """
        return self.interval_hours / 24

    def days_since(self, date: date) -> Tuple[float, float]:
        """
        Calculates the number of days from a reference date to admission and discharge dates, respectively.

        Args:
            date (date): the date to calculate the number of days since.

        Returns:
            Tuple[float, float]: the number of days passed at the admission and discharge dates, respectively.
        """
        d1 = (self.admission_dates[0] - date).total_seconds() / 3600 / 24
        d2 = (self.admission_dates[1] - date).total_seconds() / 3600 / 24
        return d1, d2

    def equals(self, other: 'Admission') -> bool:
        """
        Compares two Admission objects for equality.

        Args:
            other (Admission): the other Admission object to compare.

        Returns:
            bool: whether the two Admission objects are equal.
        """
        for k in ('dx_codes', 'dx_codes_history', 'outcome', 'observables', 'interventions', 'leading_observable'):
            attr = getattr(self, k)
            other_attr = getattr(other, k)
            if attr is None and other_attr is None:
                continue
            elif attr is None or other_attr is None:
                return False
            elif not attr.equals(other_attr):
                return False
        return (self.admission_id == other.admission_id and
                self.admission_dates == other.admission_dates)


class SegmentedAdmission(Admission):
    observables: Optional[List[InpatientObservables]] = None
    interventions: Optional[SegmentedInpatientInterventions] = None
    leading_observable: Optional[List[InpatientObservables]] = None

    @staticmethod
    def _segment_interventions(interventions: Optional[InpatientInterventions],
                               terminal_time: float,
                               hosp_procedures_size: Optional[int], icu_procedures_size: Optional[int],
                               icu_inputs_size: Optional[int], maximum_padding: int = 100) -> Optional[
        SegmentedInpatientInterventions]:
        if interventions is None:
            return None
        return SegmentedInpatientInterventions.from_interventions(interventions, terminal_time,
                                                                  hosp_procedures_size=hosp_procedures_size,
                                                                  icu_procedures_size=icu_procedures_size,
                                                                  icu_inputs_size=icu_inputs_size,
                                                                  maximum_padding=maximum_padding)

    @staticmethod
    def _segment_observables(observables: Optional[InpatientObservables],
                             interventions: Optional[SegmentedInpatientInterventions]) -> Optional[
        List[InpatientObservables]]:
        if observables is None:
            return None
        if interventions is None:
            return [observables]
        time = interventions.time
        t_sep = time[~np.isnan(time)][1:-1]
        return observables.segment(t_sep)

    @staticmethod
    def from_admission(admission: Admission, hosp_procedures_size: Optional[int] = None,
                       icu_procedures_size: Optional[int] = None,
                       icu_inputs_size: Optional[int] = None,
                       maximum_padding: int = 100) -> 'SegmentedAdmission':
        interventions = SegmentedAdmission._segment_interventions(admission.interventions, admission.interval_hours,
                                                                  hosp_procedures_size=hosp_procedures_size,
                                                                  icu_procedures_size=icu_procedures_size,
                                                                  icu_inputs_size=icu_inputs_size,
                                                                  maximum_padding=maximum_padding)
        observables = SegmentedAdmission._segment_observables(admission.observables, interventions)
        leading_observable = SegmentedAdmission._segment_observables(admission.leading_observable, interventions)
        return SegmentedAdmission(admission_id=admission.admission_id,
                                  admission_dates=admission.admission_dates,
                                  dx_codes=admission.dx_codes,
                                  dx_codes_history=admission.dx_codes_history,
                                  outcome=admission.outcome,
                                  observables=observables,
                                  interventions=interventions,
                                  leading_observable=leading_observable)


class DemographicVectorConfig(Config):
    """
    Configuration class for demographic vector.

    Attributes:
        gender (bool): indicates whether gender is included in the vector.
        age (bool): indicates whether age is included in the vector.
        ethnicity (bool): indicates whether ethnicity is included in the vector.
    """
    gender: bool = True
    age: bool = True
    ethnicity: bool = True


class CPRDDemographicVectorConfig(DemographicVectorConfig):
    """
    Configuration class for CPRD demographic vector.
    """
    imd: bool = True


class StaticInfo(Data):
    """
    Represents static information about a patient.

    Attributes:
        demographic_vector_config (DemographicVectorConfig): Configuration for demographic vector.
        gender (Optional[CodesVector]): gender information of the patient.
        ethnicity (Optional[CodesVector]): ethnicity information of the patient.
        date_of_birth (Optional[date]): date of birth of the patient.
        constant_vec (Optional[Array]): constant vector representing the static information.

    Methods:
        __post_init__: initializes the constant vector based on the available attributes.
        age: calculates the age of the patient based on the current date.
        demographic_vector: returns the demographic vector based on the current date.
        _concat: concatenates the age and vector.

    """

    demographic_vector_config: DemographicVectorConfig
    gender: Optional[CodesVector] = None
    ethnicity: Optional[CodesVector] = None
    date_of_birth: Optional[date] = None

    @property
    def constant_attrs_list(self) -> List[Array]:
        attrs_vec = []
        if self.demographic_vector_config.gender:
            assert self.gender is not None and len(
                self.gender) > 0, "Gender is not extracted from the dataset"
            attrs_vec.append(self.gender.vec)
        if self.demographic_vector_config.ethnicity:
            assert self.ethnicity is not None, \
                "Ethnicity is not extracted from the dataset"
            attrs_vec.append(self.ethnicity.vec)
        return attrs_vec

    @cached_property
    def constant_vec(self):
        attrs_vec = self.constant_attrs_list
        if any(isinstance(a, jax.Array) for a in attrs_vec):
            _np = jnp
        else:
            _np = np

        if len(attrs_vec) == 0:
            return _np.array([], dtype=jnp.float16)
        else:
            return _np.hstack(attrs_vec)

    def age(self, current_date: date) -> float:
        """
        Calculates the age of the patient based on the current date.

        Args:
            current_date (date): the current date.

        Returns:
            float: the age of the patient in years.
        """
        return (current_date - self.date_of_birth).days / 365.25

    def demographic_vector(self, current_date: date) -> Array:
        """
        Returns the demographic vector based on the current date.

        Args:
            current_date (date): the current date.

        Returns:
            Array: the demographic vector.
        """
        if self.demographic_vector_config.age:
            return self._concat(self.age(current_date), self.constant_vec)
        return self.constant_vec

    @staticmethod
    def _concat(age, vec):
        """
        Concatenates the age and vector.

        Args:
            age (float): the age of the patient.
            vec (Array): the vector to be concatenated.

        Returns:
            Array: the concatenated vector.
        """
        return jnp.hstack((age, vec), dtype=jnp.float16)

    def to_dataframes(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
        """
        Converts the static information to a pandas DataFrame.

        Returns:
            pd.DataFrame: a pandas DataFrame containing the static information.
        """
        df = {}
        meta = {}
        if self.gender is not None:
            df['gender'] = pd.DataFrame(self.gender.vec)
            meta.update({'gender_scheme': self.gender.scheme})

        if self.ethnicity is not None:
            df['ethnicity'] = pd.DataFrame(self.ethnicity.vec)
            meta.update({'ethnicity_scheme': self.ethnicity.scheme})
        if self.date_of_birth is not None:
            meta['date_of_birth'] = self.date_of_birth
        return df, meta

    def to_hdf(self, path: str, key: str) -> None:
        """
        Save the static information to an HDF5 file.

        Args:
            path (str): the path to the HDF5 file.
            key (str): the key to use for the static information.
        """
        df, meta = self.to_dataframes()
        for k, v in df.items():
            v.to_hdf(path, key=f'{key}/{k}', format='table')
        pd.DataFrame(meta, index=[0]).to_hdf(path, key=f'{key}/meta', format='table')

    @staticmethod
    def _data_from_dataframes(dataframes: Dict[str, pd.DataFrame], meta: Dict[str, str]) -> Dict[str, Any]:
        data = {}
        if 'gender' in dataframes:
            scheme = CodingScheme.from_name(meta['gender_scheme'])
            data['gender'] = scheme.wrap_vector(vec=dataframes['gender'][0].values)
        if 'ethnicity' in dataframes:
            scheme = CodingScheme.from_name(meta['ethnicity_scheme'])
            data['ethnicity'] = scheme.wrap_vector(vec=dataframes['ethnicity'][0].values)
        if 'date_of_birth' in meta:
            data['date_of_birth'] = meta['date_of_birth']
        return data

    @staticmethod
    def from_dataframes(dataframes: Dict[str, pd.DataFrame], meta: Dict[str, str],
                        demographic_vector_config: DemographicVectorConfig) -> 'StaticInfo':
        """
        Converts a pandas DataFrame to a StaticInfo object.
        """
        return StaticInfo(demographic_vector_config=demographic_vector_config,
                          **StaticInfo._data_from_dataframes(dataframes, meta))

    @staticmethod
    def from_hdf(store: pd.HDFStore, key: str, demographic_vector_config: DemographicVectorConfig) -> 'StaticInfo':
        """
        Load the static information from an HDF5 file.

        Args:
            store (pd.HDFStore): the HDF5 store.
            key (str): the key to use for the static information.
            demographic_vector_config (DemographicVectorConfig): the demographic vector configuration.

        Returns:
            StaticInfo: the static information.
        """
        dataframes = {k.split('/')[-1]: store[k] for k in store.keys() if key in k and k != f'{key}/meta'}
        meta = store[f'{key}/meta'].iloc[0].to_dict()
        return StaticInfo.from_dataframes(dataframes, meta, demographic_vector_config)

    def equals(self, other: 'StaticInfo') -> bool:
        """
        Compares two StaticInfo objects for equality.

        Args:
            other (StaticInfo): the other StaticInfo object to compare.

        Returns:
            bool: whether the two StaticInfo objects are equal.
        """
        return self.demographic_vector_config == other.demographic_vector_config and \
            self.gender == other.gender and self.ethnicity == other.ethnicity and \
            self.date_of_birth == other.date_of_birth


class CPRDStaticInfo(StaticInfo):
    """
    Represents static information extracted from the CPRD dataset.

    Attributes:
        imd (Optional[CodesVector]): the IMD (Index of Multiple Deprivation) vector.
    """

    imd: Optional[CodesVector] = None
    demographic_vector_config: CPRDDemographicVectorConfig

    @property
    def constant_attrs_list(self) -> List[Array]:
        attrs_vec = super().constant_attrs_list
        if self.demographic_vector_config.imd:
            assert self.imd is not None, \
                "IMD is not extracted from the dataset"
            attrs_vec.append(self.imd.vec)
        return attrs_vec

    def to_dataframes(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
        """
        Converts the static information to a pandas DataFrame.

        Returns:
            pd.DataFrame: a pandas DataFrame containing the static information.
        """
        df, meta = super().to_dataframes()
        if self.imd is not None:
            df['imd'] = pd.DataFrame(self.imd.vec)
            meta.update({'imd': self.imd.scheme})
        return df, meta

    @staticmethod
    def _data_from_dataframes(dataframes: Dict[str, pd.DataFrame], meta: Dict[str, str]) -> Dict[str, Any]:
        data = super()._data_from_dataframes(dataframes, meta)
        if 'imd' in dataframes:
            scheme = CodingScheme.from_name(meta['imd_scheme'])
            data['imd'] = scheme.wrap_vector(vec=dataframes['imd'][0].values)
        return data

    def equals(self, other: 'CPRDStaticInfo') -> bool:
        """
        Compares two CPRDStaticInfo objects for equality.

        Args:
            other (CPRDStaticInfo): the other CPRDStaticInfo object to compare.

        Returns:
            bool: whether the two CPRDStaticInfo objects are equal.
        """
        return super().equals(other) and self.imd == other.imd


class Patient(Data):
    """
    Represents a patient with demographic information and a clinical history 
    of admissions.
    
    Attributes:
        subject_id: Unique identifier for the patient.
        static_info: Static demographic and geographic data for the patient.
        admissions: List of hospital/clinic admissions for the patient.
    
    
    Properties:
        d2d_interval_days: number of days between first and last discharge.  
        
        
    Methods:
        outcome_frequency_vec: accumulates outcome vectors from all admissions to get 
        aggregate vector for patient.
    """
    subject_id: int
    static_info: StaticInfo
    admissions: List[Admission]

    admission_cls: ClassVar[Type[Admission]] = Admission

    def __post_init__(self):
        self.admissions = list(sorted(self.admissions, key=lambda x: x.admission_dates[0]))

    @property
    def d2d_interval_days(self):
        """
        The interval in days between the first and last discharge dates.

        Returns:
            float: the interval in days.
        """
        d1 = self.admissions[0].admission_dates[1]
        d2 = self.admissions[-1].admission_dates[1]
        return (d2 - d1).total_seconds() / 3600 / 24

    def outcome_frequency_vec(self):
        """
        Calculate the accumulation of outcome vectors for all admissions.

        Returns:
            The total count of outcome code occurrence over all admissions.
        """
        return sum(a.outcome.vec for a in self.admissions)

    def to_hdf(self, path: str, key: str) -> None:
        """
        Save the patient data to an HDF5 file.

        Args:
            path (str): the path to the HDF5 file.
            key (str): the key to use for the patient data.
        """
        meta = {'subject_id': self.subject_id}
        self.static_info.to_hdf(path, key=f'{key}/static_info')
        self._admissions_to_hdf(path, f'{key}/admissions')
        pd.DataFrame(meta, index=[0]).to_hdf(path, key=f'{key}/meta', format='table')

    def _admissions_to_hdf(self, path: str, key: str) -> None:
        for adm in self.admissions:
            adm.to_hdf(path, f'{key}/admissions/{adm.admission_id}')
        admission_ids = [adm.admission_id for adm in self.admissions]
        pd.DataFrame({'admission_id': admission_ids}).to_hdf(path, key=f'{key}/admission_ids', format='table')

    @classmethod
    def _admissions_from_hdf(cls, hdf_store: pd.HDFStore, key: str) -> List[Admission]:
        if f'{key}/admission_ids' in hdf_store.keys():
            admission_ids_df = hdf_store[f'{key}/admission_ids']
            if 'admission_id' in admission_ids_df:
                admission_ids = admission_ids_df['admission_id'].values
                return [cls.admission_cls.from_hdf_store(hdf_store, f'{key}/admissions/{admission_id}') for
                        admission_id in admission_ids]
        return []

    @classmethod
    def from_hdf_store(cls, hdf_store: pd.HDFStore, key: str,
                       demographic_vector_config: DemographicVectorConfig) -> 'Patient':
        """
        Load the patient data from an HDF5 file.

        Args:
            hdf_store (pd.HDFStore): the HDF5 store.
            key (str): the key to use for the patient data.
            demographic_vector_config (DemographicVectorConfig): the demographic vector configuration.

        Returns:
            Patient: the patient data.
        """
        static_info = StaticInfo.from_hdf(hdf_store,
                                          key=f'{key}/static_info',
                                          demographic_vector_config=demographic_vector_config)
        admissions = cls._admissions_from_hdf(hdf_store, f'{key}/admissions')
        subject_id = hdf_store[f'{key}/meta'].loc[0].to_dict()['subject_id']
        return cls(subject_id=subject_id, static_info=static_info, admissions=admissions)

    def equals(self, other: 'Patient') -> bool:
        """
        Compares two Patient objects for equality.

        Args:
            other (Patient): the other Patient object to compare.

        Returns:
            bool: whether the two Patient objects are equal.
        """
        return (self.subject_id == other.subject_id and
                self.static_info.equals(other.static_info) and
                all(a.equals(b) for a, b in zip(self.admissions, other.admissions)))

    def extract_leading_observables(self, leading_observable_extractor: LeadingObservableExtractor) -> Patient:
        """
        Extracts the leading observable from all admissions.

        Args:
            leading_observable_extractor (LeadingObservableExtractor): the leading observable extractor.

        Returns:
            Patient: the patient data with the leading observable extracted.
        """
        admissions = [a.extract_leading_observable(leading_observable_extractor) for a in self.admissions]
        return Patient(subject_id=self.subject_id, static_info=self.static_info, admissions=admissions)

    def observables_time_binning(self, interval: float,
                                 obs_scheme: NumericScheme) -> Patient:
        """
        Bins the observables for all admissions.

        Args:
            interval (float): the interval in hours to bin the observables.

        Returns:
            Patient: the patient data with the observables binned.
        """
        admissions = [a.observables_time_binning(interval, obs_scheme) for a in self.admissions]
        return Patient(subject_id=self.subject_id, static_info=self.static_info, admissions=admissions)


class SegmentedPatient(Patient):
    admissions: List[SegmentedAdmission]
    admission_cls: ClassVar[Type[SegmentedAdmission]] = SegmentedAdmission

    def extract_leading_observables(self, leading_observable_extractor: LeadingObservableExtractor) -> Patient:
        raise NotImplementedError("SegmentedPatient does not support leading observable extraction")
