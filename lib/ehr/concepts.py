"""Data Model for Subjects in MIMIC-III and MIMIC-IV"""

from __future__ import annotations

from datetime import date
from functools import cached_property
from typing import (List, Tuple, Optional, Dict, Union, Callable)

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import numpy.typing as npt
import pandas as pd

from .coding_scheme import (CodesVector, CodingScheme)
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

    time: jnp.narray  # (time,)
    value: Array  # (time, size)
    mask: Array  # (time, size)

    @staticmethod
    def empty(size: int):
        """
        Create an empty InpatientObservables object.

        Parameters:
        - size (int): the size of the InpatientObservables object.

        Returns:
        - InpatientObservables: an empty InpatientObservables object with zero time, value, and mask arrays.
        """
        return InpatientObservables(time=np.zeros(shape=0, dtype=np.float32),
                                    value=np.zeros(shape=(0, size),
                                                   dtype=np.float16),
                                    mask=np.zeros(shape=(0, size), dtype=bool))

    def __len__(self):
        """
        Returns the length of the 'time' attribute.
        """
        return len(self.time)

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
                                             mask=_np.ones_like(value_i,
                                                                dtype=bool))
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

    def time_binning(self, hours: float) -> InpatientObservables:
        """
        Bin the time-series into time-windows of length `hours`.
        The values are averaged in each window and assigned to the
        end of the window.

        Args:
            hours (float): length of the time-windows in hours.

        Returns:
            InpatientObservables: A new instance of InpatientObservables
                with the time-series binned according to the specified
                time-windows.
        """

        if len(self) == 0:
            return self

        obs_value = np.where(self.mask, self.value, np.nan)
        last_ts = (int(self.time[-1] / hours) + 1) * hours
        new_time = np.arange(0, last_ts + hours, hours) * 1.0
        values = []
        masks = []
        for ti, tf in zip(new_time[:-1], new_time[1:]):
            mask = (ti <= self.time) & (self.time < tf)
            value = obs_value[mask]
            value = np.nanmean(value, axis=0)
            mask = np.where(np.isnan(value), False, True)
            values.append(value)
            masks.append(mask)

        values = np.vstack(values)
        masks = np.vstack(masks)
        values = np.where(masks, values, 0.0)
        return InpatientObservables(time=new_time[1:],
                                    value=values,
                                    mask=masks)


class LeadingObservableExtractorConfig(Config):
    """
    Config for LeadingObservableExtractor.

    Attributes:
        code_index (int): index of the observable in the observable scheme.
        leading_hours (List[float]): list of leading hours to extract. Must be sorted.
        recovery_window (float): time window in hours to mask out between a nonzero value and zero.
        entry_neglect_window (float): hours to mask out in the beginning.
        minimum_acquisitions (int): minimum number of acquisitions before extracting the leading observable.
        scheme (str): name of the observation coding scheme to use.
    """

    code_index: int
    scheme: str

    leading_hours: List[float]
    entry_neglect_window: float
    minimum_acquisitions: int  # minimum number of acquisitions to consider
    recovery_window: float = 0.0
    aggregation: str = 'any'

    def __post_init__(self):
        # `leading_hours` must be sorted.
        assert all(
            x <= y for x, y in zip(self.leading_hours[:-1], self.leading_hours[1:])
        ), f"leading_hours must be sorted"
        self.leading_hours = list(self.leading_hours)


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
        scheme = CodingScheme.from_name(self.config.scheme)
        desc = scheme.index2desc[self.config.index]
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
    def neutralize_first_acquisitions(x: Array, n: int) -> Array:
        """
        Neutralizes the first acquisitions in the input array.

        Args:
            x (Array): the input array.
            n (int): number of acquisitions to neutralize.


        Returns:
            Array: the neutralized input array.
        """
        return np.where(np.arange(len(x)) < n, np.nan, x)

    @staticmethod
    def neutralize_entry_neglect_window(t: Array, x: Array, neglect_window: float) -> Array:
        """
        Neutralizes the observations in the beginning of the admission within `neglect_window`.
        Neutralization is inclusive of the timestamps equal to `neglect_window`.

        Args:
            t (Array): the time array.
            x (Array): the input array.
            neglect_window (float): number of hours to neglect in the beginning.

        Returns:
            Array: the neutralized input array.
        """
        return np.where(t <= neglect_window, np.nan, x)

    @staticmethod
    def neutralize_recovery_window(t: Array, x: Array, recovery_window: float) -> Array:
        """
        Neutralizes the observations within the recovery window.

        Args:
            t (Array): the time array.
            x (Array): the input array.
            recovery_window (float): number of hours to neglect in the beginning.

        Returns:
            Array: the neutralized input array.
        """
        x = x.copy()
        if len(t) == 0 or len(t) == 1:
            return x
        x0 = x[0: -1]
        x1 = x[1:]
        next_recovery = (x0 != 0) & (~np.isnan(x0)) & (x1 == 0)

        for i in np.flatnonzero(next_recovery):
            x[i + 1:] = np.where(t[i + 1:] - t[i] <= recovery_window, np.nan, x[i + 1:])
        return x

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
    def clean_values(cls, t: Array, x: Array, entry_neglect_window: float, recovery_window: float,
                     minimum_acquisitions: int) -> Array:
        # neutralize the first acquisitions
        x = cls.neutralize_first_acquisitions(x, minimum_acquisitions)
        # neutralize the observations in the beginning within the entry neglect window.
        x = cls.neutralize_entry_neglect_window(t, x, entry_neglect_window)
        # neutralize the observations within the recovery window.
        x = cls.neutralize_recovery_window(t, x, recovery_window)
        return x

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
        masks = []

        for t in leading_hours:
            # select the rows where the time-window is less than t
            # if time = [t0, t1, t2, t3]
            # then mask = [[0 < t, t1 - t0 < t, t2 - t0 < t, t3 - t0 < t],
            #              [0 < t, t2 - t1 < t, t3 - t1 < t, nan < t],
            #              [0 < t, t3 - t2 < t, nan < t, nan < t]],
            #              [0 < t, nan < t, nan < t, nan < t]]
            mask = delta_leads < t
            v_lead = np.where(mask, v_leads, np.nan)
            values.append(aggregation(v_lead, axis=1).flatten())

            masks.append(np.where(np.isnan(values[-1]), False, True))

        values = np.stack(values, axis=1)
        masks = np.stack(masks, axis=1)
        return InpatientObservables(time=t, value=values, mask=masks)

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
        mask = observables.mask[:, self.config.code_index]
        value = np.where(mask, observables.value[:, self.config.code_index], np.nan)
        value = self.clean_values(time, value, entry_neglect_window=self.config.entry_neglect_window,
                                  recovery_window=self.config.recovery_window,
                                  minimum_acquisitions=self.config.minimum_acquisitions)
        return self.extract_leading_window(time, value, self.config.leading_hours, self.config.aggregation)


class MaskedAggregator(eqx.Module):
    """
    A class representing a masked aggregator.

    Attributes:
        mask (jnp.array): The mask used for aggregation.
    """

    mask: jnp.array = eqx.static_field()

    def __init__(self,
                 subsets: Union[Array, List[int]],
                 input_size: int,
                 backend: str = 'jax'):
        super().__init__()
        mask = np.zeros((len(subsets), input_size), dtype=bool)
        for i, s in enumerate(subsets):
            mask[i, s] = True
        if backend == 'jax':
            self.mask = jnp.array(mask)
        else:
            self.mask = mask

    def __call__(self, x: Array) -> Array:
        raise NotImplementedError


class MaskedPerceptron(MaskedAggregator):
    """
    A masked perceptron model for classification tasks.

    Parameters:
    - subsets: an array of subsets used for masking.
    - input_size: the size of the input features.
    - key: a PRNGKey for random initialization.
    - backend: the backend framework to use (default: 'jax').

    Attributes:
    - linear: a linear layer for the perceptron.

    Methods:
    - __call__(self, x): performs forward pass of the perceptron.
    """

    linear: eqx.nn.Linear

    def __init__(self,
                 subsets: Array,
                 input_size: int,
                 key: "jax.random.PRNGKey",
                 backend: str = 'jax'):
        """
        Initialize the MaskedPerceptron class.

        Args:
            subsets (Array): the subsets of input features.
            input_size (int): the size of the input.
            key (jax.random.PRNGKey): the random key for initialization.
            backend (str, optional): the backend to use. Defaults to 'jax'.
        """
        super().__init__(subsets, input_size, backend)
        self.linear = eqx.nn.Linear(input_size,
                                    len(subsets),
                                    use_bias=False,
                                    key=key)

    @property
    def weight(self):
        """
        Returns the weight of the linear layer.

        Returns:
            Array: the weights of the linear layer.
        """
        return self.linear.weight

    @eqx.filter_jit
    def __call__(self, x: Array) -> Array:
        """
        Performs forward pass of the perceptron.

        Args:
            x (Array): the input features.

        Returns:
            Array: the output of the perceptron.
        """
        return (self.weight * self.mask) @ x


class MaskedSum(MaskedAggregator):

    def __call__(self, x: Array) -> Array:
        """
        performs a masked sum aggregation on the input array x.

        Args:
            x (Array): input array to aggregate.

        Returns:
            Array: sum of x for True mask locations.  
        """
        return self.mask @ x


class MaskedOr(MaskedAggregator):

    def __call__(self, x):
        """Performs a masked OR aggregation.

        For each row of the input array `x`, performs a logical OR operation between 
        elements of that row and the mask. Returns a boolean array indicating if there 
        was at least one `True` value for each row.

        Args:
            x: Input array to aggregate. Can be numpy ndarray or jax ndarray.

        Returns: 
            Boolean ndarray indicating if there was at least one True value in each 
            row of x after applying the mask.
        """
        if isinstance(x, jax.Array):
            return jnp.any(self.mask & (x != 0), axis=1)
        else:
            return np.any(self.mask & (x != 0), axis=1)


class AggregateRepresentation(eqx.Module):
    """
    AggregateRepresentation aggregates input codes into target codes.
    
    It initializes masked aggregators based on the target scheme's 
    aggregation and aggregation groups. On call, it splits the input, 
    aggregates each split with the corresponding aggregator, and 
    concatenates the results.
    
    Handles both jax and numpy arrays for the input.

    Attributes:
        aggregators: a list of masked aggregators.
        splits: a tuple of integers indicating the splits of the input.

    Methods:
        __call__(self, x): performs the aggregation.
    """
    aggregators: List[MaskedAggregator]
    splits: Tuple[int] = eqx.static_field()

    def __init__(self,
                 source_scheme: CodingScheme,
                 target_scheme: CodingScheme,
                 key: "jax.random.PRNGKey" = None,
                 backend: str = 'numpy'):
        """
        Initializes an AggregateRepresentation.
    
        Constructs the masked aggregators based on the target scheme's 
        aggregation and aggregation groups. Splits the input into sections 
        for each aggregator.
        
        Args:
            source_scheme: Source coding scheme to aggregate from 
            target_scheme: Target coding scheme to aggregate to
            key: JAX PRNGKey for initializing perceptrons
            backend: 'numpy' or 'jax' 
        """
        super().__init__()
        self.aggregators = []
        aggs = target_scheme.aggregation
        agg_grps = target_scheme.aggregation_groups
        grps = target_scheme.groups
        splits = []

        def is_contagious(x):
            return x.max() - x.min() == len(x) - 1 and len(set(x)) == len(x)

        for agg in aggs:
            selectors = []
            agg_offset = len(source_scheme)
            for grp in agg_grps[agg]:
                input_codes = grps[grp]
                input_index = sorted(source_scheme.index[c]
                                     for c in input_codes)
                input_index = np.array(input_index, dtype=np.int32)
                assert is_contagious(input_index), (
                    f"Selectors must be contiguous, but {input_index} is not. Codes: {input_codes}. Group: {grp}"
                )
                agg_offset = min(input_index.min(), agg_offset)
                selectors.append(input_index)
            selectors = [s - agg_offset for s in selectors]
            agg_input_size = sum(len(s) for s in selectors)
            max_index = max(s.max() for s in selectors)
            assert max_index == agg_input_size - 1, (
                f"Selectors must be contiguous, max index is {max_index} but size is {agg_input_size}"
            )
            splits.append(agg_input_size)

            if agg == 'w_sum':
                self.aggregators.append(
                    MaskedPerceptron(selectors, agg_input_size, key, backend))
                (key,) = jrandom.split(key, 1)
            elif agg == 'sum':
                self.aggregators.append(
                    MaskedSum(selectors, agg_input_size, backend))
            elif agg == 'or':
                self.aggregators.append(
                    MaskedOr(selectors, agg_input_size, backend))
            else:
                raise ValueError(f"Aggregation {agg} not supported")
        splits = np.cumsum([0] + splits)[1:-1]
        self.splits = tuple(splits.tolist())

    @eqx.filter_jit
    def __call__(self, inpatient_input: Array) -> Array:
        """
        Apply aggregators to the input data.

        Args:
            inpatient_input (Array): the input data to be processed.

        Returns:
            Array: the processed data after applying aggregators.
        """
        if isinstance(inpatient_input, jax.Array):
            splitted = jnp.hsplit(inpatient_input, self.splits)
            return jnp.hstack(
                [agg(x) for x, agg in zip(splitted, self.aggregators)])

        splitted = np.hsplit(inpatient_input, self.splits)
        return np.hstack(
            [agg(x) for x, agg in zip(splitted, self.aggregators)])


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
        return InpatientInput(df['code_index'].values, df['rate'].values, df['starttime'].values, df['endtime'].values)

    def __call__(self, t: float) -> Array:
        """
        Returns the vectorized input at time t.

        Args:
            t (float): the time at which to return the input.

        Returns:
            Array: the input at time t.
        """

        mask = (self.starttime <= t) & (t < self.endtime)
        index = self.code_index[mask]
        rate = self.rate[mask]
        adm_input = np.zeros(len(self.code_index), dtype=rate.dtype)
        adm_input[index] += rate
        return adm_input

    @classmethod
    def empty(cls) -> InpatientInput:
        """
        Returns an empty InpatientInput object.

        Args:
            size (int): the size of the input.

        Returns:
            InpatientInput: an empty InpatientInput object.
        """
        zvec = np.zeros(0, dtype=bool)
        return cls(zvec.astype(int), zvec, zvec, zvec)


class InpatientInterventions(Data):
    # TODO: Add docstring.
    hosp_procedures: Optional[InpatientInput]
    icu_inputs: Optional[InpatientInput]
    icu_procedures: Optional[InpatientInput]

    def to_dataframes(self) -> Dict[str, pd.DataFrame]:
        df1 = self.hosp_procedures.to_dataframe() if self.hosp_procedures is not None else None
        df2 = self.icu_procedures.to_dataframe() if self.icu_procedures is not None else None
        df3 = self.icu_inputs.to_dataframe() if self.icu_inputs is not None else None
        return {'hosp_procedures': df1, 'icu_procedures': df2, 'icu_inputs': df3}

    @staticmethod
    def from_dataframes(dfs: Dict[str, pd.DataFrame]) -> "InpatientInterventions":
        df1 = dfs.get('hosp_procedures')
        df2 = dfs.get('icu_procedures')
        df3 = dfs.get('icu_inputs')
        hosp_procedures = InpatientInput.from_dataframe(df1) if df1 is not None else None
        icu_procedures = InpatientInput.from_dataframe(df2) if df2 is not None else None
        icu_inputs = InpatientInput.from_dataframe(df3) if df3 is not None else None
        return InpatientInterventions(hosp_procedures, icu_procedures, icu_inputs)

    @property
    def timestamps(self) -> List[float]:
        timestamps = []
        for k in self.__dict__.keys():
            ii = getattr(self, k, None)
            if isinstance(ii, InpatientInput):
                timestamps.extend(ii.starttime)
                timestamps.extend(ii.endtime)
        return timestamps


class SegmentedInpatientInterventions(Data):
    time: Array
    hosp_procedures: Array
    icu_procedures: Array
    icu_inputs: Array

    @classmethod
    def from_interventions(cls, inpatient_interventions: InpatientInterventions, terminal_time: float):
        time = [np.clip(t, 0.0, terminal_time)
                for t in inpatient_interventions.timestamps
                ] + [0.0, terminal_time]
        time = np.unique(np.hstack(time, dtype=np.float32))
        time = cls.pad_array(time, value=np.nan)
        t0 = time[:-1]
        hosp_procedures = cls._segment(t0, inpatient_interventions.hosp_procedures)
        icu_procedures = cls._segment(t0, inpatient_interventions.icu_procedures)
        icu_inputs = cls._segment(t0, inpatient_interventions.icu_inputs)
        return cls(time, hosp_procedures, icu_procedures, icu_inputs)

    @staticmethod
    def pad_array(array: Array,
                  maximum_padding: int = 100,
                  value: float = 0.0) -> Array:
        """
        Pad array to be a multiple of maximum_padding. This is efficient to 
        avoid jit-compiling a different function for each array shape.
        
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
    def _segment(t0_padded: Array, inpatient_input: InpatientInput) -> Array:
        """
        Generate segmented procedures and apply the specified procedure transformation.
        Returns:
            numpy.ndarray: the processed segments.
        """
        t = t0_padded[~np.isnan(t0_padded)]
        t_nan = t0_padded[np.isnan(t0_padded)]
        out = np.vstack([inpatient_input(ti) for ti in t])
        pad = np.zeros((len(t_nan), out[0].shape[0]), dtype=out.dtype)
        return np.vstack([out, pad])


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
    interventions: Optional[InpatientInterventions]
    leading_observable: Optional[InpatientObservables] = None

    def to_hdf(self, path: str, admissions_key: str) -> None:
        """
        Save the admission data to an HDF5 file.

        Args:
            path (str): the path to the HDF5 file.
            admissions_key (str): the key to use for the admission data.
        """
        key = f'{admissions_key}/{self.admission_id}'
        adm_dates = pd.DataFrame({'start': self.admission_dates[0],
                                  'end': self.admission_dates[1],
                                  'dx_codes_scheme': self.dx_codes.scheme,
                                  'dx_codes_history_scheme': self.dx_codes_history.scheme,
                                  'outcome_scheme': self.outcome.scheme}, index=[0])

        adm_dates.to_hdf(path, key=f'{key}/admission_meta', format='table')
        pd.DataFrame(self.dx_codes.vec).to_hdf(path, key=f'{key}/dx_codes', format='table')
        pd.DataFrame(self.dx_codes_history.vec).to_hdf(path, key=f'{key}/dx_codes_history', format='table')
        pd.DataFrame(self.outcome.vec).to_hdf(path, key=f'{key}/outcome', format='table')
        if self.observables is not None:
            self.observables.to_dataframe().to_hdf(path, key=f'{key}/observables', format='table')
        if self.leading_observable is not None:
            self.leading_observable.to_dataframe().to_hdf(path, key=f'{key}/leading_observable', format='table')
        if self.interventions is not None:
            for k, v in self.interventions.to_dataframes().items():
                v.to_hdf(path, key=f'{key}/interventions/{k}', format='table')

    @staticmethod
    def from_hdf_store(hdf_store: pd.HDFStore, admissions_key: str, admission_id: str) -> 'Admission':
        """
        Load the admission data from an HDF5 file.

        Args:
            hdf_store (pd.HDFStore): the HDF5 store.
            admissions_key (str): the key to use for the admission data.
            admission_id (str): the admission ID.

        Returns:
            Admission: the admission data.
        """
        key = f'{admissions_key}/{admission_id}'
        adm_meta = hdf_store[f'{key}/admission_meta']
        adm_dates = (adm_meta['start'].iloc[0], adm_meta['end'].iloc[0])
        dx_codes_scheme = adm_meta['dx_codes_scheme'].iloc[0]
        dx_codes_history_scheme = adm_meta['dx_codes_history_scheme'].iloc[0]
        outcome_scheme = adm_meta['outcome_scheme'].iloc[0]
        dx_codes = CodesVector(hdf_store[f'{key}/dx_codes'], dx_codes_scheme)
        dx_codes_history = CodesVector(hdf_store[f'{key}/dx_codes_history'], dx_codes_history_scheme)
        outcome = CodesVector(hdf_store[f'{key}/outcome'], outcome_scheme)

        observables = None
        leading_observables = None
        interventions = None

        if f'{key}/observables' in hdf_store:
            observables = InpatientObservables.from_dataframe(hdf_store[f'{key}/observables'])
        if f'{key}/leading_observable' in hdf_store:
            leading_observable = InpatientObservables.from_dataframe(hdf_store[f'{key}/leading_observable'])
        if f'{key}/interventions' in hdf_store:
            interventions = InpatientInterventions.from_dataframes({
                k: hdf_store[f'{key}/interventions/{k}'] for k in hdf_store[f'{key}/interventions'].keys()
            })

        return Admission(admission_id=admission_id,
                         admission_dates=adm_dates,
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


class DemographicVectorConfig(Config):
    """
    Configuration class for demographic vector.

    Attributes:
        gender (bool): indicates whether gender is included in the vector.
        age (bool): indicates whether age is included in the vector.
        ethnicity (bool): indicates whether ethnicity is included in the vector.
    """
    gender: bool = False
    age: bool = False
    ethnicity: bool = False


class CPRDDemographicVectorConfig(DemographicVectorConfig):
    """
    Configuration class for CPRD demographic vector.
    """
    imd: bool = False


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

    @classmethod
    @property
    def separator(cls) -> str:
        return f':{cls.__name__}:'

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the static information to a pandas DataFrame.

        Returns:
            pd.DataFrame: a pandas DataFrame containing the static information.
        """
        df = pd.DataFrame()
        if self.gender is not None:
            df.loc[0, 'gender'] = self.separator.join(self.gender.to_codeset())
            df.loc[0, 'gender_scheme'] = self.gender.scheme
        if self.ethnicity is not None:
            df.loc[0, 'ethnicity'] = self.separator.join(self.ethnicity.to_codeset())
            df.loc[0, 'ethnicity_scheme'] = self.ethnicity.scheme
        if self.date_of_birth is not None:
            df.loc[0, 'date_of_birth'] = self.date_of_birth
        return df

    @classmethod
    def _from_dataframe_data(cls, df: pd.DataFrame):
        """
        Extracts the static information from a pandas DataFrame.

        Args:
            df (pd.DataFrame): the pandas DataFrame containing the static information.
        """
        data = {}
        if 'gender_scheme' in df.columns:
            scheme = CodingScheme.from_name(df.loc[0, 'gender_scheme'])
            codes = df.loc[0, 'gender'].split(cls.separator)
            data['gender'] = scheme.codeset2vec(codes)
        if 'ethnicity_scheme' in df.columns:
            scheme = CodingScheme.from_name(df.loc[0, 'ethnicity_scheme'])
            codes = df.loc[0, 'ethnicity'].split(cls.separator)
            data['ethnicity'] = scheme.codeset2vec(codes)
        if 'date_of_birth' in df.columns:
            data['date_of_birth'] = df.loc[0, 'date_of_birth']
        return data

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, demographic_vector_config: DemographicVectorConfig) -> 'StaticInfo':
        """
        Converts a pandas DataFrame to a StaticInfo object.
        """

        return cls(demographic_vector_config=demographic_vector_config,
                   **cls._from_dataframe_data(df))

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

    @classmethod
    def _from_dataframe_data(cls, df: pd.DataFrame):
        data = super()._from_dataframe_data(df)
        if 'imd_scheme' in df.columns:
            scheme = CodingScheme.from_name(df.loc[0, 'imd_scheme'])
            codes = df.loc[0, 'imd'].split(cls.separator)
            data['imd'] = scheme.codeset2vec(codes)
        return data


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

    def to_hdf(self, path: str, patients_key: str) -> None:
        """
        Save the patient data to an HDF5 file.

        Args:
            path (str): the path to the HDF5 file.
            patients_key (str): the key to use for the patient data.
        """
        key = f'{patients_key}/{self.subject_id}'
        self.static_info.to_dataframe().to_hdf(path, key=f'{key}/static_info', format='table')
        for i, adm in enumerate(self.admissions):
            adm.to_hdf(path, f'{key}/admissions')

        admission_ids = [adm.admission_id for adm in self.admissions]
        pd.DataFrame(admission_ids).to_hdf(path, key=f'{key}/admission_ids', format='table')

    @staticmethod
    def from_hdf_store(hdf_store: pd.HDFStore, patients_key: str, subject_id: int,
                       demographic_vector_config: DemographicVectorConfig) -> 'Patient':
        """
        Load the patient data from an HDF5 file.

        Args:
            hdf_store (pd.HDFStore): the HDF5 store.
            patients_key (str): the key to use for the patient data.
            subject_id (int): the subject ID.

        Returns:
            Patient: the patient data.
        """
        key = f'{patients_key}/{subject_id}'
        static_info = StaticInfo.from_dataframe(hdf_store[f'{key}/static_info'], demographic_vector_config)
        admission_ids = hdf_store[f'{key}/admission_ids']
        admissions = [Admission.from_hdf_store(hdf_store, f'{key}/admissions', aid) for aid in admission_ids]
        return Patient(subject_id=subject_id, static_info=static_info, admissions=admissions)
