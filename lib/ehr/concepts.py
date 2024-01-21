"""Data Model for Subjects in MIMIC-III and MIMIC-IV"""

from __future__ import annotations

from datetime import date
from typing import (List, Tuple, Optional, Dict, Union)

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pandas as pd

from .coding_scheme import (AbstractScheme, AbstractGroupedProcedures,
                            CodesVector, scheme_from_classname)
from ..base import Config, Data, Module


class InpatientObservables(Data):
    """
    Vectorized representation of inpatient observables.

    Attributes:
        time (jnp.ndarray): array of time values.
        value (jnp.ndarray): array of observable values.
        mask (jnp.ndarray): array indicating missing values.

    Methods:
        empty(size: int): creates an empty instance of InpatientObservables.
        __len__(): returns the length of the time array.
        as_dataframe(scheme: AbstractScheme, filter_missing_columns=False): converts the observables to a pandas DataFrame.
        groupby_code(index2code: Dict[int, str]): groups the observables by code.
        segment(t_sep: jnp.ndarray): splits the observables into segments based on time values.
        concat(observables: Union[InpatientObservables, List[InpatientObservables]]): concatenates multiple instances of InpatientObservables.
        time_binning(hours: float): bins the time-series into time-windows and averages the values in each window.
    """

    time: jnp.narray  # (time,)
    value: jnp.ndarray  # (time, size)
    mask: jnp.ndarray  # (time, size)

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

    def as_dataframe(self,
                     scheme: AbstractScheme,
                     filter_missing_columns=False) -> pd.DataFrame:
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
        if isinstance(self.time, jnp.ndarray):
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

    def segment(self, t_sep: jnp.ndarray) -> List[InpatientObservables]:
        """
        Splits the InpatientObservables object into multiple segments based on the given time points.

        Args:
            t_sep (jnp.ndarray): array of time points used to split the InpatientObservables object.

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
        if isinstance(observables[0].time, jnp.ndarray):
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
        index (int): index of the observable in the observable scheme.
        leading_hours (List[float]): list of leading hours to extract. Must be sorted.
        recovery_window (float): time window in hours to mask out between a nonzero value and zero.
        entry_neglect_window (float): hours to mask out in the beginning.
        minimum_acquisitions (int): minimum number of acquisitions before extracting the leading observable.
        scheme (str): name of the observation coding scheme to use.
    """

    index: int
    leading_hours: List[float]
    recovery_window: float
    entry_neglect_window: float
    minimum_acquisitions: int  # minimum number of acquisitions to consider
    scheme: str

    def __init__(
            self,
            index: int,
            leading_hours: List[float],
            recovery_window: float,
            entry_neglect_window: float,
            minimum_acquisitions: int,
            scheme: str,
    ):
        super().__init__()
        self.index = index
        self.leading_hours = list(leading_hours)
        self.recovery_window = recovery_window
        self.entry_neglect_window = entry_neglect_window
        self.minimum_acquisitions = minimum_acquisitions
        self.scheme = scheme

        # `leading_hours` must be sorted.
        assert all(
            x <= y for x, y in zip(self.leading_hours[:-1], self.leading_hours[1:])
        ), f"leading_hours must be sorted"


class LeadingObservableExtractor(Module):
    """
    Extracts leading observables from a given timestamp based on a specified configuration.

    Attributes:
        config (LeadingObservableExtractorConfig): the configuration for the extractor.
        _scheme (AbstractScheme): the scheme used for indexing.
        _index2code (Dict[int, str]): a dictionary mapping index to code.
        _code2index (Dict[str, int]): a dictionary mapping code to index.
    """
    config: LeadingObservableExtractorConfig
    _scheme: AbstractScheme
    _index2code: Dict[int, str]
    _code2index: Dict[str, int]

    def __init__(self, config: LeadingObservableExtractorConfig):
        """
        Initializes the LeadingObservableExtractor.

        Args:
            config (LeadingObservableExtractorConfig): the configuration for the extractor.
        """
        super().__init__(config=config)
        self._scheme = scheme_from_classname(config.scheme)
        desc = self._scheme.index2desc[config.index]
        self._index2code = dict(
            zip(range(len(config.leading_hours)),
                [f'{desc}_next_{h}hrs' for h in config.leading_hours]))
        self._code2index = {v: k for k, v in self._index2code.items()}

    def __len__(self):
        """
        Returns the number of leading hours.

        Returns:
            int: the number of leading hours.
        """
        return len(self.config.leading_hours)

    @property
    def index2code(self):
        """
        Returns the mapping of index to code.

        Returns:
            Dict[int, str]: the mapping of index to code.
        """
        return self._index2code

    @property
    def index2desc(self):
        """
        Returns the mapping of index to description.

        Returns:
            Dict[int, str]: the mapping of index to description.
        """
        return self.index2code

    @property
    def code2index(self):
        """
        Returns the mapping of code to index.

        Returns:
            Dict[str, int]: the mapping of code to index.
        """
        return self._code2index

    def empty(self):
        """
        Returns an empty InpatientObservables object.

        Returns:
            InpatientObservables: an empty InpatientObservables object.
        """
        return InpatientObservables.empty(len(self.config.leading_hours))

    @staticmethod
    def _nan_concat_leading_windows(x: np.ndarray) -> np.ndarray:
        """
        Generates sliding windows of the input array, padded with NaN values.

        Args:
            x (np.ndarray): the input array.

        Returns:
            np.ndarray: the sliding windows of the input array, padded with NaN values.
        """
        n = len(x)
        add_arr = np.full(n - 1, np.nan)
        x_ext = np.concatenate((add_arr, x[::-1]))
        strided = np.lib.stride_tricks.as_strided
        nrows = len(x_ext) - n + 1
        s = x_ext.strides[0]
        return strided(x_ext, shape=(nrows, n), strides=(s, s))[::-1, ::-1]

    @staticmethod
    def _nan_agg_nonzero(x, axis):
        """
        Aggregates the values in a given array along the specified axis, treating NaN values as zero.

        Args:
            x (np.ndarray): The input array.
            axis: The axis along which to aggregate.

        Returns:
            np.ndarray: The aggregated array.
        """
        all_nan = np.all(np.isnan(x), axis=axis) * 1.0
        replaced_nan = np.where(np.isnan(x), 0, x)
        return np.where(all_nan, np.nan, np.any(replaced_nan, axis=axis) * 1.0)

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

        mask = observables.mask[:, self.config.index]
        value = np.where(mask, observables.value[:, self.config.index], np.nan)
        time = observables.time
        # a time-window starting from timestamp_i for each row_i
        # if time = [t0, t1, t2, t3]
        # then t_leads = [[t0, t1, t2, t3],
        #                 [t1, t2, t3, nan],
        #                 [t2, t3, nan, nan],
        #                 [t3, nan, nan, nan]]
        t_leads = self._nan_concat_leading_windows(time)

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
        v_leads = self._nan_concat_leading_windows(value)

        values = []
        masks = []

        for t in self.config.leading_hours:
            # select the rows where the time-window is less than t
            # if time = [t0, t1, t2, t3]
            # then mask = [[0 < t, t1 - t0 < t, t2 - t0 < t, t3 - t0 < t],
            #              [0 < t, t2 - t1 < t, t3 - t1 < t, nan < t],
            #              [0 < t, t3 - t2 < t, nan < t, nan < t]],
            #              [0 < t, nan < t, nan < t, nan < t]]
            mask = delta_leads < t
            v_lead = np.where(mask, v_leads, np.nan)
            values.append(self._nan_agg_nonzero(v_lead, axis=1).flatten())

            masks.append(np.where(np.isnan(values[-1]), False, True))

        values = np.stack(values, axis=1)
        masks = np.stack(masks, axis=1)
        return InpatientObservables(time=time, value=values, mask=masks)


class MaskedAggregator(eqx.Module):
    """
    A class representing a masked aggregator.

    Attributes:
        mask (jnp.array): The mask used for aggregation.
    """

    mask: jnp.array = eqx.static_field()

    def __init__(self,
                 subsets: Union[np.ndarray, List[int]],
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

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
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

    linear: eqx.Linear

    def __init__(self,
                 subsets: jnp.ndarray,
                 input_size: int,
                 key: "jax.random.PRNGKey",
                 backend: str = 'jax'):
        """
        Initialize the MaskedPerceptron class.

        Args:
            subsets (jnp.ndarray): the subsets of input features.
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
            jnp.ndarray: the weights of the linear layer.
        """
        return self.linear.weight

    @eqx.filter_jit
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Performs forward pass of the perceptron.

        Args:
            x (jnp.ndarray): the input features.

        Returns:
            jnp.ndarray: the output of the perceptron.
        """
        return (self.weight * self.mask) @ x


class MaskedSum(MaskedAggregator):

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        performs a masked sum aggregation on the input array x.

        Args:
            x (jnp.ndarray): input array to aggregate.

        Returns:
            jnp.ndarray: sum of x for True mask locations.  
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
        if isinstance(x, np.ndarray):
            return np.any(self.mask & (x != 0), axis=1)
        else:
            return jnp.any(self.mask & (x != 0), axis=1)


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
                 source_scheme: AbstractScheme,
                 target_scheme: AbstractGroupedProcedures,
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
    def __call__(self, inpatient_input: Union[jnp.ndarray, np.ndarray]) -> Union[jnp.ndarray, np.ndarray]:
        """
        Apply aggregators to the input data.

        Args:
            inpatient_input (Union[jnp.ndarray, np.ndarray]): the input data to be processed.

        Returns:
            Union[jnp.ndarray, np.ndarray]: the processed data after applying aggregators.
        """
        if isinstance(inpatient_input, np.ndarray):
            splitted = np.hsplit(inpatient_input, self.splits)
            return np.hstack(
                [agg(x) for x, agg in zip(splitted, self.aggregators)])

        splitted = jnp.hsplit(inpatient_input, self.splits)
        return jnp.hstack(
            [agg(x) for x, agg in zip(splitted, self.aggregators)])


class InpatientInput(Data):
    """
    Represents inpatient input data.

    Attributes:
        index (jnp.ndarray): the index array. Each index value correspond to a unique input code.
        rate (jnp.ndarray): the rate array.
        starttime (jnp.ndarray): the start time array.
        endtime (jnp.ndarray): the end time array.
        size (int): the size of the input data.
    """

    index: jnp.ndarray
    rate: jnp.ndarray
    starttime: jnp.ndarray
    endtime: jnp.ndarray
    size: int

    def __call__(self, t: float) -> Union[jnp.ndarray, np.ndarray]:
        """
        Returns the vectorized input at time t.

        Args:
            t (float): the time at which to return the input.

        Returns:
            jnp.ndarray: the input at time t.
        """

        mask = (self.starttime <= t) & (t < self.endtime)
        if isinstance(self.index, np.ndarray):
            index = self.index[mask]
            rate = self.rate[mask]
            adm_input = np.zeros(self.size, dtype=rate.dtype)
            adm_input[index] += rate
            return adm_input
        else:
            index = jnp.where(mask, self.index, 0)
            rate = jnp.where(mask, self.rate, 0.0)
            adm_input = jnp.zeros(self.size, dtype=rate.dtype)
            return adm_input.at[index].add(rate)

    @classmethod
    def empty(cls, size: int):
        """
        Returns an empty InpatientInput object.

        Args:
            size (int): the size of the input.

        Returns:
            InpatientInput: an empty InpatientInput object.
        """
        zvec = np.zeros(0, dtype=bool)
        return cls(zvec.astype(int), zvec, zvec, zvec, size)


class InpatientInterventions(Data):
    """
    Represents a class for handling inpatient interventions data.

    Attributes:
        proc (Optional[InpatientInput]): the procedures' representation.
        input_ (Optional[InpatientInput]): the inputs' representation.
        time (jnp.ndarray): the unique timestamps of the interventions starts and ends.
        segmented_input (Optional[np.ndarray]): the segmented input array for the interventions.
        segmented_proc (Optional[np.ndarray]): the segmented procedure array for the interventions.

    Methods:
        segment_proc(self, proc_repr: Optional[AggregateRepresentation] = None): segments the procedure array and updates the segmented_proc attribute.
        segment_input(self, input_repr: Optional[AggregateRepresentation] = None): segments the input array and updates the segmented_input attribute.
    """
    proc: Optional[InpatientInput]
    input_: Optional[InpatientInput]

    time: jnp.ndarray
    segmented_input: Optional[np.ndarray]
    segmented_proc: Optional[np.ndarray]

    def __init__(self, proc: InpatientInput, input_: InpatientInput,
                 adm_interval: float):
        """
        Initialize the InpatientInterventions object.

        Args:
            proc (InpatientInput): the procedures' representation.
            input_ (InpatientInput): the inputs' representation.
            adm_interval (float): the admission interval.
        """
        super().__init__()
        self.proc = proc
        self.input_ = input_
        self.segmented_proc = None
        self.segmented_input = None

        time = [
            np.clip(t, 0.0, adm_interval)
            for t in (proc.starttime, proc.endtime, input_.starttime,
                      input_.endtime)
        ]
        time = np.unique(
            np.hstack(time + [0.0, adm_interval], dtype=np.float32))
        time = self.pad_array(time, value=np.nan)
        self.time = time

    @staticmethod
    def pad_array(array: np.ndarray,
                  maximum_padding: int = 100,
                  value: float = 0.0) -> np.ndarray:
        """
        Pad array to be a multiple of maximum_padding. This is efficient to 
        avoid jit-compiling a different function for each array shape.
        
        Args:
            array (np.ndarray): the array to be padded.
            maximum_padding (int, optional): the maximum padding. Defaults to 100.
            value (float, optional): the value to pad with. Defaults to 0.0.
            
        Returns:
            np.ndarray: the padded array."""

        n = len(array)
        n_pad = maximum_padding - (n % maximum_padding)
        if n_pad == maximum_padding:
            return array

        if isinstance(array, np.ndarray):
            _np = np
        else:
            _np = jnp

        return _np.pad(array,
                       pad_width=(0, n_pad),
                       mode='constant',
                       constant_values=value)

    @property
    def _np(self):
        if isinstance(self.time, np.ndarray):
            return np
        else:
            return jnp

    @property
    def t0_padded(self):
        """Start times for segmenting the interventions, padded to the `maximum_padding`."""
        return self.time[:-1]

    @property
    def t0(self):
        """Start times for segmenting the interventions."""
        t = self.time
        return t[~self._np.isnan(t)][:-1]

    @property
    def t1_padded(self):
        """End times for segmenting the interventions, padded to the `maximum_padding`."""
        return self.time[1:]

    @property
    def t1(self):
        """End times for segmenting the interventions."""
        t = self.time
        return t[~self._np.isnan(t)][1:]

    @property
    def t_sep(self):
        """Separation times for segmenting the interventions."""
        t = self.time
        return t[~self._np.isnan(t)][1:-1]

    @property
    def interval(self):
        """Length of the admission interval"""
        return jnp.nanmax(self.time) - jnp.nanmin(self.time)

    def _np_segment_proc(self, proc_repr: Optional[AggregateRepresentation]) -> np.ndarray:
        """
        Generate segmented procedures and apply the specified procedure transformation.

        Args:
            proc_repr (Optional[AggregateRepresentation]): the procedure transformation to use.

        Returns:
            numpy.ndarray: the processed segments.
        """
        t = self.t0_padded[~np.isnan(self.t0_padded)]
        t_nan = self.t0_padded[np.isnan(self.t0_padded)]

        if proc_repr is None:
            out = np.vstack([self.proc(ti) for ti in t])
        else:
            out = np.vstack([proc_repr(self.proc(ti)) for ti in t])
        pad = np.zeros((len(t_nan), out[0].shape[0]), dtype=out.dtype)
        return np.vstack([out, pad])

    def _np_segment_input(self, input_repr: Optional[AggregateRepresentation]) -> np.ndarray:
        """
        Generate segmented inputs and apply the specified input transformation.

        Args:
            input_repr (Optional[AggregateRepresentation]): the input transformation to use.

        Returns:
            numpy.ndarray: the processed segments.
        """
        t = self.t0_padded[~np.isnan(self.t0_padded)]
        t_nan = self.t0_padded[np.isnan(self.t0_padded)]

        if input_repr is None:
            out = np.vstack([self.input_(ti) for ti in t])
        else:
            out = np.vstack([input_repr(self.input_(ti)) for ti in t])

        pad = np.zeros((len(t_nan), out[0].shape[0]), dtype=out.dtype)
        return np.vstack([out, pad])

    def segment_proc(self,
                     proc_repr: Optional[AggregateRepresentation] = None) -> InpatientInterventions:
        """
        Segment the procedures and apply the specified procedure transformation.

        Args:
            proc_repr (Optional[AggregateRepresentation]): the procedure transformation to use.

        Returns:
            InpatientInterventions: the updated object.
        """
        proc_segments = self._np_segment_proc(proc_repr)
        update = eqx.tree_at(lambda x: x.segmented_proc,
                             self,
                             proc_segments,
                             is_leaf=lambda x: x is None)
        update = eqx.tree_at(lambda x: x.proc, update, None)
        return update

    def segment_input(self,
                      input_repr: Optional[AggregateRepresentation] = None) -> InpatientInterventions:
        """
        Segment the inputs and apply the specified input transformation.

        Args:
            input_repr (Optional[AggregateRepresentation]): the input transformation to use.

        Returns:
            InpatientInterventions: the updated object.
        """
        inp_segments = self._np_segment_input(input_repr)
        update = eqx.tree_at(lambda x: x.segmented_input,
                             self,
                             inp_segments,
                             is_leaf=lambda x: x is None)
        update = eqx.tree_at(lambda x: x.input_, update, None)
        return update


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
    observables: Optional[Union[InpatientObservables,
    List[InpatientObservables]]]
    interventions: Optional[InpatientInterventions]
    leading_observable: Optional[Union[InpatientObservables,
    List[InpatientObservables]]] = None

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
        constant_vec (Optional[jnp.ndarray]): constant vector representing the static information.

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
    constant_vec: Optional[Union[jnp.ndarray, np.array]] = eqx.static_field(init=False)

    def __post_init__(self):
        attrs_vec = []
        if self.demographic_vector_config.gender:
            assert self.gender is not None and len(
                self.gender) > 0, "Gender is not extracted from the dataset"
            attrs_vec.append(self.gender.vec)
        if self.demographic_vector_config.ethnicity:
            assert self.ethnicity is not None, \
                "Ethnicity is not extracted from the dataset"
            attrs_vec.append(self.ethnicity.vec)

        if len(attrs_vec) == 0:
            self.constant_vec = np.array([], dtype=jnp.float16)
        else:
            self.constant_vec = np.hstack(attrs_vec)

    def age(self, current_date: date) -> float:
        """
        Calculates the age of the patient based on the current date.

        Args:
            current_date (date): the current date.

        Returns:
            float: the age of the patient in years.
        """
        return (current_date - self.date_of_birth).days / 365.25

    def demographic_vector(self, current_date: date) -> jnp.ndarray:
        """
        Returns the demographic vector based on the current date.

        Args:
            current_date (date): the current date.

        Returns:
            jnp.ndarray: the demographic vector.
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
            vec (jnp.ndarray): the vector to be concatenated.

        Returns:
            jnp.ndarray: the concatenated vector.
        """
        return jnp.hstack((age, vec), dtype=jnp.float16)


class CPRDStaticInfo(StaticInfo):
    """
    Represents static information extracted from the CPRD dataset.

    Attributes:
        imd (Optional[CodesVector]): the IMD (Index of Multiple Deprivation) vector.
    """

    imd: Optional[CodesVector] = None

    def __post_init__(self):
        attrs_vec = []
        if self.demographic_vector_config.gender:
            assert self.gender is not None and len(
                self.gender) > 0, "Gender is not extracted from the dataset"
            attrs_vec.append(self.gender.vec)
        if self.demographic_vector_config.ethnicity:
            assert self.ethnicity is not None, \
                "Ethnicity is not extracted from the dataset"
            attrs_vec.append(self.ethnicity.vec)

        if self.demographic_vector_config.imd:
            assert self.imd is not None, \
                "IMD is not extracted from the dataset"
            attrs_vec.append(self.imd.vec)

        if len(attrs_vec) == 0:
            self.constant_vec = np.array([], dtype=jnp.float16)
        else:
            self.constant_vec = np.hstack(attrs_vec)


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
