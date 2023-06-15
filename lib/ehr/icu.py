from __future__ import annotations
from datetime import date
from collections import namedtuple
from dataclasses import dataclass
from typing import List, Tuple, Set, Callable, Optional, Union
from absl import logging
import numpy as np

@dataclass
class NumericMarkers:
    """
    NumericMarkers class encapsulates the patient EHRs numeric markers.
    """
    values: np.ndarray
    mask: np.ndarray
    marker_scheme


@dataclass
class PointInputs:
    """
    PointInput class encapsulates the patient EHRs single timestamp inputs.
    """
    amount: np.ndarray
    input_scheme

@dataclass
class IntervalInputs:
    """
    IntervalInputs class encapsulates the patient EHRs interval inputs.
    """
    rates: np.ndarray
    input_scheme


