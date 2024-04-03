from typing import (AbstractSet, Any, Callable, Dict, Iterable, List, Mapping,
                    Optional, Tuple, Union)

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.nn as jnn
import equinox as eqx

from ..metric.loss import mse

