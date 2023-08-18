"""Integration Test for EHR Dx predictive model
To execute this test from the root directory:
    python -m unittest discover -s test/integration/ml -v -t .
"""

import unittest
import string
import os
import uuid
import random
from typing import List, Tuple, Type
from collections import namedtuple

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import equinox as eqx

from lib import ml
from lib.ehr.coding_scheme import DxICD10, DxICD9, DxCCS, PrICD9
from lib.ehr.interface import Patients
from lib.ehr.dataset import MIMIC3EHRDataset, MIMIC4EHRDataset
from lib.utils import load_config
from lib.metric import (CodeAUC, UntilFirstCodeAUC, AdmissionAUC,
                        MetricsCollection)


def setUpModule():
    pass

def tearDownModule():
    pass

class DxCommonTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def test_train_read_write_params(self):
        pass

if __name__ == '__main__':
    unittest.main()
