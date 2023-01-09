import unittest
import random

import numpy as np
from jax.tree_util import tree_leaves

from icenode import ehr
from icenode.embeddings import GRAM, CachedGRAM


def setUpModule():
    global m3_interface, gram_config, subjects_sample

    m3_dataset = ehr.MIMIC3EHRDataset.from_meta_json(
        'test/integration/fixtures/synthetic_mimic/mimic3_syn_meta.json')

    m3_interface = ehr.Subject_JAX.from_dataset(
        m3_dataset, {
            'dx': 'dx_icd9',
            'dx_dagvec': True,
            'dx_outcome': 'dx_icd9_filter_v1'
        })

    m3_subjects = list(m3_interface.keys())
    subjects_sample = random.sample(m3_subjects, 5)

    gram_config = {
        "category": "dx",
        "subject_interface": m3_interface,
        "train_ids": m3_subjects,
        "attention_dim": 20,
        "attention_method": "tanh",
        "embeddings_dim": 30,
        "glove_config": {
            "iterations": 2,
            "window_size_days": 300
        }
    }


def tearDownModule():
    pass


class CachedGRAMTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.gram0 = GRAM(**gram_config)
        cls.gramC = CachedGRAM(**gram_config)

        cls.gram0_params = [cls.gram0.init_params(s) for s in (0, 10, 11)]
        cls.gramC_params = [cls.gramC.init_params(s) for s in (0, 10, 11)]

    def test_encoding_difference(self):
        for gram0_p, gramC_p in zip(self.gram0_params, self.gramC_params):

            # Compare paramters themselves
            for p0, pC in zip(tree_leaves(gram0_p), tree_leaves(gramC_p)):
                self.assertSequenceEqual(p0.tolist(), pC.tolist())

            G0 = self.gram0.compute_embeddings_mat(gram0_p)
            GC = self.gramC.compute_embeddings_mat(gramC_p)

            for i in subjects_sample:
                for adm in m3_interface[i]:
                    v = adm.dx_vec

                    v0 = self.gram0.encode(G0, v)
                    vC = self.gramC.encode(GC, v)
                    np.testing.assert_allclose(v0, vC, atol=0, rtol=1e-5)
