"""."""

import unittest

import jax.random as jrandom
import jax.numpy as jnp
import numpy as onp

from lib import ml
from lib.ehr.coding_scheme import DxICD10, DxICD9, DxCCS, PrICD9
from lib.ehr import (Subject_JAX, OutcomeExtractor, MIMIC3EHRDataset,
                     MIMIC4EHRDataset)
from lib.utils import load_config
from lib.embeddings import embeddings_from_conf

from lib.metric.stat import (MetricsCollection, CodeAUC, DeLongTest,
                             CodeGroupTopAlarmAccuracy, AdmissionAUC,
                             UntilFirstCodeAUC, VisitsAUC)


def setUpModule():
    global m3_interface, splits

    m3_dataset = MIMIC3EHRDataset.from_meta_json(
        'test/integration/fixtures/synthetic_mimic/mimic3_syn_meta.json')
    icd9_outcome = OutcomeExtractor('dx_icd9_filter_v1')
    m3_interface = Subject_JAX.from_dataset(
        m3_dataset, dict(dx=DxICD9(), dx_outcome=icd9_outcome))
    splits = m3_interface.random_splits(0.7, 0.85, 42)


class MetricsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.random_predictions = m3_interface.random_predictions(
            splits[0], splits[1])
        cls.recency_predictions = m3_interface.recency_predictions(
            splits[0], splits[1])
        cls.historical_predictions = m3_interface.historical_predictions(
            splits[0], splits[1])
        cls.mean_predictions = m3_interface.mean_predictions(
            splits[0], splits[1])
        cls.cheating_predictions = m3_interface.cheating_predictions(
            splits[0], splits[1])

    def test_visits_auc(self):
        metric = VisitsAUC(m3_interface)
        df_rnd = metric.to_df(0, self.random_predictions)

        # AUC of randomized decisions almost in (0.4, 0.6)
        self.assertGreater(df_rnd.max().max(), 0.4)
        self.assertLess(df_rnd.min().min(), 0.6)

        df_truth = metric.to_df(0, self.cheating_predictions)

        # AUC of perfect decision = 1.0
        self.assertEqual(set(onp.unique(df_truth)), {1.0}, msg=df_truth)

    def test_code_auc(self):
        metric = CodeAUC(m3_interface)
        df_rnd = metric.to_df(0, self.random_predictions)
        df_truth = metric.to_df(0, self.cheating_predictions)
        cols = df_rnd.columns
        auc_cols = [c for c in cols if c.split('.')[-1] == 'auc']
        self.assertEqual(len(auc_cols), len(metric.index2code))

        a_rnd = df_rnd[auc_cols].values.flatten()
        a_truth = df_truth[auc_cols].values.flatten()
        mask = onp.isnan(a_rnd)
        self.assertEqual(set(a_truth[~mask]), {1.0})
        self.assertTrue((a_rnd[~mask] <= a_truth[~mask]).all())

    def test_until_first_code_auc(self):
        metric = UntilFirstCodeAUC(m3_interface)
        df_rnd = metric.to_df(0, self.random_predictions)
        df_truth = metric.to_df(0, self.cheating_predictions)
        cols = df_rnd.columns
        auc_cols = [c for c in cols if c.split('.')[-1] == 'auc']
        self.assertEqual(len(auc_cols), len(metric.index2code))

        a_rnd = df_rnd[auc_cols].values.flatten()
        a_truth = df_truth[auc_cols].values.flatten()
        mask = onp.isnan(a_rnd)
        self.assertEqual(set(a_truth[~mask]), {1.0})
        self.assertTrue((a_rnd[~mask] <= a_truth[~mask]).all())


    def test_code_group_alarm_acc(self):
        m3_groups = m3_interface.dx_outcome_by_percentiles(percentile_range=20,
                                                           subjects=splits[0])

        metric = CodeGroupTopAlarmAccuracy(code_groups=m3_groups,
                                           top_k_list=[1, 5, 10, 15],
                                           subject_interface=m3_interface)
        df = metric.to_df(0, self.random_predictions)

    def test_admission_auc(self):
        metric = AdmissionAUC(subject_interface=m3_interface)
        df = metric.to_df(0, self.random_predictions)

    def test_metrics_collection(self):
        m3_groups = m3_interface.dx_outcome_by_percentiles(percentile_range=20,
                                                           subjects=splits[0])

        metric = MetricsCollection([
            VisitsAUC(m3_interface),
            CodeAUC(m3_interface),
            UntilFirstCodeAUC(m3_interface),
            CodeGroupTopAlarmAccuracy(code_groups=m3_groups,
                                      top_k_list=[1, 5, 10, 15],
                                      subject_interface=m3_interface)
        ])

        df = metric.to_df(0, self.random_predictions)

    def test_delong_tests(self):
        metric = DeLongTest(subject_interface=m3_interface)
        df = metric.to_df({
            'random': self.random_predictions,
            'recency': self.recency_predictions,
            'mean': self.mean_predictions,
            'historical': self.historical_predictions,
            'cheating': self.cheating_predictions
        })
