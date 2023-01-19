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
        cls.random_predictions = m3_interface.random_predictions(splits[0],
                                                                 splits[1],
                                                                 seed=0)
        cls.random_predictions2 = m3_interface.random_predictions(splits[0],
                                                                  splits[1],
                                                                  seed=2)

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
        micro_auc = metric.value_extractor({'field': 'micro_auc'})
        macro_auc = metric.value_extractor({'field': 'macro_auc'})

        df_rnd = metric.to_df(0, self.random_predictions)

        # AUC of randomized decisions almost in (0.4, 0.6)
        self.assertGreater(df_rnd.max().max(), 0.4)
        self.assertLess(df_rnd.min().min(), 0.6)

        df_truth = metric.to_df(0, self.cheating_predictions)

        # AUC of perfect decision = 1.0
        self.assertEqual(set(onp.unique(df_truth)), {1.0}, msg=df_truth)

        self.assertLess(micro_auc(df_rnd), micro_auc(df_truth))
        self.assertLess(macro_auc(df_rnd), macro_auc(df_truth))

    def test_code_auc(self):
        metric = CodeAUC(m3_interface)
        mean_auc = metric.aggregate_extractor({
            'aggregate': 'mean',
            'field': 'auc'
        })

        df_rnd = metric.to_df(0, self.random_predictions)
        df_truth = metric.to_df(0, self.cheating_predictions)
        cols = df_rnd.columns
        auc_cols = [c for c in cols if c.split('.')[-1] == 'auc']
        self.assertEqual(len(auc_cols), len(metric.index2code))

        a_rnd = df_rnd[auc_cols].values.flatten()
        a_truth = df_truth[auc_cols].values.flatten()
        mask = onp.isnan(a_rnd)

        # Perfect predictions
        self.assertEqual(set(a_truth[~mask]), {1.0})
        # AUC(prefect) > AUC(random)
        self.assertTrue((a_rnd[~mask] <= a_truth[~mask]).all())

        # Test value extractors.
        self.assertEqual(mean_auc(df_rnd), a_rnd[~mask].mean())
        self.assertEqual(mean_auc(df_truth), a_truth[~mask].mean())

    def test_until_first_code_auc(self):
        metric = UntilFirstCodeAUC(m3_interface)
        mean_auc = metric.aggregate_extractor({
            'aggregate': 'mean',
            'field': 'auc'
        })

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

        self.assertEqual(mean_auc(df_rnd), a_rnd[~mask].mean())
        self.assertEqual(mean_auc(df_truth), a_truth[~mask].mean())

    def test_code_group_alarm_acc(self):
        m3_groups = m3_interface.dx_outcome_by_percentiles(percentile_range=20,
                                                           subjects=splits[0])
        ks = [1, 5, 10, 15]
        metric = CodeGroupTopAlarmAccuracy(code_groups=m3_groups,
                                           top_k_list=ks,
                                           subject_interface=m3_interface)

        top5acc_leastfrequent = metric.value_extractor({
            'k':
            5,
            'group_index':
            len(m3_groups) - 1
        })

        df_rand = metric.to_df(0, self.random_predictions)
        df_truth = metric.to_df(0, self.cheating_predictions)

        self.assertLess(top5acc_leastfrequent(df_rand),
                        top5acc_leastfrequent(df_truth))

    def test_admission_auc(self):
        metric = AdmissionAUC(subject_interface=m3_interface)
        mean_auc = metric.aggregate_extractor({
            'field': 'auc',
            'aggregate': 'mean'
        })

        df_rnd = metric.to_df(0, self.random_predictions)
        df_truth = metric.to_df(0, self.cheating_predictions)
        self.assertEqual(mean_auc(df_truth), 1.0)
        self.assertLess(mean_auc(df_rnd), mean_auc(df_truth))

    def test_metrics_collection(self):
        m3_groups = m3_interface.dx_outcome_by_percentiles(percentile_range=20,
                                                           subjects=splits[0])
        vauc_m = VisitsAUC(m3_interface)
        vmicroauc = vauc_m.value_extractor({'field': 'micro_auc'})

        cauc_m = CodeAUC(m3_interface)
        cauc_median = cauc_m.aggregate_extractor({
            'aggregate': 'median',
            'field': 'auc'
        })

        fcauc_m = UntilFirstCodeAUC(m3_interface)
        fcauc_median = fcauc_m.aggregate_extractor({
            'aggregate': 'median',
            'field': 'auc'
        })

        alarm_m = CodeGroupTopAlarmAccuracy(code_groups=m3_groups,
                                            top_k_list=[1, 5, 10, 15],
                                            subject_interface=m3_interface)
        alarm10g0 = alarm_m.value_extractor({'k': 10, 'group_index': 0})

        all_metrics = MetricsCollection([vauc_m, cauc_m, fcauc_m, alarm_m])

        rand_df = all_metrics.to_df(0, self.random_predictions)
        truth_df = all_metrics.to_df(0, self.cheating_predictions)

        for objective in (vmicroauc, cauc_median, fcauc_median, alarm10g0):
            self.assertGreater(objective(truth_df), objective(rand_df))

    def test_delong_tests(self):
        delong = DeLongTest(subject_interface=m3_interface)
        df = delong.to_df({
            'random1': self.random_predictions,
            'random2': self.random_predictions2,
            'recency': self.recency_predictions,
            'mean': self.mean_predictions,
            'historical': self.historical_predictions,
            'cheating': self.cheating_predictions
        })

        pvalue = lambda i, m1, m2: delong.value_extractor({
            'field': 'p_val',
            'pair': (m1, m2),
            'code_index': i
        })(df)
        auc = lambda i, m: delong.value_extractor({
            'field': 'auc',
            'model': m,
            'code_index': i
        })(df)
        aucvar = lambda i, m: delong.value_extractor({
            'field': 'auc_var',
            'model': m,
            'code_index': i
        })(df)
        npos = lambda i: delong.value_extractor({
            'field': 'n_pos',
            'code_index': i
        })(df)

        for i, code in delong.index2code.items():
            if npos(i) < 2:
                continue

            if npos(i) > 10:
                # Random against random
                self.assertGreater(pvalue(i, 'random1', 'random2'), 0.05)

                # Check the pair-order is normalized.
                self.assertEqual(pvalue(i, 'mean', 'recency'),
                                pvalue(i, 'recency', 'mean'))

                # Random against perfect
                self.assertLess(pvalue(i, 'random1', 'cheating'), 0.05)

            self.assertGreater(auc(i, 'cheating'), auc(i, 'random1'))
            self.assertLess(aucvar(i, 'cheating'), aucvar(i, 'random1'))
