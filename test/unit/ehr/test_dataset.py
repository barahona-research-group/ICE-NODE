import unittest

from parameterized import parameterized

from lib.ehr.dataset import TableConfig, AdmissionLinkedTableConfig, SubjectLinkedTableConfig, TimestampedTableConfig, \
    IntervalBasedTableConfig, DatasetTables, Dataset
from lib.ehr.dataset import DatasetPipeline, ProcessOverlappingAdmissions


class TestTableConfig(unittest.TestCase):

    def test_alias_dict(self):
        config = TableConfig(name='test')
        self.assertEqual(config.alias_dict, {})
        self.fail()

    def test_alias_id_dict(self):
        config = TableConfig(name='test')
        self.assertEqual(config.alias_id_dict, {})
        self.fail()

    def test_index(self):
        config = TableConfig(name='test')
        self.assertIsNone(config.index)
        self.fail()

    def test_time_cols(self):
        config = TableConfig(name='test')
        self.assertEqual(config.time_cols, ())
        self.fail()

    def test_coded_cols(self):
        config = TableConfig(name='test')
        self.assertEqual(config.coded_cols, ())
        self.fail()


class TestAdmissionLinkedTableConfig(unittest.TestCase):

    def test_admission_id_alias(self):
        config = AdmissionLinkedTableConfig(name='test')
        self.assertEqual(config.admission_id_alias, 'hadm_id')
        self.fail()


class TestSubjectLinkedTableConfig(unittest.TestCase):

    def test_subject_id_alias(self):
        config = SubjectLinkedTableConfig(name='test')
        self.assertEqual(config.subject_id_alias, 'subject_id')
        self.fail()


# Additional test cases for other TableConfig subclasses

import unittest
from lib.ehr.dataset import DatasetTablesConfig


class TestDatasetTablesConfig(unittest.TestCase):

    def test_assert_consistent_aliases(self):
        # Arrange
        config = DatasetTablesConfig()

        # Act
        config._assert_consistent_aliases()

        # Assert
        # No assertion, just validate no exceptions
        self.fail()

    def test_table_config_dict(self):
        # Arrange
        config = DatasetTablesConfig()

        # Act
        result = config.table_config_dict

        # Assert
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)
        self.assertTrue(all(isinstance(v, TableConfig) for v in result.values()))
        self.fail()

    def test_timestamped_table_config_dict(self):
        # Arrange
        config = DatasetTablesConfig()

        # Act
        result = config.timestamped_table_config_dict

        # Assert
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)
        self.assertTrue(all(isinstance(v, TimestampedTableConfig) for v in result.values()))
        self.fail()

    def test_interval_based_table_config_dict(self):
        # Arrange
        config = DatasetTablesConfig()

        # Act
        result = config.interval_based_table_config_dict

        # Assert
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)
        self.assertTrue(all(isinstance(v, IntervalBasedTableConfig) for v in result.values()))
        self.fail()

    def test_indices(self):
        # Arrange
        config = DatasetTablesConfig()

        # Act
        result = config.indices

        # Assert
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)
        self.assertTrue(all(isinstance(k, str) and isinstance(v, str) for k, v in result.items()))
        self.fail()

    def test_time_cols(self):
        # Arrange
        config = DatasetTablesConfig()

        # Act
        result = config.time_cols

        # Assert
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)
        self.assertTrue(all(isinstance(k, str) and isinstance(v, tuple) for k, v in result.items()))
        self.fail()

    def test_code_column(self):
        # Arrange
        config = DatasetTablesConfig()

        # Act
        result = config.code_column

        # Assert
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)
        self.assertTrue(all(isinstance(k, str) and isinstance(v, str) for k, v in result.items()))

        self.fail()


class TestDatasetTables(unittest.TestCase):

    def test_tables_dict_property(self):
        tables = DatasetTables(static=None, admissions=None)
        self.assertEqual(tables.tables_dict, {})
        self.fail()

    def test_save(self):
        tables = DatasetTables(static=None, admissions=None)
        tables.save('test.h5', overwrite=True)
        loaded = DatasetTables.load('test.h5')
        self.assertEqual(loaded.static, None)
        self.assertEqual(loaded.admissions, None)
        self.fail()

    def test_load(self):
        tables = DatasetTables(static=None, admissions=None)
        tables.save('test.h5', overwrite=True)
        loaded = DatasetTables.load('test.h5')
        self.assertIsInstance(loaded, DatasetTables)

        self.fail()


class TestDataset(unittest.TestCase):

    def test_setup_core_pipeline(self):
        # Arrange
        config = None # DatasetConfig object
        dataset = Dataset(config, None)

        # Act
        pipeline = dataset._setup_core_pipeline(config)

        # Assert
        self.assertIsNotNone(pipeline)
        self.assertIsInstance(pipeline, DatasetPipeline)
        self.fail()

    def test_execute_pipeline(self):
        # Arrange
        dataset = Dataset(None, None)
        dataset.core_pipeline_report = pd.DataFrame()

        # Act
        updated_dataset = dataset.execute_pipeline()

        # Assert
        self.assertTrue(updated_dataset.config.pipeline_executed)
        self.assertIsInstance(updated_dataset.core_pipeline_report, pd.DataFrame)
        self.fail()

    def test_random_splits_default(self):
        # Arrange
        dataset = Dataset(None, None)
        dataset.subject_ids = [1, 2, 3]

        # Act
        splits = dataset.random_splits([0.6, 0.2, 0.2])

        # Assert
        self.assertEqual(len(splits), 3)
        self.assertAlmostEqual(len(splits[0]), 2)
        self.assertAlmostEqual(len(splits[1]), 1)
        self.assertAlmostEqual(len(splits[2]), 1)
        self.fail()

    def test_random_splits_custom_ids(self):
        # Arrange
        dataset = Dataset(None, None)
        subject_ids = [1, 2, 3, 4]

        # Act
        splits = dataset.random_splits([0.5, 0.5], subject_ids=subject_ids)

        # Assert
        self.assertEqual(len(splits), 2)
        self.assertAlmostEqual(len(splits[0]), 2)
        self.assertAlmostEqual(len(splits[1]), 2)

        self.fail()



class MockTable:
    pass



if __name__ == '__main__':
    unittest.main()
