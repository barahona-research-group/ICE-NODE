import os
from typing import Tuple

import equinox as eqx
import pytest

from lib.ehr.dataset import TableConfig, DatasetTablesConfig, DatasetTables


@pytest.fixture(params=[('x_id_alias', 'y_id_alias'), tuple()])
def id_alias_attrs(request) -> Tuple[str, ...]:
    return request.param


@pytest.fixture(params=[('a_alias', 'b_alias'), tuple()])
def alias_attrs(request) -> Tuple[str, ...]:
    return request.param


@pytest.fixture(params=[('timex_alias', 'datex_alias'), tuple()])
def time_attrs(request) -> Tuple[str, ...]:
    return request.param


@pytest.fixture(params=[('alpha_code_alias', 'beta_code_alias'), tuple()])
def coded_attrs(request) -> Tuple[str, ...]:
    return request.param


@pytest.fixture(params=[('derived_lambda', 'derived_sigma'), tuple()])
def other_attrs(request) -> Tuple[str, ...]:
    return request.param


def test_table_config(id_alias_attrs: Tuple[str, ...], alias_attrs: Tuple[str, ...],
                      time_attrs: Tuple[str, ...], coded_attrs: Tuple[str, ...],
                      other_attrs: Tuple[str, ...]):
    id_alias_dict = {k: '_' for k in id_alias_attrs}
    alias_dict = {k: '_' for k in alias_attrs}
    time_dict = {k: f'{k}_t' for k in time_attrs}
    coded_dict = {k: f'{k}_code' for k in coded_attrs}
    other_dict = {k: '_' for k in other_attrs}

    all_alias_dict = id_alias_dict | alias_dict | time_dict | coded_dict
    all_dict = all_alias_dict | other_dict
    config = TableConfig()
    # TODO: Is there a better way to do this?
    config.__dict__.update(all_dict)

    assert config.alias_dict == all_alias_dict
    assert config.alias_id_dict == id_alias_dict
    assert set(config.time_cols) == set(time_dict.values())
    assert set(config.coded_cols) == set(coded_dict.values())


class TestDatasetTablesConfig:

    def test_assert_consistent_aliases(self, dataset_tables_config: DatasetTablesConfig):
        dataset_tables_config._assert_consistent_aliases()

    def test_assert_consistent_subject_alias_fail(self, dataset_tables_config: DatasetTablesConfig):
        with pytest.raises(AssertionError):
            updated = eqx.tree_at(lambda x: x.admissions.subject_id_alias,
                                  dataset_tables_config,
                                  f'{dataset_tables_config.subject_id_alias}_')
            updated._assert_consistent_aliases()

    @pytest.mark.parametrize("table_name", ['dx_discharge', 'obs', 'icu_procedures', 'icu_inputs', 'hosp_procedures'])
    def test_assert_consistent_admission_alias_fail(self, dataset_tables_config: DatasetTablesConfig,
                                                    table_name: str):
        if getattr(dataset_tables_config, table_name) is None:
            pytest.skip(f'{table_name} is not in the dataset')

        with pytest.raises(AssertionError):
            updated = eqx.tree_at(lambda x: getattr(getattr(x, table_name), 'admission_id_alias'),
                                  dataset_tables_config,
                                  f'{dataset_tables_config.admission_id_alias}_')
            updated._assert_consistent_aliases()

    def test_assert_consistent_subject_alias_pass(self, dataset_tables_config: DatasetTablesConfig):
        all_tables_keys = ('static', 'admissions', 'dx_discharge', 'obs',
                           'icu_procedures', 'icu_inputs', 'hosp_procedures')
        timestamped_tables = ('obs',)
        interval_tables = ('icu_procedures', 'icu_inputs', 'hosp_procedures')

        assert set(dataset_tables_config.table_config_dict.keys()) == set(k for k in all_tables_keys
                                                                          if
                                                                          getattr(dataset_tables_config, k) is not None)
        assert set(dataset_tables_config.timestamped_table_config_dict.keys()) == set(k for k in timestamped_tables
                                                                                      if getattr(dataset_tables_config,
                                                                                                 k) is not None)
        assert set(dataset_tables_config.interval_based_table_config_dict.keys()) == set(k for k in interval_tables
                                                                                         if
                                                                                         getattr(dataset_tables_config,
                                                                                                 k) is not None)

        assert set(dataset_tables_config.indices.keys()) == {'static', 'admissions'}


class TestDatasetTables:

    def test_tables_dict_property(self, dataset_tables: DatasetTables):
        all_tables_keys = ('static', 'admissions', 'dx_discharge', 'obs',
                           'icu_procedures', 'icu_inputs', 'hosp_procedures')

        assert set(dataset_tables.tables_dict.keys()) == set(k for k in all_tables_keys
                                                             if getattr(dataset_tables, k) is not None)

    @pytest.mark.expensive_test
    def test_save_load(self, dataset_tables: DatasetTables, tmpdir: str):
        dataset_tables.save(f'{tmpdir}/test.h5', overwrite=False)
        loaded = DatasetTables.load(f'{tmpdir}/test.h5')

        assert loaded.equals(dataset_tables)

    @pytest.mark.expensive_test
    def test_load_overwrite(self, dataset_tables: DatasetTables, tmpdir: str):
        dataset_tables.save(f'{tmpdir}/test.h5', overwrite=False)
        dataset_tables.save(f'{tmpdir}/test.h5', overwrite=True)
        with pytest.raises(RuntimeError):
            dataset_tables.save(f'{tmpdir}/test.h5', overwrite=False)

#
# class TestDataset(unittest.TestCase):
#
#     def test_setup_core_pipeline(self):
#         # Arrange
#         config = None # DatasetConfig object
#         dataset = Dataset(config, None)
#
#         # Act
#         pipeline = dataset._setup_core_pipeline(config)
#
#         # Assert
#         self.assertIsNotNone(pipeline)
#         self.assertIsInstance(pipeline, DatasetPipeline)
#         self.fail()
#
#     def test_execute_pipeline(self):
#         # Arrange
#         dataset = Dataset(None, None)
#         dataset.core_pipeline_report = pd.DataFrame()
#
#         # Act
#         updated_dataset = dataset.execute_pipeline()
#
#         # Assert
#         self.assertTrue(updated_dataset.config.pipeline_executed)
#         self.assertIsInstance(updated_dataset.core_pipeline_report, pd.DataFrame)
#         self.fail()
#
#     def test_random_splits_default(self):
#         # Arrange
#         dataset = Dataset(None, None)
#         dataset.subject_ids = [1, 2, 3]
#
#         # Act
#         splits = dataset.random_splits([0.6, 0.2, 0.2])
#
#         # Assert
#         self.assertEqual(len(splits), 3)
#         self.assertAlmostEqual(len(splits[0]), 2)
#         self.assertAlmostEqual(len(splits[1]), 1)
#         self.assertAlmostEqual(len(splits[2]), 1)
#         self.fail()
#
#     def test_random_splits_custom_ids(self):
#         # Arrange
#         dataset = Dataset(None, None)
#         subject_ids = [1, 2, 3, 4]
#
#         # Act
#         splits = dataset.random_splits([0.5, 0.5], subject_ids=subject_ids)
#
#         # Assert
#         self.assertEqual(len(splits), 2)
#         self.assertAlmostEqual(len(splits[0]), 2)
#         self.assertAlmostEqual(len(splits[1]), 2)
#
#         self.fail()
#
#
#
# class MockTable:
#     pass
#
#
#
# if __name__ == '__main__':
#     unittest.main()
