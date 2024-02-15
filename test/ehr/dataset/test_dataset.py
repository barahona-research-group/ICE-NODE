import itertools
from typing import Tuple, Dict, List
from unittest import mock

import equinox as eqx
import pandas as pd
import pytest

from lib.ehr import CodingScheme, DatasetConfig
from lib.ehr.coding_scheme import CodeMapConfig, CodeMap, CodingSchemeConfig, FlatScheme
from lib.ehr.dataset import TableConfig, DatasetTablesConfig, DatasetTables, DatasetSchemeConfig, DatasetScheme, \
    Dataset, AbstractDatasetPipeline
from lib.ehr.pipeline import SetIndex
from test.ehr.dataset.conftest import NaiveDataset, Pipeline, IndexedDataset


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


@pytest.fixture(params=[{'dx_discharge': 2, 'obs': 2, 'icu_procedures': 1, 'icu_inputs': 1, 'hosp_procedures': 2}, {}])
def dataset_scheme_targets(dataset_scheme_config, request) -> Dict[str, Tuple[str]]:
    """
    Make different schemes where the input schemes can be mapped to.
    """

    targets = {}
    for space, n_targets in request.param.items():
        source_scheme = CodingScheme.from_name(getattr(dataset_scheme_config, space))
        target_schemes = []
        for i in range(n_targets):
            target_name = f'{source_scheme.name}_target_{i}'
            target_codes = [f'{c}_target_{i}' for c in source_scheme.codes]
            target_desc = dict(zip(target_codes, target_codes))

            map_config = CodeMapConfig(source_scheme=source_scheme.name,
                                       target_scheme=target_name,
                                       mapped_to_dag_space=False)
            map_data = {c: {c_target} for c, c_target in zip(source_scheme.codes, target_codes)}
            CodingScheme.register_scheme(FlatScheme(config=CodingSchemeConfig(name=target_name),
                                                    codes=target_codes,
                                                    desc=target_desc))
            CodeMap.register_map(source_scheme.name,
                                 target_name,
                                 CodeMap(map_config, map_data))
            target_schemes.append(target_name)
        targets[space] = tuple(target_schemes)
    return targets


def traverse_all_targets(dataset_scheme_target: Dict[str, Tuple[str]]):
    default_target_config = {}
    for space, targets in dataset_scheme_target.items():
        if len(targets) == 0:
            continue
        elif len(targets) == 1:
            default_target_config[space] = targets[0]

    comb_space_names = list(sorted(k for k, v in dataset_scheme_target.items() if len(v) > 1))
    comb_space = [dataset_scheme_target[name] for name in comb_space_names]
    target_combinations = []
    for comb in itertools.product(*comb_space):
        target_conf = default_target_config.copy()
        for i, space in enumerate(comb_space_names):
            target_conf[space] = comb[i]
        target_combinations.append(target_conf)
    return target_combinations


@pytest.fixture
def cleanup_dataset_scheme_targets(dataset_scheme_config, dataset_scheme_targets) -> None:
    yield  # The following code is run after the test.
    # Clean up
    for space, target_names in dataset_scheme_targets.items():
        source_scheme = CodingScheme.from_name(getattr(dataset_scheme_config, space))
        for target_name in target_names:
            CodingScheme.deregister_scheme(target_name)
            CodeMap.deregister_map(source_scheme.name, target_name)


class TestDatasetScheme:

    def test_scheme_dict(self, dataset_scheme_config: DatasetSchemeConfig):
        scheme = DatasetScheme(config=dataset_scheme_config)

        for space, scheme_name in dataset_scheme_config.as_dict().items():
            assert hasattr(scheme, space)
            if scheme_name is not None:
                assert isinstance(getattr(scheme, space), CodingScheme)

    @pytest.mark.usefixtures('cleanup_dataset_scheme_targets')
    def test_supported_target_schemes_options(self, dataset_scheme_config: DatasetSchemeConfig,
                                              dataset_scheme_targets: Dict[str, Tuple[str]]):
        scheme = DatasetScheme(config=dataset_scheme_config)
        supported_target_schemes = scheme.supported_target_scheme_options

        for space, targets in supported_target_schemes.items():
            support_set = set(dataset_scheme_targets.get(space, set()))
            if getattr(scheme, space) is not None:
                support_set.add(getattr(scheme, space).name)

            assert set(targets) == support_set

    @pytest.mark.usefixtures('cleanup_dataset_scheme_targets')
    def test_make_target_scheme(self, dataset_scheme_config: DatasetSchemeConfig,
                                dataset_scheme_targets: Dict[str, str]):
        for target_config in traverse_all_targets(dataset_scheme_targets):
            scheme = DatasetScheme(config=dataset_scheme_config)
            target_scheme = scheme.make_target_scheme(**target_config)
            assert isinstance(target_scheme, DatasetScheme)
            for space, target_name in target_config.items():
                assert isinstance(getattr(target_scheme, space), CodingScheme)
                assert getattr(target_scheme, space).name == target_name

            if scheme.dx_discharge is not None and target_scheme.dx_discharge is not None:
                assert isinstance(scheme.dx_mapper(target_scheme), CodeMap)
            if scheme.ethnicity is not None and target_scheme.ethnicity is not None:
                assert isinstance(scheme.ethnicity_mapper(target_scheme), CodeMap)

    def test_demographic_size(self, dataset_scheme_config: DatasetSchemeConfig):
        pass
        # scheme = DatasetScheme(config=dataset_scheme_config)
        # assert scheme.demographic_size == 3





class TestDataset:

    def test_execute_pipeline(self, dataset: Dataset):
        assert isinstance(dataset.core_pipeline, AbstractDatasetPipeline)
        assert dataset.core_pipeline_report.equals(pd.DataFrame())
        assert dataset.config.pipeline_executed is False

        dataset2 = dataset.execute_pipeline()
        # Because we use identity pipeline, the dataset tables should be the same
        # but the new dataset should have a different report (metadata).
        assert not dataset2.equals(dataset) and dataset2.tables.equals(dataset.tables)
        assert dataset2.core_pipeline_report.equals(pd.DataFrame({'action': ['*']}))
        assert dataset2.config.pipeline_executed is True

        with mock.patch('logging.warning') as mocker:
            dataset3 = dataset2.execute_pipeline()
            assert dataset3.equals(dataset2)
            mocker.assert_called_once_with("Pipeline has already been executed. Doing nothing.")

    def test_subject_ids(self, dataset_config: DatasetConfig, dataset_tables: DatasetTables):
        naive_dataset = NaiveDataset(config=dataset_config, tables=dataset_tables)
        indexed_dataset = IndexedDataset(config=dataset_config, tables=dataset_tables)

        with pytest.raises(AssertionError):
            naive_dataset.subject_ids
        with pytest.raises(AssertionError):
            indexed_dataset.subject_ids

        naive_dataset = naive_dataset.execute_pipeline()
        indexed_dataset = indexed_dataset.execute_pipeline()
        with pytest.raises(AssertionError):
            naive_dataset.subject_ids

        assert set(indexed_dataset.subject_ids) == set(indexed_dataset.tables.static.index.unique())

    @pytest.mark.expensive_test
    @pytest.mark.parametrize('overwrite', [True, False])
    @pytest.mark.parametrize('execute_pipeline', [True, False])
    def test_save_load(self, dataset: NaiveDataset, tmpdir: str, execute_pipeline: bool, overwrite: bool):
        raw_dataset = dataset
        if execute_pipeline:
            dataset = dataset.execute_pipeline()

        dataset.save(f'{tmpdir}/test_dataset', overwrite=False)
        if overwrite:
            dataset.save(f'{tmpdir}/test_dataset', overwrite=True)

        with pytest.raises(RuntimeError):
            dataset.save(f'{tmpdir}/test_dataset', overwrite=False)

        loaded = type(dataset).load(f'{tmpdir}/test_dataset')
        assert loaded.equals(dataset)
        if execute_pipeline:
            assert not loaded.equals(raw_dataset)
            assert loaded.equals(raw_dataset.execute_pipeline())

    @pytest.mark.parametrize('splits', [[0.1], [0.4, 0.8, 0.9], [0.3, 0.5, 0.7, 0.9], [0.5, 0.2]])
    @pytest.mark.parametrize('balance', ['subjects', 'admissions', 'admissions_intervals', 'unsupported'])
    def test_random_split(self, indexed_dataset: IndexedDataset, splits: List[float], balance: str):
        dataset = indexed_dataset.execute_pipeline()
        subjects = dataset.subject_ids
        skip = False
        if balance not in ('subjects', 'admissions', 'admissions_intervals'):
            with pytest.raises(AssertionError):
                dataset.random_splits(splits, balance=balance)
            skip = True
        if sorted(splits) != splits:
            with pytest.raises(AssertionError):
                dataset.random_splits(splits, balance=balance)
            skip = True
        if len(subjects) == 0:
            with pytest.raises(AssertionError):
                dataset.random_splits(splits, balance=balance)
            skip = True
        if balance in ('admissions', 'admissions_intervals') and len(dataset.tables.admissions) == 0:
            with pytest.raises(AssertionError):
                dataset.random_splits(splits, balance=balance)
            skip = True
        if skip:
            return

        subject_splits = dataset.random_splits(splits, balance=balance)
        assert set.union(*list(map(set, subject_splits))) == set(subjects)
