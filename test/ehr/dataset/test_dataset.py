import itertools
from typing import Tuple, Dict, List, Callable
from unittest import mock

import equinox as eqx
import pandas as pd
import pytest
import tables as tb
from lib.ehr import CodingScheme
from lib.ehr.coding_scheme import CodeMapConfig, CodeMap, CodingSchemeConfig, FlatScheme
from lib.ehr.dataset import TableConfig, DatasetTablesConfig, DatasetTables, DatasetSchemeConfig, DatasetScheme, \
    Dataset, AbstractDatasetPipeline
from test.ehr.conftest import NaiveDataset


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
    assert TableConfig._alias_dict(all_dict) == all_alias_dict
    assert TableConfig._alias_id_dict(all_dict) == id_alias_dict
    assert set(TableConfig._time_cols(all_dict)) == set(time_dict.values())
    assert set(TableConfig._coded_cols(all_dict)) == set(coded_dict.values())


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

    def test_save_load(self, dataset_tables: DatasetTables, tmpdir):
        with tb.open_file(f'{tmpdir}/test_dataset_tables.h5', 'w') as hf5:
            dataset_tables.save(hf5.create_group('/', 'dataset_tables'))
        with tb.open_file(f'{tmpdir}/test_dataset_tables.h5', 'r') as hf5:
            loaded = DatasetTables.load(hf5.root['dataset_tables'])
        assert loaded.equals(dataset_tables)




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
            CodeMap.register_map(CodeMap(map_config, map_data))
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

    def test_demographic_size(self, dataset_scheme_config: DatasetSchemeConfig):
        pass
        # scheme = DatasetScheme(config=dataset_scheme_config)
        # assert scheme.demographic_size == 3


TEST_DATASET_SCOPE = 'function'


class TestDataset:
    @pytest.fixture
    def dataset_after_identity_pipeline(self, dataset: Dataset):
        return dataset.execute_pipeline()

    def test_execute_pipeline(self, dataset: Dataset, dataset_after_identity_pipeline: Dataset):
        assert isinstance(dataset.pipeline, AbstractDatasetPipeline)
        assert isinstance(dataset, Dataset)
        assert isinstance(dataset_after_identity_pipeline, Dataset)
        assert dataset.pipeline_report.equals(pd.DataFrame())

        # Because we use identity pipeline, the dataset tables should be the same
        # but the new dataset should have a different report (metadata).
        assert not dataset_after_identity_pipeline.equals(dataset)
        assert not dataset_after_identity_pipeline.pipeline_report.equals(dataset.pipeline_report)
        assert dataset_after_identity_pipeline.tables.equals(dataset.tables)
        assert len(dataset_after_identity_pipeline.pipeline_report) == 1
        assert dataset_after_identity_pipeline.pipeline_report.loc[0, 'transformation'] == 'identity'

        with mock.patch('logging.warning') as mocker:
            dataset3 = dataset_after_identity_pipeline.execute_pipeline()
            assert dataset3.equals(dataset_after_identity_pipeline)
            mocker.assert_called_once_with("Pipeline has already been executed. Doing nothing.")

    def test_subject_ids(self, dataset: NaiveDataset,
                         dataset_after_identity_pipeline: NaiveDataset,
                         indexed_dataset: NaiveDataset):
        with pytest.raises(AssertionError):
            dataset.subject_ids

        with pytest.raises(AssertionError):
            dataset_after_identity_pipeline.subject_ids

        assert set(indexed_dataset.subject_ids) == set(indexed_dataset.tables.static.index.unique())

    @pytest.mark.expensive_test
    @pytest.mark.parametrize('overwrite', [True, False])
    def test_save_load(self, dataset: NaiveDataset,
                       dataset_after_identity_pipeline: NaiveDataset,
                       tmpdir: str, overwrite: bool):
        dataset_after_identity_pipeline.save(f'{tmpdir}/test_dataset', overwrite=False)
        if overwrite:
            dataset_after_identity_pipeline.save(f'{tmpdir}/test_dataset', overwrite=True)

        with pytest.raises(RuntimeError):
            dataset_after_identity_pipeline.save(f'{tmpdir}/test_dataset', overwrite=False)

        loaded = NaiveDataset.load(f'{tmpdir}/test_dataset')
        assert loaded.equals(dataset_after_identity_pipeline)
        assert not loaded.equals(dataset)
        assert loaded.equals(dataset.execute_pipeline())

    @pytest.fixture(scope=TEST_DATASET_SCOPE)
    def subject_ids(self, indexed_dataset: NaiveDataset):
        return indexed_dataset.subject_ids

    @pytest.mark.parametrize('valid_split', [[1.0]])
    @pytest.mark.parametrize('valid_balance', ['subjects', 'admissions', 'admissions_intervals'])
    @pytest.mark.parametrize('invalid_splits', [[], [0.3, 0.8, 0.7, 0.9], [0.5, 0.2]])  # should be sorted.
    @pytest.mark.parametrize('invalid_balance', ['hi', 'unsupported'])
    def test_random_split_invalid_args(self, indexed_dataset: NaiveDataset, subject_ids: List[str],
                                       valid_split: List[float], valid_balance: str,
                                       invalid_splits: List[float], invalid_balance: str):
        if len(subject_ids) == 0:
            with pytest.raises(AssertionError):
                indexed_dataset.random_splits(valid_split, balance=valid_balance)
            return

        if len(indexed_dataset.tables.admissions) == 0 and 'admissions' in valid_balance:
            with pytest.raises(AssertionError):
                indexed_dataset.random_splits(valid_split, balance=valid_balance)
            return

        assert set(indexed_dataset.random_splits(valid_split, balance=valid_balance)[0]) == set(subject_ids)

        with pytest.raises(AssertionError):
            indexed_dataset.random_splits(valid_split, balance=invalid_balance)

        with pytest.raises(AssertionError):
            indexed_dataset.random_splits(invalid_splits, balance=valid_balance)

        with pytest.raises(AssertionError):
            indexed_dataset.random_splits(invalid_splits, balance=invalid_balance)

    @pytest.fixture(scope=TEST_DATASET_SCOPE)
    def split_measure(self, indexed_dataset: NaiveDataset, balance: str):
        return {
            'subjects': lambda x: len(x),
            'admissions': lambda x: sum(indexed_dataset.subjects_n_admissions.loc[x]),
            'admissions_intervals': lambda x: sum(indexed_dataset.subjects_intervals_sum.loc[x])
        }[balance]

    @pytest.fixture(scope=TEST_DATASET_SCOPE, params=['subjects', 'admissions', 'admissions_intervals'])
    def balance(self, request):
        return request.param

    @pytest.fixture(scope=TEST_DATASET_SCOPE, params=[[0.1], [0.4, 0.8, 0.9], [0.3, 0.5, 0.7, 0.9]])
    def split_quantiles(self, request):
        return request.param

    @pytest.fixture(scope=TEST_DATASET_SCOPE,
                    params=[1, 11, 111, 1111, 11111])
    def subject_splits(self, indexed_dataset: NaiveDataset, subject_ids: List[str], balance: str,
                       split_quantiles: List[float], request):
        random_seed = request.param
        if len(subject_ids) == 0 or (len(indexed_dataset.tables.admissions) == 0 and 'admissions' in balance):
            pytest.skip("No admissions in dataset or no subjects.")
        return indexed_dataset.random_splits(split_quantiles, balance=balance, random_seed=random_seed)

    def test_random_split(self, indexed_dataset: NaiveDataset, subject_ids: List[str],
                          subject_splits: List[List[str]],
                          split_quantiles: List[float]):
        assert set.union(*list(map(set, subject_splits))) == set(subject_ids)
        assert len(subject_splits) == len(split_quantiles) + 1
        # No overlaps.
        assert sum(len(v) for v in subject_splits) == len(indexed_dataset.subject_ids)
        assert set.union(*[set(v) for v in subject_splits]) == set(indexed_dataset.subject_ids)

    @pytest.fixture(scope=TEST_DATASET_SCOPE)
    def split_proportions(self, split_quantiles: List[float]):
        return [p1 - p0 for p0, p1 in zip([0] + split_quantiles, split_quantiles + [1])]

    def test_random_split_balance(self, subject_ids: List[str],
                                  subject_splits: List[List[str]],
                                  split_proportions: List[float],
                                  balance: str,
                                  split_measure: Callable[[List[str]], float]):
        if len(subject_ids) < 2:
            pytest.skip("No enough subjects in dataset to split.")
        # # test proportionality
        # NOTE: no specified behaviour when splits have equal proportions, so comparing argsorts
        # is not appropriate.
        p_threshold = 1 / len(subject_ids)
        tolerance = min(abs(split_measure([i]) - split_measure([j])) for i in subject_ids for j in subject_ids if i != j)
        for i in range(len(split_proportions)):
            for j in range(i + 1, len(split_proportions)):
                if abs(split_proportions[i] - split_proportions[j]) < p_threshold:
                    if balance == 'subjects':
                        # Difference between subjects is at most 1 when balance is applied
                        # on subjects count AND proportions are (almost) equal.
                        assert abs(len(subject_splits[i]) - len(subject_splits[j])) <= 1
                elif split_proportions[i] > split_proportions[j]:
                    assert (split_measure(subject_splits[i]) - split_measure(subject_splits[j])) >= -tolerance
                else:
                    assert (split_measure(subject_splits[i]) - split_measure(subject_splits[j])) <= tolerance
