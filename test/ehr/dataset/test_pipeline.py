import random
import string
from collections import defaultdict
from typing import List, Tuple, Set, Dict

import equinox as eqx
import numpy as np
import pandas as pd
import pytest

from lib.ehr import Dataset
from lib.ehr.pipeline import DatasetTransformation, SampleSubjects, CastTimestamps, \
    FilterUnsupportedCodes, SetAdmissionRelativeTimes, SetCodeIntegerIndices, SetIndex, ProcessOverlappingAdmissions, \
    FilterClampTimestampsToAdmissionInterval, FilterInvalidInputRatesSubjects, ICUInputRateUnitConversion, \
    ObsIQROutlierRemover, RandomSplits, FilterSubjectsNegativeAdmissionLengths
from test.ehr.dataset.conftest import IndexedDataset


@pytest.mark.parametrize('cls, params', [
    (SampleSubjects, {'n_subjects': 10, 'seed': 0, 'offset': 5}),
    (CastTimestamps, {}),
    (FilterUnsupportedCodes, {}),
    (SetAdmissionRelativeTimes, {}),
    (SetCodeIntegerIndices, {}),
    (SetIndex, {}),
    (ProcessOverlappingAdmissions, {'merge': False}),
    (FilterClampTimestampsToAdmissionInterval, {}),
    (FilterInvalidInputRatesSubjects, {}),
    (ICUInputRateUnitConversion, {'conversion_table': pd.DataFrame()}),
    (RandomSplits, {'splits': [0.5], 'splits_key': 'splits', 'seed': 0, 'balance': 'subjects',
                    'discount_first_admission': False}),
    (ObsIQROutlierRemover, {'fit_only': False, 'fitted_processor': 'x',
                            'splits_key': 'splits', 'training_split_index': 0,
                            'outlier_q1': 0.0, 'outlier_q2': 0.0,
                            'outlier_iqr_scale': 0.0, 'outlier_z1': 0.0,
                            'outlier_z2': 0.0, 'transformer_key': 'key'})])
def test_additional_parameters(cls, params):
    # Test that additional_parameters returns dict without name
    assert cls(name='test', **params).additional_parameters == params


@pytest.fixture
def indexed_dataset(dataset_config, dataset_tables):
    return IndexedDataset(config=dataset_config, tables=dataset_tables).execute_pipeline()


def test_synchronize_index_subjects(indexed_dataset: Dataset):
    if len(indexed_dataset.tables.admissions) == 0:
        pytest.skip("No admissions table found in dataset.")

    c_subject_id = indexed_dataset.config.tables.subject_id_alias
    sample_subject_id = indexed_dataset.tables.admissions[c_subject_id].iloc[0]
    static = indexed_dataset.tables.static
    static = static.drop(index=sample_subject_id)
    indexed_dataset = eqx.tree_at(lambda x: x.tables.static, indexed_dataset, static)

    synced_indexed_dataset, _ = DatasetTransformation.synchronize_index(indexed_dataset, 'static', c_subject_id, aux={},
                                                                        report=lambda *args, **kwargs: None)
    assert sample_subject_id in indexed_dataset.tables.admissions[c_subject_id].values
    assert not sample_subject_id in synced_indexed_dataset.tables.admissions[c_subject_id].values
    assert set(indexed_dataset.tables.static.index) == set(synced_indexed_dataset.tables.admissions[c_subject_id])


def test_synchronize_index_admissions(indexed_dataset: Dataset):
    if len(indexed_dataset.tables.admissions) == 0:
        pytest.skip("No admissions table found in dataset.")

    c_admission_id = indexed_dataset.config.tables.admission_id_alias
    sample_admission_id = indexed_dataset.tables.admissions.index[0]
    admissions = indexed_dataset.tables.admissions
    admissions = admissions.drop(index=sample_admission_id)
    indexed_dataset = eqx.tree_at(lambda x: x.tables.admissions, indexed_dataset, admissions)

    synced_indexed_dataset, _ = DatasetTransformation.synchronize_index(indexed_dataset, 'admissions', c_admission_id,
                                                                        aux={},
                                                                        report=lambda *args, **kwargs: None)
    for table_name, table in indexed_dataset.tables.tables_dict.items():
        if table is not None and c_admission_id in table.columns and len(table) > 0:
            assert sample_admission_id in table[c_admission_id].values
            synced_table = getattr(synced_indexed_dataset.tables, table_name)
            assert sample_admission_id not in synced_table[c_admission_id].values
            assert set(synced_table[c_admission_id]).issubset(set(synced_indexed_dataset.tables.admissions.index))


def test_filter_no_admissions_subjects(indexed_dataset: Dataset):
    if len(indexed_dataset.tables.admissions) == 0:
        pytest.skip("No admissions table found in dataset.")

    c_subject_id = indexed_dataset.config.tables.subject_id_alias
    sample_subject_id = indexed_dataset.tables.admissions[c_subject_id].iloc[0]
    assert sample_subject_id in indexed_dataset.tables.static.index
    admissions = indexed_dataset.tables.admissions
    admissions = admissions[admissions[c_subject_id] != sample_subject_id]
    indexed_dataset = eqx.tree_at(lambda x: x.tables.admissions, indexed_dataset, admissions)
    filtered_dataset, _ = DatasetTransformation.filter_no_admission_subjects(indexed_dataset, {},
                                                                             report=lambda *args, **kwargs: None)
    assert sample_subject_id not in filtered_dataset.tables.static


@pytest.mark.parametrize('seed', [0, 1])
def test_sample_subjects(indexed_dataset: Dataset, seed: int):
    if len(indexed_dataset.tables.static) <= 1:
        pytest.skip("Only one subject in dataset.")
    if len(indexed_dataset.tables.admissions) == 0:
        pytest.skip("No admissions table found in dataset. The sampling will result on empty dataset.")

    n_subjects = len(indexed_dataset.tables.static)
    n_sample = int(n_subjects / 5)

    for offset in range(n_subjects - n_sample):
        sampled_dataset, _ = SampleSubjects(n_subjects=n_sample, seed=seed, offset=offset)(indexed_dataset, {})
        sampled_subjects = sampled_dataset.subject_ids
        assert len(sampled_subjects) == n_sample
        assert len(set(sampled_subjects)) == n_sample
        assert set(sampled_subjects).issubset(set(indexed_dataset.subject_ids))


@pytest.fixture
def indexed_dataset_str_timestamps(indexed_dataset: Dataset):
    for table_name, time_cols in indexed_dataset.config.tables.time_cols.items():
        if len(time_cols) == 0:
            continue
        table = indexed_dataset.tables.tables_dict[table_name]
        for col in time_cols:
            table[col] = table[col].astype(str)
        indexed_dataset = eqx.tree_at(lambda x: getattr(x.tables, table_name), indexed_dataset, table)
    return indexed_dataset


def test_cast_timestamps(indexed_dataset_str_timestamps: Dataset):
    dataset, _ = CastTimestamps()(indexed_dataset_str_timestamps, {})

    for table_name, time_cols in dataset.config.tables.time_cols.items():
        if len(time_cols) == 0:
            continue
        table = dataset.tables.tables_dict[table_name]
        for col in time_cols:
            assert table[col].dtype == np.dtype('datetime64[ns]')


@pytest.fixture
def indexed_dataset_unsupported_codes(indexed_dataset: Dataset):
    if all(len(getattr(indexed_dataset.tables, k)) == 0 for k in indexed_dataset.config.tables.code_column.keys()):
        pytest.skip("No coded tables or they are all empty.")
    unsupported_codes = {}
    for table_name, code_col in indexed_dataset.config.tables.code_column.items():
        table = indexed_dataset.tables.tables_dict[table_name]
        unsupported_code = f'UNSUPPORTED_CODE_{"".join(random.choices(string.ascii_uppercase, k=5))}'
        table.loc[table.index[0], code_col] = unsupported_code
        unsupported_codes[table_name] = unsupported_code
        indexed_dataset = eqx.tree_at(lambda x: getattr(x.tables, table_name), indexed_dataset, table)
    return indexed_dataset, unsupported_codes


def test_filter_unsupported_codes(indexed_dataset_unsupported_codes: Tuple[Dataset, Dict[str, Set[str]]]):
    indexed_dataset, unsupported_codes = indexed_dataset_unsupported_codes
    filtered_dataset, _ = FilterUnsupportedCodes()(indexed_dataset, {})
    for table_name, code_col in filtered_dataset.config.tables.code_column.items():
        assert unsupported_codes[table_name] in getattr(indexed_dataset.tables, table_name)[code_col].values
        assert unsupported_codes[table_name] not in getattr(filtered_dataset.tables, table_name)[code_col].values


def test_set_relative_times(indexed_dataset: Dataset):
    if len(indexed_dataset.tables.admissions) == 0 or len(indexed_dataset.tables.obs) == 0:
        pytest.skip("No admissions table found in dataset.")
    dataset, _ = SetAdmissionRelativeTimes()(indexed_dataset, {})

    admissions = indexed_dataset.tables.admissions.copy()
    c_admittime = dataset.config.tables.admissions.admission_time_alias
    c_dischtime = dataset.config.tables.admissions.discharge_time_alias
    admissions['los_hours'] = (admissions[c_dischtime] - admissions[c_admittime]).dt.total_seconds() / (60 * 60)

    for table_name, time_cols in dataset.config.tables.time_cols.items():
        if table_name in ('admissions', 'static'):
            continue

        table = getattr(dataset.tables, table_name).copy()
        table = table.merge(admissions[['los_hours']],
                            left_on=dataset.config.tables.admissions.admission_id_alias,
                            right_index=True)

        for col in time_cols:
            assert table[col].dtype == float
            assert table[col].min() >= 0
            assert all(table[col] <= table['los_hours'])


@pytest.fixture
def indexed_dataset_first_negative_admission(indexed_dataset: Dataset):
    if len(indexed_dataset.tables.admissions) == 0:
        pytest.skip("No admissions table found in dataset.")
    admissions = indexed_dataset.tables.admissions.copy()
    first_admission_i = admissions.index[0]
    c_admittime = indexed_dataset.config.tables.admissions.admission_time_alias
    c_dischtime = indexed_dataset.config.tables.admissions.discharge_time_alias
    admittime = admissions.loc[first_admission_i, c_admittime]
    dischtime = admissions.loc[first_admission_i, c_dischtime]
    admissions.loc[first_admission_i, c_admittime] = dischtime
    admissions.loc[first_admission_i, c_dischtime] = admittime
    return eqx.tree_at(lambda x: x.tables.admissions, indexed_dataset, admissions)


def test_filter_subjects_negative_admission_length(indexed_dataset_first_negative_admission: Dataset):
    dataset0 = indexed_dataset_first_negative_admission
    admissions0 = dataset0.tables.admissions

    dataset1, _ = FilterSubjectsNegativeAdmissionLengths()(dataset0, {})
    admissions1 = dataset1.tables.admissions
    c_admittime = dataset1.config.tables.admissions.admission_time_alias
    c_dischtime = dataset1.config.tables.admissions.discharge_time_alias

    assert admissions0.shape[0] == admissions1.shape[0] + 1
    assert admissions0.loc[admissions0.index[0], c_admittime] > admissions0.loc[admissions0.index[0], c_dischtime]
    assert admissions0.index[0] not in admissions1.index
    # Also assert synchronization.


def set_code_integer_indices(indexed_dataset: Dataset):
    dataset, _ = SetCodeIntegerIndices()(indexed_dataset, {})
    for table_name, code_col in dataset.config.tables.code_column.items():
        table = getattr(dataset.tables, table_name)
        scheme = getattr(dataset.scheme, table_name).code_scheme
        assert table[code_col].dtype == int
        assert all(table[code_col].isin(scheme.codes.index.values()))


def generate_admissions_from_pattern(pattern: List[str]) -> pd.DataFrame:
    if len(pattern) == 0:
        return pd.DataFrame(columns=['admittime', 'dischtime'])
    random_monotonic_positive_integers = np.random.randint(1, 30, size=len(pattern)).cumsum()
    sequence_dates = list(
        map(lambda x: pd.Timestamp.today().normalize() + pd.Timedelta(days=x), random_monotonic_positive_integers))
    event_times = dict(zip(pattern, sequence_dates))
    admittimes = {k: v for k, v in event_times.items() if k.startswith('A')}
    dischtimes = {k.replace('D', 'A'): v for k, v in event_times.items() if k.startswith('D')}
    admissions = pd.DataFrame(index=admittimes.keys())
    admissions['admittime'] = admissions.index.map(admittimes)
    admissions['dischtime'] = admissions.index.map(dischtimes)
    # shuffle the rows shuffled.
    return admissions.sample(frac=1)


@pytest.mark.parametrize('admission_pattern, expected_out', [
    # Below are a list of tuples of the form (admission_pattern, expected_super_sub).
    # The admission pattern is a sequence of admission/discharge.
    # In the database it is possible to have overlapping admissions, so we merge them.
    # A1 and D1 represent the admission and discharge time of a particular admission record in the database.
    # A2 and D2 means the same and that A2 happened after A1 (and not necessarily after D1).
    (['A1', 'A2', 'A3', 'D1', 'D3', 'D2'], {'A1': ['A2', 'A3']}),
    (['A1', 'A2', 'A3', 'D1', 'D2', 'D3'], {'A1': ['A2', 'A3']}),
    (['A1', 'A2', 'A3', 'D2', 'D1', 'D3'], {'A1': ['A2', 'A3']}),
    (['A1', 'A2', 'A3', 'D2', 'D3', 'D1'], {'A1': ['A2', 'A3']}),
    (['A1', 'A2', 'A3', 'D3', 'D1', 'D2'], {'A1': ['A2', 'A3']}),
    (['A1', 'A2', 'A3', 'D3', 'D2', 'D1'], {'A1': ['A2', 'A3']}),
    ##
    (['A1', 'A2', 'D1', 'A3', 'D3', 'D2'], {'A1': ['A2', 'A3']}),
    (['A1', 'A2', 'D1', 'A3', 'D2', 'D3'], {'A1': ['A2', 'A3']}),
    ##
    (['A1', 'A2', 'D1', 'D2', 'A3', 'D3'], {'A1': ['A2']}),
    (['A1', 'A2', 'D1', 'D2'], {'A1': ['A2']}),
    ##
    (['A1', 'A2', 'D2', 'A3', 'D1', 'D3'], {'A1': ['A2', 'A3']}),
    (['A1', 'A2', 'D2', 'A3', 'D3', 'D1'], {'A1': ['A2', 'A3']}),
    (['A1', 'A2', 'D2', 'D1', 'A3', 'D3'], {'A1': ['A2']}),
    ##
    (['A1', 'D1', 'A2', 'A3', 'D2', 'D3'], {'A2': ['A3']}),
    (['A1', 'D1', 'A2', 'A3', 'D3', 'D2'], {'A2': ['A3']}),
    (['A1', 'D1', 'A2', 'D2', 'A3', 'D3'], {}),
    ##
    (['A1', 'D1'], {}),
    ([], {}),
    (['A1', 'D1', 'A2', 'A3', 'D3', 'A4', 'D2', 'D4'], {'A2': ['A3', 'A4']}),
    (['A1', 'A2', 'D2', 'D1', 'A3', 'A4', 'D3', 'D4'], {'A1': ['A2'], 'A3': ['A4']}),
])
def test_overlapping_cases(admission_pattern, expected_out):
    admissions = generate_admissions_from_pattern(admission_pattern)
    sub2sup = ProcessOverlappingAdmissions._collect_overlaps(admissions, 'admittime', 'dischtime')
    sup2sub = defaultdict(list)
    for sub, sup in sub2sup.items():
        sup2sub[sup].append(sub)
    assert sup2sub == expected_out


@pytest.fixture
def admission_ids_map(indexed_dataset: Dataset):
    if len(indexed_dataset.tables.admissions) < 10:
        pytest.skip("Not enough admissions for the test in dataset.")
    index = indexed_dataset.tables.admissions.index
    return {
        index[1]: index[0],
        index[2]: index[0],
        index[3]: index[0],
        index[5]: index[4],
        index[6]: index[4],
    }


def test_map_admission_ids(indexed_dataset: Dataset, admission_ids_map: Dict[str, str]):
    # Assert no data loss from records.
    # Assert children admissions are mapped in records and removed from admissions.
    dataset, _ = ProcessOverlappingAdmissions._merge_overlapping_admissions(indexed_dataset, {},
                                                                            admission_ids_map,
                                                                            lambda *args, **kwargs: None)
    admissions0 = indexed_dataset.tables.admissions
    admissions1 = dataset.tables.admissions

    assert len(admissions0) == len(admissions1) + len(admission_ids_map)
    assert set(admissions1.index).issubset(set(admissions0.index))

    c_admission_id = dataset.config.tables.admissions.admission_id_alias
    for table_name, table in dataset.tables.tables_dict.items():
        table0 = getattr(indexed_dataset.tables, table_name)
        if c_admission_id in table.columns:
            assert len(table) == len(table0)
            assert set(table[c_admission_id]) - set(admissions1.index.values) == set()
            assert set(table0[c_admission_id]) - set(admissions0.index.values) == set()
            assert set(table[c_admission_id]) - set(table0[c_admission_id]) == set()


def test_merge_overlapping_admissions(indexed_dataset: Dataset):
    if len(indexed_dataset.tables.admissions) == 0:
        pytest.skip("No admissions in dataset.")
    admissions = indexed_dataset.tables.admissions
    c_subject_id = indexed_dataset.config.tables.admissions.subject_id_alias
    c_admittime = indexed_dataset.config.tables.admissions.admission_time_alias
    c_dischtime = indexed_dataset.config.tables.admissions.discharge_time_alias

    sub2sup = {adm_id: super_adm_id for _, subject_adms in admissions.groupby(c_subject_id)
               for adm_id, super_adm_id in ProcessOverlappingAdmissions._collect_overlaps(subject_adms,
                                                                                          c_admittime,
                                                                                          c_dischtime).items()}

    if len(sub2sup) == 0:
        pytest.skip("No overlapping admissions in dataset.")

    merged_dataset, _ = ProcessOverlappingAdmissions(merge=True)(indexed_dataset, {})
    filtered_dataset, _ = ProcessOverlappingAdmissions(merge=False)(indexed_dataset, {})
    admissions0 = indexed_dataset.tables.admissions
    admissions_m = merged_dataset.tables.admissions
    admissions_f = filtered_dataset.tables.admissions

    assert len(admissions0) == len(admissions_m) + len(sub2sup)
    assert len(admissions_m) > len(admissions_f)

    c_admission_id = indexed_dataset.config.tables.admissions.admission_id_alias
    for table_name, table0 in indexed_dataset.tables.tables_dict.items():
        table_m = getattr(merged_dataset.tables, table_name)
        table_f = getattr(filtered_dataset.tables, table_name)

        if c_admission_id in table0.columns and len(table0) > 0:
            assert len(table_m) == len(table0)
            assert set(table_m[c_admission_id]) - set(admissions_m.index.values) == set()
            assert set(table0[c_admission_id]) - set(admissions0.index.values) == set()
            assert set(table_m[c_admission_id]) - set(table0[c_admission_id]) == set()
            assert len(table_f) <= len(table0)
            assert set(table_f[c_admission_id]).intersection(set(sub2sup.keys()) | set(sub2sup.values())) == set()


@pytest.fixture
def shifted_timestamps_dataset(indexed_dataset: Dataset):
    if any(len(getattr(indexed_dataset.tables, k)) == 0 for k in indexed_dataset.config.tables.time_cols.keys()):
        pytest.skip("No temporal data in dataset.")

    admissions = indexed_dataset.tables.admissions

    c_admission_id = indexed_dataset.config.tables.admissions.admission_id_alias
    c_admittime = indexed_dataset.config.tables.admissions.admission_time_alias
    c_dischtime = indexed_dataset.config.tables.admissions.discharge_time_alias
    admission_id = admissions.index[0]
    admittime = admissions.loc[admission_id, c_admittime]
    dischtime = admissions.loc[admission_id, c_dischtime]
    if 'obs' in indexed_dataset.tables.tables_dict:
        c_time = indexed_dataset.config.tables.obs.time_alias

        obs = indexed_dataset.tables.obs.copy()
        admission_obs = obs[obs[c_admission_id] == admission_id]
        if len(admission_obs) > 0:
            obs.loc[admission_obs.index[0], c_time] = dischtime + pd.Timedelta(days=1)
        if len(admission_obs) > 1:
            obs.loc[admission_obs.index[1], c_time] = admittime + pd.Timedelta(days=-1)
        indexed_dataset = eqx.tree_at(lambda x: x.tables.obs, indexed_dataset, obs)
    for k in ('hosp_procedures', 'icu_procedures', 'icu_inputs'):
        if k in indexed_dataset.tables.tables_dict:
            c_starttime = getattr(indexed_dataset.config.tables, k).start_time_alias
            c_endtime = getattr(indexed_dataset.config.tables, k).end_time_alias
            procedures = getattr(indexed_dataset.tables, k).copy()

            admission_procedures = procedures[procedures[c_admission_id] == admission_id]
            if len(admission_procedures) > 0:
                procedures.loc[admission_procedures.index[0], c_starttime] = admittime + pd.Timedelta(days=-1)
                procedures.loc[admission_procedures.index[0], c_endtime] = dischtime + pd.Timedelta(days=1)

            if len(admission_procedures) > 1:
                procedures.loc[admission_procedures.index[1], c_starttime] = dischtime + pd.Timedelta(days=1)
                procedures.loc[admission_procedures.index[1], c_endtime] = dischtime + pd.Timedelta(days=2)

            if len(admission_procedures) > 2:
                procedures.loc[admission_procedures.index[2], c_starttime] = admittime + pd.Timedelta(days=-2)
                procedures.loc[admission_procedures.index[2], c_endtime] = admittime + pd.Timedelta(days=-1)

            indexed_dataset = eqx.tree_at(lambda x: getattr(x.tables, k), indexed_dataset, procedures)

    return indexed_dataset


def test_clamp_timestamps_to_admission_interval(shifted_timestamps_dataset: Dataset):
    fixed_dataset, _ = FilterClampTimestampsToAdmissionInterval()(shifted_timestamps_dataset, {})
    admissions = shifted_timestamps_dataset.tables.admissions

    c_admission_id = shifted_timestamps_dataset.config.tables.admissions.admission_id_alias
    c_admittime = shifted_timestamps_dataset.config.tables.admissions.admission_time_alias
    c_dischtime = shifted_timestamps_dataset.config.tables.admissions.discharge_time_alias
    admission_id = admissions.index[0]
    admittime = admissions.loc[admission_id, c_admittime]
    dischtime = admissions.loc[admission_id, c_dischtime]

    if 'obs' in shifted_timestamps_dataset.tables.tables_dict:
        c_time = shifted_timestamps_dataset.config.tables.obs.time_alias

        obs0 = shifted_timestamps_dataset.tables.obs
        obs1 = fixed_dataset.tables.obs

        admission_obs0 = obs0[obs0[c_admission_id] == admission_id]
        admission_obs1 = obs1[obs1[c_admission_id] == admission_id]
        if len(admission_obs0) > 0:
            assert len(admission_obs0) > len(admission_obs1)
            assert not admission_obs0[c_time].between(admittime, dischtime).all()
            assert admission_obs1[c_time].between(admittime, dischtime).all()

    for k in ('hosp_procedures', 'icu_procedures', 'icu_inputs'):
        if k in shifted_timestamps_dataset.tables.tables_dict:
            c_starttime = getattr(shifted_timestamps_dataset.config.tables, k).start_time_alias
            c_endtime = getattr(shifted_timestamps_dataset.config.tables, k).end_time_alias
            procedures0 = getattr(shifted_timestamps_dataset.tables, k)
            procedures1 = getattr(fixed_dataset.tables, k)

            admission_procedures0 = procedures0[procedures0[c_admission_id] == admission_id]
            admission_procedures1 = procedures1[procedures1[c_admission_id] == admission_id]
            if len(admission_procedures0) > 0:
                assert len(admission_procedures0) >= len(admission_procedures1)
                assert admission_procedures1[c_starttime].between(admittime, dischtime).all()
                assert admission_procedures1[c_endtime].between(admittime, dischtime).all()

            if len(admission_procedures0) > 1:
                assert len(admission_procedures0) > len(admission_procedures1)


def test_filter_invalid_input_rates_subjects(indexed_dataset: Dataset):
    assert False


def test_icu_input_rate_unit_conversion(indexed_dataset: Dataset):
    assert False


def test_random_splits(indexed_dataset: Dataset):
    assert False


def test_obs_iqr_outlier_remover(indexed_dataset: Dataset):
    assert False


def test_obs_zscore_scaler(indexed_dataset: Dataset):
    assert False


def test_obs_minmax_scaler(indexed_dataset: Dataset):
    assert False


def test_obs_adaptive_scaler(indexed_dataset: Dataset):
    assert False
