import random
import string
from collections import defaultdict
from typing import List, Tuple, Set, Dict, Type

import equinox as eqx
import numpy as np
import pandas as pd
import pytest

from lib.ehr import Dataset
from lib.ehr.pipeline import DatasetTransformation, SampleSubjects, CastTimestamps, \
    FilterUnsupportedCodes, SetAdmissionRelativeTimes, SetCodeIntegerIndices, SetIndex, ProcessOverlappingAdmissions, \
    FilterClampTimestampsToAdmissionInterval, FilterInvalidInputRatesSubjects, ICUInputRateUnitConversion, \
    ObsIQROutlierRemover, RandomSplits, FilterSubjectsNegativeAdmissionLengths, CodedValueScaler, ObsAdaptiveScaler, \
    InputScaler, TrainableTransformation, DatasetPipeline


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
    (ObsIQROutlierRemover, {'fit_only': False,
                            'splits_key': 'splits', 'training_split_index': 0,
                            'outlier_q1': 0.0, 'outlier_q2': 0.0,
                            'outlier_iqr_scale': 0.0, 'outlier_z1': 0.0,
                            'outlier_z2': 0.0, 'transformer_key': 'key'})])
def test_additional_parameters(cls, params):
    # Test that additional_parameters returns dict without name
    assert cls(name='test', **params).additional_parameters == params


def test_synchronize_index_subjects(indexed_dataset: Dataset):
    if len(indexed_dataset.tables.admissions) == 0:
        pytest.skip("No admissions table found in dataset.")
    indexed_dataset = indexed_dataset.execute_pipeline()

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
    indexed_dataset = indexed_dataset.execute_pipeline()

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
    indexed_dataset = indexed_dataset.execute_pipeline()

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
    indexed_dataset = indexed_dataset.execute_pipeline()

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
    indexed_dataset = indexed_dataset.execute_pipeline()

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
    indexed_dataset = indexed_dataset.execute_pipeline()

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
    indexed_dataset = indexed_dataset.execute_pipeline()

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
    indexed_dataset = indexed_dataset.execute_pipeline()

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
    static0 = dataset0.tables.static

    dataset1, _ = FilterSubjectsNegativeAdmissionLengths()(dataset0, {})
    admissions1 = dataset1.tables.admissions
    static1 = dataset1.tables.static
    c_admittime = dataset1.config.tables.admissions.admission_time_alias
    c_dischtime = dataset1.config.tables.admissions.discharge_time_alias

    assert admissions0.shape[0] > admissions1.shape[0]
    assert static0.shape[0] == static1.shape[0] + 1

    assert admissions0.loc[admissions0.index[0], c_admittime] > admissions0.loc[admissions0.index[0], c_dischtime]
    assert any(admissions0[c_admittime] > admissions0[c_dischtime])
    assert all(admissions1[c_admittime] <= admissions1[c_dischtime])
    assert admissions0.index[0] not in admissions1.index


def test_set_code_integer_indices(indexed_dataset: Dataset):
    indexed_dataset = indexed_dataset.execute_pipeline()

    dataset, _ = SetCodeIntegerIndices()(indexed_dataset, {})
    for table_name, code_col in dataset.config.tables.code_column.items():
        table = getattr(dataset.tables, table_name)
        scheme = getattr(dataset.scheme, table_name)
        assert table[code_col].dtype == int
        assert all(table[code_col].isin(scheme.index.values()))


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
    indexed_dataset = indexed_dataset.execute_pipeline()

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
    indexed_dataset = indexed_dataset.execute_pipeline()

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
    indexed_dataset = indexed_dataset.execute_pipeline()

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
    assert len(merged_dataset.tables.static) > len(filtered_dataset.tables.static)

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


def test_select_subjects_with_observation(indexed_dataset: Dataset):
    assert False


@pytest.fixture
def shifted_timestamps_dataset(indexed_dataset: Dataset):
    if any(len(getattr(indexed_dataset.tables, k)) == 0 for k in indexed_dataset.config.tables.time_cols.keys()):
        pytest.skip("No temporal data in dataset.")

    indexed_dataset = indexed_dataset.execute_pipeline()

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


@pytest.fixture
def unit_converter_table(dataset_config, dataset_tables):
    if 'icu_inputs' not in dataset_tables.tables_dict or len(dataset_tables.icu_inputs) == 0:
        pytest.skip("No ICU inputs in dataset.")
    c_code = dataset_config.tables.icu_inputs.code_alias
    c_amount_unit = dataset_config.tables.icu_inputs.amount_unit_alias
    c_norm_factor = dataset_config.tables.icu_inputs.derived_unit_normalization_factor
    c_universal_unit = dataset_config.tables.icu_inputs.derived_universal_unit
    icu_inputs = dataset_tables.icu_inputs

    table = pd.DataFrame(columns=[c_code, c_amount_unit],
                         data=[(code, unit) for code, unit in
                               icu_inputs.groupby([c_code, c_amount_unit]).groups.keys()])

    for code, df in table.groupby(c_code):
        units = df[c_amount_unit].unique()
        universal_unit = np.random.choice(units, size=1)[0]
        norm_factor = 1
        if len(units) > 1:
            norm_factor = np.random.choice([1e-3, 100, 10, 1e3], size=len(units))
            norm_factor = np.where(units == universal_unit, 1, norm_factor)
        table.loc[df.index, c_norm_factor] = norm_factor
        table.loc[df.index, c_universal_unit] = universal_unit

    return table


def test_icu_input_rate_unit_conversion(indexed_dataset: Dataset, unit_converter_table: pd.DataFrame):
    indexed_dataset = indexed_dataset.execute_pipeline()

    fixed_dataset, _ = ICUInputRateUnitConversion(conversion_table=unit_converter_table)(indexed_dataset, {})
    icu_inputs0 = indexed_dataset.tables.icu_inputs
    icu_inputs1 = fixed_dataset.tables.icu_inputs
    c_code = indexed_dataset.config.tables.icu_inputs.code_alias
    c_amount = indexed_dataset.config.tables.icu_inputs.amount_alias
    c_amount_unit = indexed_dataset.config.tables.icu_inputs.amount_unit_alias
    c_norm_factor = indexed_dataset.config.tables.icu_inputs.derived_unit_normalization_factor
    c_universal_unit = indexed_dataset.config.tables.icu_inputs.derived_universal_unit
    c_norm_amount = indexed_dataset.config.tables.icu_inputs.derived_normalized_amount
    c_rate = indexed_dataset.config.tables.icu_inputs.derived_normalized_amount_per_hour
    _derived_cols = [c_norm_amount, c_rate, c_norm_factor, c_universal_unit]
    assert all(c not in icu_inputs0.columns for c in _derived_cols)
    assert all(c in icu_inputs1.columns for c in _derived_cols)

    # For every (code, unit) pair, a unique normalization factor and universal unit is assigned.
    for (code, unit), inputs_df in icu_inputs1.groupby([c_code, c_amount_unit]):
        ctable = unit_converter_table[(unit_converter_table[c_code] == code)]
        ctable = ctable[ctable[c_amount_unit] == unit]

        norm_factor = ctable[c_norm_factor].iloc[0]
        universal_unit = ctable[c_universal_unit].iloc[0]

        assert inputs_df[c_universal_unit].unique() == universal_unit
        assert inputs_df[c_norm_factor].unique() == norm_factor
        assert inputs_df[c_norm_amount].equals(inputs_df[c_amount] * norm_factor)


@pytest.fixture
def preprocessed_dataset(indexed_dataset, unit_converter_table):
    conv = ICUInputRateUnitConversion(conversion_table=unit_converter_table)
    transformations = [SetIndex(), conv, SetCodeIntegerIndices()]
    return eqx.tree_at(lambda x: x.core_pipeline.transformations, indexed_dataset, transformations).execute_pipeline()


@pytest.fixture
def nan_inputs_dataset(preprocessed_dataset: Dataset):
    if 'icu_inputs' not in preprocessed_dataset.tables.tables_dict or len(preprocessed_dataset.tables.icu_inputs) == 0:
        pytest.skip("No ICU inputs in dataset.")

    preprocessed_dataset = preprocessed_dataset.execute_pipeline()

    c_rate = preprocessed_dataset.config.tables.icu_inputs.derived_normalized_amount_per_hour
    c_admission_id = preprocessed_dataset.config.tables.icu_inputs.admission_id_alias
    icu_inputs = preprocessed_dataset.tables.icu_inputs.copy()
    admission_id = icu_inputs.iloc[0][c_admission_id]
    icu_inputs.loc[icu_inputs[c_admission_id] == admission_id, c_rate] = np.nan
    return eqx.tree_at(lambda x: x.tables.icu_inputs, preprocessed_dataset, icu_inputs)


def test_filter_invalid_input_rates_subjects(nan_inputs_dataset: Dataset):
    fixed_dataset, _ = FilterInvalidInputRatesSubjects()(nan_inputs_dataset, {})
    icu_inputs0 = nan_inputs_dataset.tables.icu_inputs
    admissions0 = nan_inputs_dataset.tables.admissions
    static0 = nan_inputs_dataset.tables.static

    icu_inputs1 = fixed_dataset.tables.icu_inputs
    admissions1 = fixed_dataset.tables.admissions
    static1 = fixed_dataset.tables.static

    c_rate = nan_inputs_dataset.config.tables.icu_inputs.derived_normalized_amount_per_hour
    c_admission_id = nan_inputs_dataset.config.tables.icu_inputs.admission_id_alias
    c_subject_id = nan_inputs_dataset.config.tables.admissions.subject_id_alias
    admission_id = icu_inputs0.iloc[0][c_admission_id]
    subject_id = static0[static0.index == admissions0.loc[admission_id, c_subject_id]].index[0]
    subject_admissions = admissions0[admissions0[c_subject_id] == subject_id]

    assert not subject_id in static1.index
    assert not subject_admissions.index.isin(admissions1.index).all()
    assert not subject_admissions.index.isin(icu_inputs1[c_admission_id]).all()
    assert icu_inputs0[c_rate].isna().any()
    assert not icu_inputs1[c_rate].isna().any()


@pytest.mark.parametrize('splits', [[0.5], [0.2, 0.5, 0.7], [0.1, 0.2, 0.3, 0.4, 0.5]])
def test_random_splits(indexed_dataset: Dataset, splits: List[float]):
    indexed_dataset = indexed_dataset.execute_pipeline()

    if len(indexed_dataset.tables.admissions) == 0 or len(splits) >= len(indexed_dataset.subject_ids):
        pytest.skip("No admissions in dataset or splits requested exceeds the number of subjects.")

    _, aux_subjs = RandomSplits(splits=splits, splits_key='splits', seed=0, balance='subjects',
                                discount_first_admission=False)(indexed_dataset, {})
    _, aux_adms = RandomSplits(splits=splits, splits_key='splits', seed=0, balance='admissions',
                               discount_first_admission=False)(indexed_dataset, {})
    _, aux_los = RandomSplits(splits=splits, splits_key='splits', seed=0, balance='admissions_intervals',
                              discount_first_admission=False)(indexed_dataset, {})

    for aux in (aux_subjs, aux_adms, aux_los):
        assert len(aux['splits']) == len(splits) + 1
        # No overlaps.
        assert sum(len(v) for v in aux['splits']) == len(indexed_dataset.subject_ids)
        assert set.union(*[set(v) for v in aux['splits']]) == set(indexed_dataset.subject_ids)

    # # test proportionality
    # NOTE: no specified behaviour when splits have equal proportions, so comparing argsorts
    # is not appropriate.
    splits_proportions = [p1 - p0 for p0, p1 in zip([0] + splits, splits + [1])]
    n_adms = lambda subjects: sum(indexed_dataset.subjects_n_admissions.loc[subjects])
    total_los = lambda subjects: sum(indexed_dataset.subjects_intervals_sum.loc[subjects])
    p_threshold = 1 / len(indexed_dataset.subject_ids)
    for i in range(len(splits_proportions)):
        for j in range(i + 1, len(splits_proportions)):
            if abs(splits_proportions[i] - splits_proportions[j]) < p_threshold:
                assert abs(len(aux_subjs['splits'][i]) - len(aux_subjs['splits'][j])) <= 1
            elif splits_proportions[i] > splits_proportions[j]:
                assert len(aux_subjs['splits'][i]) >= len(aux_subjs['splits'][j])
                assert n_adms(aux_adms['splits'][i]) >= n_adms(aux_adms['splits'][j])
                assert total_los(aux_los['splits'][i]) >= total_los(aux_los['splits'][j])
            else:
                assert len(aux_subjs['splits'][i]) <= len(aux_subjs['splits'][j])
                assert n_adms(aux_adms['splits'][i]) <= n_adms(aux_adms['splits'][j])
                assert total_los(aux_los['splits'][i]) <= total_los(aux_los['splits'][j])


@pytest.mark.parametrize('fit_only', [True, False])
@pytest.mark.parametrize('use_float16', [True, False])
@pytest.mark.parametrize('scaler', [('obs', ObsAdaptiveScaler), ('icu_inputs', InputScaler)])
def test_trainable_transformer(preprocessed_dataset: Dataset, use_float16: bool, fit_only: bool,
                               scaler: Tuple[str, Type[TrainableTransformation]]):
    if len(preprocessed_dataset.tables.static) < 5 or len(getattr(preprocessed_dataset.tables, scaler[0])) == 0:
        pytest.skip("Not enough subjects in dataset or no data to scale.")
    table_name, scaler_class = scaler
    scaler_name = f'{table_name}_scaler'

    with pytest.raises(AssertionError):
        scaler_class(use_float16=use_float16, transformer_key=scaler_name,
                     fit_only=fit_only, splits_key='splits',
                     training_split_index=0)(preprocessed_dataset, {})

    aux = {'splits': [preprocessed_dataset.subject_ids[:3], preprocessed_dataset.subject_ids[3:]]}
    transformer = scaler_class(use_float16=use_float16, transformer_key=scaler_name,
                               splits_key='splits',
                               fit_only=fit_only,
                               training_split_index=0)

    assert isinstance(transformer, TrainableTransformation)
    assert transformer.fit_only == fit_only
    assert transformer.transformer_key == scaler_name

    scaled_ds, aux = transformer(preprocessed_dataset, aux)
    scaler = aux[scaler_name]
    assert scaler is not None
    assert isinstance(scaler, CodedValueScaler)
    assert scaler.table(scaled_ds) is getattr(scaled_ds.tables, table_name)
    assert scaler.table(preprocessed_dataset) is getattr(preprocessed_dataset.tables, table_name)
    assert scaler.use_float16 == use_float16

    table0 = scaler.table(preprocessed_dataset)
    table1 = scaler.table(scaled_ds)
    c_value = scaler.value_column(scaled_ds)
    c_code = scaler.code_column(scaled_ds)
    assert c_value in table1.columns
    assert c_code in table1.columns
    if fit_only:
        assert table1 is table0
        assert table1[c_value].dtype == scaler.original_dtype
    else:
        assert table1 is not table0

        if use_float16:
            assert table1[c_value].dtype == np.float16
        else:
            assert table1[c_value].dtype == table0[c_value].dtype


# def test_obs_minmax_scaler(int_indexed_dataset: Dataset):
#     assert False
#
#
# def test_obs_adaptive_scaler(int_indexed_dataset: Dataset):
#     assert False
#
#
# def test_obs_iqr_outlier_remover(indexed_dataset: Dataset):
#     assert False


@pytest.mark.parametrize('illegal_transformation_sequence', [
    (SetIndex(), SetCodeIntegerIndices(), SetIndex()),  # No duplicates.
    (SetIndex(), SetCodeIntegerIndices(), SetCodeIntegerIndices()),  # No duplicates.
    (SampleSubjects(n_subjects=1),),  # Needs SetIndex before.
    # SetAdmissionRelativeTimes() needs SetIndex, CastTimestamps before.
    (SetAdmissionRelativeTimes(),),
    (SetIndex(), SetAdmissionRelativeTimes()),
    (CastTimestamps(), SetAdmissionRelativeTimes()),
    # FilterSubjectsNegativeAdmissionLengths() needs SetIndex, CastTimestamps before.
    (FilterSubjectsNegativeAdmissionLengths(),),
    (SetIndex(), FilterSubjectsNegativeAdmissionLengths()),
    (CastTimestamps(), FilterSubjectsNegativeAdmissionLengths()),
    # FilterUnsupportedCodes() blocked by SetCodeIntegerIndices before.
    (SetCodeIntegerIndices(), FilterUnsupportedCodes(),),
    # ProcessOverlappingAdmissions() needs SetIndex, CastTimestamps before.
    (ProcessOverlappingAdmissions(merge=True),),
    (SetIndex(), ProcessOverlappingAdmissions(merge=True)),
    (CastTimestamps(), ProcessOverlappingAdmissions(merge=True)),
    # FilterClampTimestampsToAdmissionInterval() needs SetIndex, CastTimestamps before.
    # Blocked by SetAdmissionRelativeTimes
    (FilterClampTimestampsToAdmissionInterval(),),
    (SetIndex(), FilterClampTimestampsToAdmissionInterval()),
    (CastTimestamps(), FilterClampTimestampsToAdmissionInterval()),
    (SetIndex(), CastTimestamps(), SetAdmissionRelativeTimes(), FilterClampTimestampsToAdmissionInterval()),
    # ICUInputRateUnitConversion() is blocked by SetCodeIntegerIndices.
    (SetCodeIntegerIndices(), ICUInputRateUnitConversion(conversion_table=None),),
    # FilterInvalidInputRatesSubjects() needs SetIndex, ICUInputRateUnitConversion before.
    (FilterInvalidInputRatesSubjects(),),
    (SetIndex(), FilterInvalidInputRatesSubjects()),
    (ICUInputRateUnitConversion(conversion_table=None), FilterInvalidInputRatesSubjects()),
    # RandomSplits() needs SetIndex, CastTimestamps before.
    (RandomSplits(splits=[0.5], splits_key=''),),
    (SetIndex(), RandomSplits(splits=[0.5], splits_key='')),
    (CastTimestamps(), RandomSplits(splits=[0.5], splits_key='')),
    # ObsIQROutlierRemover(TrainableTransformation) needs RandomSplits, SetIndex, SetCodeIntegerIndices before.
    (ObsIQROutlierRemover(splits_key=''),),
    (RandomSplits(splits=[0.5], splits_key=''), ObsIQROutlierRemover(splits_key='')),
    (SetIndex(), ObsIQROutlierRemover(splits_key='')),
    (SetCodeIntegerIndices(), ObsIQROutlierRemover(splits_key='')),
    (SetIndex(), SetCodeIntegerIndices(), ObsIQROutlierRemover(splits_key='')),
    (SetIndex(), RandomSplits(splits=[0.5], splits_key=''), ObsIQROutlierRemover(splits_key='')),
    (SetCodeIntegerIndices(), RandomSplits(splits=[0.5], splits_key=''), ObsIQROutlierRemover(splits_key='')),
    # ObsAdaptiveScaler(TrainableTransformation) needs RandomSplits, SetIndex, SetCodeIntegerIndices,
    # ObsIQROutlierRemover before.
    (ObsAdaptiveScaler(splits_key=''),),
    (SetCodeIntegerIndices(), RandomSplits(splits=[0.5], splits_key=''), ObsAdaptiveScaler(splits_key='')),
    (SetIndex(), SetCodeIntegerIndices(), RandomSplits(splits=[0.5], splits_key=''), ObsAdaptiveScaler(splits_key='')),
    # InputScaler(TrainableTransformation) needs RandomSplits, SetIndex, SetCodeIntegerIndices,
    # ICUInputRateUnitConversion, FilterInvalidInputRatesSubjects before.
    (SetIndex(), SetCodeIntegerIndices(), RandomSplits(splits=[0.5], splits_key=''),
     ICUInputRateUnitConversion(conversion_table=None), InputScaler(splits_key='')),
    (SetIndex(), SetCodeIntegerIndices(), RandomSplits(splits=[0.5], splits_key=''), FilterInvalidInputRatesSubjects(),
     InputScaler(splits_key=''))])
def test_pipeline_transformers_sequence(illegal_transformation_sequence: List[DatasetTransformation]):
    with pytest.raises(AssertionError):
        DatasetPipeline(transformations=illegal_transformation_sequence)
