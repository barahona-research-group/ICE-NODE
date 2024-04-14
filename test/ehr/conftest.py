import dataclasses
import random
from typing import List
from unittest.mock import patch, PropertyMock

import equinox as eqx
import numpy as np
import numpy.random as nrand
import pandas as pd
import pytest
import tables as tb

from lib.ehr import CodingScheme, \
    TVxEHR, TVxEHRConfig, DemographicVectorConfig, InpatientObservables
from lib.ehr.coding_scheme import ExcludingOutcomeExtractor, NumericScheme, FrozenDict11, CodingSchemesManager, \
    FrozenDict1N, ReducedCodeMapN1, CodesVector, OutcomeExtractor
from lib.ehr.dataset import StaticTableConfig, AdmissionTableConfig, AdmissionLinkedCodedValueTableConfig, \
    AdmissionIntervalBasedCodedTableConfig, RatedInputTableConfig, AdmissionTimestampedCodedValueTableConfig, \
    DatasetTablesConfig, DatasetSchemeConfig, DatasetTables, Dataset, DatasetConfig, AbstractDatasetPipeline, \
    DatasetScheme
from lib.ehr.transformations import SetIndex, ICUInputRateUnitConversion, CastTimestamps, SetAdmissionRelativeTimes, \
    ValidatedDatasetPipeline
from lib.ehr.tvx_concepts import SegmentedAdmission, InpatientInterventions, AdmissionDates, Admission, \
    LeadingObservableExtractor, SegmentedInpatientInterventions, LeadingObservableExtractorConfig, InpatientInput, \
    Patient, SegmentedPatient, StaticInfo
from lib.ehr.tvx_ehr import TVxEHRSchemeConfig, AbstractTVxPipeline

DATASET_SCOPE = "function"
MAX_STAY_DAYS = 356


def scheme(name: str, codes: List[str]) -> CodingScheme:
    return CodingScheme(name=name, codes=tuple(sorted(codes)),
                        desc=FrozenDict11.from_dict(dict(zip(codes, codes))))


def sample_codes(scheme: CodingScheme, n: int) -> List[str]:
    codes = scheme.codes
    return random.choices(codes, k=n)


# Static Columns
@pytest.fixture(params=['SUBJECT_IDXYZ'])
def subject_id_alias(request):
    return request.param


@pytest.fixture(params=['race'])
def race_alias(request):
    return request.param


@pytest.fixture(params=['gender'])
def gender_alias(request):
    return request.param


@pytest.fixture(params=['dob'])
def date_of_birth_alias(request):
    return request.param


@pytest.fixture
def static_table_config(subject_id_alias, gender_alias, race_alias, date_of_birth_alias):
    return StaticTableConfig(subject_id_alias=subject_id_alias,
                             gender_alias=gender_alias, race_alias=race_alias,
                             date_of_birth_alias=date_of_birth_alias)


@pytest.fixture(params=['ADMISSION_IDX'])
def admission_id_alias(request):
    return request.param


@pytest.fixture(params=['admission_time'])
def admission_time_alias(request):
    return request.param


@pytest.fixture(params=['discharge_time'])
def discharge_time_alias(request):
    return request.param


@pytest.fixture
def admission_table_config(static_table_config, admission_id_alias, admission_time_alias,
                           discharge_time_alias):
    return AdmissionTableConfig(subject_id_alias=static_table_config.subject_id_alias,
                                admission_id_alias=admission_id_alias,
                                admission_time_alias=admission_time_alias,
                                discharge_time_alias=discharge_time_alias)


# Obs Columns
@pytest.fixture(params=['time_bin'])
def obs_time_alias(request):
    return request.param


@pytest.fixture(params=['measurement'])
def obs_code_alias(request):
    return request.param


@pytest.fixture(params=['measurement description'])
def obs_code_desc_alias(request):
    return request.param


@pytest.fixture(params=['obs_val'])
def obs_value_alias(request):
    return request.param


@pytest.fixture
def obs_table_config(admission_table_config, obs_time_alias, obs_code_alias,
                     obs_code_desc_alias,
                     obs_value_alias):
    return AdmissionTimestampedCodedValueTableConfig(admission_id_alias=admission_table_config.admission_id_alias,
                                                     time_alias=obs_time_alias,
                                                     code_alias=obs_code_alias,
                                                     description_alias=obs_code_desc_alias,
                                                     value_alias=obs_value_alias)


# Dx Columns
@pytest.fixture(params=['dx_code'])
def dx_code_alias(request):
    return request.param


@pytest.fixture(params=['dx_code_desc'])
def dx_code_desc_alias(request):
    return request.param


@pytest.fixture
def dx_discharge_table_config(admission_table_config, dx_code_alias, dx_code_desc_alias):
    return AdmissionLinkedCodedValueTableConfig(admission_id_alias=admission_table_config.admission_id_alias,
                                                code_alias=dx_code_alias, description_alias=dx_code_desc_alias)


# Hosp Proc Columns
@pytest.fixture(params=['hosp_proc_code'])
def hosp_proc_code_alias(request):
    return request.param


@pytest.fixture(params=['hosp_proc_code_desc'])
def hosp_proc_code_desc_alias(request):
    return request.param


@pytest.fixture(params=['hosp_proc_start_time'])
def hosp_proc_start_time_alias(request):
    return request.param


@pytest.fixture(params=['hosp_proc_end_time'])
def hosp_proc_end_time_alias(request):
    return request.param


@pytest.fixture
def hosp_proc_table_config(admission_table_config, hosp_proc_code_alias, hosp_proc_code_desc_alias,
                           hosp_proc_start_time_alias, hosp_proc_end_time_alias):
    return AdmissionIntervalBasedCodedTableConfig(admission_id_alias=admission_table_config.admission_id_alias,
                                                  code_alias=hosp_proc_code_alias,
                                                  description_alias=hosp_proc_code_desc_alias,
                                                  start_time_alias=hosp_proc_start_time_alias,
                                                  end_time_alias=hosp_proc_end_time_alias)


# ICU Proc Columns
@pytest.fixture(params=['icu_proc_code'])
def icu_proc_code_alias(request):
    return request.param


@pytest.fixture(params=['icu_proc_code_desc'])
def icu_proc_code_desc_alias(request):
    return request.param


@pytest.fixture(params=['icu_proc_start_time'])
def icu_proc_start_time_alias(request):
    return request.param


@pytest.fixture(params=['icu_proc_end_time'])
def icu_proc_end_time_alias(request):
    return request.param


@pytest.fixture
def icu_proc_table_config(admission_table_config, icu_proc_code_alias, icu_proc_code_desc_alias,
                          icu_proc_start_time_alias, icu_proc_end_time_alias):
    return AdmissionIntervalBasedCodedTableConfig(admission_id_alias=admission_table_config.admission_id_alias,
                                                  code_alias=icu_proc_code_alias,
                                                  description_alias=icu_proc_code_desc_alias,
                                                  start_time_alias=icu_proc_start_time_alias,
                                                  end_time_alias=icu_proc_end_time_alias)


# ICU Input Columns
@pytest.fixture(params=['icu_input_code'])
def icu_input_code_alias(request):
    return request.param


@pytest.fixture(params=['icu_input_code_desc'])
def icu_input_code_desc_alias(request):
    return request.param


@pytest.fixture(params=['icu_input_start_time'])
def icu_input_start_time_alias(request):
    return request.param


@pytest.fixture(params=['icu_input_end_time'])
def icu_input_end_time_alias(request):
    return request.param


@pytest.fixture(params=['icu_input_amount'])
def icu_input_amount_alias(request):
    return request.param


@pytest.fixture(params=['icu_input_amount_uom'])
def icu_input_amount_unit_alias(request):
    return request.param


@pytest.fixture(params=['icu_input_derived_normalized_amount'])
def icu_input_derived_normalized_amount(request):
    return request.param


@pytest.fixture(params=['icu_input_derived_normalized_amount_per_hour'])
def icu_input_derived_normalized_amount_per_hour(request):
    return request.param


@pytest.fixture(params=['icu_input_derived_unit_normalization_factor'])
def icu_input_derived_unit_normalization_factor(request):
    return request.param


@pytest.fixture(params=['icu_input_derived_universal_unit'])
def icu_input_derived_universal_unit(request):
    return request.param


@pytest.fixture
def icu_input_table_config(admission_table_config: AdmissionTableConfig,
                           icu_input_code_alias: str,
                           icu_input_code_desc_alias: str, icu_input_start_time_alias: str,
                           icu_input_end_time_alias: str, icu_input_amount_alias: str,
                           icu_input_amount_unit_alias: str,
                           icu_input_derived_normalized_amount: str,
                           icu_input_derived_normalized_amount_per_hour: str,
                           icu_input_derived_unit_normalization_factor: str,
                           icu_input_derived_universal_unit: str) -> RatedInputTableConfig:
    return RatedInputTableConfig(admission_id_alias=admission_table_config.admission_id_alias,
                                 code_alias=icu_input_code_alias,
                                 description_alias=icu_input_code_desc_alias,
                                 start_time_alias=icu_input_start_time_alias,
                                 end_time_alias=icu_input_end_time_alias,
                                 amount_alias=icu_input_amount_alias,
                                 amount_unit_alias=icu_input_amount_unit_alias,
                                 derived_normalized_amount=icu_input_derived_normalized_amount,
                                 derived_normalized_amount_per_hour=icu_input_derived_normalized_amount_per_hour,
                                 derived_unit_normalization_factor=icu_input_derived_unit_normalization_factor,
                                 derived_universal_unit=icu_input_derived_universal_unit)


def sample_subjects_dataframe(static_table_config: StaticTableConfig,
                              ethnicity_scheme: CodingScheme, gender_scheme: CodingScheme, n: int) -> pd.DataFrame:
    return pd.DataFrame({
        static_table_config.subject_id_alias: list(str(i) for i in range(n)),
        static_table_config.race_alias: random.choices(ethnicity_scheme.codes, k=n),
        static_table_config.gender_alias: random.choices(gender_scheme.codes, k=n),
        static_table_config.date_of_birth_alias: pd.to_datetime(
            random.choices(pd.date_range(start='1/1/1900', end='1/1/2000', freq='D'), k=n))
    })


def sample_admissions_dataframe(subjects_df: pd.DataFrame,
                                static_table_config: StaticTableConfig,
                                admission_table_config: AdmissionTableConfig,
                                n: int) -> pd.DataFrame:
    c_subject = static_table_config.subject_id_alias
    c_admission = admission_table_config.admission_id_alias
    c_admission_time = admission_table_config.admission_time_alias
    c_discharge_time = admission_table_config.discharge_time_alias
    admit_dates = pd.to_datetime(random.choices(pd.date_range(start='1/1/2000', end='1/1/2020', freq='D'), k=n))
    disch_dates = admit_dates + pd.to_timedelta(random.choices(range(1, MAX_STAY_DAYS), k=n), unit='D')

    return pd.DataFrame({
        c_subject: random.choices(subjects_df[c_subject], k=n),
        c_admission: list(str(i) for i in range(n)),
        c_admission_time: admit_dates,
        c_discharge_time: disch_dates
    })


def sample_dx_dataframe(admissions_df: pd.DataFrame,
                        admission_table_config: AdmissionTableConfig,
                        dx_discharge_table_config: AdmissionLinkedCodedValueTableConfig,
                        dx_scheme: CodingScheme,
                        n: int) -> pd.DataFrame:
    c_admission = admission_table_config.admission_id_alias
    c_dx = dx_discharge_table_config.code_alias
    dx_codes = sample_codes(dx_scheme, n)
    return pd.DataFrame({
        c_admission: random.choices(admissions_df[c_admission], k=n),
        c_dx: dx_codes
    })


def _sample_proc_dataframe(admissions_df: pd.DataFrame,
                           admission_table_config: AdmissionTableConfig,
                           table_config: AdmissionIntervalBasedCodedTableConfig,
                           scheme: CodingScheme,
                           n: int) -> pd.DataFrame:
    c_admission = admission_table_config.admission_id_alias
    c_admittime = admission_table_config.admission_time_alias
    c_dischtime = admission_table_config.discharge_time_alias
    c_code = table_config.code_alias
    c_start = table_config.start_time_alias
    c_end = table_config.end_time_alias
    codes = sample_codes(scheme, n)
    df = pd.DataFrame({
        c_admission: random.choices(admissions_df[c_admission], k=n),
        c_code: codes
    })
    df = pd.merge(df, admissions_df[[c_admission, c_admittime, c_dischtime]],
                  on=c_admission,
                  suffixes=(None, '_admission'))
    df['los'] = (df[c_dischtime] - df[c_admittime]).dt.total_seconds() / 3600

    relative_start = np.random.uniform(0, df['los'].values, size=n)
    df[c_start] = df[c_admittime] + pd.to_timedelta(relative_start, unit='H')

    relative_end = np.random.uniform(low=relative_start,
                                     high=df['los'].values, size=n)

    df[c_end] = df[c_admittime] + pd.to_timedelta(relative_end, unit='H')
    return df[[c_admission, c_code, c_start, c_end]]


def sample_icu_inputs_dataframe(admissions_df: pd.DataFrame,
                                admission_table_config: AdmissionTableConfig,
                                table_config: RatedInputTableConfig,
                                icu_input_scheme: CodingScheme,
                                n: int) -> pd.DataFrame:
    df = _sample_proc_dataframe(admissions_df, admission_table_config, table_config, icu_input_scheme, n)
    c_amount = table_config.amount_alias
    c_unit = table_config.amount_unit_alias
    df[c_amount] = np.random.uniform(low=0, high=1000, size=n)
    df[c_unit] = random.choices(['mg', 'g', 'kg', 'cm', 'dose', 'ml'], k=n)
    return df


def sample_obs_dataframe(admissions_df: pd.DataFrame,
                         admission_table_config: AdmissionTableConfig,
                         obs_table_config: AdmissionTimestampedCodedValueTableConfig,
                         obs_scheme: CodingScheme,
                         n: int) -> pd.DataFrame:
    c_admission = admission_table_config.admission_id_alias
    c_admittime = admission_table_config.admission_time_alias
    c_dischtime = admission_table_config.discharge_time_alias
    c_obs = obs_table_config.code_alias
    c_time = obs_table_config.time_alias
    c_value = obs_table_config.value_alias

    codes = sample_codes(obs_scheme, n)
    df = pd.DataFrame({
        c_admission: random.choices(admissions_df[c_admission], k=n),
        c_obs: codes
    })
    df = pd.merge(df, admissions_df[[c_admission, c_admittime, c_dischtime]], on=c_admission,
                  suffixes=(None, '_y'))
    df['los'] = (df[c_dischtime] - df[c_admittime]).dt.total_seconds() / 3600
    relative_time = np.random.uniform(0, df['los'].values, size=n)
    df[c_time] = df[c_admittime] + pd.to_timedelta(relative_time, unit='H')

    assert isinstance(obs_scheme, NumericScheme), 'Only numeric schemes are supported'
    df['obs_type'] = df[c_obs].map(obs_scheme.type_hint)
    df.loc[df.obs_type == 'N', c_value] = np.random.uniform(low=0, high=1000, size=(df.obs_type == 'N').sum())
    df.loc[df.obs_type.isin(('C', 'O')), c_value] = random.choices([0, 1, 2],
                                                                   k=df.obs_type.isin(('C', 'O')).sum())
    df.loc[df.obs_type == 'B', c_value] = random.choices([0, 1], k=(df.obs_type == 'B').sum())

    return df[[c_admission, c_obs, c_time, c_value]]


BINARY_OBSERVATION_CODE_INDEX = 0
CATEGORICAL_OBSERVATION_CODE_INDEX = 1
ORDINAL_OBSERVATION_CODE_INDEX = 2
NUMERIC_OBSERVATION_CODE_INDEX = 3


@pytest.fixture(params=[('ethnicity1', ['E1', 'E2', 'E3'])])
def ethnicity_scheme(request) -> CodingScheme:
    return scheme(*request.param)


@pytest.fixture(params=[('gender1', ['M', 'F'])])
def gender_scheme(request) -> CodingScheme:
    return scheme(*request.param)


@pytest.fixture(params=[('dx1', ['Dx1', 'Dx2', 'Dx3', 'Dx4', 'Dx5'])])
def dx_scheme(request) -> CodingScheme:
    return scheme(*request.param)


@pytest.fixture(params=[('hosp_proc1', ['P1', 'P2', 'P3'])])
def hosp_proc_scheme(request) -> CodingScheme:
    return scheme(*request.param)


@pytest.fixture(params=[('icu_proc1', ['P1', 'P2', 'P3'])])
def icu_proc_scheme(request) -> CodingScheme:
    return scheme(*request.param)


@pytest.fixture(params=[('icu_inputs1', ['I1', 'I2', 'I3'])])
def icu_inputs_scheme(request) -> CodingScheme:
    return scheme(*request.param)


@pytest.fixture(params=[('observation1',
                         ('O1', 'O2', 'O3', 'O4', 'O5'),
                         ('B', 'C', 'O', 'N', 'N'))])
def observation_scheme(request) -> CodingScheme:
    name, codes, types = request.param
    return NumericScheme(name=name,
                         codes=tuple(sorted(codes)),
                         desc=FrozenDict11.from_dict(dict(zip(codes, codes))),
                         type_hint=FrozenDict11.from_dict(dict(zip(codes, types))))


@pytest.fixture
def outcome_extractor(dx_scheme: CodingScheme) -> ExcludingOutcomeExtractor:
    name = f'{dx_scheme.name}_outcome'
    k = max(3, len(dx_scheme.codes) - 1)
    random.seed(0)
    excluded = random.sample(dx_scheme.codes, k=k)
    manager = CodingSchemesManager().add_scheme(dx_scheme).add_outcome(ExcludingOutcomeExtractor(name=name,
                                                                                                 base_name=dx_scheme.name,
                                                                                                 exclude_codes=excluded))
    return manager.outcome[name]


@pytest.fixture
def dataset_scheme_config(ethnicity_scheme: CodingScheme,
                          gender_scheme: CodingScheme,
                          dx_scheme: CodingScheme,
                          icu_proc_scheme: CodingScheme,
                          icu_inputs_scheme: CodingScheme,
                          observation_scheme: CodingScheme,
                          hosp_proc_scheme: CodingScheme) -> DatasetSchemeConfig:
    return DatasetSchemeConfig(ethnicity=ethnicity_scheme.name,
                               gender=gender_scheme.name,
                               dx_discharge=dx_scheme.name,
                               icu_procedures=icu_proc_scheme.name,
                               icu_inputs=icu_inputs_scheme.name,
                               obs=observation_scheme.name,
                               hosp_procedures=hosp_proc_scheme.name)


@pytest.fixture(params=[('icu_inputs1_target', ['I1_target', 'I2_target', 'I3_target', 'I4_target', 'I5_target'])])
def icu_inputs_target_scheme(request) -> CodingScheme:
    return scheme(*request.param)


@pytest.fixture
def tvx_ehr_scheme_config(ethnicity_scheme: CodingScheme,
                          gender_scheme: CodingScheme,
                          dx_scheme: CodingScheme,
                          outcome_extractor: CodingScheme,
                          icu_proc_scheme: CodingScheme,
                          icu_inputs_target_scheme: CodingScheme,
                          observation_scheme: CodingScheme,
                          hosp_proc_scheme: CodingScheme) -> DatasetSchemeConfig:
    return TVxEHRSchemeConfig(ethnicity=ethnicity_scheme.name,
                              gender=gender_scheme.name,
                              dx_discharge=dx_scheme.name,
                              outcome=outcome_extractor.name,
                              icu_procedures=icu_proc_scheme.name,
                              icu_inputs=icu_inputs_target_scheme.name,
                              obs=observation_scheme.name,
                              hosp_procedures=hosp_proc_scheme.name)


@pytest.fixture
def icu_inputs_aggregation(icu_inputs_target_scheme) -> FrozenDict11:
    return FrozenDict11.from_dict({c: 'w_sum' for c in icu_inputs_target_scheme.codes})


@pytest.fixture
def icu_inputs_mapping_data(icu_inputs_scheme, icu_inputs_target_scheme) -> FrozenDict1N:
    return FrozenDict1N.from_dict({c: {random.choice(icu_inputs_target_scheme.codes)} for c in icu_inputs_scheme.codes})


@pytest.fixture
def dataset_scheme_manager(ethnicity_scheme: CodingScheme,
                           gender_scheme: CodingScheme,
                           dx_scheme: CodingScheme,
                           outcome_extractor: ExcludingOutcomeExtractor,
                           icu_proc_scheme: CodingScheme,
                           icu_inputs_scheme: CodingScheme,
                           icu_inputs_target_scheme: CodingScheme,
                           icu_inputs_mapping_data: FrozenDict1N,
                           icu_inputs_aggregation: FrozenDict11,
                           observation_scheme: CodingScheme,
                           hosp_proc_scheme: CodingScheme
                           ) -> CodingSchemesManager:
    manager = CodingSchemesManager().add_outcome(outcome_extractor)
    for scheme in (ethnicity_scheme, gender_scheme, dx_scheme, icu_inputs_target_scheme,
                   icu_proc_scheme, icu_inputs_scheme, observation_scheme, hosp_proc_scheme):
        manager = manager.add_scheme(scheme)

    m = ReducedCodeMapN1.from_data(icu_inputs_scheme.name,
                                   icu_inputs_target_scheme.name,
                                   icu_inputs_mapping_data, icu_inputs_aggregation)
    manager = manager.add_map(m)
    return manager


@pytest.fixture
def dataset_tables_config(static_table_config: StaticTableConfig,
                          admission_table_config: AdmissionTableConfig,
                          obs_table_config: AdmissionTimestampedCodedValueTableConfig,
                          dx_discharge_table_config: AdmissionLinkedCodedValueTableConfig,
                          hosp_proc_table_config: AdmissionIntervalBasedCodedTableConfig,
                          icu_proc_table_config: AdmissionIntervalBasedCodedTableConfig,
                          icu_input_table_config: AdmissionIntervalBasedCodedTableConfig) -> DatasetTablesConfig:
    return DatasetTablesConfig(static=static_table_config,
                               admissions=admission_table_config,
                               obs=obs_table_config,
                               dx_discharge=dx_discharge_table_config,
                               hosp_procedures=hosp_proc_table_config,
                               icu_procedures=icu_proc_table_config,
                               icu_inputs=icu_input_table_config)


@pytest.fixture(params=[(1, 0, 0), (1, 10, 0), (1, 10, 10),
                        (30, 0, 0), (10, 10, 50)],
                ids=lambda x: f"_{x[0]}_subjects_{x[0] * x[1]}_admissions_{x[0] * x[1] * x[2]}_records")
def dataset_tables(dataset_tables_config: DatasetTablesConfig,
                   dataset_scheme_config: DatasetSchemeConfig,
                   dataset_scheme_manager: CodingSchemesManager,
                   request) -> DatasetTables:
    n_subjects, n_admission_per_subject, n_per_admission = request.param
    subjects_df = sample_subjects_dataframe(dataset_tables_config.static,
                                            dataset_scheme_manager.scheme[dataset_scheme_config.ethnicity],
                                            dataset_scheme_manager.scheme[dataset_scheme_config.gender],
                                            n_subjects)
    admissions_df = sample_admissions_dataframe(subjects_df, dataset_tables_config.static,
                                                dataset_tables_config.admissions,
                                                n_admission_per_subject * n_subjects)
    dx_df = sample_dx_dataframe(admissions_df, dataset_tables_config.admissions,
                                dataset_tables_config.dx_discharge,
                                dataset_scheme_manager.scheme[dataset_scheme_config.dx_discharge],
                                n_per_admission * n_subjects * n_admission_per_subject)

    obs_df = sample_obs_dataframe(admissions_df, dataset_tables_config.admissions,
                                  dataset_tables_config.obs,
                                  dataset_scheme_manager.scheme[dataset_scheme_config.obs],
                                  n_per_admission * n_subjects * n_admission_per_subject)

    icu_proc_df = _sample_proc_dataframe(admissions_df, dataset_tables_config.admissions,
                                         dataset_tables_config.icu_procedures,
                                         dataset_scheme_manager.scheme[dataset_scheme_config.icu_procedures],
                                         n_per_admission * n_subjects * n_admission_per_subject)

    hosp_proc_df = _sample_proc_dataframe(admissions_df, dataset_tables_config.admissions,
                                          dataset_tables_config.hosp_procedures,
                                          dataset_scheme_manager.scheme[dataset_scheme_config.hosp_procedures],
                                          n_per_admission * n_subjects * n_admission_per_subject)

    icu_inputs_df = sample_icu_inputs_dataframe(admissions_df, dataset_tables_config.admissions,
                                                dataset_tables_config.icu_inputs,
                                                dataset_scheme_manager.scheme[dataset_scheme_config.icu_inputs],
                                                n_per_admission * n_subjects * n_admission_per_subject)

    return DatasetTables(static=subjects_df,
                         admissions=admissions_df,
                         dx_discharge=dx_df,
                         obs=obs_df,
                         icu_procedures=icu_proc_df,
                         hosp_procedures=hosp_proc_df,
                         icu_inputs=icu_inputs_df)


class NaiveDataset(Dataset):

    @classmethod
    def _setup_pipeline(cls, config: DatasetConfig) -> AbstractDatasetPipeline:
        return AbstractDatasetPipeline(transformations=[])

    @classmethod
    def load_tables(cls, config: DatasetConfig, scheme: DatasetScheme) -> DatasetTables:
        return None


NaiveDataset.register()


@pytest.fixture
def dataset_config(dataset_scheme_config, dataset_tables_config):
    return DatasetConfig(scheme=dataset_scheme_config, tables=dataset_tables_config)


@pytest.fixture
def dataset(dataset_config, dataset_tables, dataset_scheme_manager):
    ds = NaiveDataset(scheme_manager=dataset_scheme_manager, config=dataset_config)
    return dataclasses.replace(ds, tables=dataset_tables)


@pytest.fixture
def indexed_dataset(dataset) -> NaiveDataset:
    return dataset.execute_external_transformations([SetIndex()])


@pytest.fixture
def has_admissions_dataset(indexed_dataset):
    if len(indexed_dataset.tables.admissions) == 0:
        pytest.skip("No admissions data found in dataset.")
    return indexed_dataset.execute_pipeline()


@pytest.fixture
def has_codes_dataset(has_admissions_dataset: Dataset):
    if all(len(getattr(has_admissions_dataset.tables, k)) == 0 for k in
           has_admissions_dataset.config.tables.code_column.keys()):
        pytest.skip("No coded tables or they are all empty.")
    return has_admissions_dataset


@pytest.fixture
def has_obs_dataset(has_admissions_dataset: Dataset):
    if len(has_admissions_dataset.tables.obs) == 0:
        pytest.skip("No obs data found in dataset.")
    return has_admissions_dataset


@pytest.fixture
def subject_id_column(indexed_dataset: Dataset) -> str:
    return indexed_dataset.config.tables.subject_id_alias


@pytest.fixture
def admission_id_column(indexed_dataset: Dataset) -> str:
    return indexed_dataset.config.tables.admission_id_alias


@pytest.fixture
def sample_subject_id(has_admissions_dataset: Dataset, subject_id_column: str) -> str:
    return has_admissions_dataset.tables.admissions[subject_id_column].iloc[0]


@pytest.fixture
def sample_admission_id(has_admissions_dataset: Dataset) -> str:
    return has_admissions_dataset.tables.admissions.index[0]


class MockMIMICIVDatasetSchemeConfig(DatasetSchemeConfig):

    @property
    def icu_inputs_uom_normalization_table(self):
        return 0


MockMIMICIVDatasetSchemeConfig.register()


class MockMIMICIVDataset(NaiveDataset):

    @staticmethod
    def icu_inputs_uom_normalization(icu_inputs_config: RatedInputTableConfig,
                                     icu_inputs_uom_normalization_table: pd.DataFrame):
        return icu_inputs_uom_normalization_table

    @classmethod
    def _setup_pipeline(cls, config: DatasetConfig) -> AbstractDatasetPipeline:
        return ValidatedDatasetPipeline(transformations=[SetIndex(), CastTimestamps(), ICUInputRateUnitConversion(),
                                                         SetAdmissionRelativeTimes()])


MockMIMICIVDataset.register()


@pytest.fixture
def unit_converter_table(dataset_config, dataset_tables):
    if 'icu_inputs' not in dataset_tables.tables_dict or len(dataset_tables.icu_inputs) == 0:
        pytest.skip("No ICU inputs in dataset. Required for the unit conversion table generation.")
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


@pytest.fixture
def mimiciv_dataset_scheme_config(ethnicity_scheme: CodingScheme,
                                  gender_scheme: CodingScheme,
                                  dx_scheme: CodingScheme,
                                  outcome_extractor: CodingScheme,
                                  icu_proc_scheme: CodingScheme,
                                  icu_inputs_scheme: CodingScheme,
                                  observation_scheme: CodingScheme,
                                  hosp_proc_scheme: CodingScheme) -> MockMIMICIVDatasetSchemeConfig:
    return MockMIMICIVDatasetSchemeConfig(
        ethnicity=ethnicity_scheme.name,
        gender=gender_scheme.name,
        dx_discharge=dx_scheme.name,
        icu_procedures=icu_proc_scheme.name,
        icu_inputs=icu_inputs_scheme.name,
        obs=observation_scheme.name,
        hosp_procedures=hosp_proc_scheme.name)


@pytest.fixture
def mimiciv_dataset_config(mimiciv_dataset_scheme_config, dataset_tables_config):
    return DatasetConfig(scheme=mimiciv_dataset_scheme_config, tables=dataset_tables_config)


@pytest.fixture
def mimiciv_dataset_no_conv(dataset_scheme_manager, mimiciv_dataset_config, dataset_tables,
                            unit_converter_table) -> MockMIMICIVDataset:
    ds = MockMIMICIVDataset(scheme_manager=dataset_scheme_manager, config=mimiciv_dataset_config)
    with patch(__name__ + '.MockMIMICIVDatasetSchemeConfig.icu_inputs_uom_normalization_table',
               return_value=unit_converter_table,
               new_callable=PropertyMock):
        yield eqx.tree_at(lambda x: x.tables, ds, dataset_tables,
                          is_leaf=lambda x: x is None).execute_external_transformations([SetIndex(), CastTimestamps()])


@pytest.fixture
def mimiciv_dataset(dataset_scheme_manager, mimiciv_dataset_config, dataset_tables,
                    unit_converter_table) -> MockMIMICIVDataset:
    ds = MockMIMICIVDataset(scheme_manager=dataset_scheme_manager, config=mimiciv_dataset_config)
    with patch(__name__ + '.MockMIMICIVDatasetSchemeConfig.icu_inputs_uom_normalization_table',
               return_value=unit_converter_table,
               new_callable=PropertyMock):
        yield eqx.tree_at(lambda x: x.tables, ds, dataset_tables,
                          is_leaf=lambda x: x is None).execute_pipeline()


class NaiveEHR(TVxEHR):
    @classmethod
    def _setup_pipeline(cls, config: DatasetConfig) -> AbstractDatasetPipeline:
        return AbstractTVxPipeline(transformations=[])

    def __repr__(self):
        return 'NaiveEHR'


NaiveEHR.register()


@pytest.fixture
def tvx_ehr_config(tvx_ehr_scheme_config: TVxEHRSchemeConfig) -> TVxEHRConfig:
    return TVxEHRConfig(scheme=tvx_ehr_scheme_config, demographic=DemographicVectorConfig())


@pytest.fixture
def tvx_ehr(mimiciv_dataset: MockMIMICIVDataset, tvx_ehr_config: TVxEHRConfig) -> TVxEHR:
    return NaiveEHR(dataset=mimiciv_dataset, config=tvx_ehr_config)


@pytest.fixture
def hf5_writer_file(tmpdir) -> tb.File:
    with tb.open_file(tmpdir.join('test.h5'), 'w') as h5f:
        yield h5f


@pytest.fixture
def hf5_group(hf5_writer_file: tb.File) -> tb.Group:
    return hf5_writer_file.create_group('/', 'test')


@pytest.fixture
def hf5_reader_file(tmpdir) -> tb.File:
    with tb.open_file(tmpdir.join('test.h5'), 'r') as h5f:
        yield h5f


@pytest.fixture
def hf5_write_group(hf5_writer_file: tb.File) -> tb.Group:
    return hf5_writer_file.create_group('/', 'test')


LENGTH_OF_STAY = 10.0


def _singular_codevec(scheme: CodingScheme) -> CodesVector:
    return scheme.codeset2vec({random.choice(scheme.codes)})


@pytest.fixture
def gender(gender_scheme: CodingScheme) -> CodesVector:
    return _singular_codevec(gender_scheme)


@pytest.fixture
def ethnicity(ethnicity_scheme: CodingScheme) -> CodesVector:
    return _singular_codevec(ethnicity_scheme)


def date_of_birth() -> pd.Timestamp:
    return pd.to_datetime(pd.Timestamp('now') - pd.to_timedelta(nrand.randint(0, 100 * 365), unit='D'))


def demographic_vector_config() -> DemographicVectorConfig:
    flags = random.choices([True, False], k=3)
    return DemographicVectorConfig(*flags)


def _static_info(ethnicity: CodesVector, gender: CodesVector) -> StaticInfo:
    return StaticInfo(ethnicity=ethnicity, gender=gender,
                      date_of_birth=date_of_birth())


@pytest.fixture
def static_info(ethnicity: CodesVector, gender: CodesVector) -> StaticInfo:
    return _static_info(ethnicity, gender)


def _dx_codes(dx_scheme: CodingScheme):
    v = nrand.binomial(1, 0.5, size=len(dx_scheme)).astype(bool)
    return CodesVector(vec=v, scheme=dx_scheme.name)


@pytest.fixture
def dx_codes(dx_scheme: CodingScheme):
    return _dx_codes(dx_scheme)


def _dx_codes_history(dx_codes: CodesVector):
    v = nrand.binomial(1, 0.5, size=len(dx_codes)).astype(bool)
    return CodesVector(vec=v + dx_codes.vec, scheme=dx_codes.scheme)


@pytest.fixture
def dx_codes_history(dx_codes: CodesVector):
    return _dx_codes_history(dx_codes)


def _outcome(outcome_extractor_: OutcomeExtractor, dataset_scheme_manager: CodingSchemesManager,
             dx_codes: CodesVector):
    return outcome_extractor_.map_vector(dataset_scheme_manager, dx_codes)


@pytest.fixture
def outcome(dx_codes: CodesVector, outcome_extractor: OutcomeExtractor, dataset_scheme_manager):
    return _outcome(outcome_extractor, dataset_scheme_manager, dx_codes)


def _inpatient_observables(observation_scheme: CodingScheme, n_timestamps: int):
    d = len(observation_scheme)
    timestamps_grid = np.linspace(0, LENGTH_OF_STAY, 1000, dtype=np.float64)
    t = np.array(sorted(nrand.choice(timestamps_grid, replace=False, size=n_timestamps)))
    v = nrand.randn(n_timestamps, d)
    mask = nrand.binomial(1, 0.5, size=(n_timestamps, d)).astype(bool)
    return InpatientObservables(t, v, mask)


@pytest.fixture(params=[0, 1, 501])
def inpatient_observables(observation_scheme: CodingScheme, request):
    n_timestamps = request.param
    return _inpatient_observables(observation_scheme, n_timestamps)


def inpatient_binary_input(n: int, p: int):
    starttime = np.array(
        sorted(nrand.choice(np.linspace(0, LENGTH_OF_STAY, max(1000, n)), replace=False, size=n)))
    endtime = starttime + nrand.uniform(0, LENGTH_OF_STAY - starttime, size=(n,))
    code_index = nrand.choice(p, size=n, replace=True)
    return InpatientInput(starttime=starttime, endtime=endtime, code_index=code_index)


def inpatient_rated_input(n: int, p: int):
    bin_input = inpatient_binary_input(n, p)
    return InpatientInput(starttime=bin_input.starttime, endtime=bin_input.endtime, code_index=bin_input.code_index,
                          rate=nrand.uniform(0, 1, size=(n,)))


def _icu_inputs(icu_inputs_scheme: CodingScheme, n_timestamps: int):
    return inpatient_rated_input(n_timestamps, len(icu_inputs_scheme))


@pytest.fixture(params=[0, 1, 501])
def icu_inputs(icu_inputs_scheme: CodingScheme, request):
    return _icu_inputs(icu_inputs_scheme, request.param)


def _proc(scheme: CodingScheme, n_timestamps: int):
    return inpatient_binary_input(n_timestamps, len(scheme))


@pytest.fixture(params=[0, 1, 501])
def icu_proc(icu_proc_scheme: CodingScheme, request):
    return _proc(icu_proc_scheme, request.param)


@pytest.fixture(params=[0, 1, 501])
def hosp_proc(hosp_proc_scheme: CodingScheme, request):
    return _proc(hosp_proc_scheme, n_timestamps=request.param)


def _inpatient_interventions(hosp_proc, icu_proc, icu_inputs):
    return InpatientInterventions(hosp_proc, icu_proc, icu_inputs)


@pytest.fixture(params=[0, 1, 2, -1])
def inpatient_interventions(hosp_proc, icu_proc, icu_inputs, request):
    whoisnull = request.param
    return _inpatient_interventions(None if whoisnull == 0 else hosp_proc,
                                    None if whoisnull == 1 else icu_proc,
                                    None if whoisnull == 2 else icu_inputs)


def _segmented_inpatient_interventions(inpatient_interventions: InpatientInterventions, hosp_proc_scheme,
                                       icu_proc_scheme,
                                       icu_inputs_scheme,
                                       maximum_padding: int = 1) -> SegmentedInpatientInterventions:
    assert all(isinstance(scheme, CodingScheme) for scheme in [hosp_proc_scheme, icu_proc_scheme, icu_inputs_scheme])
    return SegmentedInpatientInterventions.from_interventions(inpatient_interventions, LENGTH_OF_STAY,
                                                              hosp_procedures_size=len(hosp_proc_scheme),
                                                              icu_procedures_size=len(icu_proc_scheme),
                                                              icu_inputs_size=len(icu_inputs_scheme),
                                                              maximum_padding=maximum_padding)


@pytest.fixture
def segmented_inpatient_interventions(inpatient_interventions: InpatientInterventions, hosp_proc_scheme,
                                      icu_proc_scheme,
                                      icu_inputs_scheme) -> SegmentedInpatientInterventions:
    return _segmented_inpatient_interventions(inpatient_interventions,
                                              hosp_proc_scheme=hosp_proc_scheme,
                                              icu_proc_scheme=icu_proc_scheme,
                                              icu_inputs_scheme=icu_inputs_scheme,
                                              maximum_padding=1)


def leading_observables_extractor(observation_scheme: NumericScheme,
                                  leading_hours: List[float] = (1.0,),
                                  entry_neglect_window: float = 0.0,
                                  recovery_window: float = 0.0,
                                  minimum_acquisitions: int = 0,
                                  code_index: int = BINARY_OBSERVATION_CODE_INDEX) -> LeadingObservableExtractor:
    config = LeadingObservableExtractorConfig(observable_code=observation_scheme.codes[code_index],
                                              scheme=observation_scheme.name,
                                              entry_neglect_window=entry_neglect_window,
                                              recovery_window=recovery_window,
                                              minimum_acquisitions=minimum_acquisitions,
                                              leading_hours=leading_hours)
    return LeadingObservableExtractor(config=config, observable_scheme=observation_scheme)


@pytest.fixture
def leading_observable(observation_scheme: NumericScheme,
                       inpatient_observables: InpatientObservables,
                       dataset_scheme_manager) -> InpatientObservables:
    return leading_observables_extractor(observation_scheme=observation_scheme)(inpatient_observables)


def _admission(admission_id: str, admission_date: pd.Timestamp,
               dx_codes: CodesVector,
               dx_codes_history: CodesVector, outcome: CodesVector, observables: InpatientObservables,
               interventions: InpatientInterventions, leading_observable: InpatientObservables):
    discharge_date = pd.to_datetime(admission_date + pd.to_timedelta(LENGTH_OF_STAY, unit='H'))

    return Admission(admission_id=admission_id, admission_dates=AdmissionDates(admission_date, discharge_date),
                     dx_codes=dx_codes,
                     dx_codes_history=dx_codes_history, outcome=outcome, observables=observables,
                     interventions=interventions, leading_observable=leading_observable)


@pytest.fixture
def admission(dx_codes: CodesVector, dx_codes_history: CodesVector,
              outcome: CodesVector, inpatient_observables: InpatientObservables,
              inpatient_interventions: InpatientInterventions,
              leading_observable: InpatientObservables) -> Admission:
    admission_id = 'test'
    return _admission(admission_id=admission_id, admission_date=pd.to_datetime('now'),
                      dx_codes=dx_codes, dx_codes_history=dx_codes_history, outcome=outcome,
                      observables=inpatient_observables, interventions=inpatient_interventions,
                      leading_observable=leading_observable)


@pytest.fixture
def segmented_admission(admission: Admission, icu_inputs_scheme: CodingScheme, icu_proc_scheme: CodingScheme,
                        hosp_proc_scheme: CodingScheme) -> SegmentedAdmission:
    return SegmentedAdmission.from_admission(admission=admission, maximum_padding=1,
                                             icu_inputs_size=len(icu_inputs_scheme),
                                             icu_procedures_size=len(icu_proc_scheme),
                                             hosp_procedures_size=len(hosp_proc_scheme))


@pytest.fixture
def segmented_patient(patient: Patient, icu_inputs_scheme: CodingScheme, icu_proc_scheme: CodingScheme,
                      hosp_proc_scheme: CodingScheme) -> SegmentedPatient:
    return SegmentedPatient.from_patient(patient=patient, maximum_padding=1,
                                         icu_inputs_size=len(icu_inputs_scheme),
                                         icu_procedures_size=len(icu_proc_scheme),
                                         hosp_procedures_size=len(hosp_proc_scheme))


def _admissions(n_admissions, dx_scheme: CodingScheme,
                outcome_extractor_: OutcomeExtractor, observation_scheme: NumericScheme,
                icu_inputs_scheme: CodingScheme, icu_proc_scheme: CodingScheme,
                hosp_proc_scheme: CodingScheme,
                dataset_scheme_manager: CodingSchemesManager) -> List[Admission]:
    admissions = []
    for i in range(n_admissions):
        dx_codes = _dx_codes(dx_scheme)
        obs = _inpatient_observables(observation_scheme, n_timestamps=nrand.randint(0, 100))
        lead = leading_observables_extractor(observation_scheme=observation_scheme)(obs)
        icu_proc = _proc(icu_proc_scheme, n_timestamps=nrand.randint(0, 50))
        hosp_proc = _proc(hosp_proc_scheme, n_timestamps=nrand.randint(0, 50))
        icu_inputs = _icu_inputs(icu_inputs_scheme, n_timestamps=nrand.randint(0, 50))

        admissions.append(_admission(admission_id=f'test_{i}', admission_date=pd.to_datetime('now'),
                                     dx_codes=dx_codes,
                                     dx_codes_history=_dx_codes_history(dx_codes),
                                     outcome=_outcome(outcome_extractor_, dataset_scheme_manager, dx_codes),
                                     observables=obs,
                                     interventions=_inpatient_interventions(hosp_proc=hosp_proc, icu_proc=icu_proc,
                                                                            icu_inputs=icu_inputs),
                                     leading_observable=lead))
    return admissions


@pytest.fixture(params=[0, 1, 50])
def patient(request, static_info: StaticInfo,
            dx_scheme: CodingScheme,
            outcome_extractor: OutcomeExtractor, observation_scheme: NumericScheme,
            icu_inputs_scheme: CodingScheme, icu_proc_scheme: CodingScheme,
            hosp_proc_scheme: CodingScheme,
            dataset_scheme_manager: CodingSchemesManager) -> List[Patient]:
    admissions = _admissions(n_admissions=request.param, dx_scheme=dx_scheme,
                             outcome_extractor_=outcome_extractor, observation_scheme=observation_scheme,
                             icu_inputs_scheme=icu_inputs_scheme, icu_proc_scheme=icu_proc_scheme,
                             hosp_proc_scheme=hosp_proc_scheme,
                             dataset_scheme_manager=dataset_scheme_manager)
    return Patient(subject_id='test', admissions=admissions, static_info=static_info)
