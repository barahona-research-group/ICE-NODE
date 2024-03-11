import random
from typing import List

import equinox as eqx
import numpy as np
import pandas as pd
import pytest

from lib.ehr import CodingScheme, FlatScheme, \
    CodingSchemeConfig, OutcomeExtractor
from lib.ehr.coding_scheme import ExcludingOutcomeExtractorConfig, ExcludingOutcomeExtractor, NumericScheme
from lib.ehr.dataset import StaticTableConfig, AdmissionTableConfig, AdmissionLinkedCodedValueTableConfig, \
    AdmissionIntervalBasedCodedTableConfig, RatedInputTableConfig, AdmissionTimestampedCodedValueTableConfig, \
    DatasetTablesConfig, DatasetSchemeConfig, DatasetTables, Dataset, DatasetConfig, AbstractDatasetPipeline, \
    AbstractDatasetPipelineConfig, DatasetScheme
from lib.ehr.transformations import SetIndex

DATASET_SCOPE = "function"


def scheme(name: str, codes: List[str]) -> str:
    CodingScheme.register_scheme(FlatScheme(CodingSchemeConfig(name), codes=codes,
                                            desc=dict(zip(codes, codes))))
    return name


def sample_codes(scheme: str, n: int) -> List[str]:
    codes = CodingScheme.from_name(scheme).codes
    return random.choices(codes, k=n)


# Static Columns
class StaticTableConfigFixture:
    @pytest.fixture(params=['SUBJECT_IDXYZ'])
    def subject_id_alias(self, request):
        return request.param

    @pytest.fixture(params=['race'])
    def race_alias(self, request):
        return request.param

    @pytest.fixture(params=['gender'])
    def gender_alias(self, request):
        return request.param

    @pytest.fixture(params=['dob'])
    def date_of_birth_alias(self, request):
        return request.param

    @pytest.fixture
    def static_table_config(self, subject_id_alias, gender_alias, race_alias, date_of_birth_alias):
        return StaticTableConfig(subject_id_alias=subject_id_alias,
                                 gender_alias=gender_alias, race_alias=race_alias,
                                 date_of_birth_alias=date_of_birth_alias)


class AdmissionsTableConfigFxiture:
    @pytest.fixture(params=['ADMISSION_IDX'])
    def admission_id_alias(self, request):
        return request.param

    @pytest.fixture(params=['admission_time'])
    def admission_time_alias(self, request):
        return request.param

    @pytest.fixture(params=['discharge_time'])
    def discharge_time_alias(self, request):
        return request.param

    @pytest.fixture
    def admission_table_config(self, static_table_config, admission_id_alias, admission_time_alias,
                               discharge_time_alias):
        return AdmissionTableConfig(subject_id_alias=static_table_config.subject_id_alias,
                                    admission_id_alias=admission_id_alias,
                                    admission_time_alias=admission_time_alias,
                                    discharge_time_alias=discharge_time_alias)


# Obs Columns
class ObsTableConfigFixture:
    @pytest.fixture(params=['time_bin'])
    def obs_time_alias(self, request):
        return request.param

    @pytest.fixture(params=['measurement'])
    def obs_code_alias(self, request):
        return request.param

    @pytest.fixture(params=['measurement description'])
    def obs_code_desc_alias(self, request):
        return request.param

    @pytest.fixture(params=['obs_val'])
    def obs_value_alias(self, request):
        return request.param

    @pytest.fixture
    def obs_table_config(self, admission_table_config, obs_time_alias, obs_code_alias,
                         obs_code_desc_alias,
                         obs_value_alias):
        return AdmissionTimestampedCodedValueTableConfig(admission_id_alias=admission_table_config.admission_id_alias,
                                                         time_alias=obs_time_alias,
                                                         code_alias=obs_code_alias,
                                                         description_alias=obs_code_desc_alias,
                                                         value_alias=obs_value_alias)


# Dx Columns
class DxTableConfigFixture:
    @pytest.fixture(params=['dx_code'])
    def dx_code_alias(self, request):
        return request.param

    @pytest.fixture(params=['dx_code_desc'])
    def dx_code_desc_alias(self, request):
        return request.param

    @pytest.fixture
    def dx_discharge_table_config(self, admission_table_config, dx_code_alias, dx_code_desc_alias):
        return AdmissionLinkedCodedValueTableConfig(admission_id_alias=admission_table_config.admission_id_alias,
                                                    code_alias=dx_code_alias, description_alias=dx_code_desc_alias)


# Hosp Proc Columns
class HospProcTableConfigFixture:
    @pytest.fixture(params=['hosp_proc_code'])
    def hosp_proc_code_alias(self, request):
        return request.param

    @pytest.fixture(params=['hosp_proc_code_desc'])
    def hosp_proc_code_desc_alias(self, request):
        return request.param

    @pytest.fixture(params=['hosp_proc_start_time'])
    def hosp_proc_start_time_alias(self, request):
        return request.param

    @pytest.fixture(params=['hosp_proc_end_time'])
    def hosp_proc_end_time_alias(self, request):
        return request.param

    @pytest.fixture
    def hosp_proc_table_config(self, admission_table_config, hosp_proc_code_alias, hosp_proc_code_desc_alias,
                               hosp_proc_start_time_alias, hosp_proc_end_time_alias):
        return AdmissionIntervalBasedCodedTableConfig(admission_id_alias=admission_table_config.admission_id_alias,
                                                      code_alias=hosp_proc_code_alias,
                                                      description_alias=hosp_proc_code_desc_alias,
                                                      start_time_alias=hosp_proc_start_time_alias,
                                                      end_time_alias=hosp_proc_end_time_alias)


# ICU Proc Columns
class ICUProcTableConfigFixture:
    @pytest.fixture(params=['icu_proc_code'])
    def icu_proc_code_alias(self, request):
        return request.param

    @pytest.fixture(params=['icu_proc_code_desc'])
    def icu_proc_code_desc_alias(self, request):
        return request.param

    @pytest.fixture(params=['icu_proc_start_time'])
    def icu_proc_start_time_alias(self, request):
        return request.param

    @pytest.fixture(params=['icu_proc_end_time'])
    def icu_proc_end_time_alias(self, request):
        return request.param

    @pytest.fixture
    def icu_proc_table_config(self, admission_table_config, icu_proc_code_alias, icu_proc_code_desc_alias,
                              icu_proc_start_time_alias, icu_proc_end_time_alias):
        return AdmissionIntervalBasedCodedTableConfig(admission_id_alias=admission_table_config.admission_id_alias,
                                                      code_alias=icu_proc_code_alias,
                                                      description_alias=icu_proc_code_desc_alias,
                                                      start_time_alias=icu_proc_start_time_alias,
                                                      end_time_alias=icu_proc_end_time_alias)


# ICU Input Columns
class ICUInputTableConfigFixture:
    @pytest.fixture(params=['icu_input_code'])
    def icu_input_code_alias(self, request):
        return request.param

    @pytest.fixture(params=['icu_input_code_desc'])
    def icu_input_code_desc_alias(self, request):
        return request.param

    @pytest.fixture(params=['icu_input_start_time'])
    def icu_input_start_time_alias(self, request):
        return request.param

    @pytest.fixture(params=['icu_input_end_time'])
    def icu_input_end_time_alias(self, request):
        return request.param

    @pytest.fixture(params=['icu_input_amount'])
    def icu_input_amount_alias(self, request):
        return request.param

    @pytest.fixture(params=['icu_input_amount_uom'])
    def icu_input_amount_unit_alias(self, request):
        return request.param

    @pytest.fixture(params=['icu_input_derived_normalized_amount'])
    def icu_input_derived_normalized_amount(self, request):
        return request.param

    @pytest.fixture(params=['icu_input_derived_normalized_amount_per_hour'])
    def icu_input_derived_normalized_amount_per_hour(self, request):
        return request.param

    @pytest.fixture(params=['icu_input_derived_unit_normalization_factor'])
    def icu_input_derived_unit_normalization_factor(self, request):
        return request.param

    @pytest.fixture(params=['icu_input_derived_universal_unit'])
    def icu_input_derived_universal_unit(self, request):
        return request.param

    @pytest.fixture
    def icu_input_table_config(self, admission_table_config: AdmissionTableConfig,
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
                              ethnicity_scheme: str, gender_scheme: str, n: int) -> pd.DataFrame:
    ethnicity_scheme = CodingScheme.from_name(ethnicity_scheme)
    gender_scheme = CodingScheme.from_name(gender_scheme)

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
    disch_dates = admit_dates + pd.to_timedelta(random.choices(range(1, 1000), k=n), unit='D')

    return pd.DataFrame({
        c_subject: random.choices(subjects_df[c_subject], k=n),
        c_admission: list(str(i) for i in range(n)),
        c_admission_time: admit_dates,
        c_discharge_time: disch_dates
    })


def sample_dx_dataframe(admissions_df: pd.DataFrame,
                        admission_table_config: AdmissionTableConfig,
                        dx_discharge_table_config: AdmissionLinkedCodedValueTableConfig,
                        dx_scheme: str,
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
                           scheme: str,
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
                                icu_input_scheme: str,
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
                         obs_scheme: str,
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

    scheme_object = CodingScheme.from_name(obs_scheme)
    assert isinstance(scheme_object, NumericScheme), 'Only numeric schemes are supported'
    df['obs_type'] = df[c_obs].map(scheme_object.type_hint)
    df.loc[df.obs_type == 'N', c_value] = np.random.uniform(low=0, high=1000, size=(df.obs_type == 'N').sum())
    df.loc[df.obs_type.isin(('C', 'O')), c_value] = random.choices([0, 1, 2],
                                                                   k=df.obs_type.isin(('C', 'O')).sum())
    df.loc[df.obs_type == 'B', c_value] = random.choices([0, 1], k=(df.obs_type == 'B').sum())

    return df[[c_admission, c_obs, c_time, c_value]]


BINARY_OBSERVATION_CODE_INDEX = 0
CATEGORICAL_OBSERVATION_CODE_INDEX = 1
ORDINAL_OBSERVATION_CODE_INDEX = 2
NUMERIC_OBSERVATION_CODE_INDEX = 3


class DatasetSchemeFixture:

    @pytest.fixture(params=[('ethnicity1', ['E1', 'E2', 'E3'])])
    def ethnicity_scheme_name(self, request):
        return scheme(*request.param)

    @pytest.fixture(params=[('gender1', ['M', 'F'])])
    def gender_scheme_name(self, request):
        return scheme(*request.param)

    @pytest.fixture(params=[('dx1', ['Dx1', 'Dx2', 'Dx3', 'Dx4', 'Dx5'])])
    def dx_scheme_name(self, request) -> str:
        return scheme(*request.param)

    @pytest.fixture(params=[('hosp_proc1', ['P1', 'P2', 'P3'])])
    def hosp_proc_scheme_name(self, request) -> str:
        return scheme(*request.param)

    @pytest.fixture(params=[('icu_proc1', ['P1', 'P2', 'P3'])])
    def icu_proc_scheme_name(self, request) -> str:
        return scheme(*request.param)

    @pytest.fixture(params=[('icu_inputs1', ['I1', 'I2', 'I3'])])
    def icu_inputs_scheme_name(self, request) -> str:
        return scheme(*request.param)

    @pytest.fixture(params=[('observation1',
                             ('O1', 'O2', 'O3', 'O4', 'O5'),
                             ('B', 'C', 'O', 'N', 'N'))])
    def observation_scheme_name(self, request) -> str:
        name, codes, types = request.param
        CodingScheme.register_scheme(NumericScheme(config=CodingSchemeConfig(name),
                                                   codes=codes,
                                                   desc=dict(zip(codes, codes)),
                                                   type_hint=dict(zip(codes, types))))
        return name

    @pytest.fixture
    def outcome_extractor_name(self, dx_scheme_name: str) -> str:
        name = f'{dx_scheme_name}_outcome'
        base_scheme = CodingScheme.from_name(dx_scheme_name)
        k = max(3, len(base_scheme.codes) - 1)
        random.seed(0)
        excluded = random.sample(base_scheme.codes, k=k)
        config = ExcludingOutcomeExtractorConfig(name=name,
                                                 base_scheme=base_scheme.name,
                                                 exclude_codes=excluded)
        outcome_extractor_scheme = ExcludingOutcomeExtractor(config)
        OutcomeExtractor.register_scheme(outcome_extractor_scheme)
        return name

    @pytest.fixture
    def gender_scheme(self, gender_scheme_name: str) -> CodingScheme:
        return CodingScheme.from_name(gender_scheme_name)

    @pytest.fixture
    def ethnicity_scheme(self, ethnicity_scheme_name: str) -> CodingScheme:
        return CodingScheme.from_name(ethnicity_scheme_name)

    @pytest.fixture
    def outcome_extractor(self, outcome_extractor_name: str) -> CodingScheme:
        return OutcomeExtractor.from_name(outcome_extractor_name)

    @pytest.fixture
    def obs_scheme(self, observation_scheme_name: str) -> CodingScheme:
        return CodingScheme.from_name(observation_scheme_name)

    @pytest.fixture
    def icu_inputs_scheme(self, icu_inputs_scheme_name: str) -> CodingScheme:
        return CodingScheme.from_name(icu_inputs_scheme_name)

    @pytest.fixture
    def icu_proc_scheme(self, icu_proc_scheme_name: str) -> CodingScheme:
        return CodingScheme.from_name(icu_proc_scheme_name)

    @pytest.fixture
    def hosp_proc_scheme(self, hosp_proc_scheme_name: str) -> CodingScheme:
        return CodingScheme.from_name(hosp_proc_scheme_name)

    @pytest.fixture
    def dataset_scheme_config(self, ethnicity_scheme_name: str,
                              gender_scheme_name: str,
                              dx_scheme_name: str,
                              icu_proc_scheme_name: str,
                              icu_inputs_scheme_name: str,
                              observation_scheme_name: str,
                              hosp_proc_scheme_name: str) -> DatasetSchemeConfig:
        return DatasetSchemeConfig(ethnicity=ethnicity_scheme_name,
                                   gender=gender_scheme_name,
                                   dx_discharge=dx_scheme_name,
                                   icu_procedures=icu_proc_scheme_name,
                                   icu_inputs=icu_inputs_scheme_name,
                                   obs=observation_scheme_name,
                                   hosp_procedures=hosp_proc_scheme_name)


class DatasetTablesFixture:
    @pytest.fixture
    def dataset_tables_config(self, static_table_config: StaticTableConfig,
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
                            (50, 0, 0), (50, 10, 10)],
                    ids=lambda x: f"_{x[0]}_subjects_{x[0] * x[1]}_admissions_{x[0] * x[1] * x[2]}_records")
    def dataset_tables(self, dataset_tables_config: DatasetTablesConfig,
                       dataset_scheme_config: DatasetSchemeConfig,
                       request) -> DatasetTables:
        n_subjects, n_admission_per_subject, n_per_admission = request.param
        subjects_df = sample_subjects_dataframe(dataset_tables_config.static,
                                                dataset_scheme_config.ethnicity,
                                                dataset_scheme_config.gender,
                                                n_subjects)
        admissions_df = sample_admissions_dataframe(subjects_df, dataset_tables_config.static,
                                                    dataset_tables_config.admissions,
                                                    n_admission_per_subject * n_subjects)
        dx_df = sample_dx_dataframe(admissions_df, dataset_tables_config.admissions,
                                    dataset_tables_config.dx_discharge,
                                    dataset_scheme_config.dx_discharge,
                                    n_per_admission * n_subjects * n_admission_per_subject)

        obs_df = sample_obs_dataframe(admissions_df, dataset_tables_config.admissions,
                                      dataset_tables_config.obs,
                                      dataset_scheme_config.obs,
                                      n_per_admission * n_subjects * n_admission_per_subject)

        icu_proc_df = _sample_proc_dataframe(admissions_df, dataset_tables_config.admissions,
                                             dataset_tables_config.icu_procedures,
                                             dataset_scheme_config.icu_procedures,
                                             n_per_admission * n_subjects * n_admission_per_subject)

        hosp_proc_df = _sample_proc_dataframe(admissions_df, dataset_tables_config.admissions,
                                              dataset_tables_config.hosp_procedures,
                                              dataset_scheme_config.hosp_procedures,
                                              n_per_admission * n_subjects * n_admission_per_subject)

        icu_inputs_df = sample_icu_inputs_dataframe(admissions_df, dataset_tables_config.admissions,
                                                    dataset_tables_config.icu_inputs,
                                                    dataset_scheme_config.icu_inputs,
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
        return AbstractDatasetPipeline(config=config.pipeline, transformations=[])

    @classmethod
    def load_tables(cls, config: DatasetConfig, scheme: DatasetScheme) -> DatasetTables:
        return None


class DatasetFixture:
    @pytest.fixture
    def dataset_config(self, dataset_scheme_config, dataset_tables_config):
        return DatasetConfig(scheme=dataset_scheme_config, tables=dataset_tables_config,
                             pipeline=AbstractDatasetPipelineConfig())

    @pytest.fixture(scope=DATASET_SCOPE)
    def dataset(self, dataset_config, dataset_tables):
        ds = NaiveDataset(config=dataset_config)
        return eqx.tree_at(lambda x: x.tables, ds, dataset_tables,
                           is_leaf=lambda x: x is None)

    @pytest.fixture(scope=DATASET_SCOPE)
    def indexed_dataset(self, dataset):
        return dataset.execute_external_transformations([SetIndex()])
