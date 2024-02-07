import random
from typing import List

import numpy as np
import pandas as pd
import pytest

from lib.ehr import CodingScheme, FlatScheme, \
    CodingSchemeConfig
from lib.ehr.dataset import StaticTableConfig, AdmissionTableConfig, AdmissionLinkedCodedValueTableConfig, \
    AdmissionIntervalBasedCodedTableConfig, RatedInputTableConfig, AdmissionTimestampedCodedValueTableConfig, \
    DatasetTablesConfig, DatasetSchemeConfig


def scheme(name: str, codes: List[str]) -> str:
    CodingScheme.register_scheme(FlatScheme(CodingSchemeConfig(name), codes=codes,
                                            desc=dict(zip(codes, codes))))
    return name


def sample_codes(scheme: str, n: int) -> List[str]:
    codes = CodingScheme.from_name(scheme).codes
    return random.choices(codes, k=n)


# Static Columns

@pytest.fixture(scope="module", params=['subject_id', 'SUBJECT_ID'])
def subject_id_alias(alias):
    return alias


@pytest.fixture(scope="module", params=['race'])
def race_alias(alias):
    return alias


@pytest.fixture(scope="module", params=['gender'])
def gender_alias(alias):
    return alias


@pytest.fixture(scope="module", params=['dob'])
def date_of_birth_alias(alias):
    return alias


@pytest.fixture(scope="module")
def static_table_config(subject_id_alias, gender_alias, race_alias, date_of_birth_alias):
    return StaticTableConfig(subject_id_alias=subject_id_alias,
                             gender_alias=gender_alias, race_alias=race_alias,
                             date_of_birth_alias=date_of_birth_alias)


@pytest.fixture(scope="module", params=['hadm_id', 'ADMISSION_ID'])
def admission_id_alias(alias):
    return alias


# Admission Columns
@pytest.fixture(scope="module", params=['admission_time'])
def admission_time_alias(alias):
    return alias


@pytest.fixture(scope="module", params=['discharge_time'])
def discharge_time_alias(alias):
    return alias


@pytest.fixture(scope="module")
def admission_table_config(static_table_config, admission_id_alias, admission_time_alias, discharge_time_alias):
    return AdmissionTableConfig(subject_id_alias=static_table_config.subject_id_alias,
                                admission_id_alias=admission_id_alias,
                                admission_time_alias=admission_time_alias,
                                discharge_time_alias=discharge_time_alias)


# Obs Columns
@pytest.fixture(scope="module", params=['time_bin'])
def obs_time_alias(alias):
    return alias


@pytest.fixture(scope="module", params=['measurement'])
def obs_code_alias(alias):
    return alias


@pytest.fixture(scope="module", params=['measurement description'])
def obs_code_desc_alias(alias):
    return alias


@pytest.fixture(scope="module", params=['obs_val'])
def obs_value_alias(alias):
    return alias


@pytest.fixture(scope="module")
def obs_table_config(admission_table_config, obs_time_alias, obs_code_alias, obs_value_alias):
    return AdmissionLinkedCodedValueTableConfig(admission_id_alias=admission_table_config.admission_id_alias,
                                                time_alias=obs_time_alias,
                                                code_alias=obs_code_alias,
                                                description_alias=obs_code_desc_alias,
                                                value_alias=obs_value_alias)


# Dx Columns
@pytest.fixture(scope="module", params=['dx_code'])
def dx_code_alias(alias):
    return alias


@pytest.fixture(scope="module", params=['dx_code_desc'])
def dx_code_desc_alias(alias):
    return alias


@pytest.fixture(scope="module")
def dx_discharge_table_config(admission_table_config,
                              dx_code_alias, dx_code_desc_alias):
    return AdmissionLinkedCodedValueTableConfig(admission_id_alias=admission_table_config.admission_id_alias,
                                                code_alias=dx_code_alias, description_alias=dx_code_desc_alias)


# Hosp Proc Columns
@pytest.fixture(scope="module", params=['hosp_proc_code'])
def hosp_proc_code_alias(alias):
    return alias


@pytest.fixture(scope="module", params=['hosp_proc_code_desc'])
def hosp_proc_code_desc_alias(alias):
    return alias


@pytest.fixture(scope="module", params=['hosp_proc_start_time'])
def hosp_proc_start_time_alias(alias):
    return alias


@pytest.fixture(scope="module", params=['hosp_proc_end_time'])
def hosp_proc_end_time_alias(alias):
    return alias


@pytest.fixture(scope="module")
def hosp_proc_table_config(admission_table_config: AdmissionTableConfig,
                           hosp_proc_code_alias: str,
                           hosp_proc_code_desc_alias: str, hosp_proc_start_time_alias: str,
                           hosp_proc_end_time_alias: str) -> AdmissionIntervalBasedCodedTableConfig:
    return AdmissionIntervalBasedCodedTableConfig(admission_id_alias=admission_table_config.admission_id_alias,
                                                  code_alias=hosp_proc_code_alias,
                                                  description_alias=hosp_proc_code_desc_alias,
                                                  start_time_alias=hosp_proc_start_time_alias,
                                                  end_time_alias=hosp_proc_end_time_alias)


# ICU Proc Columns
@pytest.fixture(scope="module", params=['icu_proc_code'])
def icu_proc_code_alias(alias):
    return alias


@pytest.fixture(scope="module", params=['icu_proc_code_desc'])
def icu_proc_code_desc_alias(alias):
    return alias


@pytest.fixture(scope="module", params=['icu_proc_start_time'])
def icu_proc_start_time_alias(alias):
    return alias


@pytest.fixture(scope="module", params=['icu_proc_end_time'])
def icu_proc_end_time_alias(alias):
    return alias


@pytest.fixture(scope="module")
def icu_proc_table_config(admission_table_config: AdmissionTableConfig,
                          icu_proc_code_alias: str,
                          icu_proc_code_desc_alias: str, icu_proc_start_time_alias: str,
                          icu_proc_end_time_alias: str) -> AdmissionIntervalBasedCodedTableConfig:
    return AdmissionIntervalBasedCodedTableConfig(admission_id_alias=admission_table_config.admission_id_alias,
                                                  code_alias=icu_proc_code_alias,
                                                  description_alias=icu_proc_code_desc_alias,
                                                  start_time_alias=icu_proc_start_time_alias,
                                                  end_time_alias=icu_proc_end_time_alias)


# ICU Input Columns

@pytest.fixture(scope="module", params=['icu_input_code'])
def icu_input_code_alias(alias):
    return alias


@pytest.fixture(scope="module", params=['icu_input_code_desc'])
def icu_input_code_desc_alias(alias):
    return alias


@pytest.fixture(scope="module", params=['icu_input_start_time'])
def icu_input_start_time_alias(alias):
    return alias


@pytest.fixture(scope="module", params=['icu_input_end_time'])
def icu_input_end_time_alias(alias):
    return alias


@pytest.fixture(scope="module", params=['icu_input_amount'])
def icu_input_amount_alias(alias):
    return alias


@pytest.fixture(scope="module", params=['icu_input_amount_uom'])
def icu_input_amount_unit_alias(alias):
    return alias


@pytest.fixture(scope="module", params=['icu_input_derived_amount_per_hour'])
def icu_input_derived_amount_per_hour(alias):
    return alias


@pytest.fixture(scope="module", params=['icu_input_derived_normalized_amount_per_hour'])
def icu_input_derived_normalized_amount_per_hour(alias):
    return alias


@pytest.fixture(scope="module", params=['icu_input_derived_unit_normalization_factor'])
def icu_input_derived_unit_normalization_factor(alias):
    return alias


@pytest.fixture(scope="module", params=['icu_input_derived_universal_unit'])
def icu_input_derived_universal_unit(alias):
    return alias


@pytest.fixture(scope="module")
def icu_input_table_config(admission_table_config: AdmissionTableConfig,
                           icu_input_code_alias: str,
                           icu_input_code_desc_alias: str, icu_input_start_time_alias: str,
                           icu_input_end_time_alias: str, icu_input_amount_alias: str, icu_input_amount_unit_alias: str,
                           icu_input_derived_amount_per_hour: str, icu_input_derived_normalized_amount_per_hour: str,
                           icu_input_derived_unit_normalization_factor: str,
                           icu_input_derived_universal_unit: str) -> AdmissionIntervalBasedCodedTableConfig:
    return AdmissionIntervalBasedCodedTableConfig(admission_id_alias=admission_table_config.admission_id_alias,
                                                  code_alias=icu_input_code_alias,
                                                  description_alias=icu_input_code_desc_alias,
                                                  start_time_alias=icu_input_start_time_alias,
                                                  end_time_alias=icu_input_end_time_alias,
                                                  value_alias=icu_input_amount_alias,
                                                  unit_alias=icu_input_amount_unit_alias,
                                                  derived_amount_per_hour_alias=icu_input_derived_amount_per_hour,
                                                  derived_normalized_amount_per_hour_alias=icu_input_derived_normalized_amount_per_hour,
                                                  derived_unit_normalization_factor_alias=icu_input_derived_unit_normalization_factor,
                                                  derived_universal_unit_alias=icu_input_derived_universal_unit)


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
    df = pd.merge(df, admissions_df[[c_admission, c_admittime, c_dischtime]], on=c_admission)
    df['los'] = (df[c_dischtime] - df[c_admittime]).dt.total_seconds() / 3600

    relative_start = np.random.uniform(0, df['los'].values, size=n)
    df[c_start] = df[c_admittime] + pd.to_timedelta(relative_start, unit='H')

    relative_end = np.random.uniform(low=df[c_start].values,
                                     high=(df[c_dischtime] - df[c_start]).values, size=n)

    df[c_end] = df[c_start] + pd.to_timedelta(relative_end, unit='H')
    return df[[c_admission, c_code, c_start, c_end]]


def sample_icu_inputs_dataframe(admissions_df: pd.DataFrame,
                                admission_table_config: AdmissionTableConfig,
                                table_config: RatedInputTableConfig,
                                icu_input_scheme: str,
                                n: int) -> pd.DataFrame:
    df = _sample_proc_dataframe(admissions_df, admission_table_config, table_config, icu_input_scheme, n)
    c_amount = table_config.amount_alias
    c_unit = table_config.amount_unit_alias
    c_derived_amount_per_hour = table_config.derived_amount_per_hour
    c_derived_normalized_amount_per_hour = table_config.derived_normalized_amount_per_hour
    c_derived_unit_normalization_factor = table_config.derived_unit_normalization_factor
    c_derived_universal_unit = table_config.derived_universal_unit

    df[c_amount] = np.random.uniform(low=0, high=1000, size=n)
    df[c_unit] = random.choices(['mg', 'g', 'kg'], k=n)
    df[c_derived_amount_per_hour] = np.random.uniform(low=0, high=1000, size=n)
    df[c_derived_normalized_amount_per_hour] = np.random.uniform(low=0, high=1000, size=n)
    df[c_derived_unit_normalization_factor] = np.random.uniform(low=0, high=1000, size=n)
    df[c_derived_universal_unit] = random.choices(['mg', 'g', 'kg'], k=n)
    return df


def sample_obs_table_dataframe(admissions_df: pd.DataFrame,
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
    df = pd.merge(df, admissions_df[[c_admission, c_admittime, c_dischtime]], on=c_admission)
    df['los'] = (df[c_dischtime] - df[c_admittime]).dt.total_seconds() / 3600
    relative_time = np.random.uniform(0, df['los'].values, size=n)
    df[c_time] = df[c_admittime] + pd.to_timedelta(relative_time, unit='H')
    df[c_value] = np.random.uniform(low=0, high=1000, size=n)
    return df[[c_admission, c_obs, c_time, c_value]]


@pytest.fixture(scope="module", params=[('ethnicity1', ['E1', 'E2', 'E3'])])
def ethnicity_scheme(name, codes):
    return scheme(name, codes)


@pytest.fixture(scope="module", params=[('gender1', ['M', 'F'])])
def gender_scheme(name, codes):
    return scheme(name, codes)


@pytest.fixture(scope="module", params=[('dx1', ['Dx1', 'Dx2', 'Dx3'])])
def dx_scheme(name, codes):
    return scheme(name, codes)


@pytest.fixture(scope="module", params=[('hosp_proc1', ['P1', 'P2', 'P3'])])
def hosp_proc_scheme(name, codes):
    return scheme(name, codes)


@pytest.fixture(scope="module", params=[('icu_proc1', ['P1', 'P2', 'P3'])])
def icu_proc_scheme(name, codes):
    return scheme(name, codes)


@pytest.fixture(scope="module", params=[('icu_inputs1', ['I1', 'I2', 'I3'])])
def icu_inputs_scheme(name, codes):
    return scheme(name, codes)


@pytest.fixture(scope="module", params=[('observation1', ['O1', 'O2', 'O3'])])
def observation_scheme(name, codes):
    return scheme(name, codes)


@pytest.fixture(scope="module")
def dataset_table_config(static_table_config: StaticTableConfig,
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
                               hosp_proc=hosp_proc_table_config,
                               icu_proc=icu_proc_table_config,
                               icu_inputs=icu_input_table_config)


@pytest.fixture(scope="module")
def dataset_scheme_config(ethnicity_scheme: str,
                          gender_scheme: str,
                          dx_scheme: str,
                          icu_proc_scheme: str,
                          icu_inputs_scheme: str,
                          observation_scheme: str,
                          hosp_proc_scheme: str) -> DatasetSchemeConfig:
    return DatasetSchemeConfig(ethnicity=ethnicity_scheme,
                               gender=gender_scheme,
                               dx=dx_scheme,
                               icu_proc=icu_proc_scheme,
                               icu_inputs=icu_inputs_scheme,
                               observation=observation_scheme,
                               hosp_proc=hosp_proc_scheme)

@pytest.fixture(scope="module")