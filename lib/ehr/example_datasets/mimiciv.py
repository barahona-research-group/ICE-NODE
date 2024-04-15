"""."""
from __future__ import annotations

import os
import warnings
from dataclasses import field
from functools import cached_property
from typing import Dict, List, Optional, Iterable, Tuple, Final

import pandas as pd
import sqlalchemy
from sqlalchemy import Engine

from lib.base import Module, Config
from lib.ehr.coding_scheme import (CodingScheme, resources_dir, CodingSchemesManager, FrozenDict11)
from lib.ehr.dataset import (DatasetScheme, StaticTableConfig,
                             AdmissionTimestampedMultiColumnTableConfig, AdmissionIntervalBasedCodedTableConfig,
                             AdmissionTimestampedCodedValueTableConfig, AdmissionLinkedCodedValueTableConfig,
                             TableConfig, CodedTableConfig, AdmissionTableConfig,
                             RatedInputTableConfig, DatasetTablesConfig,
                             DatasetTables, DatasetConfig, DatasetSchemeConfig, Dataset, AbstractDatasetPipelineConfig)
from lib.ehr.dataset import SECONDS_TO_HOURS_SCALER
from lib.ehr.example_schemes.icd import setup_standard_icd_ccs, CCSICDSchemeSelection, CCSICDOutcomeSelection
from lib.ehr.example_schemes.mimic import MixedICDScheme, AggregatedICUInputsScheme, ObservableMIMICScheme
from lib.utils import tqdm_constructor

warnings.filterwarnings('error',
                        category=RuntimeWarning,
                        message=r'overflow encountered in cast')


class SQLTableConfig(TableConfig):
    # resource file.
    # TODO: Add an attribute for the version using the last relevant git commit hash.
    query_template: Optional[str] = None

    @property
    def query(self) -> str:
        return open(resources_dir(self.query_template), "r").read()


class CodedSQLTableConfig(SQLTableConfig, CodedTableConfig):
    space_query_template: Optional[str] = None

    @property
    def space_query(self) -> str:
        return open(resources_dir(self.space_query_template), "r").read()


class AdmissionTimestampedMultiColumnSQLTableConfig(SQLTableConfig,
                                                    AdmissionTimestampedMultiColumnTableConfig):
    admission_id_alias: str = 'admission_id'
    time_alias: str = 'time'


class AdmissionTimestampedCodedValueSQLTableConfig(CodedSQLTableConfig, AdmissionTimestampedCodedValueTableConfig):
    components: List[AdmissionTimestampedMultiColumnSQLTableConfig] = field(kw_only=True)
    admission_id_alias: str = 'admission_id'
    time_alias: str = 'time'
    value_alias: str = 'value'
    code_alias: str = 'code'
    description_alias: str = 'description'


class AdmissionSQLTableConfig(SQLTableConfig, AdmissionTableConfig):
    admission_id_alias: str = 'admission_id'
    subject_id_alias: str = 'subject_id'
    admission_time_alias: str = 'admission_time'
    discharge_time_alias: str = 'discharge_time'


class StaticSQLTableConfig(SQLTableConfig, StaticTableConfig):
    gender_space_query_template: str = field(kw_only=True)
    race_space_query_template: str = field(kw_only=True)
    anchor_year_alias: str = 'anchor_year'
    anchor_age_alias: str = 'anchor_age'
    subject_id_alias: str = 'subject_id'
    gender_alias: str = 'gender'
    date_of_birth_alias: str = 'date_of_birth'
    race_alias: str = 'race'

    @property
    def gender_space_query(self) -> str:
        return open(resources_dir(self.gender_space_query_template), "r").read()

    @property
    def race_space_query(self) -> str:
        return open(resources_dir(self.race_space_query_template), "r").read()


class AdmissionMixedICDSQLTableConfig(AdmissionLinkedCodedValueTableConfig, CodedSQLTableConfig):
    admission_id_alias: str = 'admission_id'
    code_alias: str = 'code'
    description_alias: str = 'description'
    icd_code_alias: str = 'icd_code'
    icd_version_alias: str = 'icd_version'


class IntervalICUProcedureSQLTableConfig(CodedSQLTableConfig, AdmissionIntervalBasedCodedTableConfig):
    admission_id_alias: str = 'admission_id'
    code_alias: str = 'code'
    description_alias: str = 'description'
    start_time_alias: str = 'start_time'
    end_time_alias: str = 'end_time'


class RatedInputSQLTableConfig(CodedSQLTableConfig, RatedInputTableConfig):
    admission_id_alias: str = 'admission_id'
    amount_alias: str = 'amount'
    amount_unit_alias: str = 'amount_unit'
    start_time_alias: str = 'start_time'
    end_time_alias: str = 'end_time'
    code_alias: str = 'code'
    description_alias: str = 'description'
    derived_normalized_amount: str = 'derived_normalized_amount'
    derived_universal_unit: str = 'derived_universal_unit'
    derived_unit_normalization_factor: str = 'derived_unit_normalization_factor'
    derived_normalized_amount_per_hour: str = 'derived_normalized_amount_per_hour'


class AdmissionIntervalBasedMixedICDTableConfig(AdmissionMixedICDSQLTableConfig,
                                                AdmissionIntervalBasedCodedTableConfig):
    start_time_alias: str = 'start_time'
    end_time_alias: str = 'end_time'


class SQLTable(Module):
    config: SQLTableConfig

    # TODO: Document this class.
    @staticmethod
    def _coerce_columns_to_str(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
        # TODO: test this method.
        is_str = pd.api.types.is_string_dtype
        # coerce to integers then fix as strings.
        int_dtypes = {k: int for k in columns if k in df.columns if not is_str(df.dtypes[k])}
        str_dtypes = {k: str for k in columns if k in df.columns if not is_str(df.dtypes[k])}
        return df.astype(int_dtypes).astype(str_dtypes)

    def _coerce_id_to_str(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Some of the integer ids in the database when downloaded are stored as floats.
        A fix is to coerce them to integers then fix as strings.
        """
        return self._coerce_columns_to_str(df, self.config.alias_id_dict.values())

    def _coerce_code_to_str(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Some of the integer codes in the database when downloaded are stored as floats or integers.
        A fix is to coerce them to integers then fix as strings.
        """
        return self._coerce_columns_to_str(df, self.config.coded_cols)

    def process_table_types(self, table):
        return self._coerce_id_to_str(self._coerce_id_to_str(table))

    def __call__(self, engine: Engine):
        query = self.config.query.format(**self.config.alias_dict)
        return self.process_table_types(pd.read_sql(query, engine,
                                                    coerce_float=False))


class CategoricalSQLTable(SQLTable):
    # TODO: Document this class.

    config: CodedSQLTableConfig

    def process_table_types(self, table):
        return self._coerce_code_to_str(self._coerce_id_to_str(table))

    def space(self, engine: Engine):
        query = self.config.space_query.format(**self.config.alias_dict)
        return self.process_table_types(pd.read_sql(query, engine, coerce_float=False))


class ObservablesSQLTable(SQLTable):
    # TODO: Document this class.mimi

    config: AdmissionTimestampedCodedValueSQLTableConfig

    def melted_table(self, engine: Engine, table_name: str, attribute2code: Dict[str, str]) -> pd.DataFrame:
        c_code = self.config.code_alias
        c_value = self.config.value_alias
        attributes = list(attribute2code.keys())
        # download the table.
        table = self.table_interface(str(table_name))
        obs_df = table(engine, attributes)
        # melt the table. (admission_id, time, attribute, value)
        melted_obs_df = obs_df.melt(id_vars=[table.config.admission_id_alias, table.config.time_alias],
                                    var_name='attribute', value_name=c_value, value_vars=attributes)
        # add the code. (admission_id, time, attribute, value, code)
        melted_obs_df[c_code] = melted_obs_df['attribute'].map(attribute2code)
        melted_obs_df = melted_obs_df[melted_obs_df.value.notnull()]
        # drop the attribute. (admission_id, time, value, code)
        return melted_obs_df.drop('attribute', axis=1)

    def __call__(self, engine: Engine, obs_scheme: ObservableMIMICScheme) -> pd.DataFrame:
        # TODO: test this method with a mock engine.
        dfs = []
        for table_name, attrs_df in obs_scheme.as_dataframe().groupby('table_name'):
            attr2code = attrs_df.set_index('attribute')['code'].to_dict()
            dfs.append(self.melted_table(engine, table_name, attr2code))
        return pd.concat(dfs, ignore_index=True)

    def _time_stat(self, code_table: pd.DataFrame) -> pd.Series:
        c_admission_id = self.config.admission_id_alias
        c_time = self.config.time_alias
        timestamps = code_table[[c_admission_id, c_time]].sort_values([c_admission_id, c_time])
        time_deltas = (timestamps[c_time].diff().dt.total_seconds() * SECONDS_TO_HOURS_SCALER).iloc[1:]
        in_admission = pd.Series(timestamps[c_admission_id] == timestamps[c_admission_id].shift()).iloc[1:]
        time_deltas_stats = time_deltas[in_admission].describe()
        return time_deltas_stats.rename(index={k: f'time_delta_{k}' for k in time_deltas_stats.index})

    def _stats(self, code: str, code_table: pd.DataFrame) -> pd.DataFrame:
        values = code_table[self.config.value_alias]
        stats = values.describe()
        stats['nunique'] = values.nunique()
        time_stats = self._time_stat(code_table)
        stats = pd.concat([stats, time_stats])
        stats = stats.to_frame().T
        stats[self.config.code_alias] = code

        return stats.set_index(self.config.code_alias)

    def gen_melted_tables(self, engine: Engine, obs_scheme: ObservableMIMICScheme):
        for table_name, attrs_df in tqdm_constructor(obs_scheme.as_dataframe().groupby('table_name'), leave=False):
            attr2code = attrs_df.set_index('attribute')['code'].to_dict()
            yield self.melted_table(engine, table_name, attr2code)

    def stats(self, engine: Engine, obs_scheme: ObservableMIMICScheme) -> pd.DataFrame:
        dfs = []
        for table in self.gen_melted_tables(engine, obs_scheme):
            for code, code_table in table.groupby(self.config.code_alias):
                dfs.append(self._stats(code, code_table))
        return pd.concat(dfs, axis=0)

    def space(self, engine: Engine) -> pd.DataFrame:
        df_list = [self.table_interface(c.name).space(engine) for c in self.config.components]
        return pd.concat(df_list).sort_values(['attribute']).sort_index()

    def table_interface(self, table_name: str) -> TimestampedMultiColumnSQLTable:
        table_conf = next(t for t in self.config.components if t.name == table_name)
        return TimestampedMultiColumnSQLTable(config=table_conf)

    def register_scheme(self, manager: CodingSchemesManager,
                        name: str,
                        engine: Engine, attributes_selection: Optional[pd.DataFrame]) -> CodingSchemesManager:
        supported_space = self.space(engine)
        if attributes_selection is None:
            attributes_selection = supported_space
        else:
            if attributes_selection.index.name != 'table_name':
                attributes_selection = attributes_selection.set_index('table_name', drop=True)
            if 'type' not in attributes_selection.columns:
                attributes_selection = pd.merge(attributes_selection, supported_space,
                                                left_on=['table_name', 'attribute'],
                                                right_on=['table_name', 'attribute'],
                                                suffixes=(None, '_y'),
                                                how='inner')

        return manager.add_scheme(ObservableMIMICScheme.from_selection(name, attributes_selection))


class MixedICDSQLTable(CategoricalSQLTable):
    # TODO: Document this class.

    config: AdmissionMixedICDSQLTableConfig

    @staticmethod
    def _register_scheme(manager: CodingSchemesManager,
                         name: str,
                         icd_version_schemes: FrozenDict11,
                         supported_space: pd.DataFrame,
                         icd_version_selection: Optional[pd.DataFrame],
                         c_version: str, c_code: str, c_desc: str) -> CodingSchemesManager:
        # TODO: test this method.
        if icd_version_selection is None:
            icd_version_selection = supported_space[[c_version, c_code, c_desc]].drop_duplicates()
            icd_version_selection = icd_version_selection.astype(str)
        else:
            if c_desc not in icd_version_selection.columns:
                icd_version_selection = pd.merge(icd_version_selection,
                                                 supported_space[[c_version, c_code, c_desc]],
                                                 on=[c_version, c_code], how='left')
            icd_version_selection = icd_version_selection[[c_version, c_code, c_desc]]
            icd_version_selection = icd_version_selection.drop_duplicates()
            icd_version_selection = icd_version_selection.astype(str)
            for version, codes in icd_version_selection.groupby(c_version):
                support_subset = supported_space[supported_space[c_version] == version]
                unsupported_codes = codes[~codes[c_code].isin(support_subset[c_code])]

                assert len(unsupported_codes) == 0, f'Codes {unsupported_codes} are not supported for version {version}'

        manager = manager.add_scheme(MixedICDScheme.from_selection(manager, name, icd_version_selection,
                                                                   icd_version_alias=c_version,
                                                                   icd_code_alias=c_code,
                                                                   description_alias=c_desc,
                                                                   icd_version_schemes=icd_version_schemes))
        scheme: MixedICDScheme = manager.scheme[name]
        return scheme.register_standard_icd_maps(manager)

    def register_scheme(self, manager: CodingSchemesManager,
                        name: str,
                        engine: Engine,
                        icd_version_schemes: FrozenDict11,
                        icd_version_selection: pd.DataFrame,
                        target_name: Optional[str],
                        c_target_code: Optional[str],
                        c_target_desc: Optional[str],
                        mapping: Optional[pd.DataFrame]) -> CodingSchemesManager:
        c_version = self.config.icd_version_alias
        c_code = self.config.icd_code_alias
        c_desc = self.config.description_alias
        manager = self._register_scheme(manager=manager,
                                        name=name, icd_version_schemes=icd_version_schemes,
                                        supported_space=self.space(engine),
                                        icd_version_selection=icd_version_selection,
                                        c_version=c_version, c_code=c_code, c_desc=c_desc)
        if target_name is not None and mapping is not None:
            mixed_icd_scheme: MixedICDScheme = manager.scheme[name]
            manager = mixed_icd_scheme.register_map(manager=manager, target_name=target_name, mapping=mapping,
                                                    c_code=c_code, c_icd_code=c_code, c_icd_version=c_version,
                                                    c_target_code=c_target_code,
                                                    c_target_desc=c_target_desc)
        return manager

    def _coerce_version_to_str(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Some of the integer codes in the database when downloaded are stored as floats or integers.
        A fix is to coerce them to integers then fix as strings.
        """
        return self._coerce_columns_to_str(df, (self.config.icd_version_alias,))

    def process_table_types(self, table):
        table = super().process_table_types(table)
        table[self.config.icd_code_alias] = table[self.config.icd_code_alias].str.strip()
        return self._coerce_version_to_str(table)


class CodedSQLTable(CategoricalSQLTable):
    # TODO: Document this class.

    config: CodedSQLTableConfig

    def register_scheme(self, manager: CodingSchemesManager, name: str,
                        engine: Engine, code_selection: Optional[pd.DataFrame]) -> CodingSchemesManager:
        return manager.register_scheme_from_selection(name=name, supported_space=self.space(engine),
                                                      code_selection=code_selection, c_code=self.config.code_alias,
                                                      c_desc=self.config.description_alias)


class TimestampedMultiColumnSQLTable(SQLTable):
    # TODO: Document this class.

    config: AdmissionTimestampedMultiColumnSQLTableConfig

    def _coerce_value_to_real(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Some of the values in the `value_alias` column are stored as strings.
        """
        types = {k: float for k in self.config.attributes if k in df.columns
                 and not pd.api.types.is_float_dtype(df.dtypes[k])}
        return df.astype(types)

    def process_table_types(self, table):
        return self._coerce_value_to_real(self._coerce_code_to_str(self._coerce_id_to_str(table)))

    def __call__(self, engine: Engine, attributes: List[str]) -> pd.DataFrame:
        assert len(set(attributes)) == len(attributes), f"Duplicate attributes {attributes}"
        assert all(a in self.config.attributes for a in attributes), \
            f"Some attributes {attributes} not in {self.config.attributes}"
        query = self.config.query.format(attributes=','.join(attributes),
                                         **self.config.alias_dict)
        return self.process_table_types(pd.read_sql(query, engine,
                                                    coerce_float=False))

    def space(self, engine: Engine):
        space = pd.DataFrame(
            [(self.config.name, a, self.config.type_hint[i]) for i, a in enumerate(self.config.attributes)],
            columns=['table_name', 'attribute', 'type_hint'])
        return space.set_index('table_name', drop=True).sort_values('attribute').sort_index()


class StaticSQLTable(SQLTable):
    # TODO: Document this class.

    config: StaticSQLTableConfig

    def __call__(self, engine: Engine):
        query = self.config.query.format(**self.config.alias_dict)
        df = self._coerce_id_to_str(pd.read_sql(query, engine,
                                                coerce_float=False))

        anchor_date = pd.to_datetime(df[self.config.anchor_year_alias],
                                     format='%Y').dt.normalize()
        anchor_age = df[self.config.anchor_age_alias].map(
            lambda y: pd.DateOffset(years=-y))
        df[self.config.date_of_birth_alias] = anchor_date + anchor_age

        return df

    def gender_space(self, engine: Engine):
        query = self.config.gender_space_query.format(**self.config.alias_dict)
        return pd.read_sql(query, engine)

    def ethnicity_space(self, engine: Engine):
        query = self.config.race_space_query.format(**self.config.alias_dict)
        return pd.read_sql(query, engine)

    def register_gender_scheme(self, manager: CodingSchemesManager,
                               scheme_config: MIMICIVDatasetSchemeConfig,
                               engine: Engine) -> CodingSchemesManager:

        manager = manager.register_scheme_from_selection(name=scheme_config.gender,
                                                         supported_space=self.gender_space(engine),
                                                         code_selection=scheme_config.gender_selection,
                                                         c_code=self.config.gender_alias,
                                                         c_desc=self.config.gender_alias)
        if scheme_config.gender_map is not None:
            manager = manager.register_target_scheme(
                scheme_config.gender,
                scheme_config.propose_target_scheme_name(scheme_config.gender),
                scheme_config.gender_map,
                c_code=self.config.gender_alias,
                c_target_code=scheme_config.target_column_name(self.config.gender_alias),
                c_target_desc=scheme_config.target_column_name(self.config.gender_alias))
        return manager

    def register_ethnicity_scheme(self, manager: CodingSchemesManager, scheme_config: MIMICIVDatasetSchemeConfig,
                                  engine: Engine) -> CodingSchemesManager:

        manager = manager.register_scheme_from_selection(name=scheme_config.ethnicity,
                                                         supported_space=self.ethnicity_space(engine),
                                                         code_selection=scheme_config.ethnicity_selection,
                                                         c_code=self.config.race_alias,
                                                         c_desc=self.config.race_alias)
        if scheme_config.ethnicity_map is not None:
            manager = manager.register_target_scheme(scheme_config.ethnicity,
                                                     scheme_config.propose_target_scheme_name(scheme_config.ethnicity),
                                                     scheme_config.ethnicity_map,
                                                     c_code=self.config.race_alias,
                                                     c_target_code=scheme_config.target_column_name(
                                                         self.config.race_alias),
                                                     c_target_desc=scheme_config.target_column_name(
                                                         self.config.race_alias))
        return manager


ENV_MIMICIV_HOST: Final[str] = 'MIMICIV_HOST'
ENV_MIMICIV_PORT: Final[str] = 'MIMICIV_PORT'
ENV_MIMICIV_USER: Final[str] = 'MIMICIV_USER'
ENV_MIMICIV_PASSWORD: Final[str] = 'MIMICIV_PASSWORD'
ENV_MIMICIV_DBNAME: Final[str] = 'MIMICIV_DBNAME'
ENV_MIMICIV_URL: Final[str] = 'MIMICIV_URL'


class MIMICIVSQLTablesConfig(DatasetTablesConfig):
    static: StaticSQLTableConfig = field(default_factory=lambda: STATIC_CONF, kw_only=True)
    admissions: AdmissionSQLTableConfig = field(default_factory=lambda: ADMISSIONS_CONF, kw_only=True)
    dx_discharge: AdmissionMixedICDSQLTableConfig = field(default_factory=lambda: DX_DISCHARGE_CONF, kw_only=True)
    obs: AdmissionTimestampedCodedValueSQLTableConfig = field(default_factory=lambda: OBS_TABLE_CONFIG,
                                                              kw_only=True)
    icu_procedures: IntervalICUProcedureSQLTableConfig = field(default_factory=lambda: ICU_PROCEDURES_CONF,
                                                               kw_only=True)
    icu_inputs: RatedInputSQLTableConfig = field(default_factory=lambda: ICU_INPUTS_CONF, kw_only=True)
    hosp_procedures: AdmissionIntervalBasedMixedICDTableConfig = field(default_factory=lambda: HOSP_PROCEDURES_CONF,
                                                                       kw_only=True)

    @staticmethod
    def url() -> str:
        if ENV_MIMICIV_URL in os.environ:
            return os.environ[ENV_MIMICIV_URL]
        elif all(e in os.environ for e in
                 [ENV_MIMICIV_USER, ENV_MIMICIV_PASSWORD, ENV_MIMICIV_HOST, ENV_MIMICIV_PORT, ENV_MIMICIV_DBNAME]):
            return MIMICIVSQLTablesConfig.url_from_credentials(
                user=os.environ[ENV_MIMICIV_USER],
                password=os.environ[ENV_MIMICIV_PASSWORD],
                host=os.environ[ENV_MIMICIV_HOST],
                port=os.environ[ENV_MIMICIV_PORT],
                dbname=os.environ[ENV_MIMICIV_DBNAME]
            )
        else:
            credentials_env_list = [ENV_MIMICIV_USER, ENV_MIMICIV_PASSWORD, ENV_MIMICIV_HOST, ENV_MIMICIV_PORT,
                                    ENV_MIMICIV_DBNAME]
            raise ValueError(f"Environment variables ({ENV_MIMICIV_URL}) or "
                             f"({', '.join(credentials_env_list)}) "
                             f"are not set.")

    @staticmethod
    def url_from_credentials(user: str, password: str, host: str, port: str, dbname: str) -> str:
        return f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}'


class DatasetSchemeMapsFiles(Config):
    gender: Optional[str] = 'gender.csv'
    ethnicity: Optional[str] = 'ethnicity.csv'
    icu_inputs: Optional[str] = 'icu_inputs.csv'
    icu_procedures: Optional[str] = 'icu_procedures.csv'
    hosp_procedures: Optional[str] = 'hosp_procedures.csv'


class DatasetSchemeSelectionFiles(Config):
    gender: Optional[str] = 'gender.csv'
    ethnicity: Optional[str] = 'ethnicity.csv'
    icu_inputs: Optional[str] = 'icu_inputs.csv'
    icu_procedures: Optional[str] = 'icu_procedures.csv'
    hosp_procedures: Optional[str] = 'hosp_procedures.csv'
    obs: Optional[str] = 'obs.csv'
    dx_discharge: Optional[str] = 'dx_discharge.csv'


class MIMICIVDatasetSchemeConfig(DatasetSchemeConfig):
    gender: str = 'fm_gender'
    ethnicity: str = 'ethnicity'
    dx_discharge: str = 'dx_mixed_icd'
    obs: str = 'obs'
    icu_inputs: str = 'icu_inputs'
    icu_procedures: str = 'icu_procedures'
    hosp_procedures: str = 'pr_mixed_icd'

    name_separator: str = '.'
    name_prefix: str = 'mimiciv'
    resources_dir: str = 'mimiciv'
    selection_subdir: str = 'selection'
    map_subdir: str = 'map'
    map_files: DatasetSchemeMapsFiles = field(default_factory=DatasetSchemeMapsFiles)
    selection_files: DatasetSchemeSelectionFiles = field(default_factory=DatasetSchemeSelectionFiles)
    icu_inputs_uom_normalization: Optional[Tuple[str]] = ("uom_normalization", "icu_inputs.csv")
    icu_inputs_aggregation_column: Optional[str] = 'aggregation'

    def __post_init__(self):
        for k in ('gender', 'ethnicity', 'dx_discharge', 'obs', 'icu_inputs', 'icu_procedures', 'hosp_procedures'):
            setattr(self, k, self._scheme_name(k))

    @property
    def icu_inputs_uom_normalization_table(self) -> pd.DataFrame:
        return pd.read_csv(resources_dir(self.resources_dir, *self.icu_inputs_uom_normalization)).astype(str)

    def print_expected_configuration_layout_disk(self):
        raise NotImplementedError()

    def generate_empty_configuration_layout_disk(self):
        raise NotImplementedError()

    def check_configuration_layout_disk(self):
        raise NotImplementedError()

    def selection_file(self, path: str) -> Optional[pd.DataFrame]:
        try:
            return pd.read_csv(resources_dir(self.resources_dir, self.selection_subdir, path)).astype(str)
        except FileNotFoundError:
            return None

    def map_file(self, path: str) -> Optional[pd.DataFrame]:
        try:
            return pd.read_csv(resources_dir(self.resources_dir, self.map_subdir, path)).astype(str)
        except FileNotFoundError:
            return None

    def _scheme_name(self, key: str) -> str:
        return f'{self.name_prefix}{self.name_separator}{key}'

    def propose_target_scheme_name(self, key: str) -> str:
        return self._scheme_name(f'target_{key}')

    def target_column_name(self, key: str) -> str:
        return f'target_{key}'

    @property
    def gender_selection(self) -> Optional[pd.DataFrame]:
        return self.selection_file(self.selection_files.gender)

    @property
    def ethnicity_selection(self) -> Optional[pd.DataFrame]:
        return self.selection_file(self.selection_files.ethnicity)

    @property
    def icu_inputs_selection(self) -> Optional[pd.DataFrame]:
        return self.selection_file(self.selection_files.icu_inputs)

    @property
    def icu_procedures_selection(self) -> Optional[pd.DataFrame]:
        return self.selection_file(self.selection_files.icu_procedures)

    @property
    def hosp_procedures_selection(self) -> Optional[pd.DataFrame]:
        return self.selection_file(self.selection_files.hosp_procedures)

    @property
    def obs_selection(self) -> Optional[pd.DataFrame]:
        df = self.selection_file(self.selection_files.obs)
        return df.set_index('table_name', drop=True).sort_values('attribute').sort_index()

    @property
    def dx_discharge_selection(self) -> Optional[pd.DataFrame]:
        return self.selection_file(self.selection_files.dx_discharge)

    @property
    def gender_map(self) -> Optional[pd.DataFrame]:
        return self.map_file(self.map_files.gender)

    @property
    def ethnicity_map(self) -> Optional[pd.DataFrame]:
        return self.map_file(self.map_files.ethnicity)

    @property
    def icu_inputs_map(self) -> Optional[pd.DataFrame]:
        return self.map_file(self.map_files.icu_inputs)

    @property
    def icu_procedures_map(self) -> Optional[pd.DataFrame]:
        return self.map_file(self.map_files.icu_procedures)

    @property
    def hosp_procedures_map(self) -> Optional[pd.DataFrame]:
        return self.map_file(self.map_files.hosp_procedures)


class MIMICIVSQLConfig(DatasetConfig):
    tables: MIMICIVSQLTablesConfig
    scheme: DatasetSchemeConfig


class MIMICIVSQLTablesInterface(Module):
    # TODO: Document this class.

    config: MIMICIVSQLTablesConfig

    def create_engine(self) -> Engine:
        return sqlalchemy.create_engine(self.config.url())

    def register_gender_scheme(self, manager: CodingSchemesManager,
                               config: MIMICIVDatasetSchemeConfig) -> CodingSchemesManager:
        """
        TODO: document me.
        """
        table = StaticSQLTable(self.config.static)
        return table.register_gender_scheme(manager, config, self.create_engine())

    def register_ethnicity_scheme(self, manager: CodingSchemesManager,
                                  config: MIMICIVDatasetSchemeConfig) -> CodingSchemesManager:
        """
        TODO: document me.
        """
        table = StaticSQLTable(self.config.static)
        return table.register_ethnicity_scheme(manager, config, self.create_engine())

    def register_obs_scheme(self, manager: CodingSchemesManager,
                            config: MIMICIVDatasetSchemeConfig) -> CodingSchemesManager:
        """
        From the given selection of observable variables `attributes_selection`, generate a new scheme
        that can be used to generate for vectorisation of the timestamped observations.

        Args:
            name : The name of the scheme.
            attributes_selection : A table containing the MIMICIV table names and their corresponding attributes.
                If None, all supported variables will be used.

        Returns:
            (CodingScheme.FlatScheme) A new scheme that is also registered in the current runtime.
        """
        table = ObservablesSQLTable(self.config.obs)
        return table.register_scheme(manager, config.obs,
                                     self.create_engine(),
                                     config.obs_selection)

    def register_icu_inputs_scheme(self, manager: CodingSchemesManager,
                                   config: MIMICIVDatasetSchemeConfig) -> CodingSchemesManager:
        """
        From the given selection of ICU input items `code_selection`, generate a new scheme
        that can be used to generate for vectorisation of the ICU inputs. If `code_selection` is None,
        all supported items will be used.

        Args:
            name : The name of the new scheme.
            code_selection : A dataframe containing the `code`s to generate the new scheme. If None, all supported items will be used.

        Returns:
            (CodingScheme.FlatScheme) A new scheme that is also registered in the current runtime.
        """

        code_selection = config.icu_inputs_selection
        mapping = config.icu_inputs_map
        c_aggregation = config.icu_inputs_aggregation_column
        table = CodedSQLTable(self.config.icu_inputs)
        manager = table.register_scheme(manager, config.icu_inputs, self.create_engine(), code_selection)
        if mapping is not None and c_aggregation is not None:
            source_scheme = manager.scheme[config.icu_inputs]
            manager = AggregatedICUInputsScheme.register_aggregated_scheme(
                manager=manager, scheme=source_scheme,
                target_scheme_name=config.propose_target_scheme_name(source_scheme.name),
                code_column=self.config.icu_inputs.code_alias,
                target_code_column=config.target_column_name(self.config.icu_inputs.code_alias),
                target_desc_column=config.target_column_name(self.config.icu_inputs.description_alias),
                target_aggregation_column=c_aggregation,
                mapping_table=mapping
            )
        return manager

    def register_icu_procedures_scheme(self, manager: CodingSchemesManager,
                                       config: MIMICIVDatasetSchemeConfig) -> CodingSchemesManager:
        """
        From the given selection of ICU procedure items `code_selection`, generate a new scheme
        that can be used to generate for vectorisation of the ICU procedures. If `code_selection` is None,
        all supported items will be used.

        Args:
            name : The name of the new scheme.
            code_selection : A dataframe containing the `code`s of choice. If None, all supported items will be used.

        Returns:
            (CodingScheme.FlatScheme) A new scheme that is also registered in the current runtime.
        """

        table = CodedSQLTable(self.config.icu_procedures)
        manager = table.register_scheme(manager, config.icu_procedures, self.create_engine(),
                                        config.icu_procedures_selection)
        mapping = config.icu_procedures_map
        if mapping is not None:
            c_target_code = config.target_column_name(self.config.icu_procedures.code_alias)
            c_target_desc = config.target_column_name(self.config.icu_procedures.description_alias)
            manager = manager.register_target_scheme(config.icu_procedures,
                                                     config.propose_target_scheme_name(config.icu_procedures), mapping,
                                                     c_code=self.config.icu_procedures.code_alias,
                                                     c_target_code=c_target_code,
                                                     c_target_desc=c_target_desc)
        return manager

    def register_hosp_procedures_scheme(self, manager: CodingSchemesManager,
                                        config: MIMICIVDatasetSchemeConfig) -> CodingSchemesManager:
        """
        From the given selection of hospital procedure items `icd_version_selection`, generate a new scheme.

        Args:
            name : The name of the new scheme.
            icd_version_selection : A dataframe containing the `icd_code`s to generate the new scheme. If None, all supported items will be used. The dataframe should have the following columns:
                - icd_version: The version of the ICD.
                - icd_code: The ICD code.

        Returns:
            (CodingScheme.FlatScheme) A new scheme that is also registered in the current runtime.
        """

        table = MixedICDSQLTable(self.config.hosp_procedures)

        icd_version_schemes = FrozenDict11.from_dict({'9': 'pr_icd9', '10': 'pr_flat_icd10'})
        return table.register_scheme(manager, name=config.hosp_procedures,
                                     engine=self.create_engine(),
                                     icd_version_schemes=icd_version_schemes,
                                     icd_version_selection=config.hosp_procedures_selection,
                                     target_name=config.propose_target_scheme_name(config.hosp_procedures),
                                     c_target_code=config.target_column_name(self.config.hosp_procedures.code_alias),
                                     c_target_desc=config.target_column_name(
                                         self.config.hosp_procedures.description_alias),
                                     mapping=config.hosp_procedures_map)

    def register_dx_discharge_scheme(self, manager: CodingSchemesManager,
                                     config: MIMICIVDatasetSchemeConfig) -> CodingSchemesManager:
        """
        From the given selection of discharge diagnosis items `icd_version_selection`, generate a new scheme.

        Args:
            name : The name of the new scheme.
            icd_version_selection : A dataframe containing the `icd_code`s to generate the new scheme. If None, all supported items will be used. The dataframe should have the following columns:
                - icd_version: The version of the ICD.
                - icd_code: The ICD code.

        Returns:
            (CodingScheme.FlatScheme) A new scheme that is also registered in the current runtime.
        """
        table = MixedICDSQLTable(self.config.dx_discharge)

        return table.register_scheme(manager, config.dx_discharge, self.create_engine(),
                                     FrozenDict11.from_dict({'9': 'dx_icd9', '10': 'dx_flat_icd10'}),
                                     config.dx_discharge_selection,
                                     target_name=None,
                                     c_target_code=None,
                                     c_target_desc=None,
                                     mapping=None)

    @cached_property
    def supported_gender(self) -> pd.DataFrame:
        table = StaticSQLTable(self.config.static)
        return table.gender_space(self.create_engine())

    @cached_property
    def supported_ethnicity(self) -> pd.DataFrame:
        table = StaticSQLTable(self.config.static)
        return table.ethnicity_space(self.create_engine())

    def obs_stats(self, scheme: ObservableMIMICScheme) -> pd.DataFrame:
        table = ObservablesSQLTable(self.config.obs)
        return table.stats(self.create_engine(), scheme)

    @cached_property
    def supported_obs_variables(self) -> pd.DataFrame:
        table = ObservablesSQLTable(self.config.obs)
        return table.space(self.create_engine())

    @cached_property
    def supported_icu_procedures(self) -> pd.DataFrame:
        table = CodedSQLTable(self.config.icu_procedures)
        return table.space(self.create_engine())

    @cached_property
    def supported_icu_inputs(self) -> pd.DataFrame:
        table = CodedSQLTable(self.config.icu_inputs)
        return table.space(self.create_engine())

    @cached_property
    def supported_hosp_procedures(self) -> pd.DataFrame:
        table = MixedICDSQLTable(self.config.hosp_procedures)
        return table.space(self.create_engine())

    @cached_property
    def supported_dx_discharge(self) -> pd.DataFrame:
        table = MixedICDSQLTable(self.config.dx_discharge)
        return table.space(self.create_engine())

    def _extract_static_table(self, engine: Engine) -> pd.DataFrame:
        table = StaticSQLTable(self.config.static)
        return table(engine)

    def _extract_admissions_table(self, engine: Engine) -> pd.DataFrame:
        table = SQLTable(self.config.admissions)
        return table(engine)

    def _extract_dx_discharge_table(self, engine: Engine, dx_discharge_scheme: MixedICDScheme) -> pd.DataFrame:
        table = MixedICDSQLTable(self.config.dx_discharge)
        dataframe = table(engine)
        c_icd_version = self.config.dx_discharge.icd_version_alias
        c_icd_code = self.config.dx_discharge.icd_code_alias
        return dx_discharge_scheme.mixedcode_format_table(dataframe, c_icd_code, c_icd_version,
                                                          table.config.code_alias)

    def _extract_obs_table(self, engine: Engine, obs_scheme: ObservableMIMICScheme) -> pd.DataFrame:
        table = ObservablesSQLTable(self.config.obs)
        return table(engine, obs_scheme)

    def _extract_icu_procedures_table(self, engine: Engine, icu_procedure_scheme: CodingScheme) -> pd.DataFrame:
        table = CodedSQLTable(self.config.icu_procedures)
        c_code = self.config.icu_procedures.code_alias
        dataframe = table(engine)
        dataframe = dataframe[dataframe[c_code].isin(icu_procedure_scheme.codes)]
        return dataframe.reset_index(drop=True)

    def _extract_icu_inputs_table(self, engine: Engine, icu_input_scheme: CodingScheme) -> pd.DataFrame:
        table = CodedSQLTable(self.config.icu_inputs)
        c_code = self.config.icu_inputs.code_alias
        dataframe = table(engine)
        dataframe = dataframe[dataframe[c_code].isin(icu_input_scheme.codes)]
        return dataframe.reset_index(drop=True)

    def _extract_hosp_procedures_table(self, engine: Engine,
                                       procedure_icd_scheme: MixedICDScheme) -> pd.DataFrame:
        table = MixedICDSQLTable(self.config.hosp_procedures)
        c_icd_code = self.config.hosp_procedures.icd_code_alias
        c_icd_version = self.config.hosp_procedures.icd_version_alias
        c_code = self.config.hosp_procedures.code_alias
        dataframe = table(engine)
        return procedure_icd_scheme.mixedcode_format_table(dataframe, c_icd_code, c_icd_version, c_code)

    def dataset_scheme_manager_from_selection(self, config: MIMICIVDatasetSchemeConfig) -> CodingSchemesManager:
        """
        Create a dataset scheme from the given config.
        TODO: document me.
        Returns:
            (CodingScheme.DatasetScheme) A new scheme that is also registered in the current runtime.
        """
        manager = setup_standard_icd_ccs(CodingSchemesManager(),
                                         scheme_selection=CCSICDSchemeSelection.all(),
                                         outcome_selection=CCSICDOutcomeSelection.all())
        manager = self.register_gender_scheme(manager, config)
        manager = self.register_ethnicity_scheme(manager, config)
        manager = self.register_icu_inputs_scheme(manager, config)
        manager = self.register_icu_procedures_scheme(manager, config)
        manager = self.register_hosp_procedures_scheme(manager, config)
        manager = self.register_dx_discharge_scheme(manager, config)
        manager = self.register_obs_scheme(manager, config)
        return manager

    def load_tables(self, dataset_scheme: DatasetScheme) -> DatasetTables:
        if dataset_scheme.hosp_procedures is not None:
            hosp_procedures = self._extract_hosp_procedures_table(self.create_engine(),
                                                                  dataset_scheme.hosp_procedures)
        else:
            hosp_procedures = None
        if dataset_scheme.icu_procedures is not None:
            icu_procedures = self._extract_icu_procedures_table(self.create_engine(),
                                                                dataset_scheme.icu_procedures)
        else:
            icu_procedures = None
        if dataset_scheme.icu_inputs is not None:
            icu_inputs = self._extract_icu_inputs_table(self.create_engine(),
                                                        dataset_scheme.icu_inputs)
        else:
            icu_inputs = None
        if dataset_scheme.obs is not None:
            obs = self._extract_obs_table(self.create_engine(), dataset_scheme.obs)
        else:
            obs = None

        static = self._extract_static_table(self.create_engine())
        admissions = self._extract_admissions_table(self.create_engine())
        dx_discharge = self._extract_dx_discharge_table(self.create_engine(), dataset_scheme.dx_discharge)
        return DatasetTables(
            static=static,
            admissions=admissions,
            dx_discharge=dx_discharge,
            obs=obs, icu_procedures=icu_procedures, icu_inputs=icu_inputs, hosp_procedures=hosp_procedures)

    def __call__(self):
        return self


class MIMICIVDatasetPipelineConfig(AbstractDatasetPipelineConfig):
    overlap_merge: bool = True


class MIMICIVDatasetConfig(MIMICIVSQLConfig):
    tables: MIMICIVSQLTablesConfig = field(default_factory=MIMICIVSQLTablesConfig, kw_only=True)
    pipeline: MIMICIVDatasetPipelineConfig = field(default_factory=MIMICIVDatasetPipelineConfig)
    scheme: MIMICIVDatasetSchemeConfig = field(default_factory=MIMICIVDatasetSchemeConfig)


class MIMICIVDataset(Dataset):
    config: MIMICIVDatasetConfig

    @staticmethod
    def icu_inputs_uom_normalization(icu_inputs_config: RatedInputTableConfig,
                                     icu_inputs_uom_normalization_table: pd.DataFrame) -> pd.DataFrame:
        c_universal_uom = icu_inputs_config.derived_universal_unit
        c_code = icu_inputs_config.code_alias
        c_unit = icu_inputs_config.amount_unit_alias
        c_normalization = icu_inputs_config.derived_unit_normalization_factor
        df = icu_inputs_uom_normalization_table.astype({c_normalization: float})

        columns = [c_code, c_unit, c_normalization]
        assert all(c in df.columns for c in columns), \
            f"Columns {columns} not found in the normalization table."

        if c_universal_uom not in df.columns:
            df[c_universal_uom] = ''
            for code, code_df in df.groupby(c_code):
                # Select the first unit associated with 1.0 as a normalization factor.
                index = code_df[code_df[c_normalization] == 1.0].first_valid_index()
                if index is not None:
                    df.loc[code_df.index, c_universal_uom] = code_df.loc[index, c_unit]
        return df

    @classmethod
    def load_scheme_manager(cls, config: MIMICIVDatasetConfig) -> CodingSchemesManager:
        sql = MIMICIVSQLTablesInterface(config.tables)
        return sql.dataset_scheme_manager_from_selection(config=config.scheme)

    @classmethod
    def load_tables(cls, config: MIMICIVSQLConfig, scheme: DatasetScheme) -> DatasetTables:
        sql = MIMICIVSQLTablesInterface(config.tables)
        return sql.load_tables(scheme)


ADMISSIONS_CONF = AdmissionSQLTableConfig(query_template="mimiciv/sql/admissions.tsql")
STATIC_CONF = StaticSQLTableConfig(query_template="mimiciv/sql/static.tsql",
                                   gender_space_query_template="mimiciv/sql/static_gender_space.tsql",
                                   race_space_query_template="mimiciv/sql/static_race_space.tsql")
DX_DISCHARGE_CONF = AdmissionMixedICDSQLTableConfig(query_template="mimiciv/sql/dx_discharge.tsql",
                                                    space_query_template="mimiciv/sql/dx_discharge_space.tsql")

RENAL_OUT_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="renal_out",
                                                               attributes=['uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr'],
                                                               query_template="mimiciv/sql/renal_out.tsql")

RENAL_CREAT_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="renal_creat",
                                                                 attributes=['creat'],
                                                                 query_template="mimiciv/sql/renal_creat.tsql")

RENAL_AKI_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="renal_aki",
                                                               attributes=['aki_stage_smoothed', 'aki_binary'],
                                                               query_template="mimiciv/sql/renal_aki.tsql",
                                                               type_hint=('O', 'B'))  # Ordinal, Binary.

SOFA_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="sofa",
                                                          attributes=["sofa_24hours"],
                                                          query_template="mimiciv/sql/sofa.tsql",
                                                          default_type_hint='O')  # Ordinal.
BLOOD_GAS_ATTRIBUTES = ['so2', 'po2', 'pco2', 'fio2',
                        'fio2_chartevents',
                        'aado2', 'aado2_calc',
                        'pao2fio2ratio', 'ph',
                        'baseexcess', 'bicarbonate', 'totalco2',
                        'hematocrit',
                        'hemoglobin',
                        'carboxyhemoglobin', 'methemoglobin',
                        'chloride', 'calcium', 'temperature',
                        'potassium',
                        'sodium', 'lactate', 'glucose']
BLOOD_GAS_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="blood_gas",
                                                               attributes=BLOOD_GAS_ATTRIBUTES,
                                                               query_template="mimiciv/sql/blood_gas.tsql")

BLOOD_CHEMISTRY_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="blood_chemistry",
                                                                     attributes=['albumin', 'globulin',
                                                                                 'total_protein',
                                                                                 'aniongap', 'bicarbonate', 'bun',
                                                                                 'calcium', 'chloride',
                                                                                 'creatinine', 'glucose', 'sodium',
                                                                                 'potassium'],
                                                                     query_template="mimiciv/sql/blood_chemistry.tsql")

CARDIAC_MARKER_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="cardiac_marker",
                                                                    attributes=['troponin_t2', 'ntprobnp', 'ck_mb'],
                                                                    query_template="mimiciv/sql/cardiac_marker.tsql")

WEIGHT_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="weight",
                                                            attributes=['weight'],
                                                            query_template="mimiciv/sql/weight.tsql")

CBC_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="cbc",
                                                         attributes=['hematocrit', 'hemoglobin', 'mch', 'mchc',
                                                                     'mcv',
                                                                     'platelet',
                                                                     'rbc', 'rdw', 'wbc'],
                                                         query_template="mimiciv/sql/cbc.tsql")

VITAL_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="vital",
                                                           attributes=['heart_rate', 'sbp', 'dbp', 'mbp', 'sbp_ni',
                                                                       'dbp_ni',
                                                                       'mbp_ni', 'resp_rate',
                                                                       'temperature', 'spo2',
                                                                       'glucose'],
                                                           query_template="mimiciv/sql/vital.tsql")

# Glasgow Coma Scale, a measure of neurological function
GCS_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="gcs",
                                                         attributes=['gcs', 'gcs_motor', 'gcs_verbal', 'gcs_eyes',
                                                                     'gcs_unable'],
                                                         query_template="mimiciv/sql/gcs.tsql",
                                                         default_type_hint='O')  # Ordinal.

# Intracranial pressure
ICP_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="icp",
                                                         attributes=['icp'],
                                                         query_template="mimiciv/sql/icp.tsql")

# Inflammation
INFLAMMATION_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="inflammation",
                                                                  attributes=['crp'],
                                                                  query_template="mimiciv/sql/inflammation.tsql")

# Coagulation
COAGULATION_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="coagulation",
                                                                 attributes=['pt', 'ptt', 'inr', 'd_dimer',
                                                                             'fibrinogen', 'thrombin'],
                                                                 query_template="mimiciv/sql/coagulation.tsql")
# Blood differential
BLOOD_DIFF_ATTRIBUTES = ['neutrophils', 'lymphocytes',
                         'monocytes',
                         'eosinophils', 'basophils',
                         'atypical_lymphocytes',
                         'bands', 'immature_granulocytes',
                         'metamyelocytes',
                         'nrbc',
                         'basophils_abs', 'eosinophils_abs',
                         'lymphocytes_abs',
                         'monocytes_abs', 'neutrophils_abs']
BLOOD_DIFF_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="blood_diff",
                                                                attributes=BLOOD_DIFF_ATTRIBUTES,
                                                                query_template="mimiciv/sql/blood_diff.tsql")
ENZYMES_ATTRIBUTES = ['ast', 'alt', 'alp', 'ld_ldh', 'ck_cpk',
                      'ck_mb',
                      'amylase', 'ggt', 'bilirubin_direct',
                      'bilirubin_total', 'bilirubin_indirect']
# Enzymes
ENZYMES_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="enzymes",
                                                             attributes=ENZYMES_ATTRIBUTES,
                                                             query_template="mimiciv/sql/enzymes.tsql")

OBS_COMPONENTS = [
    RENAL_OUT_CONF,
    RENAL_CREAT_CONF,
    RENAL_AKI_CONF,
    SOFA_CONF,
    BLOOD_GAS_CONF,
    BLOOD_CHEMISTRY_CONF,
    CARDIAC_MARKER_CONF,
    WEIGHT_CONF,
    CBC_CONF,
    VITAL_CONF,
    # GCS_CONF,
    ICP_CONF,
    INFLAMMATION_CONF,
    COAGULATION_CONF,
    BLOOD_DIFF_CONF,
    ENZYMES_CONF
]
OBS_TABLE_CONFIG = AdmissionTimestampedCodedValueSQLTableConfig(components=OBS_COMPONENTS)

ICU_INPUTS_CONF = RatedInputSQLTableConfig(query_template="mimiciv/sql/icu_inputs.tsql",
                                           space_query_template="mimiciv/sql/icu_inputs_space.tsql")
ICU_PROCEDURES_CONF = IntervalICUProcedureSQLTableConfig(query_template="mimiciv/sql/icu_procedures.tsql",
                                                         space_query_template="mimiciv/sql/icu_procedures_space.tsql")
HOSP_PROCEDURES_CONF = AdmissionIntervalBasedMixedICDTableConfig(
    query_template="mimiciv/sql/hosp_procedures.tsql", space_query_template="mimiciv/sql/hosp_procedures_space.tsql")
