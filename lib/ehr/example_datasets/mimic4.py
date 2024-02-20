"""."""
from __future__ import annotations

import logging
import warnings
from collections import defaultdict
from dataclasses import field
from functools import cached_property
from typing import Dict, List, Optional, Iterable

import pandas as pd
import sqlalchemy
from sqlalchemy import Engine

from lib.base import Module
from lib.ehr.coding_scheme import (CodingSchemeConfig, FlatScheme, CodeMap)
from lib.ehr.dataset import (DatasetScheme, StaticTableConfig,
                             AdmissionTimestampedMultiColumnTableConfig, AdmissionIntervalBasedCodedTableConfig,
                             AdmissionTimestampedCodedValueTableConfig, AdmissionLinkedCodedValueTableConfig,
                             TableConfig, CodedTableConfig, AdmissionTableConfig,
                             RatedInputTableConfig, DatasetTablesConfig,
                             DatasetTables)
from lib.ehr.example_schemes.icd import ICDScheme

warnings.filterwarnings('error',
                        category=RuntimeWarning,
                        message=r'overflow encountered in cast')


class SQLTableConfig(TableConfig):
    query: str = field(kw_only=True)


class CodedSQLTableConfig(SQLTableConfig, CodedTableConfig):
    space_query: str = field(kw_only=True)


class AdmissionTimestampedMultiColumnSQLTableConfig(SQLTableConfig,
                                                    AdmissionTimestampedMultiColumnTableConfig):
    admission_id_alias: str = 'admission_id'
    time_alias: str = 'time'


class AdmissionTimestampedCodedValueSQLTableConfig(CodedSQLTableConfig, AdmissionTimestampedCodedValueTableConfig):
    components: List[AdmissionTimestampedMultiColumnSQLTableConfig] = field(kw_only=True)
    query: Optional[str] = None
    space_query: Optional[str] = None
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
    gender_space_query: str = field(kw_only=True)
    race_space_query: str = field(kw_only=True)
    anchor_year_alias: str = 'anchor_year'
    anchor_age_alias: str = 'anchor_age'
    subject_id_alias: str = 'subject_id'
    gender_alias: str = 'gender'
    date_of_birth_alias: str = 'date_of_birth'
    race_alias: str = 'race'


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
        # coerce to integers then fix as strings.
        int_dtypes = {k: int for k in columns if k in df.columns}
        str_dtypes = {k: str for k in columns if k in df.columns}
        return df.astype(int_dtypes).astype(str_dtypes)

    def _coerce_id_to_str(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Some of the integer ids in the database when downloaded are stored as floats.
        A fix is to coerce them to integers then fix as strings.
        """
        return self._coerce_columns_to_str(df, self.config.alias_id_dict.values())

    def __call__(self, engine: Engine):
        query = self.config.query.format(**self.config.alias_dict)
        return self._coerce_id_to_str(pd.read_sql(query, engine,
                                                  coerce_float=False))


class TimestampedMultiColumnSQLTable(SQLTable):
    # TODO: Document this class.

    config: AdmissionTimestampedMultiColumnSQLTableConfig

    def __call__(self, engine: Engine, attributes: List[str]) -> pd.DataFrame:
        assert len(set(attributes)) == len(attributes), f"Duplicate attributes {attributes}"
        assert all(a in self.config.attributes for a in attributes), \
            f"Some attributes {attributes} not in {self.config.attributes}"
        query = self.config.query.format(attributes=','.join(attributes),
                                         **self.config.alias_dict)
        return self._coerce_id_to_str(pd.read_sql(query, engine,
                                                  coerce_float=False))

    def space(self, engine: Engine):
        space = pd.DataFrame([(self.config.name, a) for a in self.config.attributes],
                             columns=['table_name', 'attribute'])
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

    def register_gender_scheme(self, name: str,
                               engine: Engine,
                               gender_selection: Optional[pd.DataFrame]):
        return CodedSQLTableConfig._register_scheme(name=name,
                                                    supported_space=self.gender_space(engine),
                                                    code_selection=gender_selection,
                                                    c_code=self.config.gender_alias,
                                                    c_desc=self.config.gender_alias)

    def register_ethnicity_scheme(self, name: str,
                                  engine: Engine,
                                  ethnicity_selection: Optional[pd.DataFrame]):
        return CodedSQLTableConfig._register_scheme(name=name,
                                                    supported_space=self.ethnicity_space(engine),
                                                    code_selection=ethnicity_selection,
                                                    c_code=self.config.race_alias,
                                                    c_desc=self.config.race_alias)


class CategoricalSQLTable(SQLTable):
    # TODO: Document this class.

    config: CodedSQLTableConfig

    def space(self, engine: Engine):
        query = self.config.space_query.format(**self.config.alias_dict)
        return self._coerce_id_to_str(pd.read_sql(query, engine,
                                                  coerce_float=False))


class ObservablesSQLTable(SQLTable):
    # TODO: Document this class.

    config: AdmissionTimestampedCodedValueSQLTableConfig

    def __cal__(self, engine: Engine, obs_scheme: ObservableMIMICScheme) -> pd.DataFrame:
        # TODO: test this method with a mock engine.
        dfs = []
        c_code = self.config.code_alias
        c_value = self.config.value_alias

        for table_name, attrs_df in obs_scheme.as_dataframe().groupby('table_name'):
            attributes = attrs_df['attribute'].tolist()
            attr2code = attrs_df.set_index('attribute')['code'].to_dict()
            # download the table.
            table = self.table_interface(str(table_name))
            obs_df = table(engine, attributes)
            # melt the table. (admission_id, time, attribute, value)
            melted_obs_df = obs_df.melt(id_vars=[table.config.admission_id_alias, table.config.time_alias],
                                        var_name=['attribute'], value_name=c_value, value_vars=attributes)
            # add the code. (admission_id, time, attribute, value, code)
            melted_obs_df[c_code] = melted_obs_df['attribute'].map(attr2code)
            melted_obs_df = melted_obs_df[melted_obs_df.value.notnull()]
            # drop the attribute. (admission_id, time, value, code)
            melted_obs_df = melted_obs_df.drop('attribute', axis=1)
            dfs.append(melted_obs_df)

        return pd.concat(dfs, ignore_index=True)

    def space(self, engine: Engine) -> pd.DataFrame:
        df_list = [self.table_interface(c.name).space(engine) for c in self.config.components]
        return pd.concat(df_list).sort_values(['attribute']).sort_index()

    def table_interface(self, table_name: str) -> TimestampedMultiColumnSQLTable:
        table_conf = next(t for t in self.config.components if t.name == table_name)
        return TimestampedMultiColumnSQLTable(config=table_conf)

    def register_scheme(self, name: str,
                        engine: Engine, attributes_selection: Optional[pd.DataFrame]) -> ObservableMIMICScheme:
        supported_space = self.space(engine)
        if attributes_selection is None:
            attributes_selection = supported_space

        scheme = ObservableMIMICScheme.from_selection(name, attributes_selection)
        FlatScheme.register_scheme(scheme)
        return scheme


class MixedICDSQLTable(CategoricalSQLTable):
    # TODO: Document this class.

    config: AdmissionMixedICDSQLTableConfig

    @staticmethod
    def _register_scheme(name: str,
                         icd_schemes: Dict[str, str],
                         supported_space: pd.DataFrame,
                         icd_version_selection: Optional[pd.DataFrame],
                         c_version: str, c_code: str, c_desc: str) -> MixedICDScheme:
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
            # Check that all the ICD codes-versions are supported.
            for version, codes in icd_version_selection.groupby(c_version):
                support_subset = supported_space[supported_space[c_version] == version]
                assert codes[c_code].isin(support_subset[c_code]).all(), "Some ICD codes are not supported"
        scheme = MixedICDScheme.from_selection(name, icd_version_selection,
                                               icd_version_alias=c_version,
                                               icd_code_alias=c_code,
                                               description_alias=c_desc,
                                               icd_schemes=icd_schemes)
        MixedICDScheme.register_scheme(scheme)
        scheme.register_maps_loaders()

        return scheme

    def register_scheme(self, name: str,
                        engine: Engine,
                        icd_schemes: Dict[str, str],
                        icd_version_selection: Optional[pd.DataFrame]) -> MixedICDScheme:
        c_version = self.config.icd_version_alias
        c_code = self.config.icd_code_alias
        c_desc = self.config.description_alias
        return self._register_scheme(name=name, icd_schemes=icd_schemes,
                                     supported_space=self.space(engine),
                                     icd_version_selection=icd_version_selection,
                                     c_version=c_version, c_code=c_code, c_desc=c_desc)


class CodedSQLTable(CategoricalSQLTable):
    # TODO: Document this class.

    config: CodedSQLTableConfig

    @classmethod
    def _register_scheme(cls, name: str,
                         supported_space: pd.DataFrame,
                         code_selection: Optional[pd.DataFrame],
                         c_code: str, c_desc: str) -> FlatScheme:
        # TODO: test this method.
        if code_selection is None:
            code_selection = supported_space[c_code].drop_duplicates().astype(str).tolist()
        else:
            code_selection = code_selection[c_code].drop_duplicates().astype(str).tolist()

            assert len(set(code_selection) - set(supported_space[c_code])) == 0, \
                "Some item ids are not supported."
        desc = supported_space.set_index(c_code)[c_desc].to_dict()
        desc = {k: v for k, v in desc.items() if k in code_selection}
        scheme = FlatScheme(CodingSchemeConfig(name),
                            codes=sorted(code_selection),
                            desc=desc)
        FlatScheme.register_scheme(scheme)
        return scheme

    def register_scheme(self, name: str,
                        engine: Engine, code_selection: Optional[pd.DataFrame]) -> FlatScheme:
        return self._register_scheme(name=name, supported_space=self.space(engine),
                                     code_selection=code_selection, c_code=self.config.code_alias,
                                     c_desc=self.config.description_alias)


class ObservableMIMICScheme(FlatScheme):
    # TODO: Document this class.

    @classmethod
    def from_selection(cls, name: str, obs_variables: pd.DataFrame):
        # TODO: test this method.
        """
        Create a scheme from a selection of observation variables.

        Args:
            name: Name of the scheme.
            obs_variables: A DataFrame containing the variables to include in the scheme.
                The DataFrame should have the following columns:
                    - table_name (index): The name of the table containing the variable.
                    - attribute: The name of the variable.

        Returns:
            (CodingScheme.FlatScheme) A new scheme containing the variables in obs_variables.
        """
        obs_variables = obs_variables.sort_values(['table_name', 'attribute'])
        # format codes to be of the form 'table_name.attribute'
        codes = (obs_variables.index + '.' + obs_variables['attribute']).tolist()
        desc = dict(zip(codes, codes))

        return cls(CodingSchemeConfig(name),
                   codes=codes,
                   desc=desc)

    def as_dataframe(self):
        columns = ['code', 'desc', 'code_index', 'table_name', 'attribute']
        return pd.DataFrame([(c, self.desc[c], self.index[c], *c.split('.')) for c in self.codes],
                            columns=columns)


class MixedICDScheme(FlatScheme):
    # TODO: Document this class.

    _icd_schemes: Dict[str, ICDScheme]
    _sep: str = ':'

    def __init__(self, config: CodingSchemeConfig, codes: List[str], desc: Dict[str, str],
                 icd_schemes: Dict[str, ICDScheme], sep: str = ':'):
        super().__init__(config, codes=codes, desc=desc)
        self._icd_schemes = icd_schemes
        self._sep = sep

    @classmethod
    def from_selection(cls, name: str, icd_version_selection: pd.DataFrame,
                       icd_version_alias: str, icd_code_alias: str, description_alias: str,
                       icd_schemes: Dict[str, str], sep: str = ':') -> MixedICDScheme:
        # TODO: test this method.
        icd_version_selection = icd_version_selection.sort_values([icd_version_alias, icd_code_alias])
        icd_version_selection = icd_version_selection.drop_duplicates([icd_version_alias, icd_code_alias]).astype(str)
        assert icd_version_selection[icd_version_alias].isin(icd_schemes).all(), \
            f"Only {', '.join(map(lambda x: f'ICD-{x}', icd_schemes))} are expected."

        # assert no duplicate (icd_code, icd_version)
        assert icd_version_selection.groupby([icd_version_alias, icd_code_alias]).size().max() == 1, \
            "Duplicate (icd_code, icd_version) pairs are not allowed."

        icd_schemes_loaded: Dict[str, ICDScheme] = {k: ICDScheme.from_name(v) for k, v in icd_schemes.items()}

        assert all(isinstance(s, ICDScheme) for s in icd_schemes_loaded.values()), \
            "Only ICD schemes are expected."

        for version, icd_df in icd_version_selection.groupby(icd_version_alias):
            scheme = icd_schemes_loaded[version]
            icd_version_selection.loc[icd_df.index, icd_code_alias] = \
                icd_df[icd_code_alias].str.replace(' ', '').str.replace('.', '').map(scheme.add_dots)

        codes = (icd_version_selection[icd_version_alias] + sep + icd_version_selection[icd_code_alias]).tolist()
        df = icd_version_selection.copy()
        df['code'] = codes
        desc = df.set_index('code')[description_alias].to_dict()

        return MixedICDScheme(config=CodingSchemeConfig(name),
                              codes=codes,
                              desc=desc,
                              icd_schemes=icd_schemes_loaded,
                              sep=sep)

    def mixedcode_format_table(self, table: pd.DataFrame, icd_code_alias: str,
                               icd_version_alias: str, code_alias: str) -> pd.DataFrame:
        # TODO: test this method.
        """
        Format a table with mixed codes to the ICD version:icd_code format and filter out codes that are not in the scheme.
        """
        table = table.copy()
        assert icd_version_alias in table.columns, f"Column {icd_version_alias} not found."
        assert icd_code_alias in table.columns, f"Column {icd_code_alias} not found."
        assert table[icd_version_alias].isin(self._icd_schemes).all(), \
            f"Only ICD version {list(self._icd_schemes.keys())} are expected."

        for version, icd_df in table.groupby(icd_version_alias):
            scheme = self._icd_schemes[str(version)]
            # Remove dots and spaces and then add them back.
            fixed_icd = icd_df[icd_code_alias].str.replace(' ', '').str.replace('.', '').map(scheme.add_dots)
            table.loc[icd_df.index, icd_code_alias] = fixed_icd

        # the version:icd_code format.
        table[code_alias] = table[icd_version_alias] + self._sep + table[icd_code_alias]

        # filter out codes that are not in the scheme.
        return table[table[code_alias].isin(self.codes)].reset_index(drop=True)

    def generate_maps(self):
        """
        Register the mappings between the Mixed ICD scheme and the individual ICD scheme.
        For example, if the current `MixedICD` is mixing ICD-9 and ICD-10,
        then register the two mappings between this scheme and ICD-9 and ICD-10 separately.
        This assumes that the current runtime has already registered mappings
        between the individual ICD schemes.
        """

        # mixed2pure_maps has the form {icd_version: {mixed_code: {icd}}}.
        mixed2pure_maps = {}
        lost_codes = []
        dataframe = self.as_dataframe()
        for pure_version, pure_scheme in self._icd_schemes.items():
            # mixed2pure has the form {mixed_code: {icd}}.
            mixed2pure = defaultdict(set)
            for mixed_version, mixed_version_df in dataframe.groupby('icd_version'):
                icd2mixed = mixed_version_df.set_index('icd_code')['code'].to_dict()
                icd2mixed = {icd: mixed_code for icd, mixed_code in icd2mixed.items() if icd in pure_scheme.codes}
                assert len(icd2mixed) > 0, "No mapping between the mixed and pure ICD schemes was found."
                if mixed_version == pure_version:
                    mixed2pure.update({c: {icd} for icd, c in icd2mixed.items()})
                else:
                    # if mixed_version != pure_version, then retrieve
                    # the mapping between ICD-{mixed_version} and ICD-{pure_version}
                    pure_map = CodeMap.get_mapper(self._icd_schemes[mixed_version].name,
                                                  self._icd_schemes[pure_version].name)

                    for icd, mixed_code in icd2mixed.items():
                        if icd in pure_map:
                            mixed2pure[mixed_code].update(pure_map[icd])
                        else:
                            lost_codes.append(mixed_code)

            mixed2pure_maps[pure_version] = mixed2pure
        # register the mapping between the mixed and pure ICD schemes.
        for pure_version, mixed2pure in mixed2pure_maps.items():
            pure_scheme = self._icd_schemes[pure_version]
            conf = CodingSchemeConfig(self.name, pure_scheme.name)

            CodeMap.register_map(self.name,
                                 pure_scheme.name,
                                 CodeMap(conf, mixed2pure))

        lost_df = dataframe[dataframe['code'].isin(lost_codes)]
        if len(lost_df) > 0:
            logging.warning(f"Lost {len(lost_df)} codes when generating the mapping between the Mixed ICD"
                            "scheme and the individual ICD scheme.")
            logging.warning(lost_df.to_string().replace('\n', '\n\t'))

    def register_maps_loaders(self):
        """
        Register the lazy-loading of mapping between the Mixed ICD scheme and the individual ICD scheme.
        """
        for pure_version, pure_scheme in self._icd_schemes.items():
            CodeMap.register_map_loader(self.name, pure_scheme.name, self.generate_maps)

    def as_dataframe(self):
        columns = ['code', 'desc', 'code_index', 'icd_version', 'icd_code']
        return pd.DataFrame([(c, self.desc[c], self.index[c], *c.split(self._sep)) for c in self.codes],
                            columns=columns)


class MIMICIVSQLConfig(DatasetTablesConfig):
    # TODO: Document this class.

    host: str
    port: int
    user: str
    password: str
    dbname: str

    static_table: StaticSQLTableConfig = field(default_factory=lambda: STATIC_CONF)
    admissions_table: AdmissionSQLTableConfig = field(default_factory=lambda: ADMISSIONS_CONF)
    dx_discharge_table: AdmissionMixedICDSQLTableConfig = field(default_factory=lambda: DX_DISCHARGE_CONF)
    obs_table: AdmissionTimestampedCodedValueSQLTableConfig = field(default_factory=lambda: OBS_TABLE_CONFIG)
    icu_procedures_table: IntervalICUProcedureSQLTableConfig = field(default_factory=lambda: ICU_PROC_CONF)
    icu_inputs_table: RatedInputSQLTableConfig = field(default_factory=lambda: ICU_INPUT_CONF)
    hosp_procedures_table: AdmissionIntervalBasedMixedICDTableConfig = field(
        default_factory=lambda: HOSP_PROC_CONF)


class MIMICIVDatasetScheme(DatasetScheme):
    ethnicity: FlatScheme
    gender: FlatScheme
    dx_discharge: MixedICDScheme
    obs: Optional[ObservableMIMICScheme] = None
    icu_procedures: Optional[FlatScheme] = None
    hosp_procedures: Optional[MixedICDScheme] = None
    icu_inputs: Optional[FlatScheme] = None


class MIMICIVSQL(Module):
    # TODO: Document this class.

    config: MIMICIVSQLConfig

    def create_engine(self) -> Engine:
        return sqlalchemy.create_engine(
            f'postgresql+psycopg2://{self.config.user}:{self.config.password}@'
            f'{self.config.host}:{self.config.port}/{self.config.dbname}')

    def register_obs_scheme(self, name: str,
                            attributes_selection: Optional[pd.DataFrame]):
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
        table = ObservablesSQLTable(self.config.obs_table)
        return table.register_scheme(name, self.create_engine(), attributes_selection)

    def register_icu_input_scheme(self, name: str, code_selection: Optional[pd.DataFrame]):
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
        table = CodedSQLTable(self.config.icu_inputs_table)
        return table.register_scheme(name, self.create_engine(), code_selection)

    def register_icu_procedure_scheme(self, name: str, code_selection: Optional[pd.DataFrame]):
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
        table = CodedSQLTable(self.config.icu_procedures_table)
        return table.register_scheme(name, self.create_engine(), code_selection)

    def register_hosp_procedure_scheme(self, name: str, icd_version_selection: Optional[pd.DataFrame]):
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
        table = MixedICDSQLTable(self.config.hosp_procedures_table)
        return table.register_scheme(name, self.create_engine(), {'9': 'pr_icd9', '10': 'pr_flat_icd10'},
                                     icd_version_selection)

    def register_dx_discharge_scheme(self, name: str, icd_version_selection: Optional[pd.DataFrame]):
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
        table = MixedICDSQLTable(self.config.dx_discharge_table)
        return table.register_scheme(name, self.create_engine(), {'9': 'dx_icd9', '10': 'dx_flat_icd10'},
                                     icd_version_selection)

    @cached_property
    def supported_obs_variables(self) -> pd.DataFrame:
        table = ObservablesSQLTable(self.config.obs_table)
        return table.space(self.create_engine())

    @cached_property
    def supported_icu_procedures(self) -> pd.DataFrame:
        table = CodedSQLTable(self.config.icu_procedures_table)
        return table.space(self.create_engine())

    @cached_property
    def supported_icu_inputs(self) -> pd.DataFrame:
        table = CodedSQLTable(self.config.icu_inputs_table)
        return table.space(self.create_engine())

    @cached_property
    def supported_hosp_procedures(self) -> pd.DataFrame:
        table = MixedICDSQLTable(self.config.hosp_procedures_table)
        return table.space(self.create_engine())

    @cached_property
    def supported_dx_discharge(self) -> pd.DataFrame:
        table = MixedICDSQLTable(self.config.dx_discharge_table)
        return table.space(self.create_engine())

    def _extract_static_table(self, engine: Engine) -> pd.DataFrame:
        table = StaticSQLTable(self.config.static_table)
        return table(engine)

    def _extract_admissions_table(self, engine: Engine) -> pd.DataFrame:
        table = SQLTable(self.config.admissions_table)
        return table(engine)

    def _extract_dx_discharge_table(self, engine: Engine, dx_discharge_scheme: MixedICDScheme) -> pd.DataFrame:
        table = MixedICDSQLTable(self.config.dx_discharge_table)
        dataframe = table(engine)
        c_icd_version = self.config.dx_discharge_table.icd_version_alias
        c_icd_code = self.config.dx_discharge_table.icd_code_alias
        return dx_discharge_scheme.mixedcode_format_table(dataframe, c_icd_code, c_icd_version)

    def _extract_obs_table(self, engine: Engine, obs_scheme: ObservableMIMICScheme) -> pd.DataFrame:
        table = ObservablesSQLTable(self.config.obs_table)
        return table(engine, obs_scheme)

    def _extract_icu_procedures_table(self, engine: Engine, icu_procedure_scheme: FlatScheme) -> pd.DataFrame:
        table = SQLTable(self.config.icu_procedures_table)
        c_code = self.config.icu_procedures_table.code_alias
        dataframe = table(engine)
        dataframe = dataframe[dataframe[c_code].isin(icu_procedure_scheme.codes)]
        dataframe[c_code] = dataframe[c_code]
        return dataframe.reset_index(drop=True)

    def _extract_icu_inputs_table(self, engine: Engine, icu_input_scheme: FlatScheme) -> pd.DataFrame:
        table = CodedSQLTable(self.config.icu_inputs_table)
        c_code = self.config.icu_inputs_table.code_alias
        dataframe = table(engine)
        dataframe = dataframe[dataframe[c_code].isin(icu_input_scheme.codes)]
        dataframe[c_code] = dataframe[c_code]
        return dataframe.reset_index(drop=True)

    def _extract_hosp_procedures_table(self, engine: Engine,
                                       procedure_icd_scheme: MixedICDScheme) -> pd.DataFrame:
        table = MixedICDSQLTable(self.config.hosp_procedures_table)
        c_icd_code = self.config.hosp_procedures_table.icd_code_alias
        c_icd_version = self.config.hosp_procedures_table.icd_version_alias
        c_code = self.config.hosp_procedures_table.code_alias
        dataframe = table(engine)
        return procedure_icd_scheme.mixedcode_format_table(dataframe, c_icd_code, c_icd_version, c_code)

    def dataset_scheme_from_selection(
            self, name_prefix: str,
            gender: Optional[pd.DataFrame] = None,
            ethnicity: Optional[pd.DataFrame] = None,
            dx_discharge: Optional[pd.DataFrame] = None,
            obs: Optional[pd.DataFrame] = None,
            icu_inputs: Optional[pd.DataFrame] = None,
            icu_procedures: Optional[pd.DataFrame] = None,
            hosp_procedures: Optional[pd.DataFrame] = None) -> MIMICIVDatasetScheme:
        """
        Create a dataset scheme from the given selection of variables.

        Args:
            gender: A dataframe containing the `gender`s to generate the new scheme. If None, all supported items will be used.
            ethnicity: A dataframe containing the `ethnicity`s to generate the new scheme. If None, all supported items will be used.
            dx_discharge: A dataframe containing the `icd_code`s to generate the new scheme. If None, all supported items will be used. The dataframe should have the following columns:
                - icd_version: The version of the ICD.
                - icd_code: The ICD code.
            obs: A dictionary of observation table names and their corresponding attributes.
                If None, all supported variables will be used.
            icu_inputs: A dataframe containing the `code`s to generate the new scheme. If None, all supported items will be used.
            icu_procedures: A dataframe containing the `code`s of choice. If None, all supported items will be used.
            hosp_procedures: A dataframe containing the `icd_code`s to generate the new scheme. If None, all supported items will be used. The dataframe should have the following columns:
                - icd_version: The version of the ICD.
                - icd_code: The ICD code.

        Returns:
            (CodingScheme.DatasetScheme) A new scheme that is also registered in the current runtime.
        """
        pass

    def load_tables(self, dataset_scheme: MIMICIVDatasetScheme) -> DatasetTables:
        if dataset_scheme.obs is not None:
            obs = self._extract_obs_table(self.create_engine(), dataset_scheme.obs)
        else:
            obs = None
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
        if dataset_scheme.hosp_procedures is not None:
            hosp_procedures = self._extract_hosp_procedures_table(self.create_engine(),
                                                                  dataset_scheme.hosp_procedures)
        else:
            hosp_procedures = None

        return DatasetTables(
            static=self._extract_static_table(self.create_engine()),
            admissions=self._extract_admissions_table(self.create_engine()),
            dx_discharge=self._extract_dx_discharge_table(self.create_engine(), dataset_scheme.dx_discharge),
            obs=obs, icu_procedures=icu_procedures, icu_inputs=icu_inputs, hosp_procedures=hosp_procedures)

    def __call__(self):
        return self


#     def subject_info_extractor(self, subject_ids, target_scheme):
#
#         static_df = self.df['static']
#         c_gender = self.colname["static"].gender
#         c_anchor_year = self.colname["static"].anchor_year
#         c_anchor_age = self.colname["static"].anchor_age
#         c_eth = self.colname["static"].ethnicity
#
#         static_df = static_df.loc[subject_ids]
#         gender = static_df[c_gender].map(self.scheme.gender.codeset2vec)
#         subject_gender = gender.to_dict()
#
#         anchor_date = pd.to_datetime(static_df[c_anchor_year],
#                                      format='%Y').dt.normalize()
#         anchor_age = static_df[c_anchor_age].map(
#             lambda y: pd.DateOffset(years=-y))
#         dob = anchor_date + anchor_age
#         subject_dob = dict(zip(static_df.index.values, dob))
#         subject_eth = dict()
#         eth_mapper = self.scheme.ethnicity_mapper(target_scheme)
#         for subject_id in static_df.index.values:
#             eth_code = eth_mapper.map_codeset(
#                 [static_df.loc[subject_id, c_eth]])
#             subject_eth[subject_id] = eth_mapper.codeset2vec(eth_code)
#
#         return subject_dob, subject_gender, subject_eth
#
#     def dx_codes_extractor(self, admission_ids_list, target_scheme):
#         c_adm_id = self.colname["dx_discharge"].admission_id
#         c_code = self.colname["dx_discharge"].code
#         c_version = self.colname["dx_discharge"].version
#
#         df = self.df["dx_discharge"]
#         df = df[df[c_adm_id].isin(admission_ids_list)]
#         codes_df = {
#             adm_id: codes_df
#             for adm_id, codes_df in df.groupby(c_adm_id)
#         }
#         empty_vector = target_scheme.dx_discharge.empty_vector()
#
#         dx_mapper = self.scheme.dx_mapper(target_scheme)
#
#         def _extract_codes(adm_id):
#             _codes_df = codes_df.get(adm_id)
#             if _codes_df is None:
#                 return (adm_id, empty_vector)
#
#             vec = empty_vector
#             for version, version_df in _codes_df.groupby(c_version):
#                 mapper = dx_mapper[str(version)]
#                 codeset = mapper.map_codeset(version_df[c_code])
#                 vec = vec.union(mapper.codeset2vec(codeset))
#             return (adm_id, vec)
#
#         return map(_extract_codes, admission_ids_list)
#
#
# class MIMIC4ICUDataset(Dataset):
#
#     @classmethod
#     def _setup_core_pipeline(cls, config: DatasetConfig) -> DatasetPipeline:
#         raise NotImplementedError("Not implemented")
#
#     def to_subjects(self,
#                     subject_ids: List[int],
#                     num_workers: int,
#                     demographic_vector_config: DemographicVectorConfig,
#                     leading_observable_config: LeadingObservableExtractorConfig,
#                     target_scheme: MIMIC4ICUDatasetScheme,
#                     time_binning: Optional[int] = None,
#                     **kwargs):
#
#         subject_dob, subject_gender, subject_eth = self.subject_info_extractor(
#             subject_ids, target_scheme)
#         admission_ids = self.adm_extractor(subject_ids)
#         adm_ids_list = sum(map(list, admission_ids.values()), [])
#         logging.debug('Extracting dx_discharge codes...')
#         dx_codes = dict(self.dx_codes_extractor(adm_ids_list, target_scheme))
#         logging.debug('[DONE] Extracting dx_discharge codes')
#         logging.debug('Extracting dx_discharge codes history...')
#         dx_codes_history = dict(
#             self.dx_codes_history_extractor(dx_codes, admission_ids,
#                                             target_scheme))
#         logging.debug('[DONE] Extracting dx_discharge codes history')
#         logging.debug('Extracting outcome...')
#         outcome = dict(self.outcome_extractor(dx_codes, target_scheme))
#         logging.debug('[DONE] Extracting outcome')
#         logging.debug('Extracting procedures...')
#         procedures = dict(self.procedure_extractor(adm_ids_list))
#         logging.debug('[DONE] Extracting procedures')
#         logging.debug('Extracting inputs...')
#         inputs = dict(self.inputs_extractor(adm_ids_list))
#         logging.debug('[DONE] Extracting inputs')
#         logging.debug('Extracting observables...')
#         observables = dict(
#             self.observables_extractor(adm_ids_list, num_workers))
#
#         if time_binning is not None:
#             observables = dict((k, v.time_binning(time_binning))
#                                for k, v in observables.items())
#
#         logging.debug('[DONE] Extracting observables')
#
#         logging.debug('Compiling admissions...')
#         c_admittime = self.colname['adm'].admittime
#         c_dischtime = self.colname['adm'].dischtime
#         c_adm_interval = self.colname['adm'].adm_interval
#         adf = self.df['adm']
#         adm_dates = dict(
#             zip(adf.index, zip(adf[c_admittime], adf[c_dischtime])))
#         adm_interval = dict(zip(adf.index, adf[c_adm_interval]))
#         proc_repr = AggregateRepresentation(self.scheme.int_proc,
#                                             target_scheme.int_proc)
#
#         leading_obs_extractor = LeadingObservableExtractor(leading_observable_config)
#
#         def gen_admission(i):
#             interventions = InpatientInterventions(
#                 proc=procedures[i],
#                 input_=inputs[i],
#                 adm_interval=adm_interval[i])
#
#             obs = observables[i]
#             lead_obs = leading_obs_extractor(obs)
#
#             if time_binning is None:
#                 interventions = interventions.segment_proc(proc_repr)
#                 interventions = interventions.segment_input()
#                 lead_obs = lead_obs.segment(interventions.t_sep)
#                 obs = obs.segment(interventions.t_sep)
#
#             return Admission(admission_id=i,
#                              admission_dates=adm_dates[i],
#                              dx_codes=dx_codes[i],
#                              dx_codes_history=dx_codes_history[i],
#                              outcome=outcome[i],
#                              observables=obs,
#                              leading_observable=lead_obs,
#                              interventions=interventions)
#
#         def _gen_subject(subject_id):
#
#             _admission_ids = admission_ids[subject_id]
#             # for subject_id, subject_admission_ids in admission_ids.items():
#             _admission_ids = sorted(_admission_ids,
#                                     key=lambda aid: adm_dates[aid][0])
#
#             static_info = StaticInfo(
#                 date_of_birth=subject_dob[subject_id],
#                 gender=subject_gender[subject_id],
#                 ethnicity=subject_eth[subject_id],
#                 demographic_vector_config=demographic_vector_config)
#
#             with ThreadPoolExecutor(max_workers=num_workers) as executor:
#                 admissions = list(executor.map(gen_admission, _admission_ids))
#             return Patient(subject_id=subject_id,
#                            admissions=admissions,
#                            static_info=static_info)
#
#         return list(map(_gen_subject, subject_ids))
#
#     def procedure_extractor(self, admission_ids_list):
#         c_adm_id = self.colname["int_proc"].admission_id
#         c_code_index = self.colname["int_proc"].code_source_index
#         c_start_time = self.colname["int_proc"].start_time
#         c_end_time = self.colname["int_proc"].end_time
#         df = self.df["int_proc"]
#         df = df[df[c_adm_id].isin(admission_ids_list)]
#
#         def group_fun(x):
#             return pd.Series({
#                 0: x[c_code_index].to_numpy(),
#                 1: x[c_start_time].to_numpy(),
#                 2: x[c_end_time].to_numpy()
#             })
#
#         grouped = df.groupby(c_adm_id).apply(group_fun)
#         adm_arr = grouped.index.tolist()
#         input_size = len(self.scheme.int_proc)
#         for i in adm_arr:
#             yield (i,
#                    InpatientInput(index=grouped.loc[i, 0],
#                                   rate=np.ones_like(grouped.loc[i, 0],
#                                                     dtype=bool),
#                                   starttime=grouped.loc[i, 1],
#                                   endtime=grouped.loc[i, 2],
#                                   size=input_size))
#
#         for adm_id in set(admission_ids_list) - set(adm_arr):
#             yield (adm_id, InpatientInput.empty(input_size))
#
#     def inputs_extractor(self, admission_ids_list):
#         c_adm_id = self.colname["int_input"].admission_id
#         c_start_time = self.colname["int_input"].start_time
#         c_end_time = self.colname["int_input"].end_time
#         c_rate = self.colname["int_input"].rate
#         c_code_index = self.colname["int_input"].code_source_index
#
#         df = self.df["int_input"]
#         df = df[df[c_adm_id].isin(admission_ids_list)]
#
#         def group_fun(x):
#             return pd.Series({
#                 0: x[c_code_index].to_numpy(),
#                 1: x[c_rate].to_numpy(),
#                 2: x[c_start_time].to_numpy(),
#                 3: x[c_end_time].to_numpy()
#             })
#
#         grouped = df.groupby(c_adm_id).apply(group_fun)
#         adm_arr = grouped.index.tolist()
#         input_size = len(self.scheme.int_input)
#         for i in adm_arr:
#             yield (i,
#                    InpatientInput(index=grouped.loc[i, 0],
#                                   rate=grouped.loc[i, 1],
#                                   starttime=grouped.loc[i, 2],
#                                   endtime=grouped.loc[i, 3],
#                                   size=input_size))
#         for adm_id in set(admission_ids_list) - set(adm_arr):
#             yield (adm_id, InpatientInput.empty(input_size))
#
#     def observables_extractor(self, admission_ids_list, num_workers):
#         c_adm_id = self.colname["obs"].admission_id
#         c_time = self.colname["obs"].timestamp
#         c_value = self.colname["obs"].value
#         c_code_index = self.colname["obs"].code_source_index
#
#         df = self.df["obs"][[c_adm_id, c_time, c_value, c_code_index]]
#         logging.debug("obs: filter adms")
#         df = df[df[c_adm_id].isin(admission_ids_list)]
#
#         obs_dim = len(self.scheme.obs)
#
#         def ret_put(a, *args):
#             np.put(a, *args)
#             return a
#
#         def val_mask(x):
#             idx = x[c_code_index]
#             val = ret_put(np.zeros(obs_dim, dtype=np.float16), idx, x[c_value])
#             mask = ret_put(np.zeros(obs_dim, dtype=bool), idx, 1.0)
#             adm_id = x.index[0]
#             time = x[c_time].iloc[0]
#             return pd.Series({0: adm_id, 1: time, 2: val, 3: mask})
#
#         def gen_observation(val_mask):
#             time = val_mask[1].to_numpy()
#             value = val_mask[2]
#             mask = val_mask[3]
#             mask = np.vstack(mask.values).reshape((len(time), obs_dim))
#             value = np.vstack(value.values).reshape((len(time), obs_dim))
#             return InpatientObservables(time=time, value=value, mask=mask)
#
#         def partition_fun(part_df):
#             g = part_df.groupby([c_adm_id, c_time], sort=True, as_index=False)
#             return g.apply(val_mask).groupby(0).apply(gen_observation)
#
#         logging.debug("obs: dasking")
#         df = df.set_index(c_adm_id)
#         df = dd.from_pandas(df, npartitions=12, sort=True)
#         logging.debug("obs: groupby")
#         obs_obj_df = df.map_partitions(partition_fun, meta=(None, object))
#         logging.debug("obs: undasking")
#         obs_obj_df = obs_obj_df.compute()
#         logging.debug("obs: extract")
#
#         collected_adm_ids = obs_obj_df.index.tolist()
#         assert len(collected_adm_ids) == len(set(collected_adm_ids)), \
#             "Duplicate admission ids in obs"
#
#         for adm_id, obs in obs_obj_df.items():
#             yield (adm_id, obs)
#
#         logging.debug("obs: empty")
#         for adm_id in set(admission_ids_list) - set(obs_obj_df.index):
#             yield (adm_id, InpatientObservables.empty(obs_dim))
#

ADMISSIONS_CONF = AdmissionSQLTableConfig(query=(r"""
select hadm_id as {admission_id_alias}, 
subject_id as {subject_id_alias},
admittime as {admission_time_alias},
dischtime as {discharge_time_alias}
from mimiciv_hosp.admissions 
"""))

STATIC_CONF = StaticSQLTableConfig(query=(r"""
select 
p.subject_id as {subject_id_alias},
p.gender as {gender_alias},
a.race as {race_alias},
p.anchor_age as {anchor_age_alias},
p.anchor_year as {anchor_year_alias}
from mimiciv_hosp.patients p
left join 
(select subject_id, max(race) as race
from mimiciv_hosp.admissions
group by subject_id) as a
on p.subject_id = a.subject_id
"""),
                                   gender_space_query=(r"""
select distinct gender as {gender_alias} from mimiciv_hosp.patients
where gender is not null
"""),
                                   race_space_query=(r"""
select distinct race as {race_alias} from mimiciv_hosp.admissions
where race is not null
"""))

DX_DISCHARGE_CONF = AdmissionMixedICDSQLTableConfig(query=(r"""
select hadm_id as {admission_id_alias}, 
        icd_code as {icd_code_alias}, 
        icd_version as {icd_version_alias}
from mimiciv_hosp.diagnoses_icd 
"""),
                                                    space_query=(r"""
select icd_code as {icd_code_alias},
        icd_version as {icd_version_alias},
        long_title as {description_alias}
from mimiciv_hosp.d_icd_diagnoses
"""))

RENAL_OUT_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="renal_out",
                                                               attributes=['uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr'],
                                                               query=(r"""
select icu.hadm_id {admission_id_alias}, {attributes}, charttime {time_alias}
from mimiciv_derived.kdigo_uo as uo
inner join mimiciv_icu.icustays icu
on icu.stay_id = uo.stay_id
    """))

RENAL_CREAT_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="renal_creat",
                                                                 attributes=['creat'],
                                                                 query=(r"""
select hadm_id {admission_id_alias}, {attributes}, charttime {time_alias} 
from mimiciv_derived.kdigo_creatinine
    """))

RENAL_AKI_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="renal_aki",
                                                               attributes=['aki_stage_smoothed', 'aki_binary'],
                                                               query=(r"""
select hadm_id {admission_id_alias}, {attributes}, charttime {time_alias} 
from (select hadm_id, charttime, aki_stage_smoothed, 
    case when aki_stage_smoothed = 1 then 1 else 0 end as aki_binary from mimiciv_derived.kdigo_stages)
    """))

SOFA_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="renal_aki",
                                                          attributes=["sofa_24hours"],
                                                          query=(r"""
select hadm_id {admission_id_alias}, {attributes}, s.endtime {time_alias} 
from mimiciv_derived.sofa as s
inner join mimiciv_icu.icustays icu on s.stay_id = icu.stay_id    
"""))

BLOOD_GAS_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="blood_gas",
                                                               attributes=['so2', 'po2', 'pco2', 'fio2',
                                                                           'fio2_chartevents',
                                                                           'aado2', 'aado2_calc',
                                                                           'pao2fio2ratio', 'ph',
                                                                           'baseexcess', 'bicarbonate', 'totalco2',
                                                                           'hematocrit',
                                                                           'hemoglobin',
                                                                           'carboxyhemoglobin', 'methemoglobin',
                                                                           'chloride', 'calcium', 'temperature',
                                                                           'potassium',
                                                                           'sodium', 'lactate', 'glucose'],
                                                               query=(r"""
select hadm_id {admission_id_alias}, {attributes}, charttime {time_alias} 
from mimiciv_derived.bg as bg
where hadm_id is not null
"""))

BLOOD_CHEMISTRY_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="blood_chemistry",
                                                                     attributes=['albumin', 'globulin', 'total_protein',
                                                                                 'aniongap', 'bicarbonate', 'bun',
                                                                                 'calcium', 'chloride',
                                                                                 'creatinine', 'glucose', 'sodium',
                                                                                 'potassium'],
                                                                     query=(r"""
select hadm_id {admission_id_alias}, {attributes}, charttime {time_alias} 
from mimiciv_derived.chemistry as ch
where hadm_id is not null
"""))

CARDIAC_MARKER_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="cardiac_marker",
                                                                    attributes=['troponin_t', 'ntprobnp', 'ck_mb'],
                                                                    query=(r"""
with trop as
(
    select specimen_id, MAX(valuenum) as troponin_t
    from mimiciv_hosp.labevents
    where itemid = 51003
    group by specimen_id
)
select hadm_id {admission_id_alias}, {attributes}, charttime {time_alias}
from mimiciv_hosp.admissions a
left join mimiciv_derived.cardiac_marker c
  on a.hadm_id = c.hadm_id
left join trop
  on c.specimen_id = trop.specimen_id
where c.hadm_id is not null
"""))

WEIGHT_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="weight",
                                                            attributes=['weight'],
                                                            query=(r"""
select icu.hadm_id {admission_id_alias}, {attributes}, w.time_bin {time_alias}
 from (
 (select stay_id, w.weight, w.starttime time_bin
  from mimiciv_derived.weight_durations as w)
 union all
 (select stay_id, w.weight, w.endtime time_bin
     from mimiciv_derived.weight_durations as w)
 ) w
inner join mimiciv_icu.icustays icu on w.stay_id = icu.stay_id
"""))

CBC_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="cbc",
                                                         attributes=['hematocrit', 'hemoglobin', 'mch', 'mchc', 'mcv',
                                                                     'platelet',
                                                                     'rbc', 'rdw', 'wbc'],
                                                         query=(r"""
select hadm_id {admission_id_alias}, {attributes}, cbc.charttime {time_alias}
from mimiciv_derived.complete_blood_count as cbc
where hadm_id is not null
"""))

VITAL_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="vital",
                                                           attributes=['heart_rate', 'sbp', 'dbp', 'mbp', 'sbp_ni',
                                                                       'dbp_ni',
                                                                       'mbp_ni', 'resp_rate',
                                                                       'temperature', 'spo2',
                                                                       'glucose'],
                                                           query=(r"""
select hadm_id {admission_id_alias}, {attributes}, v.charttime {time_alias}
from mimiciv_derived.vitalsign as v
inner join mimiciv_icu.icustays as icu
 on icu.stay_id = v.stay_id
"""))

# Glasgow Coma Scale, a measure of neurological function
GCS_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="gcs",
                                                         attributes=['gcs', 'gcs_motor', 'gcs_verbal', 'gcs_eyes',
                                                                     'gcs_unable'],
                                                         query=(r"""
select hadm_id {admission_id_alias}, {attributes}, gcs.charttime {time_alias}
from mimiciv_derived.gcs as gcs
inner join mimiciv_icu.icustays as icu
 on icu.stay_id = gcs.stay_id
"""))

OBS_TABLE_CONFIG = AdmissionTimestampedCodedValueSQLTableConfig(components=[
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
    GCS_CONF
])

## Inputs - Canonicalise

ICU_INPUT_CONF = RatedInputSQLTableConfig(query=(r"""
select
    a.hadm_id as {admission_id_alias}
    , inp.itemid as {code_alias}
    , inp.starttime as {start_time_alias}
    , inp.endtime as {end_time_alias}
    , di.label as {description_alias}
    , inp.rate  as {rate_alias}
    , inp.amount as {amount_alias}
    , inp.rateuom as {rate_unit_alias}
    , inp.amountuom as {amount_unit_alias}
from mimiciv_hosp.admissions a
inner join mimiciv_icu.icustays i
    on a.hadm_id = i.hadm_id
left join mimiciv_icu.inputevents inp
    on i.stay_id = inp.stay_id
left join mimiciv_icu.d_items di
    on inp.itemid = di.itemid
"""),
                                          space_query=(r"""
select di.itemid as {code_alias}, max(di.label) as {description_alias}
from  mimiciv_icu.d_items di
inner join mimiciv_icu.inputevents ie
    on di.itemid = ie.itemid
where di.itemid is not null
group by di.itemid
"""))

## Procedures - Canonicalise and Refine
ICU_PROC_CONF = IntervalICUProcedureSQLTableConfig(query=(r"""
select a.hadm_id as {admission_id_alias}
    , pe.itemid as {code_alias}
    , pe.starttime as {start_time_alias}
    , pe.endtime as {end_time_alias}
    , di.label as {description_alias}
from mimiciv_hosp.admissions a
inner join mimiciv_icu.icustays i
    on a.hadm_id = i.hadm_id
left join mimiciv_icu.procedureevents pe
    on i.stay_id = pe.stay_id
left join mimiciv_icu.d_items di
    on pe.itemid = di.itemid
"""),
                                                   space_query=(r"""
select di.itemid as {code_alias}, max(di.label) as {description_alias}
from  mimiciv_icu.d_items di
inner join mimiciv_icu.procedureevents pe
    on di.itemid = pe.itemid
where di.itemid is not null
group by di.itemid
"""))

HOSP_PROC_CONF = AdmissionIntervalBasedMixedICDTableConfig(query=(r"""
select pi.hadm_id as {admission_id_alias}
, (pi.chartdate)::timestamp as {start_time_alias}
, (pi.chartdate + interval '1 hour')::timestamp as {end_time_alias}
, pi.icd_code as {icd_code_alias}
, pi.icd_version as {icd_version_alias}
, di.long_title as {description_alias}
FROM mimiciv_hosp.procedures_icd pi
INNER JOIN mimiciv_hosp.d_icd_procedures di
  ON pi.icd_version = di.icd_version
  AND pi.icd_code = di.icd_code
INNER JOIN mimiciv_hosp.admissions a
  ON pi.hadm_id = a.hadm_id
"""),
                                                           space_query=(r"""

select di.icd_code  as {icd_code_alias},  
di.icd_version as {icd_version_alias},
max(di.long_title) as {description_alias}
from mimiciv_hosp.d_icd_procedures di
where di.icd_code is not null and di.icd_version is not null
group by di.icd_code, di.icd_version
"""))
