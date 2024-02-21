"""."""
from __future__ import annotations

import logging
import warnings
from collections import defaultdict
from dataclasses import field
from functools import cached_property
from typing import Dict, List, Optional, Iterable, Tuple

import pandas as pd
import sqlalchemy
from sqlalchemy import Engine

from lib.base import Module, Config
from lib.ehr.coding_scheme import (CodingSchemeConfig, FlatScheme, CodeMap, resources_dir, CodeMapConfig)
from lib.ehr.dataset import (DatasetScheme, StaticTableConfig,
                             AdmissionTimestampedMultiColumnTableConfig, AdmissionIntervalBasedCodedTableConfig,
                             AdmissionTimestampedCodedValueTableConfig, AdmissionLinkedCodedValueTableConfig,
                             TableConfig, CodedTableConfig, AdmissionTableConfig,
                             RatedInputTableConfig, DatasetTablesConfig,
                             DatasetTables, DatasetConfig, DatasetSchemeConfig)
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

    @staticmethod
    def fix_dots(df: pd.DataFrame, c_icd_code: str, c_icd_version: str,
                 icd_schemes: Dict[str, ICDScheme]) -> pd.DataFrame:
        df = df.copy()
        for version, icd_df in df.groupby(c_icd_version):
            scheme = icd_schemes[version]
            df.loc[icd_df.index, c_icd_code] = \
                icd_df[c_icd_code].str.replace(' ', '').str.replace('.', '').map(scheme.add_dots)
        return df

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

        df = cls.fix_dots(icd_version_selection, icd_code_alias, icd_version_alias,
                          icd_schemes_loaded)
        df['code'] = (df[icd_version_alias] + sep + df[icd_code_alias]).tolist()
        desc = df.set_index('code')[description_alias].to_dict()

        return MixedICDScheme(config=CodingSchemeConfig(name),
                              codes=df['code'].tolist(),
                              desc=desc,
                              icd_schemes=icd_schemes_loaded,
                              sep=sep)

    def mixedcode_format_table(self, table: pd.DataFrame, icd_code_alias: str,
                               icd_version_alias: str, code_alias: str) -> pd.DataFrame:
        # TODO: test this method.
        """
        Format a table with mixed codes to the ICD version:icd_code format and filter out codes that are not in the scheme.
        """
        assert icd_version_alias in table.columns, f"Column {icd_version_alias} not found."
        assert icd_code_alias in table.columns, f"Column {icd_code_alias} not found."
        assert table[icd_version_alias].isin(self._icd_schemes).all(), \
            f"Only ICD version {list(self._icd_schemes.keys())} are expected."

        table = self.fix_dots(table, icd_code_alias, icd_version_alias, self._icd_schemes)

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
            conf = CodeMapConfig(source_scheme=self.name, target_scheme=pure_scheme.name)
            CodeMap.register_map(CodeMap(conf, mixed2pure))

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

    def register_map(self, target_name: str, mapping: pd.DataFrame,
                     c_code: str, c_icd_code: str, c_icd_version: str,
                     c_target_code: str, c_target_desc: str) -> None:
        """
        Register a mapping between the current Mixed ICD scheme and a target scheme.
        """
        # TODO: test this method.
        mapping = self.fix_dots(mapping.astype(str), c_icd_code, c_icd_version, self._icd_schemes)
        mapping[c_code] = (mapping[c_icd_version] + self._sep + mapping[c_icd_code]).tolist()
        mapping = mapping[mapping[c_code].isin(self.codes)]
        assert len(mapping) > 0, "No mapping between the Mixed ICD scheme and the target scheme was found."
        tcodes = sorted(mapping[c_target_code].drop_duplicates().tolist())
        tdesc = mapping.set_index(c_target_code)[c_target_desc].to_dict()
        FlatScheme.register(FlatScheme(CodingSchemeConfig(target_name), codes=tcodes, desc=tdesc))

        mapping = mapping[[c_code, c_target_code]].astype(str)
        mapping = mapping[mapping[c_code].isin(self.codes) & mapping[c_target_code].isin(tcodes)]
        mapping = mapping.groupby(c_code)[c_target_code].apply(set).to_dict()
        CodeMap.register_map(CodeMap(CodeMapConfig(self.name, target_name), mapping))

    def as_dataframe(self):
        columns = ['code', 'desc', 'code_index', 'icd_version', 'icd_code']
        return pd.DataFrame([(c, self.desc[c], self.index[c], *c.split(self._sep)) for c in self.codes],
                            columns=columns)


class AggregatedICUInputsScheme(FlatScheme):
    aggregation: Dict[str, str]

    def __init__(self, config: CodingSchemeConfig, codes: List[str],
                 desc: Dict[str, str],
                 aggregation: Dict[str, str]):
        super().__init__(config, codes=codes, desc=desc)
        self.aggregation = aggregation

    @staticmethod
    def register_aggregated_scheme(scheme: FlatScheme, target_name: Optional[str], mapping: Optional[pd.DataFrame],
                                   c_code: str, c_target_code: str, c_target_desc: str,
                                   c_target_aggregation: str) -> FlatScheme:
        """
        Register a target scheme and its mapping.
        """

        tscheme_conf = CodingSchemeConfig(target_name)
        tcodes = sorted(mapping[c_target_code].drop_duplicates().astype(str).tolist())
        tdesc = mapping.set_index(c_target_code)[c_target_desc].to_dict()
        tagg = mapping.set_index(c_target_code)[c_target_aggregation].to_dict()
        tscheme = AggregatedICUInputsScheme(tscheme_conf, codes=tcodes, desc=tdesc, aggregation=tagg)
        AggregatedICUInputsScheme.register_scheme(tscheme)

        mapping = mapping[[c_code, c_target_code]].astype(str)
        mapping = mapping[mapping[c_code].isin(scheme.codes) & mapping[c_target_code].isin(tscheme.codes)]
        mapping = mapping.groupby(c_code)[c_target_code].apply(set).to_dict()
        CodeMap.register_map(CodeMap(CodeMapConfig(scheme.name, tscheme.name), mapping))
        return tscheme


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

    def __call__(self, engine: Engine):
        query = self.config.query.format(**self.config.alias_dict)
        return self._coerce_id_to_str(pd.read_sql(query, engine,
                                                  coerce_float=False))


class CategoricalSQLTable(SQLTable):
    # TODO: Document this class.

    config: CodedSQLTableConfig

    def _coerce_code_to_str(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Some of the integer codes in the database when downloaded are stored as floats or integers.
        A fix is to coerce them to integers then fix as strings.
        """
        return self._coerce_columns_to_str(df, self.config.coded_cols)

    def space(self, engine: Engine):
        query = self.config.space_query.format(**self.config.alias_dict)
        return self._coerce_code_to_str(self._coerce_id_to_str(pd.read_sql(query, engine, coerce_float=False)))


class ObservablesSQLTable(SQLTable):
    # TODO: Document this class.mimi

    config: AdmissionTimestampedCodedValueSQLTableConfig

    def __call__(self, engine: Engine, obs_scheme: ObservableMIMICScheme) -> pd.DataFrame:
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
                        icd_version_selection: Optional[pd.DataFrame],
                        target_name: Optional[str] = None,
                        mapping: Optional[pd.DataFrame] = None) -> MixedICDScheme:
        c_version = self.config.icd_version_alias
        c_code = self.config.icd_code_alias
        c_desc = self.config.description_alias
        scheme = self._register_scheme(name=name, icd_schemes=icd_schemes,
                                       supported_space=self.space(engine),
                                       icd_version_selection=icd_version_selection,
                                       c_version=c_version, c_code=c_code, c_desc=c_desc)

        if target_name is not None and mapping is not None:
            scheme.register_map(target_name, mapping,
                                c_code=c_code, c_icd_code=c_code, c_icd_version=c_version,
                                c_target_code=f'target_{c_code}',
                                c_target_desc=f'target_{c_desc}')
        return scheme

    def _coerce_version_to_str(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Some of the integer codes in the database when downloaded are stored as floats or integers.
        A fix is to coerce them to integers then fix as strings.
        """
        return self._coerce_columns_to_str(df, (self.config.icd_version_alias,))

    def space(self, engine: Engine) -> pd.DataFrame:
        df = super().space(engine)
        return self._coerce_version_to_str(df)


class CodedSQLTable(CategoricalSQLTable):
    # TODO: Document this class.

    config: CodedSQLTableConfig

    @staticmethod
    def _register_scheme(name: str,
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

    def register_target_scheme(self, target_name: Optional[str], mapping: pd.DataFrame,
                               c_code: str, c_target_code: str, c_target_desc: str) -> FlatScheme:
        """
        Register a target scheme and its mapping.
        """
        tscheme_conf = CodingSchemeConfig(target_name)
        tcodes = sorted(mapping[c_target_code].drop_duplicates().astype(str).tolist())
        tdesc = mapping.set_index(c_target_code)[c_target_desc].to_dict()
        tscheme = FlatScheme(tscheme_conf, codes=tcodes, desc=tdesc)
        FlatScheme.register_scheme(tscheme)

        mapping = mapping[[c_code, c_target_code]].astype(str)
        mapping = mapping[mapping[c_code].isin(self.codes) & mapping[c_target_code].isin(tscheme.codes)]
        mapping = mapping.groupby(c_code)[c_target_code].apply(set).to_dict()
        CodeMap.register_map(CodeMap(CodeMapConfig(self.name, tscheme.name), mapping))
        return tscheme

    def register_scheme(self, name: str,
                        engine: Engine, code_selection: Optional[pd.DataFrame]) -> FlatScheme:
        return self._register_scheme(name=name, supported_space=self.space(engine),
                                     code_selection=code_selection, c_code=self.config.code_alias,
                                     c_desc=self.config.description_alias)


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
                               gender_selection: Optional[pd.DataFrame],
                               target_name: Optional[str] = None,
                               mapping: Optional[pd.DataFrame] = None):
        scheme = CodedSQLTable._register_scheme(name=name,
                                                supported_space=self.gender_space(engine),
                                                code_selection=gender_selection,
                                                c_code=self.config.gender_alias,
                                                c_desc=self.config.gender_alias)
        if target_name is not None and mapping is not None:
            scheme.register_target_scheme(target_name, mapping,
                                          c_code=self.config.gender_alias,
                                          c_target_code=f'target_{self.config.gender_alias}',
                                          c_target_desc=f'target_{self.config.gender_alias}')
        return scheme

    def register_ethnicity_scheme(self, name: str,
                                  engine: Engine,
                                  ethnicity_selection: Optional[pd.DataFrame],
                                  target_name: Optional[str] = None,
                                  mapping: Optional[pd.DataFrame] = None):
        scheme = CodedSQLTable._register_scheme(name=name,
                                                supported_space=self.ethnicity_space(engine),
                                                code_selection=ethnicity_selection,
                                                c_code=self.config.race_alias,
                                                c_desc=self.config.race_alias)

        if target_name is not None and mapping is not None:
            scheme.register_target_scheme(target_name, mapping,
                                          c_code=self.config.race_alias,
                                          c_target_code=f'target_{self.config.race_alias}',
                                          c_target_desc=f'target_{self.config.race_alias}')
        return scheme


class MIMICIVSQLTablesConfig(DatasetTablesConfig):
    # TODO: Document this class.

    host: str
    port: int
    user: str
    password: str
    dbname: str

    static: StaticSQLTableConfig = field(default_factory=lambda: STATIC_CONF, kw_only=True)
    admissions: AdmissionSQLTableConfig = field(default_factory=lambda: ADMISSIONS_CONF, kw_only=True)
    dx_discharge: AdmissionMixedICDSQLTableConfig = field(default_factory=lambda: DX_DISCHARGE_CONF, kw_only=True)
    obs: AdmissionTimestampedCodedValueSQLTableConfig = field(default_factory=lambda: OBS_TABLE_CONFIG,
                                                              kw_only=True)
    icu_procedures: IntervalICUProcedureSQLTableConfig = field(default_factory=lambda: ICU_PROC_CONF, kw_only=True)
    icu_inputs: RatedInputSQLTableConfig = field(default_factory=lambda: ICU_INPUT_CONF, kw_only=True)
    hosp_procedures: AdmissionIntervalBasedMixedICDTableConfig = field(default_factory=lambda: HOSP_PROC_CONF,
                                                                       kw_only=True)


class MIMICIVDatasetSchemeMapsFiles(Config):
    gender: Optional[str] = 'gender.csv'
    ethnicity: Optional[str] = 'ethnicity.csv'
    icu_inputs: Optional[str] = 'icu_inputs.csv'
    icu_procedures: Optional[str] = 'icu_procedures.csv'
    hosp_procedures: Optional[str] = 'hosp_procedures.csv'


class MIMICIVDatasetSchemeSelectionFiles(Config):
    gender: Optional[str] = 'gender.csv'
    ethnicity: Optional[str] = 'ethnicity.csv'
    icu_inputs: Optional[str] = 'icu_inputs.csv'
    icu_procedures: Optional[str] = 'icu_procedures.csv'
    hosp_procedures: Optional[str] = 'hosp_procedures.csv'
    obs: Optional[str] = 'obs.csv'
    dx_discharge: Optional[str] = 'dx_discharge.csv'


class MIMICIVDatasetSchemeConfig(DatasetSchemeConfig):
    name_prefix: str = ''
    name_separator: str = '.'
    resources_dir: str = ''
    selection_subdir: str = 'selection'
    map_subdir: str = 'map'
    map_files: MIMICIVDatasetSchemeMapsFiles = field(default_factory=lambda: MIMICIVDatasetSchemeMapsFiles(),
                                                     kw_only=True)
    selection_files: MIMICIVDatasetSchemeSelectionFiles = field(
        default_factory=lambda: MIMICIVDatasetSchemeSelectionFiles(),
        kw_only=True)
    icu_inputs_uom_normalization: Optional[Tuple[str]] = ("uom_normalization", "icu_inputs.csv")
    icu_inputs_aggregation_column: Optional[str] = 'aggregation'

    @property
    def uom_normalization_file(self) -> pd.DataFrame:
        return pd.read_csv(resources_dir(self.resources_dir, *self.icu_inputs_uom_normalization))

    def print_configuration_layout_disk(self):
        raise NotImplementedError()

    def create_empty_configuration_layout_disk(self):
        raise NotImplementedError()

    def scan_layout_disk(self):
        raise NotImplementedError()

    def selection_file(self, path: str) -> pd.DataFrame:
        return pd.read_csv(resources_dir(self.resources_dir, self.selection_subdir, path))

    def map_file(self, path: str) -> Optional[pd.DataFrame]:
        try:
            return pd.read_csv(resources_dir(self.resources_dir, self.map_subdir, path))
        except FileNotFoundError:
            return None

    def scheme_name(self, key: str) -> str:
        return f'{self.name_prefix}{self.name_separator}{key}'

    def target_scheme_name(self, key: str) -> str:
        return self.scheme_name(f'target_{key}')

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
        return self.selection_file(self.selection_files.obs)

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


class MIMICIVDatasetScheme(DatasetScheme):
    ethnicity: FlatScheme
    gender: FlatScheme
    dx_discharge: MixedICDScheme
    obs: Optional[ObservableMIMICScheme] = None
    icu_procedures: Optional[FlatScheme] = None
    hosp_procedures: Optional[MixedICDScheme] = None
    icu_inputs: Optional[FlatScheme] = None


class MIMICIVSQLConfig(DatasetConfig):
    tables: MIMICIVSQLTablesConfig
    scheme: MIMICIVDatasetSchemeConfig


class MIMICIVSQLTablesInterface(Module):
    # TODO: Document this class.

    config: MIMICIVSQLTablesConfig

    def create_engine(self) -> Engine:
        return sqlalchemy.create_engine(
            f'postgresql+psycopg2://{self.config.user}:{self.config.password}@'
            f'{self.config.host}:{self.config.port}/{self.config.dbname}')

    def register_gender_scheme(self, config: MIMICIVDatasetSchemeConfig) -> FlatScheme:
        """
        TODO: document me.
        """
        table = StaticSQLTable(self.config.static)
        return table.register_gender_scheme(config.scheme_name("gender"), self.create_engine(), config.gender_selection,
                                            config.target_scheme_name("gender"), config.gender_map)

    def register_ethnicity_scheme(self, config: MIMICIVDatasetSchemeConfig):
        """
        TODO: document me.
        """
        table = StaticSQLTable(self.config.static)
        return table.register_ethnicity_scheme(config.scheme_name("ethnicity"), self.create_engine(),
                                               config.ethnicity_selection,
                                               config.target_scheme_name("ethnicity"), config.ethnicity_map)

    def register_obs_scheme(self, config: MIMICIVDatasetSchemeConfig):
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
        return table.register_scheme(config.scheme_name("obs"), self.create_engine(), config.obs_selection)

    def register_icu_inputs_scheme(self, config: MIMICIVDatasetSchemeConfig):
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
        name = config.scheme_name("icu_inputs")
        code_selection = config.icu_inputs_selection
        mapping = config.icu_inputs_map
        c_aggregation = config.icu_inputs_aggregation_column
        target_name = config.target_scheme_name("icu_inputs")

        table = CodedSQLTable(self.config.icu_inputs)
        scheme = table.register_scheme(name, self.create_engine(), code_selection)
        if mapping is not None and c_aggregation is not None:
            AggregatedICUInputsScheme.register_aggregated_scheme(
                scheme, target_name, mapping,
                c_code=self.config.icu_inputs.code_alias,
                c_target_code=config.target_column_name(self.config.icu_inputs.code_alias),
                c_target_desc=config.target_column_name(self.config.icu_inputs.description_alias),
                c_target_aggregation=c_aggregation)
        return scheme

    def register_icu_procedures_scheme(self, config: MIMICIVDatasetSchemeConfig):
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
        scheme = table.register_scheme(config.scheme_name("icu_procedures"), self.create_engine(),
                                       config.icu_procedures_selection)
        mapping = config.icu_procedures_map
        if mapping is not None:
            c_target_code = config.target_column_name(self.config.icu_procedures.code_alias)
            c_target_desc = config.target_column_name(self.config.icu_procedures.description_alias)
            table.register_target_scheme(config.target_scheme_name("icu_procedures"), mapping,
                                         c_code=self.config.icu_procedures.code_alias,
                                         c_target_code=c_target_code,
                                         c_target_desc=c_target_desc)
        return scheme

    def register_hosp_procedures_scheme(self, config: MIMICIVDatasetSchemeConfig):
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
        return table.register_scheme(name=config.scheme_name("hosp_procedures"),
                                     engine=self.create_engine(),
                                     icd_schemes={'9': 'pr_icd9', '10': 'pr_flat_icd10'},
                                     icd_version_selection=config.hosp_procedures_selection,
                                     target_name=None, mapping=None)

    def register_dx_discharge_scheme(self, config: MIMICIVDatasetSchemeConfig):
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
        return table.register_scheme(config.scheme_name("dx_discharge"), self.create_engine(),
                                     {'9': 'dx_icd9', '10': 'dx_flat_icd10'},
                                     config.dx_discharge_selection)

    @cached_property
    def supported_gender(self) -> pd.DataFrame:
        table = StaticSQLTable(self.config.static)
        return table.gender_space(self.create_engine())

    @cached_property
    def supported_ethnicity(self) -> pd.DataFrame:
        table = StaticSQLTable(self.config.static)
        return table.ethnicity_space(self.create_engine())

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
        return dx_discharge_scheme.mixedcode_format_table(dataframe, c_icd_code, c_icd_version)

    def _extract_obs_table(self, engine: Engine, obs_scheme: ObservableMIMICScheme) -> pd.DataFrame:
        table = ObservablesSQLTable(self.config.obs)
        return table(engine, obs_scheme)

    def _extract_icu_procedures_table(self, engine: Engine, icu_procedure_scheme: FlatScheme) -> pd.DataFrame:
        table = SQLTable(self.config.icu_procedures)
        c_code = self.config.icu_procedures.code_alias
        dataframe = table(engine)
        dataframe = dataframe[dataframe[c_code].isin(icu_procedure_scheme.codes)]
        dataframe[c_code] = dataframe[c_code]
        return dataframe.reset_index(drop=True)

    def _extract_icu_inputs_table(self, engine: Engine, icu_input_scheme: FlatScheme) -> pd.DataFrame:
        table = CodedSQLTable(self.config.icu_inputs)
        c_code = self.config.icu_inputs.code_alias
        dataframe = table(engine)
        dataframe = dataframe[dataframe[c_code].isin(icu_input_scheme.codes)]
        dataframe[c_code] = dataframe[c_code]
        return dataframe.reset_index(drop=True)

    def _extract_hosp_procedures_table(self, engine: Engine,
                                       procedure_icd_scheme: MixedICDScheme) -> pd.DataFrame:
        table = MixedICDSQLTable(self.config.hosp_procedures)
        c_icd_code = self.config.hosp_procedures.icd_code_alias
        c_icd_version = self.config.hosp_procedures.icd_version_alias
        c_code = self.config.hosp_procedures.code_alias
        dataframe = table(engine)
        return procedure_icd_scheme.mixedcode_format_table(dataframe, c_icd_code, c_icd_version, c_code)

    def dataset_scheme_from_selection(self, config: MIMICIVDatasetSchemeConfig) -> MIMICIVDatasetScheme:
        """
        Create a dataset scheme from the given config.
        TODO: document me.
        Returns:
            (CodingScheme.DatasetScheme) A new scheme that is also registered in the current runtime.
        """
        return MIMICIVDatasetScheme(config=config,
                                    gender=self.register_gender_scheme(config),
                                    ethnicity=self.register_ethnicity_scheme(config),
                                    icu_inputs=self.register_icu_inputs_scheme(config),
                                    icu_procedures=self.register_icu_procedures_scheme(config),
                                    hosp_procedures=self.register_hosp_procedures_scheme(config),
                                    dx_discharge=self.register_dx_discharge_scheme(config),
                                    obs=self.register_obs_scheme(config))

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
    case when aki_stage_smoothed = 0 then 0 else 1 end as aki_binary from mimiciv_derived.kdigo_stages)
    """))

SOFA_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="sofa",
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
                                                                     attributes=['albumin', 'globulin',
                                                                                 'total_protein',
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
                                                         attributes=['hematocrit', 'hemoglobin', 'mch', 'mchc',
                                                                     'mcv',
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

# Intracranial pressure
ICP_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="icp",
                                                         attributes=['icp'],
                                                         query=(r"""
select icu.hadm_id {admission_id_alias}, {attributes}, icp.charttime {time_alias}
from mimiciv_derived.icp as icp
where hadm_id is not null
inner join mimiciv_icu.icustays as icu
 on icu.stay_id = icp.stay_id
"""))

# Inflammation
INFLAMMATION_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="inflammation",
                                                                  attributes=['crp'],
                                                                  query=(r"""
select hadm_id {admission_id_alias}, {attributes}, charttime {time_alias}
from mimiciv_derived.inflammation
where hadm_id is not null
"""))

# Coagulation
COAGULATION_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="coagulation",
                                                                 attributes=['pt', 'ptt', 'inr', 'd_dimer',
                                                                             'fibrinogen', 'thrombin'],
                                                                 query=(r"""
select hadm_id {admission_id_alias}, {attributes}, charttime {time_alias}
from mimiciv_derived.coagulation
where hadm_id is not null
"""))

# Blood differential
BLOOD_DIFF_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="blood_diff",
                                                                attributes=['neutrophils', 'lymphocytes',
                                                                            'monocytes',
                                                                            'eosinophils', 'basophils',
                                                                            'atypical_lymphocytes',
                                                                            'bands', 'immature_granulocytes',
                                                                            'metamyelocytes',
                                                                            'nrbc',
                                                                            'basophils_abs', 'eosinophils_abs',
                                                                            'lymphocytes_abs',
                                                                            'monocytes_abs', 'neutrophils_abs'],
                                                                query=(r"""
select hadm_id {admission_id_alias}, {attributes}, charttime {time_alias}
from mimiciv_derived.blood_differential
where hadm_id is not null
"""))

# Enzymes
ENZYMES_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="enzymes",
                                                             attributes=['ast', 'alt', 'alp', 'ld_ldh', 'ck_cpk',
                                                                         'ck_mb',
                                                                         'amylase', 'ggt', 'bilirubin_direct',
                                                                         'bilirubin_total', 'bilirubin_indirect'],
                                                             query=(r"""
select hadm_id {admission_id_alias}, {attributes}, charttime {time_alias}
from mimiciv_derived.enzyme
where hadm_id is not null
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
    GCS_CONF,
    ICP_CONF,
    INFLAMMATION_CONF,
    COAGULATION_CONF,
    BLOOD_DIFF_CONF,
    ENZYMES_CONF
])

## Inputs - Canonicalise

ICU_INPUT_CONF = RatedInputSQLTableConfig(query=(r"""
select
    a.hadm_id as {admission_id_alias}
    , inp.itemid as {code_alias}
    , inp.starttime as {start_time_alias}
    , inp.endtime as {end_time_alias}
    , di.label as {description_alias}
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
