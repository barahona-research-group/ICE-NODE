import logging
from collections import defaultdict
from dataclasses import field
from functools import cached_property
from typing import Dict, Self

import pandas as pd

from lib.ehr.coding_scheme import FrozenDict11, FrozenDict1N, CodingScheme, CodingSchemesManager, \
    CodeMap, NumericScheme, ReducedCodeMapN1
from lib.ehr.example_schemes.icd import ICDScheme


class ObservableMIMICScheme(NumericScheme):
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
                    - type_hint: The type of the variable. 'B' for boolean, 'N' for numeric, 'O' for ordinal,
                        'C' for categorical.

        Returns:
            (CodingScheme.FlatScheme) A new scheme containing the variables in obs_variables.
        """
        # format codes to be of the form 'table_name.attribute'
        codes = tuple(sorted(obs_variables.index + '.' + obs_variables['attribute'].tolist()))
        desc = FrozenDict11.from_dict(dict(zip(codes, codes)))
        type_hint = FrozenDict11.from_dict(dict(zip(codes, obs_variables['type_hint'].tolist())))
        return cls(name=name,
                   codes=codes,
                   desc=desc,
                   type_hint=type_hint)

    def as_dataframe(self):
        columns = ['code', 'desc', 'type_hint', 'code_index', 'table_name', 'attribute']
        return pd.DataFrame([(c, self.desc[c], self.type_hint[c], self.index[c], *c.split('.')) for c in self.codes],
                            columns=columns)


class MixedICDScheme(CodingScheme):
    # TODO: Document this class.

    icd_version_schemes: FrozenDict11
    sep: str = ':'

    @cached_property
    def icd_schemes(self) -> Dict[str, ICDScheme]:
        return {k: self.context_view.scheme[v] for k, v in self.icd_version_schemes.items()}

    @staticmethod
    def fix_dots(df: pd.DataFrame, c_icd_code: str, c_icd_version: str,
                 icd_schemes: Dict[str, ICDScheme]) -> pd.DataFrame:
        df = df.copy()
        for version, icd_df in df.groupby(c_icd_version):
            scheme = icd_schemes[str(version)]
            df.loc[icd_df.index, c_icd_code] = \
                icd_df[c_icd_code].str.replace(' ', '').str.replace('.', '').map(scheme.ops.add_dots)
        return df

    @classmethod
    def from_selection(cls, manager: CodingSchemesManager, name: str, icd_version_selection: pd.DataFrame,
                       icd_version_alias: str, icd_code_alias: str, description_alias: str,
                       icd_version_schemes: FrozenDict11, sep: str = ':') -> Self:
        # TODO: test this method.
        icd_version_selection = icd_version_selection.sort_values([icd_version_alias, icd_code_alias])
        icd_version_selection = icd_version_selection.drop_duplicates([icd_version_alias, icd_code_alias]).astype(str)
        assert icd_version_selection[icd_version_alias].isin(icd_version_schemes).all(), \
            f"Only {', '.join(map(lambda x: f'ICD-{x}', icd_version_schemes))} are expected."

        # assert no duplicate (icd_code, icd_version)
        assert icd_version_selection.groupby([icd_version_alias, icd_code_alias]).size().max() == 1, \
            "Duplicate (icd_code, icd_version) pairs are not allowed."

        icd_schemes_loaded: Dict[str, ICDScheme] = {k: manager.scheme[v] for k, v in icd_version_schemes.items()}

        assert all(isinstance(s, ICDScheme) for s in icd_schemes_loaded.values()), \
            "Only ICD schemes are expected."

        df = cls.fix_dots(icd_version_selection, icd_code_alias, icd_version_alias,
                          icd_schemes_loaded)
        df['code'] = (df[icd_version_alias] + sep + df[icd_code_alias]).tolist()
        desc = df.set_index('code')[description_alias].to_dict()

        return MixedICDScheme(name=name,
                              codes=tuple(sorted(df['code'].tolist())),
                              desc=FrozenDict11.from_dict(desc),
                              icd_version_schemes=icd_version_schemes,
                              sep=sep)

    def mixedcode_format_table(self, table: pd.DataFrame, icd_code_alias: str,
                               icd_version_alias: str, code_alias: str) -> pd.DataFrame:
        # TODO: test this method.
        """
        Format a table with mixed codes to the ICD version:icd_code format and filter out codes that are not in the scheme.
        """
        assert icd_version_alias in table.columns, f"Column {icd_version_alias} not found."
        assert icd_code_alias in table.columns, f"Column {icd_code_alias} not found."
        assert table[icd_version_alias].isin(self.icd_schemes).all(), \
            f"Only ICD version {list(self.icd_schemes.keys())} are expected."

        table = self.fix_dots(table, icd_code_alias, icd_version_alias, self.icd_schemes)

        # the version:icd_code format.
        table[code_alias] = table[icd_version_alias] + self._sep + table[icd_code_alias]

        # filter out codes that are not in the scheme.
        return table[table[code_alias].isin(self.codes)].reset_index(drop=True)

    def register_standard_icd_maps(self, manager: CodingSchemesManager) -> CodingSchemesManager:
        """
        Register the mappings between the Mixed ICD scheme and the individual ICD scheme.
        For example, if the current `MixedICD` is mixing ICD-9 and ICD-10,
        then register the two mappings between this scheme and ICD-9 and ICD-10 separately.
        This assumes that the current runtime has already registered mappings
        between the individual ICD schemes.
        """

        # mixed2pure_maps has the form {icd_version: {mixed_code: {icd}}}.
        mixed2pure_maps: Dict[str, FrozenDict1N] = {}
        lost_codes = []
        dataframe = self.as_dataframe()
        for pure_version, pure_scheme in self.icd_schemes.items():
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
                    pure_map = manager.map[
                        (self.icd_schemes[mixed_version].name, self.icd_schemes[pure_version].name)]

                    for icd, mixed_code in icd2mixed.items():
                        if icd in pure_map:
                            mixed2pure[mixed_code].update(pure_map[icd])
                        else:
                            lost_codes.append(mixed_code)

            mixed2pure = FrozenDict1N.from_dict(mixed2pure)
            mixed2pure_maps[pure_version] = mixed2pure
        # register the mapping between the mixed and pure ICD schemes.
        for pure_version, mixed2pure in mixed2pure_maps.items():
            pure_scheme = self.icd_schemes[pure_version]
            manager = manager.add_map(CodeMap(source_name=self.name, target_name=pure_scheme.name, data=mixed2pure))

        lost_df = dataframe[dataframe['code'].isin(lost_codes)]
        if len(lost_df) > 0:
            logging.warning(f"Lost {len(lost_df)} codes when generating the mapping between the Mixed ICD"
                            "scheme and the individual ICD scheme.")
            logging.warning(lost_df.to_string().replace('\n', '\n\t'))

        return manager

    def register_map(self, manager: CodingSchemesManager, target_name: str, mapping: pd.DataFrame,
                     c_code: str, c_icd_code: str, c_icd_version: str,
                     c_target_code: str, c_target_desc: str) -> CodingSchemesManager:
        """
        Register a mapping between the current Mixed ICD scheme and a target scheme.
        """
        # TODO: test this method.
        mapping = self.fix_dots(mapping.astype(str), c_icd_code, c_icd_version, self._icd_schemes)
        mapping[c_code] = (mapping[c_icd_version] + self._sep + mapping[c_icd_code]).tolist()
        mapping = mapping[mapping[c_code].isin(self.codes)]
        assert len(mapping) > 0, "No mapping between the Mixed ICD scheme and the target scheme was found."
        target_codes = tuple(sorted(mapping[c_target_code].drop_duplicates().tolist()))
        target_desc = FrozenDict11.from_dict(mapping.set_index(c_target_code)[c_target_desc].to_dict())
        manager = manager.add_scheme(CodingScheme(name=target_name, codes=target_codes, desc=target_desc))

        mapping = mapping[[c_code, c_target_code]].astype(str)
        mapping = mapping[mapping[c_code].isin(self.codes) & mapping[c_target_code].isin(target_codes)]
        mapping = FrozenDict1N.from_dict(mapping.groupby(c_code)[c_target_code].apply(set).to_dict())
        return manager.add_map(CodeMap(source_name=self.name, target_name=target_name, data=mapping))

    def as_dataframe(self):
        columns = ['code', 'desc', 'code_index', 'icd_version', 'icd_code']
        return pd.DataFrame([(c, self.desc[c], self.index[c], *c.split(self._sep)) for c in self.codes],
                            columns=columns)


class AggregatedICUInputsScheme(CodingScheme):
    aggregation: FrozenDict11 = field(kw_only=True)

    @staticmethod
    def register_aggregated_scheme(manager: CodingSchemesManager,
                                   scheme: CodingScheme,
                                   target_scheme_name: str,
                                   code_column: str,
                                   target_code_column: str,
                                   target_desc_column: str,
                                   target_aggregation_column: str,
                                   mapping_table: pd.DataFrame) -> CodingSchemesManager:
        """
        Register a target scheme and its mapping.
        """
        target_codes = tuple(sorted(mapping_table[target_code_column].drop_duplicates().astype(str).tolist()))
        target_desc = FrozenDict11.from_dict(mapping_table.set_index(target_code_column)[target_desc_column].to_dict())
        target_agg = FrozenDict11.from_dict(
            mapping_table.set_index(target_code_column)[target_aggregation_column].to_dict())
        target_scheme = CodingScheme(name=target_scheme_name, codes=target_codes, desc=target_desc)
        manager = manager.add_scheme(target_scheme)

        mapping = mapping_table[[code_column, target_code_column]].astype(str)
        mapping = mapping[
            mapping[code_column].isin(scheme.codes) & mapping[target_code_column].isin(target_scheme.codes)]
        mapping = FrozenDict1N.from_dict(mapping.groupby(code_column)[target_code_column].apply(set).to_dict())
        return manager.add_map(ReducedCodeMapN1.from_data(source_name=scheme.name,
                                                          target_name=target_scheme.name,
                                                          map_data=mapping,
                                                          set_aggregation=target_agg))
