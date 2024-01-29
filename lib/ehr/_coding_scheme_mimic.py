from __future__ import annotations

import os
from typing import Dict, Set, List

import pandas as pd

from .coding_scheme import (CodingSchemeConfig, FlatScheme,
                            CodeMap, CodeMapConfig,
                            _RSC_DIR, Ethnicity)

ETH_SCHEME_FILE = 'mimic4_race_grouper.csv.gz'
ETH32_NAME = 'mimic4_eth32'
ETH32_COLNAME = 'eth32'
ETH5_NAME = 'mimic4_eth5'
ETH5_COLNAME = 'eth5'

MIMIC3_ETH_SCHEME_FILE = 'mimic3_race_grouper.csv.gz'
MIMIC3_ETH37_NAME = 'mimic3_eth37'
MIMIC3_ETH37_COLNAME = 'ETH37'
MIMIC3_ETH7_NAME = 'mimic3_eth7'
MIMIC3_ETH7_COLNAME = 'ETH7'


class AbstractGroupedProcedures(FlatScheme):
    """
    AbstractGroupedProcedures is a subclass of FlatScheme that represents a coding scheme for grouped procedures. The grouping enables a more concise abstract representation based on aggregation operations applied on each procedure group.

    Attributes:
        groups (Dict[str, set]): a dictionary mapping group names to sets of procedure codes.
        aggregation (List[str]): a list of aggregation methods for the groups.
        aggregation_groups (Dict[str, set]): a dictionary mapping aggregation methods to sets of group names.
    """

    _groups: Dict[str, Set[str]]
    _aggregation: List[str]
    _aggregation_groups: Dict[str, Set[str]]

    def __init__(self, config: CodingSchemeConfig, groups: Dict[str, Set[str]], aggregation: List[str],
                 aggregation_groups: Dict[str, Set[str]], **init_kwargs):
        super().__init__(config=config, **init_kwargs)
        self._groups = groups
        self._aggregation = aggregation
        self._aggregation_groups = aggregation_groups

    @property
    def groups(self):
        return self._groups

    @property
    def aggregation(self):
        return self._aggregation

    @property
    def aggregation_groups(self):
        return self._aggregation_groups


def register_mimic_ethnicity(scheme: str, filename: str, colname: str):
    """
    Register a MIMIC ethnicity coding scheme.

    Args:
        scheme (str): the name of the coding scheme.
        filename (str): the filename of the ethnicity scheme file.
        colname (str): the column name of the ethnicity codes in the file.
    """
    filepath = os.path.join(_RSC_DIR, filename)
    df = pd.read_csv(filepath, dtype=str)
    codes = sorted(set(df[colname]))
    desc = dict(zip(codes, codes))
    Ethnicity.register_scheme(Ethnicity(CodingSchemeConfig(name=scheme),
                                        codes=codes,
                                        index=dict(zip(codes, range(len(codes)))),
                                        desc=desc))


def register_mimic_ethnicity_loaders():
    """
    Register loaders for the mimic ethnicity coding schemes.
    """
    Ethnicity.register_scheme_loader(ETH32_NAME,
                                     lambda: register_mimic_ethnicity(ETH32_NAME, ETH_SCHEME_FILE, ETH32_COLNAME))
    Ethnicity.register_scheme_loader(ETH5_NAME,
                                     lambda: register_mimic_ethnicity(ETH5_NAME, ETH_SCHEME_FILE, ETH5_COLNAME))
    Ethnicity.register_scheme_loader(MIMIC3_ETH37_NAME,
                                     lambda: register_mimic_ethnicity(MIMIC3_ETH37_NAME, MIMIC3_ETH_SCHEME_FILE,
                                                                      MIMIC3_ETH37_COLNAME))
    Ethnicity.register_scheme_loader(MIMIC3_ETH7_NAME,
                                     lambda: register_mimic_ethnicity(MIMIC3_ETH7_NAME, MIMIC3_ETH_SCHEME_FILE,
                                                                      MIMIC3_ETH7_COLNAME))


def register_mimic_procedures():
    """
    Register the mimic procedure coding scheme.
    """
    filepath = os.path.join(_RSC_DIR, 'mimic4_int_grouper_proc.csv.gz')
    df = pd.read_csv(filepath, dtype=str)
    df = df[df.group != 'exclude']
    df = df.sort_values(['group', 'label'])
    codes = df.code.tolist()
    labels = df.label.tolist()
    desc = dict(zip(codes, labels))
    FlatScheme.register_scheme(FlatScheme(CodingSchemeConfig(name='int_mimic4_proc'),
                                          codes=codes,
                                          index=dict(zip(codes, range(len(codes)))),
                                          desc=desc))


def register_mimic_procedure_groups():
    """
    Register the mimic procedure groups coding scheme.
    """
    filepath = os.path.join(_RSC_DIR, 'mimic4_int_grouper_proc.csv.gz')
    df = pd.read_csv(filepath, dtype=str)
    df = df[df.group != 'exclude']
    df = df.sort_values(['group', 'label'])
    codes = df.group.unique().tolist()
    desc = dict(zip(codes, codes))

    groups = {
        group: set(group_df['code'])
        for group, group_df in df.groupby('group')
    }
    aggregation_groups = {'or': set(codes)}
    AbstractGroupedProcedures.register_scheme(
        AbstractGroupedProcedures(CodingSchemeConfig(name='int_mimic4_grouped_proc'),
                                  groups=groups,
                                  aggregation=['or'],
                                  aggregation_groups=aggregation_groups,
                                  codes=codes,
                                  index=dict(zip(codes, range(len(codes)))),
                                  desc=desc))


class MIMICInputGroups(AbstractGroupedProcedures):
    """
    This class represents grouped input items in the MIMIC dataset.
    It provides a property `dose_impact` that returns a dictionary mapping
    intervention names to flags indicating that the input rate/dose should be respected for the model adopting this scheme.

    Attributes:
        _dose_impact (Dict[str, str]): A dictionary mapping intervention names
            to their corresponding `dose_impact` flag.

    """

    _dose_impact: Dict[str, str]

    @property
    def dose_impact(self):
        return self._dose_impact

    def __init__(self, config: CodingSchemeConfig, dose_impact: Dict[str, str], **kwargs):
        super().__init__(config=config, **kwargs)
        self._dose_impact = dose_impact


def register_mimic_input_groups():
    """
    Register the MIMIC input groups coding scheme.

    This function reads a CSV file containing the MIMIC input groups data, which is manually curated and stored at `lib/ehr/resources`, filters out rows with group_decision 'E' (for exclude),
    sorts the data by 'group_decision', 'group', and 'label', and then registers the coding scheme using the extracted
    information.

    Returns:
        None
    """
    filepath = os.path.join(_RSC_DIR, 'mimic4_int_grouper_input.csv.gz')
    df = pd.read_csv(filepath, dtype=str)
    df = df[df.group_decision != 'E']
    df = df.sort_values(by=['group_decision', 'group', 'label'])
    codes = df.group.unique().tolist()
    desc = dict(zip(codes, codes))

    aggs = df.group_decision.unique().tolist()

    dose_impact = dict()
    aggregation_groups = dict()
    groups = dict()
    for agg, agg_df in df.groupby('group_decision'):
        aggregation_groups[agg] = set(agg_df['group'])
        for group, group_df in agg_df.groupby('group'):
            assert group not in groups, "Group should be unique"
            groups[group] = set(group_df['label'])
            dose_impact[group] = group_df['dose_impact'].iloc[0]

    MIMICInputGroups.register_scheme(MIMICInputGroups(CodingSchemeConfig(name='int_mimic4_input_group'),
                                                      groups=groups,
                                                      aggregation=aggs,
                                                      aggregation_groups=aggregation_groups,
                                                      codes=codes,
                                                      index=dict(zip(codes, range(len(codes)))),
                                                      desc=desc,
                                                      dose_impact=dose_impact))


def register_mimic_input():
    """
    Register the MIMIC input coding scheme.

    This function reads a CSV file containing mimic4_int_grouper input data, which is manually curated and stored at `lib/ehr/resources/mimic4_int_grouper_input.csv.gz`,
    filters out rows with group_decision 'E', sorts the data by 'group_decision',
    'group', and 'label', and registers the coding scheme using the unique labels
    as codes and their corresponding labels as descriptions.

    Returns:
        None
    """
    filepath = os.path.join(_RSC_DIR, 'mimic4_int_grouper_input.csv.gz')
    df = pd.read_csv(filepath, dtype=str)
    df = df[df.group_decision != 'E']
    df = df.sort_values(by=['group_decision', 'group', 'label'])
    codes = df.label.unique().tolist()
    desc = dict(zip(codes, codes))

    FlatScheme.register_scheme(FlatScheme(CodingSchemeConfig(name='int_mimic4_input'),
                                          codes=codes,
                                          index=dict(zip(codes, range(len(codes)))),
                                          desc=desc))


def register_mimic_eth_mapping(s_scheme: str, t_scheme: str, filename: str,
                               s_colname: str, t_colname: str):
    """
    Register the mapping between two ethnicity coding schemes using a CSV file.

    Args:
        s_scheme (str): the source coding scheme.
        t_scheme (str): the target coding scheme.
        filename (str): the name of the CSV file containing the mapping.
        s_colname (str): the column name in the CSV file representing the source codes.
        t_colname (str): the column name in the CSV file representing the target codes.

    Returns:
        None
    """
    filepath = os.path.join(_RSC_DIR, filename)
    df = pd.read_csv(filepath, dtype=str)
    mapper = df.groupby(s_colname)[t_colname].apply(set).to_dict()
    CodeMap.register_map(s_scheme, t_scheme, CodeMap(CodeMapConfig(s_scheme, t_scheme), mapper))


def register_mimic4proc_mapping(s_scheme: str,
                                t_scheme: str):
    """
    Register the mapping for MIMIC-IV procedure codes.

    Parameters:
    - s_scheme (str): the source coding scheme.
    - t_scheme (str): the target coding scheme.

    Returns:
    None
    """
    filepath = os.path.join(_RSC_DIR, 'mimic4_int_grouper_proc.csv.gz')
    df = pd.read_csv(filepath, dtype=str)
    mapper = {}
    for group, group_df in df.groupby('group'):
        mapper.update({c: {group} for c in group_df.code})
    CodeMap.register_map(s_scheme, t_scheme, CodeMap(CodeMapConfig(s_scheme, t_scheme), mapper))


def register_mimic4input_mapping(s_scheme: str,
                                 t_schame: str):
    """
    Register the mapping for MIMIC-IV input codes.

    Parameters:
    - s_scheme (str): The source coding scheme.
    - t_scheme (str): The target coding scheme.

    Returns:
    None
    """
    filepath = os.path.join(_RSC_DIR, 'mimic4_int_grouper_input.csv.gz')
    df = pd.read_csv(filepath, dtype=str)

    mapper = {}
    for group, group_df in df.groupby('group'):
        mapper.update({c: {group} for c in group_df.label})

    CodeMap.register_map(s_scheme, t_schame, CodeMap(CodeMapConfig(s_scheme, t_schame), mapper))


def setup_scheme_loaders():
    """
    Sets up the scheme loaders for the MIMIC-IV coding scheme.

    This function registers various scheme loaders for different components of the MIMIC-IV coding scheme,
    such as procedures, procedure groups, input groups, inputs, observables, and ethnicity loaders.

    Returns:
        None
    """
    FlatScheme.register_scheme_loader('int_mimic4_proc', register_mimic_procedures)
    AbstractGroupedProcedures.register_scheme_loader('int_mimic4_grouped_proc', register_mimic_procedure_groups)
    MIMICInputGroups.register_scheme_loader('int_mimic4_input_group', register_mimic_input_groups)
    FlatScheme.register_scheme_loader('int_mimic4_input', register_mimic_input)
    register_mimic_ethnicity_loaders()


def setup_maps_loaders():
    """
    Set up the map loaders for the coding schemes in the EHR module.
    """
    CodeMap.register_map_loader(ETH32_NAME, ETH5_NAME,
                                lambda: register_mimic_eth_mapping(ETH32_NAME, ETH5_NAME, ETH_SCHEME_FILE,
                                                                   ETH32_COLNAME, ETH5_COLNAME))
    CodeMap.register_map_loader(MIMIC3_ETH37_NAME, MIMIC3_ETH7_NAME,
                                lambda: register_mimic_eth_mapping(MIMIC3_ETH37_NAME, MIMIC3_ETH7_NAME,
                                                                   MIMIC3_ETH_SCHEME_FILE, MIMIC3_ETH37_COLNAME,
                                                                   MIMIC3_ETH7_COLNAME))
    CodeMap.register_map_loader('int_mimic4_proc', 'int_mimic4_grouped_proc',
                                lambda: register_mimic4proc_mapping('int_mimic4_proc', 'int_mimic4_grouped_proc'))
    CodeMap.register_map_loader('int_mimic4_input', 'int_mimic4_input_group',
                                lambda: register_mimic4input_mapping('int_mimic4_input', 'int_mimic4_input_group'))


def setup_mimic():
    """
    Sets up the MIMIC-IV coding schemes by loading the scheme loaders and maps loaders.
    """
    setup_scheme_loaders()
    setup_maps_loaders()
