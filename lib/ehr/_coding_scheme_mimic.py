from __future__ import annotations

import os
from typing import Dict

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

    def __init__(self, groups, aggregation, aggregation_groups, **init_kwargs):
        super().__init__(**init_kwargs)
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
    filepath = os.path.join(_RSC_DIR, filename)
    df = pd.read_csv(filepath, dtype=str)
    codes = sorted(set(df[colname]))
    desc = dict(zip(codes, codes))
    Ethnicity.register_scheme(Ethnicity(CodingSchemeConfig(name=scheme),
                                        codes=codes,
                                        index=dict(zip(codes, range(len(codes)))),
                                        desc=desc))


def register_mimic_ethnicity_loaders():
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
    InterventionGroup class encapsulates the similar interventions.
    """

    _dose_impact: Dict[str, str]

    @property
    def dose_impact(self):
        return self._dose_impact

    def __init__(self, dose_impact: Dict[str, str], **kwargs):
        super().__init__(**kwargs)
        self._dose_impact = dose_impact


def register_mimic_input_groups():
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


class MIMICObservables(FlatScheme):
    _groups: Dict[str, str]

    def __init__(self, groups: Dict[str, str], **kwargs):
        super().__init__(**kwargs)
        self._groups = groups

    @property
    def groups(self):
        return self._groups


def register_observables_scheme():
    filepath = os.path.join(_RSC_DIR, 'mimic4_obs_codes.csv.gz')
    df = pd.read_csv(filepath, dtype=str)
    codes = df.code.tolist()
    desc = dict(zip(codes, df.label.tolist()))
    groups = dict(zip(codes, df.group.tolist()))
    MIMICObservables.register_scheme(MIMICObservables(CodingSchemeConfig(name='mimic4_obs'),
                                                      codes=codes,
                                                      index=dict(zip(codes, range(len(codes)))),
                                                      desc=desc,
                                                      groups=groups))


def register_mimic_eth_mapping(s_scheme: str, t_scheme: str, filename: str,
                               s_colname: str, t_colname: str):
    filepath = os.path.join(_RSC_DIR, filename)
    df = pd.read_csv(filepath, dtype=str)
    mapper = df.groupby(s_colname)[t_colname].apply(set).to_dict()
    CodeMap.register_map(s_scheme, t_scheme, CodeMap(CodeMapConfig(s_scheme, t_scheme), mapper))


def register_mimic4proc_mapping(s_scheme: str,
                                t_scheme: str):
    filepath = os.path.join(_RSC_DIR, 'mimic4_int_grouper_proc.csv.gz')
    df = pd.read_csv(filepath, dtype=str)
    mapper = {}
    for group, group_df in df.groupby('group'):
        mapper.update({c: {group} for c in group_df.code})
    CodeMap.register_map(s_scheme, t_scheme, CodeMap(CodeMapConfig(s_scheme, t_scheme), mapper))


def register_mimic4input_mapping(s_scheme: str,
                                 t_schame: str):
    filepath = os.path.join(_RSC_DIR, 'mimic4_int_grouper_input.csv.gz')
    df = pd.read_csv(filepath, dtype=str)

    mapper = {}
    for group, group_df in df.groupby('group'):
        mapper.update({c: {group} for c in group_df.label})

    CodeMap.register_map(s_scheme, t_schame, CodeMap(CodeMapConfig(s_scheme, t_schame), mapper))


def setup_scheme_loaders():
    FlatScheme.register_scheme_loader('int_mimic4_proc', register_mimic_procedures)
    AbstractGroupedProcedures.register_scheme_loader('int_mimic4_grouped_proc', register_mimic_procedure_groups)
    MIMICInputGroups.register_scheme_loader('int_mimic4_input_group', register_mimic_input_groups)
    FlatScheme.register_scheme_loader('int_mimic4_input', register_mimic_input)
    MIMICObservables.register_scheme_loader('mimic4_obs', register_observables_scheme)
    register_mimic_ethnicity_loaders()


def setup_maps_loaders():
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
    setup_scheme_loaders()
    setup_maps_loaders()
