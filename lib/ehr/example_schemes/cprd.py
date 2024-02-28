from __future__ import annotations

import os
from typing import Set, Dict, List

import pandas as pd

from lib.ehr.coding_scheme import (CodingSchemeConfig, CodingScheme, FlatScheme,
                                   CodeMap, CodeMapConfig, resources_dir,
                                   SchemeWithMissing, Ethnicity, OutcomeExtractor, FileBasedOutcomeExtractor)

_CPRD_LTC_FILE = resources_dir('CPRD_212_LTC_ALL.csv.gz')
ETH16_CODE_CNAME = 'eth16'
ETH5_CODE_CNAME = 'eth5'
ETH16_DESC_CNAME = 'eth16_desc'
ETH5_DESC_CNAME = 'eth5_desc'
ETH_SCHEME_FILE = 'cprd_eth.csv'


class DxLTC212FlatCodes(FlatScheme):
    _medcodes: Dict[str, Set[str]]
    _system: Dict[str, str]

    def __init__(self, config: CodingSchemeConfig,
                 codes: List[str], desc: Dict[str, str], system: Dict[str, str],
                 medcodes: Dict[str, Set[str]]):
        super().__init__(config=config,
                         codes=codes,
                         desc=desc)
        self._system = system
        self._medcodes = medcodes

    @property
    def medcodes(self):
        return self._medcodes

    @property
    def system(self):
        return self._system

    @classmethod
    def from_file(cls, name: str = 'dx_cprd_ltc212', filepath: str = _CPRD_LTC_FILE):
        df = pd.read_csv(filepath, dtype=str)

        medcode_cname = 'medcodeid'
        disease_num_cname = 'disease_num'
        disease_cname = 'disease'

        system_cname = 'system'
        system_num_cname = 'system_num'

        desc = dict()
        system = dict()
        medcodes = dict()
        for disease_num, disease_df in df.groupby(disease_num_cname):
            disease_set = set(disease_df[disease_cname])
            assert len(disease_set) == 1, "Disease name should be unique"
            (disease_name,) = disease_set

            system_set = set(disease_df[system_cname])
            system_num_set = set(disease_df[system_num_cname])
            assert len(system_set) == 1, "System name should be unique"
            assert len(system_num_set) == 1, "System num should be unique"

            (system_name,) = system_set
            (system_num,) = system_num_set

            medcodes_list = sorted(set(disease_df[medcode_cname]))

            desc[disease_num] = disease_name
            system[disease_num] = system_num
            medcodes[disease_num] = medcodes_list

        codes = sorted(set(df[disease_num_cname]))
        return cls(config=CodingSchemeConfig(name),
                   codes=codes,
                   desc=desc,
                   system=system,
                   medcodes=medcodes)


class DxLTC9809FlatMedcodes(FlatScheme):
    _diseases: Dict[str, List[str]]
    _systems: Dict[str, List[str]]
    _diseases_desc: Dict[str, str]
    _systems_desc: Dict[str, str]

    def __init__(self, config: CodingSchemeConfig,
                 codes: List[str], desc: Dict[str, str], diseases: Dict[str, List[str]],
                 systems: Dict[str, List[str]], diseases_desc: Dict[str, str], systems_desc: Dict[str, str]):
        super().__init__(config=config,
                         codes=codes,
                         desc=desc)
        self._diseases = diseases
        self._systems = systems
        self._diseases_desc = diseases_desc
        self._systems_desc = systems_desc

    @property
    def systems(self):
        return self._systems

    @property
    def diseases(self):
        return self._diseases

    @property
    def diseases_desc(self):
        return self._diseases_desc

    @property
    def systems_desc(self):
        return self._systems_desc

    @classmethod
    def from_file(cls, name: str = 'dx_cprd_ltc9809', filepath: str = _CPRD_LTC_FILE):
        filepath = resources_dir(filepath)
        df = pd.read_csv(filepath, dtype=str)

        medcode_cname = 'medcodeid'
        disease_num_cname = 'disease_num'
        disease_cname = 'disease'
        desc_cname = 'descr'

        system_cname = 'system'
        system_num_cname = 'system_num'

        desc = dict()
        systems = dict()
        diseases = dict()
        for medcodeid, medcode_df in df.groupby(medcode_cname):
            desc[medcodeid] = medcode_df[desc_cname].iloc[0]
            systems[medcodeid] = sorted(set(medcode_df[system_num_cname]))
            diseases[medcodeid] = sorted(set(medcode_df[disease_num_cname]))

        codes = sorted(set(df[medcode_cname]))
        diseases_desc = df.groupby(disease_num_cname)[desc_cname].apply(list).to_dict()
        systems_desc = df.groupby(system_num_cname)[desc_cname].apply(list).to_dict()
        return cls(config=CodingSchemeConfig(name),
                   codes=codes,
                   desc=desc,
                   systems=systems,
                   diseases=diseases,
                   diseases_desc=diseases_desc,
                   systems_desc=systems_desc)


def register_cprd_ethnicity(name, eth_code_colname, eth_desc_colname):
    filepath = resources_dir(ETH_SCHEME_FILE)
    df = pd.read_csv(filepath, dtype=str)
    desc = dict()
    for eth_code, eth_df in df.groupby(eth_code_colname):
        eth_set = set(eth_df[eth_desc_colname])
        assert len(eth_set) == 1, "Ethnicity description should be unique"
        (eth_desc,) = eth_set
        desc[eth_code] = eth_desc

    codes = sorted(set(df[eth_code_colname]))
    Ethnicity.register_scheme(Ethnicity(CodingSchemeConfig(name),
                                        codes=codes,
                                        desc=desc))


def register_cprd_ethnicity_scheme_loaders():
    Ethnicity.register_scheme_loader('eth_cprd_16',
                                     lambda: register_cprd_ethnicity('eth_cprd_16', ETH16_CODE_CNAME, ETH16_DESC_CNAME))
    Ethnicity.register_scheme_loader('eth_cprd_5',
                                     lambda: register_cprd_ethnicity('eth_cprd_5', ETH5_CODE_CNAME, ETH5_DESC_CNAME))


def register_cprd_eth_mapping():
    filepath = resources_dir(ETH_SCHEME_FILE)
    df = pd.read_csv(filepath, dtype=str)
    data = df.set_index(ETH16_CODE_CNAME)[ETH5_CODE_CNAME].to_dict()
    CodeMap.register_map('eth_cprd_16', 'eth_cprd_5', CodeMap(CodeMapConfig('eth_cprd_16', 'eth_cprd_5'), data=data))


def register_cprd_gender():
    missing_code = '9'
    codes = ['0', '1', '2']
    desc = {
        '0': 'female',
        '1': 'male',
        '2': 'intermediate'
    }
    name = 'cprd_gender'
    SchemeWithMissing.register_scheme(
        SchemeWithMissing(CodingSchemeConfig(name),
                          codes=codes, desc=desc, missing_code=missing_code))


def register_cprd_imd():
    missing_code = '99'
    codes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    desc = dict(zip(codes, codes))
    name = 'cprd_imd_cat'

    SchemeWithMissing.register_scheme(SchemeWithMissing(CodingSchemeConfig(name),
                                                        codes=codes, desc=desc,
                                                        missing_code=missing_code))


def register_medcode_mapping(medcode_scheme: str,
                             disease_num_scheme: str):
    medcode = FlatScheme.from_name(medcode_scheme)
    CodeMap.register_map(medcode_scheme, disease_num_scheme,
                         CodeMap(CodeMapConfig(medcode_scheme, disease_num_scheme), data=medcode.diseases))


def setup_scheme_loaders():
    CodingScheme.register_scheme_loader('dx_cprd_ltc212',
                                        lambda: CodingScheme.register_scheme(DxLTC212FlatCodes.from_file()))
    CodingScheme.register_scheme_loader('dx_cprd_ltc9809',
                                        lambda: CodingScheme.register_scheme(DxLTC9809FlatMedcodes.from_file()))
    register_cprd_ethnicity_scheme_loaders()
    CodingScheme.register_scheme_loader('cprd_gender', register_cprd_gender)
    CodingScheme.register_scheme_loader('cprd_imd_cat', register_cprd_imd)
    FileBasedOutcomeExtractor.register_outcome_extractor_loader('dx_cprd_ltc212', 'dx_cprd_ltc212_v1.json')
    FileBasedOutcomeExtractor.register_outcome_extractor_loader('dx_cprd_ltc9809', 'dx_cprd_ltc9809_v1.json')


def setup_maps_loaders():
    # CPRD conversions
    # LTC9809 -> LTC212
    # Eth16 -> Eth5
    CodeMap.register_map_loader('dx_cprd_ltc9809', 'dx_cprd_ltc212',
                                lambda: register_medcode_mapping('dx_cprd_ltc9809', 'dx_cprd_ltc212'))

    CodeMap.register_map_loader('eth_cprd_16', 'eth_cprd_5', register_cprd_eth_mapping)


def setup_cprd():
    setup_scheme_loaders()
    setup_maps_loaders()
