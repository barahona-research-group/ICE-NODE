from __future__ import annotations

import gzip
import xml.etree.ElementTree as ET
from abc import abstractmethod
from collections import defaultdict
from dataclasses import fields
from typing import Set, Dict, List, Union, Any, ClassVar, Type, Tuple, Final

import pandas as pd

from lib import Config
from lib.ehr.coding_scheme import (CodingScheme, HierarchicalScheme,
                                   CodeMap, resources_dir, FileBasedOutcomeExtractor, CodingSchemesManager,
                                   FrozenDict11, FrozenDict1N)


class ICDOps:
    """
    Class representing the ICD (International Classification of Diseases) coding scheme.

    This class provides additional methods
    for loading conversion tables, analyzing conversions, and registering mappings.


    Methods:
        add_dots(code: str) -> str:
            return the dotted code textual representation.

        create_scheme():
            abstract method for creating the ICD scheme.

        _deselect_subtree(pt2ch: Dict[str, Set[str]], sub_root: str) -> Dict[str, Set[str]]:
            deselects a subtree from the given parent-to-child dictionary.

        _select_subtree(pt2ch: Dict[str, Set[str]], sub_root: str) -> Dict[str, Set[str]]:
            selects a subtree from the given parent-to-child dictionary.

        load_conv_table(s_scheme: ICD, t_scheme: ICD, conv_fname: str) -> Dict[str, Union[pd.DataFrame, str, Set[str]]]:
            loads the conversion table from the specified file.

        analyse_conversions(s_scheme: ICD, t_scheme: ICD, conv_fname: str) -> pd.DataFrame:
            analyzes the conversions between the source and target schemes using the given conversion file.

        register_mappings(s_scheme: str, t_scheme: str, conv_fname: str):
            registers the mappings between the source and target schemes using the given conversion file.
    """

    @staticmethod
    @abstractmethod
    def add_dots(code: str) -> str:
        pass

    @staticmethod
    def deselect_subtree(pt2ch: Dict[str, Set[str]], sub_root: str) -> Dict[str, Set[str]]:
        to_del = HierarchicalScheme._bfs_traversal(pt2ch, sub_root, True)
        pt2ch = pt2ch.copy()
        to_del = set(to_del) & set(pt2ch.keys())
        for node_idx in to_del:
            del pt2ch[node_idx]
        return pt2ch

    @staticmethod
    def select_subtree(pt2ch: Dict[str, Set[str]], sub_root: str) -> Dict[str, Set[str]]:
        to_keep = HierarchicalScheme._bfs_traversal(pt2ch, sub_root, True)
        to_keep = set(to_keep) & set(pt2ch.keys())
        return {idx: pt2ch[idx] for idx in to_keep}


class ICDScheme(CodingScheme):
    ops: ClassVar[Type[ICDOps]] = ICDOps

    @classmethod
    def create_scheme(cls, manager: CodingSchemesManager) -> CodingSchemesManager:
        raise NotImplementedError


class ICDFlatScheme(ICDScheme, CodingScheme):
    def __init__(self, *args, **kwargs):
        CodingScheme.__init__(self, *args, **kwargs)


class ICDHierarchicalScheme(ICDScheme, HierarchicalScheme):

    def __init__(self, *args, **kwargs):
        HierarchicalScheme.__init__(self, *args, **kwargs)


class ICDMapOps:
    @staticmethod
    def load_conversion_table(source_scheme: ICDScheme, target_scheme: ICDScheme,
                              conversion_filename: str) -> Tuple[pd.DataFrame, Dict[str, Union[str, Set[str]]]]:
        df = pd.read_csv(resources_dir("ICD", conversion_filename),
                         sep='\s+',
                         dtype=str,
                         names=['source', 'target', 'meta'])
        df['approximate'] = df['meta'].apply(lambda s: s[0])
        df['no_map'] = df['meta'].apply(lambda s: s[1])
        df['combination'] = df['meta'].apply(lambda s: s[2])
        df['scenario'] = df['meta'].apply(lambda s: s[3])
        df['choice_list'] = df['meta'].apply(lambda s: s[4])
        df['source'] = df['source'].map(source_scheme.ops.add_dots)
        df['target'] = df['target'].map(target_scheme.ops.add_dots)

        valid_target = df['target'].isin(target_scheme.index)
        valid_source = df['source'].isin(source_scheme.index)

        conversion_table = df[valid_target & valid_source]

        return conversion_table, {"conversion_filename": conversion_filename,
                                  "unrecognised_target": set(df[~valid_target]["target"]),
                                  "unrecognised_source": set(df[~valid_source]["source"])}

    @staticmethod
    def conversion_status(conversion_table: pd.DataFrame) -> Dict[str, str]:
        def _get_status(groupby_df: pd.DataFrame):
            if (groupby_df['no_map'] == '1').all():
                return 'no_map'
            elif len(groupby_df) == 1:
                return '11_map'
            elif groupby_df['scenario'].nunique() > 1:
                return 'ambiguous'
            elif groupby_df['choice_list'].nunique() < len(groupby_df):
                return '1n_map(resolved)'
            else:
                return '1n_map'

        return conversion_table.groupby('source').apply(_get_status).to_dict()


    @staticmethod
    def register_mappings(manager: CodingSchemesManager, source_scheme: str, target_scheme: str,
                          conversion_filename: str) -> CodingSchemesManager:
        source_scheme: ICDScheme = manager.scheme[source_scheme]
        target_scheme: ICDScheme = manager.scheme[target_scheme]
        table, _ = ICDMapOps.load_conversion_table(source_scheme=source_scheme, target_scheme=target_scheme,
                                                   conversion_filename=conversion_filename)
        conversion_status = ICDMapOps.conversion_status(table)
        table['status'] = table['source'].map(conversion_status)
        table = table[table['status'] != 'no_map']
        data = FrozenDict1N.from_dict(table.groupby('source')['target'].apply(set).to_dict())
        return manager.add_map(CodeMap(source_name=source_scheme.name, target_name=target_scheme.name, data=data))


class DxICD10Ops(ICDOps):
    """
    NOTE: for prediction targets, remember to exclude the following chapters:
        - 'chapter:19': 'Injury, poisoning and certain \
            other consequences of external causes (S00-T88)',
        - 'chapter:20': 'External causes of morbidity (V00-Y99)',
        - 'chapter:21': 'Factors influencing health status and \
            contact with health services (Z00-Z99)',
        - 'chapter:22': 'Codes for special purposes (U00-U85)'
    """

    @staticmethod
    def add_dots(code: str) -> str:
        if '.' in code:
            # logging.debug(f'Code {code} already is in decimal format')
            return code
        if len(code) > 3:
            return code[:3] + '.' + code[3:]
        else:
            return code

    @staticmethod
    def distill_icd10_xml(filename: str, hierarchical: bool = True) -> Dict[str, Any]:
        # https://www.cdc.gov/nchs/icd/Comprehensive-Listing-of-ICD-10-CM-Files.htm
        _ICD10_FILE = resources_dir("ICD", filename)
        with gzip.open(_ICD10_FILE, 'r') as f:
            tree = ET.parse(f)
        root = tree.getroot()
        pt2ch = defaultdict(set)
        root_node = f'root:{root.tag}'
        desc = {root_node: 'root'}
        chapters = [ch for ch in root if ch.tag == 'chapter']

        def _traverse_diag_dfs(parent_name, dx_element):
            dx_name = next(e for e in dx_element if e.tag == 'name').text
            dx_desc = next(e for e in dx_element if e.tag == 'desc').text
            dx_name = f'dx_icd10:{dx_name}'
            desc[dx_name] = dx_desc
            pt2ch[parent_name].add(dx_name)

            diags = [dx for dx in dx_element if dx.tag == 'diag']
            for dx in diags:
                _traverse_diag_dfs(dx_name, dx)

        for chapter in chapters:
            ch_name = next(e for e in chapter if e.tag == 'name').text
            ch_desc = next(e for e in chapter if e.tag == 'desc').text
            ch_name = f'chapter:{ch_name}'
            pt2ch[root_node].add(ch_name)
            desc[ch_name] = ch_desc

            sections = [sec for sec in chapter if sec.tag == 'section']
            for section in sections:
                sec_name = section.attrib['id']
                sec_desc = next(e for e in section if e.tag == 'desc').text
                sec_name = f'section:{sec_name}'

                pt2ch[ch_name].add(sec_name)
                desc[sec_name] = sec_desc

                diags = [dx for dx in section if dx.tag == 'diag']
                for dx in diags:
                    _traverse_diag_dfs(sec_name, dx)

        icd_codes = tuple(sorted(c.split(':')[1] for c in desc if 'dx_icd10:' in c))
        icd_desc = {c: desc[f'dx_icd10:{c}'] for c in icd_codes}

        if not hierarchical:
            return {
                'codes': tuple(sorted(icd_codes)),
                'desc': FrozenDict11.from_dict(icd_desc)
            }

        def leaf_code(c):
            return 'dx_icd10_leaf:' + c

        def node_code(c):
            return 'dx_icd10_node:' + c

        raw_nodes = set(c for c in desc if 'dx_icd10:' not in c)

        dag_codes_nodes = tuple(sorted(node_code(c) for c in raw_nodes))
        dag_desc_nodes = {node_code(c): desc[c] for c in raw_nodes}
        pt2ch = {node_code(k): set(node_code(v) for v in vs) for k, vs in pt2ch.items()}
        return {
            'codes': icd_codes,
            'desc': FrozenDict11.from_dict(icd_desc),
            'code2dag': FrozenDict11.from_dict({c: leaf_code(c) for c in icd_codes}),
            'dag_codes': tuple(sorted(leaf_code(c) for c in icd_codes)) + dag_codes_nodes,
            'dag_desc': FrozenDict11.from_dict({leaf_code(c): icd_desc[c] for c in icd_codes} | dag_desc_nodes),
            'ch2pt': HierarchicalScheme.reverse_connection(pt2ch)
        }


class PrICD10Ops(ICDOps):

    @staticmethod
    def add_dots(code: str) -> str:
        # No decimal point in ICD10-PCS
        return code

    @staticmethod
    def distill_icd10_xml(filename: str) -> Dict[str, Any]:
        _ICD10_FILE = resources_dir("ICD", filename)

        with gzip.open(_ICD10_FILE, 'rt') as f:
            desc = {
                code: desc
                for code, desc in map(lambda line: line.strip().split(' ', 1),
                                      f.readlines())
            }
        return {
            'codes': tuple(sorted(desc)),
            'desc': FrozenDict11.from_dict(desc)
        }


class ICD9Ops(ICDOps):
    ICD9CM_FILE: Final[str] = resources_dir('ICD', 'HOM-ICD9.csv.gz')
    DUMMY_ROOT_CLASS_ID: Final[str] = 'owl#Thing'
    PR_ROOT_CLASS_ID: Final[str] = 'MM_CLASS_2'
    DX_ROOT_CLASS_ID: Final[str] = 'MM_CLASS_21'

    @staticmethod
    def icd9_columns() -> Dict[str, List[str]]:
        # https://bioportal.bioontology.org/ontologies/HOM-ICD9
        df = pd.read_csv(ICD9Ops.ICD9CM_FILE, dtype=str)
        df = df.fillna('')

        def retain_suffix(cell):
            if 'http' in cell:
                return cell.split('/')[-1]
            else:
                return cell

        df = df.map(retain_suffix)
        df.columns = list(map(retain_suffix, df.columns))

        df['level'] = 0
        for j in range(1, 7):
            level_rows = df[f'ICD9_LEVEL{j}'] != ''
            df.loc[level_rows, 'level'] = j

        return {
            'ICD9': list(df['C_BASECODE'].apply(lambda c: c.split(':')[-1])),
            'NODE_IDX': list(df['Class ID']),
            'PARENT_IDX': list(df['Parents']),
            'LABEL': list(df['Preferred Label']),
            'LEVEL': list(df['level'])
        }

    @staticmethod
    def parent_child_mappings(df: pd.DataFrame) -> Dict[str, Set[str]]:
        pt2ch = {}
        for pt, ch_df in df.groupby('PARENT_IDX'):
            pt2ch[str(pt)] = set(ch_df['NODE_IDX'])

        # Remove dummy parent of diagnoses.
        del pt2ch[ICD9Ops.DUMMY_ROOT_CLASS_ID]
        return pt2ch

    @staticmethod
    def generate_dictionaries(df: pd.DataFrame) -> Dict[str, Any]:
        # df version for leaf nodes only (with non-empty ICD9 codes)
        df_leaves = df[df['ICD9'] != '']

        icd2dag = dict(zip(df_leaves['ICD9'], df_leaves['NODE_IDX']))

        # df version for internal nodes only (with empty ICD9 codes)
        df_internal = df[(df['ICD9'] == '') | df['ICD9'].isnull()]

        icd_codes = sorted(df_leaves['ICD9'])
        icd_desc = dict(zip(df_leaves['ICD9'], df_leaves['LABEL']))

        dag_codes = list(map(icd2dag.get, icd_codes)) + sorted(
            df_internal['NODE_IDX'])
        dag_desc = dict(zip(df['NODE_IDX'], df['LABEL']))

        return {
            'codes': tuple(sorted(icd_codes)),
            'desc': FrozenDict11.from_dict(icd_desc),
            'code2dag': FrozenDict11.from_dict(icd2dag),
            'dag_codes': tuple(sorted(dag_codes)),
            'dag_desc': FrozenDict11.from_dict(dag_desc)
        }


class DxICD9Ops(ICD9Ops):
    @staticmethod
    def add_dots(code: str) -> str:
        if '.' in code:
            # logging.debug(f'Code {code} already is in decimal format')
            return code
        if code[0] == 'E':
            if len(code) > 4:
                return code[:4] + '.' + code[4:]
            else:
                return code
        else:
            if len(code) > 3:
                return code[:3] + '.' + code[3:]
            else:
                return code


class PrICD9Ops(DxICD9Ops):

    @staticmethod
    def add_dots(code: str) -> str:
        if '.' in code:
            # logging.debug(f'Code {code} already is in decimal format')
            return code
        if len(code) > 2:
            return code[:2] + '.' + code[2:]
        else:
            return code


class DxHierarchicalICD10(ICDHierarchicalScheme):
    ops: ClassVar[Type[DxICD10Ops]] = DxICD10Ops

    @classmethod
    def create_scheme(cls, manager: CodingSchemesManager) -> CodingSchemesManager:
        # https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2023/
        return manager.add_scheme(
            cls(name='dx_icd10', **cls.ops.distill_icd10_xml('icd10cm_tabular_2023.xml.gz', True)))


class DxFlatICD10(ICDFlatScheme):
    ops: ClassVar[Type[DxICD10Ops]] = DxICD10Ops

    @classmethod
    def create_scheme(cls, manager: CodingSchemesManager) -> CodingSchemesManager:
        # https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2023/

        return manager.add_scheme(
            cls(name='dx_flat_icd10', **cls.ops.distill_icd10_xml('icd10cm_tabular_2023.xml.gz', False)))


class PrFlatICD10(ICDFlatScheme):
    ops: ClassVar[Type[PrICD10Ops]] = PrICD10Ops

    @classmethod
    def create_scheme(cls, manager: CodingSchemesManager) -> CodingSchemesManager:
        scheme = PrFlatICD10(name='pr_flat_icd10',
                             **cls.ops.distill_icd10_xml('icd10pcs_codes_2023.txt.gz'))
        return manager.add_scheme(scheme)


class DxHierarchicalICD9(ICDHierarchicalScheme):
    ops: ClassVar[Type[DxICD9Ops]] = DxICD9Ops

    @classmethod
    def create_scheme(cls, manager: CodingSchemesManager) -> CodingSchemesManager:
        df = pd.DataFrame(cls.ops.icd9_columns())
        pt2ch = cls.ops.parent_child_mappings(df)

        # Remove the procedure codes.
        pt2ch = cls.ops.deselect_subtree(pt2ch, cls.ops.PR_ROOT_CLASS_ID)

        # Remaining node indices in one set.
        nodes = set().union(set(pt2ch), *pt2ch.values())

        # Filter out the procedure code from the df.
        df = df[df['NODE_IDX'].isin(nodes)]
        scheme = DxHierarchicalICD9(name='dx_icd9',
                                    **cls.ops.generate_dictionaries(df),
                                    ch2pt=HierarchicalScheme.reverse_connection(pt2ch))
        return manager.add_scheme(scheme)


class PrHierarchicalICD9(ICDHierarchicalScheme):
    ops: ClassVar[Type[PrICD9Ops]] = PrICD9Ops

    @classmethod
    def create_scheme(cls, manager: CodingSchemesManager) -> CodingSchemesManager:
        df = pd.DataFrame(cls.ops.icd9_columns())
        pt2ch = cls.ops.parent_child_mappings(df)

        # Remove the procedure codes.
        pt2ch = cls.ops.select_subtree(pt2ch, cls.ops.PR_ROOT_CLASS_ID)

        # Remaining node indices in one set.
        nodes = set().union(set(pt2ch), *pt2ch.values())

        # Filter out the procedure code from the df.
        df = df[df['NODE_IDX'].isin(nodes)]
        scheme = PrHierarchicalICD9(name='pr_icd9',
                                    **cls.ops.generate_dictionaries(df),
                                    ch2pt=HierarchicalScheme.reverse_connection(pt2ch))
        return manager.add_scheme(scheme)


class CCSMapOps:
    SCHEME_FILE: str = None
    N_LEVELS: int = None

    @classmethod
    def ccs_columns(cls, icd9_scheme: ICDHierarchicalScheme) -> Tuple[Dict[str, List[str]], Dict[str, Set[str] | str]]:
        df = pd.read_csv(resources_dir("CCS", cls.SCHEME_FILE), dtype=str)
        icd_cname = '\'ICD-9-CM CODE\''

        df[icd_cname] = df[icd_cname].str.strip('\'').str.strip()
        df[icd_cname] = df[icd_cname].map(icd9_scheme.ops.add_dots)
        valid_icd = df[icd_cname].isin(icd9_scheme.index)
        df = df[valid_icd]

        cols = {}
        for i in range(1, cls.N_LEVELS + 1):
            cols[f'I{i}'] = df[f'\'CCS LVL {i}\''].str.strip('\'').str.strip().tolist()
            cols[f'L{i}'] = df[f'\'CCS LVL {i} LABEL\''].str.strip('\'').str.strip().tolist()
        cols['ICD'] = df[icd_cname].tolist()

        return cols, {
            "unrecognised_icd9": set(df[~valid_icd][icd_cname]),
            "conversion_filename": cls.SCHEME_FILE
        }

    @classmethod
    def register_mappings(cls, manager: CodingSchemesManager, ccs_scheme: str,
                          icd9_scheme: str) -> CodingSchemesManager:
        ccs_scheme: CodingScheme = manager.scheme[ccs_scheme]
        icd9_scheme: ICDHierarchicalScheme = manager.scheme[icd9_scheme]

        cols, _ = cls.ccs_columns(icd9_scheme)

        # TODO: Check if the mapping is correct
        icd92ccs = defaultdict(set)
        ccs2icd9 = defaultdict(set)

        n_rows = len(cols['ICD'])
        for i in range(n_rows):
            last_index = None
            for j in range(1, cls.N_LEVELS + 1):
                level = cols[f'I{j}'][i]
                if level != '':
                    last_index = level
            if last_index is not None:
                icd_code = cols['ICD'][i]
                icd92ccs[icd_code].add(last_index)
                ccs2icd9[last_index].add(icd_code)

        manager = manager.add_map(CodeMap(source_name=icd9_scheme.name,
                                          target_name=ccs_scheme.name, data=FrozenDict1N.from_dict(dict(icd92ccs))))
        manager = manager.add_map(CodeMap(source_name=ccs_scheme.name,
                                          target_name=icd9_scheme.name, data=FrozenDict1N.from_dict(dict(ccs2icd9))))
        return manager

    @classmethod
    def parent_child_mappings(cls, df: pd.DataFrame) -> Dict[str, Set[str]]:
        """Make dictionary for parent-child connections."""
        pt2ch = {'root': set(df['I1'])}
        levels = list(map(lambda i: f'I{i}', range(1, cls.N_LEVELS + 1)))

        for pt_col, ch_col in zip(levels[:-1], levels[1:]):
            df_ = df[(df[pt_col] != '') & (df[ch_col] != '')]
            df_ = df_[[pt_col, ch_col]].drop_duplicates()
            for parent_code, ch_df in df_.groupby(pt_col):
                pt2ch[parent_code] = set(ch_df[ch_col])
        return pt2ch

    @classmethod
    def desc_mappings(cls, df: pd.DataFrame) -> Dict[str, str]:
        """Make a dictionary for CCS labels."""
        desc = {'root': 'root'}
        levels = list(map(lambda i: f'I{i}', range(1, cls.N_LEVELS + 1)))
        descs = list(map(lambda i: f'L{i}', range(1, cls.N_LEVELS + 1)))
        for code_col, desc_col in zip(levels, descs):
            df_ = df[df[code_col] != '']
            df_ = df_[[code_col, desc_col]].drop_duplicates()
            code_desc = dict(zip(df_[code_col], df_[desc_col]))
            desc.update(code_desc)
        return desc

    @staticmethod
    def code_ancestors_dots(code: str, include_itself: bool = True) -> Set[str]:

        ancestors = {code} if include_itself else set()
        if code == 'root':
            return ancestors
        else:
            ancestors.add('root')

        indices = code.split('.')
        for i in reversed(range(1, len(indices))):
            parent = '.'.join(indices[0:i])
            ancestors.add(parent)
        return ancestors

    @classmethod
    def code_ancestors(cls, code: str, include_itself: bool) -> Set[str]:
        return cls.code_ancestors_dots(code, include_itself)


class DxCCSMapOps(CCSMapOps):
    SCHEME_FILE = 'ccs_multi_dx_tool_2015.csv.gz'
    N_LEVELS = 4


class PrCCSMapOps(CCSMapOps):
    SCHEME_FILE = 'ccs_multi_pr_tool_2015.csv.gz'
    N_LEVELS = 3


class CCSHierarchicalScheme(HierarchicalScheme):
    ops: ClassVar[Type[CCSMapOps]] = CCSMapOps
    ICD9_SCHEME_NAME: str = None
    SCHEME_NAME: str = None

    @classmethod
    def create_scheme(cls, manager: CodingSchemesManager) -> CodingSchemesManager:
        icd9_scheme: ICDHierarchicalScheme = manager.scheme[cls.ICD9_SCHEME_NAME]
        cols, _ = cls.ops.ccs_columns(icd9_scheme)
        df = pd.DataFrame(cols)
        ch2pt = HierarchicalScheme.reverse_connection(FrozenDict1N.from_dict(cls.ops.parent_child_mappings(df)))
        desc = FrozenDict11.from_dict(cls.ops.desc_mappings(df))
        codes = tuple(sorted(desc.keys()))
        return manager.add_scheme(cls(name=cls.SCHEME_NAME,
                                      ch2pt=ch2pt,
                                      codes=codes,
                                      desc=desc))


class DxCCS(CCSHierarchicalScheme):
    ops: ClassVar[Type[DxCCSMapOps]] = DxCCSMapOps
    ICD9_SCHEME_NAME: str = 'dx_icd9'
    SCHEME_NAME: str = 'dx_ccs'


class PrCCS(CCSHierarchicalScheme):
    ops: ClassVar[Type[PrCCSMapOps]] = PrCCSMapOps
    ICD9_SCHEME_NAME: str = 'pr_icd9'
    SCHEME_NAME: str = 'pr_ccs'


class FlatCCSMapOps:
    SCHEME_FILE = None

    @classmethod
    def flat_ccs_columns(cls, icd9_scheme: ICDHierarchicalScheme) -> Tuple[pd.DataFrame, Dict[str, Set[str] | str]]:
        filepath = resources_dir("CCS", cls.SCHEME_FILE)
        df = pd.read_csv(filepath, skiprows=[0, 2], dtype=str)
        icd9_cname = '\'ICD-9-CM CODE\''
        cat_cname = '\'CCS CATEGORY\''
        desc_cname = '\'CCS CATEGORY DESCRIPTION\''
        for cname in (icd9_cname, cat_cname, desc_cname):
            df[cname] = df[cname].str.strip('\'').str.strip()
        df[icd9_cname] = df[icd9_cname].map(icd9_scheme.ops.add_dots)
        # Filter out unrecognised ICD-9 codes.
        valid_icd9 = df[icd9_cname].isin(icd9_scheme.index)
        df = df[valid_icd9]
        df = df.rename(columns={icd9_cname: 'icd9', cat_cname: 'code', desc_cname: 'desc'})
        return df[['code', 'icd9', 'desc']], {
            'unrecognised_icd9': set(df[~valid_icd9]['icd9']),
            'conversion_filename': cls.SCHEME_FILE
        }

    @classmethod
    def register_ccs_flat_mappings(cls, manager: CodingSchemesManager, flat_ccs_scheme: CodingScheme,
                                   icd9_scheme: ICDHierarchicalScheme):
        columns, _ = cls.flat_ccs_columns(icd9_scheme)
        assert len(columns['icd9']) == columns['icd9'].nunique(), "1toN mapping expected"
        flat_ccs2icd9 = defaultdict(set)
        icd92flat_ccs = defaultdict(set)
        for icd_code, ccode in columns.set_index('icd9')['code'].to_dict().items():
            flat_ccs2icd9[ccode].add(icd_code)
            icd92flat_ccs[icd_code].add(ccode)

        manager = manager.add_map(CodeMap(source_name=flat_ccs_scheme.name,
                                          target_name=icd9_scheme.name, data=FrozenDict1N.from_dict(flat_ccs2icd9)))
        manager = manager.add_map(CodeMap(source_name=icd9_scheme.name,
                                          target_name=flat_ccs_scheme.name, data=FrozenDict1N.from_dict(icd92flat_ccs)))
        return manager


class DxFlatCCSMapOps(FlatCCSMapOps):
    SCHEME_FILE = '$dxref 2015.csv.gz'


class PrFlatCCSMapOps(FlatCCSMapOps):
    SCHEME_FILE = '$prref 2015.csv.gz'


class FlatCCSScheme(CodingScheme):
    ops: ClassVar[Type[FlatCCSMapOps]] = FlatCCSMapOps
    ICD9_SCHEME_NAME: str = None
    SCHEME_NAME: str = None

    @classmethod
    def create_scheme(cls, manager: CodingSchemesManager) -> CodingSchemesManager:
        icd9_scheme: ICDHierarchicalScheme = manager.scheme[cls.ICD9_SCHEME_NAME]
        cols, _ = cls.ops.flat_ccs_columns(icd9_scheme)
        codes = tuple(sorted(set(cols['code'])))
        return manager.add_scheme(cls(name=cls.SCHEME_NAME,
                                      codes=codes,
                                      desc=FrozenDict11.from_dict(dict(zip(cols['code'], cols['desc'])))))


class DxFlatCCS(FlatCCSScheme):
    ops: ClassVar[Type[DxFlatCCSMapOps]] = DxFlatCCSMapOps
    ICD9_SCHEME_NAME: str = 'dx_icd9'
    SCHEME_NAME: str = 'dx_flat_ccs'


class PrFlatCCS(FlatCCSScheme):
    ops: ClassVar[Type[PrFlatCCSMapOps]] = PrFlatCCSMapOps
    ICD9_SCHEME_NAME: str = 'pr_icd9'
    SCHEME_NAME: str = 'pr_flat_ccs'


class Flags(Config):
    @property
    def flag_set(self) -> Tuple[str, ...]:
        return tuple(f.name for f in fields(self) if isinstance(getattr(self, f.name), bool) and getattr(self, f.name))


class CCSICDSchemeSelection(Flags):
    dx_icd9: bool = False
    dx_icd10: bool = False
    pr_icd9: bool = False
    dx_flat_icd10: bool = False
    pr_flat_icd10: bool = False
    dx_ccs: bool = False
    pr_ccs: bool = False
    dx_flat_ccs: bool = False
    pr_flat_ccs: bool = False


icd_ccs_schemes = {
    'dx_icd9': DxHierarchicalICD9,
    'dx_icd10': DxHierarchicalICD10,
    'dx_flat_icd10': DxFlatICD10,
    'dx_ccs': DxCCS,
    'dx_flat_ccs': DxFlatCCS,
    'pr_icd9': PrHierarchicalICD9,
    'pr_flat_icd10': PrFlatICD10,
    'pr_ccs': PrCCS,
    'pr_flat_ccs': PrFlatCCS
}


class CCSICDOutcomeSelection(Flags):
    dx_icd9_v1: bool = False
    dx_icd9_v2_groups: bool = False
    dx_icd9_v3_groups: bool = False
    dx_flat_ccs_mlhc_groups: bool = False
    dx_flat_ccs_v1: bool = False


def setup_icd_schemes(manager: CodingSchemesManager, scheme_selection: CCSICDSchemeSelection) -> CodingSchemesManager:
    for name in scheme_selection.flag_set:
        scheme_cls = icd_ccs_schemes[name]
        manager = scheme_cls.create_scheme(manager)
    return manager


def setup_icd_outcomes(manager: CodingSchemesManager,
                       outcome_selection: CCSICDOutcomeSelection) -> CodingSchemesManager:
    for outcome_name in outcome_selection.flag_set:
        manager = manager.add_outcome(FileBasedOutcomeExtractor.from_spec_file(f'{outcome_name}.json'))
    return manager


def setup_icd_maps(manager: CodingSchemesManager, scheme_selection: CCSICDSchemeSelection):
    # ICD9 <-> ICD10s
    if scheme_selection.dx_icd9 and scheme_selection.dx_icd10:
        manager = ICDMapOps.register_mappings(manager, 'dx_icd10', 'dx_icd9', '2018_gem_cm_I10I9.txt.gz')
        manager = ICDMapOps.register_mappings(manager, 'dx_icd9', 'dx_icd10', '2018_gem_cm_I9I10.txt.gz')

    if scheme_selection.dx_flat_icd10 and scheme_selection.dx_icd9:
        manager = ICDMapOps.register_mappings(manager, 'dx_icd9', 'dx_flat_icd10',
                                              '2018_gem_cm_I9I10.txt.gz')
        manager = ICDMapOps.register_mappings(manager, 'dx_flat_icd10', 'dx_icd9',
                                              '2018_gem_cm_I10I9.txt.gz')
    if scheme_selection.pr_icd9 and scheme_selection.pr_flat_icd10:
        manager = ICDMapOps.register_mappings(manager, 'pr_flat_icd10', 'pr_icd9',
                                              '2018_gem_pcs_I10I9.txt.gz')
        manager = ICDMapOps.register_mappings(manager, 'pr_icd9', 'pr_flat_icd10',
                                              '2018_gem_pcs_I9I10.txt.gz')

    # ICD9 <-> CCS
    if scheme_selection.dx_icd9 and scheme_selection.dx_ccs:
        manager = DxCCSMapOps.register_mappings(manager, 'dx_ccs', 'dx_icd9')
    if scheme_selection.pr_icd9 and scheme_selection.pr_ccs:
        manager = PrCCSMapOps.register_mappings(manager, 'pr_ccs', 'pr_icd9')

    if scheme_selection.dx_flat_ccs and scheme_selection.dx_icd9:
        manager = DxCCSMapOps.register_mappings(manager, 'dx_flat_ccs', 'dx_icd9')
    if scheme_selection.pr_flat_ccs and scheme_selection.pr_icd9:
        manager = PrCCSMapOps.register_mappings(manager, 'pr_flat_ccs', 'pr_icd9')

    return manager


def setup_standard_icd_ccs(manager: CodingSchemesManager, scheme_selection: CCSICDSchemeSelection,
                           outcome_selection: CCSICDOutcomeSelection) -> CodingSchemesManager:
    manager = setup_icd_schemes(manager, scheme_selection)
    manager = setup_icd_maps(manager, scheme_selection)
    manager = setup_icd_outcomes(manager, outcome_selection)
    return manager
