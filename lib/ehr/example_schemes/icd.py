from __future__ import annotations

import gzip
import xml.etree.ElementTree as ET
from abc import abstractmethod
from collections import defaultdict
from typing import Set, Dict, List, Union, Any, ClassVar, Type, Tuple, Final

import pandas as pd

from lib.ehr.coding_scheme import (CodingSchemeConfig, CodingScheme, FlatScheme, HierarchicalScheme,
                                   CodeMapConfig, CodeMap, resources_dir, FileBasedOutcomeExtractor)


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
    def create_scheme(cls):
        raise NotImplementedError


class ICDFlatScheme(ICDScheme, FlatScheme):
    def __init__(self, *args, **kwargs):
        FlatScheme.__init__(self, *args, **kwargs)


class ICDHierarchicalScheme(ICDScheme, HierarchicalScheme):

    def __init__(self, *args, **kwargs):
        HierarchicalScheme.__init__(self, *args, **kwargs)


class ICDMapOps:
    @staticmethod
    def load_conversion_table(source_scheme: ICDScheme, target_scheme: ICDScheme,
                              conversion_filename: str) -> Tuple[pd.DataFrame, Dict[str, Union[str, Set[str]]]]:
        df = pd.read_csv(resources_dir("CCS", conversion_filename),
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
    def analyse_conversions(conversion_table: pd.DataFrame) -> pd.DataFrame:
        codes = list(conversion_table['source'][conversion_table['no_map'] == '1'])
        status = ['no_map' for _ in codes]
        for code, source_df in conversion_table[conversion_table['no_map'] == '0'].groupby('source'):
            codes.append(code)
            if len(source_df) == 1:
                status.append('11_map')
            elif len(set(source_df['scenario'])) > 1:
                status.append('ambiguous')
            elif len(set(source_df['choice_list'])) < len(source_df):
                status.append('1n_map(resolved)')
            else:
                status.append('1n_map')

        return pd.DataFrame({'code': codes, 'status': status})

    @staticmethod
    def register_mappings(source_scheme: str, target_scheme: str,
                          conversion_filename: str):
        source_scheme: ICDScheme = CodingScheme.from_name(source_scheme)
        target_scheme: ICDScheme = CodingScheme.from_name(target_scheme)
        table, _ = ICDMapOps.load_conversion_table(source_scheme=source_scheme, target_scheme=target_scheme,
                                                   conversion_filename=conversion_filename)
        status_df = ICDMapOps.analyse_conversions(table)
        data = dict()
        mapping_status = status_df.set_index('code')['status'].to_dict()
        for code, code_targets_table in table.groupby('source'):
            if mapping_status[code] == 'no_map':
                continue
            data[code] = set(code_targets_table['target'])
        config = CodeMapConfig(source_scheme=source_scheme.name,
                               target_scheme=target_scheme.name,
                               mapped_to_dag_space=False)
        CodeMap.register_map(CodeMap(config=config, data=data))


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
        _ICD10_FILE = resources_dir(filename)
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
            dx_name = f'dx_discharge:{dx_name}'
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

        icd_codes = sorted(c.split(':')[1] for c in desc if 'dx_discharge:' in c)
        icd_desc = {c: desc[f'dx_discharge:{c}'] for c in icd_codes}

        if not hierarchical:
            return {
                'codes': icd_codes,
                'desc': icd_desc
            }
        icd2dag = {c: f'dx_discharge:{c}' for c in icd_codes}
        dag_codes = [f'dx_discharge:{c}' for c in icd_codes] + sorted(
            c for c in set(desc) - set(icd2dag.values()))

        return {
            'codes': icd_codes,
            'desc': icd_desc,
            'code2dag': icd2dag,
            'dag_codes': dag_codes,
            'dag_desc': desc,
            'pt2ch': dict(pt2ch)
        }


class PrICD10Ops(ICDOps):

    @staticmethod
    def add_dots(code: str) -> str:
        # No decimal point in ICD10-PCS
        return code

    @staticmethod
    def distill_icd10_xml(filename: str, hierarchical: bool = True) -> Dict[str, Any]:
        _ICD10_FILE = resources_dir(filename)

        with gzip.open(_ICD10_FILE, 'rt') as f:
            desc = {
                code: desc
                for code, desc in map(lambda line: line.strip().split(' ', 1),
                                      f.readlines())
            }
        codes = sorted(desc)

        if not hierarchical: return {
            'codes': codes,
            'desc': desc
        }

        dag_desc = {'_': 'root'}
        code2dag = {}

        pt2ch = {}
        pos = list(f'p{i}' for i in range(7))
        df = pd.DataFrame(list(list(pos for pos in code) for code in desc),
                          columns=pos)

        def _distill_connections(branch_df, ancestors, ancestors_str,
                                 next_positions):
            if len(next_positions) == 0:
                leaf_code = ''.join(ancestors[1:])
                code2dag[leaf_code] = ancestors_str
            else:
                children = set()
                for branch, _df in branch_df.groupby(next_positions[0]):
                    child = ancestors + [branch]
                    child_str = ':'.join(child)
                    children.add(child_str)
                    _distill_connections(_df, child, child_str,
                                         next_positions[1:])
                dag_desc[ancestors_str] = ancestors_str
                pt2ch[ancestors_str] = children

        _distill_connections(df, ['_'], '_', pos)

        dag_codes = list(map(code2dag.get, codes))
        dag_codes.extend(sorted(pt2ch))
        dag_desc.update({c: c for c in set(dag_codes) - set(dag_desc)})

        return {
            'codes': codes,
            'desc': desc,
            'code2dag': code2dag,
            'dag_codes': dag_codes,
            'dag_desc': dag_desc,
            'pt2ch': pt2ch
        }


class ICD9Ops(ICDOps):
    ICD9CM_FILE: Final[str] = resources_dir('HOM-ICD9.csv.gz')
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
            'codes': icd_codes,
            'desc': icd_desc,
            'code2dag': icd2dag,
            'dag_codes': dag_codes,
            'dag_desc': dag_desc
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
    def create_scheme(cls):
        # https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2023/
        config = CodingSchemeConfig(name='dx_icd10')
        CodingScheme.register_scheme(
            cls(config=config, **cls.ops.distill_icd10_xml('icd10cm_tabular_2023.xml.gz', True)))


class DxFlatICD10(ICDFlatScheme):
    ops: ClassVar[Type[DxICD10Ops]] = DxICD10Ops

    @classmethod
    def create_scheme(cls):
        # https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2023/
        config = CodingSchemeConfig(name='dx_flat_icd10')
        scheme = cls(config=config, **cls.ops.distill_icd10_xml('icd10cm_tabular_2023.xml.gz', False))
        CodingScheme.register_scheme(scheme)


class PrHierarchicalICD10(ICDHierarchicalScheme):
    ops: ClassVar[Type[PrICD10Ops]] = PrICD10Ops

    @classmethod
    def create_scheme(cls):
        scheme = PrHierarchicalICD10(config=CodingSchemeConfig(name='pr_icd10'),
                                     **cls.ops.distill_icd10_xml('icd10pcs_codes_2023.txt.gz', True))
        CodingScheme.register_scheme(scheme)


class PrFlatICD10(ICDFlatScheme):
    ops: ClassVar[Type[PrICD10Ops]] = PrICD10Ops

    @classmethod
    def create_scheme(cls):
        CodingScheme.register_scheme(PrFlatICD10(CodingSchemeConfig(name='pr_flat_icd10'),
                                                 **cls.ops.distill_icd10_xml('icd10pcs_codes_2023.txt.gz', False)))


class DxHierarchicalICD9(ICDHierarchicalScheme):
    ops: ClassVar[Type[DxICD9Ops]] = DxICD9Ops

    @classmethod
    def create_scheme(cls):
        df = pd.DataFrame(cls.ops.icd9_columns())
        pt2ch = cls.ops.parent_child_mappings(df)

        # Remove the procedure codes.
        pt2ch = cls.ops.deselect_subtree(pt2ch, cls.ops.PR_ROOT_CLASS_ID)

        # Remaining node indices in one set.
        nodes = set().union(set(pt2ch), *pt2ch.values())

        # Filter out the procedure code from the df.
        df = df[df['NODE_IDX'].isin(nodes)]
        CodingScheme.register_scheme(DxHierarchicalICD9(config=CodingSchemeConfig(name='dx_icd9'),
                                                        **cls.ops.generate_dictionaries(df), pt2ch=pt2ch))


class PrHierarchicalICD9(ICDHierarchicalScheme):
    ops: ClassVar[Type[PrICD9Ops]] = PrICD9Ops

    @classmethod
    def create_scheme(cls):
        df = pd.DataFrame(cls.ops.icd9_columns())
        pt2ch = cls.ops.parent_child_mappings(df)

        # Remove the procedure codes.
        pt2ch = cls.ops.select_subtree(pt2ch, cls.ops.PR_ROOT_CLASS_ID)

        # Remaining node indices in one set.
        nodes = set().union(set(pt2ch), *pt2ch.values())

        # Filter out the procedure code from the df.
        df = df[df['NODE_IDX'].isin(nodes)]
        CodingScheme.register_scheme(PrHierarchicalICD9(config=CodingSchemeConfig(name='pr_icd9'),
                                                        **cls.ops.generate_dictionaries(df), pt2ch=pt2ch))


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
    def register_mappings(cls, ccs_scheme: str,
                          icd9_scheme: str):
        ccs_scheme: CodingScheme = CodingScheme.from_name(ccs_scheme)
        icd9_scheme: ICDHierarchicalScheme = CodingScheme.from_name(icd9_scheme)

        res = cls.ccs_columns(icd9_scheme)

        # TODO: Check if the mapping is correct
        icd92ccs_config = CodeMapConfig(icd9_scheme.name,
                                        ccs_scheme.name,
                                        t_dag_space=False)
        ccs2icd9_config = CodeMapConfig(ccs_scheme.name,
                                        icd9_scheme.name,
                                        t_dag_space=False)
        icd92ccs = defaultdict(set)
        ccs2icd9 = defaultdict(set)

        cols = res["cols"]
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

        CodeMap.register_map(CodeMap(icd92ccs_config, dict(icd92ccs)))
        CodeMap.register_map(CodeMap(ccs2icd9_config, dict(ccs2icd9)))

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
    def create_scheme(cls):
        icd9_scheme: ICDHierarchicalScheme = CodingScheme.from_name(cls.ICD9_SCHEME_NAME)
        cols, _ = cls.ops.ccs_columns(icd9_scheme)
        df = pd.DataFrame(cols)
        pt2ch = cls.ops.parent_child_mappings(df)
        desc = cls.ops.desc_mappings(df)
        codes = sorted(desc.keys())
        CodingScheme.register_scheme(cls(CodingSchemeConfig(name=cls.SCHEME_NAME),
                                         pt2ch=pt2ch,
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
    def flatccs_columns(cls, icd9_scheme: ICDHierarchicalScheme) -> Tuple[pd.DataFrame, Dict[str, Set[str] | str]]:
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
    def register_ccs_flat_mappings(cls, flatccs_scheme: CodingScheme, icd9_scheme: ICDHierarchicalScheme):
        columns, _ = cls.flatccs_columns(icd9_scheme)

        flatccs2icd9_config = CodeMapConfig(
            flatccs_scheme.name,
            icd9_scheme.name,
            t_dag_space=False)
        icd92flatccs_config = CodeMapConfig(
            icd9_scheme.name,
            flatccs_scheme.name,
            t_dag_space=False)

        assert len(columns['icd9']) == columns['icd9'].nunique(), "1toN mapping expected"
        flatccs2icd9 = defaultdict(set)
        icd92flatccs = defaultdict(set)
        for icd_code, ccode in columns.set_index('icd9')['code'].to_dict().items():
            flatccs2icd9[ccode].add(icd_code)
            icd92flatccs[icd_code].add(ccode)

        CodeMap.register_map(CodeMap(flatccs2icd9_config, dict(flatccs2icd9)))
        CodeMap.register_map(CodeMap(icd92flatccs_config, dict(icd92flatccs)))


class DxFlatCCSMapOps(FlatCCSMapOps):
    SCHEME_FILE = '$dxref 2015.csv.gz'


class PrFlatCCSMapOps(FlatCCSMapOps):
    SCHEME_FILE = '$prref 2015.csv.gz'


class FlatCCSScheme(FlatScheme):
    ops: ClassVar[Type[FlatCCSMapOps]] = FlatCCSMapOps
    ICD9_SCHEME_NAME: str = None
    SCHEME_NAME: str = None

    @classmethod
    def create_scheme(cls):
        icd9_scheme: ICDHierarchicalScheme = CodingScheme.from_name(cls.ICD9_SCHEME_NAME)
        cols, _ = cls.ops.flatccs_columns(icd9_scheme)
        codes = sorted(set(cols['code']))
        CodingScheme.register_scheme(cls(config=CodingSchemeConfig(cls.SCHEME_NAME),
                                         codes=codes,
                                         desc=dict(zip(cols['code'], cols['desc']))))


class DxFlatCCS(FlatCCSScheme):
    ops: ClassVar[Type[DxFlatCCSMapOps]] = DxFlatCCSMapOps
    ICD9_SCHEME_NAME: str = 'dx_icd9'
    SCHEME_NAME: str = 'dx_flatccs'

class PrFlatCCS(FlatCCSScheme):
    ops: ClassVar[Type[PrFlatCCSMapOps]] = PrFlatCCSMapOps
    ICD9_SCHEME_NAME: str = 'pr_icd9'
    SCHEME_NAME: str = 'pr_flatccs'

def setup_scheme_loaders():
    CodingScheme.register_scheme_loader('dx_icd10', DxHierarchicalICD10.create_scheme)
    CodingScheme.register_scheme_loader('pr_icd10', PrHierarchicalICD10.create_scheme)
    CodingScheme.register_scheme_loader('dx_flat_icd10', DxFlatICD10.create_scheme)
    CodingScheme.register_scheme_loader('pr_flat_icd10', PrFlatICD10.create_scheme)
    CodingScheme.register_scheme_loader('dx_icd9', DxHierarchicalICD9.create_scheme)
    CodingScheme.register_scheme_loader('pr_icd9', PrHierarchicalICD9.create_scheme)
    CodingScheme.register_scheme_loader('dx_ccs', DxCCS.create_scheme)
    CodingScheme.register_scheme_loader('pr_ccs', PrCCS.create_scheme)
    CodingScheme.register_scheme_loader('dx_flatccs', DxFlatCCS.create_scheme)
    CodingScheme.register_scheme_loader('pr_flatccs', PrFlatCCS.create_scheme)

    for name in ('dx_flatccs_mlhc_groups', 'dx_flatccs_v1', 'dx_icd9_v1', 'dx_icd9_v2_groups', 'dx_icd9_v3_groups'):
        FileBasedOutcomeExtractor.register_outcome_extractor_loader(name, f'{name}.json')


def setup_maps_loaders():
    # ICD9 <-> ICD10s
    CodeMap.register_map_loader('dx_icd10', 'dx_icd9',
                                lambda: ICDMapOps.register_mappings('dx_icd10', 'dx_icd9', '2018_gem_cm_I10I9.txt.gz'))
    CodeMap.register_map_loader('dx_icd9', 'dx_icd10',
                                lambda: ICDMapOps.register_mappings('dx_icd9', 'dx_icd10', '2018_gem_cm_I9I10.txt.gz'))
    CodeMap.register_map_loader('pr_icd10', 'pr_icd9',
                                lambda: ICDMapOps.register_mappings('pr_icd10', 'pr_icd9', '2018_gem_pcs_I10I9.txt.gz'))
    CodeMap.register_map_loader('pr_icd9', 'pr_icd10',
                                lambda: ICDMapOps.register_mappings('pr_icd9', 'pr_icd10', '2018_gem_pcs_I9I10.txt.gz'))
    CodeMap.register_map_loader('dx_flat_icd10', 'dx_icd9',
                                lambda: ICDMapOps.register_mappings('dx_flat_icd10', 'dx_icd9',
                                                                    '2018_gem_cm_I10I9.txt.gz'))
    CodeMap.register_map_loader('dx_icd9', 'dx_flat_icd10',
                                lambda: ICDMapOps.register_mappings('dx_icd9', 'dx_flat_icd10',
                                                                    '2018_gem_cm_I9I10.txt.gz'))
    CodeMap.register_map_loader('pr_flat_icd10', 'pr_icd9',
                                lambda: ICDMapOps.register_mappings('pr_flat_icd10', 'pr_icd9',
                                                                    '2018_gem_pcs_I10I9.txt.gz'))
    CodeMap.register_map_loader('pr_icd9', 'pr_flat_icd10',
                                lambda: ICDMapOps.register_mappings('pr_icd9', 'pr_flat_icd10',
                                                                    '2018_gem_pcs_I9I10.txt.gz'))

    # ICD9 <-> CCS
    bimap_dx_ccs_icd9 = lambda: DxCCSMapOps.register_mappings('dx_ccs', 'dx_icd9')

    CodeMap.register_map_loader('dx_icd9', 'dx_ccs',
                                bimap_dx_ccs_icd9)
    CodeMap.register_map_loader('dx_ccs', 'dx_icd9',
                                bimap_dx_ccs_icd9)

    bimap_pr_ccs_icd9 = lambda: PrCCSMapOps.register_mappings('pr_ccs', 'pr_icd9')
    CodeMap.register_map_loader('pr_icd9', 'pr_ccs',
                                bimap_pr_ccs_icd9)
    CodeMap.register_map_loader('pr_ccs', 'pr_icd9',
                                bimap_pr_ccs_icd9)

    bimap_dx_flatccs_icd9 = lambda: DxCCSMapOps.register_mappings('dx_flatccs', 'dx_icd9')
    CodeMap.register_map_loader('dx_flatccs', 'dx_icd9',
                                bimap_dx_flatccs_icd9)
    CodeMap.register_map_loader('dx_icd9', 'dx_flatccs',
                                bimap_dx_flatccs_icd9)
    bimap_pr_flatccs_icd9 = lambda: PrCCSMapOps.register_mappings('pr_flatccs', 'pr_icd9')
    CodeMap.register_map_loader('pr_flatccs', 'pr_icd9',
                                bimap_pr_flatccs_icd9)
    CodeMap.register_map_loader('pr_icd9', 'pr_flatccs',
                                bimap_pr_flatccs_icd9)
    # ICD10 <-> CCS (Through ICD9 as an intermediate scheme)
    for dx_icd10_str in ('dx_icd10', 'dx_flat_icd10'):
        CodeMap.register_chained_map_loader('dx_ccs', 'dx_icd9', dx_icd10_str)
        CodeMap.register_chained_map_loader(dx_icd10_str, 'dx_icd9', 'dx_ccs')
        CodeMap.register_chained_map_loader('dx_flatccs', 'dx_icd9', dx_icd10_str)
        CodeMap.register_chained_map_loader(dx_icd10_str, 'dx_icd9', 'dx_flatccs')

    for pr_icd10_str in ('pr_icd10', 'pr_flat_icd10'):
        CodeMap.register_chained_map_loader('pr_ccs', 'pr_icd9', pr_icd10_str)
        CodeMap.register_chained_map_loader(pr_icd10_str, 'pr_icd9', 'pr_ccs')
        CodeMap.register_chained_map_loader('pr_flatccs', 'pr_icd9', pr_icd10_str)
        CodeMap.register_chained_map_loader(pr_icd10_str, 'pr_icd9', 'pr_flatccs')

    # CCS <-> FlatCCS (Through ICD9 as an intermediate scheme)
    CodeMap.register_chained_map_loader('dx_flatccs', 'dx_icd9', 'dx_ccs')
    CodeMap.register_chained_map_loader('dx_ccs', 'dx_icd9', 'dx_flatccs')
    CodeMap.register_chained_map_loader('pr_flatccs', 'pr_icd9', 'pr_ccs')
    CodeMap.register_chained_map_loader('pr_ccs', 'pr_icd9', 'pr_flatccs')


def setup_icd():
    setup_scheme_loaders()
    setup_maps_loaders()
