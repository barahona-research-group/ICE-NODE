from __future__ import annotations

import gzip
import logging
import os
import xml.etree.ElementTree as ET
from abc import abstractmethod
from collections import defaultdict
from typing import Set, Dict, List, Union, Callable, Any

import pandas as pd

from .coding_scheme import (CodingSchemeConfig, CodingScheme, FlatScheme, HierarchicalScheme,
                            _RSC_DIR, CodeMapConfig, CodeMap, OutcomeExtractor)

_CCS_DIR = os.path.join(_RSC_DIR, "CCS")


class ICDOps:

    @staticmethod
    @abstractmethod
    def add_dots(code: str) -> str:
        pass

    @staticmethod
    @classmethod
    def register_scheme(cls):
        pass

    @staticmethod
    def _deselect_subtree(pt2ch: Dict[str, Set[str]], sub_root: str) -> Dict[str, Set[str]]:
        to_del = HierarchicalScheme._bfs_traversal(pt2ch, sub_root, True)
        pt2ch = pt2ch.copy()
        to_del = set(to_del) & set(pt2ch.keys())
        for node_idx in to_del:
            del pt2ch[node_idx]
        return pt2ch

    @staticmethod
    def _select_subtree(pt2ch: Dict[str, Set[str]], sub_root: str) -> Dict[str, Set[str]]:
        to_keep = HierarchicalScheme._bfs_traversal(pt2ch, sub_root, True)
        to_keep = set(to_keep) & set(pt2ch.keys())
        return {idx: pt2ch[idx] for idx in to_keep}

    @staticmethod
    def load_conv_table(s_scheme: CodingScheme, t_scheme: CodingScheme,
                        s_scheme_add_dots: Callable[[str], str],
                        t_scheme_add_dots: Callable[[str], str],
                        conv_fname: str) -> Dict[str, Union[pd.DataFrame, str, Set[str]]]:
        df = pd.read_csv(os.path.join(_RSC_DIR, conv_fname),
                         sep='\s+',
                         dtype=str,
                         names=['source', 'target', 'meta'])
        df['approximate'] = df['meta'].apply(lambda s: s[0])
        df['no_map'] = df['meta'].apply(lambda s: s[1])
        df['combination'] = df['meta'].apply(lambda s: s[2])
        df['scenario'] = df['meta'].apply(lambda s: s[3])
        df['choice_list'] = df['meta'].apply(lambda s: s[4])
        df['source'] = df['source'].map(s_scheme_add_dots)
        df['target'] = df['target'].map(t_scheme_add_dots)

        valid_target = df['target'].isin(t_scheme.index)
        unrecognised_target = set(df[~valid_target]["target"])
        if len(unrecognised_target) > 0:
            logging.debug(f"""
                            {s_scheme}->{t_scheme} Unrecognised t_codes
                            ({len(unrecognised_target)}):
                            {sorted(unrecognised_target)[:20]}...""")

        valid_source = df['source'].isin(s_scheme.index)
        unrecognised_source = set(df[~valid_source]["source"])
        if len(unrecognised_source) > 0:
            logging.debug(f"""
                            {s_scheme}->{t_scheme} Unrecognised s_codes
                            ({len(unrecognised_source)}):
                            {sorted(unrecognised_source)[:20]}...""")

        df = df[valid_target & valid_source]
        # df['target'] = df['target'].map(t_scheme.code2dag)

        return {
            "df": df,
            "conv_file": conv_fname,
            "unrecognised_target": unrecognised_target,
            "unrecognised_source": unrecognised_source
        }

    @staticmethod
    def analyse_conversions(s_scheme: CodingScheme, t_scheme: CodingScheme,
                            s_scheme_add_dots: Callable[[str], str],
                            t_scheme_add_dots: Callable[[str], str],
                            conv_fname: str) -> pd.DataFrame:
        df = ICDOps.load_conv_table(s_scheme, t_scheme,
                                    s_scheme_add_dots,
                                    t_scheme_add_dots,
                                    conv_fname)["df"]
        codes = list(df['source'][df['no_map'] == '1'])
        status = ['no_map' for _ in codes]
        for code, source_df in df[df['no_map'] == '0'].groupby('source'):
            codes.append(code)
            if len(source_df) == 1:
                status.append('11_map')
            elif len(set(source_df['scenario'])) > 1:
                status.append('ambiguous')
            elif len(set(source_df['choice_list'])) < len(source_df):
                status.append('1n_map(resolved)')
            else:
                status.append('1n_map')

        status = pd.DataFrame({'code': codes, 'status': status})
        return status

    @staticmethod
    def register_mappings(s_scheme: str, t_scheme: str,
                          s_scheme_add_dots: Callable[[str], str], t_scheme_add_dots: Callable[[str], str],
                          conv_fname: str):
        s_scheme = CodingScheme.from_name(s_scheme)
        t_scheme = CodingScheme.from_name(t_scheme)

        res = ICDOps.load_conv_table(s_scheme, t_scheme,
                                     s_scheme_add_dots,
                                     t_scheme_add_dots,
                                     conv_fname)
        conv_df = res["df"]
        status_df = ICDOps.analyse_conversions(s_scheme, t_scheme,
                                               s_scheme_add_dots,
                                               t_scheme_add_dots,
                                               conv_fname)
        map_kind = dict(zip(status_df['code'], status_df['status']))

        config = CodeMapConfig(source_scheme=s_scheme.name,
                               target_scheme=t_scheme.name,
                               mapped_to_dag_space=False)
        data = dict()
        for code, df in conv_df.groupby('source'):
            kind = map_kind[code]
            if kind == 'no_map':
                continue
            data[code] = set(df['target'])
        CodeMap.register_map(s_scheme.name, t_scheme.name, CodeMap(config, data))


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
    def distill_icd10_xml(filename: str) -> Dict[str, Any]:
        # https://www.cdc.gov/nchs/icd/Comprehensive-Listing-of-ICD-10-CM-Files.htm
        _ICD10_FILE = os.path.join(_RSC_DIR, filename)
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
            dx_name = f'dx:{dx_name}'
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

        icd_codes = sorted(c.split(':')[1] for c in desc if 'dx:' in c)
        icd_index = dict(zip(icd_codes, range(len(icd_codes))))
        icd_desc = {c: desc[f'dx:{c}'] for c in icd_codes}
        icd2dag = {c: f'dx:{c}' for c in icd_codes}
        dag_codes = [f'dx:{c}' for c in icd_codes] + sorted(
            c for c in set(desc) - set(icd2dag.values()))
        dag_index = dict(zip(dag_codes, range(len(dag_codes))))

        return {
            'codes': icd_codes,
            'index': icd_index,
            'desc': icd_desc,
            'code2dag': icd2dag,
            'dag_codes': dag_codes,
            'dag_index': dag_index,
            'dag_desc': desc,
            'pt2ch': dict(pt2ch)
        }

    @classmethod
    def register_scheme(cls):
        # https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2023/
        config = CodingSchemeConfig(name='dx_icd10')
        CodingScheme.register_scheme(HierarchicalScheme(config, **cls.distill_icd10_xml('icd10cm_codes_2023.txt.gz')))


class PrICD10Ops(ICDOps):

    @staticmethod
    def add_dots(code: str) -> str:
        # No decimal point in ICD10-PCS
        return code

    @staticmethod
    def distill_icd10_xml(filename: str) -> Dict[str, Any]:
        _ICD10_FILE = os.path.join(_RSC_DIR, filename)

        with gzip.open(_ICD10_FILE, 'rt') as f:
            desc = {
                code: desc
                for code, desc in map(lambda line: line.strip().split(' ', 1),
                                      f.readlines())
            }
        codes = sorted(desc)
        index = dict(zip(codes, range(len(codes))))

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
        dag_index = dict(zip(dag_codes, range(len(dag_codes))))

        return {
            'codes': codes,
            'index': index,
            'desc': desc,
            'code2dag': code2dag,
            'dag_codes': dag_codes,
            'dag_index': dag_index,
            'dag_desc': desc,
            'pt2ch': pt2ch
        }

    @classmethod
    def register_scheme(cls):
        CodingScheme.register_scheme(HierarchicalScheme(CodingSchemeConfig(name='pr_icd10'),
                                                        **cls.distill_icd10_xml('icd10pcs_codes_2023.txt.gz')))


class DxICD9Ops(ICDOps):
    _PR_ROOT_CLASS_ID = 'MM_CLASS_2'
    _DX_ROOT_CLASS_ID = 'MM_CLASS_21'
    _DX_DUMMY_ROOT_CLASS_ID = 'owl#Thing'

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

    @staticmethod
    def icd9_columns() -> Dict[str, List[str]]:
        # https://bioportal.bioontology.org/ontologies/HOM-ICD9
        ICD9CM_FILE = os.path.join(_RSC_DIR, 'HOM-ICD9.csv.gz')
        df = pd.read_csv(ICD9CM_FILE, dtype=str)
        df = df.fillna('')

        def retain_suffix(cell):
            if 'http' in cell:
                return cell.split('/')[-1]
            else:
                return cell

        df = df.applymap(retain_suffix)
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
            pt2ch[pt] = set(ch_df['NODE_IDX'])

        # Remove dummy parent of diagnoses.
        del pt2ch[DxICD9Ops._DX_DUMMY_ROOT_CLASS_ID]
        return pt2ch

    @staticmethod
    def generate_dictionaries(df: pd.DataFrame) -> Dict[str, Any]:
        # df version for leaf nodes only (with non-empty ICD9 codes)
        df_leaves = df[df['ICD9'] != '']

        icd2dag = dict(zip(df_leaves['ICD9'], df_leaves['NODE_IDX']))

        # df version for internal nodes only (with empty ICD9 codes)
        df_internal = df[(df['ICD9'] == '') | df['ICD9'].isnull()]

        icd_codes = sorted(df_leaves['ICD9'])
        icd_index = dict(zip(icd_codes, range(len(icd_codes))))
        icd_desc = dict(zip(df_leaves['ICD9'], df_leaves['LABEL']))

        dag_codes = list(map(icd2dag.get, icd_codes)) + sorted(
            df_internal['NODE_IDX'])
        dag_index = dict(zip(dag_codes, range(len(dag_codes))))
        dag_desc = dict(zip(df['NODE_IDX'], df['LABEL']))

        return {
            'icd_codes': icd_codes,
            'icd_index': icd_index,
            'icd_desc': icd_desc,
            'icd2dag': icd2dag,
            'dag_codes': dag_codes,
            'dag_index': dag_index,
            'dag_desc': dag_desc
        }

    @classmethod
    def register_scheme(cls):
        df = pd.DataFrame(cls.icd9_columns())
        pt2ch = cls.parent_child_mappings(df)

        # Remove the procedure codes.
        pt2ch = cls._deselect_subtree(pt2ch, cls._PR_ROOT_CLASS_ID)

        # Remaining node indices in one set.
        nodes = set().union(set(pt2ch), *pt2ch.values())

        # Filter out the procedure code from the df.
        df = df[df['NODE_IDX'].isin(nodes)]

        d = cls.generate_dictionaries(df)

        CodingScheme.register_scheme(HierarchicalScheme(CodingSchemeConfig(name='dx_icd9'),
                                                        **d, pt2ch=pt2ch))


class PrICD9Ops(ICDOps):

    @staticmethod
    def add_dots(code: str) -> str:
        if '.' in code:
            # logging.debug(f'Code {code} already is in decimal format')
            return code
        if len(code) > 2:
            return code[:2] + '.' + code[2:]
        else:
            return code

    @classmethod
    def register_scheme(cls):
        df = pd.DataFrame(DxICD9Ops.icd9_columns())
        pt2ch = DxICD9Ops.parent_child_mappings(df)

        # Remove the procedure codes.
        pt2ch = cls._select_subtree(pt2ch, DxICD9Ops._PR_ROOT_CLASS_ID)

        # Remaining node indices in one set.
        nodes = set().union(set(pt2ch), *pt2ch.values())

        # Filter out the procedure code from the df.
        df = df[df['NODE_IDX'].isin(nodes)]

        d = DxICD9Ops.generate_dictionaries(df)

        CodingScheme.register_scheme(HierarchicalScheme(CodingSchemeConfig(name='pr_icd9'),
                                                        **d, pt2ch=pt2ch))


class CCSOps:
    _SCHEME_FILE = None
    _N_LEVELS = None

    @classmethod
    def ccs_columns(cls, icd9_scheme: CodingScheme, icd9_add_dots: Callable[[str], str]) -> Dict[str, List[str]]:
        df = pd.read_csv(os.path.join(_CCS_DIR, cls._SCHEME_FILE), dtype=str)
        icd_cname = '\'ICD-9-CM CODE\''

        df[icd_cname] = df[icd_cname].apply(lambda l: l.strip('\'').strip())
        df[icd_cname] = df[icd_cname].map(icd9_add_dots)
        valid_icd = df[icd_cname].isin(icd9_scheme.index)
        unrecognised_icd9 = set(df[~valid_icd][icd_cname])
        df = df[valid_icd]

        cols = {}
        for i in range(1, cls._N_LEVELS + 1):
            cols[f'I{i}'] = list(
                df[f'\'CCS LVL {i}\''].apply(lambda l: l.strip('\'').strip()))
            cols[f'L{i}'] = list(df[f'\'CCS LVL {i} LABEL\''].apply(
                lambda l: l.strip('\'').strip()))
        cols['ICD'] = list(df[icd_cname])

        return {
            "cols": cols,
            "unrecognised_icd9": unrecognised_icd9,
            "conv_file": cls._SCHEME_FILE
        }

    @staticmethod
    def register_mappings(ccs_scheme: str, icd9_scheme: str, icd9_add_dots: Callable[[str], str]):
        ccs_scheme = CodingScheme.from_name(ccs_scheme)
        icd9_scheme = CodingScheme.from_name(icd9_scheme)

        res = ccs_scheme.ccs_columns(icd9_scheme, icd9_add_dots)

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
            for j in range(1, ccs_scheme._N_LEVELS + 1):
                level = cols[f'I{j}'][i]
                if level != '':
                    last_index = level
            if last_index != None:
                icd_code = cols['ICD'][i]
                icd92ccs[icd_code].add(last_index)
                ccs2icd9[last_index].add(icd_code)

        CodeMap.register_map(icd9_scheme.name, ccs_scheme.name,
                             CodeMap(icd92ccs_config, dict(icd92ccs)))
        CodeMap.register_map(ccs_scheme.name, icd9_scheme.name,
                             CodeMap(ccs2icd9_config, dict(ccs2icd9)))

    @classmethod
    def parent_child_mappings(cls, df: pd.DataFrame) -> Dict[str, Set[str]]:
        """Make dictionary for parent-child connections."""
        pt2ch = {'root': set(df['I1'])}
        levels = list(map(lambda i: f'I{i}', range(1, cls._N_LEVELS + 1)))

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
        levels = list(map(lambda i: f'I{i}', range(1, cls._N_LEVELS + 1)))
        descs = list(map(lambda i: f'L{i}', range(1, cls._N_LEVELS + 1)))
        for code_col, desc_col in zip(levels, descs):
            df_ = df[df[code_col] != '']
            df_ = df_[[code_col, desc_col]].drop_duplicates()
            code_desc = dict(zip(df_[code_col], df_[desc_col]))
            desc.update(code_desc)
        return desc

    @staticmethod
    def _code_ancestors_dots(code: str, include_itself: bool = True) -> Set[str]:

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
        return cls._code_ancestors_dots(code, include_itself)

    @abstractmethod
    @classmethod
    def register_scheme(cls):
        pass


class DxCCSOps(CCSOps):
    _SCHEME_FILE = 'ccs_multi_dx_tool_2015.csv.gz'
    _N_LEVELS = 4

    @classmethod
    def register_scheme(cls):
        icd9_scheme = CodingScheme.from_name('dx_icd9')
        icd9_add_dots = DxICD9Ops.add_dots

        cols = cls.ccs_columns(icd9_scheme, icd9_add_dots)["cols"]
        df = pd.DataFrame(cols)
        pt2ch = cls.parent_child_mappings(df)
        desc = cls.desc_mappings(df)
        codes = sorted(desc.keys())
        CodingScheme.register_scheme(HierarchicalScheme(CodingSchemeConfig(name='dx_ccs'),
                                                        pt2ch=pt2ch,
                                                        codes=codes,
                                                        index=dict(zip(codes, range(len(codes)))),
                                                        desc=desc))


class PrCCSOps(CCSOps):
    _SCHEME_FILE = 'ccs_multi_pr_tool_2015.csv.gz'
    _N_LEVELS = 3

    @classmethod
    def register_scheme(cls):
        icd9_scheme = CodingScheme.from_name('pr_icd9')
        icd9_add_dots = PrICD9Ops.add_dots

        cols = cls.ccs_columns(icd9_scheme, icd9_add_dots)["cols"]
        df = pd.DataFrame(cols)
        pt2ch = cls.parent_child_mappings(df)
        desc = cls.desc_mappings(df)
        codes = sorted(desc.keys())
        CodingScheme.register_scheme(HierarchicalScheme(CodingSchemeConfig(name='pr_ccs'),
                                                        pt2ch=pt2ch,
                                                        codes=codes,
                                                        index=dict(zip(codes, range(len(codes)))),
                                                        desc=desc))


class FlatCCSOps(CCSOps):
    _SCHEME_FILE = None

    @classmethod
    def flatccs_columns(cls, icd9_scheme: CodingScheme, icd9_add_dots: Callable[[str], str]) -> Dict[str, List[str]]:
        filepath = os.path.join(_CCS_DIR, cls._SCHEME_FILE)
        df = pd.read_csv(filepath, skiprows=[0, 2], dtype=str)
        icd9_cname = '\'ICD-9-CM CODE\''
        cat_cname = '\'CCS CATEGORY\''
        desc_cname = '\'CCS CATEGORY DESCRIPTION\''
        df[icd9_cname] = df[icd9_cname].map(lambda c: c.strip('\'').strip())
        df[icd9_cname] = df[icd9_cname].map(icd9_add_dots)

        valid_icd9 = df[icd9_cname].isin(icd9_scheme.index)

        unrecognised_icd9 = set(df[~valid_icd9][icd9_cname])
        df = df[valid_icd9]

        code_col = list(df[cat_cname].map(lambda c: c.strip('\'').strip()))
        icd9_col = list(df[icd9_cname])
        desc_col = list(df[desc_cname].map(lambda d: d.strip('\'').strip()))

        return {
            'code': code_col,
            'icd9': icd9_col,
            'desc': desc_col,
            'unrecognised_icd9': unrecognised_icd9,
            'conv_file': cls._SCHEME_FILE
        }

    @staticmethod
    def register_mappings(flatccs_scheme: str, icd9_scheme: str, icd9_add_dots: Callable[[str], str]):
        flatccs_scheme = CodingScheme.from_name(flatccs_scheme)
        icd9_scheme = CodingScheme.from_name(icd9_scheme)

        res = flatccs_scheme.flatccs_columns(icd9_scheme, icd9_add_dots)

        flatccs2icd9_config = CodeMapConfig(
            flatccs_scheme.name,
            icd9_scheme.name,
            t_dag_space=False)
        icd92flatccs_config = CodeMapConfig(
            icd9_scheme.name,
            flatccs_scheme.name,
            t_dag_space=False)

        map_n1 = dict(zip(res['icd9'], res['code']))
        assert len(map_n1) == len(res['icd9']), "1toN mapping expected"

        flatccs2icd9 = defaultdict(set)
        icd92flatccs = defaultdict(set)
        for icd_code, ccode in map_n1.items():
            flatccs2icd9[ccode].add(icd_code)
            icd92flatccs[icd_code].add(ccode)
        flatccs2icd9 = dict(flatccs2icd9)
        icd92flatccs = dict(icd92flatccs)
        CodeMap.register_map(flatccs_scheme.name, icd9_scheme.name,
                             CodeMap(flatccs2icd9_config, flatccs2icd9))
        CodeMap.register_map(icd9_scheme.name, flatccs_scheme.name,
                             CodeMap(icd92flatccs_config, icd92flatccs))


class DxFlatCCSOps(FlatCCSOps):
    _SCHEME_FILE = '$dxref 2015.csv.gz'

    @classmethod
    def register_scheme(self):
        dx_icd9 = CodingScheme.from_name('dx_icd9')
        dx_add_dots = DxICD9Ops.add_dots

        cols = self.flatccs_columns(dx_icd9, dx_add_dots)
        codes = sorted(set(cols['code']))
        CodingScheme.register_scheme(FlatScheme(CodingSchemeConfig('dx_flatccs'),
                                                codes=codes,
                                                index=dict(zip(codes, range(len(codes)))),
                                                desc=dict(zip(cols['code'], cols['desc']))))


class PrFlatCCSOps(FlatCCSOps):
    _SCHEME_FILE = '$prref 2015.csv.gz'

    @classmethod
    def register_scheme(cls):
        pr_icd9 = CodingScheme.from_name('pr_icd9')
        pr_add_dots = PrICD9Ops.add_dots

        cols = cls.flatccs_columns(pr_icd9, pr_add_dots)
        codes = sorted(set(cols['code']))
        CodingScheme.register_scheme(FlatScheme(CodingSchemeConfig('pr_flatccs'),
                                                codes=codes,
                                                index=dict(zip(codes, range(len(codes)))),
                                                desc=dict(zip(cols['code'], cols['desc']))))


def setup_scheme_loaders():
    CodingScheme.register_scheme_loader('dx_icd10', DxICD10Ops.register_scheme)
    CodingScheme.register_scheme_loader('pr_icd10', PrICD10Ops.register_scheme)
    CodingScheme.register_scheme_loader('dx_icd9', DxICD9Ops.register_scheme)
    CodingScheme.register_scheme_loader('pr_icd9', PrICD9Ops.register_scheme)
    CodingScheme.register_scheme_loader('dx_ccs', DxCCSOps.register_scheme)
    CodingScheme.register_scheme_loader('pr_ccs', PrCCSOps.register_scheme)
    CodingScheme.register_scheme_loader('dx_flatccs', DxFlatCCSOps.register_scheme)
    CodingScheme.register_scheme_loader('pr_flatccs', PrFlatCCSOps.register_scheme)

    OutcomeExtractor.register_outcome_extractor_loader('dx_flatccs_mlhc_groups', 'dx_flatccs_mlhc_groups.json')
    OutcomeExtractor.register_outcome_extractor_loader('dx_flatccs_filter_v1', 'dx_flatccs_v1.json')
    OutcomeExtractor.register_outcome_extractor_loader('dx_icd9_filter_v1', 'dx_icd9_v1.json')
    OutcomeExtractor.register_outcome_extractor_loader('dx_icd9_filter_v2_groups', 'dx_icd9_v2_groups.json')
    OutcomeExtractor.register_outcome_extractor_loader('dx_icd9_filter_v3_groups', 'dx_icd9_v3_groups.json')


def setup_maps_loaders():
    # ICD9 <-> ICD10s
    CodeMap.register_map_loader('dx_icd10', 'dx_icd9',
                                lambda: ICDOps.register_mappings('dx_icd10', 'dx_icd9', DxICD10Ops.add_dots,
                                                                 DxICD9Ops.add_dots, '2018_gem_cm_I10I9.txt.gz'))
    CodeMap.register_map_loader('dx_icd9', 'dx_icd10',
                                lambda: ICDOps.register_mappings('dx_icd9', 'dx_icd10', DxICD9Ops.add_dots,
                                                                 DxICD10Ops.add_dots, '2018_gem_cm_I9I10.txt.gz'))
    CodeMap.register_map_loader('pr_icd10', 'pr_icd9',
                                lambda: ICDOps.register_mappings('pr_icd10', 'pr_icd9', PrICD10Ops.add_dots,
                                                                 PrICD9Ops.add_dots, '2018_gem_pcs_I10I9.txt.gz'))
    CodeMap.register_map_loader('pr_icd9', 'pr_icd10',
                                lambda: ICDOps.register_mappings('pr_icd9', 'pr_icd10', PrICD9Ops.add_dots,
                                                                 PrICD10Ops.add_dots, '2018_gem_pcs_I9I10.txt.gz'))

    # ICD9 <-> CCS
    bimap_dx_ccs_icd9 = lambda: CCSOps.register_mappings('dx_ccs', 'dx_icd9', DxICD9Ops.add_dots)

    CodeMap.register_map_loader('dx_icd9', 'dx_ccs',
                                bimap_dx_ccs_icd9)
    CodeMap.register_map_loader('dx_ccs', 'dx_icd9',
                                bimap_dx_ccs_icd9)

    bimap_pr_ccs_icd9 = lambda: CCSOps.register_mappings('pr_ccs', 'pr_icd9', PrICD9Ops.add_dots)
    CodeMap.register_map_loader('pr_icd9', 'pr_ccs',
                                bimap_pr_ccs_icd9)
    CodeMap.register_map_loader('pr_ccs', 'pr_icd9',
                                bimap_pr_ccs_icd9)

    bimap_dx_flatccs_icd9 = lambda: FlatCCSOps.register_mappings('dx_flatccs', 'dx_icd9', DxICD9Ops.add_dots)
    CodeMap.register_map_loader('dx_flatccs', 'dx_icd9',
                                bimap_dx_flatccs_icd9)
    CodeMap.register_map_loader('dx_icd9', 'dx_flatccs',
                                bimap_dx_flatccs_icd9)
    bimap_pr_flatccs_icd9 = lambda: FlatCCSOps.register_mappings('pr_flatccs', 'pr_icd9', PrICD9Ops.add_dots)
    CodeMap.register_map_loader('pr_flatccs', 'pr_icd9',
                                bimap_pr_flatccs_icd9)
    CodeMap.register_map_loader('pr_icd9', 'pr_flatccs',
                                bimap_pr_flatccs_icd9)
    # ICD10 <-> CCS (Through ICD9 as an intermediate scheme)
    CodeMap.register_chained_map_loader('dx_ccs', 'dx_icd9', 'dx_icd10')
    CodeMap.register_chained_map_loader('dx_icd10', 'dx_icd9', 'dx_ccs')
    CodeMap.register_chained_map_loader('pr_ccs', 'pr_icd9', 'pr_icd10')
    CodeMap.register_chained_map_loader('pr_icd10', 'pr_icd9', 'pr_ccs')
    CodeMap.register_chained_map_loader('dx_flatccs', 'dx_icd9', 'dx_icd10')
    CodeMap.register_chained_map_loader('dx_icd10', 'dx_icd9', 'dx_flatccs')
    CodeMap.register_chained_map_loader('pr_flatccs', 'pr_icd9', 'pr_icd10')
    CodeMap.register_chained_map_loader('pr_icd10', 'pr_icd9', 'pr_flatccs')

    # CCS <-> FlatCCS (Through ICD9 as an intermediate scheme)
    CodeMap.register_chained_map_loader('dx_flatccs', 'dx_icd9', 'dx_ccs')
    CodeMap.register_chained_map_loader('dx_ccs', 'dx_icd9', 'dx_flatccs')
    CodeMap.register_chained_map_loader('pr_flatccs', 'pr_icd9', 'pr_ccs')
    CodeMap.register_chained_map_loader('pr_ccs', 'pr_icd9', 'pr_flatccs')