"""Extract diagnostic/procedure information of CCS files into new
data structures to support conversion between CCS and ICD9."""

from collections import defaultdict, OrderedDict
import os
import gzip
import xml.etree.ElementTree as ET

import pandas as pd

from ..utils import OOPError, LazyDict

_DIR = os.path.dirname(__file__)
_RSC_DIR = os.path.join(_DIR, 'resources')
_CCS_DIR = os.path.join(_RSC_DIR, 'CCS')
"""
Testing ideas:

    - ICD9: #nodes > #ICD codes
    - test *dfs vs *bfs algorithms.


"""
"""
Ideas in mapping

- For a certain scenario, a choice list of multiple code is represented by their common ancestor.
- Use OrderedSet in ancestor/successor retrieval to retain the order of traversal (important in Breadth-firth search to have them sorted by level).
"""


class AbstractScheme:
    maps = {}

    def __init__(self, codes, index, desc):
        self.codes = codes
        self.index = index
        self.desc = desc

    @staticmethod
    def add_map(src_cls, target_cls, mapping):
        AbstractScheme.maps[(src_cls, target_cls)] = mapping

    @staticmethod
    def get_map(src_obj, target_obj):
        return AbstractScheme.maps[(type(src_obj), type(target_obj))]


class HierarchicalScheme(AbstractScheme):

    def __init__(self,
                 dag_codes=None,
                 dag_index=None,
                 dag_desc=None,
                 code2dag=None,
                 pt2ch=None,
                 ch2pt=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.dag_codes = dag_codes or kwargs['codes']
        self.dag_index = dag_index or kwargs['index']
        self.dag_desc = dag_desc or kwargs['desc']

        if code2dag is not None:
            self.code2dag = code2dag
        else:
            # Identity
            self.code2dag = {c: c for c in kwargs['codes']}

        assert pt2ch or ch2pt, "Should provide ch2pt or pt2ch connection dictionary"
        if ch2pt and pt2ch:
            self.ch2pt = ch2pt
            self.pt2ch = pt2ch
        if pt2ch is not None:
            # Direct parent-children relationship
            self.pt2ch = pt2ch
            # Direct children-parents relashionship
            self.ch2pt = self.reverse_connection(pt2ch)
        elif ch2pt is not None:
            self.ch2pt = ch2pt
            self.pt2ch = self.reverse_connection(ch2pt)

    @staticmethod
    def reverse_connection(connection):
        rconnection = defaultdict(set)
        for node, conns in connection.items():
            for conn in conns:
                rconnection[conn].add(node)
        return rconnection

    @staticmethod
    def _bfs_traversal(connection, code, include_itself):
        result = OrderedDict()
        q = [code]

        while len(q) != 0:
            # remove the first element from the stack
            current_code = q.pop(0)
            current_connections = connection.get(current_code, [])
            q.extend([c for c in current_connections if c not in result])
            if current_code not in result:
                result[current_code] = 1

        if not include_itself:
            del result[code]
        return list(result.keys())

    @staticmethod
    def _dfs_traversal(connection, code, include_itself):
        result = {code} if include_itself else set()

        def _traversal(_node):
            for conn in connection.get(_node, []):
                result.add(conn)
                _traversal(conn)

        _traversal(code)

        return list(result)

    @staticmethod
    def _dfs_edges(connection, code):
        result = set()

        def _edges(_node):
            for conn in connection.get(_node, []):
                result.add((_node, conn))
                _edges(conn)

        _edges(code)
        return result

    def code_ancestors_bfs(self, code, include_itself=True):
        return self._bfs_traversal(self.ch2pt, code, include_itself)

    def code_ancestors_dfs(self, code, include_itself=True):
        return self._dfs_traversal(self.ch2pt, code, include_itself)

    def code_successors_bfs(self, code, include_itself=True):
        return self._bfs_traversal(self.pt2ch, code, include_itself)

    def code_successors_dfs(self, code, include_itself=True):
        return self._dfs_traversal(self.pt2ch, code, include_itself)

    def ancestors_edges_dfs(self, code):
        return self._dfs_edges(self.ch2pt, code)

    def successors_edges_dfs(self, code):
        return self._dfs_edges(self.pt2ch, code)

    def least_common_ancestor(self, codes):
        if len(codes) == 1:
            return codes[0]
        else:
            a, b = codes[:2]
            a_ancestors = self.code_ancestors_bfs(a, True)
            b_ancestors = self.code_ancestors_bfs(b, True)
            for ancestor in a_ancestors:
                if ancestor in b_ancestors:
                    return self.least_common_ancestor([ancestor] + codes[2:])
            raise RuntimeError('Unresolved common ancestor!')


class CCSCommons:

    @staticmethod
    def _code_ancestors_dots(code, include_itself=True):

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


class ICDCommons:

    @staticmethod
    def _deselect_subtree(pt2ch, sub_root):
        to_del = HierarchicalScheme._bfs_traversal(pt2ch, sub_root, True)
        pt2ch = pt2ch.copy()
        to_del = set(to_del) & set(pt2ch.keys())
        for node_idx in to_del:
            del pt2ch[node_idx]
        return pt2ch

    @staticmethod
    def _select_subtree(pt2ch, sub_root):
        to_keep = HierarchicalScheme._bfs_traversal(pt2ch, sub_root, True)
        to_keep = set(to_keep) & set(pt2ch.keys())
        return {idx: pt2ch[idx] for idx in to_keep}

    @classmethod
    def load_conv_table(cls, conv_fname):
        conv_fname = os.path.join(_RSC_DIR, conv_fname)
        df = pd.read_csv(conv_fname,
                         sep='\s+',
                         dtype=str,
                         names=['source', 'target', 'meta'])
        df['approximate'] = df['meta'].apply(lambda s: s[0])
        df['no_map'] = df['meta'].apply(lambda s: s[1])
        df['combination'] = df['meta'].apply(lambda s: s[2])
        df['scenario'] = df['meta'].apply(lambda s: s[3])
        df['choice_list'] = df['meta'].apply(lambda s: s[4])
        df['source'] = df['source'].map(cls.add_dot)
        df['target'] = df['target'].map(cls.add_dot)

        return df

    @classmethod
    def analyse_conversions(cls, conv_fname):
        df = cls.load_conv_table(conv_fname)
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

    def distill_conversion_table(self, conv_fname):
        # For choice_list, represent each group by there common ancestor
        def _resolve_choice_list(df):
            represent = set()
            for _, choice_list_df in df.groupby('choice_list'):
                choice_list = choice_list_df['target'].map(self.code2dag)
                if choice_list.isnull().sum() > 0:
                    raise RuntimeError(
                        f'Failed mapping to DAG space: {list(choice_list_df["target"])} -> {list(choice_list)}'
                    )

                choice_list = list(choice_list)
                if len(choice_list) > 1:
                    lca = self.least_common_ancestor(choice_list)
                    represent.add(lca)
                else:
                    represent.add(choice_list[0])
            return represent

        conv_df = self.load_conv_table(conv_fname)
        status_df = self.analyse_conversions(conv_fname)
        map_kind = dict(zip(status_df['code'], status_df['status']))
        mapping = {}
        for code, df in conv_df.groupby('source'):
            kind = map_kind[code]
            if kind == 'no_map':
                continue
            elif kind == '11_map' or kind == '1n_map':
                mapping[code] = set(df['target'])
            elif kind == '1n_map(resolved)':
                mapping[code] = _resolve_choice_list(df)
            elif kind == 'ambiguous':
                represent = set()
                for _, scenario_df in df.groupby('scenario'):
                    represent.update(_resolve_choice_list(scenario_df))
                mapping[code] = represent
        return mapping


class DxICD10(HierarchicalScheme, ICDCommons):
    """
    NOTE: for prediction targets, remember to exclude the following chapters:
        - 'chapter:19': 'Injury, poisoning and certain other consequences of external causes (S00-T88)',
        - 'chapter:20': 'External causes of morbidity (V00-Y99)',
        - 'chapter:21': 'Factors influencing health status and contact with health services (Z00-Z99)',
        - 'chapter:22': 'Codes for special purposes (U00-U85)'
    """

    @staticmethod
    def add_dot(code):
        if len(code) > 3:
            return code[:3] + '.' + code[3:]
        else:
            return code

    @staticmethod
    def distill_icd10_xml(filename):
        # https://www.cdc.gov/nchs/icd/Comprehensive-Listing-of-ICD-10-CM-Files.htm
        _ICD10_FILE = os.path.join(_RSC_DIR, filename)
        with gzip.open(_ICD10_FILE, 'r') as f:
            tree = ET.parse(f)
        root = tree.getroot()
        pt2ch = defaultdict(set)
        desc = {root.tag: root.tag}
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
            pt2ch[root.tag].add(ch_name)
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
            'pt2ch': pt2ch
        }

    def __init__(self):
        # https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2023/
        super().__init__(
            **self.distill_icd10_xml('icd10cm_tabular_2023.xml.gz'))

        map9to10 = self.distill_conversion_table('2018_gem_cm_I9I10.txt.gz')
        self.add_map(DxICD9, DxICD10, map9to10)


class PrICD10(HierarchicalScheme, ICDCommons):

    @staticmethod
    def add_dot(code):
        # No decimal point in ICD10-PCS
        return code

    @staticmethod
    def distill_icd10_xml(filename):
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

    def __init__(self):
        super().__init__(
            **self.distill_icd10_xml('icd10pcs_codes_2023.txt.gz'))

        map9to10 = self.distill_conversion_table('2018_gem_pcs_I9I10.txt.gz')
        self.add_map(PrICD9, PrICD10, map9to10)


class DxICD9(HierarchicalScheme, ICDCommons):
    _PR_ROOT_CLASS_ID = 'MM_CLASS_2'

    @staticmethod
    def add_dot(code):
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
    def icd9_columns():
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
    def parent_child_mappings(df):
        pt2ch = {}
        for pt, ch_df in df.groupby('PARENT_IDX'):
            pt2ch[pt] = set(ch_df['NODE_IDX'])
        return pt2ch

    @staticmethod
    def generate_dictionaries(df):
        # df version for leaf nodes only (with non-empty ICD9 codes)
        df_leaves = df[df['ICD9'] != '']

        icd2dag = dict(zip(df_leaves['ICD9'], df_leaves['NODE_IDX']))

        # df version for internal nodes only (with empty ICD9 codes)
        df_internal = df[df['ICD9'] == '']

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

    def __init__(self):
        df = pd.DataFrame(self.icd9_columns())
        pt2ch = self.parent_child_mappings(df)

        # Remove the procedure codes.
        pt2ch = self._deselect_subtree(pt2ch, self._PR_ROOT_CLASS_ID)

        # Remaining node indices in one set.
        nodes = set().union(set(pt2ch), *pt2ch.values())

        # Filter out the procedure code from the df.
        df = df[df['NODE_IDX'].isin(nodes)]

        d = self.generate_dictionaries(df)

        super().__init__(dag_codes=d['dag_codes'],
                         dag_index=d['dag_index'],
                         dag_desc=d['dag_desc'],
                         code2dag=d['icd2dag'],
                         pt2ch=pt2ch,
                         codes=d['icd_codes'],
                         index=d['icd_index'],
                         desc=d['icd_desc'])

        map10to9 = self.distill_conversion_table('2018_gem_cm_I10I9.txt.gz')
        self.add_map(DxICD10, DxICD9, map10to9)


class PrICD9(HierarchicalScheme, ICDCommons):

    @staticmethod
    def add_dot(code):
        if len(code) > 2:
            return code[:2] + '.' + code[2:]
        else:
            return code

    def __init__(self):
        df = pd.DataFrame(DxICD9.icd9_columns())
        pt2ch = DxICD9.parent_child_mappings(df)

        # Remove the procedure codes.
        pt2ch = self._select_subtree(pt2ch, DxICD9._PR_ROOT_CLASS_ID)

        # Remaining node indices in one set.
        nodes = set().union(set(pt2ch), *pt2ch.values())

        # Filter out the procedure code from the df.
        df = df[df['NODE_IDX'].isin(nodes)]

        d = DxICD9.generate_dictionaries(df)

        super().__init__(dag_codes=d['dag_codes'],
                         dag_index=d['dag_index'],
                         dag_desc=d['dag_desc'],
                         code2dag=d['icd2dag'],
                         pt2ch=pt2ch,
                         codes=d['icd_codes'],
                         index=d['icd_index'],
                         desc=d['icd_desc'])

        map10to9 = self.distill_conversion_table('2018_gem_pcs_I10I9.txt.gz')
        self.add_map(PrICD10, PrICD9, map10to9)


class DxCCS(HierarchicalScheme):

    @staticmethod
    def ccs_columns(filename, n_levels):
        DX_CCS_FILE = os.path.join(_CCS_DIR, filename)
        df = pd.read_csv(DX_CCS_FILE)
        cols = {}
        for i in range(1, n_levels + 1):
            cols[f'I{i}'] = list(
                df[f'\'CCS LVL {i}\''].apply(lambda l: l.strip('\'').strip()))
            cols[f'L{i}'] = list(df[f'\'CCS LVL {i} LABEL\''].apply(
                lambda l: l.strip('\'').strip()))
        cols['ICD'] = list(
            df['\'ICD-9-CM CODE\''].apply(lambda l: l.strip('\'').strip()))
        return cols

    @staticmethod
    def icd9_mappings(cols, n_levels):
        icd92ccs = defaultdict(set)
        icd92ccs['root'] = {'root'}
        ccs2icd9 = defaultdict(set)
        ccs2icd9['root'] = {'root'}
        n_rows = len(cols['ICD'])
        for i in range(n_rows):
            last_index = None
            for j in range(1, n_levels + 1):
                level = cols[f'I{j}'][i]
                if level != '':
                    last_index = level
            if last_index != None:
                icode = cols['ICD'][i]
                icd92ccs[icode].add(last_index)
                ccs2icd9[last_index].add(icode)
        return icd92ccs, ccs2icd9

    @staticmethod
    def parent_child_mappings(df, n_levels):
        """Make dictionary for parent-child connections."""
        pt2ch = {'root': set(df['I1'])}
        levels = list(map(lambda i: f'I{i}', range(1, n_levels + 1)))

        for pt_col, ch_col in zip(levels[:-1], levels[1:]):
            df_ = df[(df[pt_col] != '') & (df[ch_col] != '')]
            df_ = df_[[pt_col, ch_col]].drop_duplicates()
            for parent_code, ch_df in df_.groupby(pt_col):
                pt2ch[parent_code] = set(ch_df[ch_col])
        return pt2ch

    @staticmethod
    def desc_mappings(df, n_levels):
        """Make a dictionary for CCS labels."""
        desc = {'root': 'root'}
        levels = list(map(lambda i: f'I{i}', range(1, n_levels + 1)))
        descs = list(map(lambda i: f'L{i}', range(1, n_levels + 1)))
        for code_col, desc_col in zip(levels, descs):
            df_ = df[df[code_col] != '']
            df_ = df_[[code_col, desc_col]].drop_duplicates()
            code_desc = dict(zip(df_[code_col], df_[desc_col]))
            desc.update(code_desc)
        return desc

    def __init__(self):
        cols = self.ccs_columns('ccs_multi_dx_tool_2015.csv.gz', 4)
        df = pd.DataFrame(cols)
        icd92ccs, ccs2icd9 = self.icd9_mappings(cols, 4)
        pt2ch = self.parent_child_mappings(df, 4)
        desc = self.desc_mappings(df, 4)
        codes = sorted(desc.keys())

        super().__init__(pt2ch=pt2ch,
                         codes=codes,
                         index=dict(zip(codes, range(len(codes)))),
                         desc=desc)

        self.add_map(DxCCS, DxICD9, ccs2icd9)
        self.add_map(DxICD9, DxCCS, icd92ccs)

    @classmethod
    def code_ancestors(cls, code, include_itself):
        return cls._code_ancestors_dots(code, include_itself)


class PrCCS(HierarchicalScheme):

    def __init__(self):
        cols = DxCCS.ccs_columns('ccs_multi_pr_tool_2015.csv.gz', 3)
        df = pd.DataFrame(cols)
        icd92ccs, ccs2icd9 = DxCCS.icd9_mappings(cols, 3)
        pt2ch = DxCCS.parent_child_mappings(df, 3)
        desc = DxCCS.desc_mappings(df, 3)
        codes = sorted(desc.keys())

        super().__init__(pt2ch=pt2ch,
                         codes=codes,
                         index=dict(zip(codes, range(len(codes)))),
                         desc=desc)

        self.add_map(PrCCS, PrICD9, ccs2icd9)
        self.add_map(PrICD9, PrCCS, icd92ccs)

    @classmethod
    def code_ancestors(cls, code, include_itself=True):
        return cls._code_ancestors_dots(code, include_itself)


class DxFlatCCS(AbstractScheme):

    @staticmethod
    def flatccs_columns(fname):
        filepath = os.path.join(_CCS_DIR, fname)
        df = pd.read_csv(filepath, skiprows=1)

        code_col = list(
            df['\'CCS CATEGORY\''].apply(lambda cat: cat.strip('\'').strip()))
        icd9_col = list(
            df['\'ICD-9-CM CODE\''].apply(lambda c: c.strip('\'').strip()))
        desc_col = df['\'CCS CATEGORY DESCRIPTION\''].apply(
            lambda desc: desc.strip('\'').strip()).tolist()

        return {'code': code_col, 'icd9': icd9_col, 'desc': desc_col}

    def __init__(self):
        cols = self.flatccs_columns('$dxref 2015.csv.gz')
        codes = sorted(set(cols['code']))
        super().__init__(codes=codes,
                         index=dict(zip(codes, range(len(codes)))),
                         desc=dict(zip(cols['code'], cols['desc'])))

        icd92flatccs = dict(zip(cols['icd9'], cols['code']))
        assert len(icd92flatccs) == len(cols['icd9']), "1toN mapping expected"

        flatccs2icd9 = defaultdict(set)
        for icode, ccode in icd92flatccs.items():
            flatccs2icd9[ccode].add(icode)

        icd92flatccs = {k: {v} for k, v in icd92flatccs.items()}

        self.add_map(DxFlatCCS, DxICD9, flatccs2icd9)
        self.add_map(DxICD9, DxFlatCCS, icd92flatccs)


class PrFlatCCS(AbstractScheme):

    def __init__(self):
        cols = DxFlatCCS.flatccs_columns('$prref 2015.csv.gz')
        codes = sorted(set(cols['code']))
        super().__init__(codes=codes,
                         index=dict(zip(codes, range(len(codes)))),
                         desc=dict(zip(cols['code'], cols['desc'])))

        icd92flatccs = dict(zip(cols['icd9'], cols['code']))
        assert len(icd92flatccs) == len(cols['icd9']), "1toN mapping expected"

        flatccs2icd9 = defaultdict(set)
        for icode, ccode in icd92flatccs.items():
            flatccs2icd9[ccode].add(icode)

        icd92flatccs = {k: {v} for k, v in icd92flatccs.items()}
        self.add_map(PrFlatCCS, PrICD9, flatccs2icd9)
        self.add_map(PrICD9, PrFlatCCS, icd92flatccs)


# Singleton instance.
code_scheme = LazyDict({
    'dx_flatccs': lambda: DxFlatCCS(),
    'dx_ccs': lambda: DxCCS(),
    'dx_icd9': lambda: DxICD9(),
    'dx_icd10': lambda: DxICD10(),
    'pr_flatccs': lambda: PrFlatCCS(),
    'pr_ccs': lambda: PrCCS(),
    'pr_icd9': lambda: PrICD9(),
    'pr_icd10': lambda: PrICD10()
})
