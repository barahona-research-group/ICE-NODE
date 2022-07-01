"""Extract diagnostic/procedure information of CCS files into new
data structures to support conversion between CCS and ICD9."""

from collections import defaultdict
import os
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
class AbstractScheme:
    maps = {}

    def __init__(self, codes, index, desc):
        self.codes = codes
        self.index = index
        self.desc = desc

    @staticmethod
    def add_map(src_cls, target_cls, mapping):
        AbstractScheme.maps[(src_cls, target_cls)] = mapping


class HierarchicalScheme(AbstractScheme):
    def __init__(self,
                 dag_codes=None,
                 dag_index=None,
                 dag_desc=None,
                 pt2ch=None,
                 ch2pt=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.dag_codes = dag_codes or kwargs['codes']
        self.dag_index = dag_index or kwargs['index']
        self.dag_desc = dag_desc or kwargs['desc']

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

    def code_ancestors(self, code):
        raise OOPError('Must be overriden')

    @staticmethod
    def _code_ancestors_dots(code):
        if code == 'root':
            return []

        indices = code.split('.')
        ancestors = {'root'}
        for i in reversed(range(1, len(indices))):
            parent = '.'.join(indices[0:i])
            ancestors.add(parent)
        return ancestors

    @staticmethod
    def _bfs_traversal(connection, code, include_itself):
        result = set()
        q = [code]

        while len(q) != 0:
            # remove the first element from the stack
            current_code = q.pop(0)
            current_connections = connection.get(current_code, [])
            q.extend([c for c in current_connections if c not in result])
            if current_code not in result:
                result.add(current_code)

        if not include_itself:
            result.remove(code)
        return result

    @staticmethod
    def _dfs_traversal(connection, code, include_itself):
        result = {code} if include_itself else set()

        def _traversal(_node):
            for conn in connection.get(_node, []):
                result.add(conn)
                _traversal(conn)

        _traversal(code)

        return result

    @staticmethod
    def _dfs_edges(connection, code):
        result = set()

        def _edges(_node):
            for conn in connection.get(_node, []):
                result.add((_node, conn))
                _edges(conn)

        _edges(code)
        return result

    @staticmethod
    def _deselect_subtree(pt2ch, sub_root):
        to_del = HierarchicalScheme._bfs_traversal(pt2ch, sub_root, True)
        pt2ch = pt2ch.copy()
        to_del = to_del & set(pt2ch.keys())
        for node_idx in to_del:
            del pt2ch[node_idx]
        return pt2ch

    @staticmethod
    def _select_subtree(pt2ch, sub_root):
        to_keep = HierarchicalScheme._bfs_traversal(pt2ch, sub_root, True)
        to_keep = to_keep & set(pt2ch.keys())
        return {idx: pt2ch[idx] for idx in to_keep}

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


class DxICD10(HierarchicalScheme):
    pass


class PrICD10(HierarchicalScheme):
    pass


class DxICD9(HierarchicalScheme):
    _PR_ROOT_CLASS_ID = 'MM_CLASS_2'

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
        nodes = set.union(pt2ch.keys(), *pt2ch.values())

        # Filter out the procedure code from the df.
        df = df[df['NODE_IDX'].isin(nodes)]

        d = self.generate_dictionaries(df)

        super().__init__(dag_codes=d['dag_codes'],
                         dag_index=d['dag_index'],
                         dag_desc=d['dag_desc'],
                         pt2ch=pt2ch,
                         codes=d['icd_codes'],
                         index=d['icd_index'],
                         desc=d['icd_desc'])


class PrICD9(HierarchicalScheme):
    def __init__(self):
        df = pd.DataFrame(DxICD9.icd9_columns())
        pt2ch = DxICD9.parent_child_mappings(df)

        # Remove the procedure codes.
        pt2ch = DxICD9._select_subtree(pt2ch, DxICD9._PR_ROOT_CLASS_ID)

        # Remaining node indices in one set.
        nodes = set.union(pt2ch.keys(), *pt2ch.values())

        # Filter out the procedure code from the df.
        df = df[df['NODE_IDX'].isin(nodes)]

        d = DxICD9.generate_dictionaries(df)

        super().__init__(dag_codes=d['dag_codes'],
                         dag_index=d['dag_index'],
                         dag_desc=d['dag_desc'],
                         pt2ch=pt2ch,
                         codes=d['icd_codes'],
                         index=d['icd_index'],
                         desc=d['icd_desc'])


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
        icd92ccs = {'root': 'root'}
        ccs2icd9 = defaultdict(list)
        ccs2icd9['root'] = ['root']
        n_rows = len(cols['ICD'])
        for i in range(n_rows):
            last_index = None
            for j in range(1, n_levels + 1):
                level = cols[f'I{j}'][i]
                if level != '':
                    last_index = level
            if last_index != None:
                icode = cols['ICD'][i]
                icd92ccs[icode] = last_index
                ccs2icd9[last_index].append(icode)
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
        cols = self.ccs_columns('ccs_multi_dx_tool_2015.csv', 4)
        df = pd.DataFrame(cols)
        icd92ccs, ccs2icd9 = self.icd9_mappings(cols, 4)
        pt2ch = self.parent_child_mappings(df, 4)
        desc = self.desc_mappings(df, 4)
        codes = sorted(desc.keys())

        super().__init__(pt2ch=pt2ch,
                         codes=codes,
                         index=dict(zip(codes, range(len(codes)))),
                         desc=desc)

        AbstractScheme.add_map(DxCCS, DxICD9, ccs2icd9)
        AbstractScheme.add_map(DxICD9, DxCCS, icd92ccs)

    # Get parents of CCS code
    @classmethod
    def code_parents(cls, code):
        return cls._code_ancestors_dots(code)


class PrCCS(HierarchicalScheme):
    def __init__(self):
        cols = DxCCS.ccs_columns('ccs_multi_pr_tool_2015.csv', 3)
        df = pd.DataFrame(cols)
        icd92ccs, ccs2icd9 = DxCCS.icd9_mappings(cols, 3)
        pt2ch = DxCCS.parent_child_mappings(df, 3)
        desc = DxCCS.desc_mappings(df, 3)
        codes = sorted(desc.keys())

        super().__init__(pt2ch=pt2ch,
                         codes=codes,
                         index=dict(zip(codes, range(len(codes)))),
                         desc=desc)

        AbstractScheme.add_map(PrCCS, PrICD9, ccs2icd9)
        AbstractScheme.add_map(PrICD9, PrCCS, icd92ccs)


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
        cols = self.flatccs_columns('$dxref 2015 filtered.csv')
        codes = sorted(cols['code'])
        super().__init__(codes=codes,
                         index=dict(zip(codes, range(len(codes)))),
                         desc=dict(zip(cols['code'], cols['desc'])))

        icd92flatccs = dict(zip(cols['icd9'], cols['code']))
        flatccs2icd9 = defaultdict(list)
        for icode, ccode in icd92flatccs.items():
            flatccs2icd9[ccode].append(icode)

        self.add_map(DxFlatCCS, DxICD9, flatccs2icd9)
        self.add_map(DxICD9, DxFlatCCS, icd92flatccs)


class PrFlatCCS(AbstractScheme):
    def __init__(self):
        cols = DxFlatCCS.flatccs_columns('$prref 2015.csv')
        codes = sorted(cols['code'])
        super().__init__(codes=codes,
                         index=dict(zip(codes, range(len(codes)))),
                         desc=dict(zip(cols['code'], cols['desc'])))

        icd92flatccs = dict(zip(cols['icd9'], cols['code']))
        flatccs2icd9 = defaultdict(list)
        for icode, ccode in icd92flatccs.items():
            flatccs2icd9[ccode].append(icode)
        self.add_map(PrFlatCCS, PrICD9, flatccs2icd9)
        self.add_map(PrICD9, PrFlatCCS, icd92flatccs)


# Singleton instance.
code_scheme = {
    'dx_flatccs': lambda: DxFlatCCS(),
    'dx_ccs': lambda: DxCCS(),
    'dx_icd9': lambda: DxICD9(),
    'dx_icd10': lambda: DxICD10(),
    'pr_flatccs': lambda: PrFlatCCS(),
    'pr_ccs': lambda: PrCCS(),
    'pr_icd9': lambda: PrICD9(),
    'pr_icd10': lambda: PrICD10()
}
