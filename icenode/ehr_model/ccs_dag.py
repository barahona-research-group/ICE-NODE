"""Extract diagnostic/procedure information of CCS files into new
data structures to support conversion between CCS and ICD9."""

from collections import defaultdict
import os
import pandas as pd


class CCSDAG:
    """
    A class that wraps CCS dictionaries and enable functionalities
    related to the hierarchical structure of CCS,
    like retrieving ancestry and descedants.
    """
    DIR = os.path.dirname(__file__)
    CCS_DIR = os.path.join(DIR, 'resources', 'CCS')
    DIAG_SINGLE_CCS_FILE = os.path.join(CCS_DIR, '$dxref 2015 filtered.csv')
    PROC_SINGLE_CCS_FILE = os.path.join(CCS_DIR, '$prref 2015.csv')
    DIAG_MULTI_CCS_FILE = os.path.join(CCS_DIR, 'ccs_multi_dx_tool_2015.csv')
    PROC_MULTI_CCS_FILE = os.path.join(CCS_DIR, 'ccs_multi_pr_tool_2015.csv')

    def __init__(self):

        self.dx_flatccs_df = pd.read_csv(self.DIAG_SINGLE_CCS_FILE, skiprows=1)
        self.proc_flatccs_df = pd.read_csv(self.PROC_SINGLE_CCS_FILE,
                                           skiprows=1)
        self.dx_ccs_df = pd.read_csv(self.DIAG_MULTI_CCS_FILE)
        self.proc_ccs_df = pd.read_csv(self.PROC_MULTI_CCS_FILE)

        self.dx_icd_label = self.make_dx_icd_dict()
        self.proc_icd_label = self.make_proc_icd_dict()

        self.dx_icd_codes = list(sorted(self.dx_icd_label.keys()))
        self.proc_icd_codes = list(sorted(self.proc_icd_label.keys()))

        (self.dx_icd2flatccs, self.dx_flatccs2icd,
         self.dx_flatccs_desc) = self.make_dx_icd2ccs_dict()
        (self.proc_icd2flatccs,
         self.proc_flatccs2icd) = self.make_proc_icd2ccs_dict()

        (self.dx_ccs_pt2ch, self.dx_icd2ccs, self.dx_ccs2icd,
         self.dx_ccs_codes,
         self.dx_ccs_labels) = self.make_dx_multi_dictionaries()
        (self.proc_ccs_pt2ch, self.proc_icd2ccs, self.proc_ccs2icd,
         self.proc_ccs_codes,
         self.proc_ccs_labels) = self.make_proc_multi_dictionaries()

        self.dx_flatccs_codes = list(sorted(self.dx_flatccs2icd.keys()))

        self.dx_ccs_idx = dict(
            zip(self.dx_ccs_codes, range(len(self.dx_ccs_codes))))
        self.dx_flatccs_idx = dict(
            zip(self.dx_flatccs_codes, range(len(self.dx_flatccs_codes))))

    def make_dx_icd_dict(self):
        dx_ccs_label_list = self.dx_flatccs_df[
            '\'ICD-9-CM CODE DESCRIPTION\''].apply(
                lambda cat: cat.strip('\'').strip()).tolist()
        dx_ccs_icd_list = self.dx_flatccs_df['\'ICD-9-CM CODE\''].apply(
            lambda c: c.strip('\'').strip()).tolist()
        return dict(zip(dx_ccs_icd_list, dx_ccs_label_list))

    def make_proc_icd_dict(self):
        proc_ccs_label_list = self.proc_flatccs_df[
            '\'ICD-9-CM CODE DESCRIPTION\''].apply(
                lambda cat: cat.strip('\'').strip()).tolist()
        proc_ccs_icd_list = self.proc_flatccs_df['\'ICD-9-CM CODE\''].apply(
            lambda c: c.strip('\'').strip()).tolist()
        return dict(zip(proc_ccs_icd_list, proc_ccs_label_list))

    def make_dx_icd2ccs_dict(self):
        dx_ccs_cat_list = self.dx_flatccs_df['\'CCS CATEGORY\''].apply(
            lambda cat: cat.strip('\'').strip()).tolist()
        dx_ccs_icd_list = self.dx_flatccs_df['\'ICD-9-CM CODE\''].apply(
            lambda c: c.strip('\'').strip()).tolist()

        dx_ccs_cat_desc = self.dx_flatccs_df[
            '\'CCS CATEGORY DESCRIPTION\''].apply(
                lambda desc: desc.strip('\'').strip()).tolist()

        dx_ccs_desc = dict(zip(dx_ccs_cat_list, dx_ccs_cat_desc))

        dx_icd2ccs_dict = dict(zip(dx_ccs_icd_list, dx_ccs_cat_list))

        dx_ccs2icd_dict = defaultdict(list)
        for code, cat in zip(dx_ccs_icd_list, dx_ccs_cat_list):
            dx_ccs2icd_dict[cat].append(code)

        return dx_icd2ccs_dict, dx_ccs2icd_dict, dx_ccs_desc

    def make_proc_icd2ccs_dict(self):
        proc_ccs_cat_list = self.proc_flatccs_df['\'CCS CATEGORY\''].apply(
            lambda cat: cat.strip('\'').strip()).tolist()
        proc_ccs_icd_list = self.proc_flatccs_df['\'ICD-9-CM CODE\''].apply(
            lambda c: c.strip('\'').strip()).tolist()

        proc_icd2ccs_dict = dict(zip(proc_ccs_icd_list, proc_ccs_cat_list))

        proc_ccs2icd_dict = defaultdict(list)
        for code, cat in zip(proc_ccs_icd_list, proc_ccs_cat_list):
            proc_ccs2icd_dict[cat].append(code)

        return proc_icd2ccs_dict, proc_ccs2icd_dict

    def make_dx_multi_dictionaries(self):
        df = self.dx_ccs_df.copy()
        df['I1'] = df['\'CCS LVL 1\''].apply(lambda l: l.strip('\'').strip())
        df['I2'] = df['\'CCS LVL 2\''].apply(lambda l: l.strip('\'').strip())
        df['I3'] = df['\'CCS LVL 3\''].apply(lambda l: l.strip('\'').strip())
        df['I4'] = df['\'CCS LVL 4\''].apply(lambda l: l.strip('\'').strip())
        df['L1'] = df['\'CCS LVL 1 LABEL\''].apply(
            lambda l: l.strip('\'').strip())
        df['L2'] = df['\'CCS LVL 2 LABEL\''].apply(
            lambda l: l.strip('\'').strip())
        df['L3'] = df['\'CCS LVL 3 LABEL\''].apply(
            lambda l: l.strip('\'').strip())
        df['L4'] = df['\'CCS LVL 4 LABEL\''].apply(
            lambda l: l.strip('\'').strip())
        df['ICD'] = df['\'ICD-9-CM CODE\''].apply(
            lambda l: l.strip('\'').strip())

        df = df[['I1', 'I2', 'I3', 'I4', 'L1', 'L2', 'L3', 'L4', 'ICD']]

        dx_multi_icd2ccs = {'root': 'root'}
        dx_multi_ccs2icd = defaultdict(list)
        dx_multi_ccs2icd['root'] = ['root']
        for row in df.itertuples():
            code = row.ICD
            i1, i2, i3, i4 = row.I1, row.I2, row.I3, row.I4
            last_index = None
            if i1 != '':
                last_index = i1
            if i2 != '':
                last_index = i2
            if i3 != '':
                last_index = i3
            if i4 != '':
                last_index = i4
            if last_index != None:
                dx_multi_icd2ccs[code] = last_index
                dx_multi_ccs2icd[last_index].append(code)

        # Make dictionary for parent-child connections
        dx_multi_ccs_pt2ch = {'root': set(df['I1'])}
        for pt_col, ch_col in zip(('I1', 'I2', 'I3'), ('I2', 'I3', 'I4')):
            df_ = df[(df[pt_col] != '') & (df[ch_col] != '')]
            df_ = df_[[pt_col, ch_col]].drop_duplicates()
            for parent_ccs_code, ch_ccs_df in df_.groupby(pt_col):
                dx_multi_ccs_pt2ch[parent_ccs_code] = set(ch_ccs_df[ch_col])

        # Make a dictionary for CCS labels
        dx_multi_ccs_labels = {'root': 'root'}
        for idx_col, label_col in zip(('I1', 'I2', 'I3', 'I4'),
                                      ('L1', 'L2', 'L3', 'L4')):
            df_ = df[df[idx_col] != '']
            df_ = df_[[idx_col, label_col]].drop_duplicates()
            idx_label = dict(zip(df_[idx_col], df_[label_col]))
            dx_multi_ccs_labels.update(idx_label)

        dx_multi_ccs_codes = list(sorted(dx_multi_ccs_labels.keys()))
        return (dx_multi_ccs_pt2ch, dx_multi_icd2ccs, dx_multi_ccs2icd,
                dx_multi_ccs_codes, dx_multi_ccs_labels)

    def make_proc_multi_dictionaries(self):
        df = self.proc_ccs_df.copy()
        df['I1'] = df['\'CCS LVL 1\''].apply(lambda l: l.strip('\'').strip())
        df['I2'] = df['\'CCS LVL 2\''].apply(lambda l: l.strip('\'').strip())
        df['I3'] = df['\'CCS LVL 3\''].apply(lambda l: l.strip('\'').strip())
        df['L1'] = df['\'CCS LVL 1 LABEL\''].apply(
            lambda l: l.strip('\'').strip())
        df['L2'] = df['\'CCS LVL 2 LABEL\''].apply(
            lambda l: l.strip('\'').strip())
        df['L3'] = df['\'CCS LVL 3 LABEL\''].apply(
            lambda l: l.strip('\'').strip())
        df['ICD'] = df['\'ICD-9-CM CODE\''].apply(
            lambda l: l.strip('\'').strip())

        df = df[['I1', 'I2', 'I3', 'L1', 'L2', 'L3', 'ICD']]

        proc_multi_icd2ccs = {'root': 'root'}
        proc_multi_ccs2icd = defaultdict(list)
        proc_multi_ccs2icd['root'] = ['root']
        for row in df.itertuples():
            code = row.ICD
            i1, i2, i3 = row.I1, row.I2, row.I3
            last_index = i1
            if i2 != '':
                last_index = i2
            if i3 != '':
                last_index = i3
            proc_multi_icd2ccs[code] = last_index
            proc_multi_ccs2icd[last_index].append(code)

        # Make dictionary for parent-child connections
        proc_multi_ccs_pt2ch = {'root': set(df['I1'])}
        for pt_col, ch_col in zip(('I1', 'I2'), ('I2', 'I3')):
            df_ = df[(df[pt_col] != '') & (df[ch_col] != '')]
            df_ = df_[[pt_col, ch_col]].drop_duplicates()
            for parent_ccs_code, ch_ccs_df in df_.groupby(pt_col):
                proc_multi_ccs_pt2ch[parent_ccs_code] = set(ch_ccs_df[ch_col])

        # Make a dictionary for CCS labels
        proc_multi_ccs_labels = {'root': 'root'}
        for idx_col, label_col in zip(('I1', 'I2', 'I3'), ('L1', 'L2', 'L3')):
            df_ = df[df[idx_col] != '']
            df_ = df_[[idx_col, label_col]].drop_duplicates()
            idx_label = dict(zip(df_[idx_col], df_[label_col]))
            proc_multi_ccs_labels.update(idx_label)

        proc_multi_ccs_codes = list(sorted(proc_multi_ccs_labels.keys()))
        return (proc_multi_ccs_pt2ch, proc_multi_icd2ccs, proc_multi_ccs2icd,
                proc_multi_ccs_codes, proc_multi_ccs_labels)

    def find_dx_icd_name(self, code):
        return self.dx_icd_label[code]

    def find_proc_icd_name(self, code):
        return self.proc_icd_label[code]

    def get_dx_ccs(self, icd_dx_code):
        return self.dx_icd2ccs[icd_dx_code]

    def get_proc_ccs(self, icd_proc_code):
        return self.proc_icd2ccs[icd_proc_code]

    # Get parents of CCS code
    def get_ccs_parents(self, ccs_code):
        if ccs_code == 'root':
            return []

        indices = ccs_code.split('.')
        parents = ['root']
        for i in reversed(range(1, len(indices))):
            parent = '.'.join(indices[0:i])
            parents.append(parent)
        return parents

    # Get all children of CCS code
    def get_dx_ccs_children(self, ccs_code):
        result = []
        q = [ccs_code]

        while len(q) != 0:
            # remove the first element from the stack
            current_ccs = q.pop(0)
            expanded_ccs = self.dx_ccs_pt2ch.get(current_ccs, [])
            q.extend([c for c in expanded_ccs if c not in result])
            if current_ccs not in result:
                result.append(current_ccs)
        result.remove(ccs_code)
        return result

    # Get all children of CCS code
    def get_proc_ccs_children(self, ccs_code):
        result = []
        q = [ccs_code]

        while len(q) != 0:
            # remove the first element from the stack
            current_ccs = q.pop(0)
            expanded_ccs = self.proc_ccs_pt2ch.get(current_ccs, [])
            q.extend([c for c in expanded_ccs if c not in result])
            if current_ccs not in result:
                result.append(current_ccs)
        result.remove(ccs_code)
        return result

    def dx_ccs_children_traversal(self, ccs_code):
        children_set = set()

        def _children_traversal(_node):
            for ch in self.dx_ccs_pt2ch.get(_node, []):
                children_set.add(ch)
                _children_traversal(ch)

        _children_traversal(ccs_code)
        return children_set

    def proc_ccs_children_traversal(self, ccs_code):
        children_set = set()

        def _children_traversal(_node):
            for ch in self.proc_ccs_pt2ch.get(_node, []):
                children_set.add(ch)
                _children_traversal(ch)

        _children_traversal(ccs_code)
        return children_set

    def ancestors_linkage(self, ccs_code):
        ancestors_set = set()

        def _ancestors_linkage(c):
            for pt in self.get_ccs_parents(c):
                ancestors_set.add((c, pt))
                _ancestors_linkage(pt)

        _ancestors_linkage(ccs_code)
        return ancestors_set

    def common_parent(self, ccs1, ccs2):
        for parent in self.get_ccs_parents(ccs1):
            if parent in self.get_ccs_parents(ccs2):
                return parent


# Singleton instance.
ccs_dag = CCSDAG()
