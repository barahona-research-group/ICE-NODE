from collections import defaultdict
import json
import re
import os

import pandas as pd

import networkx as nx


class CCSDAG:
    """
    A class that wraps CCS dictionaries and enable functionalities
    related to the hierarchical structure of CCS,
    like retrieving ancestry and descedants.
    """
    DIR = os.path.dirname(__file__)
    CCS_DIR = os.path.join(DIR, 'resources', 'CCS')
    DIAG_SINGLE_CCS_FILE = os.path.join(CCS_DIR, '$dxref 2015.csv')
    PROC_SINGLE_CCS_FILE = os.path.join(CCS_DIR, '$prref 2015.csv')
    DIAG_MULTI_CCS_FILE = os.path.join(CCS_DIR, 'ccs_multi_dx_tool_2015.csv')
    PROC_MULTI_CCS_FILE = os.path.join(CCS_DIR, 'ccs_multi_pr_tool_2015.csv')

    def __init__(self):

        self.diag_single_ccs_df = pd.read_csv(self.DIAG_SINGLE_CCS_FILE,
                                              skiprows=1)
        self.proc_single_ccs_df = pd.read_csv(self.PROC_SINGLE_CCS_FILE,
                                              skiprows=1)
        self.diag_multi_ccs_df = pd.read_csv(self.DIAG_MULTI_CCS_FILE)
        self.proc_multi_ccs_df = pd.read_csv(self.PROC_MULTI_CCS_FILE)

        self.diag_icd_label = self.make_diag_icd_dict()
        self.proc_icd_label = self.make_proc_icd_dict()

        self.diag_icd_codes = list(self.diag_icd_label.keys())
        self.proc_icd_codes = list(self.proc_icd_label.keys())

        (self.diag_single_icd2ccs,
         self.diag_single_ccs2icd) = self.make_diag_icd2ccs_dict()
        (self.proc_signle_icd2ccs,
         self.proc_single_ccs2icd) = self.make_proc_icd2ccs_dict()

        (self.diag_multi_ccs_pt2ch, self.diag_multi_icd2ccs,
         self.diag_multi_ccs2icd) = self.make_diag_multi_dictionaries()
        (self.proc_multi_ccs_pt2ch, self.proc_multi_icd2ccs,
         self.proc_multi_ccs2icd) = self.make_proc_multi_dictionaries()

        self.diag_single_ccs_codes = list(self.diag_single_ccs2icd.keys())
        self.diag_multi_ccs_codes = list(self.diag_multi_ccs2icd.keys())
        self.proc_multi_ccs_codes = list(self.proc_multi_ccs2icd.keys())

    def make_diag_icd_dict(self):
        diag_ccs_label_list = self.diag_single_ccs_df[
            '\'ICD-9-CM CODE DESCRIPTION\''].apply(
                lambda cat: cat.strip('\'').strip()).tolist()
        diag_ccs_icd_list = self.diag_single_ccs_df['\'ICD-9-CM CODE\''].apply(
            lambda c: c.strip('\'').strip()).tolist()
        return dict(zip(diag_ccs_icd_list, diag_ccs_label_list))

    def make_proc_icd_dict(self):
        proc_ccs_label_list = self.proc_single_ccs_df[
            '\'ICD-9-CM CODE DESCRIPTION\''].apply(
                lambda cat: cat.strip('\'').strip()).tolist()
        proc_ccs_icd_list = self.proc_single_ccs_df['\'ICD-9-CM CODE\''].apply(
            lambda c: c.strip('\'').strip()).tolist()
        return dict(zip(proc_ccs_icd_list, proc_ccs_label_list))

    def make_diag_icd2ccs_dict(self):
        diag_ccs_cat_list = self.diag_single_ccs_df['\'CCS CATEGORY\''].apply(
            lambda cat: cat.strip('\'').strip()).tolist()
        diag_ccs_icd_list = self.diag_single_ccs_df['\'ICD-9-CM CODE\''].apply(
            lambda c: c.strip('\'').strip()).tolist()

        diag_icd2ccs_dict = dict(zip(diag_ccs_icd_list, diag_ccs_cat_list))

        diag_ccs2icd_dict = defaultdict(list)
        for code, cat in zip(diag_ccs_icd_list, diag_ccs_cat_list):
            diag_ccs2icd_dict[cat].append(code)

        return diag_icd2ccs_dict, diag_ccs2icd_dict

    def make_proc_icd2ccs_dict(self):
        proc_ccs_cat_list = self.proc_single_ccs_df['\'CCS CATEGORY\''].apply(
            lambda cat: cat.strip('\'').strip()).tolist()
        proc_ccs_icd_list = self.proc_single_ccs_df['\'ICD-9-CM CODE\''].apply(
            lambda c: c.strip('\'').strip()).tolist()

        proc_icd2ccs_dict = dict(zip(proc_ccs_icd_list, proc_ccs_cat_list))

        proc_ccs2icd_dict = defaultdict(list)
        for code, cat in zip(proc_ccs_icd_list, proc_ccs_cat_list):
            proc_ccs2icd_dict[cat].append(code)

        return proc_icd2ccs_dict, proc_ccs2icd_dict

    def make_diag_multi_dictionaries(self):
        df = self.diag_multi_ccs_df.copy()
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
        diag_multi_ccs_pt2ch = defaultdict(list)
        diag_multi_icd2ccs = {}
        diag_multi_ccs2icd = defaultdict(list)

        for row in df.itertuples():
            code = row.ICD
            l1, l2, l3, l4 = row.L1, row.L2, row.L3, row.L4
            i1, i2, i3, i4 = row.I1, row.I2, row.I3, row.I4

            last_index = i1

            if i2:
                diag_multi_ccs_pt2ch[i1].append(i2)
                last_index = i2

            if i3:
                diag_multi_ccs_pt2ch[i2].append(i3)
                last_index = i3
            if i4:
                diag_multi_ccs_pt2ch[i3].append(i4)
                last_index = i4

            diag_multi_icd2ccs[code] = last_index
            diag_multi_ccs2icd[last_index].append(code)

        return diag_multi_ccs_pt2ch, diag_multi_icd2ccs, diag_multi_ccs2icd

    def make_proc_multi_dictionaries(self):
        df = self.proc_multi_ccs_df.copy()
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


        proc_multi_ccs_pt2ch = defaultdict(list)
        proc_multi_icd2ccs = {}
        proc_multi_ccs2icd = defaultdict(list)

        for row in df.itertuples():
            code = row.ICD
            l1, l2, l3 = row.L1, row.L2, row.L3
            i1, i2, i3 = row.I1, row.I2, row.I3

            last_index = i1

            if i2:
                proc_multi_ccs_pt2ch[i1].append(i2)
                last_index = i2

            if i3:
                proc_multi_ccs_pt2ch[i2].append(i3)
                last_index = i3

            proc_multi_icd2ccs[code] = last_index
            proc_multi_ccs2icd[last_index].append(code)

        return proc_multi_ccs_pt2ch, proc_multi_icd2ccs, proc_multi_ccs2icd

    def find_diag_icd_name(self, code):
        return self.diag_icd_label[code]

    def find_proc_icd_name(self, code):
        return self.proc_icd_label[code]

    def get_diag_multi_ccs(self, icd9_diag_code):
        return self.diag_multi_icd2ccs[icd9_diag_code]

    def get_proc_multi_ccs(self, icd9_proc_code):
        return self.proc_multi_icd2ccs[icd9_proc_code]

    # Get parents of CCS code
    def get_ccs_parents(self, ccs_code):
        indices = ccs_code.split('.')
        parents = []
        for i in reversed(range(1, len(indices))):
            parent = '.'.join(indices[0:i])
            parents.append(parent)
        return parents

    # Get all children of CCS code
    def get_diag_ccs_children(self, ccs_code):
        result = []
        q = [ccs_code]

        while len(q) != 0:
            # remove the first element from the stack
            current_ccs = q.pop(0)
            expanded_ccs = self.diag_multi_ccs_pt2ch.get(current_ccs, [])
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
            expanded_ccs = self.proc_multi_ccs_pt2ch.get(current_ccs, [])
            q.extend([c for c in expanded_ccs if c not in result])
            if current_ccs not in result:
                result.append(current_ccs)
        result.remove(ccs_code)
        return result

    def diag_ccs_children_traversal(self, ccs_code):
        children_set = set()

        def _children_traversal(_node):
            for ch in self.diag_multi_ccs_pt2ch.get(_node, []):
                children_set.add(ch)
                _children_traversal(ch)

        _children_traversal(ccs_code)
        return children_set

    def proc_ccs_children_traversal(self, ccs_code):
        children_set = set()

        def _children_traversal(_node):
            for ch in self.proc_multi_ccs_pt2ch.get(_node, []):
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

    def digraph_from_dataframe(self, df, node_attrs=None):
        # TODO: DataFrame will have ICD, CCS_MULTI, CCS_SINGLE columns
        _ch2pt = set()
        dag = nx.DiGraph()

        def parents_traversal(ccs_code):
            parents = self.get_ccs_parents(ccs_code)
            _ch2pt.update(zip([ccs_code] + parents[:-1], parents))

        df['CCS_MULTI'].apply(lambda c: parents_traversal(c))

        for ch, pt in _ch2pt:
            dag.add_edge(ch, pt)

        if node_attrs:
            for node in dag.nodes:
                for attr_name, attr_dict in node_attrs.items():
                    dag.nodes[node][attr_name] = attr_dict.get(node, '')
        return dag
