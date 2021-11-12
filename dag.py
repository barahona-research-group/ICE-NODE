import pandas as pd
import json
import networkx as nx
import re


class SNOMEDDAG:
    """
    A class that wraps SNOMED dictionaries and enable functionalities
    related to the hierarchical structure of SNOMED CT scheme,
    like retrieving ancestry and descedants.
    """
    def __init__(self, snomed_cdb_active, snomed_cdb_inactive, ch2pt_json,
                 pt2ch_json):
        self.snomed_cdb_df = pd.read_csv(snomed_cdb_active, index_col=0)
        self.active_terms = set(self.snomed_cdb_df.cui.unique())
        _df1 = self.snomed_cdb_df[self.snomed_cdb_df.tty == 1].reset_index(
            drop=True)

        _df1 = (_df1.groupby('cui',
                             as_index=False).agg(name=('str',
                                                       lambda x: x.values[0])))

        if snomed_cdb_inactive is not None:
            self.snomed_cdb_inactive_df = pd.read_csv(snomed_cdb_inactive,
                                                      index_col=0)
            self.inactive_terms = set(self.snomed_cdb_inactive_df.cui.unique())

            _df2 = self.snomed_cdb_inactive_df[self.snomed_cdb_inactive_df.tty
                                               == 1].reset_index(drop=True)

            _df2 = (_df2.groupby(
                'cui',
                as_index=False).agg(name=('str', lambda x: x.values[0])))

            self.concept_name = {
                **dict(zip(_df1.cui, _df1.name)),
                **dict(zip(_df2.cui, _df2.name))
            }

        else:
            self.concept_name = {**dict(zip(_df1.cui, _df1.name))}

        with open(ch2pt_json) as json_file:
            self.ch2pt = json.load(json_file)

        with open(pt2ch_json) as json_file:
            self.pt2ch = json.load(json_file)

    def __contains__(self, code):
        return code in self.ch2pt or code in self.pt2ch

    def find_name(self, code):
        return self.concept_name.get(code, '')

    def find_names(self, code):
        """
        Converts SNOMED code to Fully specified name and finds any Synonyms
        """
        if code in self.active_terms:
            df = self.snomed_cdb_df[(self.snomed_cdb_df['cui'] == code)
                                    & (self.snomed_cdb_df['tty'] == 1)]
            concept_name = df['str'].values
            return f"{' | '.join(concept_name)}"
        elif code in self.inactive_terms:
            df = self.snomed_cdb_inactive_df[
                (self.snomed_cdb_inactive_df['cui'] == code)
                & (self.snomed_cdb_inactive_df['tty'] == 1)]
            concept_name = df['str'].values
            return f"{' | '.join(concept_name)}"
        else:
            return ''

    def find_syn(self, code):
        """
        Converts SNOMED code and finds all Synonyms. Not including concept name
        """
        df = self.snomed_cdb_df[(self.snomed_cdb_df['cui'] == code)
                                & (self.snomed_cdb_df['tty'] == 0)]
        synonym = df['str'].to_list()
        return f"{'; '.join(synonym)}"

    def search_regex(self, query, regex_flags=0):
        names1 = self.snomed_cdb_df['str']
        names2 = self.snomed_cdb_inactive_df['str']
        rows1 = names1.str.contains(query, regex=True, flags=regex_flags, na=False)
        rows2 = names2.str.contains(query, regex=True, flags=regex_flags, na=False)

        codes1 = self.snomed_cdb_df[rows1]
        codes2 = self.snomed_cdb_inactive_df[rows2]

        codes1 = codes1[~codes1.isnull()]
        codes2 = codes2[~codes2.isnull()]
        return list(set(codes1.cui) | set(codes2.cui))

    # Get parents of snomed code
    def get_parents(self, code):
        result = []
        q = [code]

        while len(q) != 0:
            # remove the first element from the stack
            current_snomed = q.pop(0)
            current_snomed_children = self.ch2pt.get(current_snomed, [])
            q.extend([c for c in current_snomed_children if c not in q])
            result.append(current_snomed)
        result.remove(code)
        return result

    # Get all children of snomed code
    def get_children(self, code):
        result = []
        q = [code]

        while len(q) != 0:
            # remove the first element from the stack
            current_snomed = q.pop(0)
            current_snomed_parent = self.pt2ch.get(current_snomed, [])
            q.extend([c for c in current_snomed_parent if c not in result])
            if current_snomed not in result:
                result.append(current_snomed)
        result.remove(code)
        return result

    def children_traversal(self, code):
        children_set = set()

        def _children_traversal(_node):
            for ch in self.pt2ch.get(_node, []):
                children_set.add(ch)
                _children_traversal(ch)

        _children_traversal(code)
        return children_set

    def ancestors_linkage(self, code):
        ancestors_set = set()

        def _ancestors_linkage(c):
            for pt in self.ch2pt.get(c, []):
                ancestors_set.add((c, pt))
                _ancestors_linkage(pt)

        _ancestors_linkage(code)
        return ancestors_set

    def common_parent(self, snomed1, snomed2):
        for parent in self.get_parents(snomed1):
            if parent in self.get_parents(snomed2):
                return parent

    def get_parent_from_list(self, snomed_list):
        if len(snomed_list) == 2:
            return self.common_parent(snomed_list[0], snomed_list[1])
        else:
            first_snomed = snomed_list[0]
            rest_of_snomed = snomed_list[1:]
            rest_of_snomed_parent = self.get_parent_from_list(rest_of_snomed)
            return self.common_parent(first_snomed, rest_of_snomed_parent)

    def prune_subsets(self, branches):
        branches = [[c, s] for c, s in branches.items()]

        branches = sorted(branches, key=lambda t: len(t[1]))

        for i in range(len(branches)):
            i_code, i_set = branches[i]
            i_parents = set(self.get_parents(i_code))
            for j in range(i + 1, len(branches)):
                j_code, j_set = branches[j]
                if j_code in i_parents:
                    branches[j][1] = j_set - i_set

        return {c: s for c, s in branches}

    def dag_from_dataframe(self, df, discard_set=None, node_attrs=None):
        _ch2pt = set()
        dag = nx.DiGraph()

        def parents_traversal(node):
            for pt in self.ch2pt.get(node, []):
                _ch2pt.add((node, pt))
                parents_traversal(pt)

        if discard_set:
            pruned_df = df[df.core_code.apply(lambda c: c not in discard_set)]
            pruned_df.core_code.apply(lambda c: parents_traversal(c))
        else:
            df.core_code.apply(lambda c: parents_traversal(c))

        for ch, pt in _ch2pt:
            dag.add_edge(ch, pt)

        if node_attrs is not None:
            for node in dag.nodes:
                for attr_name, attr_dict in node_attrs.items():
                    dag.nodes[node][attr_name] = attr_dict.get(node, '')
        return dag

    def digraph_from_dataframe(self, df, discard_set=None, node_attrs=None):
        _ch2pt = set()
        dag = nx.DiGraph()

        def parents_traversal(node):
            for pt in self.ch2pt.get(node, []):
                _ch2pt.add((node, pt))
                parents_traversal(pt)

        if discard_set:
            pruned_df = df[df.core_code.apply(lambda c: c not in discard_set)]
            pruned_df.core_code.apply(lambda c: parents_traversal(c))
        else:
            df.core_code.apply(lambda c: parents_traversal(c))

        for ch, pt in _ch2pt:
            dag.add_edge(ch, pt)

        if node_attrs:
            for node in dag.nodes:
                for attr_name, attr_dict in node_attrs.items():
                    dag.nodes[node][attr_name] = attr_dict.get(node, '')
        return dag
