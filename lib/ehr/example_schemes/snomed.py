from __future__ import annotations

import json
from typing import Set, Self

import pandas as pd
import networkx as nx
from lib.ehr.coding_scheme import (HierarchicalScheme,
                                   FrozenDict11)


class SNOMEDCT(HierarchicalScheme):
    cdb_df: pd.DataFrame
    cdb_inactive_df: pd.DataFrame
    active_terms: Set[str]
    inactive_terms: Set[str]

    @classmethod
    def from_files(cls, cdb_active_path: str, cdb_inactive_path: str, ch2pt_json_path: str) -> Self:
        cdb_df = pd.read_csv(cdb_active_path, index_col=0)
        active_terms = set(cdb_df.cui.unique())
        _df1 = cdb_df[cdb_df.tty == 1].reset_index(drop=True)
        _df1 = (_df1.groupby('cui', as_index=False).agg(name=('str', lambda x: x.values[0])))

        cdb_inactive_df = pd.read_csv(cdb_inactive_path, index_col=0)
        inactive_terms = set(cdb_inactive_df.cui.unique())

        _df2 = cdb_inactive_df[cdb_inactive_df.tty == 1].reset_index(drop=True)
        _df2 = (_df2.groupby('cui', as_index=False).agg(name=('str', lambda x: x.values[0])))

        concept_name = {
            **dict(zip(_df1.cui, _df1.name)),
            **dict(zip(_df2.cui, _df2.name))
        }

        with open(ch2pt_json_path) as json_file:
            ch2pt = json.load(json_file)

        return cls(
            codes=tuple(cdb_df.cui.unique()),
            desc=FrozenDict11.from_dict(concept_name),
            cdb_df=cdb_df, cdb_inactive_df=cdb_inactive_df, active_terms=active_terms,
            inactive_terms=inactive_terms, ch2pt=ch2pt)

    def digraph_from_dataframe(self, df, discard_set=None, node_attrs=None):
        """
        Generate a networkx.DiGraph (Directed Graph) from a table of SNOMED-CT codes.

        Args:
            df (pd.DataFrame): The table of codes, must have a column `core_code` for the SNOMED-CT codes.
            discard_set (Optional[Set[str]]): A set of codes, which if provided, they are excluded from the Graph object.
            node_attrs: A dictionary of node attributes, which if provided, used to annotate nodes with additional information,
                        such as the frequency of the corresponding SNOMED-CT code in a particular dataset.
        """
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