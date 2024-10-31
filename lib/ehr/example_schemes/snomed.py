from __future__ import annotations

import json
from dataclasses import field
from typing import Set, Self, Tuple, Optional, Dict

import networkx as nx
import pandas as pd

from lib.ehr.coding_scheme import (HierarchicalScheme, FrozenDict11, FrozenDict1N)
from lib.utils import tqdm_constructor


# from lib.utils import tqdm_constructor


class SNOMEDCT(HierarchicalScheme):
    cdb_df: pd.DataFrame = field(kw_only=True)
    cdb_inactive_df: pd.DataFrame = field(kw_only=True)
    active_terms: Set[str] = field(kw_only=True)

    @classmethod
    def from_files(cls, name: str, cdb_active_path: str, cdb_inactive_path: str, ch2pt_json_path: str) -> Self:
        cdb_df = pd.read_csv(cdb_active_path, index_col=0)
        active_terms = set(cdb_df.cui.unique())
        _df1 = cdb_df[cdb_df.tty == 1].reset_index(drop=True)
        _df1 = (_df1.groupby('cui', as_index=False).agg(name=('str', lambda x: x.values[0])))

        cdb_inactive_df = pd.read_csv(cdb_inactive_path, index_col=0)
        inactive_terms = set(cdb_inactive_df.cui.unique())
        _df2 = cdb_inactive_df[cdb_inactive_df.tty == 1].reset_index(drop=True)
        _df2 = (_df2.groupby('cui', as_index=False).agg(name=('str', lambda x: x.values[0])))

        desc = dict(zip(_df1.cui, _df1.name)) | dict(zip(_df2.cui, _df2.name))

        with open(ch2pt_json_path) as json_file:
            ch2pt = {ch: set(pts) for ch, pts in json.load(json_file).items()}

        return cls(
            name=name,
            codes=tuple(sorted(active_terms | inactive_terms)),
            desc=FrozenDict11.from_dict(desc),
            cdb_df=cdb_df, cdb_inactive_df=cdb_inactive_df, active_terms=active_terms,
            ch2pt=FrozenDict1N.from_dict(ch2pt))

    def to_networkx(self,
                    codes: Tuple[str, ...] = None,
                    discard_set: Optional[Set[str]] = None,
                    node_attrs: Optional[Dict[str, Dict[str, str]]] = None) -> nx.DiGraph:
        """
        Generate a networkx.DiGraph (Directed Graph) from a table of SNOMED-CT codes.

        Args:
            codes (Tuple[str, ...]): The table of codes, must have a column `core_code` for the SNOMED-CT codes.
            discard_set (Optional[Set[str]]): A set of codes, which, if provided, they are excluded from
                the Graph object.
            node_attrs: A dictionary of node attributes, which, if provided, used to annotate nodes with additional
                information, such as the frequency of the corresponding SNOMED-CT code in a particular dataset.
        """

        if codes is None:
            codes = set(self.codes) & set(self.ch2pt.keys())

        def parents_traversal(x):
            ch2pt_edges = set()

            def parents_traversal_(node):
                for pt in self.ch2pt.get(node, set()):
                    ch2pt_edges.add((node, pt))
                    parents_traversal_(pt)

            parents_traversal_(x)
            return ch2pt_edges

        if discard_set:
            ch2pt_edges = [parents_traversal(c) for c in tqdm_constructor((c for c in codes if c not in discard_set))]

        else:
            ch2pt_edges = [parents_traversal(c) for c in tqdm_constructor(codes)]

        dag = nx.DiGraph()

        for ch, pt in set().union(*ch2pt_edges):
            dag.add_edge(ch, pt)

        if node_attrs is not None:
            for node in tqdm_constructor(dag.nodes):
                for attr_name, attr_dict in node_attrs.items():
                    dag.nodes[node][attr_name] = attr_dict.get(node, '')
        return dag
