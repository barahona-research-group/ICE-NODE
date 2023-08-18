    # @staticmethod
    # def _emb_subtrees(pytree):
    #     return (pytree.dx_emb, pytree.dx_dec)

    # @staticmethod
    # def emb_dyn_partition(pytree):
    #     """
    #     Separate the dynamics parameters from the embedding parameters.
    #     Thanks to Patrick Kidger for the clever function of eqx.partition.
    #     """
    #     emb_leaves = jtu.tree_leaves(AbstractModel._emb_subtrees(pytree))
    #     emb_predicate = lambda _t: any(_t is t for t in emb_leaves)
    #     emb_tree, dyn_tree = eqx.partition(pytree, emb_predicate)
    #     return emb_tree, dyn_tree

    # @staticmethod
    # def emb_dyn_merge(emb_tree, dyn_tree):
    #     return eqx.combine(emb_tree, dyn_tree)


#     @staticmethod
#     def emb_dyn_partition(pytree: InICENODE):
#         """
#         Separate the dynamics parameters from the embedding parameters.
#         Thanks to Patrick Kidger for the clever function of eqx.partition.
#         """
#         dyn_leaves = jtu.tree_leaves(pytree.f_dyn)
#         dyn_predicate = lambda _t: any(_t is t for t in dyn_leaves)
#         dyn_tree, emb_tree = eqx.partition(pytree, dyn_predicate)
#         return emb_tree, dyn_tree

#     @staticmethod
#     def emb_dyn_merge(emb_tree, dyn_tree):
#         return eqx.combine(emb_tree, dyn_tree)


