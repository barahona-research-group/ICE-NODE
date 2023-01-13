from .dx_icenode_uniform import ICENODE as ICENODE_UNIFORM
from .dx_icenode_2lr import ICENODE_2LR_MIXIN


class ICENODE(ICENODE_2LR_MIXIN, ICENODE_UNIFORM):

    def __init__(self, **kwargs):
        ICENODE_UNIFORM.__init__(self, **kwargs)

    def _f_n_ode(self, params, count_nfe, state, t, c={}):
        # Identity, return the input state as it is.
        n = {i: 0 for i in t}
        r = {i: 0.0 for i in t}
        s = {i: state[i] for i in t}
        return s, r, n


