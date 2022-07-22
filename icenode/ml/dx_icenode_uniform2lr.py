from .dx_icenode_uniform import ICENODE as ICENODE_UNIFORM
from .dx_icenode_2lr import ICENODE_2LR_MIXIN


class ICENODE(ICENODE_2LR_MIXIN, ICENODE_UNIFORM):

    def __init__(self, **kwargs):
        ICENODE_UNIFORM.__init__(self, **kwargs)
