"""."""

from .dx_icenode_tl import ICENODE as ICENODE_TL


class ICENODE(ICENODE_TL):

    @staticmethod
    def _time_diff(t1, t2):
        return 7.0
