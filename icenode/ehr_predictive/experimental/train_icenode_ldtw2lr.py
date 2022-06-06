from .train_icenode_ldtw import ICENODE as ICENODE_LDTW
from .train_icenode_2lr import ICENODE_2LR_MIXIN


class ICENODE(ICENODE_2LR_MIXIN, ICENODE_LDTW):

    def __init__(self, **kwargs):
        ICENODE_LDTW.__init__(self, **kwargs)


if __name__ == '__main__':
    from .hpo_utils import capture_args, run_trials
    run_trials(model_cls=ICENODE, **capture_args())
