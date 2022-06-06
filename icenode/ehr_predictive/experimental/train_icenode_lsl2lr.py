from .train_icenode_lsl import ICENODE as ICENODE_LSL
from .train_icenode_2lr import ICENODE_2LR_MIXIN


class ICENODE(ICENODE_2LR_MIXIN, ICENODE_LSL):

    def __init__(self, **kwargs):
        ICENODE_LSL.__init__(self, **kwargs)


if __name__ == '__main__':
    from .hpo_utils import capture_args, run_trials
    run_trials(model_cls=ICENODE, **capture_args())
