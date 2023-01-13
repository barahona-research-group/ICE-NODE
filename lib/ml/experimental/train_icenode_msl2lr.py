from .train_icenode_msl import ICENODE as ICENODE_MSL
from .train_icenode_m2lr import ICENODE_2LR_MIXIN


class ICENODE(ICENODE_2LR_MIXIN, ICENODE_MSL):

    def __init__(self, **kwargs):
        ICENODE_MSL.__init__(self, **kwargs)


if __name__ == '__main__':
    from .hpo_utils import capture_args, run_trials
    run_trials(model_cls=ICENODE, **capture_args())
