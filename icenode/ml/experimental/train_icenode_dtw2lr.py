from .train_icenode_dtw import ICENODE as ICENODE_DTW
from .train_icenode_2lr import ICENODE_2LR_MIXIN


class ICENODE(ICENODE_2LR_MIXIN, ICENODE_DTW):

    def __init__(self, **kwargs):
        ICENODE_DTW.__init__(self, **kwargs)


if __name__ == '__main__':
    from .hpo_utils import capture_args, run_trials
    run_trials(model_cls=ICENODE, **capture_args())
