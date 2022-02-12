from .train_icenode_mtl import ICENODE as ICENODE_MTL
from .train_icenode_2lr import ICENODE as ICENODE_2LR

class ICENODE(ICENODE_2LR, ICENODE_MTL):
    pass

if __name__ == '__main__':
    from .hpo_utils import capture_args, run_trials
    run_trials(model_cls=ICENODE, **capture_args())
