from functools import partial
from typing import Dict, Any
import jax
from .train_icenode_tl import ICENODE as ICENODE_TL
from .utils import tree_map

class ICENODE_STAGED_MIXIN:

    @classmethod
    def step_optimizer(cls, step, model_state, batch):
        opt_state, opt_update, get_params, loss_, reg_hyperparams = model_state
        params = get_params(opt_state)
        grads = jax.grad(loss_)(params, batch)
        if step > 50:
            reg_hyperparams['L_dyn'] = 0
            for label in grads:
                if label != 'f_n_ode':
                    grads[label] = tree_map(lambda g: g * 0, grads[label])

        opt_state = opt_update(step, grads, opt_state)
        return opt_state, opt_update, get_params, loss_, reg_hyperparams


class ICENODE(ICENODE_STAGED_MIXIN, ICENODE_TL):

    def __init__(self, **kwargs):
        ICENODE_TL.__init__(self, **kwargs)


if __name__ == '__main__':
    from .hpo_utils import capture_args, run_trials
    run_trials(model_cls=ICENODE, **capture_args())
