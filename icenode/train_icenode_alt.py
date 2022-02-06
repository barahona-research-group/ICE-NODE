from typing import (Any, Callable, Dict, Iterable, List, Optional, Set)

import jax

from .jax_interface import (DiagnosisJAXInterface)
from .gram import AbstractEmbeddingsLayer
from .train_icenode_tl import ICENODE as ICENODE_TL


class ICENODE(ICENODE_TL):

    def __init__(self, subject_interface: DiagnosisJAXInterface,
                 diag_emb: AbstractEmbeddingsLayer, ode_dyn: str,
                 ode_with_bias: bool, ode_init_var: float, state_size: int,
                 timescale: float):
        super().__init__(subject_interface=subject_interface,
                         diag_emb=diag_emb,
                         ode_dyn=ode_dyn,
                         ode_with_bias=ode_with_bias,
                         ode_init_var=ode_init_var,
                         state_size=state_size,
                         timescale=timescale)

    def step_optimizer(self, step, opt_object, batch):
        opt_state, opt_update, get_params, loss_, loss_mixing = opt_object
        params = get_params(opt_state)

        if (step / 15) % 2 == 0:
            interval_norm = True
        else:
            interval_norm = False

        grads = jax.grad(loss_)(params,
                                batch,
                                count_nfe=False,
                                interval_norm=interval_norm)
        opt_state = opt_update(step, grads, opt_state)
        return opt_state, opt_update, get_params, loss_, loss_mixing


if __name__ == '__main__':
    from .hpo_utils import capture_args, run_trials
    run_trials(model_cls=ICENODE, **capture_args())
