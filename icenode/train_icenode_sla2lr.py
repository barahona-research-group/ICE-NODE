from functools import partial
from typing import (Any, Callable, Dict, Iterable, List, Optional, Set)

import jax
from jax.experimental import optimizers

import optuna

from .jax_interface import (DiagnosisJAXInterface)
from .train_icenode_sl2lr import ICENODE as ICENODE_SL2LR
from .gram import AbstractEmbeddingsLayer
from .utils import tree_map


class ICENODE(ICENODE_SL2LR):

    def __init__(self, subject_interface: DiagnosisJAXInterface,
                 diag_emb: AbstractEmbeddingsLayer, ode_dyn: str,
                 ode_with_bias: bool, ode_init_var: float,
                 loss_half_life: float, state_size: int, timescale: float):
        super().__init__(subject_interface=subject_interface,
                         diag_emb=diag_emb,
                         ode_dyn=ode_dyn,
                         ode_with_bias=ode_with_bias,
                         ode_init_var=ode_init_var,
                         loss_half_life=loss_half_life,
                         state_size=state_size,
                         timescale=timescale)

    def step_optimizer(self, step, model_state, batch):
        opt1, opt2, loss_, loss_mixing = model_state
        opt1_state, opt1_update, get_params1 = opt1
        opt2_state, opt2_update, get_params2 = opt2

        params = self.get_params(model_state)
        grads, detailed = jax.grad(loss_, has_aux=True)(params,
                                                        batch,
                                                        count_nfe=False,
                                                        interval_norm=False)

        grads1 = tree_map(lambda g: g / detailed['odeint_weeks'],
                          {'f_n_ode': grads['f_n_ode']})
        grads2 = tree_map(
            lambda g: g / detailed['admissions_count'], {
                'f_dec': grads['f_dec'],
                'f_update': grads['f_update'],
                'diag_emb': grads['diag_emb']
            })

        opt1_state = opt1_update(step, grads1, opt1_state)
        opt2_state = opt2_update(step, grads2, opt2_state)

        opt1 = (opt1_state, opt1_update, get_params1)
        opt2 = (opt2_state, opt2_update, get_params2)
        return opt1, opt2, loss_, loss_mixing


if __name__ == '__main__':
    from .hpo_utils import capture_args, run_trials
    run_trials(model_cls=ICENODE, **capture_args())
