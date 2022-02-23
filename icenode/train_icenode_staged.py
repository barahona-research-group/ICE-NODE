from functools import partial
from typing import (Any, Callable, Dict, Iterable, List, Optional, Set)
from absl import logging

import jax
from jax.experimental import optimizers

import optuna

from .jax_interface import (DiagnosisJAXInterface)
from .train_icenode_tl import ICENODE as ICENODE_TL
from .gram import AbstractEmbeddingsLayer


class ICENODE_STAGED_MIXIN:

    @classmethod
    def step_optimizer(cls, step, model_state, batch):
        opt_state, opt_update, get_params, loss_, loss_mixing = model_state
        params = get_params(opt_state)

        if step < 50:
            grads = jax.grad(loss_)(params, batch)
        else:
            loss_mixing['L_dyn'] = 0
            grads = jax.grad(loss_)(params['f_n_ode'], batch)

        opt_state = opt_update(step, grads, opt_state)
        return opt_state, opt_update, get_params, loss_, loss_mixing


class ICENODE(ICENODE_STAGED_MIXIN, ICENODE_TL):

    def __init__(self, **kwargs):
        ICENODE_TL.__init__(self, **kwargs)


if __name__ == '__main__':
    from .hpo_utils import capture_args, run_trials
    run_trials(model_cls=ICENODE, **capture_args())
