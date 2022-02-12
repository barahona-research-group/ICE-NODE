from functools import partial
from typing import (Any, Callable, Dict, Iterable, List, Optional, Set)

import jax.numpy as jnp

import optuna

from .jax_interface import (DiagnosisJAXInterface)
from .train_icenode_sl import ICENODE as ICENODE_SL
from .train_icenode_2lr import ICENODE as ICENODE_2LR
from .gram import AbstractEmbeddingsLayer


class ICENODE(ICENODE_2LR, ICENODE_SL):

    def __init__(self, subject_interface: DiagnosisJAXInterface,
                 diag_emb: AbstractEmbeddingsLayer, ode_dyn: str,
                 ode_with_bias: bool, ode_init_var: float,
                 loss_half_life: float, state_size: int, timescale: float):
        ICENODE_SL.__init__(self,
                            subject_interface=subject_interface,
                            diag_emb=diag_emb,
                            ode_dyn=ode_dyn,
                            ode_with_bias=ode_with_bias,
                            ode_init_var=ode_init_var,
                            loss_half_life=loss_half_life,
                            state_size=state_size,
                            timescale=timescale)

    @classmethod
    def sample_training_config(cls, trial: optuna.Trial):
        return {
            'loss_half_life': trial.suggest_int('lt0.5', 7, 7 * 1e2, log=True),
            **ICENODE_2LR.sample_training_config(trial)
        }


if __name__ == '__main__':
    from .hpo_utils import capture_args, run_trials
    run_trials(model_cls=ICENODE, **capture_args())
