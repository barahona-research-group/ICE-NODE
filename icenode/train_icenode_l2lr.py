from functools import partial
from typing import (Any, Callable, Dict, Iterable, List, Optional, Set)

from absl import logging
import jax
import jax.numpy as jnp

import optuna

from .jax_interface import (DiagnosisJAXInterface)
from .gram import AbstractEmbeddingsLayer
from .train_icenode_2lr import ICENODE as ICENODE_2LR


class ICENODE(ICENODE_2LR):

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

        del (self.initializers['f_update'], self.f_update)

    def _initialization_data(self):
        """
        Creates data for initializing each of the
        modules based on the shapes of init_data.
        """
        emb = jnp.zeros(self.dimensions['diag_emb'])
        state = jnp.zeros(self.dimensions['state'])
        state_emb = jnp.hstack((state, emb))
        return {
            "f_n_ode": [2, True, state_emb, 0.1],
            "f_dec": [emb],
        }

    def _f_update(self, params: Any, state_e: Dict[int, jnp.ndarray],
                  emb: jnp.ndarray) -> jnp.ndarray:
        new_state = {}
        for i in emb:
            state, _ = self.split_state_emb(state_e[i])
            new_state[i] = self.join_state_emb(state, emb[i])
        return new_state


if __name__ == '__main__':
    from .hpo_utils import capture_args, run_trials
    run_trials(model_cls=ICENODE, **capture_args())
