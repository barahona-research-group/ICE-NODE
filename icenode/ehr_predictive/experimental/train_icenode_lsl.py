from functools import partial
from typing import (Any, Dict)

import jax.numpy as jnp

from .train_icenode_sl import ICENODE as ICENODE_SL


class ICENODE(ICENODE_SL):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

    def _f_update(self, params: Any, state_seq: Dict[int, jnp.ndarray],
                  emb: jnp.ndarray) -> jnp.ndarray:
        new_state = {}
        for i in emb:
            state, _ = self.split_state_emb_seq(state_seq[i])
            new_state[i] = self.join_state_emb(state[-1], emb[i])
        return new_state


if __name__ == '__main__':
    from .hpo_utils import capture_args, run_trials
    run_trials(model_cls=ICENODE, **capture_args())
