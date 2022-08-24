"""."""

from typing import Optional

import haiku as hk
import jax.numpy as jnp
from jax.nn import leaky_relu


class StateIntervenedUpdate(hk.Module):
    """Implements discrete update based on the received observations."""

    def __init__(self, state_size: int, name: Optional[str] = None):
        super().__init__(name=name)
        self.__gates = []
        for label in ('dx_emb_hat', 'dx_emb', 'pr_emb'):
            self.__gates.append(
                hk.Sequential([
                    hk.Linear(state_size,
                              with_bias=True,
                              name=f'{name}_{label}'),
                    lambda x: leaky_relu(x, negative_slope=-0.2)
                ]))

        self.__merge = hk.Sequential(
            [hk.Linear(state_size, with_bias=True, name=f'merger'), jnp.tanh])

        self.__gru = hk.GRU(state_size)

    def __call__(self, state: jnp.ndarray, dx_emb_hat: jnp.ndarray,
                 dx_emb: jnp.ndarray, pr_emb: jnp.ndarray) -> jnp.ndarray:
        dx_emb_hat = self.__gates[0](dx_emb_hat)
        dx_emb = self.__gates[1](dx_emb)
        pr_emb = self.__gates[2](pr_emb)

        gru_input = self.__merge(jnp.hstack((dx_emb_hat, dx_emb, pr_emb)))
        _, updated_state = self.__gru(gru_input, state)
        return updated_state
