"""."""
from __future__ import annotations

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom


class GRUDynamics(eqx.Module):
    x_x: eqx.nn.Linear
    x_r: eqx.nn.Linear
    x_z: eqx.nn.Linear
    rx_g: eqx.nn.Linear

    def __init__(self, input_size: int, state_size: int, key: "jax.random.PRNGKey"):
        k0, k1, k2, k3 = jrandom.split(key, 4)
        self.x_x = eqx.nn.Linear(input_size, state_size, key=k0)
        self.x_r = eqx.nn.Linear(state_size, state_size, key=k1)
        self.x_z = eqx.nn.Linear(state_size, state_size, key=k2)
        self.rx_g = eqx.nn.Linear(state_size, state_size, key=k3)

    @eqx.filter_jit
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.x_x(x)
        r = jnn.sigmoid(self.x_r(x))
        z = jnn.sigmoid(self.x_z(x))
        g = jnn.tanh(self.rx_g(r * x))
        return (1 - z) * (g - x)
