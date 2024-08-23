import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from lib.ml.icnn_modules import ICNN


@pytest.mark.parametrize("input_size, hidden_size_multiplier, depth, positivity",
                         [(2, 2, 5, 'none'), (2, 2, 5, 'abs'), (50, 2, 5, 'squared'), (50, 2, 5, 'none'),
                          (100, 2, 5, 'abs')])
def test_icnn_convexity(input_size, hidden_size_multiplier, depth, positivity):
    f_energy = ICNN(input_size, input_size * hidden_size_multiplier, depth, positivity=positivity,
                    key=jrandom.PRNGKey(0))
    random_a = jrandom.normal(key=jrandom.PRNGKey(0), shape=(2 ** 17, input_size), dtype=jnp.float64)
    random_b = jrandom.normal(key=jrandom.PRNGKey(1), shape=(2 ** 17, input_size), dtype=jnp.float64)
    f_a = eqx.filter_vmap(f_energy)(random_a)
    f_b = eqx.filter_vmap(f_energy)(random_b)
    f_ab = eqx.filter_vmap(f_energy)((random_a + random_b) / 2)
    assert jnp.all(f_a + f_b >= 2 * f_ab)
