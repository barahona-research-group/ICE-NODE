import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from lib.ml.base_models import ICNN


@pytest.mark.parametrize("input_size, hidden_size_multiplier, depth", [(2, 2, 2), (50, 2, 3), (100, 3, 2)])
def test_icnn_convexity(input_size, hidden_size_multiplier, depth):
    f_energy = ICNN(input_size, input_size * hidden_size_multiplier, depth, key=jrandom.PRNGKey(0))
    random_a = jrandom.normal(key=jrandom.PRNGKey(0), shape=(2 ** 16, input_size))
    random_b = jrandom.normal(key=jrandom.PRNGKey(1), shape=(2 ** 16, input_size))
    f_a = eqx.filter_vmap(f_energy)(random_a)
    f_b = eqx.filter_vmap(f_energy)(random_b)
    f_ab = eqx.filter_vmap(f_energy)((random_a + random_b) / 2)
    assert jnp.all(f_a + f_b >= 2 * f_ab)
