from typing import NamedTuple

import chex
import jax.numpy as jnp
from jax import tree_util as jtu
from optax._src import transform, combine, base, numerics
from optax.tree_utils import tree_zeros_like, tree_update_moment, tree_bias_correction, tree_update_moment_per_elem_norm


class ScaleByAdemamixState(NamedTuple):
    count: chex.Array
    count_m2: chex.Array
    m1: base.Updates
    m2: base.Updates
    nu: base.Updates


def ademamix(lr,
             b1=0.9,
             b2=0.999,
             b3=0.9999,
             alpha=5.0,
             b3_scheduler=None,
             alpha_scheduler=None,
             eps=1e-8,
             weight_decay=0.0):
    return combine.chain(
        scale_by_ademamix(b1, b2, b3, alpha, b3_scheduler, alpha_scheduler, eps),
        transform.add_decayed_weights(weight_decay),
        transform.scale_by_learning_rate(lr),
    )


def scale_by_ademamix(b1,
                      b2,
                      b3,
                      alpha,
                      b3_scheduler,
                      alpha_scheduler,
                      eps):
    def init_fn(params):
        m1 = tree_zeros_like(params)  # fast EMA
        m2 = tree_zeros_like(params)  # slow EMA
        nu = tree_zeros_like(params)  # second moment estimate
        return ScaleByAdemamixState(
            count=jnp.zeros([], jnp.int32),
            count_m2=jnp.zeros([], jnp.int32),
            m1=m1,
            m2=m2,
            nu=nu
        )

    def update_fn(updates, state, params=None):
        del params
        c_b3 = b3_scheduler(state.count_m2) if b3_scheduler is not None else b3
        c_alpha = alpha_scheduler(state.count_m2) if alpha_scheduler is not None else alpha
        m1 = tree_update_moment(updates, state.m1, b1, 1)  # m1 = b1 * m1 + (1-b1) * updates
        m2 = tree_update_moment(updates, state.m2, c_b3, 1)
        nu = tree_update_moment_per_elem_norm(updates, state.nu, b2, 2)
        count_inc = numerics.safe_int32_increment(state.count)
        count_m2_inc = numerics.safe_int32_increment(state.count_m2)
        m1_hat = tree_bias_correction(m1, b1, count_inc)
        nu_hat = tree_bias_correction(nu, b2, count_inc)
        updates = jtu.tree_map(
            lambda m1_, m2_, v_: (m1_ + c_alpha * m2_) / (jnp.sqrt(v_) + eps),
            m1_hat,
            m2,
            nu_hat
        )
        return updates, ScaleByAdemamixState(
            count=count_inc,
            count_m2=count_m2_inc,
            m1=m1,
            m2=m2,
            nu=nu
        )

    return base.GradientTransformation(init_fn, update_fn)
