"""
This is a custom JVP rule for the eigendecomposition of a matrix. It is
necessary because the eigendecomposition is not differentiable in current
JAX library as of 8/20/2021.

This code is taken from:
    https://github.com/google/jax/issues/2748#issuecomment-1179511268
    by https://github.com/gboehl
"""

import jax.numpy as jnp
import jax.lax.linalg as lax_linalg
from jax import custom_jvp
from functools import partial

from jax import lax
from jax.numpy.linalg import solve


@custom_jvp
def eig(a):
    w, vl, vr = lax_linalg.eig(a)
    return w, vr


@eig.defjvp
def eig_jvp_rule(primals, tangents):
    a, = primals
    da, = tangents

    w, v = eig(a)

    eye = jnp.eye(a.shape[-1], dtype=a.dtype)
    # carefully build reciprocal delta-eigenvalue matrix, avoiding NaNs.
    Fmat = (
        jnp.reciprocal(eye + w[..., jnp.newaxis, :] - w[..., jnp.newaxis]) -
        eye)
    dot = partial(lax.dot if a.ndim == 2 else lax.batch_matmul,
                  precision=lax.Precision.HIGHEST)
    vinv_da_v = dot(solve(v, da), v)
    du = dot(v, jnp.multiply(Fmat, vinv_da_v))
    corrections = (jnp.conj(v) * du).sum(-2, keepdims=True)
    dv = du - v * corrections
    dw = jnp.diagonal(vinv_da_v, axis1=-2, axis2=-1)
    return (w, v), (dw, dv)
