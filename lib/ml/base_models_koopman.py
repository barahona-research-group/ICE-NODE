from typing import (AbstractSet, Any, Callable, Dict, Iterable, List, Mapping,
                    Optional, Tuple, Union)

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import jax.random as jrandom
import jax.nn as jnn
import equinox as eqx


class SKELKoopmanOperator(eqx.Module):
    """Koopman operator for continuous-time systems."""
    R: jnp.ndarray
    Q: jnp.ndarray
    N: jnp.ndarray
    epsI: jnp.ndarray
    dim: int = eqx.static_field()
    dim_K: int = eqx.static_field()
    eigen_decomposition: bool = eqx.static_field()

    def __init__(self,
                 input_dim,
                 dim_K,
                 key: "jax.random.PRNGKey",
                 eigen_decomposition=True):
        super().__init__()
        self.dim = input_dim
        self.dim_K = dim_K
        self.eigen_decomposition = eigen_decomposition
        keys = jrandom.split(key, 3)

        self.R = jrandom.normal(keys[0], (dim_K, dim_K))
        self.Q = jrandom.normal(keys[1], (dim_K, dim_K))
        self.N = jrandom.normal(keys[2], (dim_K, dim_K))
        self.epsI = 1e-8 * jnp.eye(dim_K)

    @eqx.filter_jit
    def compute_A(self):
        R = self.R
        Q = self.Q
        N = self.N

        skew = (R - R.T) / 2
        F = skew - Q @ Q.T - self.epsI
        E = N @ N.T + self.epsI

        A = jnp.linalg.solve(E, F)

        if self.eigen_decomposition:
            lam, V = jnp.linalg.eig(A)
            return A, (lam, V)

        return A

    def compute_K(self, t, A=None):
        if A is None:
            A = self.compute_A()
        if self.eigen_decomposition:
            _, (lam, V) = A
            return V @ jnp.diag(jnp.exp(lam * t)) @ jnp.linalg.inv(V)
        else:
            return jscipy.linalg.expm(A * t, max_squarings=32)

    @eqx.filter_jit
    def __call__(self, t, x, A=None):
        K = self.compute_K(t, A=A)
        return K @ x
