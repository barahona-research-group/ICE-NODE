from typing import (AbstractSet, Any, Callable, Dict, Iterable, List, Mapping,
                    Optional, Tuple, Union)

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import jax.random as jrandom
import jax.nn as jnn
import equinox as eqx

from ..metric.loss import mse

from ._eig_ad import eig

if len(jax.devices()) > 0:
    _flag_gpu_device = jax.devices()[0].platform == "gpu"
else:
    _flag_gpu_device = False


class SKELPhi(eqx.Module):
    """Koopman embeddings for continuous-time systems."""
    mlp: eqx.Module
    C: jnp.ndarray = eqx.static_field()
    skip: bool = eqx.static_field()

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 key: "jax.random.PRNGKey",
                 depth: int,
                 control_size: int = 0,
                 skip: bool = True):
        super().__init__()
        self.skip = skip
        self.C = jnp.eye(output_size, M=input_size + control_size)
        self.mlp = eqx.nn.MLP(
            input_size + control_size,
            output_size,
            depth=depth,
            width_size=(input_size + control_size + output_size) // 2,
            activation=jnn.relu,
            use_final_bias=not skip,
            key=key)

    @eqx.filter_jit
    def __call__(self, x, u=None):
        if u is not None:
            x = jnp.hstack((x, u))

        if self.skip:
            return self.C @ x + self.mlp(x)
        else:
            return self.mlp(x)


class VanillaKoopmanOperator(eqx.Module):
    A: jnp.ndarray
    phi: SKELPhi
    phi_inv: SKELPhi

    input_size: int = eqx.static_field()
    koopman_size: int = eqx.static_field()
    control_size: int = eqx.static_field()
    eigen_decomposition: bool = eqx.static_field()

    def __init__(self,
                 input_size: int,
                 koopman_size: int,
                 key: "jax.random.PRNGKey",
                 control_size: int = 0,
                 phi_depth: int = 1,
                 eigen_decomposition: bool = not _flag_gpu_device):
        super().__init__()
        self.input_size = input_size
        self.koopman_size = koopman_size
        self.control_size = control_size
        self.eigen_decomposition = eigen_decomposition
        keys = jrandom.split(key, 3)

        self.A = jrandom.normal(keys[0], (koopman_size, koopman_size),
                                dtype=jnp.float32)
        self.phi = SKELPhi(input_size,
                           koopman_size,
                           control_size=control_size,
                           depth=phi_depth,
                           skip=True,
                           key=keys[1])
        self.phi_inv = SKELPhi(koopman_size,
                               input_size,
                               depth=phi_depth,
                               skip=False,
                               key=keys[2])

    @eqx.filter_jit
    def compute_A(self):
        if self.eigen_decomposition:
            lam, V = eig(self.A)
            V_inv = jnp.linalg.solve(V @ jnp.diag(lam), self.A)
            return self.A, (lam, V, V_inv)

        return self.A

    def compute_K(self, t, A=None):
        if A is None:
            A = self.compute_A()
        if self.eigen_decomposition:
            _, (lam, V, V_inv) = A
            return (V @ jnp.diag(jnp.exp(lam * t)) @ V_inv).real
        else:
            return jscipy.linalg.expm(A * t, max_squarings=20)

    @eqx.filter_jit
    def __call__(self, t, x, u=None, A=None):
        z = self.phi(x, u=u)
        K = self.compute_K(t, A=A)
        z = K @ z
        return self.phi_inv(z)

    @eqx.filter_jit
    def compute_phi_loss(self, x, u=None):
        z = self.phi(x, u=u)
        return mse(x, self.phi_inv(z))

    def compute_A_spectrum(self):
        if self.eigen_decomposition:
            _, (lam, _, _) = self.compute_A()
        else:
            lam, _ = jnp.linalg.eig(self.compute_A())

        return lam.real, lam.imag


class SKELKoopmanOperator(VanillaKoopmanOperator):
    """Koopman operator for continuous-time systems."""
    R: jnp.ndarray
    Q: jnp.ndarray
    N: jnp.ndarray
    epsI: jnp.ndarray = eqx.static_field()

    def __init__(self,
                 input_size: int,
                 koopman_size: int,
                 key: "jax.random.PRNGKey",
                 control_size: int = 0,
                 phi_depth: int = 3,
                 eigen_decomposition: bool = not _flag_gpu_device):
        superkey, key = jrandom.split(key, 2)
        super().__init__(input_size=input_size,
                         koopman_size=koopman_size,
                         key=superkey,
                         control_size=control_size,
                         phi_depth=phi_depth,
                         eigen_decomposition=eigen_decomposition)
        self.A = None
        self.eigen_decomposition = eigen_decomposition
        keys = jrandom.split(key, 3)

        self.R = jrandom.normal(keys[0], (koopman_size, koopman_size),
                                dtype=jnp.float64)
        self.Q = jrandom.normal(keys[1], (koopman_size, koopman_size),
                                dtype=jnp.float64)
        self.N = jrandom.normal(keys[2], (koopman_size, koopman_size),
                                dtype=jnp.float64)
        self.epsI = 1e-9 * jnp.eye(koopman_size, dtype=jnp.float64)

        assert all(a.dtype == jnp.float64 for a in (self.R, self.Q, self.N)), \
            "SKELKoopmanOperator requires float64 precision"

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
            lam, V = eig(A)
            V_inv = jnp.linalg.solve(V @ jnp.diag(lam), A)
            return A, (lam, V, V_inv)

        return A
