from typing import Tuple, Optional, Literal

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr

import jax.numpy as jnp
import jax.lax.linalg as lax_linalg
from jax import custom_jvp
from functools import partial

from jax import lax
from jax.numpy.linalg import solve

from diffrax import SaveAt
from jaxtyping import PyTree

from lib.ml.base_models import ODEMetrics
from lib.ml.icnn_modules import ICNNObsDecoder, ImputerMetrics


@custom_jvp
def eig(a):
    """
    This is a custom JVP rule for the eigendecomposition of a matrix. It is
    necessary because the eigendecomposition is not differentiable in current
    JAX library as of 8/20/2021.

    This code is taken from:
        https://github.com/google/jax/issues/2748#issuecomment-1179511268
        by https://github.com/gboehl
    """
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


class KoopmanPrecomputes(eqx.Module):
    A: jnp.ndarray
    A_eig: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]


class KoopmanPhi(eqx.Module):
    @eqx.filter_jit
    def encode(self, x: jnp.ndarray, u: Optional[jnp.ndarray] = None,
               x_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        pass

    @eqx.filter_jit
    def decode(self, z: jnp.ndarray, u: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        pass


class KoopmanPhiMLP(KoopmanPhi):
    """Koopman embeddings for continuous-time systems."""
    mlp: eqx.Module
    mlp_inv: eqx.Module
    C: jnp.ndarray = eqx.static_field()
    C_inv: jnp.ndarray = eqx.static_field()
    skip: bool = eqx.static_field()

    def __init__(self,
                 observables_size: int,
                 embeddings_size: int,
                 depth: int,
                 control_size: int = 0,
                 skip: bool = True, *,
                 key: "jax.random.PRNGKey"):
        super().__init__()
        self.skip = skip
        self.C = jnp.eye(embeddings_size, M=observables_size + control_size)
        self.C_inv = jnp.eye(observables_size, M=embeddings_size)
        self.mlp = eqx.nn.MLP(
            observables_size + control_size,
            embeddings_size,
            depth=depth,
            width_size=(observables_size + control_size + embeddings_size) // 2,
            activation=jnn.relu,
            use_final_bias=not skip,
            key=key)
        self.mlp_inv = eqx.nn.MLP(
            embeddings_size,
            observables_size,
            depth=depth,
            width_size=(observables_size + control_size + embeddings_size) // 2,
            activation=jnn.relu,
            use_final_bias=not skip,
            key=key)

    @eqx.filter_jit
    def encode(self, x: jnp.ndarray, u: Optional[jnp.ndarray] = None) -> jnp.ndarray:

        if u is not None:
            x = jnp.hstack((x, u))

        if self.skip:
            return self.C @ x + self.mlp(x)
        else:
            return self.mlp(x)

    @eqx.filter_jit
    def decode(self, z: jnp.ndarray, u: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        if self.skip:
            return self.mlp_inv(z) + self.C_inv @ z
        else:
            return self.mlp_inv(z)


class KoopmanPhiICNN(KoopmanPhi):
    f_icnn: ICNNObsDecoder
    z_size: int = eqx.static_field()  # Embeddings
    x_size: int = eqx.static_field()  # Observables
    u_size: int = eqx.static_field()  # Control

    def __init__(self, x_size: int, z_size: int, u_size: int,
                 key: jr.PRNGKey,
                 depth: int = 4,
                 hidden_size_multiplier: float = 1.0,
                 optax_optimiser_name: Literal['adam', 'polyak_sgd', 'lamb', 'yogi'] = 'adam'):
        # Input Components: (z "embeddings", x "observables", u "control")
        super().__init__()
        self.f_icnn = ICNNObsDecoder(x_size + z_size + u_size, 0,
                                     hidden_size_multiplier, depth,
                                     optax_optimiser_name=optax_optimiser_name, key=key)
        self.u_size = u_size
        self.x_size = x_size
        self.z_size = z_size

    @eqx.filter_jit
    def encode(self, x: jnp.ndarray, u: Optional[jnp.ndarray] = None) -> jnp.ndarray:

        # fix the mask for the control input.
        if u is None:
            u = jnp.zeros(self.u_size)

        z = jnp.zeros(self.z_size)
        icnn_input = jnp.hstack((z, x, u))
        mask = jnp.hstack((z, jnp.ones_like(x), jnp.ones_like(u)))
        icnn_output, _ = self.f_icnn.partial_input_optimise(icnn_input, mask)
        z, _, _ = self.split(icnn_output)
        return z

    @eqx.filter_jit
    def decode(self, z: jnp.ndarray, u: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        if u is None:
            u = jnp.zeros(self.u_size)

        icnn_input = jnp.hstack((z, jnp.zeros(self.x_size), u))
        icnn_mask = jnp.hstack((jnp.ones(self.z_size), jnp.zeros(self.x_size), jnp.ones(self.u_size)))
        icnn_input, _ = self.f_icnn.partial_input_optimise(icnn_input, icnn_mask)
        _, x, _ = self.split(icnn_input)
        return x

    @eqx.filter_jit
    def autoimpute_x(self, x: jnp.ndarray, x_mask: jnp.ndarray, u: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        if u is None:
            u = jnp.zeros(self.u_size)

        icnn_input = jnp.hstack((jnp.zeros(self.z_size), x, u))
        icnn_mask = jnp.hstack((jnp.zeros(self.z_size), x_mask, jnp.ones(self.u_size)))
        icnn_input, _ = self.f_icnn.partial_input_optimise(icnn_input, icnn_mask)
        _, x, _ = self.split(icnn_input)
        return x

    def split(self, icnn_input: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        z, x, u = jnp.split(icnn_input, [self.z_size, self.z_size + self.x_size])
        return z, x, u


class KoopmanIVPGenerator(eqx.Module):
    persistence_ratio: float = eqx.static_field(default=0.0)

    @eqx.filter_jit
    def __call__(self,
                 k_icnn: KoopmanPhiICNN,
                 forecasted_state: jnp.ndarray,
                 true_observables: jnp.ndarray,
                 observables_mask: jnp.ndarray,
                 u: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, ImputerMetrics]:
        # init_obs: obs_decoder(forecasted_state).
        # input: (persistent_hidden_confounder, hidden_confounder, init_obs).
        # mask: (ones_like(state_mem), zeros_like(state_hidden).
        unobserved_size = len(forecasted_state) - len(true_observables)
        persistent_memory_size = int(unobserved_size * self.persistence_ratio)
        x = jnp.hstack((forecasted_state[:unobserved_size], true_observables))
        unobserved_mask = jnp.zeros_like(forecasted_state).at[:persistent_memory_size].set(1)
        x_mask = jnp.hstack((unobserved_mask, observables_mask))
        return k_icnn.autoimpute_x(x, x_mask, u), ImputerMetrics(n_steps=jnp.array([]))


class VanillaKoopmanOperator(eqx.Module):
    A: jnp.ndarray
    phi: KoopmanPhiMLP

    input_size: int = eqx.static_field()
    koopman_size: int = eqx.static_field()
    control_size: int = eqx.static_field()
    phi_depth: int = eqx.static_field()

    def __init__(self,
                 input_size: int,
                 koopman_size: int,
                 control_size: int = 0,
                 phi_depth: int = 1, *,
                 key: "jax.random.PRNGKey"):
        super().__init__()
        self.input_size = input_size
        self.koopman_size = koopman_size
        self.control_size = control_size
        self.phi_depth = phi_depth
        key1, key2 = jr.split(key, 2)

        self.A = jr.normal(key1, (koopman_size, koopman_size),
                           dtype=jnp.float32)
        self.phi = self.make_phi(key2)

    def make_phi(self, key: jr.PRNGKey) -> KoopmanPhiMLP:
        return KoopmanPhiMLP(self.input_size, self.koopman_size, control_size=self.control_size,
                             depth=self.phi_depth, skip=True, key=key)

    @eqx.filter_jit
    def compute_A(self) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        lam, V = eig(self.A)
        V_inv = jnp.linalg.solve(V @ jnp.diag(lam), self.A)
        return self.A, (lam, V, V_inv)

    def K_operator(self, t: float, z: jnp.ndarray,
                   A_eig: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        assert z.ndim == 1, f"z must be a vector, got {z.ndim}D"
        (lam, V, V_inv) = A_eig
        lam_t = jnp.exp(lam * t)
        complex_z = V @ (lam_t * (V_inv @ z))
        return complex_z.real

    @eqx.filter_jit
    def __call__(self, x0, t0: float, t1: float,
                 u: Optional[PyTree],
                 precomputes: KoopmanPrecomputes, saveat: Optional[SaveAt] = None,
                 key: Optional[jr.PRNGKey] = None) -> Tuple[jnp.ndarray, ODEMetrics]:
        z = self.phi.encode(x0, u=u)
        z = self.K_operator(t1, z, precomputes.A_eig)
        return self.phi.decode(z, u=u), ODEMetrics()

    @eqx.filter_jit
    def compute_phi_loss(self, x: jnp.ndarray, u: Optional[jnp.ndarray] = None):
        z = self.phi.encode(x, u=u)
        diff = x - self.phi.decode(z, u=u)
        return jnp.mean(diff ** 2)

    def compute_A_spectrum(self):
        _, (lam, _, _) = self.compute_A()
        return lam.real, lam.imag


class KoopmanOperator(VanillaKoopmanOperator):
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
                 phi_depth: int = 3):
        superkey, key = jr.split(key, 2)
        super().__init__(input_size=input_size,
                         koopman_size=koopman_size,
                         key=superkey,
                         control_size=control_size,
                         phi_depth=phi_depth)
        self.A = None
        keys = jr.split(key, 3)

        self.R = jr.normal(keys[0], (koopman_size, koopman_size),
                           dtype=jnp.float64)
        self.Q = jr.normal(keys[1], (koopman_size, koopman_size),
                           dtype=jnp.float64)
        self.N = jr.normal(keys[2], (koopman_size, koopman_size),
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

        lam, V = eig(A)
        V_inv = jnp.linalg.solve(V @ jnp.diag(lam), A)
        return A, (lam, V, V_inv)


class ICNNKoopmanOperator(KoopmanOperator):
    phi: KoopmanPhiICNN

    def __init__(self, input_size: int, koopman_size: int, key: jr.PRNGKey,
                 control_size: int = 0, phi_depth: int = 3):
        super().__init__(input_size, koopman_size, key, control_size, phi_depth)

    def make_phi(self, key: jr.PRNGKey) -> KoopmanPhiICNN:
        return KoopmanPhiICNN(x_size=self.input_size, z_size=self.koopman_size,
                              u_size=self.control_size, depth=self.phi_depth, key=key)
