from typing import Tuple, Optional, Literal, Union, Final, Callable, Any

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import optax
import optimistix as optx
from diffrax import diffeqsolve, Tsit5, RecursiveCheckpointAdjoint, SaveAt, ODETerm, Solution, PIDController, MultiTerm, \
    ControlTerm, VirtualBrownianTree, ReversibleHeun
from jax.experimental.jet import jet
from jaxtyping import PyTree

from .model import (Precomputes)

LeadPredictorName = Literal['monotonic', 'mlp']


class TaylorAugmented(eqx.Module):
    """
    https://github.com/jacobjinkelly/easy-neural-ode/blob/master/latent_ode.py
    By Jacob Kelly
    Recursively compute higher order derivatives of dynamics of ODE.
    MIT License

    Copyright (c) 2020 Jacob Kelly and Jesse Bettencourt

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """

    f: Callable
    order: int = 0

    def sol_recursive(self, t, x, args=None):
        if self.order < 2:
            return self.f(t, x, args), jnp.zeros_like(x)

        x_shape = x.shape

        def g(t_x):
            """
            Closure to expand z.
            """
            _t, _x = t_x[0], jnp.reshape(t_x[1:], x_shape)
            dx = jnp.ravel(self.f(_t, _x, args))
            dt = jnp.array([1.])
            dt_dx = jnp.concatenate((dt, dx))
            return dt_dx

        t_x = jnp.concatenate((jnp.array([t]), jnp.ravel(x)))

        (y0, [*yns]) = jet(g, (t_x,), ((jnp.ones_like(t_x),),))
        for _ in range(self.order - 1):
            (y0, [*yns]) = jet(g, (t_x,), ((y0, *yns),))

        return (jnp.reshape(y0[1:],
                            x_shape), jnp.reshape(yns[-2][1:], x_shape))

    def __call__(self, t, x_r: Tuple[jnp.ndarray, jnp.ndarray], args):
        x, _ = x_r

        dydt, drdt = self.sol_recursive(t, x, args)

        return dydt, jnp.mean(drdt ** 2)


class DiffusionMLP(eqx.nn.MLP):
    scale: jnp.ndarray
    brownian_size: int
    state_size: int

    def __init__(self, brownian_size: int, control_size: int, state_size: int, width_size: int, depth: int, *,
                 key: jrandom.PRNGKey, **kwargs):
        scale_key, mlp_key = jrandom.split(key)
        self.brownian_size = brownian_size
        self.state_size = state_size
        super().__init__(in_size=state_size + control_size, out_size=state_size * brownian_size,
                         activation=jnn.tanh,
                         width_size=width_size, depth=depth, key=mlp_key, **kwargs)
        self.scale = jrandom.uniform(
            scale_key, (state_size, brownian_size), minval=0.9, maxval=1.1
        )

    @eqx.filter_jit
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.scale * super().__call__(x).reshape(self.state_size, self.brownian_size)


class ForcedVectorField(eqx.Module):
    mlp: eqx.nn.MLP | DiffusionMLP

    @eqx.filter_jit
    def __call__(self, t: float, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return self.mlp(jnp.hstack((x, u)))


class NeuralODESolver(eqx.Module):
    f: ForcedVectorField
    SECOND: Final[float] = 1 / 3600.0  # Time units in one second.
    DT0: Final[float] = 60.0  # Initial time step in seconds.

    @staticmethod
    def from_mlp(mlp: eqx.nn.MLP, second: float = 1 / 3600.0, dt0: float = 60.0):
        return NeuralODESolver(f=ForcedVectorField(mlp), SECOND=second, DT0=dt0)

    @property
    def zero_force(self) -> jnp.ndarray:
        in_size = self.f.mlp.layers[0].weight.shape[1]
        out_size = self.f.mlp.layers[-1].weight.shape[0]
        return jnp.zeros((in_size - out_size,))

    @property
    def ode_term(self) -> ODETerm:
        return ODETerm(self.f)

    @eqx.filter_jit
    def __call__(self, x0, t0: float, t1: float, saveat: Optional[SaveAt] = None,
                 u: Optional[PyTree] = None,
                 precomputes: Optional[Precomputes] = None,
                 key: Optional[jrandom.PRNGKey] = None) -> Union[jnp.ndarray, Tuple[jnp.ndarray, ...]]:
        sol = diffeqsolve(
            terms=self.ode_term,
            solver=Tsit5(),
            t0=t0,
            t1=t1,
            dt0=self.DT0 * self.SECOND,
            y0=self.get_aug_x0(x0, precomputes),
            args=self.get_args(x0, u, precomputes),
            adjoint=RecursiveCheckpointAdjoint(checkpoints=10),
            # TODO: investigate the difference between checkpoints, max_steps, and BacksolveAdjoint.
            saveat=saveat or SaveAt(t1=True),
            stepsize_controller=PIDController(rtol=1.4e-8, atol=1.4e-8),
            throw=True,
            max_steps=None)
        return self.get_solution(sol)

    def get_args(self, x0: jnp.ndarray, u: Optional[jnp.ndarray], precomputes: Optional[Precomputes]) -> PyTree:
        return u if u is not None else self.zero_force

    def get_aug_x0(self, x0: jnp.ndarray, precomputes: Precomputes) -> PyTree:
        return x0

    def get_solution(self, sol: Solution) -> Union[jnp.ndarray, Tuple[jnp.ndarray, ...]]:
        return sol.ys


class StochasticNeuralODESolver(eqx.Module):
    f_drift: ForcedVectorField
    f_diffusion: ForcedVectorField
    SECOND: Final[float] = 1 / 3600.0  # Time units in one second.
    DT0: Final[float] = 60.0  # Initial time step in seconds.

    @staticmethod
    def from_mlp(drift: eqx.Module, diffusion: DiffusionMLP, second: float = 1 / 3600.0, dt0: float = 60.0):
        return StochasticNeuralODESolver(f_drift=ForcedVectorField(drift), f_diffusion=ForcedVectorField(diffusion),
                                         SECOND=second, DT0=dt0)

    @property
    def zero_force(self) -> jnp.ndarray:
        in_size = self.f_drift.mlp.layers[0].weight.shape[1]
        out_size = self.f_drift.mlp.layers[-1].weight.shape[0]
        return jnp.zeros((in_size - out_size,))

    def stochastic_ode_terms(self, t0: float, t1: float, key: jrandom.PRNGKey) -> MultiTerm:
        control = VirtualBrownianTree(
            t0=t0, t1=t1, tol=self.DT0 * self.SECOND / 2, shape=(self.f_diffusion.mlp.brownian_size,), key=key
        )
        return MultiTerm(ODETerm(self.f_drift), ControlTerm(self.f_diffusion, control))

    @eqx.filter_jit
    def __call__(self, x0, t0: float, t1: float, saveat: Optional[SaveAt] = None,
                 u: Optional[PyTree] = None,
                 precomputes: Optional[Precomputes] = None,
                 key: Optional[jrandom.PRNGKey] = None) -> Union[jnp.ndarray, Tuple[jnp.ndarray, ...]]:
        sol = diffeqsolve(
            terms=self.stochastic_ode_terms(t0, t1, key),
            solver=ReversibleHeun(),
            t0=t0,
            t1=t1,
            dt0=self.DT0 * self.SECOND,
            y0=self.get_aug_x0(x0, precomputes),
            args=self.get_args(x0, u, precomputes),
            adjoint=RecursiveCheckpointAdjoint(checkpoints=10),
            saveat=saveat or SaveAt(t1=True),
            stepsize_controller=PIDController(rtol=1.4e-8, atol=1.4e-8),
            throw=True,
            max_steps=None)
        return self.get_solution(sol)

    def get_args(self, x0: jnp.ndarray, u: Optional[jnp.ndarray], precomputes: Optional[Precomputes]) -> PyTree:
        return u if u is not None else self.zero_force

    def get_aug_x0(self, x0: jnp.ndarray, precomputes: Precomputes) -> PyTree:
        return x0

    def get_solution(self, sol: Solution) -> Union[jnp.ndarray, Tuple[jnp.ndarray, ...]]:
        return sol.ys


class SkipShortIntervalsWrapper(eqx.Module):
    solver: StochasticNeuralODESolver | NeuralODESolver
    min_interval: float = 1.0  # seconds

    def __call__(self, x0, t0: float, t1: float, saveat: Optional[SaveAt] = None,
                 u: Optional[PyTree] = None,
                 precomputes: Optional[Precomputes] = None,
                 key: Optional[jrandom.PRNGKey] = None) -> Union[jnp.ndarray, Tuple[jnp.ndarray, ...]]:
        if t1 - t0 < self.min_interval * self.solver.SECOND:
            return x0
        return self.solver(x0, t0, t1, saveat, u, precomputes, key)


class MonotonicLeadingObsPredictor(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, input_size: int,
                 n_lead_times: int,
                 key: jrandom.PRNGKey, **mlp_kwargs):
        super().__init__()
        out_size = n_lead_times + 1
        width = mlp_kwargs.get("width_size", out_size * 5)
        self.mlp = eqx.nn.MLP(input_size,
                              out_size,
                              width_size=width,
                              activation=jnn.tanh,
                              depth=2,
                              key=key)

    @eqx.filter_jit
    def __call__(self, state):
        y = self.mlp(state)
        risk = jnn.sigmoid(y[-1])
        p = jnp.cumsum(jnn.softmax(y[:-1]))
        return risk * p


class MLPLeadingObsPredictor(eqx.Module):
    _mlp: eqx.nn.MLP

    def __init__(self, input_size: int,
                 n_lead_times: int,
                 key: jax.random.PRNGKey, **mlp_kwargs):
        super().__init__()
        width = mlp_kwargs.get("width_size", n_lead_times * 5)
        self._mlp = eqx.nn.MLP(input_size,
                               n_lead_times,
                               width_size=width,
                               activation=jnn.tanh,
                               final_activation=jnn.sigmoid,
                               depth=2,
                               key=key)

    @eqx.filter_jit
    def __call__(self, state):
        return self._mlp(state)


class CompiledMLP(eqx.nn.MLP):
    # Just an eqx.nn.MLP with a compiled __call__ method.

    @eqx.filter_jit
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return super().__call__(x)


class CompiledLinear(eqx.nn.Linear):

    @eqx.filter_jit
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return super().__call__(x)


class ProbMLP(CompiledMLP):

    @eqx.filter_jit
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        mean_std = super().__call__(x)
        mean, std = jnp.split(mean_std, 2, axis=-1)
        return mean, jnn.softplus(std)


class CompiledGRU(eqx.nn.GRUCell):
    @eqx.filter_jit
    def __call__(self, x: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
        return super().__call__(h, x)


class PositiveSquaredLinear(eqx.nn.Linear):
    def __call__(self, x: jnp.ndarray, *, key: Optional[jrandom.PRNGKey] = None) -> jnp.ndarray:
        w = self.weight ** 2
        y = w @ x
        if self.bias is not None:
            y += self.bias
        return y


class ICNN(eqx.Module):
    """Input Convex Neural Network"""
    """https://github.com/atong01/ot-icnn-minimal/blob/main/icnn/icnn.py
    Principled Weight Initialisation for Input-Convex Neural Networks: https://openreview.net/pdf?id=pWZ97hUQtQ 
    """
    Wzs: Tuple[PositiveSquaredLinear, ...]
    Wxs: Tuple[PositiveSquaredLinear, ...]

    def __init__(self, input_size: int, hidden_size: int, depth: int, key: jrandom.PRNGKey):
        super().__init__()

        def new_key():
            nonlocal key
            key, subkey = jrandom.split(key)
            return subkey

        Wzs = [PositiveSquaredLinear(input_size, hidden_size, key=new_key())]
        for _ in range(depth - 1):
            Wzs.append(PositiveSquaredLinear(hidden_size, hidden_size, use_bias=False, key=new_key()))
        Wzs.append(PositiveSquaredLinear(hidden_size, 1, use_bias=False, key=new_key()))
        self.Wzs = tuple(Wzs)

        Wxs = []
        for _ in range(depth - 1):
            Wxs.append(PositiveSquaredLinear(input_size, hidden_size, key=new_key()))
        Wxs.append(PositiveSquaredLinear(input_size, 1, use_bias=False, key=new_key()))
        self.Wxs = tuple(Wxs)

    @eqx.filter_jit
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray | float:
        z = jnn.softplus(self.Wzs[0](x))
        for Wz, Wx in zip(self.Wzs[1:-1], self.Wxs[:-1]):
            z = jnn.softplus(Wz(z) + Wx(x))
        return (self.Wzs[-1](z) + self.Wxs[-1](x)).squeeze()


# def test_convexity(f):
#     rdata = torch.randn(1024, 2).to(device)
#     rdata2 = torch.randn(1024, 2).to(device)
#     return np.all(
#         (((f(rdata) + f(rdata2)) / 2 - f(rdata + rdata2) / 2) > 0)
#         .cpu()
#         .detach()
#         .numpy()
#     )
#

class MaskedOptimiser(eqx.Module):
    optim: optax.GradientTransformation
    mask: jnp.ndarray = eqx.static_field()

    def __init__(self, optim: optax.GradientTransformation, mask: jnp.ndarray):
        self.optim = optim
        self.mask = mask

    def update(
            self,
            updates: optax.Updates,
            state: optax.OptState,
            params: Optional[optax.Params] = None
    ) -> tuple[optax.Updates, optax.OptState]:
        return self.optim.update(jnp.where(self.mask, updates, 0.0), state, params)

    def init(self, params: optax.Params) -> optax.OptState:
        return self.optim.init(params)


class ICNNObsDecoder(eqx.Module):
    f_energy: ICNN
    observables_size: int
    state_size: int

    def __init__(self, observables_size: int, state_size: int, hidden_size_multiplier: int,
                 depth: int, key: jrandom.PRNGKey):
        super().__init__()
        self.observables_size = observables_size
        self.state_size = state_size
        input_size = observables_size + state_size
        self.f_energy = ICNN(input_size, input_size * hidden_size_multiplier, depth, key)

    @eqx.filter_jit
    def partial_input_optimise(self, input: jnp.ndarray, fixed_mask: jnp.ndarray):
        def masked_input_energy(y: jnp.ndarray, args: Any = None) -> jnp.ndarray:
            return self.f_energy(jnp.where(fixed_mask, input, y))

        return optx.minimise(masked_input_energy,
                             adjoint=optx.RecursiveCheckpointAdjoint(checkpoints=20),
                             solver=optx.BestSoFarMinimiser(
                                 optx.OptaxMinimiser(optim=MaskedOptimiser(optax.adam(5e-4), fixed_mask),
                                                     rtol=1e-8, atol=1e-8)),
                             max_steps=2 ** 16,
                             y0=input, throw=False).value

    @eqx.filter_jit
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray | float:
        input = jnp.hstack((state, jnp.zeros(self.observables_size)))
        mask = jnp.zeros_like(input).at[:self.state_size].set(1)
        output = self.partial_input_optimise(input, mask)
        _, obs = jnp.split(output, [self.state_size])
        return obs


class ICNNObsExtractor(ICNNObsDecoder):

    @eqx.filter_jit
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.split(x, [self.state_size])[1]
