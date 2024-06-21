from typing import Tuple, Optional, Literal, Union, Final, Callable, Self, cast

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import optax
import optimistix as optx
from diffrax import diffeqsolve, Tsit5, RecursiveCheckpointAdjoint, SaveAt, ODETerm, PIDController, MultiTerm, \
    ControlTerm, VirtualBrownianTree, ReversibleHeun
from jax.experimental.jet import jet
from jaxtyping import PyTree

from .model import (Precomputes)
from .. import VxData

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


class ODEMetrics(VxData):
    n_steps: jnp.ndarray = eqx.field(default_factory=lambda: jnp.array([]))
    n_hours: jnp.ndarray = eqx.field(default_factory=lambda: jnp.array([]))

    @property
    def n_solutions(self):
        return len(self.n_steps)

    @property
    def n_steps_per_solution(self):
        return sum(self.n_steps) / self.n_solutions

    @property
    def n_steps_per_hour(self):
        return sum(self.n_steps) / sum(self.n_hours)

    def __add__(self, other: Self) -> Self:
        return ODEMetrics(n_steps=jnp.hstack((self.n_steps, other.n_steps)),
                          n_hours=jnp.hstack((self.n_hours, other.n_hours)))


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
                 key: Optional[jrandom.PRNGKey] = None) -> Tuple[
        Union[jnp.ndarray, Tuple[jnp.ndarray, ...]], ODEMetrics]:
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
        return sol.ys, ODEMetrics(n_steps=jnp.array(sol.stats['num_steps']), n_hours=jnp.array(t1 - t0))

    def get_args(self, x0: jnp.ndarray, u: Optional[jnp.ndarray], precomputes: Optional[Precomputes]) -> PyTree:
        return u if u is not None else self.zero_force

    def get_aug_x0(self, x0: jnp.ndarray, precomputes: Precomputes) -> PyTree:
        return x0


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
                 key: Optional[jrandom.PRNGKey] = None) -> Tuple[
        Union[jnp.ndarray, Tuple[jnp.ndarray, ...]], ODEMetrics]:
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
        return sol.ys, ODEMetrics(n_steps=jnp.array(sol.stats['num_steps']), n_hours=jnp.array(t1 - t0))

    def get_args(self, x0: jnp.ndarray, u: Optional[jnp.ndarray], precomputes: Optional[Precomputes]) -> PyTree:
        return u if u is not None else self.zero_force

    def get_aug_x0(self, x0: jnp.ndarray, precomputes: Precomputes) -> PyTree:
        return x0


class SkipShortIntervalsWrapper(eqx.Module):
    solver: StochasticNeuralODESolver | NeuralODESolver
    min_interval: float = 1.0  # seconds

    def __call__(self, x0, t0: float, t1: float, saveat: Optional[SaveAt] = None,
                 u: Optional[PyTree] = None,
                 precomputes: Optional[Precomputes] = None,
                 key: Optional[jrandom.PRNGKey] = None) -> Tuple[
        Union[jnp.ndarray, Tuple[jnp.ndarray, ...]], ODEMetrics]:
        if t1 - t0 < self.min_interval * self.solver.SECOND:
            return x0, ODEMetrics()
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
        # w = jnn.relu(self.weight)
        y = w @ x
        if self.bias is not None:
            y += self.bias
        return y


class PositiveReLuLinear(eqx.nn.Linear):
    def __call__(self, x: jnp.ndarray, *, key: Optional[jrandom.PRNGKey] = None) -> jnp.ndarray:
        w = jnn.relu(self.weight)
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
    activations: Tuple[Callable[..., jnp.ndarray], ...]

    def __init__(self, input_size: int, hidden_size: int, depth: int, key: jrandom.PRNGKey):
        super().__init__()
        positive_layer_i = 0
        activation_i = 0

        def new_key():
            nonlocal key
            key, subkey = jrandom.split(key)
            return subkey

        def positive_layer(*args, **kwargs):
            nonlocal positive_layer_i
            positive_layer_i += 1
            if positive_layer_i % 2 == 0:
                return PositiveSquaredLinear(*args, **kwargs)
            return PositiveReLuLinear(*args, **kwargs)

        def activation():
            nonlocal activation_i
            activation_i += 1
            if activation_i == 0:
                return jnn.sigmoid
            if activation_i % 2 == 0:
                return jnn.softplus
            return jnn.relu

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
        self.activations = tuple(jnn.softplus for _ in range(depth))

    @eqx.filter_jit
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray | float:
        z = jnn.softplus(self.Wzs[0](x))
        for Wz, Wx, sigma in zip(self.Wzs[1:-1], self.Wxs[:-1], self.activations):
            z = jnn.softplus(Wz(z) + Wx(x))
        return (self.Wzs[-1](z) + self.Wxs[-1](x)).squeeze()


class MaskedOptaxMinimiser(optx.OptaxMinimiser):

    def step(
            self,
            fn,
            y,
            args,
            options,
            state,
            tags,
    ):
        (f, aux), grads = eqx.filter_value_and_grad(fn, has_aux=True)(y, args)
        grads = jnp.where(options['fixed_mask'], 0.0, grads)
        f = cast(jax.Array, f)
        if len(self.verbose) > 0:
            optx._misc.verbose_print(
                ("step" in self.verbose, "Step", state.step),
                ("loss" in self.verbose, "Loss", f),
                ("y" in self.verbose, "y", y),
            )
        updates, new_opt_state = self.optim.update(grads, state.opt_state, params=y)
        new_y = eqx.apply_updates(y, updates)
        terminate = optx._misc.cauchy_termination(
            self.rtol,
            self.atol,
            self.norm,
            y,
            updates,
            f,
            f - state.f,
        )
        new_state = optx._solver.optax._OptaxState(
            step=state.step + 1, f=f, opt_state=new_opt_state, terminate=terminate
        )
        return new_y, new_state, aux


class MaskedNonlinearCG(optx.NonlinearCG):
    def step(
            self,
            fn,
            y,
            args,
            options,
            state,
            tags,
    ):
        f_eval, lin_fn, aux_eval = jax.linearize(
            lambda _y: fn(_y, args), state.y_eval, has_aux=True
        )
        step_size, accept, search_result, search_state = self.search.step(
            state.first_step,
            y,
            state.y_eval,
            state.f_info,
            optx.FunctionInfo.Eval(f_eval),
            state.search_state,
        )

        def accepted(descent_state):
            (grad,) = optx._misc.lin_to_grad(lin_fn, y)
            grad = jnp.where(options['fixed_mask'], 0.0, grad)
            f_eval_info = optx.FunctionInfo.EvalGrad(f_eval, grad)
            descent_state = self.descent.query(state.y_eval, f_eval_info, descent_state)
            y_diff = (state.y_eval ** eqx.internal.ω - y ** eqx.internal.ω).ω
            f_diff = (f_eval ** eqx.internal.ω - state.f_info.f ** eqx.internal.ω).ω
            terminate = optx._misc.cauchy_termination(
                self.rtol, self.atol, self.norm, state.y_eval, y_diff, f_eval, f_diff
            )
            return state.y_eval, f_eval_info, aux_eval, descent_state, terminate

        def rejected(descent_state):
            return y, state.f_info, state.aux, descent_state, jnp.array(False)

        y, f_info, aux, descent_state, terminate = optx._misc.filter_cond(
            accept, accepted, rejected, state.descent_state
        )

        y_descent, descent_result = self.descent.step(step_size, descent_state)
        y_eval = (y ** eqx.internal.ω + y_descent ** eqx.internal.ω).ω
        result = optx.RESULTS.where(
            search_result == optx.RESULTS.successful, descent_result, search_result
        )

        state = optx._solver.gradient_methods._GradientDescentState(
            first_step=jnp.array(False),
            y_eval=y_eval,
            search_state=search_state,
            f_info=f_info,
            aux=aux,
            descent_state=descent_state,
            terminate=terminate,
            result=result,
        )
        return y, state, aux


class MaskedBFGS(optx.BFGS):

    def step(
            self,
            fn,
            y,
            args,
            options,
            state,
            tags,
    ):
        f_eval, lin_fn, aux_eval = jax.linearize(
            lambda _y: fn(_y, args), state.y_eval, has_aux=True
        )
        step_size, accept, search_result, search_state = self.search.step(
            state.first_step,
            y,
            state.y_eval,
            state.f_info,
            optx.FunctionInfo.Eval(f_eval),
            state.search_state,
        )

        def accepted(descent_state):
            (grad,) = optx._misc.lin_to_grad(lin_fn, y)
            grad = jnp.where(options['fixed_mask'], 0.0, grad)
            y_diff = (state.y_eval ** eqx.internal.ω - y ** eqx.internal.ω).ω
            if self.use_inverse:
                hessian = None
                hessian_inv = state.f_info.hessian_inv
            else:
                hessian = state.f_info.hessian
                hessian_inv = None
            f_eval_info = optx._solver.bfgs._bfgs_update(
                f_eval, grad, state.f_info.grad, hessian, hessian_inv, y_diff
            )
            descent_state = self.descent.query(
                state.y_eval,
                f_eval_info,  # pyright: ignore
                descent_state,
            )
            f_diff = (f_eval ** eqx.internal.ω - state.f_info.f ** eqx.internal.ω).ω
            terminate = optx._misc.cauchy_termination(
                self.rtol, self.atol, self.norm, state.y_eval, y_diff, f_eval, f_diff
            )
            return state.y_eval, f_eval_info, aux_eval, descent_state, terminate

        def rejected(descent_state):
            return y, state.f_info, state.aux, descent_state, jnp.array(False)

        y, f_info, aux, descent_state, terminate = optx._misc.filter_cond(
            accept, accepted, rejected, state.descent_state
        )

        y_descent, descent_result = self.descent.step(step_size, descent_state)
        y_eval = (y ** eqx.internal.ω + y_descent ** eqx.internal.ω).ω
        result = optx.RESULTS.where(
            search_result == optx.RESULTS.successful, descent_result, search_result
        )

        state = optx._solver.bfgs._BFGSState(
            first_step=jnp.array(False),
            y_eval=y_eval,
            search_state=search_state,
            f_info=f_info,
            aux=aux,
            descent_state=descent_state,
            terminate=terminate,
            result=result,
            num_accepted_steps=state.num_accepted_steps + accept,
        )
        return y, state, aux


class ImputerMetrics(VxData):
    n_steps: jnp.ndarray = eqx.field(default_factory=lambda: jnp.array([]))

    def __add__(self, other: Self) -> Self:
        return ImputerMetrics(n_steps=jnp.hstack((self.n_steps, other.n_steps)))


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
    def partial_input_optimise(self, input: jnp.ndarray, fixed_mask: jnp.ndarray) -> Tuple[jnp.ndarray, ImputerMetrics]:
        sol = optx.minimise(lambda y, args: self.f_energy(y),
                            solver=optx.BestSoFarMinimiser(solver=self.optax_solver()),
                            max_steps=2 ** 10,
                            options=dict(fixed_mask=fixed_mask),
                            y0=input, throw=False)
        num_steps = sol.stats['num_steps']
        return sol.value, ImputerMetrics(n_steps=jnp.array(num_steps))

    @staticmethod
    def optax_solver():
        return MaskedOptaxMinimiser(optax.adam(1e-7), rtol=1e-8, atol=1e-8)

    @staticmethod
    def cg_solver():
        return MaskedNonlinearCG(rtol=3e-8, atol=3e-8, method=optx.polak_ribiere)

    @staticmethod
    def bfgs_solver():
        return MaskedBFGS(rtol=3e-4, atol=3e-4)

    @eqx.filter_jit
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        input = jnp.hstack((state, jnp.zeros(self.observables_size)))
        mask = jnp.zeros_like(input).at[:self.state_size].set(1)
        output, _ = self.partial_input_optimise(input, mask)
        _, obs = jnp.split(output, [self.state_size])
        return obs


class ICNNObsExtractor(ICNNObsDecoder):

    @eqx.filter_jit
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.split(x, [self.state_size])[1]
