from typing import (AbstractSet, Any, Callable, Dict, Iterable, List, Mapping,
                    Optional, Tuple, Union)

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.experimental.jet import jet
from jax.nn import sigmoid
import jax.tree_util as jtu
import equinox as eqx

from diffrax import (ODETerm, Dopri5, diffeqsolve, SaveAt, BacksolveAdjoint,
                     NoAdjoint)


class GRUDynamics(eqx.Module):
    """
    Modified GRU unit to deal with latent state ``h(t)``.
    """

    x_x: eqx.Module
    x_r: eqx.Module
    x_z: eqx.Module
    rx_g: eqx.Module

    def __init__(self, input_size: int, state_size: int, use_bias: bool,
                 key: "jax.random.PRNGKey"):
        k0, k1, k2, k3 = jrandom.split(key, 4)
        self.x_x = self.shallow_net(input_size, state_size, lambda x: x, k0)
        self.x_r = self.shallow_net(input_size, state_size, sigmoid, k1)
        self.x_z = self.shallow_net(input_size, state_size, sigmoid, k2)
        self.rx_g = self.shallow_net(input_size, state_size, jax.nn.tanh, k3)

    @staticmethod
    def shallow_net(input_size, state_size, act, key, use_bias=True):
        return eqx.nn.Sequential([
            eqx.nn.Linear(input_size, state_size, use_bias=use_bias, key=key),
            eqx.nn.Lambda(act)
        ])

    def weights(self):
        weights = eqx.filter(self, lambda m: m.weight)
        weights, _ = jtu.tree_flatten(weights)
        return weights

    def __call__(self, x: jnp.ndarray,
                 key: "jax.random.PRNGKey") -> jnp.ndarray:
        """
        Returns a change due to one step of using GRU-ODE for all h.
        Args:
            h: hidden state (current) of the observable variables.
            c: control variables.
        Returns:
            dh/dt
        """
        x = self.x_x(x)
        r = self.x_r(x)
        z = self.x_z(x)
        g = self.rx_g(r * x)
        return (1 - z) * (g - x)


class TaylorAugmented(eqx.Module):
    f: Callable
    order: int = 0

    def sol_recursive(self, t, x, args=None):
        """
        https://github.com/jacobjinkelly/easy-neural-ode/blob/master/latent_ode.py
        By Jacob Kelly
        Recursively compute higher order derivatives of dynamics of ODE.
        # MIT License

        # Copyright (c) 2020 Jacob Kelly and Jesse Bettencourt

        # Permission is hereby granted, free of charge, to any person obtaining a copy
        # of this software and associated documentation files (the "Software"), to deal
        # in the Software without restriction, including without limitation the rights
        # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        # copies of the Software, and to permit persons to whom the Software is
        # furnished to do so, subject to the following conditions:

        # The above copyright notice and this permission notice shall be included in all
        # copies or substantial portions of the Software.

        # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        # SOFTWARE.
        """
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

        (y0, [*yns]) = jet(g, (t_x, ), ((jnp.ones_like(t_x), ), ))
        for _ in range(self.order - 1):
            (y0, [*yns]) = jet(g, (t_x, ), ((y0, *yns), ))

        return (jnp.reshape(y0[1:],
                            x_shape), jnp.reshape(yns[-2][1:], x_shape))

    def __call__(self, t, x_r: Tuple[jnp.ndarray, jnp.ndarray], args):
        x, _ = x_r

        dydt, drdt = self.sol_recursive(t, x, args)

        return dydt, jnp.mean(drdt**2)


class ControlledDynamics(eqx.Module):
    f_dyn: Callable

    def __call__(self, t, x, args):
        return self.f_dyn(t, jnp.concatenate((args['control'], x)), args)


class VectorField(eqx.Module):
    f_dyn: eqx.Module

    def __call__(self, t, x, args):
        return self.f_dyn(x)


class NeuralODE(eqx.Module):
    ode_dyn: Callable
    timescale: float = eqx.static_field()

    @staticmethod
    def timesamples(int_time, dt):
        if dt < 0:
            dt = int_time
        return jnp.linspace(0, int_time - int_time % dt,
                            round((int_time - int_time % dt) / dt + 1))

    def __call__(self, t, x, args=dict()):
        x0 = self.get_x0(x, args)

        dt0 = self.initial_step_size(0, x, 4, 1.4e-8, 1.4e-8, args)
        sampling_rate = args.get('sampling_rate', None)
        if sampling_rate:
            t = self.timesamples(float(t), sampling_rate)
        else:
            t = jnp.linspace(0.0, t / self.timescale, 2)

        term = self.ode_term(args)
        saveat = SaveAt(ts=t)
        solver = Dopri5()

        eval_only = args.get('eval_only', False)
        if eval_only:
            adjoint = NoAdjoint()
        else:
            adjoint = BacksolveAdjoint(solver=Dopri5())
        return diffeqsolve(term,
                           solver,
                           t[0],
                           t[-1],
                           dt0=dt0,
                           y0=x0,
                           args=args,
                           adjoint=adjoint,
                           saveat=saveat,
                           max_steps=2**20).ys

    def get_x0(self, x0, args):
        if args.get('tay_reg', 0) > 0:
            return (x0, jnp.array(0.0))
        else:
            return x0

    def ode_term(self, args):
        term = VectorField(self.ode_dyn)
        if len(args.get('control', jnp.array([]))) > 0:
            term = ControlledDynamics(term)
        if args.get('tay_reg', 0) > 0:
            term = TaylorAugmented(term, args['tay_reg'])

        return ODETerm(term)

    def initial_step_size(self, t0, y0, order, rtol, atol, args):
        # Algorithm from:
        # E. Hairer, S. P. Norsett G. Wanner,
        # Solving Ordinary Differential Equations I: Nonstiff Problems, Sec. II.4.
        # Code from: https://github.com/google/jax/blob/main/jax/experimental/ode.py
        ctrl = args.get('control', jnp.array([]))
        ctrl_y = lambda y: jnp.concatenate((ctrl, y))
        f0 = self.ode_dyn(ctrl_y(y0))

        dtype = y0.dtype

        scale = atol + jnp.abs(y0) * rtol
        d0 = jnp.linalg.norm(y0 / scale.astype(dtype))
        d1 = jnp.linalg.norm(f0 / scale.astype(dtype))

        h0 = jnp.where((d0 < 1e-5) | (d1 < 1e-5), 1e-6, 0.01 * d0 / d1)
        y1 = y0 + h0.astype(dtype) * f0
        f1 = self.ode_dyn(ctrl_y(y1))
        d2 = jnp.linalg.norm((f1 - f0) / scale.astype(dtype)) / h0

        h1 = jnp.where((d1 <= 1e-15) & (d2 <= 1e-15),
                       jnp.maximum(1e-6, h0 * 1e-3),
                       (0.01 / jnp.max(d1 + d2))**(1. / (order + 1.)))

        return jnp.minimum(100. * h0, h1)


class StateUpdate(eqx.Module):
    """Implements discrete update based on the received observations."""
    f_project_error: Callable
    f_update: Callable

    def __init__(self, state_size: int, embeddings_size: int,
                 key: "jax.random.PRNGKey"):
        key1, key2 = jrandom.split(key, 2)
        self.f_project_error = eqx.nn.Sequential([
            eqx.nn.Linear(embeddings_size * 2,
                          embeddings_size,
                          use_bias=True,
                          key=key1),
            eqx.nn.Lambda(jnp.tanh)
        ])
        self.f_update = eqx.nn.GRUCell(embeddings_size,
                                       state_size,
                                       use_bias=True,
                                       key=key2)

    def __call__(self, state: jnp.ndarray, predicted_emb: jnp.ndarray,
                 nominal_emb: jnp.ndarray) -> jnp.ndarray:

        projected_error = self.f_project_error(
            jnp.hstack((predicted_emb, nominal_emb)))

        return self.f_update(projected_error, state)
