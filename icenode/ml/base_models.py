from typing import (AbstractSet, Any, Callable, Dict, Iterable, List, Mapping,
                    Optional, Tuple, Union)

import jax
import jax.numpy as jnp
import jax.random as jrandom
import eqionox as eqx

from diffrax import (ODETerm, Dopri5, diffeqsolve, SaveAt, BacksolveAdjoint,
                     NoAdjoint)

from jax.experimental.jet import jet
from jax.nn import softplus, sigmoid, leaky_relu


class ControlledDynamics(eqx.Module):
    f: Callable
    c: jnp.ndarray = eqx.static_field()

    def __init__(self, t, x, args):
        return self.f(t, jnp.concatenate((self.c, x)), args)


class GRUDynamics(eqx.Module):
    """
    Modified GRU unit to deal with latent state ``h(t)``.
    """

    x_r: eqx.Module
    x_z: eqx.Module
    rx_g: eqx.Module

    def __init__(self, hidden_size: int, state_size: int, use_bias: bool,
                 key: "jax.random.PRNGKey"):
        super().__init__()
        k1, k2, k3, k4, k5, k6 = jrandom.split(key, 6)
        self.x_r = self.shallow_net(state_size, hidden_size, jax.nn.relu,
                                    jax.nn.sigmoid, k1, k2)
        self.x_z = self.shallow_net(state_size, hidden_size, jax.nn.relu,
                                    jax.nn.sigmoid, k3, k4)
        self.rx_g = self.shallow_net(state_size, hidden_size, jax.nn.relu,
                                     jax.nn.tanh, k5, k6)

    @staticmethod
    def shallow_net(state_size, hidden_size, act1, key1, use_bias=True):
        return eqx.nn.Sequential([
            eqx.nn.Linear(hidden_size, state_size, use_bias=use_bias,
                          key=key2),
            eqx.nn.Lambda(act2)
        ])

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


class VectorField(eqx.Module):
    f_dyn: eqx.Module

    def __call__(self, t, x, args):
        return self.f_dyn(x)


class NeuralODE(eqx.Module):
    timescale: float = eqx.static_field()
    ode_dyn: Callable

    def __init__(self, f: Callable, tay_reg: int, timescale: float):
        self.timescale = timescale
        self.ode_dyn = TaylorAugmented(tay_reg, f)

    def __call__(self, t, x, args=dict()):
        dt0 = self.initial_step_size(t[0], x[0, :], 4, 1.4e-8, 1.4e-8,
                                     self.f_dyn(x[0, :]))
        n_samples = args.get('n_samples', 2)
        t = jnp.linspace(0.0, t / self.timescale, n_samples)
        x0 = self.get_x0(x, args)
        term = self.ode_term(self.ode_dyn, args)
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
                           adjoint=adjoint,
                           saveat=saveat,
                           max_steps=2**20).ys

    def get_x0(self, x0, args):
        if args.get('tay_reg', 0) > 0:
            return (x0, jnp.zeros(1))
        else:
            return x0

    def ode_term(self, args):
        term = VectorField(self.f_dyn)
        if args.get('control', None) is not None:
            term = ControlledDynamics(term, args['control'])
        if args.get('tay_reg', 0) > 0:
            term = TaylorAugmented(term, args['tay_reg'])

        return ODETerm(term)

    def initial_step_size(self, t0, y0, order, rtol, atol, f0):
        # Algorithm from:
        # E. Hairer, S. P. Norsett G. Wanner,
        # Solving Ordinary Differential Equations I: Nonstiff Problems, Sec. II.4.
        # Code from: https://github.com/google/jax/blob/main/jax/experimental/ode.py
        dtype = y0.dtype

        scale = atol + jnp.abs(y0) * rtol
        d0 = jnp.linalg.norm(y0 / scale.astype(dtype))
        d1 = jnp.linalg.norm(f0 / scale.astype(dtype))

        h0 = jnp.where((d0 < 1e-5) | (d1 < 1e-5), 1e-6, 0.01 * d0 / d1)
        y1 = y0 + h0.astype(dtype) * f0
        f1 = self.f_dyn(y1)
        d2 = jnp.linalg.norm((f1 - f0) / scale.astype(dtype)) / h0

        h1 = jnp.where((d1 <= 1e-15) & (d2 <= 1e-15),
                       jnp.maximum(1e-6, h0 * 1e-3),
                       (0.01 / jnp.max(d1 + d2))**(1. / (order + 1.)))

        return jnp.minimum(100. * h0, h1)


class StateUpdate(eqx.Module):
    """Implements discrete update based on the received observations."""
    f_project_error = Callable
    f_update = Callable

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
        self.f_update = eqx.nn.GRUCell(state_size + embeddings_size,
                                       state_size,
                                       use_bias=True,
                                       key=key2)

    def __call__(self, state: jnp.ndarray, predicted_emb: jnp.ndarray,
                 nominal_emb: jnp.ndarray) -> jnp.ndarray:

        projected_error = self.f_project_error(
            jnp.hstack((predicted_emb, nominal_emb)))
        gru_input = jnp.hstack((projected_error, state))

        return self.f_update(gru_input)


class StateInitializer(eqx.Module):

    def __init__(self,
                 hidden_size: int,
                 state_size: int,
                 depth: int,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.lin = [
            hk.Linear(hidden_size, name=f'lin_{i}') for i in range(depth)
        ]
        self.lin_out = hk.Linear(state_size, name='lin_out')

    def __call__(self, state_init: jnp.ndarray):
        out = state_init
        res = jnp.zeros(self.hidden_size, dtype=out.dtype)

        for lin in self.lin:
            out = lin(out)
            out = leaky_relu(out, negative_slope=2e-1) + res
            res = out

        return jnp.tanh(self.lin_out(out))


class StateDiagnosesDecoder(hk.Module):

    def __init__(self,
                 n_layers_d1: int,
                 n_layers_d2: int,
                 embeddings_size: int,
                 diag_size: int,
                 name: Optional[str] = None):
        super().__init__(name=name)

        def build_layers(n_layers, output_size):
            layers = [
                lambda x: leaky_relu(hk.Linear(embeddings_size)(x), 0.2)
                for i in range(n_layers - 2)
            ]
            layers.append(lambda x: jnp.tanh(hk.Linear(embeddings_size)(x)))
            layers.append(hk.Linear(output_size))

            return layers

        self.__dec1 = hk.Sequential(build_layers(n_layers_d1, embeddings_size),
                                    name='dec_1')
        self.__dec2 = hk.Sequential(build_layers(n_layers_d2, diag_size),
                                    name='dec_2')

    def __call__(self, h: jnp.ndarray):
        dec_emb = self.__dec1(h)
        dec_diag = self.__dec2(dec_emb)
        return dec_emb, jax.nn.softmax(dec_diag)


class EmbeddingsDecoder(hk.Module):

    def __init__(self,
                 n_layers: int,
                 embeddings_size: int,
                 diag_size: int,
                 name: Optional[str] = None):
        super().__init__(name=name)

        def build_layers(n_layers, output_size):
            layers = [
                lambda x: leaky_relu(hk.Linear(embeddings_size)(x), 0.2)
                for i in range(n_layers - 2)
            ]
            layers.append(lambda x: jnp.tanh(hk.Linear(embeddings_size)(x)))
            layers.append(hk.Linear(output_size))

            return layers

        self.__dec = hk.Sequential(build_layers(n_layers, diag_size),
                                   name='dec')

    def __call__(self, emb: jnp.ndarray):
        dec_diag = self.__dec(emb)
        return jax.nn.softmax(dec_diag)


class EmbeddingsDecoder_Logits(hk.Module):

    def __init__(self,
                 n_layers: int,
                 embeddings_size: int,
                 diag_size: int,
                 name: Optional[str] = None):
        super().__init__(name=name)

        def build_layers(n_layers, output_size):
            layers = [
                lambda x: leaky_relu(hk.Linear(embeddings_size)(x), 0.2)
                for i in range(n_layers - 2)
            ]
            if n_layers >= 2:
                layers.append(
                    lambda x: jnp.tanh(hk.Linear(embeddings_size)(x)))
            layers.append(hk.Linear(output_size))

            return layers

        self.__dec = hk.Sequential(build_layers(n_layers, diag_size),
                                   name='dec')

    def __call__(self, emb: jnp.ndarray):
        return self.__dec(emb)


class DiagnosticSamplesCombine(hk.Module):

    def __init__(self, embeddings_size: int, name: Optional[str] = None):
        super().__init__(name=name)
        self.__gru_e = hk.GRU(embeddings_size // 2, name='gru_1')
        self.__gru_d = hk.GRU(embeddings_size // 2, name='gru_2')
        self.__att_e = hk.Linear(1)
        self.__att_d = hk.Linear(1)

    def __call__(self, emb_seq: jnp.ndarray, diag_seq):
        initial_state = self.__gru_e.initial_state(1).squeeze()
        rnn_states_e, _ = hk.dynamic_unroll(self.__gru_e,
                                            emb_seq,
                                            initial_state,
                                            reverse=True)
        att_weights_e = jax.vmap(self.__att_e)(rnn_states_e)
        att_weights_e = jax.nn.tanh(att_weights_e)

        initial_state = self.__gru_d.initial_state(1).squeeze()
        rnn_states_d, _ = hk.dynamic_unroll(self.__gru_d,
                                            emb_seq,
                                            initial_state,
                                            reverse=True)
        att_weights_d = jax.vmap(self.__att_d)(rnn_states_d)
        att_weights_d = jax.nn.softmax(att_weights_d)

        return (sum(a * e for a, e in zip(att_weights_e, emb_seq)),
                sum(a * d for a, d in zip(att_weights_d, diag_seq)))
