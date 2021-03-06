from typing import (AbstractSet, Any, Callable, Dict, Iterable, List, Mapping,
                    Optional, Tuple, Union)

import haiku as hk
import jax
import jax.numpy as jnp
from jax.experimental.jet import jet
from jax.experimental.ode import odeint
from jax.nn import softplus, sigmoid, leaky_relu

from ..ode.odeint_nfe import odeint as odeint_nfe

from ..experimental.inn_models import InvertibleLayers

# GRU-ODE: Neural Negative Feedback ODE with Bayesian jumps
'''
Note(Asem): Haiku modules don't define the input dimensions
Input dimensions are defined during initialization step afterwards.
'''

# class QuadraticAugmentation(hk.Module):
#     """
#     Dynamics for ODE as a parametric function of the terms x1^2, x1x2, x1x3,...
#     """
#     def __init__(self, name: Optional[str] = None):
#         super().__init__(name=name)

#     def __call__(self, x):
#         out = hk.Linear(jnp.size(x), with_bias=True)(x)
#         return out * x


# IDEA: research on mixed loss functions.
# Instead of using hyperparamters for mixing loss functions,
# Each training iteration optimizes one of the loss functions on a rolling
# basis.
class MLPDynamics(hk.Module):
    """
    Dynamics for ODE as an MLP.
    """

    def __init__(self,
                 state_size: int,
                 depth: int,
                 with_bias: bool,
                 w_init: hk.initializers.Initializer,
                 b_init: hk.initializers.Initializer,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.lin = [
            hk.Linear(state_size,
                      with_bias=with_bias,
                      w_init=w_init,
                      b_init=b_init,
                      name=f'lin_{i}') for i in range(depth)
        ]

    def __call__(self, h, c):
        out = jnp.hstack((h, c))

        for lin in self.lin[:-1]:
            out = jnp.tanh(lin(out))
            out = jnp.hstack((out, c))

        out = self.lin[-1](out)
        return jnp.tanh(out)


class ResDynamics(hk.Module):
    """
    Dynamics for ODE as an MLP.
    """

    def __init__(self,
                 state_size: int,
                 depth: int,
                 with_bias: bool,
                 w_init: hk.initializers.Initializer,
                 b_init: hk.initializers.Initializer,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.lin = [
            hk.Linear(state_size,
                      with_bias=with_bias,
                      w_init=w_init,
                      b_init=b_init,
                      name=f'lin_{i}') for i in range(depth)
        ]

    def __call__(self, h, c):
        out = jnp.hstack((h, c))
        res = jnp.zeros_like(h)
        for lin in self.lin[:-1]:
            out = lin(out)
            out = jnp.tanh(out) + res
            res = out
            out = jnp.hstack((out, c))

        out = self.lin[-1](out)
        return out


class GRUDynamics(hk.Module):
    """
    Modified GRU unit to deal with latent state ``h(t)``.
    """

    def __init__(self,
                 state_size: int,
                 with_bias: bool,
                 w_init: hk.initializers.Initializer,
                 b_init: hk.initializers.Initializer,
                 name: Optional[str] = None,
                 **ignored_kwargs):
        super().__init__(name=name)
        self.h_r = hk.Sequential([
            hk.Linear(state_size,
                      with_bias=with_bias,
                      w_init=w_init,
                      b_init=b_init,
                      name='hc_r'), sigmoid
        ])

        self.h_z = hk.Sequential([
            hk.Linear(state_size,
                      with_bias=with_bias,
                      w_init=w_init,
                      b_init=b_init,
                      name='hc_z'), sigmoid
        ])

        self.rh_g = hk.Sequential([
            hk.Linear(state_size,
                      with_bias=with_bias,
                      w_init=w_init,
                      b_init=b_init,
                      name='rhc_g'), jnp.tanh
        ])

    def __call__(self, h: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
        """
        Returns a change due to one step of using GRU-ODE for all h.
        Args:
            h: hidden state (current) of the observable variables.
            c: control variables.
        Returns:
            dh/dt
        """
        hc = jnp.hstack((h, c))
        r = self.h_r(hc)
        z = self.h_z(hc)
        g = self.rh_g(r * h)
        return (1 - z) * (g - h)


class TaylorAugmented(hk.Module):

    def __init__(self, order, dynamics_cls, **dynamics_kwargs):
        super().__init__(name=f"{dynamics_kwargs.get('name')}_augment")
        self.order = order or 0
        self.f = dynamics_cls(**dynamics_kwargs)

    def sol_recursive(self, h: jnp.ndarray, t: float, c: jnp.ndarray):
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
            return self.f(h, c), jnp.zeros_like(h)

        h_shape = h.shape

        def g(h_t):
            """
            Closure to expand z.
            """
            _h, _t = jnp.reshape(h_t[:-1], h_shape), h_t[-1]
            dh = jnp.ravel(self.f(_h, c))
            dt = jnp.array([1.])
            dh_t = jnp.concatenate((dh, dt))
            return dh_t

        h_t = jnp.concatenate((jnp.ravel(h), jnp.array([t])))

        (y0, [*yns]) = jet(g, (h_t, ), ((jnp.ones_like(h_t), ), ))
        for _ in range(self.order - 1):
            (y0, [*yns]) = jet(g, (h_t, ), ((y0, *yns), ))

        return (jnp.reshape(y0[:-1],
                            h_shape), jnp.reshape(yns[-2][:-1], h_shape))

    def __call__(self, h_r: Tuple[jnp.ndarray, jnp.ndarray], t: float,
                 c: jnp.ndarray):
        h, _ = h_r

        dydt, drdt = self.sol_recursive(h, t, c)

        return dydt, jnp.mean(drdt**2)


class NeuralODE(hk.Module):

    def __init__(self,
                 ode_dyn_cls: Any,
                 state_size: int,
                 depth: int,
                 tay_reg: int,
                 with_bias: True,
                 timescale: float,
                 init_var: float,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.timescale = timescale
        init = hk.initializers.VarianceScaling(init_var, mode='fan_avg')
        init_kwargs = {'with_bias': with_bias, 'b_init': None, 'w_init': init}
        self.ode_dyn = TaylorAugmented(order=tay_reg,
                                       dynamics_cls=ode_dyn_cls,
                                       state_size=state_size,
                                       depth=depth,
                                       name='ode_dyn',
                                       **init_kwargs)

    def __call__(self, n_samples, count_nfe, h, t, c=None):
        if c is None:
            c = jnp.array([])

        t = jnp.linspace(0.0, t / self.timescale, n_samples)
        if hk.running_init():
            h, r = self.ode_dyn((h, jnp.zeros(1)), t[0], c)
            h = jnp.broadcast_to(h, (len(t), len(h)))
            return h, jnp.zeros(1), 0
        if count_nfe:
            (h, r), nfe = odeint_nfe(self.ode_dyn, (h, jnp.zeros(1)), t, c)
        else:
            h, r = odeint(self.ode_dyn, (h, jnp.zeros(1)), t, c)
            nfe = 0
        return h, r[-1], nfe


class TaylorAndLossAugmentedDynamics(hk.Module):

    def __init__(self, loss_f: Callable[[jnp.ndarray], float], order: int,
                 dynamics_cls, **dynamics_kwargs):
        super().__init__(name=f"{dynamics_kwargs.get('name')}_augment")
        self.order = order or 0
        self.f = dynamics_cls(**dynamics_kwargs)
        self.loss_f = loss_f

    def sol_recursive(self, h: jnp.ndarray, t: float, c: jnp.ndarray):
        """
        https://github.com/jacobjinkelly/easy-neural-ode/blob/master/latent_ode.py
        By Jacob Kelly
        Recursively compute higher order derivatives of dynamics of ODE.
        """
        if self.order < 2:
            return self.f(h, t), jnp.zeros_like(h)

        h_shape = h.shape

        def g(h_t):
            """
            Closure to expand z.
            """
            _h, _t = jnp.reshape(h_t[:-1], h_shape), h_t[-1]
            dh = jnp.ravel(self.f(_h, c))
            dt = jnp.array([1.])
            dh_t = jnp.concatenate((dh, dt))
            return dh_t

        h_t = jnp.concatenate((jnp.ravel(h), jnp.array([t])))

        (h0, [*hns]) = jet(g, (h_t, ), ((jnp.ones_like(h_t), ), ))
        for _ in range(self.order - 1):
            (h0, [*hns]) = jet(g, (h_t, ), ((h0, *hns), ))

        return (jnp.reshape(h0[:-1],
                            h_shape), jnp.reshape(hns[-2][:-1], h_shape))

    def __call__(self, h_l_r: Tuple[jnp.ndarray, jnp.ndarray], t: float,
                 c: jnp.ndarray, *loss_args):
        h, _, _ = h_l_r

        dhdt, _drdt = self.sol_recursive(h, t, c)

        dLdt = self.loss_f(h, dhdt, t, *loss_args)
        return dhdt, dLdt, jnp.mean(_drdt**2)


class NeuralODE_IL(hk.Module):

    def __init__(self,
                 ode_dyn_cls: Any,
                 state_size: int,
                 depth: int,
                 loss_f: Callable[[jnp.ndarray], float],
                 tay_reg: int,
                 with_bias: bool,
                 init_var: float,
                 timescale: float,
                 name: Optional[str] = None):
        super().__init__(name=name)
        init = hk.initializers.VarianceScaling(init_var, mode='fan_avg')
        init_kwargs = {'with_bias': with_bias, 'b_init': None, 'w_init': init}
        self.ode_dyn = TaylorAndLossAugmentedDynamics(loss_f=loss_f,
                                                      order=tay_reg,
                                                      dynamics_cls=ode_dyn_cls,
                                                      state_size=state_size,
                                                      depth=depth,
                                                      name='ode_dyn',
                                                      **init_kwargs)
        self.timescale = timescale

    def __call__(self, count_nfe: bool, h: jnp.ndarray, tf, c, *loss_args):
        if c is None:
            c = jnp.array([])

        t = jnp.array([0.0, tf / self.timescale])

        if hk.running_init():
            h, l, r = self.ode_dyn((h, 0.0, 0.0), t[0], c, *loss_args)
            return h, 0.0, 0.0, 0
        if count_nfe:
            (h, l, r), nfe = odeint_nfe(self.ode_dyn, (h, 0.0, 0.0), t, c,
                                        *loss_args)
        else:
            h, l, r = odeint(self.ode_dyn, (h, 0.0, 0.0), t, c, *loss_args)
            nfe = 0
        return h[-1, :], l[-1], r[-1], nfe


class StateUpdate(hk.Module):
    """Implements discrete update based on the received observations."""

    def __init__(self,
                 state_size: int,
                 embeddings_size: int,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.__project = hk.Sequential([
            hk.Linear(embeddings_size, with_bias=True, name=f'{name}_prep'),
            jnp.tanh
        ])

        self.__gru = hk.GRU(state_size)

    def __call__(self, state: jnp.ndarray, emb: jnp.ndarray,
                 nominal_emb: jnp.ndarray) -> jnp.ndarray:

        gru_input = self.__project(jnp.hstack((emb, nominal_emb)))
        _, updated_state = self.__gru(gru_input, state)
        return updated_state


class StateInitializer(hk.Module):

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
