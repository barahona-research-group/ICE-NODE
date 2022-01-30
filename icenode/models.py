from typing import (AbstractSet, Any, Callable, Dict, Iterable, List, Mapping,
                    Optional, Tuple, Union)

import haiku as hk
import jax
import jax.numpy as jnp
from jax.experimental.jet import jet
from jax.experimental.ode import odeint
from jax.nn import softplus, sigmoid, leaky_relu

from .odeint_nfe import odeint as odeint_nfe

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

    def __call__(self, h, t, c):
        out = jnp.hstack((c, h, t))

        for lin in self.lin[:-1]:
            out = lin(out)
            out = leaky_relu(out, negative_slope=2e-1)
            out = jnp.hstack((c, out, t))
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

    def __call__(self, h, t, c):
        out = jnp.hstack((c, h, t))

        res = jnp.zeros_like(h)
        for lin in self.lin[:-1]:
            out = lin(out)
            out = leaky_relu(out, negative_slope=2e-1) + res
            res = out
            out = jnp.hstack((c, out, t))
        out = self.lin[-1](out)
        return jnp.tanh(out)


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
        self.__hc_r = hk.Sequential([
            hk.Linear(state_size,
                      with_bias=with_bias,
                      w_init=w_init,
                      b_init=b_init,
                      name='hc_r'), sigmoid
        ])

        self.__hc_z = hk.Sequential([
            hk.Linear(state_size,
                      with_bias=with_bias,
                      w_init=w_init,
                      b_init=b_init,
                      name='hc_z'), sigmoid
        ])

        self.__rhc_g = hk.Sequential([
            hk.Linear(state_size,
                      with_bias=with_bias,
                      w_init=w_init,
                      b_init=b_init,
                      name='rhc_g'), jnp.tanh
        ])

    def __call__(self, h: jnp.ndarray, t: float,
                 c: jnp.ndarray) -> jnp.ndarray:
        """
        Returns a change due to one step of using GRU-ODE for all h.
        Args:
            h: hidden state (current) of the observable variables.
            c: control variables.
        Returns:
            dh/dt
        """
        htc = jnp.hstack((h, c))

        r = self.__hc_r(htc)
        z = self.__hc_z(htc)
        rhtc = jnp.hstack([r * h, c])
        g = self.__rhc_g(rhtc)

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
        """
        if self.order < 2:
            return self.f(h, t, c), jnp.zeros_like(h)

        h_shape = h.shape

        def g(h_t):
            """
            Closure to expand z.
            """
            _h, _t = jnp.reshape(h_t[:-1], h_shape), h_t[-1]
            dh = jnp.ravel(self.f(_h, _t, c))
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

        dydt, _drdt = self.sol_recursive(h, t, c)

        return dydt, jnp.mean(_drdt**2)


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

    def __call__(self, count_nfe, h, t, c):
        t = t / self.timescale
        if hk.running_init():
            return self.ode_dyn((h, jnp.zeros(1)), t, c), jnp.zeros(1), 0
        if count_nfe:
            (h, r), nfe = odeint_nfe(self.ode_dyn, (h, jnp.zeros(1)),
                                     jnp.array([0.0, t]), c)
        else:
            h, r = odeint(self.ode_dyn, (h, jnp.zeros(1)), jnp.array([0.0, t]),
                          c)
            nfe = 0
        h1 = jnp.split(h, 2)[1].squeeze()
        r1 = jnp.split(r, 2)[1].squeeze()
        return h1, r1, nfe


class NumericObsModel(hk.Module):
    """
    The mapping from hidden h to the distribution parameters of Y(t).
    The dimension of the transformation is |h|->|obs_hidden|->|2D|.
    """
    def __init__(self,
                 numeric_size: int,
                 numeric_hidden_size: int,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.__numeric_size = numeric_size
        self.__numeric_hidden_size = numeric_hidden_size
        self.__lin_mean = hk.Linear(numeric_size)
        self.__lin_logvar = hk.Linear(numeric_size)

    def __call__(self, h: jnp.ndarray) -> jnp.ndarray:
        out = hk.Linear(self.__numeric_hidden_size)(h)
        out = leaky_relu(out, negative_slope=2e-1)

        out_logvar = leaky_relu(self.__lin_logvar(out),
                                negative_slope=2e-1) - 5
        out_mean = 4 * jnp.tanh(self.__lin_mean(out))
        return out_mean, out_logvar


class GRUBayes(hk.Module):
    """Implements discrete update based on the received observations."""
    def __init__(self, state_size: int, name: Optional[str] = None):
        super().__init__(name=name)
        self.__prep = hk.Sequential([
            hk.Linear(state_size, with_bias=True, name=f'{name}_prep1'),
            lambda o: leaky_relu(o, negative_slope=2e-1),
            hk.Linear(state_size, with_bias=True,
                      name=f'{name}_prep2'), jnp.tanh
        ])

        self.__gru_d = hk.GRU(state_size)

    def __call__(self, state: jnp.ndarray, error_numeric: jnp.ndarray,
                 numeric_mask: jnp.ndarray,
                 error_gram: jnp.ndarray) -> jnp.ndarray:

        error_numeric = error_numeric * numeric_mask
        gru_input = self.__prep(jnp.hstack((error_numeric, error_gram)))
        _, updated_state = self.__gru_d(gru_input, state)
        return updated_state


class DiagnosesUpdate(hk.Module):
    """Implements discrete update based on the received observations."""
    def __init__(self, state_size: int, name: Optional[str] = None):
        super().__init__(name=name)
        self.__prep = hk.Sequential([
            hk.Linear(state_size, with_bias=True, name=f'{name}_prep1'),
            lambda o: leaky_relu(o, negative_slope=2e-1),
            hk.Linear(state_size, with_bias=True,
                      name=f'{name}_prep2'), jnp.tanh
        ])

        self.__gru_d = hk.GRU(state_size)

    def __call__(self, state: jnp.ndarray,
                 error_gram: jnp.ndarray) -> jnp.ndarray:

        gru_input = self.__prep(error_gram)
        _, updated_state = self.__gru_d(gru_input, state)
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
                 hidden_size: int,
                 embeddings_size: int,
                 output_size: int,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.__lin_h = hk.Linear(hidden_size // 2, name='lin_h_hidden')
        self.__lin_emb = hk.Linear(embeddings_size, name='lin_gram')
        self.__lin_out = hk.Linear(output_size, name='lin_out')

    def __call__(self, h: jnp.ndarray):
        out_h = jnp.tanh(self.__lin_h(h))
        dec_emb = self.__lin_emb(out_h)
        logits = self.__lin_out(leaky_relu(dec_emb, negative_slope=2e-1))
        return dec_emb, logits