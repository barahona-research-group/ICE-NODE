from typing import (AbstractSet, Any, Callable, Dict, Iterable, List, Mapping,
                    Optional, Tuple, Union)

import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as R
from jax.nn import softplus, sigmoid, leaky_relu


class InvertibleCoupledLayer(hk.Module):
    """
    Coupling Block following the RealNVP design (Dinh et al, 2017).

    References:
        https://github.com/VLL-HD/FrEIA/blob/master/FrEIA/modules/coupling_layers.py
        https://arxiv.org/pdf/1605.08803.pdf
        https://arxiv.org/pdf/1808.04730.pdf
    """

    def __init__(self,
                 size: int,
                 depth: int,
                 init_scale: float,
                 split_len: float = 0.5,
                 clamp_val: float = 2.,
                 clamp_activation: str = 'atan',
                 name: str = None):
        """
        Args:
          size: the input/output dimension.
          depth: number of layers for each function of s1, s2, t1, t2
          init_scale: the variance of the randomly initialized parameters.
          split_len: splitting ratio of the input/output.
          clamp_val: soft-clamping values such that s1, s2 around bounded by exp(±clamp),
          clamp_activation: the activation function used in soft-clamping.
        """
        super().__init__(name=name)

        if clamp_activation == "atan":
            f_clamp = (lambda u: 0.636 * jnp.arctan(u) * clamp_val)
        elif clamp_activation == "tanh":
            f_clamp = (lambda u: jnp.tanh(u) * clamp_val)
        elif clamp_activation == "sigmoid":
            f_clamp = (lambda u: 2. * (jax.nn.sigmoid(u) - 0.5) * clamp_val)
        else:
            raise ValueError(f'Unknown clamp activation "{clamp_activation}"')

        # index at which input u is split to u1, u2
        self.split_index = round(size * split_len)

        # u1 size, also output size for s2, t2
        len1 = self.split_index
        # u2 size, also output size for s1, t1
        len2 = size - self.split_index

        init = hk.initializers.VarianceScaling(init_scale)

        def build_layers(out_size):
            l = [
                lambda x: jax.nn.leaky_relu(
                    hk.Linear(out_size, w_init=init)(x))
                for _ in range(depth - 1)
            ]
            l.append(lambda x: f_clamp(hk.Linear(out_size, w_init=init)(x)))
            return l

        self.s1 = hk.Sequential(build_layers(len2), f'{name}_s1')
        self.s2 = hk.Sequential(build_layers(len1), f'{name}_s2')
        self.t1 = hk.Sequential(build_layers(len2), f'{name}_t1')
        self.t2 = hk.Sequential(build_layers(len1), f'{name}_t2')

    def forward(self, u):
        u1, u2 = jnp.split(u, (self.split_index, ))
        s2 = self.s2(u2)
        v1 = u1 * jnp.exp(s2) + self.t2(u2)

        s1 = self.s1(v1)
        v2 = u2 * jnp.exp(s1) + self.t1(v1)

        return jnp.concatenate((v1, v2))

    def inverse(self, v):
        v1, v2 = jnp.split(v, (self.split_index, ))

        s1 = self.s1(v1)
        u2 = (v2 - self.t1(v1)) * jnp.exp(-s1)

        s2 = self.s2(u2)
        u1 = (v1 - self.t2(u2)) * jnp.exp(-s2)

        return jnp.concatenate((u1, u2))

    def __call__(self, forward: bool, u_or_v):
        if forward:
            return self.forward(u_or_v)
        else:
            return self.inverse(u_or_v)


class InvertibleLayers(hk.Module):

    def __init__(self, size: int, n_layers: int, name=None):
        super().__init__(name=name)
        icl_config = {
            'size': size,
            'split_len': 0.5,
            'clamp_val': 2.,
            'clamp_activation': 'atan',
            'depth': 4,
            'init_scale': 1e-1
        }

        self.layers = tuple(
            InvertibleCoupledLayer(**icl_config, name=f'ICL{i}')
            for i in range(n_layers))

        prng = R.split(R.PRNGKey(10), num=n_layers - 1)

        # Fixed permutations, used between layers in the forward function.
        self.permuters = tuple(R.permutation(key, size) for key in prng)

        # Sorters are used between layers in the inverse function,
        # to reverse the effect of permuters from the last to the first.
        self.sorters = tuple(jnp.argsort(p) for p in reversed(self.permuters))

    def forward(self, u):
        for layer, permuter in zip(self.layers[:-1], self.permuters):
            v = layer(True, u)
            u = v[permuter]

        return self.layers[-1](True, u)

    def inverse(self, v):
        for layer, sorter in zip(reversed(self.layers[1:]), self.sorters):
            u = layer(False, v)
            v = u[sorter]

        return self.layers[0](False, v)

    def __call__(self, forward: bool, u_or_v):
        if forward:
            return self.forward(u_or_v)
        else:
            return self.inverse(u_or_v)
