"""Hard and Soft (differentiable) implementations of
the Dynamic Time Warping (DTW) distance
"""

from jax.scipy.special import logsumexp

import jax
import jax.numpy as jnp
from jax.nn import sigmoid

from .utils import OOPError

# MIT License

# Copyright (c) 2021 Konrad Heidler

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
class AbstractDTW:
    """https://github.com/khdlr/softdtw_jax/blob/main/softdtw_jax/softdtw_jax.py"""

    def __init__(self, pairwise_distance_f):
        self.pairwise_distance_f = pairwise_distance_f

    def minimum(self, *args):
        raise OOPError('This should be implemented')

    def __call__(self, target, prediction):
        """
        Compute the DTW distance.

        """

        def pad_inf(inp, before, after):
            return jnp.pad(inp, (before, after), constant_values=jnp.inf)

        D = self.pairwise_distance_f(target, prediction)

        # wlog: H >= W
        if D.shape[0] < D.shape[1]:
            D = D.T
        H, W = D.shape

        rows = []
        for row in range(H):
            rows.append(pad_inf(D[row], row, H - row - 1))

        model_matrix = jnp.stack(rows, axis=1)
        init = (pad_inf(model_matrix[0], 1,
                        0), pad_inf(model_matrix[1] + model_matrix[0, 0], 1,
                                    0))

        def scan_step(carry, current_antidiagonal):
            two_ago, one_ago = carry

            diagonal = two_ago[:-1]
            right = one_ago[:-1]
            down = one_ago[1:]
            best = self.minimum(jnp.stack([diagonal, right, down], axis=-1))
            next_row = best + current_antidiagonal
            next_row = pad_inf(next_row, 1, 0)

            return (one_ago, next_row), next_row

        # Manual unrolling:
        # carry = init
        # for i, row in enumerate(model_matrix[2:]):
        #     carry, y = scan_step(carry, row)

        carry, ys = jax.lax.scan(scan_step, init, model_matrix[2:], unroll=4)
        return carry[1][-1]


class DTW(AbstractDTW):
    """
    SoftDTW as proposed in the paper "Dynamic programming algorithm optimization for spoken word recognition"
    by Hiroaki Sakoe and Seibi Chiba (https://arxiv.org/abs/1703.01541)

    Expects inputs of the shape [T, D], where T is the time dimension
    and D is the feature dimension.
    """
    __name__ = 'DTW'

    def minimum(self, args):
        return jnp.min(args, axis=-1)


class SoftDTW(AbstractDTW):
    """
    SoftDTW as proposed in the paper "Soft-DTW: a Differentiable Loss Function for Time-Series"
    by Marco Cuturi and Mathieu Blondel (https://arxiv.org/abs/1703.01541)

    Expects inputs of the shape [T, D], where T is the time dimension
    and D is the feature dimension.
    """
    __name__ = 'SoftDTW'

    def __init__(self, pairwise_distance_f, gamma=1.0):
        super().__init__(pairwise_distance_f=pairwise_distance_f)

        assert gamma > 0, "Gamma needs to be positive."
        self.gamma = gamma
        self.__name__ = f'SoftDTW({self.gamma})'
        self.minimum_impl = self.make_softmin(gamma)

    def make_softmin(self, gamma):
        """
        We need to manually define the gradient of softmin
        to ensure (1) numerical stability and (2) prevent nans from
        propagating over valid values.
        """

        def softmin_raw(array):
            return -gamma * logsumexp(array / -gamma, axis=-1)

        softmin = jax.custom_vjp(softmin_raw)

        def softmin_fwd(array):
            return softmin(array), (array / -gamma, )

        def softmin_bwd(res, g):
            scaled_array, = res
            grad = jnp.where(
                jnp.isinf(scaled_array), jnp.zeros(scaled_array.shape),
                jax.nn.softmax(scaled_array) * jnp.expand_dims(g, 1))
            return grad,

        softmin.defvjp(softmin_fwd, softmin_bwd)
        return softmin

    def minimum(self, args):
        return self.minimum_impl(args)


@jax.jit
def distance_matrix_bce(a, b_logits):
    """
    Return pairwise crossentropy between two timeseries.
    Args:
        a: First time series (m, p).
        b: Second time series (n, p).
    Returns:
        An (m, n) distance matrix computed by a pairwise distance function.
            on the elements of a and b.
    """
    m, p = a.shape
    n, p = b_logits.shape
    assert a.shape[1] == b_logits.shape[1], "Dimensions mismatch."

    b_logits = jnp.broadcast_to(b_logits, (m, n, p))
    a = jnp.expand_dims(a, 1)
    a = jnp.broadcast_to(a, (m, n, p))

    D = a * jax.nn.softplus(-b_logits) + (1 - a) * jax.nn.softplus(b_logits)

    return jnp.mean(D, axis=-1)


@jax.jit
def distance_matrix_euc(a, b_logits):
    b = sigmoid(b_logits)
    a = jnp.expand_dims(a, axis=1)
    b = jnp.expand_dims(b, axis=0)
    D = jnp.square(a - b)
    return jnp.mean(D, axis=-1)
