import math
from typing import Tuple

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr


class ConvexInitialiser(eqx.Module):
    """
    Initialisation method for input-convex networks.

    Parameters
    ----------
    var : float, optional
        The target variance fixed point.
        Should be a positive number.
    corr : float, optional
        The target correlation fixed point.
        Should be a value between -1 and 1, but typically positive.
    bias_noise : float, optional
        The fraction of variance to originate from the bias parameters.
        Should be a value between 0 and 1
    alpha : float, optional
        Scaling parameter for leaky ReLU.
        Should be a positive number.

    Examples
    --------
    Default initialisation

    >>> icnn = torch.nn.Sequential(
    ...     torch.nn.Linear(200, 400),
    ...     torch.nn.ReLU(),
    ...     ConvexLinear(400, 300),
    ... )
    >>> torch.nn.init.kaiming_uniform_(icnn[0].weight, nonlinearity="linear")
    >>> torch.nn.init.zeros_(icnn[0].bias)
    >>> convex_init = ConvexInitialiser()
    >>> w1, b1 = icnn[1].parameters()
    >>> convex_init(w1, b1)
    >>> assert torch.all(w1 >= 0) and torch.isclose(b1.var(), torch.zeros(1))

    Initialisation with random bias parameters

    >>> convex_bias_init = ConvexInitialiser(bias_noise=0.5)
    >>> convex_bias_init(w1, b1)
    >>> assert torch.all(w1 >= 0) and b1.var() > 0
    """

    var: float = 1.0
    corr: float = 0.5
    bias_noise: float = 0.0
    alpha: float = 0.0
    relu_scale: float = eqx.field(init=False)

    def __post_init__(self):
        self.relu_scale = 2. / (1. + self.alpha ** 2)

    @staticmethod
    def init_log_normal(weight_shape: Tuple[int, ...], mean_sq: float, var: float, key: jr.PRNGKey) -> jnp.ndarray:
        """
        Initialise weights with samples from a log-normal distribution.

        Parameters
        ----------
        weight : torch.Tensor
            The parameter to be initialised.
        mean_sq : float
            The squared mean of the normal distribution underlying the log-normal.
        var : float
            The variance of the normal distribution underlying the log-normal.

        Returns
        -------
        weight : torch.Tensor
            A reference to the inputs that have been modified in-place.
        """
        log_mom2 = math.log(mean_sq + var)
        log_mean = math.log(mean_sq) - log_mom2 / 2.
        log_var = log_mom2 - math.log(mean_sq)
        return jnp.exp(jr.normal(key, weight_shape) * jnp.sqrt(log_var) + log_mean)

    def __call__(self, weight_shape: Tuple[int, ...], bias_shape: Tuple[int, ...], key: jr.PRNGKey) -> Tuple[
        jnp.ndarray, jnp.ndarray]:
        assert bias_shape is not None, "Principled Initialisation for ICNNs requires bias parameter"
        assert len(weight_shape) == 2, "Principled Initialisation for ICNNs requires 2D weight parameters"
        w_key, b_key = jr.split(key)
        fan_in = weight_shape[1]
        weight_dist, bias_dist = self.compute_parameters(fan_in)
        weight_mean_sq, weight_var = weight_dist
        weight = self.init_log_normal(weight_shape, weight_mean_sq, weight_var, w_key)

        bias_mean, bias_var = bias_dist
        bias = jr.normal(b_key, bias_shape) * bias_var ** .5 + bias_mean

        return weight, bias

    def compute_parameters(self, fan_in: int) -> tuple[
        tuple[float, float], tuple[float, float] | None
    ]:
        """
        Compute the distribution parameters for the initialisation.

        Parameters
        ----------
        fan_in : int
            Number of incoming connections.

        Returns
        -------
        (weight_mean_sq, weight_var) : tuple of 2 float
            The squared mean and variance for weight parameters.
        (bias_mean, bias_var): tuple of 2 float, optional
            The mean and variance for the bias parameters.
            If `no_bias` is `True`, `None` is returned instead.
        """
        target_mean_sq = self.corr / self.corr_func(fan_in)
        target_variance = self.relu_scale * (1. - self.corr) / fan_in

        shift = fan_in * (target_mean_sq * self.var / (2 * math.pi)) ** .5
        bias_var = 0.
        if self.bias_noise > 0.:
            target_variance *= (1 - self.bias_noise)
            bias_var = self.bias_noise * (1. - self.corr) * self.var

        return (target_mean_sq, target_variance), (-shift, bias_var)

    def corr_func(self, fan_in: int) -> float:
        """ Helper function for correlation (cf. $f_\mathrm{c}$, eq. 35). """
        rho = self.corr
        mix_mom = (1 - rho ** 2) ** .5 + rho * math.acos(-rho)
        return fan_in * (math.pi - fan_in + (fan_in - 1) * mix_mom) / (2 * math.pi)
