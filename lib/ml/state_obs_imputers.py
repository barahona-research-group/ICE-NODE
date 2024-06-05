"""."""
from __future__ import annotations

from typing import Tuple, ClassVar, Callable, Self

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import optimistix as optx

from lib import VxData
from lib.ml.base_models import ICNNObsDecoder, ImputerMetrics


class DirectGRUStateImputer(eqx.Module):
    """Implements discrete update based on the received observations."""
    f_project_error: eqx.nn.MLP
    f_update: eqx.nn.GRUCell

    def __init__(self, state_size: int, obs_size: int,
                 key: "jax.random.PRNGKey"):
        super().__init__()
        key1, key2 = jrandom.split(key, 2)

        self.f_project_error = eqx.nn.MLP(obs_size,
                                          obs_size,
                                          width_size=obs_size * 5,
                                          depth=1,
                                          use_bias=False,
                                          activation=jnn.tanh,
                                          key=key1)
        self.f_update = eqx.nn.GRUCell(obs_size * 2,
                                       state_size,
                                       use_bias=False,
                                       key=key2)

    @eqx.filter_jit
    def __call__(self,
                 f_obs_decoder: Callable[[jnp.ndarray], jnp.ndarray],
                 forecasted_state: jnp.ndarray,
                 true_observables: jnp.ndarray, observables_mask: jnp.ndarray) -> Tuple[jnp.ndarray, ImputerMetrics]:
        error = jnp.where(observables_mask, f_obs_decoder(forecasted_state) - true_observables, 0.0)
        projected_error = self.f_project_error(error)
        return self.f_update(jnp.hstack((observables_mask, projected_error)), forecasted_state), ImputerMetrics()


class DirectGRUStateProbabilisticImputer(eqx.Module):
    f_project_error: eqx.nn.Linear
    f_update: eqx.nn.GRUCell
    f_project_error_bias: jnp.ndarray
    obs_size: int
    prep_hidden: ClassVar[int] = 4

    def __init__(self, obs_size: int, state_size: int, key: jrandom.PRNGKey):
        super().__init__()
        gru_key, prep_key = jrandom.split(key)
        self.obs_size = obs_size
        self.f_update = eqx.nn.GRUCell(self.prep_hidden * obs_size, state_size, use_bias=True, key=gru_key)
        self.f_project_error = eqx.nn.Linear(obs_size * 3, self.prep_hidden * obs_size, key=prep_key, use_bias=False)
        self.f_project_error_bias = jnp.zeros((self.obs_size, self.prep_hidden))

    @eqx.filter_jit
    def __call__(self,
                 f_obs_decoder: Callable[[jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]],
                 forecasted_state: jnp.ndarray,
                 true_observables: jnp.ndarray, observables_mask: jnp.ndarray) -> Tuple[jnp.ndarray, ImputerMetrics]:
        mean_hat, std_hat = f_obs_decoder(forecasted_state)
        # dimension: (obs_dim, )
        error = (true_observables - mean_hat) / (std_hat + 1e-6)
        gru_input = jnp.hstack([mean_hat, std_hat, error])
        gru_input = self.f_project_error(gru_input).reshape(self.obs_size, self.prep_hidden)
        gru_input = gru_input * observables_mask.reshape(-1, 1) + self.f_project_error_bias
        return self.f_update(jnp.tanh(gru_input.flatten()), forecasted_state), ImputerMetrics()


class StateObsLinearLeastSquareImpute(eqx.Module):
    # https://alexhwilliams.info/itsneuronalblog/2018/02/26/censored-lstsq/

    @staticmethod
    def censored_lstsq(A, B, M):
        """Solves least squares problem subject to missing data.

        Note: uses a broadcasted solve for speed.

        Args
        ----
        A (ndarray) : m x r matrix
        B (ndarray) : m x n matrix
        M (ndarray) : m x n binary matrix (zeros indicate missing values)

        Returns
        -------
        X (ndarray) : r x n matrix that minimizes norm(M*(AX - B))
        """

        # else solve via tensor representation
        rhs = jnp.dot(A.T, M * B).T[:, :, None]  # n x r x 1 tensor
        T = jnp.matmul(A.T[None, :, :], M.T[:, :, None] * A[None, :, :])  # n x r x r tensor
        return jnp.squeeze(jnp.linalg.solve(T, rhs)).T  # transpose to get r x n

    @eqx.filter_jit
    def __call__(self,
                 obs_decoder: eqx.nn.Linear,
                 forecasted_state: jnp.ndarray,
                 true_observables: jnp.ndarray,
                 observables_mask: jnp.ndarray) -> Tuple[jnp.ndarray, ImputerMetrics]:
        A = obs_decoder.weight
        B = jnp.expand_dims(true_observables, axis=1)
        M = jnp.expand_dims(observables_mask, axis=1)
        return self.censored_lstsq(A, B, M), ImputerMetrics()


class StateObsICNNImputer(eqx.Module):
    persistent_memory_size: int  # e.g., obs_decoder.state_size // 3

    @eqx.filter_jit
    def __call__(self,
                 obs_decoder: ICNNObsDecoder,
                 forecasted_state: jnp.ndarray,
                 true_observables: jnp.ndarray,
                 observables_mask: jnp.ndarray) -> Tuple[jnp.ndarray, ImputerMetrics]:
        # init_obs: obs_decoder(forecasted_state).
        # input: (state_mem, state_hidden, init_obs).
        # mask: (ones_like(state_mem), zeros_like(state_hidden).
        init_obs = jnp.where(observables_mask, true_observables, obs_decoder(forecasted_state))
        input = jnp.hstack((forecasted_state, init_obs))
        mask = jnp.zeros_like(input).at[:self.persistent_memory_size].set(1)
        mask = mask.at[obs_decoder.state_size:].set(observables_mask)
        output, stats = obs_decoder.partial_input_optimise(input, mask)
        state, _ = jnp.split(output, [obs_decoder.state_size])
        return state, stats


class HiddenObsICNNImputer(eqx.Module):
    persistent_memory_size: int  # e.g., obs_decoder.state_size // 3

    @eqx.filter_jit
    def __call__(self,
                 obs_decoder: ICNNObsDecoder,
                 forecasted_state: jnp.ndarray,
                 true_observables: jnp.ndarray,
                 observables_mask: jnp.ndarray) -> Tuple[jnp.ndarray, ImputerMetrics]:
        # init_obs: obs_decoder(forecasted_state).
        # input: (persistent_hidden_confounder, hidden_confounder, init_obs).
        # mask: (ones_like(state_mem), zeros_like(state_hidden).
        init_obs = jnp.where(observables_mask, true_observables, obs_decoder(forecasted_state))
        hidden_confounder = forecasted_state[:obs_decoder.state_size]
        input = jnp.hstack((hidden_confounder, init_obs))
        mask = jnp.zeros_like(input).at[:self.persistent_memory_size].set(1)
        mask = mask.at[obs_decoder.state_size:].set(observables_mask)
        return obs_decoder.partial_input_optimise(input, mask)

## TODO: use Invertible NN for embeddings: https://proceedings.mlr.press/v162/zhi22a/zhi22a.pdf
