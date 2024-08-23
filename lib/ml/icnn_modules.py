import math
import time
import zipfile
from abc import abstractmethod
from typing import Tuple, Optional, Literal, Callable, Self, cast, Dict

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import lineax
import numpy as np
import optax
import optimistix as optx

from .. import VxData, Config
from ..metric.loss import log_normal, gaussian_kl, mse
from ..utils import tqdm_constructor, translate_path


class ConvexInitialiser(eqx.Module):
    """https://github.com/ml-jku/convex-init"""

    var: float = 1.0
    corr: float = 0.5
    bias_noise: float = 0.0
    alpha: float = 0.0
    relu_scale: float = eqx.field(init=False)

    def __post_init__(self):
        self.relu_scale = 2. / (1. + self.alpha ** 2)

    @staticmethod
    def init_log_normal(weight_shape: Tuple[int, ...], mean_sq: float, var: float, key: jr.PRNGKey) -> jnp.ndarray:
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

    def compute_parameters(self, fan_in: int) -> tuple[tuple[float, float], tuple[float, float] | None]:
        target_mean_sq = self.corr / self.corr_func(fan_in)
        target_variance = self.relu_scale * (1. - self.corr) / fan_in

        shift = fan_in * (target_mean_sq * self.var / (2 * math.pi)) ** .5
        bias_var = 0.
        if self.bias_noise > 0.:
            target_variance *= (1 - self.bias_noise)
            bias_var = self.bias_noise * (1. - self.corr) * self.var

        return (target_mean_sq, target_variance), (-shift, bias_var)

    def corr_func(self, fan_in: int) -> float:
        rho = self.corr
        mix_mom = (1 - rho ** 2) ** .5 + rho * math.acos(-rho)
        return fan_in * (math.pi - fan_in + (fan_in - 1) * mix_mom) / (2 * math.pi)


class PositivityLayer(eqx.nn.Linear):

    def __init__(self, in_size: int, out_size: int, use_bias: bool = True, key: jr.PRNGKey = None):
        super().__init__(in_size, out_size, use_bias=use_bias, key=key)
        self.weight, self.bias = self.re_init_params(self.weight, self.bias, key)

    @staticmethod
    def re_init_params(weight: jnp.ndarray, bias: jnp.ndarray, key: jr.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return weight, bias

    @abstractmethod
    def transform(self, x: jnp.ndarray) -> jnp.ndarray:
        pass

    def __call__(self, x: jnp.ndarray, *, key: Optional[jr.PRNGKey] = None) -> jnp.ndarray:
        w = self.transform(self.weight)
        y = w @ x
        if self.bias is not None:
            y += self.bias
        return y


class PositiveSquaredLinear(PositivityLayer):
    def transform(self, x: jnp.ndarray) -> jnp.ndarray:
        return x ** 2


class PositiveReLuLinear(PositivityLayer):
    @staticmethod
    def re_init_params(weight: jnp.ndarray, bias: jnp.ndarray, key: jr.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return ConvexInitialiser()(weight.shape, bias.shape, key)

    def transform(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.nn.relu(x)


class PositiveAbsLinear(PositivityLayer):

    @staticmethod
    def re_init_params(weight: jnp.ndarray, bias: jnp.ndarray, key: jr.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return ConvexInitialiser()(weight.shape, bias.shape, key)

    def transform(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.abs(x)


class ICNN(eqx.Module):
    """Input Convex Neural Network"""
    """https://github.com/atong01/ot-icnn-minimal/blob/main/icnn/icnn.py
    Principled Weight Initialisation for Input-Convex Neural Networks: https://openreview.net/pdf?id=pWZ97hUQtQ 
    """
    Wzs: Tuple[PositivityLayer, ...]
    Wxs: Tuple[eqx.nn.Linear, ...]
    activations: Tuple[Callable[..., jnp.ndarray], ...]
    input_size: int = eqx.field(init=False)

    def __init__(self, input_size: int, hidden_size: int, depth: int, positivity: Literal['abs', 'squared'],
                 key: jr.PRNGKey):
        super().__init__()

        def new_key():
            nonlocal key
            key, subkey = jr.split(key)
            return subkey

        if positivity == 'squared':
            PositivityLayer = PositiveSquaredLinear
        elif positivity == 'abs':
            PositivityLayer = PositiveAbsLinear
        else:
            raise ValueError(f"Unknown positivity parameter: {positivity}")

        Wzs = [eqx.nn.Linear(input_size, hidden_size, key=new_key())]
        for _ in range(depth - 1):
            Wzs.append(PositivityLayer(hidden_size, hidden_size, use_bias=True, key=new_key()))
        Wzs.append(PositivityLayer(hidden_size, 1, use_bias=True, key=new_key()))
        self.Wzs = tuple(Wzs)

        Wxs = []
        for _ in range(depth - 1):
            Wxs.append(eqx.nn.Linear(input_size, hidden_size, key=new_key()))
        Wxs.append(eqx.nn.Linear(input_size, 1, use_bias=True, key=new_key()))
        self.Wxs = tuple(Wxs)
        self.activations = tuple(jnn.softplus for _ in range(depth))
        self.input_size = input_size

    @eqx.filter_jit
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray | float:
        # https://arxiv.org/pdf/1609.07152 (Eq (2)). Fully input convex.
        z = jnn.softplus(self.Wzs[0](x))
        for Wz, Wx, sigma in zip(self.Wzs[1:-1], self.Wxs[:-1], self.activations):
            z = sigma(Wz(z) + Wx(x))
        return self.activations[-1](self.Wzs[-1](z) + self.Wxs[-1](x)).squeeze()


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
        updates, new_opt_state = self.optim.update(grads, state.opt_state, params=y,
                                                   value=f, grad=grads,
                                                   value_fn=lambda yi: fn(yi, args)[0])
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
    energy: jnp.ndarray = eqx.field(default_factory=lambda: jnp.array([]))

    def __add__(self, other: Self) -> Self:
        return ImputerMetrics(n_steps=jnp.hstack((self.n_steps, other.n_steps)),
                              energy=jnp.hstack((self.energy, other.energy)))


class ICNNObsDecoder(eqx.Module):
    f_energy: ICNN
    observables_size: int
    state_size: int
    optimiser_name: Literal['adam', 'polyak_sgd', 'lamb', 'yogi', 'bfgs', 'nonlinear_cg']
    max_steps: int
    lr: float

    def __init__(self, observables_size: int, state_size: int, hidden_size_multiplier: float,
                 depth: int,
                 positivity: Literal['abs', 'squared'] = 'abs',
                 optimiser_name: Literal['adam', 'polyak_sgd', 'lamb', 'yogi', 'bfgs', 'nonlinear_cg'] = 'lamb',
                 max_steps: int = 2 ** 9,
                 lr: float = 1e-2, *,
                 key: jr.PRNGKey):
        super().__init__()
        self.observables_size = observables_size
        self.state_size = state_size
        input_size = observables_size + state_size
        self.f_energy = ICNN(input_size, int(input_size * hidden_size_multiplier), depth, positivity, key)
        self.optimiser_name = optimiser_name
        self.max_steps = max_steps
        self.lr = lr

    @eqx.filter_jit
    def partial_input_optimise(self, input: jnp.ndarray, fixed_mask: jnp.ndarray) -> Tuple[jnp.ndarray, ImputerMetrics]:
        sol = optx.minimise(lambda y, args: self.f_energy(y),
                            solver=self.solver(self.optimiser_name),
                            adjoint=optx.ImplicitAdjoint(linear_solver=lineax.AutoLinearSolver(well_posed=False)),
                            max_steps=self.max_steps,
                            options=dict(fixed_mask=fixed_mask),
                            y0=input, throw=False)
        return sol.value, ImputerMetrics(n_steps=sol.stats['num_steps'],
                                         energy=self.f_energy(sol.value))

    @eqx.filter_jit
    def full_optimise(self):
        return self.partial_input_optimise(jnp.zeros(self.observables_size + self.state_size),
                                           jnp.zeros(self.observables_size + self.state_size))

    def solver(self, solver_name: str):
        if solver_name in ['adam', 'polyak_sgd', 'lamb', 'yogi']:
            return optx.BestSoFarMinimiser(solver=self.optax_solver(solver_name, self.lr))
        elif solver_name == 'bfgs':
            return optx.BestSoFarMinimiser(solver=MaskedBFGS(rtol=1e-8, atol=1e-8))
        elif solver_name == 'nonlinear_cg':
            return optx.BestSoFarMinimiser(solver=MaskedNonlinearCG(rtol=1e-8, atol=1e-8))
        else:
            raise ValueError(f'Unknown solver name {solver_name}')

    @staticmethod
    def optax_solver(optax_optimiser_name: str, lr: float) -> MaskedOptaxMinimiser:
        optimiser = ICNNObsDecoder.optax_solver_from_name(optax_optimiser_name)
        return MaskedOptaxMinimiser(optimiser(lr), rtol=1e-8, atol=1e-8)

    @staticmethod
    def optax_solver_from_name(optax_solver_name: str) -> Callable[[float], optax.GradientTransformation]:
        return {
            'adam': optax.adam,
            'polyak_sgd': optax.polyak_sgd,
            'novograd': optax.novograd,
            'lamb': optax.lamb,
            'yogi': optax.yogi,
        }[optax_solver_name]

    @eqx.filter_jit
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        input = jnp.hstack((state, jnp.zeros(self.observables_size)))
        mask = jnp.zeros_like(input).at[:self.state_size].set(1)
        output, _ = self.partial_input_optimise(input, mask)
        _, obs = jnp.split(output, [self.state_size])
        return obs

    def load_params(self, params_file):
        """
        Load the parameters in `params_file`\
            filepath and return as PyTree Object.
        """
        with open(translate_path(params_file), 'rb') as file_rsc:
            return eqx.tree_deserialise_leaves(file_rsc, self)

    def write_params(self, params_file):
        """
        Store the parameters (PyTree object) into a new file
        given by `params_file`.
        """
        with open(translate_path(params_file), 'wb') as file_rsc:
            eqx.tree_serialise_leaves(file_rsc, self)

    def load_params_from_archive(self, zipfile_fname: str, params_fname: str):
        with zipfile.ZipFile(translate_path(zipfile_fname),
                             compression=zipfile.ZIP_STORED,
                             mode="r") as archive:
            with archive.open(params_fname, "r") as zip_member:
                return eqx.tree_deserialise_leaves(zip_member, self)


class ICNNObsExtractor(ICNNObsDecoder):

    @eqx.filter_jit
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.split(x, [self.state_size])[1]


class ICNNImputerConfig(Config):
    model_type: Literal['staged', 'stacked', 'standard']
    observables_size: int
    state_size: int
    hidden_size_multiplier: float
    depth: int
    positivity: Literal['abs', 'squared']
    optimiser_name: Literal['adam', 'polyak_sgd', 'lamb', 'yogi', 'bfgs', 'nonlinear_cg']
    optimiser_lr: float
    optimiser_max_steps: int


class ProbStagedICNNImputer(eqx.Module):
    icnn_mean: ICNNObsDecoder
    icnn_var: ICNNObsDecoder

    def __init__(self, observables_size: int, state_size: int, hidden_size_multiplier: float,
                 depth: int,
                 positivity: Literal['abs', 'squared'] = 'abs',
                 optimiser_name: Literal['adam', 'polyak_sgd', 'lamb', 'yogi', 'bfgs', 'nonlinear_cg'] = 'lamb',
                 max_steps: int = 2 ** 9,
                 lr: float = 1e-2,
                 *,
                 key: jr.PRNGKey):
        key_mu, key_sigma = jr.split(key, 2)
        self.icnn_mean = ICNNObsDecoder(observables_size, state_size, hidden_size_multiplier, depth, positivity,
                                        optimiser_name,
                                        max_steps=max_steps, lr=lr,
                                        key=key_mu)
        self.icnn_var = ICNNObsDecoder(observables_size * 2, state_size, hidden_size_multiplier // 2, depth, positivity,
                                       optimiser_name,
                                       max_steps=max_steps, lr=lr,
                                       key=key_sigma)

    @eqx.filter_jit
    def prob_partial_input_optimise(self, input: jnp.ndarray, fixed_mask: jnp.ndarray) -> Tuple[
        Tuple[jnp.ndarray, jnp.ndarray], ImputerMetrics]:
        mu, metrics = self.icnn_mean.partial_input_optimise(input, fixed_mask)

        mu_std, _ = self.icnn_var.partial_input_optimise(jnp.hstack((mu, jnp.where(fixed_mask, -4., 10.))),
                                                         jnp.hstack((jnp.ones_like(mu), fixed_mask)))
        mu, std = jnp.hsplit(mu_std, 2)
        std = jnn.softplus(std)
        return (mu, std), metrics

    @eqx.filter_jit
    def partial_input_optimise(self, input: jnp.ndarray, fixed_mask: jnp.ndarray, **kwargs) -> Tuple[
        jnp.ndarray, ImputerMetrics]:
        (mu, _), metrics = self.prob_partial_input_optimise(input, fixed_mask, **kwargs)
        return mu, metrics


class ProbStackedICNNImputer(ICNNObsDecoder):
    f_energy: ICNN

    def __init__(self, observables_size: int, state_size: int, hidden_size_multiplier: float, depth: int,
                 positivity: Literal['abs', 'squared'] = 'abs',
                 optimiser_name: Literal['adam', 'polyak_sgd', 'lamb', 'yogi', 'bfgs', 'nonlinear_cg'] = 'adam',
                 max_steps: int = 2 ** 9, lr: float = 1e-2,
                 *,
                 key: jr.PRNGKey):
        super().__init__(observables_size=observables_size * 2, state_size=state_size,
                         hidden_size_multiplier=hidden_size_multiplier,
                         depth=depth, positivity=positivity, optimiser_name=optimiser_name,
                         max_steps=max_steps, lr=lr,
                         key=key)

    @eqx.filter_jit
    def prob_partial_input_optimise(self, input: jnp.ndarray, fixed_mask: jnp.ndarray) -> Tuple[
        Tuple[jnp.ndarray, jnp.ndarray], ImputerMetrics]:
        mu_std, metrics = super().partial_input_optimise(jnp.hstack((input, jnp.where(fixed_mask, -4., 10.))),
                                                         jnp.hstack((fixed_mask, fixed_mask)))
        mu, std = jnp.hsplit(mu_std, 2)
        std = jnn.softplus(std) + jnn.softplus(-3)
        return (mu, std), metrics

    @eqx.filter_jit
    def energy0(self, input: jnp.ndarray, fixed_mask: jnp.ndarray) -> jnp.ndarray:
        return self.f_energy(jnp.hstack((input, jnp.where(fixed_mask, -4., 10.))))

    @eqx.filter_jit
    def partial_input_optimise(self, input: jnp.ndarray, fixed_mask: jnp.ndarray) -> Tuple[
        jnp.ndarray, ImputerMetrics]:
        (mu, _), metrics = self.prob_partial_input_optimise(input, fixed_mask)
        return mu, metrics


class ProbICNNImputerTrainer(eqx.Module):
    # Config
    prob_loss_name: Literal['log_normal', 'kl_divergence'] = 'log_normal'
    optimiser_name: Literal['adam', 'novograd'] = 'adam'
    loss_feature_normalisation: bool = False
    lr: float = 1e-3
    steps: int = 1000000
    train_batch_size: int = 256
    seed: int = 0
    model_snapshot_frequency: int = 100
    artificial_missingness: float = 0.8
    model_config: Optional[ICNNImputerConfig] = None
    loss_function: Callable = None
    # State
    model: Optional[ProbStackedICNNImputer | ProbStagedICNNImputer] = None
    model_snapshots: Dict[int, ProbStackedICNNImputer | ProbStagedICNNImputer] = eqx.field(default_factory=dict)
    train_history: Tuple[Dict[str, float], ...] = eqx.field(default_factory=tuple)

    def __init__(self, icnn_hidden_size_multiplier: float = 3,
                 icnn_depth: int = 4,
                 icnn_model_name: Literal['staged', 'stacked'] = 'stacked',
                 icnn_positivity: Literal['abs', 'squared'] = 'abs',
                 icnn_optimiser: Literal['adam', 'polyak_sgd', 'lamb', 'yogi', 'bfgs', 'nonlinear_cg'] = 'lamb',
                 icnn_max_steps: int = 2 ** 9,
                 icnn_lr: float = 1e-2,
                 loss: Literal['log_normal', 'kl_divergence', 'jsd_gaussian'] = 'log_normal',
                 loss_feature_normalisation: bool = False,
                 optimiser_name: Literal['adam', 'novograd', 'lamb'] = 'adam',
                 lr: float = 1e-3, steps: int = 1000000, train_batch_size: int = 256, seed: int = 0,
                 model_snapshot_frequency: int = 100, artificial_missingness: float = 0.5):
        self.prob_loss_name = loss
        self.optimiser_name = optimiser_name
        self.lr = lr
        self.steps = steps
        self.train_batch_size = train_batch_size
        self.seed = seed
        self.model_snapshot_frequency = model_snapshot_frequency
        self.artificial_missingness = artificial_missingness
        self.model_config = ICNNImputerConfig(observables_size=0,
                                              model_type=icnn_model_name,
                                              state_size=0,
                                              hidden_size_multiplier=icnn_hidden_size_multiplier,
                                              depth=icnn_depth,
                                              positivity=icnn_positivity,
                                              optimiser_name=icnn_optimiser,
                                              optimiser_lr=icnn_lr,
                                              optimiser_max_steps=icnn_max_steps)

        self.model = None
        self.model_snapshots = {}
        self.train_history = ()
        self.loss_feature_normalisation = loss_feature_normalisation
        if loss == 'kl_divergence':
            self.loss_function = gaussian_kl
        else:
            self.loss_function = log_normal

    def init_model(self, X: jnp.ndarray) -> ProbStackedICNNImputer | ProbStagedICNNImputer:
        model_cls = ProbStagedICNNImputer if self.model_config.model_type == 'staged' else ProbStackedICNNImputer
        return model_cls(observables_size=X.shape[1],
                         state_size=self.model_config.state_size,
                         hidden_size_multiplier=self.model_config.hidden_size_multiplier,
                         depth=self.model_config.depth,
                         positivity=self.model_config.positivity,
                         optimiser_name=self.model_config.optimiser_name,
                         max_steps=self.model_config.optimiser_max_steps,
                         lr=self.model_config.optimiser_lr,
                         key=jr.PRNGKey(self.seed))

    @eqx.filter_jit
    def loss(self, model: ProbStackedICNNImputer | ProbStagedICNNImputer,
             batch_X: jnp.ndarray, batch_M: jnp.ndarray,
             batch_M_art: jnp.ndarray) -> Tuple[jnp.ndarray, ImputerMetrics]:
        # Zero for artificially missig values
        batch_X_art = jnp.where(batch_M_art, batch_X, 0.)
        # Tune for artificially masked-out values, fix mask-in (batch_M_art) values.
        (X_imp, std_imp), aux = eqx.filter_vmap(model.prob_partial_input_optimise)(batch_X_art, batch_M_art)
        # Penalise discrepancy with artifially masked-out values.
        mask = (1 - batch_M_art) * batch_M
        # Compute loss
        batch_std = jnp.zeros_like(batch_X) + 0.01
        if self.loss_feature_normalisation:
            L_norm = self.loss_function((batch_X, batch_std), (X_imp, std_imp), mask, axis=0)
            L = jnp.nanmean(L_norm)
        else:
            L = self.loss_function((batch_X, batch_std), (X_imp, std_imp), mask)
        return jnp.where(jnp.isnan(L), 0., L), aux

    @staticmethod
    def r_squared(y: jnp.ndarray, y_hat: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        y = y.squeeze()
        y_hat = y_hat.squeeze()
        mask = mask.squeeze()

        y_bar = jnp.nanmean(y, where=mask)
        ss_tot = jnp.nansum((y - y_bar) ** 2, where=mask)
        ss_res = jnp.nansum((y - y_hat) ** 2, where=mask)

        return jnp.where(mask.sum() > 1, 1 - (ss_res / ss_tot), jnp.nan)

    @staticmethod
    def r_squared_micro_average(y: jnp.ndarray, y_hat: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        y_bar = jnp.nanmean(y, where=mask, axis=0, keepdims=True)
        ss_tot = jnp.nansum((y - y_bar) ** 2, where=mask, axis=0)
        ss_res = jnp.nansum((y - y_hat) ** 2, where=mask, axis=0)

        return jnp.where(mask.sum() > 1, 1 - (np.nansum(ss_res) / np.nansum(ss_tot)), jnp.nan)

    @staticmethod
    @eqx.filter_jit
    def r_squared_ranked_prob(y: jnp.ndarray, y_hat: jnp.ndarray, mask: jnp.ndarray, sigma: jnp.ndarray,
                              k: int) -> jnp.ndarray:
        sigma = jnp.where(mask, sigma, jnp.inf)
        sigma_sorter = jnp.argpartition(sigma, k, axis=0)[:k]
        y = jnp.take_along_axis(y, sigma_sorter, axis=0)
        y_hat = jnp.take_along_axis(y_hat, sigma_sorter, axis=0)
        mask = jnp.take_along_axis(mask, sigma_sorter, axis=0)
        return jnp.where(jnp.all(mask), ProbICNNImputerTrainer.r_squared(y, y_hat, mask), jnp.nan)

    @staticmethod
    @eqx.filter_jit
    def r_squared_thresholded_prob(y: jnp.ndarray, y_hat: jnp.ndarray, mask: jnp.ndarray, sigma: jnp.ndarray,
                                   thresh: float) -> jnp.ndarray:
        sigma = jnp.where(mask, sigma, jnp.inf)
        mask = mask * (sigma < thresh)
        return ProbICNNImputerTrainer.r_squared(y, y_hat, mask)

    @staticmethod
    @eqx.filter_jit
    def model_r_squared(model: ICNNObsDecoder, batch_X: jnp.ndarray, batch_M: jnp.ndarray,
                        batch_M_art: jnp.ndarray) -> jnp.ndarray:
        # Zero for artificially missig values
        batch_X_art = jnp.where(batch_M_art, batch_X, 0.)
        # Tune for artificially masked-out values, fix mask-in (batch_M_art) values.
        X_imp, aux = eqx.filter_vmap(model.partial_input_optimise)(batch_X_art, batch_M_art)
        # Penalise discrepancy with artifially masked-out values.
        mask = (1 - batch_M_art) * batch_M
        r2_vec = eqx.filter_vmap(ProbICNNImputerTrainer.r_squared)(batch_X.T, X_imp.T, mask.T)
        return r2_vec

    @eqx.filter_jit
    def model_r_squared_ranked_prob(self, model: ProbStackedICNNImputer, batch_X: jnp.ndarray, batch_M: jnp.ndarray,
                                    batch_M_art: jnp.ndarray, k: int) -> jnp.ndarray:
        mask = (1 - batch_M_art) * batch_M
        # Zero for artificially missing values
        batch_X_art = jnp.where(batch_M_art, batch_X, 0.)
        # Tune for artificially masked-out values, fix mask-in (batch_M_art) values.
        (X_imp, std), _ = eqx.filter_vmap(model.prob_partial_input_optimise)(batch_X_art, batch_M_art)
        r2_vec = eqx.filter_vmap(self.r_squared_ranked_prob)(batch_X.T, X_imp.T, mask.T, std.T, k)
        return r2_vec

    @eqx.filter_jit
    def make_step(self, model: ProbStackedICNNImputer, optim, opt_state, batch_X: jnp.ndarray, batch_M: jnp.ndarray,
                  batch_M_art: jnp.ndarray):
        (loss, aux), grads = eqx.filter_value_and_grad(self.loss, has_aux=True)(model, batch_X, batch_M,
                                                                                batch_M_art)
        updates, opt_state = optim.update(grads, opt_state,
                                          params=eqx.filter(model, eqx.is_inexact_array),
                                          value=loss, grad=grads,
                                          value_fn=lambda m: self.loss(eqx.combine(m, model), batch_X,
                                                                       batch_M,
                                                                       batch_M_art))

        model = eqx.apply_updates(model, updates)
        return (loss, aux), model, opt_state

    @staticmethod
    def dataloader(arrays, batch_size, *, key):
        dataset_size = arrays[0].shape[0]
        indices = jnp.arange(dataset_size)
        while True:
            perm = jr.permutation(key, indices)
            (key,) = jr.split(key, 1)
            start = 0
            end = batch_size
            while end < dataset_size:
                batch_perm = perm[start:end]
                yield tuple(array[batch_perm] for array in arrays)
                start = end
                end = start + batch_size

    def new_optimiser(self):
        return {
            'adam': optax.adam(self.lr),
            'novograd': optax.novograd(self.lr),
            'lamb': optax.lamb(self.lr)
        }[self.optimiser_name]

    def fit(self, X: jnp.ndarray) -> Self:
        model = self.init_model(X)
        X = jnp.array(X)
        train_M = jnp.where(jnp.isnan(X), 0., 1.)
        train_X = jnp.where(jnp.isnan(X), 0., X)
        train_M_art = jr.bernoulli(jr.PRNGKey(self.seed), p=self.artificial_missingness, shape=train_M.shape) * train_M
        train_loader = self.dataloader((train_X, train_M, train_M_art), self.train_batch_size,
                                       key=jr.PRNGKey(self.seed))
        optim = self.new_optimiser()
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        train_history = []
        model_snapshots = []

        with tqdm_constructor(range(self.steps)) as pbar:
            for step, (batch_X, batch_M, batch_M_art) in zip(pbar, train_loader):
                start = time.time()
                (train_loss, aux), model, opt_state = self.make_step(model, optim, opt_state, batch_X, batch_M,
                                                                     batch_M_art)
                r2_vec = self.model_r_squared(model, batch_X, batch_M, batch_M_art)
                r2_vec_rank = self.model_r_squared_ranked_prob(model, batch_X, batch_M, batch_M_art, k=10)
                r2_vec = np.array(r2_vec)
                train_n_steps = int(sum(aux.n_steps) / len(aux.n_steps))
                train_history.append({'R2': r2_vec,
                                      'R2_rank10': r2_vec_rank,
                                      'loss': train_loss,
                                      'n_opt_steps': train_n_steps})
                end = time.time()

                if (step % self.model_snapshot_frequency) == 0 or step == self.steps - 1:
                    model_snapshots.append(model)

                pbar.set_description(
                    f"Trn-L: {train_loss:.3f}, Trn-R2: ({np.nanmax(r2_vec_rank):.2f}, {np.nanmin(r2_vec_rank):.2f}, {np.nanmean(r2_vec_rank):.2f}, {np.nanmedian(r2_vec_rank):.2f}),  Trn-N-steps: {train_n_steps}, "
                    f"Computation time: {end - start:.2f}, ")
        this = self
        this = eqx.tree_at(lambda x: x.model, this, model, is_leaf=lambda x: x is None)
        this = eqx.tree_at(lambda x: x.train_history, this, tuple(train_history), is_leaf=lambda x: x is None)
        this = eqx.tree_at(lambda x: x.model_snapshots, this, tuple(model_snapshots), is_leaf=lambda x: x is None)
        return this

    def transform(self, X: jnp.ndarray) -> jnp.ndarray:
        X = jnp.array(X)
        M = jnp.where(jnp.isnan(X), 0., 1.)
        X = jnp.where(jnp.isnan(X), 0., X)
        X_imp, _ = eqx.filter_vmap(self.model.partial_input_optimise)(X, M)
        return X_imp


class StandardICNNImputerTrainer(ProbICNNImputerTrainer):
    prob_loss_name: Literal['mse']

    def __init__(self, icnn_hidden_size_multiplier: float = 3,
                 icnn_depth: int = 4,
                 icnn_positivity: Literal['abs', 'squared'] = 'abs',
                 icnn_optimiser: Literal['adam', 'polyak_sgd', 'lamb', 'yogi'] = 'adam',
                 icnn_max_steps: int = 2 ** 9,
                 icnn_lr: float = 1e-2,
                 loss_feature_normalisation: bool = False,
                 optimiser_name: Literal['adam', 'novograd', 'lamb'] = 'adam',
                 lr: float = 1e-3, steps: int = 1000000, train_batch_size: int = 256, seed: int = 0,
                 model_snapshot_frequency: int = 100, artificial_missingness: float = 0.5):
        super().__init__(icnn_hidden_size_multiplier=icnn_hidden_size_multiplier,
                         icnn_depth=icnn_depth,
                         icnn_positivity=icnn_positivity,
                         icnn_optimiser=icnn_optimiser,
                         icnn_max_steps=icnn_max_steps,
                         icnn_lr=icnn_lr,
                         loss_feature_normalisation=loss_feature_normalisation,
                         optimiser_name=optimiser_name,
                         lr=lr, steps=steps, train_batch_size=train_batch_size, seed=seed,
                         model_snapshot_frequency=model_snapshot_frequency,
                         artificial_missingness=artificial_missingness)
        self.prob_loss_name = 'mse'
        self.model_config = ICNNImputerConfig(observables_size=0,
                                              model_type='standard',
                                              state_size=0,
                                              hidden_size_multiplier=icnn_hidden_size_multiplier,
                                              depth=icnn_depth,
                                              positivity=icnn_positivity,
                                              optimiser_name=icnn_optimiser,
                                              optimiser_lr=icnn_lr,
                                              optimiser_max_steps=icnn_max_steps)
        self.model = None
        self.model_snapshots = {}
        self.train_history = ()
        self.loss_feature_normalisation = loss_feature_normalisation
        self.loss_function = mse

    def init_model(self, X: jnp.ndarray) -> ICNNObsDecoder:
        return ICNNObsDecoder(observables_size=X.shape[1],
                              state_size=self.model_config.state_size,
                              hidden_size_multiplier=self.model_config.hidden_size_multiplier,
                              depth=self.model_config.depth,
                              positivity=self.model_config.positivity,
                              optimiser_name=self.model_config.optimiser_name,
                              lr=self.model_config.optimiser_lr,
                              max_steps=self.model_config.optimiser_max_steps,
                              key=jr.PRNGKey(self.seed))

    @eqx.filter_jit
    def loss(self, model: ICNNObsDecoder,
             batch_X: jnp.ndarray, batch_M: jnp.ndarray,
             batch_M_art: jnp.ndarray) -> Tuple[jnp.ndarray, ImputerMetrics]:
        # Zero for artificially missig values
        batch_X_art = jnp.where(batch_M_art, batch_X, 0.)
        # Tune for artificially masked-out values, fix mask-in (batch_M_art) values.
        X_imp, aux = eqx.filter_vmap(model.partial_input_optimise)(batch_X_art, batch_M_art)
        # Penalise discrepancy with artifially masked-out values.
        mask = (1 - batch_M_art) * batch_M
        # Compute loss
        if self.loss_feature_normalisation:
            L_norm = self.loss_function(batch_X, X_imp, mask, axis=0)
            L = jnp.nanmean(L_norm)
        else:
            L = self.loss_function(batch_X, X_imp, mask)
        return jnp.where(jnp.isnan(L), 0., L), aux

    def fit(self, X: jnp.ndarray) -> Self:
        model = self.init_model(X)
        X = jnp.array(X)
        train_M = jnp.where(jnp.isnan(X), 0., 1.)
        train_X = jnp.where(jnp.isnan(X), 0., X)
        train_M_art = jr.bernoulli(jr.PRNGKey(self.seed), p=self.artificial_missingness, shape=train_M.shape) * train_M
        train_loader = self.dataloader((train_X, train_M, train_M_art), self.train_batch_size,
                                       key=jr.PRNGKey(self.seed))
        optim = optax.adam(self.lr) if self.trainer_name == 'adam' else optax.novograd(self.lr)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        train_history = []
        model_snapshots = []

        with tqdm_constructor(range(self.steps)) as pbar:
            for step, (batch_X, batch_M, batch_M_art) in zip(pbar, train_loader):
                start = time.time()
                (train_loss, aux), model, opt_state = self.make_step(model, optim, opt_state, batch_X, batch_M,
                                                                     batch_M_art)
                r2_vec = self.model_r_squared(model, batch_X, batch_M, batch_M_art)
                r2_vec = np.array(r2_vec)
                train_n_steps = int(sum(aux.n_steps) / len(aux.n_steps))
                train_history.append({'R2': r2_vec,
                                      'loss': train_loss,
                                      'n_opt_steps': train_n_steps})
                end = time.time()

                if (step % self.model_snapshot_frequency) == 0 or step == self.steps - 1:
                    model_snapshots.append(model)

                pbar.set_description(
                    f"Trn-L: {train_loss:.3f}, Trn-R2: ({np.nanmax(r2_vec):.2f}, {np.nanmin(r2_vec):.2f}, {np.nanmean(r2_vec):.2f}, {np.nanmedian(r2_vec):.2f}),  Trn-N-steps: {train_n_steps}, "
                    f"Computation time: {end - start:.2f}, ")
        this = self
        this = eqx.tree_at(lambda x: x.model, this, model, is_leaf=lambda x: x is None)
        this = eqx.tree_at(lambda x: x.train_history, this, tuple(train_history), is_leaf=lambda x: x is None)
        this = eqx.tree_at(lambda x: x.model_snapshots, this, tuple(model_snapshots), is_leaf=lambda x: x is None)
        return this

# class StateObsLinearLeastSquareImpute(eqx.Module):
#     # https://alexhwilliams.info/itsneuronalblog/2018/02/26/censored-lstsq/
#
#     @staticmethod
#     def censored_lstsq(A, B, M):
#         """Solves least squares problem subject to missing data.
#
#         Note: uses a broadcasted solve for speed.
#
#         Args
#         ----
#         A (ndarray) : m x r matrix
#         B (ndarray) : m x n matrix
#         M (ndarray) : m x n binary matrix (zeros indicate missing values)
#
#         Returns
#         -------
#         X (ndarray) : r x n matrix that minimizes norm(M*(AX - B))
#         """
#
#         # else solve via tensor representation
#         rhs = jnp.dot(A.T, M * B).T[:, :, None]  # n x r x 1 tensor
#         T = jnp.matmul(A.T[None, :, :], M.T[:, :, None] * A[None, :, :])  # n x r x r tensor
#         return jnp.squeeze(jnp.linalg.solve(T, rhs)).T  # transpose to get r x n
#
#     @eqx.filter_jit
#     def __call__(self,
#                  obs_decoder: eqx.nn.Linear,
#                  forecasted_state: jnp.ndarray,
#                  true_observables: jnp.ndarray,
#                  observables_mask: jnp.ndarray,
#                  u: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, ImputerMetrics]:
#         A = obs_decoder.weight
#         B = jnp.expand_dims(true_observables, axis=1)
#         M = jnp.expand_dims(observables_mask, axis=1)
#         return self.censored_lstsq(A, B, M), ImputerMetrics()
#
#
#


## TODO: use Invertible NN for embeddings: https://proceedings.mlr.press/v162/zhi22a/zhi22a.pdf
