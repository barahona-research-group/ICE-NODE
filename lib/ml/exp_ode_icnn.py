"""."""
from __future__ import annotations

import logging
from typing import Tuple, Optional, Self, Callable, Any

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from .artefacts import AdmissionsPrediction, ModelBehaviouralMetrics
from .base_models import NeuralODESolver, ODEMetrics, AutoVectorField
from .embeddings import (AdmissionEmbedding, EmbeddedAdmission)
from .embeddings import (AdmissionEmbeddingsConfig)
from .icnn_modules import ImputerMetrics, ICNNObsDecoder
from .in_models import DynamicsLiteral, ICENODEStateTrajectory, AdmissionTrajectoryPrediction, \
    GRUDynamics, DirectGRUStateImputer
from .koopman_modules import KoopmanOperator, KoopmanPrecomputes
from .model import (InpatientModel, ModelConfig,
                    Precomputes)
from ..ehr import (Admission, InpatientObservables, TVxEHR)
from ..ehr.tvx_concepts import SegmentedAdmission
from ..ehr.tvx_ehr import SegmentedTVxEHR
from ..utils import tqdm_constructor, model_params_scaler


class CompiledMLP(eqx.nn.MLP):
    # Just an eqx.nn.MLP with a compiled __call__ method.

    @eqx.filter_jit
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return super().__call__(x)


def empty_if_none(x):
    return x if x is not None else jnp.array([])


class ODEICNNConfig(ModelConfig):
    state: int = 30
    memory_ratio: float = 0.5
    dynamics: DynamicsLiteral = "mlp"
    # State: (state), state=(state_m, state_e).
    # ICNN: (state_m, state_e, obs)


class ICENODEMetrics(ModelBehaviouralMetrics):
    ode: ODEMetrics
    imputer: ImputerMetrics


class ODEICNNComponents(eqx.Module):
    emb: Optional[AdmissionEmbedding] = None
    dyn: Optional[NeuralODESolver] = None
    # All share the same ICNN:
    init: Optional[Callable] = None
    obs_dec: Optional[Callable] = None
    update: Optional[Callable] = None
    impute: Optional[Callable] = None


class AdmissionImputationAndForecasting(AdmissionTrajectoryPrediction):
    imputed_observables: InpatientObservables = None


class AutoODEICNN(InpatientModel):
    f_emb: AdmissionEmbedding
    f_dyn: NeuralODESolver
    f_icnn: ICNNObsDecoder

    config: ODEICNNConfig = eqx.static_field()

    def __init__(self, config: ODEICNNConfig,
                 observables_size: int,
                 key: "jax.random.PRNGKey"):
        super().__init__(config=config)

        (icnn_key, dyn_key) = jr.split(key, 2)
        self.f_emb = AdmissionEmbedding(config=AdmissionEmbeddingsConfig(),
                                        dx_codes_size=None,
                                        icu_inputs_grouping=None,
                                        icu_procedures_size=None,
                                        hosp_procedures_size=None,
                                        demographic_size=None,
                                        observables_size=None,
                                        key=key)
        self.f_icnn = self._make_icnn(config=config, observables_size=observables_size, key=icnn_key)
        self.f_dyn = self._make_dyn(model_config=config, observables_size=observables_size, key=dyn_key)

    @staticmethod
    def _make_dyn(model_config: ODEICNNConfig, *,
                  key: jr.PRNGKey, **kwargs) -> NeuralODESolver:

        if model_config.dynamics == "mlp":
            f_dyn = CompiledMLP(in_size=model_config.state,
                                out_size=model_config.state,
                                activation=jnn.tanh,
                                depth=2,
                                width_size=model_config.state * 5,
                                key=key)
        elif model_config.dynamics == "gru":
            f_dyn = GRUDynamics(model_config.state, model_config.state, key)
        else:
            raise ValueError(f"Unknown dynamics type: {model_config.dynamics}")
        f_dyn = model_params_scaler(f_dyn, 1e-1, eqx.is_inexact_array)
        return NeuralODESolver(AutoVectorField(f_dyn))

    @property
    def persistent_memory_size(self) -> int:
        return int(self.config.state * self.config.memory_ratio)

    @property
    def ephemeral_memory_size(self) -> int:
        return self.config.state - self.persistent_memory_size

    @eqx.filter_jit
    def _f_update(self, forecasted_state: jnp.ndarray,
                  true_observables: jnp.ndarray,
                  observables_mask: jnp.ndarray,
                  u: Optional[jnp.ndarray] = None) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], ImputerMetrics]:
        # update: keeps the memory fixed (+given obs) and tune everything else.

        # init_obs: obs_decoder(forecasted_state).
        # input: (state_mem, state_hidden, init_obs).
        # mask: (ones_like(state_mem), zeros_like(state_hidden).
        input = jnp.hstack((forecasted_state, true_observables))
        mask = jnp.hstack([jnp.ones(self.persistent_memory_size),  # Keep memory fixed.
                           jnp.zeros(self.ephemeral_memory_size),
                           observables_mask])  # Keep the given obs fixed.
        output, stats = self.f_icnn.partial_input_optimise(input, mask)
        state, imputed_obs = jnp.split(output, [self.config.state])
        return (state, imputed_obs), stats

    @eqx.filter_jit
    def _f_obs_dec(self, forecasted_state: jnp.ndarray) -> Tuple[jnp.ndarray, ImputerMetrics]:
        # update: keeps the memory and state fixed and tune everything else.
        input = jnp.hstack((forecasted_state, jnp.zeros(self.f_icnn.observables_size)))
        mask = jnp.hstack([jnp.ones(self.config.state),
                           jnp.zeros(self.f_icnn.observables_size)])

        output, stats = self.f_icnn.partial_input_optimise(input, mask)
        state, obs = jnp.split(output, [self.f_icnn.state_size])
        return obs

    @eqx.filter_jit
    def _f_init(self) -> Tuple[jnp.ndarray, ImputerMetrics]:
        # mask: (ones_like(state_mem), zeros_like(state_hidden).
        output, stats = self.f_icnn.full_optimise()
        state, _ = jnp.split(output, [self.f_icnn.state_size])
        return state

    @property
    def components(self) -> ODEICNNComponents:
        return ODEICNNComponents(emb=self.f_emb,
                                 obs_dec=self._f_obs_dec,
                                 dyn=self.f_dyn,
                                 update=self._f_update,
                                 init=self._f_init)

    @staticmethod
    def _make_icnn(config: ODEICNNConfig, observables_size: int, key) -> ICNNObsDecoder:
        return ICNNObsDecoder(observables_size=observables_size, state_size=config.state,
                              hidden_size_multiplier=3, depth=4,
                              optimiser_name='lamb',
                              max_steps=2 ** 9, lr=1e-2,
                              positivity='softplus',
                              key=key)

    @property
    def dyn_params_list(self):
        return jtu.tree_leaves(eqx.filter(self.f_dyn, eqx.is_inexact_array))

    @classmethod
    def from_tvx_ehr(cls, tvx_ehr: TVxEHR, config: ODEICNNConfig,
                     embeddings_config: AdmissionEmbeddingsConfig,
                     seed: int = 0) -> Self:
        key = jr.PRNGKey(seed)
        return cls(config=config,
                   observables_size=len(tvx_ehr.scheme.obs),
                   key=key)

    def decode_state_trajectory_observables(self,
                                            admission: SegmentedAdmission | Admission,
                                            state_trajectory: ICENODEStateTrajectory) -> InpatientObservables:
        f = self.components
        pred_obs = eqx.filter_vmap(f.obs_dec)(state_trajectory.forecasted_state)
        return InpatientObservables(time=state_trajectory.time, value=pred_obs, mask=admission.observables.mask)

    def impute_state_trajectory_observables(self,
                                            admission: SegmentedAdmission | Admission,
                                            state_trajectory: ICENODEStateTrajectory) -> InpatientObservables:
        f = self.components
        pred_obs = eqx.filter_vmap(f.obs_dec)(state_trajectory.imputed_state)
        return InpatientObservables(time=state_trajectory.time, value=pred_obs, mask=admission.observables.mask)

    def __call__(
            self, admission: SegmentedAdmission,
            embedded_admission: EmbeddedAdmission, precomputes: Precomputes, training: bool = True
    ) -> AdmissionTrajectoryPrediction:

        prediction = AdmissionImputationAndForecasting(admission=admission)

        if len(admission.observables) == 0:
            logging.debug("No observation to fit.")
            return prediction

        ode_stats = ODEMetrics()
        imputer_stats = ImputerMetrics()
        imputed_obs = ()
        t = 0.0
        state_trajectory = tuple()
        f = self.components
        state = f.init()

        for obs_t, obs_val, obs_mask in admission.observables:
            # if time-diff is more than 1 seconds, we integrate.
            forecasted_state, ode_stats_ = f.dyn(state, t0=t, t1=obs_t, precomputes=precomputes, u=jnp.array([]))
            ode_stats += ode_stats_
            forecasted_state = forecasted_state.squeeze()
            (state, imputed_obs_), imputer_stats_ = f.update(forecasted_state, obs_val, obs_mask)
            imputer_stats += imputer_stats_
            state_trajectory += ((forecasted_state, state),)
            imputed_obs += (imputed_obs_,)
            t = obs_t

        prediction = prediction.add(model_behavioural_metrics=ICENODEMetrics(ode=ode_stats, imputer=imputer_stats))
        if len(state_trajectory) > 0:
            forecasted_states, imputed_states = zip(*state_trajectory)
            state_trajectory = ICENODEStateTrajectory.compile(time=admission.observables.time,
                                                              forecasted_state=forecasted_states,
                                                              imputed_state=imputed_states)
            # TODO: test --> assert len(obs.time) == len(forecasted_states)
            prediction = prediction.add(observables=self.decode_state_trajectory_observables(
                admission=admission, state_trajectory=state_trajectory))
            if training:
                prediction = prediction.add(imputed_observables=self.impute_state_trajectory_observables(
                    admission=admission, state_trajectory=state_trajectory))

            else:
                prediction = prediction.add(imputed_observables=InpatientObservables(time=admission.observables.time,
                                                                                     value=jnp.vstack(imputed_obs),
                                                                                     mask=admission.observables.mask))

            prediction = prediction.add(trajectory=state_trajectory)
        return prediction

    def batch_predict(self, inpatients: SegmentedTVxEHR, leave_pbar: bool = False,
                      training: bool = True) -> AdmissionsPrediction:
        total_int_days = inpatients.interval_days()
        precomputes = self.precomputes(inpatients)
        admissions_emb = {
            admission.admission_id: self.f_emb(admission, inpatients.admission_demographics[admission.admission_id])
            for i, subject in tqdm_constructor(inpatients.subjects.items(),
                                               desc="Embedding",
                                               unit='subject',
                                               leave=leave_pbar) for admission in subject.admissions
        }

        r_bar = '| {n:.2f}/{total:.2f} [{elapsed}<{remaining}, ' '{rate_fmt}{postfix}]'
        bar_format = '{l_bar}{bar}' + r_bar
        with tqdm_constructor(total=total_int_days,
                              bar_format=bar_format,
                              unit='longitudinal-days',
                              leave=leave_pbar) as pbar:
            results = AdmissionsPrediction()
            for i, subject_id in enumerate(inpatients.subjects.keys()):
                pbar.set_description(
                    f"Subject: {subject_id} ({i + 1}/{len(inpatients)})")
                inpatient = inpatients.subjects[subject_id]
                for admission in inpatient.admissions:
                    results = results.add(subject_id=subject_id,
                                          prediction=self(
                                              admission,
                                              admissions_emb[admission.admission_id],
                                              precomputes=precomputes,
                                              training=training))
                    pbar.update(admission.interval_days)
            return results.filter_nans()


class KoopmanICNNConfig(ModelConfig):
    state: int = 30
    memory_ratio: float = 0.5
    koopman_size: int = 300


class AutoKoopmanICNN(AutoODEICNN):
    f_dyn: KoopmanOperator

    def _make_dyn(self, model_config: KoopmanICNNConfig, *,
                  key: jr.PRNGKey, **kwargs) -> KoopmanOperator:
        return KoopmanOperator(input_size=model_config.state,
                               koopman_size=model_config.koopman_size,
                               phi_depth=3,
                               key=key)

    def precomputes(self, *args, **kwargs):
        A, A_eig = self.f_dyn.compute_A()
        return KoopmanPrecomputes(A=A, A_eig=A_eig)

    @property
    def dyn_params_list(self):
        return jtu.tree_leaves(eqx.filter((self.f_dyn.R, self.f_dyn.Q, self.f_dyn.N), eqx.is_inexact_array))

    @eqx.filter_jit
    def pathwise_params_stats(self):
        stats = super().pathwise_params_stats()
        real_eig_A, imag_eig_A = self.f_dyn.compute_A_spectrum()
        desc = [f'f_dyn.A.lam_{i}' for i in range(real_eig_A.shape[0])]
        stats.update({
            k: {
                'real': lamr,
                'imag': lami
            }
            for k, lamr, lami in zip(desc, real_eig_A, imag_eig_A)
        })
        for component, v in (('real', real_eig_A), ('imag', imag_eig_A)):
            stats.update({
                f'f_dyn.A.{component}_lam': {
                    'mean': jnp.nanmean(v),
                    'std': jnp.nanstd(v),
                    'min': jnp.nanmin(v),
                    'max': jnp.nanmax(v),
                    'l1': jnp.abs(v).sum(),
                    'l2': jnp.square(v).sum(),
                    'nans': jnp.isnan(v).sum(),
                }
            })
        return stats


class CompiledShift(eqx.Module):
    shift: jnp.ndarray

    def __init__(self, shift: jnp.ndarray):
        self.shift = shift

    @eqx.filter_jit
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x + self.shift


class AutoICEKoopman(InpatientModel):
    f_emb: Callable[[Any, Any], None]
    f_dyn: KoopmanOperator
    f_init: CompiledShift
    f_update: DirectGRUStateImputer
    f_obs_dec: CompiledMLP

    config: KoopmanICNNConfig = eqx.static_field()

    def __init__(self, config: KoopmanICNNConfig,
                 observables_size: int,
                 key: "jax.random.PRNGKey"):
        InpatientModel.__init__(self, config=config)

        (icnn_key, dyn_key) = jr.split(key, 2)
        self.f_emb = lambda _1, _2: None
        self.f_init = self._make_init(embeddings_config=None, state_size=config.state)
        self.f_update = self._make_update(state_size=config.state, observables_size=observables_size, key=dyn_key)
        self.f_obs_dec = self._make_obs_dec(config=config, observables_size=observables_size, key=icnn_key)
        self.f_dyn = self._make_dyn(model_config=config, observables_size=observables_size, key=dyn_key)

    def _make_dyn(self, model_config: KoopmanICNNConfig, *,
                  key: jr.PRNGKey, **kwargs) -> KoopmanOperator:
        return KoopmanOperator(input_size=model_config.state,
                               koopman_size=model_config.koopman_size,
                               phi_depth=3,
                               key=key)

    @staticmethod
    def _make_init(embeddings_config: None,
                   state_size: int, **kwargs) -> CompiledShift:
        return CompiledShift(jnp.zeros(state_size))

    @eqx.filter_jit
    def _f_init(self) -> jnp.ndarray:
        return self.f_init(jnp.zeros_like(self.f_init.shift))

    @staticmethod
    def _make_update(state_size: int, observables_size: int, key: jr.PRNGKey) -> DirectGRUStateImputer:
        return DirectGRUStateImputer(state_size, observables_size, key=key)

    @staticmethod
    def _make_obs_dec(config, observables_size, key) -> CompiledMLP:
        return CompiledMLP(config.state,
                           observables_size,
                           observables_size * 5,
                           activation=jnp.tanh,
                           depth=1,
                           key=key)

    def __call__(self, admission: SegmentedAdmission,
                 admission_emb: None,
                 precomputes: Precomputes,
                 training: bool = True) -> AdmissionTrajectoryPrediction:

        prediction = AdmissionImputationAndForecasting(admission=admission)

        if len(admission.observables) == 0:
            logging.debug("No observation to fit.")
            return prediction

        ode_stats = ODEMetrics()
        imputer_stats = ImputerMetrics()
        imputed_obs = ()
        t = 0.0
        state_trajectory = tuple()
        f = self.components
        state = f.init()

        for obs_t, obs_val, obs_mask in admission.observables:
            # if time-diff is more than 1 seconds, we integrate.
            forecasted_state, ode_stats_ = f.dyn(state, t0=t, t1=obs_t, precomputes=precomputes, u=jnp.array([]))
            ode_stats += ode_stats_
            forecasted_state = forecasted_state.squeeze()
            state, imputer_stats_ = f.update(f.obs_dec, forecasted_state, obs_val, obs_mask)
            imputed_obs_ = f.obs_dec(forecasted_state)
            imputer_stats += imputer_stats_
            state_trajectory += ((forecasted_state, state),)
            imputed_obs += (imputed_obs_,)
            t = obs_t

        prediction = prediction.add(model_behavioural_metrics=ICENODEMetrics(ode=ode_stats, imputer=imputer_stats))
        if len(state_trajectory) > 0:
            forecasted_states, imputed_states = zip(*state_trajectory)
            state_trajectory = ICENODEStateTrajectory.compile(time=admission.observables.time,
                                                              forecasted_state=forecasted_states,
                                                              imputed_state=imputed_states)
            # TODO: test --> assert len(obs.time) == len(forecasted_states)
            prediction = prediction.add(observables=self.decode_state_trajectory_observables(
                admission=admission, state_trajectory=state_trajectory))
            if training:
                prediction = prediction.add(imputed_observables=self.impute_state_trajectory_observables(
                    admission=admission, state_trajectory=state_trajectory))

            else:
                prediction = prediction.add(imputed_observables=InpatientObservables(time=admission.observables.time,
                                                                                     value=jnp.vstack(imputed_obs),
                                                                                     mask=admission.observables.mask))

            prediction = prediction.add(trajectory=state_trajectory)
        return prediction

    def batch_predict(self, inpatients: SegmentedTVxEHR, leave_pbar: bool = False,
                      training: bool = True) -> AdmissionsPrediction:
        total_int_days = inpatients.interval_days()
        precomputes = self.precomputes(inpatients)
        admissions_emb = {
            admission.admission_id: self.f_emb(admission, inpatients.admission_demographics[admission.admission_id])
            for i, subject in tqdm_constructor(inpatients.subjects.items(),
                                               desc="Embedding",
                                               unit='subject',
                                               leave=leave_pbar) for admission in subject.admissions
        }

        r_bar = '| {n:.2f}/{total:.2f} [{elapsed}<{remaining}, ' '{rate_fmt}{postfix}]'
        bar_format = '{l_bar}{bar}' + r_bar
        with tqdm_constructor(total=total_int_days,
                              bar_format=bar_format,
                              unit='longitudinal-days',
                              leave=leave_pbar) as pbar:
            results = AdmissionsPrediction()
            for i, subject_id in enumerate(inpatients.subjects.keys()):
                pbar.set_description(
                    f"Subject: {subject_id} ({i + 1}/{len(inpatients)})")
                inpatient = inpatients.subjects[subject_id]
                for admission in inpatient.admissions:
                    results = results.add(subject_id=subject_id,
                                          prediction=self(
                                              admission,
                                              admissions_emb[admission.admission_id],
                                              precomputes=precomputes,
                                              training=training))
                    pbar.update(admission.interval_days)
            return results.filter_nans()

    def batch_predict(self, inpatients: SegmentedTVxEHR, leave_pbar: bool = False) -> AdmissionsPrediction:
        return AutoODEICNN.batch_predict(self, inpatients, leave_pbar=leave_pbar, training=False)

    @classmethod
    def from_tvx_ehr(cls, tvx_ehr: TVxEHR, config: KoopmanICNNConfig,
                     embeddings_config: AdmissionEmbeddingsConfig,
                     seed: int = 0) -> Self:
        key = jr.PRNGKey(seed)
        return cls(config=config,
                   observables_size=len(tvx_ehr.scheme.obs),
                   key=key)

    @property
    def components(self) -> ODEICNNComponents:
        return ODEICNNComponents(emb=lambda x: None,
                                 obs_dec=self.f_obs_dec,
                                 dyn=self.f_dyn,
                                 update=self.f_update,
                                 init=self._f_init)

    def precomputes(self, *args, **kwargs):
        A, A_eig = self.f_dyn.compute_A()
        return KoopmanPrecomputes(A=A, A_eig=A_eig)

    @property
    def dyn_params_list(self):
        return jtu.tree_leaves(eqx.filter((self.f_dyn.R, self.f_dyn.Q, self.f_dyn.N), eqx.is_inexact_array))

    @eqx.filter_jit
    def pathwise_params_stats(self):
        return AutoKoopmanICNN.pathwise_params_stats(self)
