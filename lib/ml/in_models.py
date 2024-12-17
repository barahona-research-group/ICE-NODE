"""."""
from __future__ import annotations

import logging
from typing import Tuple, Optional, Literal, Type, Self, Dict, Any, ClassVar, Callable

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu

from .artefacts import AdmissionPrediction, AdmissionsPrediction, ModelBehaviouralMetrics
from .base_models import LeadPredictorName, NeuralODESolver, DiffusionMLP, StochasticNeuralODESolver, \
    SkipShortIntervalsWrapper, ODEMetrics
from .embeddings import (AdmissionEmbedding, EmbeddedAdmission)
from .embeddings import (AdmissionSequentialEmbeddingsConfig, AdmissionSequentialObsEmbedding,
                         AdmissionEmbeddingsConfig, EmbeddedAdmissionObsSequence)
from .embeddings import (DischargeSummarySequentialEmbeddingsConfig, EmbeddedDischargeSummary)
from .icnn_modules import ImputerMetrics, ICNNObsExtractor, ICNNObsDecoder
from .koopman_modules import KoopmanOperator, KoopmanPrecomputes
from .model import (InpatientModel, ModelConfig,
                    Precomputes)
from .rectilinear_modules import RectilinearImputer
from ..ehr import (Admission, InpatientObservables, CodesVector, TVxEHR)
from ..ehr.coding_scheme import GroupingData
from ..ehr.tvx_concepts import SegmentedAdmission, ObservablesDistribution
from ..ehr.tvx_ehr import SegmentedTVxEHR
from ..utils import model_params_scaler, tqdm_constructor


class MonotonicLeadingObsPredictor(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, input_size: int,
                 n_lead_times: int,
                 key: jrandom.PRNGKey, **mlp_kwargs):
        super().__init__()
        out_size = n_lead_times + 1
        width = mlp_kwargs.get("width_size", out_size * 5)
        self.mlp = eqx.nn.MLP(input_size,
                              out_size,
                              width_size=width,
                              activation=jnn.tanh,
                              depth=2,
                              key=key)

    @eqx.filter_jit
    def __call__(self, state):
        y = self.mlp(state)
        risk = jnn.sigmoid(y[-1])
        p = jnp.cumsum(jnn.softmax(y[:-1]))
        return risk * p


class MLPLeadingObsPredictor(eqx.Module):
    _mlp: eqx.nn.MLP

    def __init__(self, input_size: int,
                 n_lead_times: int,
                 key: jax.random.PRNGKey, **mlp_kwargs):
        super().__init__()
        width = mlp_kwargs.get("width_size", n_lead_times * 5)
        self._mlp = eqx.nn.MLP(input_size,
                               n_lead_times,
                               width_size=width,
                               activation=jnn.tanh,
                               final_activation=jnn.sigmoid,
                               depth=2,
                               key=key)

    @eqx.filter_jit
    def __call__(self, state):
        return self._mlp(state)


class CompiledMLP(eqx.nn.MLP):
    # Just an eqx.nn.MLP with a compiled __call__ method.

    @eqx.filter_jit
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return super().__call__(x)


class GRUDynamics(eqx.Module):
    x_x: eqx.nn.Linear
    x_r: eqx.nn.Linear
    x_z: eqx.nn.Linear
    rx_g: eqx.nn.Linear

    def __init__(self, input_size: int, state_size: int, key: "jax.random.PRNGKey"):
        k0, k1, k2, k3 = jrandom.split(key, 4)
        self.x_x = eqx.nn.Linear(input_size, state_size, key=k0)
        self.x_r = eqx.nn.Linear(state_size, state_size, key=k1)
        self.x_z = eqx.nn.Linear(state_size, state_size, key=k2)
        self.rx_g = eqx.nn.Linear(state_size, state_size, key=k3)

    @eqx.filter_jit
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.x_x(x)
        r = jnn.sigmoid(self.x_r(x))
        z = jnn.sigmoid(self.x_z(x))
        g = jnn.tanh(self.rx_g(r * x))
        return (1 - z) * (g - x)


class CompiledLinear(eqx.nn.Linear):

    @eqx.filter_jit
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return super().__call__(x)


class ProbMLP(CompiledMLP):

    @eqx.filter_jit
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        mean_std = super().__call__(x)
        mean, std = jnp.split(mean_std, 2, axis=-1)
        return mean, jnn.softplus(std)


class CompiledGRU(eqx.nn.GRUCell):
    @eqx.filter_jit
    def __call__(self, x: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
        return super().__call__(h, x)


class indexable_empty_array:

    def __getitem__(self, item):
        return jnp.ndarray([])


def empty_if_none(x):
    return x if x is not None else indexable_empty_array()


class InpatientModelConfig(ModelConfig):
    state: int = 50
    lead_predictor: LeadPredictorName = "monotonic"


class ICENODEStateTrajectory(InpatientObservables):

    @property
    def forecasted_state(self):
        return self.value

    @property
    def imputed_state(self):
        return self.extra_layers[0]

    @staticmethod
    def empty(size: int,
              time_dtype: Type | str = jnp.float64,
              value_dtype: Type | str = jnp.float16,
              mask_dtype: Type | str = bool) -> ICENODEStateTrajectory:
        traj = InpatientObservables.empty(size, time_dtype, value_dtype, mask_dtype)
        return eqx.tree_at(lambda x: x.extra_layers, traj, (traj.value,))

    @staticmethod
    def compile(time: jnp.ndarray, forecasted_state: Tuple[jnp.ndarray, ...],
                imputed_state: Tuple[jnp.ndarray, ...]) -> ICENODEStateTrajectory:
        forecasted_state = jnp.vstack(forecasted_state)
        imputed_state = jnp.vstack(imputed_state)
        return ICENODEStateTrajectory(time=time, value=forecasted_state, extra_layers=(imputed_state,),
                                      mask=jnp.ones_like(forecasted_state, dtype=bool))


class ICENODEMetrics(ModelBehaviouralMetrics):
    ode: ODEMetrics
    imputer: ImputerMetrics


class AdmissionTrajectoryPrediction(AdmissionPrediction):
    trajectory: Optional[ICENODEStateTrajectory] = None
    model_behavioural_metrics: Optional[ICENODEMetrics] = None
    imputed_observables: InpatientObservables = None


class AdmissionGRUODEBayesPrediction(AdmissionTrajectoryPrediction):
    observables: Optional[ObservablesDistribution] = None
    imputed_observables: Optional[ObservablesDistribution] = None


class DirectLeadPredictorWrapper(eqx.Module):
    predictor: MonotonicLeadingObsPredictor | MLPLeadingObsPredictor

    def __init__(self, input_size: int,
                 lead_times: Tuple[float, ...],
                 predictor: LeadPredictorName,
                 key: jrandom.PRNGKey, **mlp_kwargs):
        super().__init__()
        if predictor == "monotonic":
            self.predictor = MonotonicLeadingObsPredictor(input_size, len(lead_times), key, **mlp_kwargs)
        elif predictor == "mlp":
            self.predictor = MLPLeadingObsPredictor(input_size, len(lead_times), key, **mlp_kwargs)
        else:
            raise ValueError(f"Unknown leading predictor type: {predictor}")

    def __call__(self, trajectory: ICENODEStateTrajectory, **kwargs) -> InpatientObservables:
        leading_values = eqx.filter_vmap(self.predictor)(trajectory.imputed_state)
        return InpatientObservables(time=trajectory.time, value=leading_values,
                                    mask=jnp.ones_like(leading_values, dtype=bool))


DynamicsLiteral = Literal["gru", "mlp"]


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
                 true_observables: jnp.ndarray, observables_mask: jnp.ndarray,
                 u: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, ImputerMetrics]:
        error = jnp.where(observables_mask, f_obs_decoder(forecasted_state) - true_observables, 0.0)
        projected_error = self.f_project_error(error)
        return self.f_update(jnp.hstack((observables_mask, projected_error)), forecasted_state), ImputerMetrics()


class ICENODEConfig(InpatientModelConfig):
    dynamics: DynamicsLiteral = "mlp"


class ICEDynComponents(eqx.Module):
    emb: Optional[AdmissionEmbedding] = None
    obs_dec: Optional[CompiledMLP] = None
    lead_dec: Optional[DirectLeadPredictorWrapper] = None
    dyn: Optional[NeuralODESolver | CompiledGRU | RETAINDynamic] = None
    update: Optional[DirectGRUStateImputer] = None
    init: Optional[CompiledMLP] = None
    outcome_dec: Optional[CompiledMLP] = None


class InICENODE(InpatientModel):
    """
    The InICENODE model. It is composed of the following components:
        - f_emb: Embedding function.
        - f_obs_dec: Observation decoder.
        - f_dx_dec: Discharge codes decoder.
        - f_dyn: Dynamics function.
        - f_update: Update function.`
    """
    f_emb: AdmissionEmbedding
    f_obs_dec: CompiledMLP
    f_lead_dec: DirectLeadPredictorWrapper
    f_dyn: NeuralODESolver
    f_update: DirectGRUStateImputer
    f_init: CompiledMLP
    f_outcome_dec: Optional[CompiledMLP] = None

    config: ICENODEConfig = eqx.static_field()

    def __init__(self, config: ICENODEConfig,
                 embeddings_config: AdmissionEmbeddingsConfig,
                 lead_times: Tuple[float, ...],
                 dx_codes_size: Optional[int] = None,
                 outcome_size: Optional[int] = None,
                 icu_inputs_grouping: Optional[GroupingData] = None,
                 icu_procedures_size: Optional[int] = None,
                 hosp_procedures_size: Optional[int] = None,
                 demographic_size: Optional[int] = None,
                 observables_size: Optional[int] = None, *,
                 key: "jax.random.PRNGKey", lead_mlp_kwargs: Dict[str, Any] = None):
        super().__init__(config=config)

        (emb_key, obs_dec_key, lead_key, outcome_dec_key, dyn_key, update_key) = jrandom.split(key, 6)
        self.f_emb = self._make_embedding(config=embeddings_config,
                                          dx_codes_size=dx_codes_size,
                                          icu_inputs_grouping=icu_inputs_grouping,
                                          icu_procedures_size=icu_procedures_size,
                                          hosp_procedures_size=hosp_procedures_size,
                                          demographic_size=demographic_size,
                                          observables_size=observables_size,
                                          key=key)
        self.f_init = self._make_init(embeddings_config=embeddings_config,
                                      state_size=config.state,
                                      observables_size=observables_size,
                                      key=emb_key)
        self.f_outcome_dec = self._make_outcome_dec(state_size=config.state,
                                                    outcome_size=outcome_size,
                                                    key=outcome_dec_key)

        self.f_obs_dec = self._make_obs_dec(config=config, observables_size=observables_size, key=obs_dec_key)
        self.f_dyn = self._make_dyn(embeddings_config=embeddings_config,
                                    model_config=config,
                                    observables_size=observables_size,
                                    key=dyn_key)

        self.f_update = self._make_update(state_size=config.state, observables_size=observables_size, key=update_key)

        self.f_lead_dec = self._make_lead_dec(input_size=config.state,
                                              lead_times=lead_times,
                                              predictor=config.lead_predictor,
                                              observables_size=observables_size,
                                              key=lead_key, mlp_kwargs=lead_mlp_kwargs or dict())

    @property
    def components(self) -> ICEDynComponents:
        return ICEDynComponents(emb=self.f_emb, obs_dec=self.f_obs_dec, lead_dec=self.f_lead_dec,
                                dyn=self.f_dyn, update=self.f_update, init=self.f_init, outcome_dec=self.f_outcome_dec)

    @property
    def dyn_params_list(self):
        return jtu.tree_leaves(eqx.filter(self.f_dyn, eqx.is_inexact_array))

    @classmethod
    def from_tvx_ehr(cls, tvx_ehr: TVxEHR, config: ICENODEConfig,
                     embeddings_config: AdmissionEmbeddingsConfig, seed: int = 0) -> Self:
        key = jrandom.PRNGKey(seed)
        return cls(config=config,
                   embeddings_config=embeddings_config,
                   lead_times=tuple(tvx_ehr.config.leading_observable.leading_hours),
                   dx_codes_size=len(tvx_ehr.scheme.dx_discharge),
                   outcome_size=tvx_ehr.scheme.outcome_size,
                   icu_inputs_grouping=tvx_ehr.icu_inputs_grouping,
                   icu_procedures_size=len(tvx_ehr.scheme.icu_procedures),
                   hosp_procedures_size=len(tvx_ehr.scheme.hosp_procedures),
                   demographic_size=tvx_ehr.demographic_vector_size,
                   observables_size=len(tvx_ehr.scheme.obs),
                   key=key)

    @staticmethod
    def _make_init(embeddings_config: AdmissionEmbeddingsConfig,
                   state_size: int, key: jrandom.PRNGKey, **kwargs) -> CompiledMLP:
        dx_codes_size = embeddings_config.dx_codes or 0
        demographic_size = embeddings_config.demographic or 0
        return CompiledMLP(dx_codes_size + demographic_size,
                           state_size,
                           state_size * 3,
                           depth=2,
                           key=key)

    @staticmethod
    def _make_lead_dec(
            input_size: int,
            lead_times: Tuple[float, ...],
            predictor: LeadPredictorName,
            key: jrandom.PRNGKey, mlp_kwargs: Dict[str, Any],
            observables_size: int, **kwargs
    ) -> DirectLeadPredictorWrapper:
        return DirectLeadPredictorWrapper(input_size, lead_times, predictor, key, **mlp_kwargs)

    @staticmethod
    def _make_dyn(model_config: ICENODEConfig,
                  embeddings_config: AdmissionEmbeddingsConfig, *,
                  key: jrandom.PRNGKey, **kwargs) -> NeuralODESolver:
        interventions_size = embeddings_config.interventions.interventions if embeddings_config.interventions else 0
        demographics_size = embeddings_config.demographic or 0
        if model_config.dynamics == "mlp":
            f_dyn = CompiledMLP(in_size=model_config.state + interventions_size + demographics_size,
                                out_size=model_config.state,
                                activation=jnn.tanh,
                                depth=2,
                                width_size=model_config.state,
                                key=key)
        elif model_config.dynamics == "gru":
            f_dyn = GRUDynamics(model_config.state + interventions_size + demographics_size, model_config.state, key)
        else:
            raise ValueError(f"Unknown dynamics type: {model_config.dynamics}")
        f_dyn = model_params_scaler(f_dyn, 1e-2, eqx.is_inexact_array)
        return NeuralODESolver.from_mlp(mlp=f_dyn, second=1 / 3600.0, dt0=60.0)

    @staticmethod
    def _make_update(state_size: int, observables_size: int, key: jrandom.PRNGKey) -> DirectGRUStateImputer:
        return DirectGRUStateImputer(state_size, observables_size, key=key)

    @staticmethod
    def _make_outcome_dec(state_size: int, outcome_size: Optional[int], key: jrandom.PRNGKey) -> Optional[CompiledMLP]:
        return CompiledMLP(state_size,
                           outcome_size,
                           state_size * 2,
                           activation=jnp.tanh,
                           depth=1,
                           key=key) if outcome_size is not None else None

    @staticmethod
    def _make_obs_dec(config, observables_size, key) -> CompiledMLP:
        return CompiledMLP(config.state,
                           observables_size,
                           observables_size * 5,
                           activation=jnp.tanh,
                           depth=1,
                           key=key)

    @staticmethod
    def _make_embedding(config: AdmissionEmbeddingsConfig,
                        dx_codes_size: Optional[int],
                        icu_inputs_grouping: Optional[GroupingData],
                        icu_procedures_size: Optional[int],
                        hosp_procedures_size: Optional[int],
                        demographic_size: Optional[int],
                        observables_size: Optional[int],
                        key: jrandom.PRNGKey, **kwargs):
        return AdmissionEmbedding(
            config=config,
            dx_codes_size=dx_codes_size,
            icu_inputs_grouping=icu_inputs_grouping,
            icu_procedures_size=icu_procedures_size,
            hosp_procedures_size=hosp_procedures_size,
            demographic_size=demographic_size,
            observables_size=observables_size,
            key=key)

    def forecasted_observables(self,
                               admission: SegmentedAdmission | Admission,
                               state_trajectory: ICENODEStateTrajectory) -> InpatientObservables:
        f = self.components
        pred_obs = eqx.filter_vmap(f.obs_dec)(state_trajectory.forecasted_state)
        return InpatientObservables(time=state_trajectory.time, value=pred_obs, mask=admission.observables.mask)

    def imputed_observables(self,
                            admission: SegmentedAdmission | Admission,
                            state_trajectory: ICENODEStateTrajectory) -> InpatientObservables:
        f = self.components
        pred_obs = eqx.filter_vmap(f.obs_dec)(state_trajectory.imputed_state)
        return InpatientObservables(time=state_trajectory.time, value=pred_obs, mask=admission.observables.mask)

    def __call__(
            self, admission: SegmentedAdmission,
            embedded_admission: EmbeddedAdmission, precomputes: Precomputes
    ) -> AdmissionTrajectoryPrediction:
        ode_stats = ODEMetrics()
        imputer_stats = ImputerMetrics()
        prediction = AdmissionTrajectoryPrediction(admission=admission)
        int_e = empty_if_none(embedded_admission.interventions)
        demo_e = empty_if_none(embedded_admission.demographic)
        obs = admission.observables
        if len(obs) == 0:
            logging.debug("No observation to fit.")
            return prediction

        n_segments = obs.n_segments
        t = 0.0
        state_trajectory = tuple()
        f = self.components
        state = f.init(jnp.hstack((embedded_admission.dx_codes_history, demo_e)))
        segments_t1 = admission.interventions.t1
        key = jrandom.PRNGKey(hash(admission.admission_id))

        for segment_index in range(n_segments):
            segment_t1 = segments_t1[segment_index]
            segment_obs = obs[segment_index]
            segment_interventions = int_e[segment_index]
            segment_force = jnp.hstack((demo_e, segment_interventions))

            for obs_t, obs_val, obs_mask in segment_obs:
                key, subkey = jrandom.split(key)
                # if time-diff is more than 1 seconds, we integrate.
                forecasted_state, ode_stats_ = f.dyn(state, t0=t, t1=obs_t, u=segment_force,
                                                     precomputes=precomputes,
                                                     key=subkey)
                ode_stats += ode_stats_
                forecasted_state = forecasted_state.squeeze()
                state, imputer_stats_ = f.update(f.obs_dec, forecasted_state, obs_val, obs_mask,
                                                 u=segment_force)
                imputer_stats += imputer_stats_
                state_trajectory += ((forecasted_state, state),)
                t = obs_t
            key, subkey = jrandom.split(key)
            state, stats = f.dyn(state, t0=t, t1=segment_t1, u=segment_force, precomputes=precomputes,
                                 key=subkey)
            ode_stats += stats
            state = state.squeeze()
            t = segment_t1

        prediction = prediction.add(model_behavioural_metrics=ICENODEMetrics(ode=ode_stats, imputer=imputer_stats))
        if f.outcome_dec is not None:
            prediction = prediction.add(outcome=CodesVector(f.outcome_dec(state), admission.outcome.scheme))
        if len(state_trajectory) > 0:
            forecasted_states, imputed_states = zip(*state_trajectory)
            icenode_state_trajectory = ICENODEStateTrajectory.compile(time=obs.time,
                                                                      forecasted_state=forecasted_states,
                                                                      imputed_state=imputed_states)
            # TODO: test --> assert len(obs.time) == len(forecasted_states)
            prediction = prediction.add(observables=self.forecasted_observables(
                admission=admission, state_trajectory=icenode_state_trajectory))
            # prediction = prediction.add(imputed_observables=self.imputed_observables(
            #     admission=admission, state_trajectory=icenode_state_trajectory))
            prediction = prediction.add(leading_observable=f.lead_dec(icenode_state_trajectory))
            # prediction = prediction.add(trajectory=icenode_state_trajectory)
        return prediction

    def batch_predict(self, inpatients: SegmentedTVxEHR, leave_pbar: bool = False) -> AdmissionsPrediction:
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
                                              precomputes=precomputes))
                    pbar.update(admission.interval_days)

                if i % 200 == 0:
                    eqx.clear_caches()
                    jax.clear_caches()
                    jax.clear_backends()
            return results.filter_nans()


class InICENODELite(InICENODE):
    # Same as InICENODE but without discharge summary outcome predictions.
    def __init__(self, config: ICENODEConfig,
                 embeddings_config: AdmissionEmbeddingsConfig,
                 lead_times: Tuple[float, ...],
                 dx_codes_size: Optional[int] = None,
                 icu_inputs_grouping: Optional[GroupingData] = None,
                 icu_procedures_size: Optional[int] = None,
                 hosp_procedures_size: Optional[int] = None,
                 demographic_size: Optional[int] = None,
                 observables_size: Optional[int] = None, *,
                 key: "jax.random.PRNGKey"):
        super().__init__(config=config, embeddings_config=embeddings_config, lead_times=lead_times,
                         dx_codes_size=dx_codes_size, outcome_size=None, icu_inputs_grouping=icu_inputs_grouping,
                         icu_procedures_size=icu_procedures_size, hosp_procedures_size=hosp_procedures_size,
                         demographic_size=demographic_size, observables_size=observables_size, key=key)

    @classmethod
    def from_tvx_ehr(cls, tvx_ehr: TVxEHR, config: ICENODEConfig,
                     embeddings_config: AdmissionEmbeddingsConfig, seed: int = 0) -> Self:
        key = jrandom.PRNGKey(seed)
        return cls(config=config,
                   embeddings_config=embeddings_config,
                   lead_times=tuple(tvx_ehr.config.leading_observable.leading_hours),
                   dx_codes_size=len(tvx_ehr.scheme.dx_discharge),
                   icu_inputs_grouping=tvx_ehr.icu_inputs_grouping,
                   icu_procedures_size=len(tvx_ehr.scheme.icu_procedures),
                   hosp_procedures_size=len(tvx_ehr.scheme.hosp_procedures),
                   demographic_size=tvx_ehr.demographic_vector_size,
                   observables_size=len(tvx_ehr.scheme.obs),
                   key=key)


class StochasticInICENODELite(InICENODELite):

    @staticmethod
    def _make_dyn(model_config: ICENODEConfig,
                  embeddings_config: AdmissionEmbeddingsConfig, *,
                  key: jrandom.PRNGKey, **kwargs) -> SkipShortIntervalsWrapper:
        interventions_size = embeddings_config.interventions.interventions if embeddings_config.interventions else 0
        demographics_size = embeddings_config.demographic or 0
        if model_config.dynamics == "mlp":
            f_dyn = CompiledMLP(in_size=model_config.state + interventions_size + demographics_size,
                                out_size=model_config.state,
                                activation=jnn.tanh,
                                depth=2,
                                width_size=model_config.state,
                                key=key)
        elif model_config.dynamics == "gru":
            f_dyn = GRUDynamics(model_config.state + interventions_size + demographics_size, model_config.state, key)
        else:
            raise ValueError(f"Unknown dynamics type: {model_config.dynamics}")

        f_diffusion = DiffusionMLP(brownian_size=model_config.state // 5,
                                   control_size=interventions_size + demographics_size,
                                   state_size=model_config.state, key=key, depth=2, width_size=model_config.state * 2)
        f_dyn = model_params_scaler(f_dyn, 1e-2, eqx.is_inexact_array)
        f_diffusion = model_params_scaler(f_diffusion, 5e-2, eqx.is_inexact_array)
        f_odeint = StochasticNeuralODESolver.from_mlp(drift=f_dyn,
                                                      diffusion=f_diffusion,
                                                      second=1 / 3600.0, dt0=60.0)
        return SkipShortIntervalsWrapper(solver=f_odeint, min_interval=1.0)


class StochasticMechanisticICENODE(InICENODELite):
    @property
    def persistent_memory_size(self):
        return self.config.state // 5

    @eqx.filter_jit
    def _f_update(self,
                  obs_decoder: ICNNObsDecoder,
                  forecasted_state: jnp.ndarray,
                  true_observables: jnp.ndarray,
                  observables_mask: jnp.ndarray,
                  u: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, ImputerMetrics]:
        # init_obs: obs_decoder(forecasted_state).
        # input: (persistent_hidden_confounder, hidden_confounder, init_obs).
        # mask: (ones_like(state_mem), zeros_like(state_hidden).
        hidden_confounder = forecasted_state[:obs_decoder.state_size]
        input = jnp.hstack((hidden_confounder, true_observables))
        mask = jnp.zeros_like(input).at[:self.persistent_memory_size].set(1)
        mask = mask.at[obs_decoder.state_size:].set(observables_mask)
        return obs_decoder.partial_input_optimise(input, mask)

    @property
    def components(self):
        return ICEDynComponents(emb=self.f_emb, obs_dec=self.f_obs_dec, lead_dec=self.f_lead_dec,
                                dyn=self.f_dyn, update=self._f_update, init=self.f_init, outcome_dec=self.f_outcome_dec)

    @staticmethod
    def _make_dyn(model_config: ICENODEConfig,
                  embeddings_config: AdmissionEmbeddingsConfig,
                  *,
                  key: jrandom.PRNGKey, observables_size: Optional[int] = None, **kwargs) -> SkipShortIntervalsWrapper:
        interventions_size = embeddings_config.interventions.interventions if embeddings_config.interventions else 0
        demographics_size = embeddings_config.demographic or 0
        integrand_size = model_config.state + observables_size
        if model_config.dynamics == "mlp":
            f_dyn = CompiledMLP(in_size=integrand_size + interventions_size + demographics_size,
                                out_size=integrand_size,
                                activation=jnn.tanh,
                                depth=2,
                                width_size=integrand_size,
                                key=key)
        elif model_config.dynamics == "gru":
            f_dyn = GRUDynamics(integrand_size + interventions_size + demographics_size, integrand_size, key)
        else:
            raise ValueError(f"Unknown dynamics type: {model_config.dynamics}")

        f_diffusion = DiffusionMLP(brownian_size=integrand_size // 5,
                                   control_size=interventions_size + demographics_size,
                                   state_size=integrand_size, key=key, depth=2, width_size=integrand_size * 2)
        f_dyn = model_params_scaler(f_dyn, 1e-2, eqx.is_inexact_array)
        f_diffusion = model_params_scaler(f_diffusion, 5e-2, eqx.is_inexact_array)
        f_odeint = StochasticNeuralODESolver.from_mlp(drift=f_dyn,
                                                      diffusion=f_diffusion,
                                                      second=1 / 3600.0, dt0=60.0)
        return SkipShortIntervalsWrapper(solver=f_odeint, min_interval=1.0)

    @staticmethod
    def _make_init(embeddings_config: AdmissionEmbeddingsConfig,
                   state_size: int, key: jrandom.PRNGKey, observables_size: int = 0, **kwargs) -> CompiledMLP:
        dx_codes_size = embeddings_config.dx_codes or 0
        demographic_size = embeddings_config.demographic or 0
        integrand_size = state_size + observables_size
        return CompiledMLP(dx_codes_size + demographic_size,
                           integrand_size,
                           integrand_size * 3,
                           depth=2,
                           key=key)

    @staticmethod
    def _make_lead_dec(
            input_size: int,
            lead_times: Tuple[float, ...],
            predictor: LeadPredictorName,
            key: jrandom.PRNGKey, mlp_kwargs: Dict[str, Any], observables_size: int, **kwargs
    ) -> DirectLeadPredictorWrapper:
        integrand_size = input_size + observables_size
        return DirectLeadPredictorWrapper(integrand_size, lead_times, predictor, key, **mlp_kwargs)

    @staticmethod
    def _make_update(state_size: int, observables_size: int, key: jrandom.PRNGKey) -> None:
        return None

    @staticmethod
    def _make_obs_dec(config, observables_size, key) -> ICNNObsExtractor:
        return ICNNObsExtractor(observables_size=observables_size, state_size=config.state, optimiser_name='lamb',
                                max_steps=2 ** 9, lr=1e-2,
                                positivity='softplus', hidden_size_multiplier=3, depth=4, key=key)


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
                 true_observables: jnp.ndarray, observables_mask: jnp.ndarray,
                 u: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, ImputerMetrics]:
        mean_hat, std_hat = f_obs_decoder(forecasted_state)
        # dimension: (obs_dim, )
        error = (true_observables - mean_hat) / (std_hat + 1e-6)
        gru_input = jnp.hstack([mean_hat, std_hat, error])
        gru_input = self.f_project_error(gru_input).reshape(self.obs_size, self.prep_hidden)
        gru_input = gru_input * observables_mask.reshape(-1, 1) + self.f_project_error_bias
        return self.f_update(jnp.tanh(gru_input.flatten()), forecasted_state), ImputerMetrics()


class GRUODEBayes(InICENODELite):
    f_obs_dec: ProbMLP
    f_update: DirectGRUStateProbabilisticImputer

    def forecasted_observables(self,
                               admission: SegmentedAdmission | Admission,
                               state_trajectory: ICENODEStateTrajectory) -> ObservablesDistribution:
        f = self.components
        pred_obs_mean, pred_obs_std = eqx.filter_vmap(f.obs_dec)(state_trajectory.forecasted_state)
        return ObservablesDistribution.compile(time=state_trajectory.time, mean=pred_obs_mean, std=pred_obs_std,
                                               mask=admission.observables.mask)

    def imputed_observables(self,
                            admission: SegmentedAdmission | Admission,
                            state_trajectory: ICENODEStateTrajectory) -> ObservablesDistribution:
        f = self.components
        pred_obs_mean, pred_obs_std = eqx.filter_vmap(f.obs_dec)(state_trajectory.imputed_state)
        return ObservablesDistribution.compile(time=state_trajectory.time, mean=pred_obs_mean, std=pred_obs_std,
                                               mask=admission.observables.mask)

    @staticmethod
    def _make_obs_dec(config, observables_size, key) -> ProbMLP:
        return ProbMLP(config.state,
                       observables_size * 2,
                       observables_size * 5,
                       activation=jnp.tanh,
                       depth=1,
                       key=key)

    @staticmethod
    def _make_update(state_size: int, observables_size: int,
                     key: jrandom.PRNGKey) -> DirectGRUStateProbabilisticImputer:
        return DirectGRUStateProbabilisticImputer(obs_size=observables_size, state_size=state_size, key=key)

    def __call__(self, admission: SegmentedAdmission,
                 embedded_admission: EmbeddedAdmission, precomputes: Precomputes) -> AdmissionGRUODEBayesPrediction:
        predictions = super().__call__(admission, embedded_admission, precomputes)
        updated = AdmissionGRUODEBayesPrediction(admission=None)
        for field in predictions.fields:
            updated = eqx.tree_at(lambda x: getattr(x, field), updated, getattr(predictions, field),
                                  is_leaf=lambda x: x is None)

        return updated


class InICENODELiteICNNImpute(InICENODELite):
    f_update: None
    f_obs_dec: ICNNObsDecoder

    @property
    def persistent_memory_size(self):
        return self.config.state // 5

    @eqx.filter_jit
    def _f_update(self,
                  obs_decoder: ICNNObsDecoder,
                  forecasted_state: jnp.ndarray,
                  true_observables: jnp.ndarray,
                  observables_mask: jnp.ndarray,
                  u: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, ImputerMetrics]:
        # init_obs: obs_decoder(forecasted_state).
        # input: (state_mem, state_hidden, init_obs).
        # mask: (ones_like(state_mem), zeros_like(state_hidden).
        input = jnp.hstack((forecasted_state, true_observables))
        mask = jnp.zeros_like(input).at[:self.persistent_memory_size].set(1)
        mask = mask.at[obs_decoder.state_size:].set(observables_mask)
        output, stats = obs_decoder.partial_input_optimise(input, mask)
        state, _ = jnp.split(output, [obs_decoder.state_size])
        return state, stats

    @property
    def components(self) -> ICEDynComponents:
        return ICEDynComponents(emb=self.f_emb, obs_dec=self.f_obs_dec, lead_dec=self.f_lead_dec,
                                dyn=self.f_dyn, update=self._f_update, init=self.f_init, outcome_dec=self.f_outcome_dec)

    @staticmethod
    def _make_update(state_size: int, observables_size: int, key: jrandom.PRNGKey) -> None:
        return None

    @staticmethod
    def _make_obs_dec(config, observables_size, key) -> ICNNObsDecoder:
        return ICNNObsDecoder(observables_size=observables_size, state_size=config.state, optimiser_name='lamb',
                              max_steps=2 ** 9, lr=1e-2,
                              positivity='softplus', hidden_size_multiplier=3, depth=4, key=key)


class InGRUJump(InICENODELite):
    # TODO: as for the original paper, use multi-layer embeddings with skip
    # connections.
    f_emb: AdmissionSequentialEmbeddingsConfig
    # GRU. Alternatives: SRU, LSTM, ..
    f_dyn: CompiledGRU

    def __init__(self, config: InpatientModelConfig,
                 embeddings_config: AdmissionSequentialEmbeddingsConfig,
                 lead_times: Tuple[float, ...],
                 dx_codes_size: Optional[int] = None,
                 demographic_size: Optional[int] = None,
                 observables_size: Optional[int] = None, *,
                 key: "jax.random.PRNGKey"):
        super().__init__(config=config, embeddings_config=embeddings_config,
                         lead_times=lead_times,
                         dx_codes_size=dx_codes_size, icu_inputs_grouping=None,
                         icu_procedures_size=None, hosp_procedures_size=None,
                         demographic_size=demographic_size, observables_size=observables_size, key=key)

    @classmethod
    def from_tvx_ehr(cls, tvx_ehr: TVxEHR, config: InpatientModelConfig,
                     embeddings_config: AdmissionSequentialEmbeddingsConfig, seed: int = 0) -> Self:
        key = jrandom.PRNGKey(seed)
        return cls(config=config,
                   embeddings_config=embeddings_config,
                   lead_times=tuple(tvx_ehr.config.leading_observable.leading_hours),
                   dx_codes_size=len(tvx_ehr.scheme.dx_discharge),
                   demographic_size=tvx_ehr.demographic_vector_size,
                   observables_size=len(tvx_ehr.scheme.obs),
                   key=key)

    @staticmethod
    def _make_init(embeddings_config: AdmissionSequentialEmbeddingsConfig,
                   state_size: int, key: jrandom.PRNGKey, **kwargs) -> CompiledMLP:
        return CompiledMLP(embeddings_config.sequence,
                           state_size,
                           state_size * 3,
                           depth=2,
                           key=key)

    @staticmethod
    def _make_embedding(config: AdmissionSequentialEmbeddingsConfig,
                        dx_codes_size: Optional[int],
                        demographic_size: Optional[int],
                        observables_size=Optional[int],
                        *,
                        key: jrandom.PRNGKey, **kwargs) -> AdmissionSequentialObsEmbedding:
        return AdmissionSequentialObsEmbedding(config=config,
                                               dx_codes_size=dx_codes_size,
                                               demographic_size=demographic_size,
                                               observables_size=observables_size,
                                               key=key)

    @staticmethod
    def _make_dyn(model_config: InpatientModelConfig,
                  embeddings_config: AdmissionSequentialEmbeddingsConfig, *,
                  key: jrandom.PRNGKey, **kwargs) -> CompiledGRU:
        return CompiledGRU(input_size=embeddings_config.sequence,
                           hidden_size=model_config.state,
                           key=key)

    def __call__(self, admission: Admission, embedded_admission: EmbeddedDischargeSummary,
                 precomputes: Precomputes) -> AdmissionPrediction:
        prediction = AdmissionPrediction(admission=admission)
        obs = admission.observables
        if len(obs) == 0:
            logging.debug("No observation to fit.")
            return prediction
        state_trajectory = tuple()
        f = self.components
        state = f.init(embedded_admission.history_summary)
        for i in range(len(obs.time)):
            forecasted_state = f.dyn(embedded_admission.history_summary, state)
            state = f.update(f.obs_dec, forecasted_state, obs.value[i], obs.mask[i])
            state_trajectory += ((forecasted_state, state),)

        forecasted_states, imputed_states = zip(*state_trajectory)
        gru_state_trajectory = ICENODEStateTrajectory.compile(time=obs.time, forecasted_state=forecasted_states,
                                                              imputed_state=imputed_states)
        prediction = prediction.add(observables=self.forecasted_observables(
            admission=admission, state_trajectory=gru_state_trajectory))
        prediction = prediction.add(leading_observable=f.lead_dec(gru_state_trajectory))
        return prediction


class InGRU(InICENODELite):
    f_emb: AdmissionSequentialObsEmbedding
    # GRU. Alternatives: SRU, LSTM, ..
    f_dyn: CompiledGRU

    def __init__(self, config: InpatientModelConfig,
                 embeddings_config: AdmissionSequentialEmbeddingsConfig,
                 lead_times: Tuple[float, ...],
                 dx_codes_size: Optional[int] = None,
                 demographic_size: Optional[int] = None,
                 observables_size: Optional[int] = None, *,
                 key: "jax.random.PRNGKey"):
        super().__init__(config=config, embeddings_config=embeddings_config,
                         lead_times=lead_times,
                         dx_codes_size=dx_codes_size, icu_inputs_grouping=None,
                         icu_procedures_size=None, hosp_procedures_size=None,
                         demographic_size=demographic_size, observables_size=observables_size, key=key)

    @classmethod
    def from_tvx_ehr(cls, tvx_ehr: TVxEHR, config: InpatientModelConfig,
                     embeddings_config: AdmissionSequentialEmbeddingsConfig, seed: int = 0) -> Self:
        key = jrandom.PRNGKey(seed)
        return cls(config=config,
                   embeddings_config=embeddings_config,
                   lead_times=tuple(tvx_ehr.config.leading_observable.leading_hours),
                   dx_codes_size=len(tvx_ehr.scheme.dx_discharge),
                   demographic_size=tvx_ehr.demographic_vector_size,
                   observables_size=len(tvx_ehr.scheme.obs),
                   key=key)

    @staticmethod
    def _make_embedding(config: DischargeSummarySequentialEmbeddingsConfig,
                        dx_codes_size: Optional[int],
                        demographic_size: Optional[int],
                        observables_size: Optional[int], *,
                        key: jrandom.PRNGKey, **kwargs) -> AdmissionSequentialObsEmbedding:
        return InGRUJump._make_embedding(config=config,
                                         dx_codes_size=dx_codes_size,
                                         demographic_size=demographic_size,
                                         observables_size=observables_size,
                                         key=key)

    @staticmethod
    def _make_dyn(model_config: InpatientModelConfig,
                  embeddings_config: AdmissionSequentialEmbeddingsConfig, *,
                  key: jrandom.PRNGKey, **kwargs) -> CompiledGRU:
        return CompiledGRU(input_size=embeddings_config.sequence,
                           hidden_size=model_config.state,
                           key=key)

    def __call__(
            self, admission: Admission,
            embedded_admission: EmbeddedAdmissionObsSequence, precomputes: Precomputes) -> AdmissionPrediction:
        emb_dx_history = empty_if_none(embedded_admission.dx_codes_history)
        emb_demo = empty_if_none(embedded_admission.demographic)
        force = jnp.hstack((emb_dx_history, emb_demo))
        prediction = AdmissionPrediction(admission=admission)
        if len(embedded_admission.sequence) == 0:
            logging.debug("No observations to fit.")
            return prediction
        f = self.components
        state = f.init(force)
        state_trajectory = tuple()
        for seq_e in embedded_admission.sequence:
            forecasted_state = state
            state = f.dyn(seq_e, state)
            state_trajectory += ((forecasted_state, state),)

        forecasted_states, imputed_states = zip(*state_trajectory)
        gru_state_trajectory = ICENODEStateTrajectory.compile(time=admission.observables.time,
                                                              forecasted_state=forecasted_states,
                                                              imputed_state=imputed_states)
        prediction = prediction.add(observables=self.forecasted_observables(
            admission=admission, state_trajectory=gru_state_trajectory))
        prediction = prediction.add(leading_observable=f.lead_dec(gru_state_trajectory))
        return prediction


class NeuralODESolverGhost(NeuralODESolver):
    f: None = None

    @staticmethod
    def from_mlp(mlp: None = None, second: float = 1 / 3600.0, dt0: float = 60.0):
        return NeuralODESolverGhost()

    @eqx.filter_jit
    def __call__(self, x0, t0: float, t1: float, saveat: None = None,
                 u: None = None,
                 precomputes: None = None,
                 key: None = None) -> Tuple[jnp.ndarray, ODEMetrics]:
        return x0, ODEMetrics()


class InNaiveSequentialGRU(InICENODELite):
    @staticmethod
    def _make_dyn(model_config: ICENODEConfig,
                  embeddings_config: AdmissionEmbeddingsConfig, *,
                  key: jrandom.PRNGKey, **kwargs) -> NeuralODESolver:
        return NeuralODESolverGhost.from_mlp()


# Baseline Lead-predictor based on RectiLinear imputations (LOCF: Last-observation-carried-forward).

class InRectilinearConfig(ModelConfig):
    lead_predictor: Literal["monotonic", "mlp"] = "monotonic"


class InRectilinear(InpatientModel):
    f_lead_dec: DirectLeadPredictorWrapper
    imputer: RectilinearImputer = eqx.static_field()
    config: InRectilinearConfig = eqx.static_field()

    def __init__(self, config: InRectilinearConfig,
                 lead_times: Tuple[float, ...],
                 observables_size: Optional[int] = None, *,
                 imputer: RectilinearImputer,
                 key: "jax.random.PRNGKey"):
        self.f_lead_dec = DirectLeadPredictorWrapper(observables_size, lead_times, config.lead_predictor, key=key)
        self.config = config
        self.imputer = imputer

    def dyn_params_list(self):
        return []

    @classmethod
    def from_tvx_ehr(cls, tvx_ehr: TVxEHR, config: InRectilinearConfig,
                     embeddings_config: AdmissionSequentialEmbeddingsConfig, seed: int = 0) -> Self:
        key = jrandom.PRNGKey(seed)
        return cls(config=config,
                   lead_times=tuple(tvx_ehr.config.leading_observable.leading_hours),
                   observables_size=len(tvx_ehr.scheme.obs),
                   imputer=RectilinearImputer.from_tvx_ehr(tvx_ehr),
                   key=key)

    def __call__(
            self, admission: Admission,
            embedded_admission: None, precomputes: Precomputes) -> AdmissionPrediction:
        prediction = AdmissionTrajectoryPrediction(admission=admission)
        predicted_obs = self.imputer(admission).observables
        prediction = prediction.add(observables=predicted_obs)
        prediction = prediction.add(imputed_observables=predicted_obs)

        leading_values = eqx.filter_vmap(self.f_lead_dec.predictor)(predicted_obs.value)
        prediction = prediction.add(
            leading_observable=InpatientObservables(time=predicted_obs.time, value=leading_values,
                                                    mask=jnp.ones_like(leading_values, dtype=bool)))
        return prediction

    def batch_predict(self, inpatients: SegmentedTVxEHR, leave_pbar: bool = False) -> AdmissionsPrediction:
        total_int_days = inpatients.interval_days()
        precomputes = self.precomputes(inpatients)

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
                                              embedded_admission=None,
                                              precomputes=precomputes))
                    pbar.update(admission.interval_days)
            return results.filter_nans()


class InRETAINConfig(ModelConfig):
    lead_predictor: str = "monotonic"


class RETAINDynamic(eqx.Module):
    gru_a: CompiledGRU
    gru_b: CompiledGRU
    att_a: CompiledLinear
    att_b: CompiledLinear


class InRETAIN(InGRUJump):
    f_dyn: RETAINDynamic
    config: InRETAINConfig = eqx.static_field()
    state_size: int = eqx.static_field()

    def __init__(self, config: InpatientModelConfig,
                 embeddings_config: DischargeSummarySequentialEmbeddingsConfig,
                 lead_times: Tuple[float, ...],
                 dx_codes_size: Optional[int] = None,
                 demographic_size: Optional[int] = None,
                 observables_size: Optional[int] = None, *,
                 key: "jax.random.PRNGKey"):
        class InRETAINProxyConfig(InRETAINConfig, InpatientModelConfig):
            state: int

        config = InRETAINProxyConfig(state=embeddings_config.summary, **config.as_dict())
        super().__init__(config=config, embeddings_config=embeddings_config,
                         lead_times=lead_times,
                         dx_codes_size=dx_codes_size,
                         demographic_size=demographic_size, observables_size=observables_size, key=key)
        self.config = config
        self.state_size = embeddings_config.summary

    @classmethod
    def from_tvx_ehr(cls, tvx_ehr: TVxEHR, config: InpatientModelConfig,
                     embeddings_config: DischargeSummarySequentialEmbeddingsConfig, seed: int = 0) -> Self:
        key = jrandom.PRNGKey(seed)
        return cls(config=config,
                   embeddings_config=embeddings_config,
                   lead_times=tuple(tvx_ehr.config.leading_observable.leading_hours),
                   dx_codes_size=len(tvx_ehr.scheme.dx_discharge),
                   demographic_size=tvx_ehr.demographic_vector_size,
                   observables_size=len(tvx_ehr.scheme.obs),
                   key=key)

    @staticmethod
    def _make_dyn(model_config: InpatientModelConfig, embeddings_config: DischargeSummarySequentialEmbeddingsConfig, *,
                  key: jrandom.PRNGKey, **kwargs) -> RETAINDynamic:
        keys = jrandom.split(key, 4)
        gru_a = CompiledGRU(model_config.state,
                            model_config.state // 2,
                            use_bias=True,
                            key=keys[0])
        gru_b = CompiledGRU(model_config.state,
                            model_config.state // 2,
                            use_bias=True,
                            key=keys[1])

        att_a = CompiledLinear(model_config.state // 2,
                               1,
                               use_bias=True,
                               key=keys[2])
        att_b = CompiledLinear(model_config.state // 2,
                               model_config.state,
                               use_bias=True,
                               key=keys[3])

        return RETAINDynamic(gru_a=gru_a,
                             gru_b=gru_b,
                             att_a=att_a,
                             att_b=att_b)

    @staticmethod
    def _make_init(embeddings_config: DischargeSummarySequentialEmbeddingsConfig,
                   state_size: int, key: jrandom.PRNGKey, **kwargs) -> CompiledMLP:
        init_input_size = embeddings_config.demographic + embeddings_config.dx_codes
        return CompiledMLP(init_input_size,
                           state_size * 3,
                           state_size * 3,
                           depth=1,
                           key=key)

    @eqx.filter_jit
    def _lead_dec(self, x):
        return self.f_lead_dec(x)

    @eqx.filter_jit
    def _obs_dec(self, x):
        return self.f_obs_dec(x)

    @eqx.filter_jit
    def _init_state(self, emb_demo, emb_dx_history):
        force = jnp.hstack((emb_dx_history, emb_demo))
        state = self.f_init(force)
        splitter = (self.state_size, 2 * self.state_size)
        state_a, state_b, context = jnp.hsplit(state, splitter)
        return state_a, state_b, jnp.tanh(self.f_dyn.att_b(state_b)) * context

    def __call__(
            self, admission: Admission,
            embedded_admission: EmbeddedAdmissionObsSequence,
            precomputes: Precomputes) -> AdmissionPrediction:

        obs = admission.observables
        prediction = AdmissionPrediction(admission=admission)

        if len(obs) == 0:
            logging.debug("No observations to fit.")
            return prediction
        state_trajectory = tuple()
        state_a0, state_b0, context = self._init_state(empty_if_none(embedded_admission.demographic),
                                                       embedded_admission.dx_codes_history)
        # for seq_e in embedded_admission.sequence:
        #     forecasted_state = state
        #     state = self.f_dyn(seq_e, state)
        #     state_trajectory += ((forecasted_state, state),)
        #
        # forecasted_states, imputed_states = zip(*state_trajectory)
        # gru_state_trajectory = ICENODEStateTrajectory.compile(time=admission.observables.time,
        #                                                       forecasted_state=forecasted_states,
        #                                                       imputed_state=imputed_states)
        # step 1 @RETAIN paper

        # v1, v2, ..., vT
        # Merge controls with embeddings
        cv_seq = embedded_admission.sequence

        for i in range(1, len(cv_seq)):
            # e: i, ..., 1
            e_seq = []

            # beta: i, ..., 1
            b_seq = []

            state_a = state_a0
            state_b = state_b0
            for j in reversed(range(i)):
                # step 2 @RETAIN paper
                state_a = self.f_dyn.gru_a(cv_seq[j], state_a)
                e_j = self.f_dyn.att_a(state_a)
                # After the for-loop apply softmax on e_seq to get
                # alpha_seq
                e_seq.append(e_j)

                # step 3 @RETAIN paper
                h_j = state_b = self.f_dyn.gru_b(cv_seq[j], state_b)
                b_j = self.f_dyn.att_b(h_j)

                b_seq.append(jnp.tanh(b_j))

            # alpha: i, ..., 1
            a_seq = jax.nn.softmax(jnp.hstack(e_seq))

            # step 4 @RETAIN paper

            # v_i, ..., v_1
            forecasted_state = context
            context = cv_seq[:i][::-1]
            context = sum(a * (b * v)
                          for a, b, v in zip(a_seq, b_seq, context))
            state_trajectory += ((forecasted_state, context),)
            # step 5 @RETAIN paper

        forecasted_states, imputed_states = zip(*state_trajectory)
        gru_state_trajectory = ICENODEStateTrajectory.compile(time=admission.observables.time,
                                                              forecasted_state=forecasted_states,
                                                              imputed_state=imputed_states)
        prediction = prediction.add(observables=self.forecasted_observables(
            admission=admission, state_trajectory=gru_state_trajectory))
        prediction = prediction.add(leading_observable=self.f_lead_dec(gru_state_trajectory))

        return prediction


class InKoopmanPrecomputes(KoopmanPrecomputes, Precomputes):
    pass


class InKoopman(StochasticMechanisticICENODE):
    f_dyn: KoopmanOperator

    # Obs autoimputation with ICNN
    # KoopmanEmbs <-> Obs with MLP-AEs

    @staticmethod
    def _make_dyn(model_config: ICENODEConfig,
                  embeddings_config: AdmissionEmbeddingsConfig,
                  *,
                  key: jrandom.PRNGKey, observables_size: Optional[int] = None, **kwargs) -> KoopmanOperator:
        interventions_size = embeddings_config.interventions.interventions if embeddings_config.interventions else 0
        demographics_size = embeddings_config.demographic or 0
        integrand_size = model_config.state + observables_size
        return KoopmanOperator(input_size=integrand_size,
                               control_size=interventions_size + demographics_size,
                               koopman_size=integrand_size * 2,
                               key=key)

    def precomputes(self, *args, **kwargs):
        A, A_eig = self.f_dyn.compute_A()
        return InKoopmanPrecomputes(A=A, A_eig=A_eig)

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
