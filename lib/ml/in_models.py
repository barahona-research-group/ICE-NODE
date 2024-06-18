"""."""
from __future__ import annotations

import logging
from dataclasses import fields
from typing import Tuple, Optional, Literal, Type, Self, Dict, Any

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from diffrax import SaveAt
from jaxtyping import PyTree

from ._eig_ad import eig
from .artefacts import AdmissionPrediction, AdmissionsPrediction, ModelBehaviouralMetrics
from .base_models import LeadPredictorName, MonotonicLeadingObsPredictor, MLPLeadingObsPredictor, CompiledMLP, \
    NeuralODESolver, ProbMLP, CompiledLinear, CompiledGRU, ICNNObsDecoder, DiffusionMLP, StochasticNeuralODESolver, \
    ICNNObsExtractor, SkipShortIntervalsWrapper, ODEMetrics, ImputerMetrics
from .embeddings import (AdmissionEmbedding, EmbeddedAdmission)
from .embeddings import (AdmissionSequentialEmbeddingsConfig, AdmissionSequentialObsEmbedding,
                         AdmissionEmbeddingsConfig, EmbeddedAdmissionObsSequence)
from .embeddings import (DischargeSummarySequentialEmbeddingsConfig, EmbeddedDischargeSummary)
from .model import (InpatientModel, ModelConfig,
                    Precomputes)
from .state_dynamics import GRUDynamics
from .state_obs_imputers import DirectGRUStateImputer, StateObsLinearLeastSquareImpute, \
    DirectGRUStateProbabilisticImputer, StateObsICNNImputer, HiddenObsICNNImputer
from ..ehr import (Admission, InpatientObservables, CodesVector, TVxEHR)
from ..ehr.coding_scheme import GroupingData
from ..ehr.tvx_concepts import SegmentedAdmission, ObservablesDistribution
from ..ehr.tvx_ehr import SegmentedTVxEHR
from ..utils import model_params_scaler, tqdm_constructor


def empty_if_none(x):
    return x if x is not None else jnp.array([])


class InpatientModelConfig(ModelConfig):
    state: int = 50
    lead_predictor: LeadPredictorName = "monotonic"


class ICENODEStateTrajectory(InpatientObservables):

    @property
    def forecasted_state(self):
        return self.value

    @property
    def adjusted_state(self):
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
                adjusted_state: Tuple[jnp.ndarray, ...]) -> ICENODEStateTrajectory:
        forecasted_state = jnp.vstack(forecasted_state)
        adjusted_state = jnp.vstack(adjusted_state)
        return ICENODEStateTrajectory(time=time, value=forecasted_state, extra_layers=(adjusted_state,),
                                      mask=jnp.ones_like(forecasted_state, dtype=bool))


class ICENODEMetrics(ModelBehaviouralMetrics):
    ode: ODEMetrics
    imputer: ImputerMetrics


class AdmissionTrajectoryPrediction(AdmissionPrediction):
    trajectory: Optional[ICENODEStateTrajectory] = None
    model_behavioural_metrics: Optional[ICENODEMetrics] = None


class AdmissionGRUODEBayesPrediction(AdmissionTrajectoryPrediction):
    observables: Optional[ObservablesDistribution] = None
    adjusted_observables: Optional[ObservablesDistribution] = None


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
        leading_values = eqx.filter_vmap(self.predictor)(trajectory.adjusted_state)
        return InpatientObservables(time=trajectory.time, value=leading_values,
                                    mask=jnp.ones_like(leading_values, dtype=bool))


DynamicsLiteral = Literal["gru", "mlp"]


class ICENODEConfig(InpatientModelConfig):
    dynamics: DynamicsLiteral = "mlp"


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
                                width_size=model_config.state * 5,
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

    def decode_state_trajectory_observables(self,
                                            admission: SegmentedAdmission | Admission,
                                            state_trajectory: ICENODEStateTrajectory) -> InpatientObservables:
        pred_obs = eqx.filter_vmap(self.f_obs_dec)(state_trajectory.forecasted_state)
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
        state = self.f_init(jnp.hstack((embedded_admission.dx_codes_history, demo_e)))
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
                forecasted_state, ode_stats_ = self.f_dyn(state, t0=t, t1=obs_t, u=segment_force,
                                                          precomputes=precomputes,
                                                          key=subkey)
                ode_stats += ode_stats_
                forecasted_state = forecasted_state.squeeze()
                state, imputer_stats_ = self.f_update(self.f_obs_dec, forecasted_state, obs_val, obs_mask)
                imputer_stats += imputer_stats_
                state_trajectory += ((forecasted_state, state),)
                t = obs_t
            key, subkey = jrandom.split(key)
            state, stats = self.f_dyn(state, t0=t, t1=segment_t1, u=segment_force, precomputes=precomputes,
                                      key=subkey)
            ode_stats += stats
            state = state.squeeze()
            t = segment_t1

        prediction = prediction.add(model_behavioural_metrics=ICENODEMetrics(ode=ode_stats, imputer=imputer_stats))
        if self.f_outcome_dec is not None:
            prediction = prediction.add(outcome=CodesVector(self.f_outcome_dec(state), admission.outcome.scheme))
        if len(state_trajectory) > 0:
            forecasted_states, adjusted_states = zip(*state_trajectory)
            icenode_state_trajectory = ICENODEStateTrajectory.compile(time=obs.time, forecasted_state=forecasted_states,
                                                                      adjusted_state=adjusted_states)
            # TODO: test --> assert len(obs.time) == len(forecasted_states)
            prediction = prediction.add(observables=self.decode_state_trajectory_observables(
                admission=admission, state_trajectory=icenode_state_trajectory))
            prediction = prediction.add(leading_observable=self.f_lead_dec(icenode_state_trajectory))
            prediction = prediction.add(trajectory=icenode_state_trajectory)
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
                                width_size=model_config.state * 5,
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
                                width_size=integrand_size * 4,
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
    def _make_update(state_size: int, observables_size: int, key: jrandom.PRNGKey) -> HiddenObsICNNImputer:
        return HiddenObsICNNImputer(persistent_memory_size=state_size // 5)

    @staticmethod
    def _make_obs_dec(config, observables_size, key) -> ICNNObsExtractor:
        return ICNNObsExtractor(observables_size=observables_size, state_size=config.state,
                                hidden_size_multiplier=3, depth=6,
                                key=key)


class GRUODEBayes(InICENODELite):
    f_obs_dec: ProbMLP
    f_update: DirectGRUStateProbabilisticImputer

    def decode_state_trajectory_observables(self,
                                            admission: SegmentedAdmission | Admission,
                                            state_trajectory: ICENODEStateTrajectory) -> ObservablesDistribution:
        pred_obs_mean, pred_obs_std = eqx.filter_vmap(self.f_obs_dec)(state_trajectory.forecasted_state)
        return ObservablesDistribution.compile(time=state_trajectory.time, mean=pred_obs_mean, std=pred_obs_std,
                                               mask=admission.observables.mask)

    def decode_state_trajectory_adjusted_observables(self,
                                                     admission: SegmentedAdmission | Admission,
                                                     state_trajectory: ICENODEStateTrajectory) -> ObservablesDistribution:
        pred_obs_mean, pred_obs_std = eqx.filter_vmap(self.f_obs_dec)(state_trajectory.adjusted_state)
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

    def __call__(
            self, admission: SegmentedAdmission,
            embedded_admission: EmbeddedAdmission, precomputes: Precomputes
    ) -> AdmissionGRUODEBayesPrediction:
        predictions = super().__call__(admission, embedded_admission, precomputes)
        predictions = AdmissionGRUODEBayesPrediction(
            **{k.name: getattr(predictions, k.name) for k in fields(predictions)})
        if predictions.trajectory is not None:
            predictions = predictions.add(adjusted_observables=self.decode_state_trajectory_adjusted_observables(
                admission=admission, state_trajectory=predictions.trajectory))
        return predictions


class InICENODELiteLinearLeastSquareImpute(InICENODELite):
    f_update: StateObsLinearLeastSquareImpute
    f_obs_dec: CompiledLinear

    @staticmethod
    def _make_update(state_size: int, observables_size: int, key: jrandom.PRNGKey) -> StateObsLinearLeastSquareImpute:
        return StateObsLinearLeastSquareImpute()

    @staticmethod
    def _make_obs_dec(config, observables_size, key) -> CompiledLinear:
        return CompiledLinear(config.state,
                              observables_size,
                              use_bias=False,
                              key=key)


class InICENODELiteICNNImpute(InICENODELite):
    f_update: StateObsICNNImputer
    f_obs_dec: ICNNObsDecoder

    @staticmethod
    def _make_update(state_size: int, observables_size: int, key: jrandom.PRNGKey) -> StateObsICNNImputer:
        return StateObsICNNImputer(persistent_memory_size=state_size // 5)

    @staticmethod
    def _make_obs_dec(config, observables_size, key) -> ICNNObsDecoder:
        return ICNNObsDecoder(observables_size=observables_size, state_size=config.state,
                              hidden_size_multiplier=3, depth=6,
                              key=key)


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
        state = self.f_init(embedded_admission.history_summary)
        for i in range(len(obs.time)):
            forecasted_state = self.f_dyn(embedded_admission.history_summary, state)
            state = self.f_update(self.f_obs_dec, forecasted_state, obs.value[i], obs.mask[i])
            state_trajectory += ((forecasted_state, state),)

        forecasted_states, adjusted_states = zip(*state_trajectory)
        gru_state_trajectory = ICENODEStateTrajectory.compile(time=obs.time, forecasted_state=forecasted_states,
                                                              adjusted_state=adjusted_states)
        prediction = prediction.add(observables=self.decode_state_trajectory_observables(
            admission=admission, state_trajectory=gru_state_trajectory))
        prediction = prediction.add(leading_observable=self.f_lead_dec(gru_state_trajectory))
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

        state = self.f_init(force)
        state_trajectory = tuple()
        for seq_e in embedded_admission.sequence:
            forecasted_state = state
            state = self.f_dyn(seq_e, state)
            state_trajectory += ((forecasted_state, state),)

        forecasted_states, adjusted_states = zip(*state_trajectory)
        gru_state_trajectory = ICENODEStateTrajectory.compile(time=admission.observables.time,
                                                              forecasted_state=forecasted_states,
                                                              adjusted_state=adjusted_states)
        prediction = prediction.add(observables=self.decode_state_trajectory_observables(
            admission=admission, state_trajectory=gru_state_trajectory))
        prediction = prediction.add(leading_observable=self.f_lead_dec(gru_state_trajectory))
        return prediction


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
        lead = admission.leading_observable
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
        # forecasted_states, adjusted_states = zip(*state_trajectory)
        # gru_state_trajectory = ICENODEStateTrajectory.compile(time=admission.observables.time,
        #                                                       forecasted_state=forecasted_states,
        #                                                       adjusted_state=adjusted_states)
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

        forecasted_states, adjusted_states = zip(*state_trajectory)
        gru_state_trajectory = ICENODEStateTrajectory.compile(time=admission.observables.time,
                                                              forecasted_state=forecasted_states,
                                                              adjusted_state=adjusted_states)
        prediction = prediction.add(observables=self.decode_state_trajectory_observables(
            admission=admission, state_trajectory=gru_state_trajectory))
        prediction = prediction.add(leading_observable=self.f_lead_dec(gru_state_trajectory))

        return prediction


class InKoopmanPrecomputes(Precomputes):
    A: jnp.ndarray
    A_eig: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]


class KoopmanPhi(eqx.Module):
    """Koopman embeddings for continuous-time systems."""
    mlp: eqx.Module
    C: jnp.ndarray = eqx.static_field()
    skip: bool = eqx.static_field()

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 key: "jax.random.PRNGKey",
                 depth: int,
                 control_size: int = 0,
                 skip: bool = True):
        super().__init__()
        self.skip = skip
        self.C = jnp.eye(output_size, M=input_size + control_size)
        self.mlp = eqx.nn.MLP(
            input_size + control_size,
            output_size,
            depth=depth,
            width_size=(input_size + control_size + output_size) // 2,
            activation=jnn.relu,
            use_final_bias=not skip,
            key=key)

    @eqx.filter_jit
    def __call__(self, x, u=None):
        if u is not None:
            x = jnp.hstack((x, u))

        if self.skip:
            return self.C @ x + self.mlp(x)
        else:
            return self.mlp(x)


class VanillaKoopmanOperator(eqx.Module):
    A: jnp.ndarray
    phi: KoopmanPhi
    phi_inv: KoopmanPhi

    input_size: int = eqx.static_field()
    koopman_size: int = eqx.static_field()
    control_size: int = eqx.static_field()

    def __init__(self,
                 input_size: int,
                 koopman_size: int,
                 key: "jax.random.PRNGKey",
                 control_size: int = 0,
                 phi_depth: int = 1):
        super().__init__()
        self.input_size = input_size
        self.koopman_size = koopman_size
        self.control_size = control_size
        keys = jrandom.split(key, 3)

        self.A = jrandom.normal(keys[0], (koopman_size, koopman_size),
                                dtype=jnp.float32)
        self.phi = KoopmanPhi(input_size,
                              koopman_size,
                              control_size=control_size,
                              depth=phi_depth,
                              skip=True,
                              key=keys[1])
        self.phi_inv = KoopmanPhi(koopman_size,
                                  input_size,
                                  depth=phi_depth,
                                  skip=False,
                                  key=keys[2])

    @eqx.filter_jit
    def compute_A(self) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        lam, V = eig(self.A)
        V_inv = jnp.linalg.solve(V @ jnp.diag(lam), self.A)
        return self.A, (lam, V, V_inv)

    def K_operator(self, t: float, z: jnp.ndarray,
                   A_eig: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        assert z.ndim == 1, f"z must be a vector, got {z.ndim}D"
        (lam, V, V_inv) = A_eig
        lam_t = jnp.exp(lam * t)
        complex_z = V @ (lam_t * (V_inv @ z))
        return complex_z.real

    @eqx.filter_jit
    def __call__(self, x0, t0: float, t1: float,
                 u: Optional[PyTree],
                 precomputes: InKoopmanPrecomputes, saveat: Optional[SaveAt]) -> jnp.ndarray:
        z = self.phi(x0, u=u)
        z = self.K_operator(t1, z, precomputes.A_eig)
        return self.phi_inv(z)

    @eqx.filter_jit
    def compute_phi_loss(self, x, u=None):
        z = self.phi(x, u=u)
        diff = x - self.phi_inv(z)
        return jnp.mean(diff ** 2)

    def compute_A_spectrum(self):
        _, (lam, _, _) = self.compute_A()
        return lam.real, lam.imag


class KoopmanOperator(VanillaKoopmanOperator):
    """Koopman operator for continuous-time systems."""
    R: jnp.ndarray
    Q: jnp.ndarray
    N: jnp.ndarray
    epsI: jnp.ndarray = eqx.static_field()

    def __init__(self,
                 input_size: int,
                 koopman_size: int,
                 key: "jax.random.PRNGKey",
                 control_size: int = 0,
                 phi_depth: int = 3):
        superkey, key = jrandom.split(key, 2)
        super().__init__(input_size=input_size,
                         koopman_size=koopman_size,
                         key=superkey,
                         control_size=control_size,
                         phi_depth=phi_depth)
        self.A = None
        keys = jrandom.split(key, 3)

        self.R = jrandom.normal(keys[0], (koopman_size, koopman_size),
                                dtype=jnp.float64)
        self.Q = jrandom.normal(keys[1], (koopman_size, koopman_size),
                                dtype=jnp.float64)
        self.N = jrandom.normal(keys[2], (koopman_size, koopman_size),
                                dtype=jnp.float64)
        self.epsI = 1e-9 * jnp.eye(koopman_size, dtype=jnp.float64)

        assert all(a.dtype == jnp.float64 for a in (self.R, self.Q, self.N)), \
            "SKELKoopmanOperator requires float64 precision"

    @eqx.filter_jit
    def compute_A(self):
        R = self.R
        Q = self.Q
        N = self.N

        skew = (R - R.T) / 2
        F = skew - Q @ Q.T - self.epsI
        E = N @ N.T + self.epsI

        A = jnp.linalg.solve(E, F)

        lam, V = eig(A)
        V_inv = jnp.linalg.solve(V @ jnp.diag(lam), A)
        return A, (lam, V, V_inv)


class InKoopman(InICENODELite):
    f_dyn: KoopmanOperator

    @staticmethod
    def _make_dyn(model_config: InpatientModelConfig, embeddings_config: AdmissionEmbeddingsConfig,
                  key: jrandom.PRNGKey) -> KoopmanOperator:
        interventions_size = embeddings_config.interventions.interventions if embeddings_config.interventions else 0
        demographics_size = embeddings_config.demographic
        return KoopmanOperator(input_size=model_config.state,
                               control_size=interventions_size + demographics_size,
                               koopman_size=model_config.state * 5,
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


class InMultiscaleKoopman(InICENODELite):
    pass


# TODO: finish this class


class TimestepsKoopmanPrecomputes(Precomputes):
    pass
    # Precompute a number of operators for given timesteps. https://www.geeksforgeeks.org/find-minimum-number-of-coins-that-make-a-change/
    # Use the minimum-number-of-coins-needed principle to determine the matrix-vector multiplication sequence.
