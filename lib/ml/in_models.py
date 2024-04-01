"""."""
from __future__ import annotations

from typing import Callable, Tuple, Optional, Literal, Type, Union, Final

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from diffrax import diffeqsolve, Tsit5, RecursiveCheckpointAdjoint, SaveAt, ODETerm, Solution, PIDController, SubSaveAt
from jaxtyping import PyTree

from .artefacts import AdmissionPrediction
from .base_models_koopman import SKELKoopmanOperator, VanillaKoopmanOperator
from .embeddings import (AdmissionEmbedding, AdmissionEmbeddingsConfig, EmbeddedAdmission)
from .embeddings import (AdmissionSequentialEmbeddingsConfig, AdmissionSequentialObsEmbedding,
                         SpecialisedAdmissionEmbeddingsConfig, EmbeddedAdmissionObsSequence)
from .embeddings import (DischargeSummarySequentialEmbeddingsConfig, DischargeSummarySequentialEmbedding,
                         EmbeddedDischargeSummary)
from .model import (InpatientModel, ModelConfig, ModelRegularisation,
                    Precomputes)
from ..ehr import (Admission, InpatientObservables, DatasetScheme, DemographicVectorConfig, CodesVector,
                   LeadingObservableExtractorConfig)
from ..ehr.coding_scheme import GroupingData
from ..ehr.tvx_concepts import SegmentedAdmission
from ..utils import model_params_scaler

LeadPredictorName = Literal['monotonic', 'mlp']


class CompiledMLP(eqx.nn.MLP):
    # Just an eqx.nn.MLP with a compiled __call__ method.

    @eqx.filter_jit
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return super().__call__(x)


class InpatientModelConfig(ModelConfig):
    state: int = 50
    lead_predictor: LeadPredictorName = "monotonic"


class ForcedVectorField(eqx.Module):
    mlp: eqx.nn.MLP

    @eqx.filter_jit
    def __call__(self, t: float, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return self.mlp(jnp.hstack((x, u)))


class NeuralODESolver(eqx.Module):
    f: ForcedVectorField
    SECOND: Final[float] = 1 / 3600.0  # Time units in one second.
    DT0: Final[float] = 60.0  # Initial time step in seconds.

    @staticmethod
    def from_mlp(mlp: eqx.nn.MLP, second: float = 1 / 3600.0, dt0: float = 60.0):
        return NeuralODESolver(f=ForcedVectorField(mlp), SECOND=second, DT0=dt0)

    @property
    def zero_force(self) -> jnp.ndarray:
        in_size = self.f.mlp.layers[0].weight.shape[1]
        out_size = self.f.mlp.layers[-1].weight.shape[0]
        return jnp.zeros((in_size - out_size,))

    @property
    def ode_term(self) -> ODETerm:
        return ODETerm(self.f)

    @eqx.filter_jit
    def __call__(self, x0, t0: float, t1: float, saveat: Optional[SaveAt] = None,
                 u: Optional[PyTree] = None,
                 precomputes: Optional[Precomputes] = None) -> Union[jnp.ndarray, Tuple[jnp.ndarray, ...]]:
        sol = diffeqsolve(
            terms=self.ode_term,
            solver=Tsit5(),
            t0=t0,
            t1=t1,
            dt0=self.DT0 * self.SECOND,
            y0=self.get_aug_x0(x0, precomputes),
            args=self.get_args(x0, u, precomputes),
            adjoint=RecursiveCheckpointAdjoint(),
            saveat=saveat or SaveAt(t1=True),
            stepsize_controller=PIDController(rtol=1.4e-8, atol=1.4e-8),
            throw=False,
            max_steps=None)
        return self.get_solution(sol)

    def get_args(self, x0: jnp.ndarray, u: Optional[jnp.ndarray], precomputes: Optional[Precomputes]) -> PyTree:
        return u or self.zero_force

    def get_aug_x0(self, x0: jnp.ndarray, precomputes: Precomputes) -> PyTree:
        return x0

    def get_solution(self, sol: Solution) -> Union[jnp.ndarray, Tuple[jnp.ndarray, ...]]:
        return sol.ys


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


class InICENODEStateAdjust(eqx.Module):
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
    def __call__(self, forecasted_state: jnp.ndarray,
                 forecasted_observables: jnp.ndarray,
                 true_observables: jnp.ndarray, observables_mask: jnp.ndarray) -> jnp.ndarray:
        error = jnp.where(observables_mask, forecasted_observables - true_observables, 0.0)
        projected_error = self.f_project_error(error)
        return self.f_update(jnp.hstack(observables_mask, projected_error), forecasted_state)


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


class AdmissionTrajectoryPrediction(AdmissionPrediction):
    trajectory: Optional[ICENODEStateTrajectory] = None


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
        leading_values = eqx.filter_vmap(self.predictor)(trajectory.forecasted_state)
        return InpatientObservables(time=trajectory.time, value=leading_values,
                                    mask=jnp.ones_like(leading_values, dtype=bool))


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
    f_update: InICENODEStateAdjust
    f_init: CompiledMLP
    f_outcome_dec: Optional[CompiledMLP] = None

    config: InpatientModelConfig = eqx.static_field()
    regularisation: ModelRegularisation = eqx.static_field(default=ModelRegularisation())

    def __init__(self, config: InpatientModelConfig,
                 embeddings_config: SpecialisedAdmissionEmbeddingsConfig,
                 lead_times: Tuple[float, ...],
                 dx_codes_size: Optional[int] = None,
                 outcome_size: Optional[int] = None,
                 icu_inputs_grouping: Optional[GroupingData] = None,
                 icu_procedures_size: Optional[int] = None,
                 hosp_procedures_size: Optional[int] = None,
                 demographic_size: Optional[int] = None,
                 observables_size: Optional[int] = None, *,
                 key: "jax.random.PRNGKey", **lead_mlp_kwargs):
        super().__init__(config=config)
        emb_config = embeddings_config.to_admission_embeddings_config()

        (emb_key, obs_dec_key, lead_key, outcome_dec_key, dyn_key,
         update_key) = jrandom.split(key, 6)
        self.f_emb = self._make_embedding(config=emb_config,
                                          dx_codes_size=dx_codes_size,
                                          icu_inputs_grouping=icu_inputs_grouping,
                                          icu_procedures_size=icu_procedures_size,
                                          hosp_procedures_size=hosp_procedures_size,
                                          demographic_size=demographic_size,
                                          observables_size=observables_size,
                                          key=key)
        self.f_init = self._make_init(embeddings_config=emb_config,

                                      state_size=config.state,
                                      key=emb_key)
        self.f_outcome_dec = self._make_outcome_dec(state_size=config.state,
                                                    outcome_size=outcome_size,
                                                    key=outcome_dec_key)

        self.f_obs_dec = self._make_obs_dec(config=config,
                                            observables_size=observables_size,
                                            key=obs_dec_key)
        self.f_dyn = self._make_dyn(state_size=config.state,
                                    embeddings_config=emb_config,
                                    key=dyn_key)

        self.f_update = self._make_update(state_size=config.state,
                                          observables_size=observables_size,
                                          key=update_key)

        self.f_lead_dec = self._make_lead_dec(input_size=config.state,
                                              lead_times=lead_times,
                                              predictor=config.lead_predictor,
                                              key=lead_key, **lead_mlp_kwargs)

    @staticmethod
    def _make_init(embeddings_config: Union[AdmissionEmbeddingsConfig, AdmissionSequentialEmbeddingsConfig],
                   state_size: int, key: jrandom.PRNGKey) -> CompiledMLP:
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
            key: jrandom.PRNGKey, **mlp_kwargs
    ) -> DirectLeadPredictorWrapper:
        return DirectLeadPredictorWrapper(input_size, lead_times, predictor, key, **mlp_kwargs)

    @staticmethod
    def _make_dyn(state_size: int, embeddings_config: AdmissionEmbeddingsConfig,
                  key: jrandom.PRNGKey) -> NeuralODESolver:
        interventions_size = embeddings_config.interventions.interventions if embeddings_config.interventions else 0
        demographics_size = embeddings_config.demographic
        f_dyn = eqx.nn.MLP(in_size=state_size + interventions_size + demographics_size,
                           out_size=state_size,
                           activation=jnn.tanh,
                           depth=2,
                           width_size=state_size * 5,
                           key=key)
        f_dyn = model_params_scaler(f_dyn, 1e-2, eqx.is_inexact_array)
        return NeuralODESolver.from_mlp(mlp=f_dyn, second=1 / 3600.0, dt0=60.0)

    @staticmethod
    def _make_update(state_size: int, observables_size: int, key: jrandom.PRNGKey) -> InICENODEStateAdjust:
        return InICENODEStateAdjust(state_size, observables_size, key=key)

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

    def decode_state_trajectory_leading_observables(self,
                                                    admission: SegmentedAdmission | Admission,
                                                    state_trajectory: ICENODEStateTrajectory) -> InpatientObservables:
        pred_lead = eqx.filter_vmap(self.f_lead_dec)(state_trajectory.adjusted_state)
        return InpatientObservables(time=state_trajectory.time, value=pred_lead, mask=admission.leading_observable.mask)

    def __call__(
            self, admission: SegmentedAdmission,
            embedded_admission: EmbeddedAdmission, precomputes: Precomputes
    ) -> AdmissionTrajectoryPrediction:
        prediction = AdmissionTrajectoryPrediction(admission=admission)
        int_e = embedded_admission.interventions or jnp.array([[]])
        demo_e = embedded_admission.demographic or jnp.array([])
        obs = admission.observables
        n_segments = obs.n_segments
        t = 0.0
        state_trajectory = tuple()
        state = self.f_init(jnp.hstack(embedded_admission.dx_codes_history, embedded_admission.demographic))
        segments_t1 = admission.interventions.t1
        for segment_index in range(n_segments):
            segment_t1 = segments_t1[segment_index]
            segment_obs = obs[segment_index]
            segment_interventions = int_e[segment_index]
            segment_force = jnp.hstack((demo_e, segment_interventions))

            for obs_t, obs_val, obs_mask in segment_obs:
                # if time-diff is more than 1 seconds, we integrate.
                forecasted_state = self.f_dyn(state, t0=t, t1=obs_t, u=segment_force, precomputes=precomputes)
                state = self.f_update(forecasted_state, self.f_obs_dec(state), obs_val, obs_mask)
                state_trajectory += ((forecasted_state, state),)
                t = obs_t
            state = self.f_dyn(state, t0=t, t1=segment_t1, u=segment_force, precomputes=precomputes)
            t = segment_t1

        if self.f_outcome_dec is not None:
            prediction = prediction.add(outcome=CodesVector(self.f_dx_dec(state), admission.outcome.scheme))
        if len(state_trajectory) > 0:
            forecasted_states, adjusted_states = zip(*state_trajectory)
            icenode_state_trajectory = ICENODEStateTrajectory.compile(time=obs.time, forecasted_state=forecasted_states,
                                                                      adjusted_state=adjusted_states)
            # TODO: test --> assert len(obs.time) == len(forecasted_states)
            prediction = prediction.add(observables=self.decode_state_trajectory_observables(
                admission=admission, state_trajectory=icenode_state_trajectory))
            prediction = prediction.add(leading_observables=self.decode_state_trajectory_leading_observables(
                admission=admission, state_trajectory=icenode_state_trajectory))
        return prediction

    @property
    def dyn_params_list(self):
        return self.params_list(self.f_dyn)


class InICENODELite(InICENODE):
    # Same as InICENODE but without discharge summary outcome predictions.
    def __init__(self, config: InpatientModelConfig,
                 embeddings_config: Union[AdmissionEmbeddingsConfig, AdmissionSequentialEmbeddingsConfig],
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


class AugmentedForcedVectorField(ForcedVectorField):
    mlp: eqx.nn.MLP  # should be shared with f_dyn.

    def __call__(self, t: float, x: PyTree, u: PyTree) -> PyTree:
        force, zero_force = u
        x_f, x_a = x
        dx_f = self.mlp(jnp.hstack((x_f, force)))
        dx_a = self.mlp(jnp.hstack((x_a, zero_force)))
        return dx_f, dx_a


class TrajectoryInterventionEffectEstimator(NeuralODESolver):
    f: AugmentedForcedVectorField

    def get_args(self, x0: PyTree, u: PyTree, precomputes: Optional[Precomputes]) -> PyTree:
        forced, unforced = u
        return forced, unforced

    def get_aug_x0(self, x0: PyTree, precomputes: Precomputes) -> PyTree:
        return x0, x0

    @staticmethod
    def from_mlp(mlp: eqx.nn.MLP, second: float = 1 / 3600.0, dt0: float = 60.0):
        raise NotImplementedError("This method is disabled for this class. Use from_shared_dyn instead.")

    @staticmethod
    def from_shared_dyn(shared_dyn: NeuralODESolver):
        return TrajectoryInterventionEffectEstimator(f=AugmentedForcedVectorField(mlp=shared_dyn.f.mlp),
                                                     SECOND=shared_dyn.SECOND, DT0=shared_dyn.DT0)


class InterventionUncertaintyWeightingScheme(eqx.Module):  # Uncertainty-Aware InICENODE
    leading_observable_index: int
    lead_times: Tuple[float, ...]

    def _estimate_intervention_effect(self,
                                      f_obs_decoder: eqx.nn.MLP,
                                      f_ua_dyn: TrajectoryInterventionEffectEstimator,
                                      timestamp_index, initial_state: jnp.ndarray,
                                      admission: SegmentedAdmission,
                                      embdedded_interventions: jnp.ndarray,
                                      embedded_demographic: Optional[jnp.ndarray],
                                      precomputes: Precomputes) -> Tuple[Tuple[float, ...], Tuple[int, jnp.ndarray]]:

        """
        Generate supplementary statistics relative to the timestamp_index of the admission observables, which
        will be used to estimate the intervention effect as an uncertainty proxy for the outcome prediction.
        The outcome prediction at time t is assumed to match up with the delayed response at
        admission.leading_observables[timestamp_index], which is a strong assumption due to:
            - The interventions applied between t and the delayed response time (t + lead_time), can mitigate
                the delayed response per se, so it will be non-ideal to penalize a positive outcome prediction
                 as a false positive, while in fact it is a correct prediction of the outcome.
            - In contrast, the interventions themselves might solely induce the patient's condition, so it
                will be misleading to penalize the model for negative outcome prediction as false negative,
                while in fact it is a correct prediction of a missing outcome.
        So this function will estimate the effect of the interventions by producing:
            - A maximum absolute difference between (a) the predicted delayed response under interventions
                (calling it forced observable response) with (b) the predicted delayed response with interventions
                masked-out (calling it autonomous response). The demographics are not considered in the intervention
                effect estimation.
            - The difference between the predicted forced observable response and the ground-truth response. It
                will be used to improve the model prediction of the intervention effect.
        """
        # Current state value/time.
        state = initial_state
        state_t = admission.observables.time[timestamp_index]

        # Collect the observable of interest.
        observables = admission.observables
        obs_mask = observables.mask[:, self.leading_observable_index]
        obs_values = observables.value[obs_mask][:, self.leading_observable_index]
        obs_times = observables.time[obs_mask]

        # Demographics and masked-out interventions.
        demo_e = embedded_demographic or jnp.array([])
        no_intervention_u = jnp.hstack((demo_e, jnp.zeros_like(embdedded_interventions[0])))

        # the defined delays (t+L) for all L in lead_times.
        delays = tuple(state_t + lead_time for lead_time in self.lead_times)
        # grid of delays to probe the intervention max effect.
        delays_grid = tuple(jnp.linspace(d_t0, d_t1, 10) for d_t0, d_t1 in zip((state_t,) + delays[:-1], delays))

        # timestamps where ground-truth observables are available.
        obs_times = tuple(t for t in obs_times if state_t < t <= delays[-1])
        forced_obs_pred_diff = tuple()  # forced observable prediction at obs_times.
        predicted_intervention_effect = tuple()  # Stores max absolute difference between forced and autonomous preds.
        current_intervention_effect = 0.0  # Stores the last max absolute difference between forced and autonomous preds.
        for segment_index in range(observables.n_segments):
            segment_t1 = admission.interventions.t1[segment_index]
            if state_t < segment_t1:
                continue
            segment_interventions = embdedded_interventions[segment_index]
            intervention_u = jnp.hstack((demo_e, segment_interventions))
            segment_delay_grid = tuple(SubSaveAt(ts=delays_grid) for delay_t, delay_grid in
                                       zip(delays, delays_grid) if state_t < delay_t <= segment_t1)
            # Limit to 5, less shape variations, less JITs.
            obs_segment_ts = tuple(t for t in obs_times if state_t < t <= segment_t1)[:5]
            segment_obs_times = SubSaveAt(ts=obs_segment_ts) if len(obs_segment_ts) > 0 else None

            saveat = SaveAt(subs=(SubSaveAt(t1=True), segment_obs_times, segment_delay_grid))
            state, obs_ts_state, delay_ts_state = f_ua_dyn(state, t0=state_t, t1=segment_t1, saveat=saveat,
                                                           u=(intervention_u, no_intervention_u),
                                                           precomputes=precomputes)
            if obs_ts_state:
                forced_state, _ = obs_ts_state
                forced_obs_pred = eqx.filter_vmap(f_obs_decoder)(forced_state)[:, self.leading_observable_index]
                forced_obs_pred_diff += (forced_obs_pred.squeeze() - obs_values[obs_times][:5],)

            for delayed_state_grid in delay_ts_state:
                forced_state, auto_state = delayed_state_grid
                forced_delayed_pred = eqx.filter_vmap(f_obs_decoder)(forced_state)[:, self.leading_observable_index]
                auto_delayed_pred = eqx.filter_vmap(f_obs_decoder)(auto_state)[:, self.leading_observable_index]
                grid_max_effect = jnp.max(jnp.abs(forced_delayed_pred - auto_delayed_pred))
                current_intervention_effect = jnp.maximum(current_intervention_effect, grid_max_effect)
                predicted_intervention_effect += (current_intervention_effect,)

            state_t = segment_t1
            if state_t >= delays[-1]:
                break

        forced_prediction_l2 = jnp.mean(jnp.hstack(forced_obs_pred_diff) ** 2)
        forced_prediction_n = sum(map(len, forced_obs_pred_diff))

        assert len(predicted_intervention_effect) <= len(self.lead_times)
        predicted_intervention_effect += (current_intervention_effect,) * (
                len(self.lead_times) - len(predicted_intervention_effect))
        return predicted_intervention_effect, (forced_prediction_n, forced_prediction_l2)

    def __call__(self,
                 f_obs_decoder: eqx.nn.MLP,
                 f_ode_dyn: NeuralODESolver,
                 initial_states: jnp.ndarray,
                 admission: SegmentedAdmission,
                 embedded_admission: EmbeddedAdmission, precomputes: Precomputes) -> Tuple[
        Tuple[float, int], InpatientObservables]:
        intervention_effect = tuple()
        forced_prediction_l2 = tuple()
        f_uncertainty_dyn = TrajectoryInterventionEffectEstimator.from_shared_dyn(f_ode_dyn)
        assert len(admission.observables.time) == len(initial_states)
        for i, (_, _, mask) in enumerate(admission.leading_observable):
            if mask.sum() == 0:
                intervention_effect += ((0.0,) * len(self.lead_times),)
                forced_prediction_l2 += ((0, 0.0),)
            else:
                estimands = self._estimate_intervention_effect(f_obs_decoder, f_uncertainty_dyn,
                                                               i, initial_states[i],
                                                               admission, embedded_admission.interventions,
                                                               embedded_admission.demographic,
                                                               precomputes)
                intervention_effect += (estimands[0],)
                forced_prediction_l2 += (estimands[1],)
        intervention_effect_array = jnp.array(intervention_effect)
        intervention_effect_struct = InpatientObservables(time=admission.leading_observable.time,
                                                          value=intervention_effect_array,
                                                          mask=admission.leading_observable.mask)
        forced_prediction_l2, n = zip(*forced_prediction_l2)
        sum_n = sum(n)
        forced_prediction_l2_mean = sum(l2 * n / sum_n for l2, n in zip(forced_prediction_l2, n))
        return (forced_prediction_l2_mean, sum_n), intervention_effect_struct


class CompiledGRU(eqx.nn.GRUCell):
    @eqx.filter_jit
    def __call__(self, x: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
        return super().__call__(h, x)


class InGRUJump(InICENODELite):
    # TODO: as for the original paper, use multi-layer embeddings with skip
    # connections.
    f_emb: DischargeSummarySequentialEmbedding
    # GRU. Alternatives: SRU, LSTM, ..
    f_dyn: CompiledGRU

    def __init__(self, config: InpatientModelConfig,
                 embeddings_config: DischargeSummarySequentialEmbeddingsConfig,
                 lead_times: Tuple[float, ...],
                 dx_codes_size: Optional[int] = None,
                 demographic_size: Optional[int] = None,
                 observables_size: Optional[int] = None, *,
                 key: "jax.random.PRNGKey"):
        super().__init__(config=config, embeddings_config=embeddings_config.to_admission_embeddings_config(),
                         lead_times=lead_times,
                         dx_codes_size=dx_codes_size, icu_inputs_grouping=None,
                         icu_procedures_size=None, hosp_procedures_size=None,
                         demographic_size=demographic_size, observables_size=observables_size, key=key)

    @staticmethod
    def _make_init(embeddings_config: DischargeSummarySequentialEmbeddingsConfig,
                   state_size: int, key: jrandom.PRNGKey) -> CompiledMLP:
        return CompiledMLP(embeddings_config.summary,
                           state_size,
                           state_size * 3,
                           depth=2,
                           key=key)

    @staticmethod
    def _make_embedding(config: DischargeSummarySequentialEmbeddingsConfig,
                        dx_codes_size: Optional[int] = None,
                        demographic_size: Optional[int] = None, *,
                        key: jrandom.PRNGKey, **kwargs) -> DischargeSummarySequentialEmbedding:
        return DischargeSummarySequentialEmbedding(config=config,
                                                   dx_codes_size=dx_codes_size,
                                                   demographic_size=demographic_size,
                                                   key=key)

    @staticmethod
    def _make_dyn(state_size: int, embeddings_config: DischargeSummarySequentialEmbeddingsConfig,
                  key: jrandom.PRNGKey) -> CompiledGRU:
        return CompiledGRU(input_size=embeddings_config.summary,
                           hidden_size=state_size,
                           key=key)

    def __call__(self, admission: Admission, embedded_admission: EmbeddedDischargeSummary,
                 precomputes: Precomputes) -> AdmissionPrediction:
        prediction = AdmissionPrediction(admission=admission)
        obs = admission.observables
        state_trajectory = tuple()
        state = self.f_init(embedded_admission.history_summary)
        for i in range(len(obs.time)):
            forecasted_state = self.f_dyn(embedded_admission.history_summary, state)
            state = self.f_update(forecasted_state, self.f_obs_dec(state), obs.value[i], obs.mask[i])
            state_trajectory += ((forecasted_state, state),)

        forecasted_states, adjusted_states = zip(*state_trajectory)
        gru_state_trajectory = ICENODEStateTrajectory.compile(time=obs.time, forecasted_state=forecasted_states,
                                                              adjusted_state=adjusted_states)
        prediction = prediction.add(observables=self.decode_state_trajectory_observables(
            admission=admission, state_trajectory=gru_state_trajectory))
        prediction = prediction.add(leading_observables=self.decode_state_trajectory_leading_observables(
            admission=admission, state_trajectory=gru_state_trajectory))
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
        super().__init__(config=config, embeddings_config=embeddings_config.to_admission_embeddings_config(),
                         lead_times=lead_times,
                         dx_codes_size=dx_codes_size, icu_inputs_grouping=None,
                         icu_procedures_size=None, hosp_procedures_size=None,
                         demographic_size=demographic_size, observables_size=observables_size, key=key)

    @staticmethod
    def _make_dyn(state_size: int, embeddings_config: AdmissionSequentialEmbeddingsConfig,
                  key: jrandom.PRNGKey) -> CompiledGRU:
        return CompiledGRU(input_size=embeddings_config.sequence,
                           hidden_size=state_size,
                           key=key)

    def __call__(
            self, admission: Admission,
            embedded_admission: EmbeddedAdmissionObsSequence, precomputes: Precomputes) -> AdmissionPrediction:
        emb_dx_history = embedded_admission.dx_codes_history or jnp.array([])
        emb_demo = embedded_admission.demographic or jnp.array([])
        force = jnp.hstack((emb_dx_history, emb_demo))
        prediction = AdmissionPrediction(admission=admission)
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
        prediction = prediction.add(leading_observables=self.decode_state_trajectory_leading_observables(
            admission=admission, state_trajectory=gru_state_trajectory))
        return prediction


class InRETAINConfig(ModelConfig):
    mem_a: int = eqx.static_field(default=45)
    mem_b: int = eqx.static_field(default=45)
    lead_predictor: str = "monotonic"


class InRETAIN(InGRUJump):
    f_gru_a: Callable
    f_gru_b: Callable
    f_att_a: Callable
    f_att_b: Callable
    f_lead_dec: Callable
    f_obs_dec: Callable
    f_dx_dec: Callable
    f_init: Callable
    f_dym: Callable = None
    config: InRETAINConfig = eqx.static_field()

    def __init__(self, config: InRETAINConfig,
                 embeddings_config: AdmissionSequentialEmbeddingsConfig,
                 lead_times: Tuple[float, ...],
                 dx_codes_size: Optional[int] = None,
                 demographic_size: Optional[int] = None,
                 observables_size: Optional[int] = None, *,
                 key: "jax.random.PRNGKey"):

        keys = jrandom.split(key, 9)
        f_emb = InGRU._make_embedding(
            config=embeddings_config,
            dx_codes_size=dx_codes_size,
            icu_inputs_grouping=None,
            icu_procedures_size=None,
            hosp_procedures_size=None,
            demographic_size=demographic_size,
            observables_size=observables_size,
            key=key)
        f_dx_dec = self._make_dx_dec(config=config,
                                     dx_size=len(schemes[1].outcome),
                                     key=keys[1])

        self.f_lead_dec = InICENODE._make_lead_dec(
            config=config,
            input_size=config.emb.sequence,
            leading_observable_config=leading_observable_config,
            key=keys[2])
        self.f_obs_dec = self._make_obs_dec(config=config,
                                            obs_size=len(schemes[1].obs),
                                            key=keys[3])

        self.f_gru_a = eqx.nn.GRUCell(config.emb.sequence,
                                      config.mem_a,
                                      use_bias=True,
                                      key=keys[4])
        self.f_gru_b = eqx.nn.GRUCell(config.emb.sequence,
                                      config.mem_b,
                                      use_bias=True,
                                      key=keys[5])

        self.f_att_a = eqx.nn.Linear(config.mem_a,
                                     1,
                                     use_bias=True,
                                     key=keys[6])
        self.f_att_b = eqx.nn.Linear(config.mem_b,
                                     config.emb.sequence,
                                     use_bias=True,
                                     key=keys[7])

        self.f_init = self._make_init(config=config, key=keys[8])

        super().__init__(config=config, f_emb=f_emb, f_dx_dec=f_dx_dec)

    @staticmethod
    def _make_dyn(config, key, **kwargs):
        return eqx.nn.GRUCell(input_size=config.emb.sequence,
                              hidden_size=config.state,
                              key=key)

    @staticmethod
    def _make_init(config, key):
        init_input_size = config.emb.demo + config.emb.dx_discharge
        return eqx.nn.MLP(init_input_size,
                          config.mem_a + config.mem_b + config.emb.sequence,
                          (config.mem_a + config.mem_b) * 3,
                          depth=2,
                          key=key)

    @staticmethod
    def _make_dx_dec(config, dx_size, key):
        return eqx.nn.MLP(config.emb.sequence,
                          dx_size,
                          config.emb.dx_discharge * 5,
                          depth=1,
                          key=key)

    @staticmethod
    def _make_obs_dec(config, obs_size, key):
        return eqx.nn.MLP(config.emb.sequence,
                          obs_size,
                          obs_size * 5,
                          depth=1,
                          key=key)

    @property
    def dyn_params_list(self):
        return self.params_list(
            (self.f_gru_a, self.f_gru_b, self.f_att_a, self.f_att_b))

    @eqx.filter_jit
    def _gru_a(self, x, state):
        return self.f_gru_a(x, state)

    @eqx.filter_jit
    def _gru_b(self, x, state):
        return self.f_gru_b(x, state)

    @eqx.filter_jit
    def _att_a(self, x):
        return self.f_att_a(x)

    @eqx.filter_jit
    def _att_b(self, x):
        return self.f_att_b(x)

    @eqx.filter_jit
    def _dx_dec(self, x):
        return self.f_dx_dec(x)

    @eqx.filter_jit
    def _lead_dec(self, x):
        return self.f_lead_dec(x)

    @eqx.filter_jit
    def _obs_dec(self, x):
        return self.f_obs_dec(x)

    @eqx.filter_jit
    def _init_state(self, demo_emb, dx0):
        state = self.f_init(jnp.hstack((demo_emb, dx0)))
        splitter = (self.config.mem_a, self.config.mem_a + self.config.mem_b)
        state_a, state_b, context = jnp.hsplit(state, splitter)
        return state_a, state_b, jnp.tanh(self._att_b(state_b)) * context

    def __call__(
            self, admission: Admission,
            embedded_admission: DeepMindEmbeddedAdmission,
            precomputes: Precomputes, regularisation: ModelRegularisation,
            store_embeddings: Optional[TrajectoryConfig]
    ) -> AdmissionPrediction:

        obs = admission.observables
        lead = admission.leading_observable

        if len(obs) == 0:
            pred_obs = InpatientObservables.empty(obs.value.shape[1])
            pred_lead = InpatientObservables.empty(lead.value.shape[1])
            pred_dx = CodesVector(jnp.zeros(len(admission.outcome.scheme)),
                                  admission.outcome.scheme)
            return AdmissionPrediction(admission=admission,
                                       outcome=pred_dx,
                                       observables=pred_obs,
                                       leading_observable=pred_lead,
                                       trajectory=None)

        state_a0, state_b0, context = self._init_state(embedded_admission.demo,
                                                       embedded_admission.dx0)

        # step 1 @RETAIN paper

        # v1, v2, ..., vT
        # Merge controls with embeddings
        cv_seq = embedded_admission.sequence

        pred_obs_l = [self._obs_dec(context)]
        pred_lead_l = [self._lead_dec(context)]

        for i in range(1, len(cv_seq)):
            # e: i, ..., 1
            e_seq = []

            # beta: i, ..., 1
            b_seq = []

            state_a = state_a0
            state_b = state_b0
            for j in reversed(range(i)):
                # step 2 @RETAIN paper
                state_a = self._gru_a(cv_seq[j], state_a)
                e_j = self._att_a(state_a)
                # After the for-loop apply softmax on e_seq to get
                # alpha_seq

                e_seq.append(e_j)

                # step 3 @RETAIN paper
                h_j = state_b = self._gru_b(cv_seq[j], state_b)
                b_j = self._att_b(h_j)

                b_seq.append(jnp.tanh(b_j))

            # alpha: i, ..., 1
            a_seq = jax.nn.softmax(jnp.hstack(e_seq))

            # step 4 @RETAIN paper

            # v_i, ..., v_1
            context = cv_seq[:i][::-1]
            context = sum(a * (b * v)
                          for a, b, v in zip(a_seq, b_seq, context))

            # step 5 @RETAIN paper
            pred_obs_l.append(self._obs_dec(context))
            pred_lead_l.append(self._lead_dec(context))

        pred_obs = InpatientObservables(obs.time, jnp.vstack(pred_obs_l),
                                        obs.mask)
        pred_lead = InpatientObservables(lead.time, jnp.vstack(pred_lead_l),
                                         lead.mask)

        pred_dx = CodesVector(self.f_dx_dec(context),
                              admission.outcome.scheme)
        return AdmissionPrediction(admission=admission,
                                   outcome=pred_dx,
                                   observables=pred_obs,
                                   leading_observable=pred_lead,
                                   trajectory=None)


class InSKELKoopmanConfig(InICENODELiteConfig):
    pass


class InSKELKoopmanRegularisation(ModelRegularisation):
    L_rec: float = 1.0


class InSKELKoopmanPrecomputes(Precomputes):
    A: jnp.ndarray


class InSKELKoopman(InICENODELite):

    def __init__(self, config: InSKELKoopmanConfig,
                 schemes: Tuple[DatasetScheme],
                 demographic_vector_config: DemographicVectorConfig,
                 leading_observable_config: LeadingObservableExtractorConfig,
                 key: "jax.random.PRNGKey"):
        super().__init__(config=config,
                         schemes=schemes,
                         demographic_vector_config=demographic_vector_config,
                         leading_observable_config=leading_observable_config,
                         key=key)

    @staticmethod
    def _make_dyn(config, key):
        return SKELKoopmanOperator(input_size=config.state,
                                   control_size=config.emb.inp_proc_demo,
                                   koopman_size=config.state * 5,
                                   key=key)

    @eqx.filter_jit
    def _safe_integrate(self, delta, state, int_e, precomputes):
        second = jnp.array(1 / 3600.0)
        delta = jnp.where((delta < second) & (delta >= 0.0), second, delta)
        return self.f_dyn(x=state, t=delta, u=int_e, A=precomputes.A)

    def precomputes(self, *args, **kwargs):
        return InSKELKoopmanPrecomputes(A=self.f_dyn.compute_A())

    @property
    def dyn_params_list(self):
        return self.params_list((self.f_dyn.R, self.f_dyn.Q, self.f_dyn.N))

    def step_segment(self, state: jnp.ndarray, int_e: jnp.ndarray,
                     obs: InpatientObservables, lead: InpatientObservables,
                     t0: float, t1: float,
                     sampling_time: Optional[jnp.ndarray],
                     precomputes: InSKELKoopmanPrecomputes):
        preds = []
        lead_preds = []
        trajectory = []
        rec_loss = []
        t = t0
        for t_obs, val, mask in zip(obs.time, obs.value, obs.mask):
            # if time-diff is more than 1 seconds, we integrate.
            rec_loss.append(self.f_dyn.compute_phi_loss(x=state, u=int_e))
            trajectory.extend(
                self.sample_state_trajectory(state, int_e, t, t_obs,
                                             sampling_time, precomputes))

            state = self._safe_integrate(t_obs - t, state, int_e, precomputes)
            pred_obs = self.f_obs_dec(state)
            pred_lead = self.f_lead_dec(state)
            state = self.f_update(state, pred_obs, val, mask)
            t = t_obs
            preds.append(pred_obs)
            lead_preds.append(pred_lead)

        rec_loss.append(self.f_dyn.compute_phi_loss(x=state, u=int_e))
        trajectory.extend(
            self.sample_state_trajectory(state, int_e, t, t1, sampling_time,
                                         precomputes))
        state = self._safe_integrate(t1 - t, state, int_e, precomputes)

        if len(preds) > 0:
            pred_obs_val = jnp.vstack(preds)
            pred_lead_val = jnp.vstack(lead_preds)
        else:
            pred_obs_val = jnp.empty_like(obs.value)
            pred_lead_val = jnp.empty_like(lead.value)

        return state, (InpatientObservables(obs.time, pred_obs_val, obs.mask),
                       InpatientObservables(lead.time, pred_lead_val,
                                            lead.mask)), rec_loss, trajectory

    def __call__(
            self, admission: Admission,
            embedded_admission: EmbeddedInAdmission, precomputes: Precomputes,
            regularisation: InSKELKoopmanRegularisation,
            store_embeddings: Optional[TrajectoryConfig]
    ) -> AdmissionPrediction:
        int_e = embedded_admission.inp_proc_demo

        state = self._init_state(int_demo_emb=int_e[0],
                                 dx0=embedded_admission.dx0)

        obs = admission.observables
        lead = admission.leading_observable
        pred_obs_l = []
        pred_lead_l = []
        rec_loss = []
        t0 = admission.interventions.t0
        t1 = admission.interventions.t1
        trajectory = [state]
        if store_embeddings is not None:
            dt = store_embeddings.sampling_rate
            sampling_time = jnp.arange(0.0, t1[-1], dt)
        else:
            sampling_time = None
        for i in range(len(t0)):
            state, pred, rloss, traj = self.step_segment(
                state, int_e[i], obs[i], lead[i], t0[i], t1[i], sampling_time,
                precomputes)
            pred_obs, pred_lead = pred
            rec_loss.extend(rloss)
            pred_obs_l.append(pred_obs)
            pred_lead_l.append(pred_lead)
            trajectory.extend(traj)

        # if len(trajectory) > 0:
        #     trajectory = PatientTrajectory(time=sampling_time,
        #                                    value=jnp.vstack(trajectory))
        # else:
        #     trajectory = None
        # return AdmissionPrediction(admission=admission,
        #                            outcome=None,
        #                            observables=pred_obs_l,
        #                            leading_observable=pred_lead_l,
        #                            trajectory=trajectory,
        #                            auxiliary_loss={'L_rec': rec_loss})

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


class InVanillaKoopmanConfig(InSKELKoopmanConfig):
    pass


class InVanillaKoopmanRegularisation(InSKELKoopmanRegularisation):
    pass


class InVanillaKoopmanPrecomputes(InSKELKoopmanPrecomputes):
    pass


class InVanillaKoopman(InSKELKoopman):

    @staticmethod
    def _make_dyn(config, key):
        return VanillaKoopmanOperator(input_size=config.state,
                                      control_size=config.emb.inp_proc_demo,
                                      koopman_size=config.state * 5,
                                      key=key)
