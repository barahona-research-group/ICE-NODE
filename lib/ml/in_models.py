"""."""
from __future__ import annotations

import logging
from typing import Tuple, Optional, Literal, Type, Union, Final

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy as jscipy
from diffrax import diffeqsolve, Tsit5, RecursiveCheckpointAdjoint, SaveAt, ODETerm, Solution, PIDController, SubSaveAt
from jaxtyping import PyTree

from ._eig_ad import eig
from .artefacts import AdmissionPrediction, AdmissionsPrediction
from .embeddings import (AdmissionEmbedding, AdmissionGenericEmbeddingsConfig, EmbeddedAdmission)
from .embeddings import (AdmissionSequentialEmbeddingsConfig, AdmissionSequentialObsEmbedding,
                         AdmissionEmbeddingsConfig, EmbeddedAdmissionObsSequence)
from .embeddings import (DischargeSummarySequentialEmbeddingsConfig, DischargeSummarySequentialEmbedding,
                         EmbeddedDischargeSummary)
from .model import (InpatientModel, ModelConfig, ModelRegularisation,
                    Precomputes)
from ..ehr import (Admission, InpatientObservables, CodesVector)
from ..ehr.coding_scheme import GroupingData
from ..ehr.tvx_concepts import SegmentedAdmission
from ..ehr.tvx_ehr import SegmentedTVxEHR
from ..utils import model_params_scaler, tqdm_constructor

LeadPredictorName = Literal['monotonic', 'mlp']


def empty_if_none(x):
    return x if x is not None else jnp.array([])


## TODO: use Invertible NN for embeddings: https://proceedings.mlr.press/v162/zhi22a/zhi22a.pdf

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
        return u if u is not None else self.zero_force

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
        return self.f_update(jnp.hstack((observables_mask, projected_error)), forecasted_state)


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
        leading_values = eqx.filter_vmap(self.predictor)(trajectory.adjusted_state)
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
    regularisation: ModelRegularisation = eqx.static_field(default_factory=ModelRegularisation)

    def __init__(self, config: InpatientModelConfig,
                 embeddings_config: AdmissionEmbeddingsConfig,
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
    def _make_init(embeddings_config: Union[AdmissionGenericEmbeddingsConfig, AdmissionSequentialEmbeddingsConfig],
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
    def _make_dyn(state_size: int, embeddings_config: AdmissionGenericEmbeddingsConfig,
                  key: jrandom.PRNGKey) -> NeuralODESolver:
        interventions_size = embeddings_config.interventions.interventions if embeddings_config.interventions else 0
        demographics_size = embeddings_config.demographic or 0
        f_dyn = CompiledMLP(in_size=state_size + interventions_size + demographics_size,
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
    def _make_embedding(config: AdmissionGenericEmbeddingsConfig,
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
        prediction = AdmissionTrajectoryPrediction(admission=admission)
        int_e = empty_if_none(embedded_admission.interventions)
        demo_e = empty_if_none(embedded_admission.demographic)
        obs = admission.observables
        if len(obs) == 0:
            logging.warning("No observation to fit.")
            return prediction

        n_segments = obs.n_segments
        t = 0.0
        state_trajectory = tuple()
        state = self.f_init(jnp.hstack((embedded_admission.dx_codes_history, demo_e)))
        segments_t1 = admission.interventions.t1
        for segment_index in range(n_segments):
            segment_t1 = segments_t1[segment_index]
            segment_obs = obs[segment_index]
            segment_interventions = int_e[segment_index]
            segment_force = jnp.hstack((demo_e, segment_interventions))

            for obs_t, obs_val, obs_mask in segment_obs:
                # if time-diff is more than 1 seconds, we integrate.
                forecasted_state = self.f_dyn(state, t0=t, t1=obs_t, u=segment_force, precomputes=precomputes)
                forecasted_state = forecasted_state.squeeze()
                state = self.f_update(forecasted_state, self.f_obs_dec(state), obs_val, obs_mask)
                state_trajectory += ((forecasted_state, state),)
                t = obs_t
            state = self.f_dyn(state, t0=t, t1=segment_t1, u=segment_force, precomputes=precomputes).squeeze()
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
            prediction = prediction.add(leading_observable=self.f_lead_dec(icenode_state_trajectory))
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

    @property
    def dyn_params_list(self):
        return self.params_list(self.f_dyn)


class InICENODELite(InICENODE):
    # Same as InICENODE but without discharge summary outcome predictions.
    def __init__(self, config: InpatientModelConfig,
                 embeddings_config: Union[AdmissionGenericEmbeddingsConfig, AdmissionSequentialEmbeddingsConfig],
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
        admission.leading_observable[timestamp_index], which is a strong assumption due to:
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
        demo_e = empty_if_none(embedded_demographic)
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
        if len(obs) == 0:
            logging.warning("No observation to fit.")
            return prediction
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
        emb_dx_history = empty_if_none(embedded_admission.dx_codes_history)
        emb_demo = empty_if_none(embedded_admission.demographic)
        force = jnp.hstack((emb_dx_history, emb_demo))
        prediction = AdmissionPrediction(admission=admission)
        if len(embedded_admission.sequence) == 0:
            logging.warning("No observations to fit.")
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


class CompiledLinear(eqx.nn.Linear):
    @eqx.filter_jit
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return super().__call__(x)


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

    @staticmethod
    def _make_dyn(state_size: int, embeddings_config: AdmissionGenericEmbeddingsConfig,
                  key: jrandom.PRNGKey) -> RETAINDynamic:
        keys = jrandom.split(key, 4)
        gru_a = CompiledGRU(state_size,
                            state_size // 2,
                            use_bias=True,
                            key=keys[0])
        gru_b = CompiledGRU(state_size,
                            state_size // 2,
                            use_bias=True,
                            key=keys[1])

        att_a = CompiledLinear(state_size // 2,
                               1,
                               use_bias=True,
                               key=keys[2])
        att_b = CompiledLinear(state_size // 2,
                               state_size,
                               use_bias=True,
                               key=keys[3])

        return RETAINDynamic(gru_a=gru_a,
                             gru_b=gru_b,
                             att_a=att_a,
                             att_b=att_b)

    @staticmethod
    def _make_init(embeddings_config: DischargeSummarySequentialEmbeddingsConfig,
                   state_size: int, key: jrandom.PRNGKey) -> CompiledMLP:
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
            logging.warning("No observations to fit.")
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


if len(jax.devices()) > 0:
    _flag_gpu_device = jax.devices()[0].platform == "gpu"
else:
    _flag_gpu_device = False


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
    eigen_decomposition: bool = eqx.static_field()

    def __init__(self,
                 input_size: int,
                 koopman_size: int,
                 key: "jax.random.PRNGKey",
                 control_size: int = 0,
                 phi_depth: int = 1,
                 eigen_decomposition: bool = not _flag_gpu_device):
        super().__init__()
        self.input_size = input_size
        self.koopman_size = koopman_size
        self.control_size = control_size
        self.eigen_decomposition = eigen_decomposition
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
    def compute_A(self):
        if self.eigen_decomposition:
            lam, V = eig(self.A)
            V_inv = jnp.linalg.solve(V @ jnp.diag(lam), self.A)
            return self.A, (lam, V, V_inv)

        return self.A

    def compute_K(self, t, A=None):
        if A is None:
            A = self.compute_A()
        if self.eigen_decomposition:
            _, (lam, V, V_inv) = A
            return (V @ jnp.diag(jnp.exp(lam * t)) @ V_inv).real
        else:
            return jscipy.linalg.expm(A * t, max_squarings=20)

    @eqx.filter_jit
    def __call__(self, x0, t0: float, t1: float, saveat: Optional[SaveAt] = None,
                 u: Optional[PyTree] = None,
                 precomputes: Optional[InKoopmanPrecomputes] = None) -> Union[jnp.ndarray, Tuple[jnp.ndarray, ...]]:
        if precomputes is None:
            precomputes = InKoopmanPrecomputes(A=self.compute_A())

        z = self.phi(x0, u=u)
        K = self.compute_K(t1, A=precomputes.A)
        z = K @ z
        return self.phi_inv(z)

    @eqx.filter_jit
    def compute_phi_loss(self, x, u=None):
        z = self.phi(x, u=u)
        diff = x - self.phi_inv(z)
        return jnp.mean(diff ** 2)

    def compute_A_spectrum(self):
        if self.eigen_decomposition:
            _, (lam, _, _) = self.compute_A()
        else:
            lam, _ = jnp.linalg.eig(self.compute_A())

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
                 phi_depth: int = 3,
                 eigen_decomposition: bool = not _flag_gpu_device):
        superkey, key = jrandom.split(key, 2)
        super().__init__(input_size=input_size,
                         koopman_size=koopman_size,
                         key=superkey,
                         control_size=control_size,
                         phi_depth=phi_depth,
                         eigen_decomposition=eigen_decomposition)
        self.A = None
        self.eigen_decomposition = eigen_decomposition
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

        if self.eigen_decomposition:
            lam, V = eig(A)
            V_inv = jnp.linalg.solve(V @ jnp.diag(lam), A)
            return A, (lam, V, V_inv)

        return A


class InKoopman(InICENODELite):
    f_dyn: KoopmanOperator

    @staticmethod
    def _make_dyn(state_size: int, embeddings_config: AdmissionGenericEmbeddingsConfig,
                  key: jrandom.PRNGKey) -> KoopmanOperator:
        interventions_size = embeddings_config.interventions.interventions if embeddings_config.interventions else 0
        demographics_size = embeddings_config.demographic
        return KoopmanOperator(input_size=state_size,
                               control_size=interventions_size + demographics_size,
                               koopman_size=state_size * 5,
                               key=key)

    def precomputes(self, *args, **kwargs):
        return InKoopmanPrecomputes(A=self.f_dyn.compute_A())

    @property
    def dyn_params_list(self):
        return self.params_list((self.f_dyn.R, self.f_dyn.Q, self.f_dyn.N))

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
