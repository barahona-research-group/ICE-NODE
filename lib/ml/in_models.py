"""."""
from __future__ import annotations

from typing import Callable, Tuple, Optional, Literal, Type

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom

from .artefacts import AdmissionPrediction
from .base_models import (NeuralODE_JAX)
from .base_models_koopman import SKELKoopmanOperator, VanillaKoopmanOperator
from .embeddings import AdmissionEmbedding, AdmissionEmbeddingsConfig, EmbeddedAdmission, AdmissionSequentialEmbedding, \
    AdmissionSequentialEmbeddingsConfig, EmbeddedAdmissionSequence
from .model import (InpatientModel, ModelConfig, ModelRegularisation,
                    Precomputes)
from ..ehr import (Admission, InpatientObservables, DatasetScheme, DemographicVectorConfig, CodesVector,
                   LeadingObservableExtractorConfig)
from ..ehr.coding_scheme import GroupingData
from ..ehr.tvx_concepts import SegmentedAdmission
from ..utils import model_params_scaler

LeadPredictorName = Literal['monotonic', 'mlp', 'ode_solve']


class InpatientModelConfig(ModelConfig):
    state: int = 50
    lead_predictor: LeadPredictorName = "monotonic"
    lead_predictor_ode_solve: bool = False


class NODERegularisation(ModelRegularisation):
    L_taylor: float = 0.0
    taylor_order: int = 0


class MonotonicLeadingObsPredictor(eqx.Module):
    _mlp: eqx.nn.MLP

    def __init__(self, input_size: int,
                 n_lead_times: int,
                 key: jrandom.PRNGKey, **mlp_kwargs):
        super().__init__()
        out_size = n_lead_times + 1
        width = mlp_kwargs.get("width_size", out_size * 5)
        self._mlp = eqx.nn.MLP(input_size,
                               out_size,
                               width_size=width,
                               activation=jnn.tanh,
                               depth=2,
                               key=key)

    @eqx.filter_jit
    def __call__(self, state):
        y = self._mlp(state)
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
    def __post_init__(self):
        assert self.forecasted_state.shape == self.adjusted_state.shape

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
    def compile(time: jnp.ndarray, forecasted_state: Tuple[jnp.ndarray],
                adjusted_state: Tuple[jnp.ndarray]) -> ICENODEStateTrajectory:
        forecasted_state = jnp.vstack(forecasted_state)
        adjusted_state = jnp.vstack(adjusted_state)
        return ICENODEStateTrajectory(time=time, forecasted_state=forecasted_state, adjusted_state=adjusted_state,
                                      mask=jnp.ones_like(forecasted_state, dtype=bool))


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


class ODESolveLeadPredictorWrapper(eqx.Module):
    predictor: eqx.nn.MLP
    lead_times: Tuple[float, ...]

    def __init__(self, input_size: int,
                 lead_times: Tuple[float, ...],
                 key: jrandom.PRNGKey, **mlp_kwargs):
        super().__init__()
        width = mlp_kwargs.get("width_size", 5)
        self.lead_times = lead_times
        self.predictor = eqx.nn.MLP(input_size,
                                    1,
                                    width_size=width,
                                    depth=2,
                                    key=key)

    def __call__(self, trajectory: ICENODEStateTrajectory, admission: SegmentedAdmission,
                 embedded_admission: EmbeddedAdmission, precomputes: Precomputes,
                 ode_dyn: Callable,
                 **kwargs) -> InpatientObservables:
        pass
        # Wrap predictor(ode_dyn) in max operator.


class InICENODE(InpatientModel):
    """
    The InICENODE model. It is composed of the following components:
        - f_emb: Embedding function.
        - f_obs_dec: Observation decoder.
        - f_dx_dec: Discharge codes decoder.
        - f_dyn: Dynamics function.
        - f_update: Update function.
    """
    f_emb: AdmissionEmbedding
    f_obs_dec: eqx.nn.MLP
    f_lead_dec: DirectLeadPredictorWrapper | ODESolveLeadPredictorWrapper
    f_dyn: NeuralODE_JAX
    f_update: InICENODEStateAdjust
    f_init: eqx.nn.MLP
    f_outcome_dec: Optional[Callable] = None

    config: InpatientModelConfig = eqx.static_field()
    regularisation: NODERegularisation = eqx.static_field(default=NODERegularisation())

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
        (emb_key, obs_dec_key, lead_key, outcome_dec_key, dyn_key,
         update_key) = jrandom.split(key, 6)
        self.f_emb = self._make_embedding(config=embeddings_config,
                                          dx_codes_size=dx_codes_size,
                                          icu_inputs_grouping=icu_inputs_grouping,
                                          icu_procedures_size=icu_procedures_size,
                                          hosp_procedures_size=hosp_procedures_size,
                                          demographic_size=demographic_size,
                                          observables_size=observables_size,
                                          key=key)
        self.f_outcome_dec = self._make_outcome_dec(state_size=config.state,
                                                    outcome_size=outcome_size,
                                                    key=outcome_dec_key)

        self.f_obs_dec = self._make_obs_dec(config=config,
                                            observables_size=observables_size,
                                            key=obs_dec_key)
        interventions_size = embeddings_config.interventions.interventions if embeddings_config.interventions else 0
        self.f_dyn = self._make_dyn(state_size=config.state,
                                    interventions_size=interventions_size,
                                    key=dyn_key)
        self.f_update = self._make_update(state_size=config.state,
                                          observables_size=observables_size,
                                          key=update_key)

        self.f_lead_dec = self._make_lead_dec()

    @staticmethod
    def _make_init(dx_codes_size: Optional[int],
                   demographic_size: Optional[int],
                   state_size: int, key: jrandom.PRNGKey):
        dx_codes_size = dx_codes_size or 0
        demographic_size = demographic_size or 0
        return eqx.nn.MLP(dx_codes_size + demographic_size,
                          state_size,
                          state_size * 3,
                          depth=2,
                          key=key)

    @staticmethod
    def _make_lead_dec(
            input_size: int,
            n_lead_times: int,
            predictor: DirectLeadPredictorName,
            ode_solve: bool,
            key: jrandom.PRNGKey, **mlp_kwargs
    ) -> DirectLeadPredictorWrapper | ODESolveLeadPredictorWrapper:
        if ode_solve:
            wrapper_cls = ODESolveLeadPredictorWrapper
        else:
            wrapper_cls = DirectLeadPredictorWrapper
        return Wrapper(input_size, n_lead_times, predictor, key, **mlp_kwargs)

    @staticmethod
    def _make_dyn(state_size: int, interventions_size: int, key: jrandom.PRNGKey):
        f_dyn = eqx.nn.MLP(in_size=state_size + interventions_size,
                           out_size=state_size,
                           activation=jnn.tanh,
                           depth=2,
                           width_size=state_size * 5,
                           key=key)
        f_dyn = model_params_scaler(f_dyn, 1e-2, eqx.is_inexact_array)
        return NeuralODE_JAX(f_dyn, timescale=1.0)

    @staticmethod
    def _make_update(state_size: int, observables_size: int, key: jrandom.PRNGKey) -> InICENODEStateAdjust:
        return InICENODEStateAdjust(state_size, observables_size, key=key)

    @staticmethod
    def _make_outcome_dec(state_size: int, outcome_size: Optional[int], key: jrandom.PRNGKey) -> Callable:
        return eqx.nn.MLP(state_size,
                          outcome_size,
                          state_size * 2,
                          activation=jnp.tanh,
                          depth=1,
                          key=key) if outcome_size is not None else None

    @staticmethod
    def _make_obs_dec(config, observables_size, key) -> eqx.nn.MLP:
        return eqx.nn.MLP(config.state,
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

    @eqx.filter_jit
    def _safe_integrate(self, delta, state, int_e, precomputes):
        second = jnp.array(1 / 3600.0)
        delta = jnp.where((delta < second) & (delta >= 0.0), second, delta)
        return self.f_dyn(delta, state, args=dict(control=int_e))[-1]

    def decode_state_trajectory_observables(self,
                                            admission: SegmentedAdmission,
                                            embedded_admission: EmbeddedAdmission,
                                            precomputes: Precomputes,
                                            state_trajectory: ICENODEStateTrajectory) -> AdmissionPrediction:
        pass

    def __call__(
            self, admission: SegmentedAdmission,
            embedded_admission: EmbeddedAdmission, precomputes: Precomputes
    ) -> AdmissionPrediction:
        int_e = embedded_admission.interventions
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

            for obs_t, obs_val, obs_mask in segment_obs:
                # if time-diff is more than 1 seconds, we integrate.
                forecasted_state = self._safe_integrate(obs_t - t, state, segment_interventions, precomputes)
                state = self.f_update(forecasted_state, self.f_obs_dec(state), obs_val, obs_mask)
                state_trajectory += ((forecasted_state, state),)
                t = obs_t
            state = self._safe_integrate(segment_t1 - t, state, segment_interventions, precomputes)
            t = segment_t1
        forecasted_states, adjusted_states = zip(*state_trajectory)
        icenode_state_trajectory = ICENODEStateTrajectory.compile(time=obs.time, forecasted_state=forecasted_states,
                                                                  adjusted_state=adjusted_states)
        outcome_prediction = AdmissionPrediction(admission=admission,
                                                 outcome=CodesVector(self.f_dx_dec(state), admission.outcome.scheme))
        obs_predictions = self.decode_state_trajectory_observables(admission=admission,
                                                                   embedded_admission=embedded_admission,
                                                                   precomputes=precomputes,
                                                                   state_trajectory=icenode_state_trajectory)
        return eqx.combine(outcome_prediction, obs_predictions)

    @property
    def dyn_params_list(self):
        return self.params_list(self.f_dyn)

    # @staticmethod
    # def _make_lead_dec(predictor: AutonomousLeadPredictorName,
    #                    input_size, leading_observable_config, key,
    #                    **kwargs):
    #     if predictor == "monotonic":
    #         return MonotonicLeadingObsPredictor(input_size,
    #                                             leading_observable_config,
    #                                             key=key,
    #                                             **kwargs)
    #     elif predictor == "mlp":
    #         return MLPLeadingObsPredictor(input_size,
    #                                       leading_observable_config,
    #                                       key=key,
    #                                       **kwargs)
    #     else:
    #         raise ValueError(
    #             f"Unknown leading predictor type: {predictor}")


class InICENODELite(InICENODE):
    """
    The InICENODE model. It is composed of the following components:
        - f_emb: Embedding function.
        - f_obs_dec: Observation decoder.
        - f_dyn: Dynamics function.
        - f_update: Update function.
    """

    def __init__(self, config: InpatientModelConfig,
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


class InGRUJump(InICENODELite):
    # TODO: as for the original paper, use multi-layer embeddings with skip
    # connections.
    f_emb: AdmissionSequentialEmbedding
    f_obs_dec: Callable
    f_lead_dec: Callable
    f_init: Callable
    # GRU. Alternatives: SRU, LSTM, ..
    f_update: Callable
    f_dyn: Callable

    def __init__(self, config: InpatientModelConfig,
                 embeddings_config: AdmissionSequentialEmbeddingsConfig,
                 lead_times: Tuple[float, ...],
                 dx_codes_size: Optional[int] = None,
                 demographic_size: Optional[int] = None,
                 observables_size: Optional[int] = None, *,
                 key: "jax.random.PRNGKey"):
        super().__init__(config=config, embeddings_config=embeddings_config, lead_times=lead_times,
                         dx_codes_size=dx_codes_size, icu_inputs_grouping=None,
                         icu_procedures_size=None, hosp_procedures_size=None,
                         demographic_size=demographic_size, observables_size=observables_size, key=key)

    @staticmethod
    def _make_embedding(config: AdmissionSequentialEmbeddingsConfig,
                        dx_codes_size: Optional[int] = None,
                        demographic_size: Optional[int] = None,
                        observables_size: Optional[int] = None, *,
                        key: jrandom.PRNGKey, **kwargs):
        return AdmissionSequentialEmbedding(config=config,
                                            dx_codes_size=dx_codes_size,
                                            demographic_size=demographic_size,
                                            observables_size=observables_size,
                                            key=key)

    @staticmethod
    def _make_dyn(config, key, **kwargs):
        return eqx.nn.GRUCell(input_size=config.emb.demo,
                              hidden_size=config.state,
                              key=key)

    @staticmethod
    def _make_init(config, key):
        init_input_size = config.emb.demo + config.emb.dx_discharge
        return eqx.nn.MLP(init_input_size,
                          config.state,
                          config.emb.dx_discharge * 5,
                          depth=2,
                          key=key)

    @eqx.filter_jit
    def _init_state(self, demographic_emb, dx_history_codes_emb):
        return self.f_init(jnp.hstack((demographic_emb, dx_history_codes_emb)))

    def __call__(
            self, admission: Admission,
            embedded_admission: EmbeddedAdmissionSequence, precomputes: Precomputes,
            store_embeddings: Optional[TrajectoryConfig] = None, **kwargs
    ) -> AdmissionPrediction:
        obs = admission.observables
        lead = admission.leading_observable
        if len(obs) == 0:
            pred_obs = InpatientObservables.empty(obs.value.shape[1])
            pred_lead = InpatientObservables.empty(lead.value.shape[1])
            return AdmissionPrediction(admission=admission,
                                       outcome=None,
                                       observables=pred_obs,
                                       leading_observable=pred_lead,
                                       trajectory=None)

        demo_e = embedded_admission.demographic

        state = self._init_state(demographic_emb=embedded_admission.demographic,
                                 dx_history_codes_emb=embedded_admission.dx_codes_history)

        pred_obs_l = []
        pred_lead_l = []
        for i in range(len(obs.time)):
            state = self.f_dyn(demo_e, state)
            pred_obs = self.f_obs_dec(state)
            pred_lead = self.f_lead_dec(state)
            state = self.f_update(state, pred_obs, obs.value[i], obs.mask[i])
            pred_obs_l.append(pred_obs)
            pred_lead_l.append(pred_lead)

        pred_obs = InpatientObservables(obs.time, jnp.vstack(pred_obs_l),
                                        obs.mask)
        pred_lead = InpatientObservables(lead.time, jnp.vstack(pred_lead_l),
                                         lead.mask)

        return AdmissionPrediction(admission=admission,
                                   outcome=None,
                                   observables=pred_obs,
                                   leading_observable=pred_lead,
                                   trajectory=None)


class InGRU(InGRUJump):
    f_emb: AdmissionSequentialEmbedding

    @staticmethod
    def _make_dyn(config, key, **kwargs):
        return eqx.nn.GRUCell(input_size=config.emb.sequence,
                              hidden_size=config.state,
                              key=key)

    def __call__(
            self, admission: Admission,
            embedded_admission: EmbeddedAdmissionSequence, precomputes: Precomputes,
            store_embeddings: Optional[TrajectoryConfig] = None, **kwargs
    ) -> AdmissionPrediction:
        obs = admission.observables
        lead = admission.leading_observable
        if len(obs) == 0:
            pred_obs = InpatientObservables.empty(obs.value.shape[1])
            pred_lead = InpatientObservables.empty(lead.value.shape[1])
            return AdmissionPrediction(admission=admission,
                                       outcome=None,
                                       observables=pred_obs,
                                       leading_observable=pred_lead,
                                       trajectory=None)

        state = self._init_state(demographic_emb=embedded_admission.demographic,
                                 dx_history_codes_emb=embedded_admission.dx_codes_history)

        sequence_e = embedded_admission.sequence
        pred_obs_l = []
        pred_lead_l = []
        for seq_e in sequence_e:
            pred_obs = self.f_obs_dec(state)
            pred_lead = self.f_lead_dec(state)
            state = self.f_dyn(seq_e, state)
            pred_obs_l.append(pred_obs)
            pred_lead_l.append(pred_lead)

        pred_obs = InpatientObservables(obs.time, jnp.vstack(pred_obs_l),
                                        obs.mask)
        pred_lead = InpatientObservables(lead.time, jnp.vstack(pred_lead_l),
                                         lead.mask)

        return AdmissionPrediction(admission=admission,
                                   outcome=None,
                                   observables=pred_obs,
                                   leading_observable=pred_lead,
                                   trajectory=None)


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
