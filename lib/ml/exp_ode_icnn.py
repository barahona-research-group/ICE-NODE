"""."""
from __future__ import annotations

import logging
from dataclasses import fields
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
from .in_models import DynamicsLiteral, ICENODEStateTrajectory, InICENODE, ICENODEConfig
from .koopman_modules import KoopmanOperator, KoopmanPrecomputes
from .model import (InpatientModel, ModelConfig,
                    Precomputes)
from ..ehr import (Admission, InpatientObservables, CodesVector, TVxEHR)
from ..ehr.coding_scheme import GroupingData
from ..ehr.tvx_concepts import SegmentedAdmission, ObservablesDistribution
from ..ehr.tvx_ehr import SegmentedTVxEHR
from ..utils import model_params_scaler, tqdm_constructor



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
    f_init: Optional[Callable] = None
    f_obs_dec: Optional[Callable] = None
    f_update: Optional[Callable] = None


class ODEICNN(InpatientModel):
    f_emb: AdmissionEmbedding
    f_dyn: NeuralODESolver
    f_icnn: ICNNObsDecoder

    config: ODEICNNConfig = eqx.static_field()

    def __init__(self, config: ODEICNNConfig,
                 embeddings_config: AdmissionEmbeddingsConfig,
                 observables_size: int,
                 icu_inputs_grouping: Optional[GroupingData] = None,
                 icu_procedures_size: Optional[int] = None,
                 hosp_procedures_size: Optional[int] = None, *,
                 key: "jax.random.PRNGKey"):
        super().__init__(config=config)

        (emb_key, obs_dec_key, lead_key, outcome_dec_key, dyn_key, update_key) = jrandom.split(key, 6)
        self.f_emb = self._make_embedding(config=embeddings_config,
                                          icu_inputs_grouping=icu_inputs_grouping,
                                          icu_procedures_size=icu_procedures_size,
                                          hosp_procedures_size=hosp_procedures_size,
                                          key=key)
        self.f_icnn = self._make_icnn(config=config, observables_size=observables_size, key=obs_dec_key)
        self.f_dyn = InICENODE._make_dyn(embeddings_config=embeddings_config,
                                    model_config=ICENODEConfig(state=config.state, dynamics=config.dynamics),
                                    observables_size=observables_size,
                                    key=dyn_key)

    @property
    def persistent_memory_size(self) -> int:
        return int(self.config.state * self.config.memory_ratio)

    @eqx.filter_jit
    def _f_update(self,
                  obs_decoder: ICNNObsDecoder,
                  forecasted_state: jnp.ndarray,
                  true_observables: jnp.ndarray,
                  observables_mask: jnp.ndarray,
                  u: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, ImputerMetrics]:
        # update: keeps the memory fixed (+given obs) and tune everything else.

        # init_obs: obs_decoder(forecasted_state).
        # input: (state_mem, state_hidden, init_obs).
        # mask: (ones_like(state_mem), zeros_like(state_hidden).
        input = jnp.hstack((forecasted_state, true_observables))
        mask = jnp.zeros_like(input).at[:self.persistent_memory_size].set(1) # Keep memory fixed.
        mask = mask.at[obs_decoder.state_size:].set(observables_mask) # Keep the given obs fixed.
        output, stats = obs_decoder.partial_input_optimise(input, mask)
        state, _ = jnp.split(output, [obs_decoder.state_size])
        return state, stats

    @eqx.filter_jit
    def _f_obs_dec(self, forecasted_state: jnp.ndarray) -> Tuple[jnp.ndarray, ImputerMetrics]:
        # update: keeps the memory and state fixed and tune everything else.
        input = jnp.hstack((forecasted_state, jnp.zeros(self.f_icnn.observables_size)))
        mask = jnp.zeros_like(input).at[:self.f_icnn.state_size].set(1)
        output, stats = self.f_icnn.partial_input_optimise(input, mask)
        state, obs = jnp.split(output, [self.f_icnn.state_size])
        return obs, stats

    @eqx.filter_jit
    def _f_init(self,
                  obs_decoder: ICNNObsDecoder,
                  forecasted_state: jnp.ndarray,
                  true_observables: jnp.ndarray,
                  observables_mask: jnp.ndarray,
                  u: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, ImputerMetrics]:
        # update:fix the given obs and tune everything else.

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
    def components(self) -> ODEICNNComponents:
        return ODEICNNComponents(emb=self.f_emb, obs_dec=self.f_obs_dec,
                                dyn=self.f_dyn, update=self._f_update, init=self.f_init)

    @staticmethod
    def _make_icnn(config: ODEICNNConfig, observables_size: int, key) -> ICNNObsDecoder:
        return ICNNObsDecoder(observables_size=observables_size, state_size=config.state,
                              hidden_size_multiplier=3, depth=4,
                              optax_optimiser_name='polyak_sgd',
                              key=key)



    @property
    def dyn_params_list(self):
        return jtu.tree_leaves(eqx.filter(self.f_dyn, eqx.is_inexact_array))

    @classmethod
    def from_tvx_ehr(cls, tvx_ehr: TVxEHR, config: ODEICNNConfig,
                     embeddings_config: AdmissionEmbeddingsConfig, seed: int = 0) -> Self:
        key = jrandom.PRNGKey(seed)
        return cls(config=config,
                   embeddings_config=embeddings_config,
                   icu_inputs_grouping=tvx_ehr.icu_inputs_grouping,
                   icu_procedures_size=len(tvx_ehr.scheme.icu_procedures),
                   hosp_procedures_size=len(tvx_ehr.scheme.hosp_procedures),
                   observables_size=len(tvx_ehr.scheme.obs),
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
        f = self.components
        pred_obs = eqx.filter_vmap(f.obs_dec)(state_trajectory.forecasted_state)
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
            forecasted_states, adjusted_states = zip(*state_trajectory)
            icenode_state_trajectory = ICENODEStateTrajectory.compile(time=obs.time, forecasted_state=forecasted_states,
                                                                      adjusted_state=adjusted_states)
            # TODO: test --> assert len(obs.time) == len(forecasted_states)
            prediction = prediction.add(observables=self.decode_state_trajectory_observables(
                admission=admission, state_trajectory=icenode_state_trajectory))
            prediction = prediction.add(leading_observable=f.lead_dec(icenode_state_trajectory))
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
