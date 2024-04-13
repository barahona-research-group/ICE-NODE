from __future__ import annotations

from abc import abstractmethod, ABCMeta
from typing import (Callable, Optional, Tuple, Union)

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom

from ..base import Config, VxData, Module
from ..ehr.coding_scheme import AggregationLiteral, CodesVector, GroupingData
from ..ehr.tvx_concepts import SegmentedInpatientInterventions, InpatientObservables, SegmentedInpatientObservables, \
    SegmentedAdmission, Admission


class GroupEmbedding(eqx.Module):
    grouping_data: GroupingData = eqx.static_field()
    groups_aggregation: Tuple[Callable, ...]
    linear: eqx.nn.Linear

    @staticmethod
    def group_function(aggregation: AggregationLiteral, input_size: int, group_key: jrandom.PRNGKey) -> Callable:
        if aggregation == 'sum':
            return jnp.sum
        if aggregation == 'or':
            return jnp.any
        if aggregation == 'w_sum':
            return eqx.nn.Linear(input_size, 1, use_bias=False, key=group_key)
        raise ValueError(f"Unrecognised aggregation: {aggregation}")

    def __init__(self, grouping_data: GroupingData, embeddings_size: int, key: jrandom.PRNGKey):
        linear_key, aggregation_key = jrandom.split(key, 2)
        n_groups = len(grouping_data.aggregation)
        self.grouping_data = grouping_data
        self.groups_aggregation = tuple(map(self.group_function,
                                            grouping_data.aggregation,
                                            grouping_data.size,
                                            jrandom.split(aggregation_key, n_groups)))

        self.linear = eqx.nn.Linear(n_groups, embeddings_size, use_bias=False, key=linear_key)

    def split_source_array(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
        assert x.ndim == 1 and len(x) == self.grouping_data.scheme_size[0]
        y = x[self.grouping_data.permute]
        return tuple(jnp.hsplit(y, self.grouping_data.split)[:-1])

    def __call__(self, x: jnp.ndarray):
        xs = self.split_source_array(x)
        y = jnp.hstack([group_apply(x_i) for group_apply, x_i in zip(self.groups_aggregation, xs)])
        return self.linear(jnn.relu(y))


class InterventionsEmbeddingsConfig(Config):
    icu_inputs: Optional[int] = 10
    icu_procedures: Optional[int] = 10
    hosp_procedures: Optional[int] = 10
    interventions: Optional[int] = 20


NullEmbeddingFunction = Callable[[jnp.ndarray], jnp.ndarray]


class InterventionsEmbeddings(Module):
    """
    Embeds an inpatient admission into fixed vectors:
        - Embdedded discharge codes history.
        - A sequence of embedded vectors each fusing the input, procedure \
            and demographic information.
    """
    config: InterventionsEmbeddingsConfig = eqx.static_field()
    f_icu_inputs_emb: GroupEmbedding | NullEmbeddingFunction
    f_icu_procedures_emb: eqx.nn.Linear | NullEmbeddingFunction
    f_hosp_procedures_emb: eqx.nn.Linear | NullEmbeddingFunction
    f_emb: eqx.nn.Linear
    activation: Callable
    final_activation: Callable

    @staticmethod
    def null_embedding(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([])

    def __init__(self, config: InterventionsEmbeddingsConfig,
                 icu_inputs_grouping: Optional[GroupingData] = None,
                 icu_procedures_size: Optional[int] = None,
                 hosp_procedures_size: Optional[int] = None,
                 activation: Callable = jnn.relu,
                 final_activation: Callable = lambda x: x, *,
                 key: jrandom.PRNGKey):
        super().__init__(config=config)
        self.config = config
        self.activation = activation
        self.final_activation = final_activation
        (icu_inputs_key, icu_procedures_key, hosp_procedures_key, interventions_key) = jrandom.split(key, 4)

        intervention_input_size = 0
        if icu_inputs_grouping and config.icu_inputs:
            intervention_input_size += config.icu_inputs
            self.f_icu_inputs_emb = GroupEmbedding(icu_inputs_grouping, config.icu_inputs, icu_inputs_key)
        else:
            self.f_icu_inputs_emb = self.null_embedding

        if icu_procedures_size and config.icu_procedures:
            intervention_input_size += config.icu_procedures
            self.f_icu_procedures_emb = eqx.nn.Linear(icu_procedures_size,
                                                      config.icu_procedures,
                                                      use_bias=False,
                                                      key=icu_procedures_key)
        else:
            self.f_icu_procedures_emb = self.null_embedding

        if hosp_procedures_size and config.hosp_procedures:
            intervention_input_size += config.hosp_procedures
            self.f_hosp_procedures_emb = eqx.nn.Linear(hosp_procedures_size,
                                                       config.hosp_procedures,
                                                       use_bias=False,
                                                       key=hosp_procedures_key)
        else:
            self.f_hosp_procedures_emb = self.null_embedding
        self.f_emb = eqx.nn.Linear(intervention_input_size,
                                   config.interventions,
                                   use_bias=False,
                                   key=interventions_key)

    def __call__(self, icu_inputs: Optional[jnp.ndarray] = None,
                 icu_procedures: Optional[jnp.ndarray] = None,
                 hosp_procedures: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        y_icu_inputs = self.f_icu_inputs_emb(icu_inputs) if icu_inputs is not None else jnp.array([])
        y_icu_procedures = self.f_icu_procedures_emb(icu_procedures) if icu_procedures is not None else jnp.array([])
        y_hosp_procedures = self.f_hosp_procedures_emb(hosp_procedures) if hosp_procedures is not None else jnp.array(
            [])
        y = jnp.hstack((y_icu_inputs, y_icu_procedures, y_hosp_procedures))
        return self.final_activation(self.f_emb(self.activation(y)))


class AdmissionEmbeddingsConfig(Config):
    dx_codes: Optional[int] = None
    interventions: Optional[InterventionsEmbeddingsConfig] = None
    demographic: Optional[int] = None
    observables: Optional[int] = None


class SpecialisedAdmissionEmbeddingsConfig(Config, metaclass=ABCMeta):
    @abstractmethod
    def to_admission_embeddings_config(self) -> AdmissionEmbeddingsConfig:
        pass


class InICENODEEmbeddingsConfig(SpecialisedAdmissionEmbeddingsConfig):
    dx_codes: int = 0
    demographic: int = 0
    interventions: Optional[InterventionsEmbeddingsConfig] = None

    def to_admission_embeddings_config(self) -> AdmissionEmbeddingsConfig:
        return AdmissionEmbeddingsConfig(dx_codes=self.dx_codes,
                                         interventions=self.interventions,
                                         demographic=self.demographic,
                                         observables=None)


class EmbeddedAdmission(VxData):
    dx_codes: Optional[jnp.ndarray] = None  # (config.dx_codes, )
    dx_codes_history: Optional[jnp.ndarray] = None  # (config.dx_codes, )
    interventions: Optional[jnp.ndarray] = None  # (n_segments, config.interventions)
    demographic: Optional[jnp.ndarray] = None  # (config.demographic, )
    observables: Union[
        Tuple[jnp.ndarray, ...], jnp.ndarray] = tuple()  # Tuple[n_segments](n_segment_timestamps, config.observation)


class AdmissionEmbedding(Module):
    """
    Embeds an inpatient admission into fixed vectors:
        - Embdedded discharge codes history.
        - A sequence of embedded vectors each fusing the input, procedure \
            and demographic information.
    """
    config: AdmissionEmbeddingsConfig
    f_dx_codes_emb: eqx.nn.MLP | None = None
    f_interventions_emb: InterventionsEmbeddings | None = None
    f_demographic_emb: eqx.nn.MLP | None = None
    f_observables_emb: eqx.nn.MLP | None = None

    @staticmethod
    def make_interventions_emb(interventions_config: InterventionsEmbeddingsConfig,
                               icu_inputs_grouping: Optional[GroupingData],
                               icu_procedures_size: Optional[int],
                               hosp_procedures_size: Optional[int],
                               key: jrandom.PRNGKey) -> Tuple[InterventionsEmbeddings, jrandom.PRNGKey]:
        key, interventions_key = jrandom.split(key)
        return InterventionsEmbeddings(config=interventions_config,
                                       icu_inputs_grouping=icu_inputs_grouping,
                                       icu_procedures_size=icu_procedures_size,
                                       hosp_procedures_size=hosp_procedures_size,
                                       key=interventions_key), key

    @staticmethod
    def make_dx_codes_emb(dx_codes_size: int, dx_codes_emb_size: int, *, key: jrandom.PRNGKey) -> Tuple[
        eqx.nn.MLP, jrandom.PRNGKey]:
        key, dx_codes_key = jrandom.split(key)
        return eqx.nn.MLP(dx_codes_size,
                          dx_codes_emb_size,
                          width_size=dx_codes_emb_size * 2,
                          depth=1,
                          final_activation=jnp.tanh,
                          key=dx_codes_key), key

    @staticmethod
    def make_demographics_emb(demographic_size: int, demographic_emb_size: int, *, key: jrandom.PRNGKey) -> \
            Tuple[eqx.nn.MLP, jrandom.PRNGKey]:
        key, dem_emb_key = jrandom.split(key)
        return eqx.nn.MLP(demographic_size,
                          demographic_emb_size,
                          width_size=demographic_emb_size * 2,
                          depth=1,
                          final_activation=jnp.tanh,
                          key=dem_emb_key), key

    def make_observables_emb(self, observables_size: int, observables_emb_size: int, *, key: jrandom.PRNGKey) -> \
            Tuple[eqx.nn.MLP, jrandom.PRNGKey]:
        key, obs_emb_key = jrandom.split(key)
        return eqx.nn.MLP(observables_size * 2,  # 2 for value and mask.
                          observables_emb_size,
                          width_size=observables_emb_size * 2,
                          depth=1,
                          final_activation=jnp.tanh,
                          key=obs_emb_key), key

    def __init__(self, config: AdmissionEmbeddingsConfig,
                 dx_codes_size: Optional[int],
                 icu_inputs_grouping: Optional[GroupingData],
                 icu_procedures_size: Optional[int],
                 hosp_procedures_size: Optional[int],
                 demographic_size: Optional[int],
                 observables_size: Optional[int],
                 key: jrandom.PRNGKey):
        super().__init__(config=config)
        if config.dx_codes:  # not None and not 0.
            assert dx_codes_size is not None, "dx_codes_size must be provided"
            self.f_dx_codes_emb, key = self.make_dx_codes_emb(dx_codes_size, config.dx_codes, key=key)

        if config.interventions:
            assert icu_inputs_grouping or icu_procedures_size or hosp_procedures_size, \
                "Any of icu_inputs_map, icu_procedures_size or hosp_procedures_size must be provided."
            self.f_interventions_emb, key = self.make_interventions_emb(config.interventions,
                                                                        icu_inputs_grouping,
                                                                        icu_procedures_size,
                                                                        hosp_procedures_size,
                                                                        key=key)
        if config.demographic:
            assert demographic_size, "demographic_size must be provided."
            self.f_demographic_emb, key = self.make_demographics_emb(demographic_size, config.demographic, key=key)

        if config.observables:
            assert observables_size, "observables_size must be provided."
            self.f_observables_emb, _ = self.make_observables_emb(observables_size, config.observables, key=key)

    @eqx.filter_jit
    def dx_embdedding_vectors(self) -> jnp.ndarray:
        in_size = self.f_dx_codes_emb.in_size
        return eqx.filter_vmap(self.f_dx_codes_emb)(jnp.eye(in_size))

    @eqx.filter_jit
    def embed_dx_codes(self, dx_codes: CodesVector, dx_codes_history: CodesVector) -> EmbeddedAdmission:
        if self.f_dx_codes_emb is None:
            return EmbeddedAdmission()
        dx_codes_emb = self.f_dx_codes_emb(dx_codes.vec)
        dx_codes_history_emb = self.f_dx_codes_emb(dx_codes_history.vec)
        return EmbeddedAdmission(dx_codes=dx_codes_emb, dx_codes_history=dx_codes_history_emb)

    @eqx.filter_jit
    def embed_interventions(self, interventions: SegmentedInpatientInterventions) -> EmbeddedAdmission:
        if self.f_interventions_emb is None:
            return EmbeddedAdmission()

        segments = [self.f_interventions_emb(icu_inputs, icu_procedures, hosp_procedures)
                    for hosp_procedures, icu_procedures, icu_inputs in interventions.iter_tuples()]
        return EmbeddedAdmission(interventions=jnp.vstack(segments))

    @eqx.filter_jit
    def embed_demographic(self, demographic: jnp.ndarray) -> EmbeddedAdmission:
        if self.f_demographic_emb is None:
            return EmbeddedAdmission()
        return EmbeddedAdmission(demographic=self.f_demographic_emb(demographic))

    def _embed_observables_segment(self, obs: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(self.f_observables_emb)(obs)

    def embed_observables(self, observables: InpatientObservables | SegmentedInpatientObservables) -> EmbeddedAdmission:
        if self.f_observables_emb is None:
            return EmbeddedAdmission()

        if isinstance(observables, SegmentedInpatientObservables):
            segments = [jnp.concatenate((obs.value, obs.mask), axis=1) for obs in observables]
            return EmbeddedAdmission(observables=tuple(self._embed_observables_segment(s) for s in segments))

        elif isinstance(observables, InpatientObservables):
            obs = jnp.concatenate((observables.value, observables.mask), axis=1)
            return EmbeddedAdmission(observables=self._embed_observables_segment(obs))
        else:
            raise ValueError(f"Unrecognised observables type: {type(observables)}")

    def __call__(self, admission: SegmentedAdmission | Admission,
                 demographic_input: Optional[jnp.ndarray]) -> EmbeddedAdmission:
        """
        Embeds an admission into fixed vectors as described above.
        """

        if not isinstance(admission, SegmentedAdmission):
            assert self.f_interventions_emb is None, "Interventions embedding is " \
                                                     "not supported for (unsegmented) admission."

        return eqx.combine(self.embed_observables(admission.observables), self.embed_demographic(demographic_input),
                           self.embed_interventions(admission.interventions),
                           self.embed_dx_codes(admission.dx_codes, admission.dx_codes_history),
                           is_leaf=lambda x: x is None or isinstance(x, tuple))


class AdmissionSequentialEmbeddingsConfig(SpecialisedAdmissionEmbeddingsConfig):
    dx_codes: int = 0
    demographic: int = 0
    observables: int = 0
    sequence: int = 50

    def to_admission_embeddings_config(self) -> AdmissionEmbeddingsConfig:
        return AdmissionEmbeddingsConfig(dx_codes=self.dx_codes,
                                         interventions=None,
                                         demographic=self.demographic,
                                         observables=self.observables)


class EmbeddedAdmissionObsSequence(VxData):
    dx_codes_history: jnp.ndarray  # (config.dx, )
    demographic: jnp.ndarray  # (config.demo, )
    sequence: jnp.ndarray  # (n_timestamps, config.sequence)


class AdmissionSequentialObsEmbedding(eqx.Module):
    f_components_emb: AdmissionEmbedding
    f_sequence_emb: Callable

    def __init__(self, config: AdmissionSequentialEmbeddingsConfig,
                 dx_codes_size: Optional[int] = None,
                 demographic_size: Optional[int] = None,
                 observables_size: Optional[int] = None, *,
                 key: jrandom.PRNGKey):
        (components_emb_key, sequence_emb_key) = jrandom.split(key, )

        self.f_components_emb = AdmissionEmbedding(config=config.to_admission_embeddings_config(),
                                                   dx_codes_size=dx_codes_size,
                                                   icu_inputs_grouping=None,
                                                   icu_procedures_size=None,
                                                   hosp_procedures_size=None,
                                                   demographic_size=demographic_size,
                                                   observables_size=observables_size,
                                                   key=components_emb_key)

        self.f_sequence_emb = eqx.nn.MLP(config.dx_codes + config.demographic + config.observables,
                                         config.sequence,
                                         config.sequence * 2,
                                         depth=1,
                                         final_activation=jnp.tanh,
                                         key=sequence_emb_key)

    @eqx.filter_jit
    def _embed_sequence_features(self, *embeddings):
        return self.f_sequence_emb(jnp.hstack(tuple(emb for emb in embeddings if emb is not None)))

    def __call__(self, admission: Admission, admission_demographic: jnp.ndarray) -> EmbeddedAdmissionObsSequence:
        assert not isinstance(admission, SegmentedAdmission), "SegmentedPatient not supported"
        """ Embeds an admission into fixed vectors as described above."""
        emb_components = self.f_components_emb(admission, admission_demographic)
        dx_history_emb = emb_components.dx_codes_history
        demo_emb = emb_components.demographic
        obs_emb = emb_components.observables
        sequence_e = [self._embed_sequence_features(dx_history_emb, demo_emb, obs_e_i) for obs_e_i in obs_emb]
        return EmbeddedAdmissionObsSequence(dx_codes_history=dx_history_emb,
                                            sequence=sequence_e,
                                            demographic=demo_emb)


class DischargeSummarySequentialEmbeddingsConfig(SpecialisedAdmissionEmbeddingsConfig):
    dx_codes: int = 0
    demographic: int = 0
    summary: int = 50

    def to_admission_embeddings_config(self) -> AdmissionEmbeddingsConfig:
        return AdmissionEmbeddingsConfig(dx_codes=self.dx_codes,
                                         interventions=None,
                                         demographic=self.demographic,
                                         observables=None)


class EmbeddedDischargeSummary(VxData):
    dx_codes: jnp.ndarray  # (config.dx, )
    demographic: jnp.ndarray  # (config.demo, )
    summary: jnp.ndarray  # (config.sequence, )
    history_summary: jnp.ndarray  # (config.sequence, )


class DischargeSummarySequentialEmbedding(eqx.Module):
    _f_components_emb: AdmissionEmbedding
    _f_summary_emb: Callable

    def __init__(self, config: DischargeSummarySequentialEmbeddingsConfig,
                 dx_codes_size: Optional[int] = None,
                 demographic_size: Optional[int] = None, *,
                 key: jrandom.PRNGKey):
        (components_emb_key, sequence_emb_key) = jrandom.split(key, )

        self.f_components_emb = AdmissionEmbedding(config=config.to_admission_embeddings_config(),
                                                   dx_codes_size=dx_codes_size,
                                                   icu_inputs_grouping=None,
                                                   icu_procedures_size=None,
                                                   hosp_procedures_size=None,
                                                   demographic_size=demographic_size,
                                                   observables_size=None,
                                                   key=components_emb_key)

        self.f_summary_emb = eqx.nn.MLP(config.dx_codes + config.demographic,
                                        config.summary,
                                        config.summary * 2,
                                        depth=1,
                                        final_activation=jnp.tanh,
                                        key=sequence_emb_key)

    @eqx.filter_jit
    def _embed_summary_features(self, dx_codes_emb, demo_emb):
        return self.f_summary_emb(jnp.hstack((dx_codes_emb, demo_emb)))

    def __call__(self, admission: Admission, admission_demographic: jnp.ndarray) -> EmbeddedDischargeSummary:
        """ Embeds an admission into fixed vectors as described above."""
        assert not isinstance(admission, SegmentedAdmission), "SegmentedAdmission not supported"
        emb_components = self.f_components_emb(admission, admission_demographic)
        summary = self._embed_summary_features(emb_components.dx_codes, emb_components.demographic)
        history_summary = self._embed_summary_features(emb_components.dx_codes_history, emb_components.demographic)
        return EmbeddedDischargeSummary(dx_codes=emb_components.dx_codes, demographic=emb_components.demographic,
                                        summary=summary, history_summary=history_summary)
