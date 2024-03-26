from __future__ import annotations

from typing import (List, Callable, Optional, Tuple, Dict)

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom

from ..base import Config, VxData
from ..ehr.coding_scheme import ReducedCodeMapN1, AggregationLiteral, CodesVector
from ..ehr.tvx_concepts import SegmentedInpatientInterventions, InpatientObservables, SegmentedInpatientObservables, \
    SegmentedAdmission, SegmentedPatient, Admission, Patient


class GroupEmbedding(eqx.Module):
    groups_split: jnp.ndarray = eqx.static_field()
    groups_permute: jnp.ndarray = eqx.static_field()
    source_size: int = eqx.static_field()
    target_size: int = eqx.static_field()
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

    def __init__(self, reduced_map: ReducedCodeMapN1, embeddings_size: int, key: jrandom.PRNGKey):
        self.groups_split = jnp.array(reduced_map.groups_split)
        self.groups_permute = jnp.array(reduced_map.groups_permute)
        self.source_size = len(reduced_map.source_scheme)
        self.target_size = len(self.groups_aggregation)
        linear_key, aggregation_key = jrandom.split(key, 2)
        self.groups_aggregation = tuple(map(self.group_function,
                                            reduced_map.groups_aggregation,
                                            reduced_map.groups_size,
                                            jrandom.split(aggregation_key, self.target_size)))
        self.linear = eqx.nn.Linear(self.target_size, embeddings_size, use_bias=False, key=linear_key)

    def split_source_array(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
        assert x.ndim == 1 and len(x) == self.source_size
        y = x[self.groups_permute]
        return tuple(jnp.hsplit(y, self.groups_split)[:-1])

    def __call__(self, x: jnp.ndarray):
        xs = self.split_source_array(x)
        y = jnp.hstack([group(x_i) for group, x_i in zip(self.groups_aggregation, xs)])
        return self.linear(jnn.relu(y))


class InterventionsEmbeddingsConfig(Config):
    icu_inputs: Optional[int] = 10
    icu_procedures: Optional[int] = 10
    hosp_procedures: Optional[int] = 10
    interventions: Optional[int] = 20


NullEmbeddingFunction = Callable[[jnp.ndarray], jnp.ndarray]


def null_embedding(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.array([])


class InterventionsEmbeddings(eqx.Module):
    """
    Embeds an inpatient admission into fixed vectors:
        - Embdedded discharge codes history.
        - A sequence of embedded vectors each fusing the input, procedure \
            and demographic information.
    """
    config: InterventionsEmbeddingsConfig = eqx.static_field()
    _f_icu_inputs_emb: GroupEmbedding | NullEmbeddingFunction = null_embedding
    _f_icu_procedures_emb: eqx.nn.Linear | NullEmbeddingFunction = null_embedding
    _f_hosp_procedures_emb: eqx.nn.Linear | NullEmbeddingFunction = null_embedding
    _f_emb: eqx.nn.Linear
    activation: Callable
    final_activation: Callable

    def __init__(self, config: InterventionsEmbeddingsConfig,
                 icu_inputs_map: Optional[ReducedCodeMapN1] = None,
                 icu_procedures_size: Optional[int] = None,
                 hosp_procedures_size: Optional[int] = None,
                 activation: Callable = jnn.relu,
                 final_activation: Callable = lambda x: x, *,
                 key: jrandom.PRNGKey):
        super().__init__()
        self.config = config
        self.activation = activation
        self.final_activation = final_activation
        (icu_inputs_key, icu_procedures_key, hosp_procedures_key, interventions_key) = jrandom.split(key, 4)

        if icu_inputs_map and config.icu_inputs:
            self._f_icu_inputs_emb = GroupEmbedding(icu_inputs_map, config.icu_inputs, icu_inputs_key)

        if icu_procedures_size and config.icu_procedures:
            self._f_icu_procedures_emb = eqx.nn.Linear(icu_procedures_size,
                                                       config.icu_procedures,
                                                       use_bias=False,
                                                       key=icu_procedures_key)
        if hosp_procedures_size and config.hosp_procedures:
            self._f_hosp_procedures_emb = eqx.nn.Linear(hosp_procedures_size,
                                                        config.hosp_procedures,
                                                        use_bias=False,
                                                        key=hosp_procedures_key)
        self._f_emb = eqx.nn.Linear(config.icu_inputs + config.icu_procedures + config.hosp_procedures,
                                    config.interventions,
                                    use_bias=False,
                                    key=interventions_key)

    def __call__(self, icu_inputs: Optional[jnp.ndarray] = None,
                 icu_procedures: Optional[jnp.ndarray] = None,
                 hosp_procedures: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        y_icu_inputs = self._f_icu_inputs_emb(icu_inputs)
        y_icu_procedures = self._f_icu_procedures_emb(icu_procedures)
        y_hosp_procedures = self._f_hosp_procedures_emb(hosp_procedures)
        y = jnp.hstack((y_icu_inputs, y_icu_procedures, y_hosp_procedures))
        return self.final_activation(self._f_emb(self.activation(y)))


class AdmissionEmbeddingsConfig(Config):
    dx_codes: Optional[int] = None
    interventions: Optional[InterventionsEmbeddingsConfig] = None
    demographic: Optional[int] = None
    observables: Optional[int] = None


class EmbeddedAdmission(VxData):
    dx_codes: Optional[jnp.ndarray] = None  # (config.dx_codes, )
    dx_codes_history: Optional[jnp.ndarray] = None  # (config.dx_codes, )
    interventions: Optional[jnp.ndarray] = None  # (n_segments, config.interventions)
    demographic: Optional[jnp.ndarray] = None  # (config.demographic, )
    observables: Tuple[jnp.ndarray, ...] = None  # Tuple[n_segments](n_timestamps, config.observation)


class PatientEmbedding(eqx.Module):
    """
    Embeds an inpatient admission into fixed vectors:
        - Embdedded discharge codes history.
        - A sequence of embedded vectors each fusing the input, procedure \
            and demographic information.
    """
    _f_dx_codes_emb: eqx.nn.MLP | None = None
    _f_interventions_emb: InterventionsEmbeddings | None = None
    _f_demographic_emb: eqx.nn.MLP | None = None
    _f_observables_emb: eqx.nn.MLP | None = None

    @staticmethod
    def make_interventions_emb(interventions_config: InterventionsEmbeddingsConfig,
                               icu_inputs_map: Optional[ReducedCodeMapN1],
                               icu_procedures_size: Optional[int],
                               hosp_procedures_size: Optional[int],
                               key: jrandom.PRNGKey) -> Tuple[InterventionsEmbeddings, jrandom.PRNGKey]:
        key, interventions_key = jrandom.split(key)
        return InterventionsEmbeddings(interventions_config,
                                       icu_inputs_map,
                                       icu_procedures_size,
                                       hosp_procedures_size,
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
                 dx_codes_size: Optional[int] = None,
                 icu_inputs_map: Optional[ReducedCodeMapN1] = None,
                 icu_procedures_size: Optional[int] = None,
                 hosp_procedures_size: Optional[int] = None,
                 demographic_size: Optional[int] = None,
                 observables_size: Optional[int] = None, *,
                 key: jrandom.PRNGKey):
        super().__init__()
        if config.dx_codes:  # not None and not 0.
            assert dx_codes_size is not None, "dx_codes_size must be provided"
            self._f_dx_codes_emb, key = self.make_dx_codes_emb(dx_codes_size, config.dx_codes, key=key)

        if config.interventions:
            assert icu_inputs_map or icu_procedures_size or hosp_procedures_size, "Any of icu_inputs_map, icu_procedures_size or hosp_procedures_size must be provided."
            self._f_interventions_emb, key = self.make_interventions_emb(config.interventions,
                                                                         icu_inputs_map,
                                                                         icu_procedures_size,
                                                                         hosp_procedures_size,
                                                                         key=key)
        if config.demographic:
            assert demographic_size, "demographic_size must be provided."
            self._f_demographic_emb, key = self.make_demographics_emb(demographic_size, config.demographic, key=key)

        if config.observables:
            assert observables_size, "observables_size must be provided."
            self._f_observables_emb, _ = self.make_observables_emb(observables_size, config.observables, key=key)

    @eqx.filter_jit
    def dx_embdedding_vectors(self) -> jnp.ndarray:
        in_size = self._f_dx_codes_emb.in_size
        return eqx.filter_vmap(self._f_dx_codes_emb)(jnp.eye(in_size))

    @eqx.filter_jit
    def embed_dx_codes(self, dx_codes: CodesVector, dx_codes_history: CodesVector) -> EmbeddedAdmission:
        if self._f_dx_codes_emb is None:
            return EmbeddedAdmission()
        dx_codes_emb = self._f_dx_codes_emb(dx_codes.vec)
        dx_codes_history_emb = self._f_dx_codes_emb(dx_codes_history.vec)
        return EmbeddedAdmission(dx_codes=dx_codes_emb, dx_codes_history=dx_codes_history_emb)

    @eqx.filter_jit
    def embed_interventions(self, interventions: SegmentedInpatientInterventions) -> EmbeddedAdmission:
        if self._f_interventions_emb is None:
            return EmbeddedAdmission()

        segments = [self._f_interventions_emb(icu_inputs, icu_procedures, hosp_procedures)
                    for icu_inputs, icu_procedures, hosp_procedures in interventions.iter_tuples()]
        return EmbeddedAdmission(interventions=jnp.vstack(segments))

    @eqx.filter_jit
    def embed_demographic(self, demographic: jnp.ndarray) -> EmbeddedAdmission:
        if self._f_demographic_emb is None:
            return EmbeddedAdmission()
        return EmbeddedAdmission(demographic=self._f_demographic_emb(demographic))

    def _embed_observables_segment(self, obs: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(self._f_observables_emb)(obs)

    def embed_observables(self, observables: InpatientObservables | SegmentedInpatientObservables) -> EmbeddedAdmission:
        if self._f_observables_emb is None:
            return EmbeddedAdmission()

        if isinstance(observables, SegmentedInpatientObservables):
            segments = [jnp.concatenate((obs.value, obs.mask), axis=1) for obs in observables]

        elif isinstance(observables, InpatientObservables):
            segments = [jnp.concatenate((observables.value, observables.mask), axis=1)]
        else:
            raise ValueError(f"Unrecognised observables type: {type(observables)}")

        emb_segments = tuple(self._embed_observables_segment(s) for s in segments)
        return EmbeddedAdmission(observables=emb_segments)

    def embed_admission(self, admission: SegmentedAdmission | Admission,
                        demographic_input: Optional[jnp.ndarray]) -> EmbeddedAdmission:
        """
        Embeds an admission into fixed vectors as described above.
        """

        return eqx.combine(self.embed_observables(admission.observables),
                           self.embed_demographic(demographic_input),
                           self.embed_interventions(admission.interventions),
                           self.embed_dx_codes(admission.dx_codes, admission.dx_codes_history))

    def __call__(self, inpatient: SegmentedPatient | Patient, admission_demographic: Dict[str, jnp.ndarray]) -> List[
        EmbeddedAdmission]:
        """
        Embeds all the admissions of an inpatient into fixed vectors as \
        described above.
        """
        if isinstance(inpatient, Patient):
            assert self._f_interventions_emb is None, "Interventions embedding is " \
                                                      "not supported for (unsegmented) patient."

        return [
            self.embed_admission(admission, admission_demographic.get(admission.admission_id))
            for admission in inpatient.admissions
        ]


class AdmissionSequentialEmbeddingsConfig(Config):
    dx_codes: int = 0
    demographic: int = 0
    observables: int = 0
    sequence: int = 50

    def to_admission_embeddings_config(self) -> AdmissionEmbeddingsConfig:
        return AdmissionEmbeddingsConfig(dx_codes=self.dx_codes,
                                         interventions=None,
                                         demographic=self.demographic,
                                         observables=self.observables)


class EmbeddedAdmissionSequence(VxData):
    dx_codes_history: jnp.ndarray  # (config.dx, )
    demographic: jnp.ndarray  # (config.demo, )
    sequence: jnp.ndarray  # (n_timestamps, config.sequence)


class AdmissionSequentialEmbedding(eqx.Module):
    _f_components_emb: PatientEmbedding
    _f_sequence_emb: Callable

    def __init__(self, config: AdmissionSequentialEmbeddingsConfig,
                 dx_codes_size: Optional[int] = None,
                 demographic_size: Optional[int] = None,
                 observables_size: Optional[int] = None, *,
                 key: jrandom.PRNGKey):
        (components_emb_key, sequence_emb_key) = jrandom.split(key, )

        self._f_components_emb = PatientEmbedding(config=config.to_admission_embeddings_config(),
                                                  dx_codes_size=dx_codes_size,
                                                  icu_inputs_map=None,
                                                  icu_procedures_size=None,
                                                  hosp_procedures_size=None,
                                                  demographic_size=demographic_size,
                                                  observables_size=observables_size,
                                                  key=components_emb_key)

        self._f_sequence_emb = eqx.nn.MLP(config.dx_codes + config.demographic + config.observables,
                                          config.sequence,
                                          config.sequence * 2,
                                          depth=1,
                                          final_activation=jnp.tanh,
                                          key=sequence_emb_key)

    @eqx.filter_jit
    def _embed_sequence_features(self, dx_history_emb, demo_emb, obs_e_i):
        return self._f_sequence_emb(jnp.hstack((dx_history_emb, demo_emb, obs_e_i)))

    def embed_admission_components(self, components: EmbeddedAdmission) -> EmbeddedAdmissionSequence:
        """ Embeds an admission into fixed vectors as described above."""
        dx_history_emb = components.dx_codes_history
        demo_emb = components.demographic
        assert len(components.observables) == 1, "Only one segment is supported"
        obs_emb = components.observables[0]
        sequence_e = [self._embed_sequence_features(dx_history_emb, demo_emb, obs_e_i) for obs_e_i in obs_emb]
        return EmbeddedAdmissionSequence(dx_codes_history=dx_history_emb,
                                         sequence=sequence_e,
                                         demographic=demo_emb)

    def __call__(self, inpatient: Patient, admission_demographic: Dict[str, jnp.ndarray]) -> List[
        EmbeddedAdmissionSequence]:
        """
        Embeds all the admissions of an inpatient into fixed vectors as \
        described above.
        """
        assert not isinstance(inpatient, SegmentedPatient), "SegmentedPatient not supported"

        return [
            self.embed_admission_components(emb_admission)
            for emb_admission in self._f_components_emb(inpatient, admission_demographic)
        ]
