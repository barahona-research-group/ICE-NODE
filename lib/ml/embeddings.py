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
    SegmentedAdmission, SegmentedPatient


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

    def embed_admission(self, admission: SegmentedAdmission,
                        demographic_input: Optional[jnp.ndarray]) -> EmbeddedAdmission:
        """
        Embeds an admission into fixed vectors as described above.
        """

        return eqx.combine(self.embed_observables(admission.observables),
                           self.embed_demographic(demographic_input),
                           self.embed_interventions(admission.interventions),
                           self.embed_dx_codes(admission.dx_codes, admission.dx_codes_history))

    def __call__(self, inpatient: SegmentedPatient, admission_demographic: Dict[str, jnp.ndarray]) -> List[
        EmbeddedAdmission]:
        """
        Embeds all the admissions of an inpatient into fixed vectors as \
        described above.
        """
        return [
            self.embed_admission(admission, admission_demographic.get(admission.admission_id))
            for admission in inpatient.admissions
        ]

# class DeepMindPatientEmbeddingConfig(Config):
#     dx: int = 30
#     demo: int = 5
#     obs: int = 20
#     sequence: int = 50
#
# class DeepMindEmbeddedAdmission(VxData):
#     dx0: jnp.ndarray
#     demo: jnp.ndarray
#     sequence: jnp.ndarray
#
# class DeepMindPatientEmbedding(PatientEmbedding):
#     _f_dx_emb: Callable
#     _f_dem_emb: Callable
#     _f_obs_emb: Callable
#     _f_sequence_emb: Callable
#
#     def __init__(self, config: OutpatientEmbeddingConfig,
#                  schemes: Tuple[DatasetScheme],
#                  demographic_vector_config: DemographicVectorConfig,
#                  key: jrandom.PRNGKey):
#         (super_key, obs_emb_key, sequence_emb_key) = jrandom.split(key, 3)
#         super().__init__(config=config,
#                          schemes=schemes,
#                          demographic_vector_config=demographic_vector_config,
#                          key=super_key)
#
#         self._f_sequence_emb = eqx.nn.MLP(config.dx + config.demo + config.obs,
#                                           config.sequence,
#                                           config.sequence * 2,
#                                           depth=1,
#                                           final_activation=jnp.tanh,
#                                           key=sequence_emb_key)
#
#     @eqx.filter_jit
#     def embed_dx(self, x: jnp.ndarray) -> jnp.ndarray:
#         """Embeds the discharge codes history into a fixed vector."""
#         return self._f_dx_emb(x)
#
#     @eqx.filter_jit
#     def dx_embdeddings(self):
#         in_size = self._f_dx_emb.in_size
#         return eqx.filter_vmap(self._f_dx_emb)(jnp.eye(in_size))
#
#     @eqx.filter_jit
#     def _embed_demo(self, demo: jnp.ndarray) -> jnp.ndarray:
#         """Embeds the demographics into a fixed vector."""
#         return self._f_dem_emb(demo)
#
#     def _embed_admission(
#             self, demo: jnp.ndarray, dx_history_vec: jnp.ndarray,
#             observables: jnp.ndarray) -> DeepMindEmbeddedAdmission:
#         """ Embeds an admission into fixed vectors as described above."""
#
#         dx_emb = self.embed_dx(dx_history_vec)
#         demo_e = self._embed_demo(demo)
#         obs_e = jax.vmap(self._f_obs_emb)(observables)
#
#         def _embed_sequence_features(obs_e_i):
#             return self._f_sequence_emb(jnp.hstack((dx_emb, demo_e, obs_e_i)))
#
#         sequence_e = [_embed_sequence_features(obs_e_i) for obs_e_i in obs_e]
#         return DeepMindEmbeddedAdmission(dx0=dx_emb,
#                                          sequence=sequence_e,
#                                          demo=demo_e)
#
#     def embed_admission(self, static_info: StaticInfo, admission: Admission):
#         demo = static_info.demographic_vector(admission.admission_dates[0])
#
#         assert (not isinstance(admission.observables,
#                                list)), "Observables should not be fragmented"
#
#         observables = jnp.hstack(
#             (admission.observables.value, admission.observables.mask))
#
#         return self._embed_admission(demo, admission.dx_codes_history.vec,
#                                      observables)
