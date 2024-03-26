from __future__ import annotations

from abc import abstractmethod
from typing import (List, Callable, Optional, Tuple)

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom

from ..base import Config, VxData
from ..ehr import (Patient, Admission, StaticInfo, DatasetScheme,
                   DemographicVectorConfig)
from ..ehr.coding_scheme import ReducedCodeMapN1, AggregationLiteral


class EmbeddedAdmission(VxData):
    pass


class EmbeddedInAdmission(EmbeddedAdmission):
    dx0: jnp.ndarray
    inp_proc_demo: Optional[jnp.ndarray]


class EmbeddedOutAdmission(EmbeddedAdmission):
    dx: jnp.ndarray
    demo: Optional[jnp.ndarray]


class LiteEmbeddedInAdmission(EmbeddedAdmission):
    dx0: jnp.ndarray
    demo: Optional[jnp.ndarray]


class PatientEmbeddingConfig(Config):
    dx: int = 30
    demo: int = 5


class OutpatientEmbeddingConfig(PatientEmbeddingConfig):
    pass


class InpatientLiteEmbeddingConfig(PatientEmbeddingConfig):
    pass


class InterventionsEmbeddingsConfig(Config):
    icu_inputs: int = 10
    icu_procedures: int = 10
    hosp_procedures: int = 10
    interventions: int = 20


class PatientEmbedding(eqx.Module):
    """
    Embeds an inpatient admission into fixed vectors:
        - Embdedded discharge codes history.
        - A sequence of embedded vectors each fusing the input, procedure \
            and demographic information.
    """
    _f_dx_emb: Callable
    _f_dem_emb: Callable

    def __init__(self, config: OutpatientEmbeddingConfig,
                 schemes: Tuple[DatasetScheme],
                 demographic_vector_config: DemographicVectorConfig,
                 key: "jax.random.PRNGKey"):
        super().__init__()
        (dx_emb_key, dem_emb_key) = jrandom.split(key, 2)

        self._f_dx_emb = eqx.nn.MLP(len(schemes[1].dx_discharge),
                                    config.dx,
                                    width_size=config.dx * 5,
                                    depth=1,
                                    final_activation=jnp.tanh,
                                    key=dx_emb_key)

        demo_input_size = schemes[1].demographic_vector_size(
            demographic_vector_config)
        if demo_input_size > 0:
            self._f_dem_emb = eqx.nn.MLP(demo_input_size,
                                         config.demo,
                                         config.demo * 5,
                                         depth=1,
                                         final_activation=jnp.tanh,
                                         key=dem_emb_key)
        else:
            self._f_dem_emb = lambda x: jnp.array([], dtype=jnp.float16)

    @eqx.filter_jit
    def dx_embdeddings(self):
        in_size = self._f_dx_emb.in_size
        return eqx.filter_vmap(self._f_dx_emb)(jnp.eye(in_size))

    @abstractmethod
    def embed_admission(self, static_info: StaticInfo,
                        admission: Admission) -> EmbeddedAdmission:
        """
        Embeds an admission into fixed vectors as described above.
        """
        pass

    def __call__(self, inpatient: Patient) -> List[EmbeddedAdmission]:
        """
        Embeds all the admissions of an inpatient into fixed vectors as \
        described above.
        """
        return [
            self.embed_admission(inpatient.static_info, admission)
            for admission in inpatient.admissions
        ]


class OutpatientEmbedding(PatientEmbedding):

    @eqx.filter_jit
    def _embed_admission(self, demo: jnp.ndarray,
                         dx_vec: jnp.ndarray) -> EmbeddedOutAdmission:
        dx_emb = self._f_dx_emb(dx_vec)
        demo_e = self._f_dem_emb(demo)
        return EmbeddedOutAdmission(dx=dx_emb, demo=demo_e)

    def embed_admission(self, static_info: StaticInfo,
                        admission: Admission) -> EmbeddedOutAdmission:
        """ Embeds an admission into fixed vectors as described above."""
        demo = static_info.demographic_vector(admission.admission_dates[0])
        return self._embed_admission(demo, admission.dx_codes.vec)


class InpatientLiteEmbedding(PatientEmbedding):

    @eqx.filter_jit
    def _embed_admission(self, demo: jnp.ndarray,
                         dx_vec: jnp.ndarray) -> EmbeddedOutAdmission:
        dx_emb = self._f_dx_emb(dx_vec)
        demo_e = self._f_dem_emb(demo)
        return LiteEmbeddedInAdmission(dx0=dx_emb, demo=demo_e)

    def embed_admission(self, static_info: StaticInfo,
                        admission: Admission) -> EmbeddedOutAdmission:
        """ Embeds an admission into fixed vectors as described above."""
        demo = static_info.demographic_vector(admission.admission_dates[0])
        return self._embed_admission(demo, admission.dx_codes_history.vec)


class GroupEmbedding(eqx.Module):
    groups_split: Tuple[int, ...] = eqx.static_field()
    groups_argsort: Tuple[int, ...] = eqx.static_field()
    source_size: int = eqx.static_field()
    target_size: int = eqx.static_field()
    groups_aggregation: Tuple[Callable, ...]
    linear: eqx.nn.Linear

    @staticmethod
    def group_function(aggregation: AggregationLiteral, input_size: int, group_key: "jax.random.PRNGKey") -> Callable:
        if aggregation == 'sum':
            return jnp.sum
        if aggregation == 'or':
            return jnp.any
        if aggregation == 'w_sum':
            return eqx.nn.Linear(input_size, 1, use_bias=False, key=group_key)
        raise ValueError(f"Unrecognised aggregation: {aggregation}")

    def __init__(self, reduced_map: ReducedCodeMapN1, embeddings_size: int, key: "jax.random.PRNGKey"):
        self.groups_split = reduced_map.groups_split
        self.groups_argsort = reduced_map.groups_argsort
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
        y = x[self.groups_argsort]
        return tuple(jnp.hsplit(y, self.groups_split)[:-1])

    def __call__(self, x: jnp.ndarray):
        y = jnp.hstack(self.split_source_array(x))
        return self.linear(y)


class InterventionsEmbeddings(eqx.Module):
    """
    Embeds an inpatient admission into fixed vectors:
        - Embdedded discharge codes history.
        - A sequence of embedded vectors each fusing the input, procedure \
            and demographic information.
    """
    config: InterventionsEmbeddingsConfig = eqx.static_field()
    _f_icu_inputs_emb: GroupEmbedding
    _f_icu_procedures_emb: eqx.nn.Linear
    _f_hosp_procedures_emb: eqx.nn.Linear
    _f_emb: eqx.nn.Linear
    activation: Callable
    final_activation: Callable

    def __init__(self, config: InterventionsEmbeddingsConfig,
                 icu_inputs_map: ReducedCodeMapN1,
                 icu_procedures_source_size: int,
                 hosp_procedures_source_size: int,
                 activation: Callable = jnn.relu,
                 final_activation: Callable = lambda x: x, *,
                 key: "jax.random.PRNGKey"):
        super().__init__()
        self.config = config
        self.activation = activation
        self.final_activation = final_activation
        (icu_inputs_key, icu_procedures_key, hosp_procedures_key, interventions_key) = jrandom.split(key, 4)

        self._f_icu_inputs_emb = GroupEmbedding(icu_inputs_map, config.icu_inputs, icu_inputs_key)
        self._f_icu_procedures_emb = eqx.nn.Linear(icu_procedures_source_size,
                                                   config.icu_procedures,
                                                   use_bias=False,
                                                   key=icu_procedures_key)
        self._f_hosp_procedures_emb = eqx.nn.Linear(hosp_procedures_source_size,
                                                    config.hosp_procedures,
                                                    use_bias=False,
                                                    key=hosp_procedures_key)
        self._f_emb = eqx.nn.Linear(config.icu_inputs + config.icu_procedures + config.hosp_procedures,
                                    config.interventions,
                                    use_bias=False,
                                    key=interventions_key)

    def __call__(self, icu_inputs: jnp.ndarray, icu_procedures: jnp.ndarray, hosp_procedures: jnp.ndarray):
        y_icu_inputs = self._f_icu_inputs_emb(icu_inputs)
        y_icu_procedures = self._f_icu_procedures_emb(icu_procedures)
        y_hosp_procedures = self._f_hosp_procedures_emb(hosp_procedures)
        y = jnp.hstack((y_icu_inputs, y_icu_procedures, y_hosp_procedures))
        return self.final_activation(self._f_emb(self.activation(y)))

    # @eqx.filter_jit
    # def _embed_demo(self, demo: jnp.ndarray) -> jnp.ndarray:
    #     """Embeds the demographics into a fixed vector."""
    #     return self._f_dem_emb(demo)
    #
    # @eqx.filter_jit
    # def _embed_segment(self, inp: InpatientInput, proc: InpatientInput,
    #                    demo_e: jnp.ndarray) -> jnp.ndarray:
    #     """
    #     Embeds a  of the intervention (procedures and inputs) \
    #     and demographics into a fixed vector.
    #     """
    #
    #     inp_emb = self._f_inp_emb(self._f_inp_agg(inp))
    #     proc_emb = self._f_proc_emb(proc)
    #     return self._f_int_emb(jnp.hstack([inp_emb, proc_emb, demo_e]))
    #
    # @eqx.filter_jit
    # def embed_dx(self, x: jnp.ndarray) -> jnp.ndarray:
    #     """Embeds the discharge codes history into a fixed vector."""
    #     return self._f_dx_emb(x)
    #
    # @eqx.filter_jit
    # def _embed_admission(self, demo: jnp.ndarray, dx_history_vec: jnp.ndarray,
    #                      segmented_inp: jnp.ndarray,
    #                      segmented_proc: jnp.ndarray) -> EmbeddedInAdmission:
    #     """ Embeds an admission into fixed vectors as described above."""
    #
    #     demo_e = self._embed_demo(demo)
    #     dx_emb = self.embed_dx(dx_history_vec)
    #
    #     def _embed_segment(inp, proc):
    #         return self._embed_segment(inp, proc, demo_e)
    #
    #     inp_proc_demo = jax.vmap(_embed_segment)(segmented_inp, segmented_proc)
    #     return EmbeddedInAdmission(dx0=dx_emb, inp_proc_demo=inp_proc_demo)
    #
    # def embed_admission(self, static_info: StaticInfo, admission: Admission):
    #     demo = static_info.demographic_vector(admission.admission_dates[0])
    #     return self._embed_admission(demo, admission.dx_codes_history.vec,
    #                                  admission.interventions.segmented_input,
    #                                  admission.interventions.segmented_proc)


class DeepMindPatientEmbeddingConfig(Config):
    dx: int = 30
    demo: int = 5
    obs: int = 20
    sequence: int = 50


class DeepMindEmbeddedAdmission(VxData):
    dx0: jnp.ndarray
    demo: jnp.ndarray
    sequence: jnp.ndarray


class DeepMindPatientEmbedding(PatientEmbedding):
    _f_dx_emb: Callable
    _f_dem_emb: Callable
    _f_obs_emb: Callable
    _f_sequence_emb: Callable

    def __init__(self, config: OutpatientEmbeddingConfig,
                 schemes: Tuple[DatasetScheme],
                 demographic_vector_config: DemographicVectorConfig,
                 key: "jax.random.PRNGKey"):
        (super_key, obs_emb_key, sequence_emb_key) = jrandom.split(key, 3)
        super().__init__(config=config,
                         schemes=schemes,
                         demographic_vector_config=demographic_vector_config,
                         key=super_key)

        self._f_obs_emb = eqx.nn.MLP(len(schemes[1].obs) * 2,
                                     config.obs,
                                     width_size=config.obs * 3,
                                     depth=1,
                                     final_activation=jnp.tanh,
                                     key=obs_emb_key)

        self._f_sequence_emb = eqx.nn.MLP(config.dx + config.demo + config.obs,
                                          config.sequence,
                                          config.sequence * 2,
                                          depth=1,
                                          final_activation=jnp.tanh,
                                          key=sequence_emb_key)

    @eqx.filter_jit
    def embed_dx(self, x: jnp.ndarray) -> jnp.ndarray:
        """Embeds the discharge codes history into a fixed vector."""
        return self._f_dx_emb(x)

    @eqx.filter_jit
    def dx_embdeddings(self):
        in_size = self._f_dx_emb.in_size
        return eqx.filter_vmap(self._f_dx_emb)(jnp.eye(in_size))

    @eqx.filter_jit
    def _embed_demo(self, demo: jnp.ndarray) -> jnp.ndarray:
        """Embeds the demographics into a fixed vector."""
        return self._f_dem_emb(demo)

    def _embed_admission(
            self, demo: jnp.ndarray, dx_history_vec: jnp.ndarray,
            observables: jnp.ndarray) -> DeepMindEmbeddedAdmission:
        """ Embeds an admission into fixed vectors as described above."""

        dx_emb = self.embed_dx(dx_history_vec)
        demo_e = self._embed_demo(demo)
        obs_e = jax.vmap(self._f_obs_emb)(observables)

        def _embed_sequence_features(obs_e_i):
            return self._f_sequence_emb(jnp.hstack((dx_emb, demo_e, obs_e_i)))

        sequence_e = [_embed_sequence_features(obs_e_i) for obs_e_i in obs_e]
        return DeepMindEmbeddedAdmission(dx0=dx_emb,
                                         sequence=sequence_e,
                                         demo=demo_e)

    def embed_admission(self, static_info: StaticInfo, admission: Admission):
        demo = static_info.demographic_vector(admission.admission_dates[0])

        assert (not isinstance(admission.observables,
                               list)), "Observables should not be fragmented"

        observables = jnp.hstack(
            (admission.observables.value, admission.observables.mask))

        return self._embed_admission(demo, admission.dx_codes_history.vec,
                                     observables)
