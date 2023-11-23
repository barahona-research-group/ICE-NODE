from __future__ import annotations
from typing import (Any, Dict, List, Callable, Optional, Tuple)
from abc import abstractmethod
import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx

from ..ehr import (Patient, Admission, StaticInfo, DatasetScheme,
                   AggregateRepresentation, InpatientInput,
                   InpatientInterventions, DemographicVectorConfig)
from ..base import Config, Data


class EmbeddedAdmission(Data):
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


class InpatientEmbeddingConfig(PatientEmbeddingConfig):
    inp: int = 10
    proc: int = 10
    inp_proc_demo: int = 15


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

        self._f_dx_emb = eqx.nn.MLP(len(schemes[1].dx),
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


class InpatientEmbedding(PatientEmbedding):
    """
    Embeds an inpatient admission into fixed vectors:
        - Embdedded discharge codes history.
        - A sequence of embedded vectors each fusing the input, procedure \
            and demographic information.
    """
    _f_inp_agg: Callable
    _f_inp_emb: Callable
    _f_proc_emb: Callable
    _f_int_emb: Callable

    def __init__(self, config: InpatientEmbeddingConfig,
                 schemes: Tuple[DatasetScheme],
                 demographic_vector_config: DemographicVectorConfig,
                 key: "jax.random.PRNGKey"):
        (super_key, inp_agg_key, inp_emb_key, proc_emb_key,
         int_emb_key) = jrandom.split(key, 5)

        if schemes[1].demographic_vector_size(demographic_vector_config) == 0:
            config = eqx.tree_at(lambda d: d.demo, config, 0)

        super().__init__(config=config,
                         schemes=schemes,
                         demographic_vector_config=demographic_vector_config,
                         key=super_key)
        self._f_inp_agg = AggregateRepresentation(schemes[0].int_input,
                                                  schemes[1].int_input,
                                                  inp_agg_key, 'jax')
        self._f_inp_emb = eqx.nn.MLP(len(schemes[1].int_input),
                                     config.inp,
                                     config.inp * 5,
                                     final_activation=jnp.tanh,
                                     depth=1,
                                     key=inp_emb_key)
        self._f_proc_emb = eqx.nn.MLP(len(schemes[1].int_proc),
                                      config.proc,
                                      config.proc * 5,
                                      final_activation=jnp.tanh,
                                      depth=1,
                                      key=proc_emb_key)
        self._f_int_emb = eqx.nn.MLP(config.inp + config.proc + config.demo,
                                     config.inp_proc_demo,
                                     config.inp_proc_demo * 5,
                                     final_activation=jnp.tanh,
                                     depth=1,
                                     key=int_emb_key)

    @eqx.filter_jit
    def _embed_demo(self, demo: jnp.ndarray) -> jnp.ndarray:
        """Embeds the demographics into a fixed vector."""
        return self._f_dem_emb(demo)

    @eqx.filter_jit
    def _embed_segment(self, inp: InpatientInput, proc: InpatientInput,
                       demo_e: jnp.ndarray) -> jnp.ndarray:
        """
        Embeds a  of the intervention (procedures and inputs) \
        and demographics into a fixed vector.
        """

        inp_emb = self._f_inp_emb(self._f_inp_agg(inp))
        proc_emb = self._f_proc_emb(proc)
        return self._f_int_emb(jnp.hstack([inp_emb, proc_emb, demo_e]))

    @eqx.filter_jit
    def embed_dx(self, x: jnp.ndarray) -> jnp.ndarray:
        """Embeds the discharge codes history into a fixed vector."""
        return self._f_dx_emb(x)

    @eqx.filter_jit
    def _embed_admission(self, demo: jnp.ndarray, dx_history_vec: jnp.ndarray,
                         segmented_inp: jnp.ndarray,
                         segmented_proc: jnp.ndarray) -> EmbeddedInAdmission:
        """ Embeds an admission into fixed vectors as described above."""

        demo_e = self._embed_demo(demo)
        dx_emb = self.embed_dx(dx_history_vec)

        def _embed_segment(inp, proc):
            return self._embed_segment(inp, proc, demo_e)

        inp_proc_demo = jax.vmap(_embed_segment)(segmented_inp, segmented_proc)
        return EmbeddedInAdmission(dx0=dx_emb, inp_proc_demo=inp_proc_demo)

    def embed_admission(self, static_info: StaticInfo, admission: Admission):
        demo = static_info.demographic_vector(admission.admission_dates[0])
        return self._embed_admission(demo, admission.dx_codes_history.vec,
                                     admission.interventions.segmented_input,
                                     admission.interventions.segmented_proc)


class DeepMindPatientEmbeddingConfig(Config):
    dx: int = 30
    demo: int = 5
    obs: int = 20
    sequence: int = 50


class DeepMindEmbeddedAdmission(Data):
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

    @eqx.filter_jit
    def _embed_admission(
            self, demo: jnp.ndarray, dx_history_vec: jnp.ndarray,
            observables: jnp.ndarray) -> DeepMindEmbeddedAdmission:
        """ Embeds an admission into fixed vectors as described above."""

        dx_emb = self.embed_dx(dx_history_vec)
        demo_e = self._embed_demo(demo)
        obs_e = jax.vmap(self._f_obs_emb)(observables)

        def _embed_sequence_features(obs_e_i):
            return self._f_sequence_emb(jnp.hstack((dx_emb, demo_e, obs_e_i)))

        sequence_e = eqx.filter_vmap(_embed_sequence_features)(obs_e)
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
