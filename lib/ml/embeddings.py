from __future__ import annotations
from typing import (Any, Dict, List, Callable, Optional)
from abc import abstractmethod
import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx

from ..ehr import (Patient, Admission, StaticInfo, MIMIC4ICUDatasetScheme,
                   MIMICDatasetScheme, AggregateRepresentation, InpatientInput,
                   InpatientInterventions, DemographicVectorConfig)

class EmbeddedAdmission(eqx.Module):
    pass

class EmbeddedInAdmission(EmbeddedAdmission):
    dx0: jnp.ndarray
    inp_proc_demo: Optional[jnp.ndarray]

class EmbeddedOutAdmission(EmbeddedAdmission):
    dx: jnp.ndarray
    demo: Optional[jnp.ndarray]


class PatientEmbeddingDimensions(eqx.Module):
    dx: int = 30
    demo: int = 5

class OutpatientEmbeddingDimensions(PatientEmbeddingDimensions):
    pass

class InpatientEmbeddingDimensions(PatientEmbeddingDimensions):
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
    f_dx_emb: Callable
    f_dem_emb: Callable

    def __init__(self, dims: OutpatientEmbeddingDimensions,
                 scheme: MIMICDatasetScheme,
                 demographic_vector_config: DemographicVectorConfig,
                 key: "jax.random.PRNGKey"):
        super().__init__()
        (dx_emb_key, dem_emb_key) = jrandom.split(key, 2)

        self.f_dx_emb = eqx.nn.MLP(len(scheme.dx_target),
                                   dims.dx,
                                   width_size=dims.dx * 5,
                                   depth=1,
                                   key=dx_emb_key)

        demo_input_size = scheme.demographic_vector_size(
            demographic_vector_config)
        self.f_dem_emb = eqx.nn.MLP(demo_input_size,
                                    dims.demo,
                                    dims.demo * 5,
                                    depth=1,
                                    key=dem_emb_key)

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
        dx_emb = self.f_dx_emb(dx_vec)
        demo_e = self.f_dem_emb(demo)
        return EmbeddedOutAdmission(dx=dx_emb, demo=demo_e)


    def embed_admission(self, static_info: StaticInfo,
                        admission: Admission) -> EmbeddedOutAdmission:
        """ Embeds an admission into fixed vectors as described above."""
        demo = static_info.demographic_vector(admission.admission_dates[0])
        return self._embed_admission(demo, admission.dx_codes.vec)




class InpatientEmbedding(PatientEmbedding):
    """
    Embeds an inpatient admission into fixed vectors:
        - Embdedded discharge codes history.
        - A sequence of embedded vectors each fusing the input, procedure \
            and demographic information.
    """
    f_inp_agg: Callable
    f_inp_emb: Callable
    f_proc_emb: Callable
    f_int_emb: Callable

    def __init__(self, dims: InpatientEmbeddingDimensions,
                 scheme: MIMIC4ICUDatasetScheme,
                 demographic_vector_config: DemographicVectorConfig,
                 key: "jax.random.PRNGKey"):
        (super_key, inp_agg_key, inp_emb_key, proc_emb_key,
         int_emb_key) = jrandom.split(key, 5)
        super().__init__(dims=dims,
                         scheme=scheme,
                         demographic_vector_config=demographic_vector_config,
                         key=super_key)
        self.f_inp_agg = AggregateRepresentation(scheme.int_input_source,
                                                 scheme.int_input_target,
                                                 inp_agg_key, 'jax')
        self.f_inp_emb = eqx.nn.MLP(len(scheme.int_input_target),
                                    dims.inp,
                                    dims.inp * 5,
                                    depth=1,
                                    key=inp_emb_key)
        self.f_proc_emb = eqx.nn.MLP(len(scheme.int_proc_target),
                                     dims.proc,
                                     dims.proc * 5,
                                     depth=1,
                                     key=proc_emb_key)
        self.f_int_emb = eqx.nn.MLP(dims.inp + dims.proc + dims.demo,
                                    dims.inp_proc_demo,
                                    dims.inp_proc_demo * 5,
                                    depth=1,
                                    key=int_emb_key)

    @eqx.filter_jit
    def _embed_demo(self, demo: jnp.ndarray) -> jnp.ndarray:
        """Embeds the demographics into a fixed vector."""
        return self.f_dem_emb(demo)

    @eqx.filter_jit
    def _embed_segment(self, inp: InpatientInput, proc: InpatientInput,
                       demo_e: jnp.ndarray) -> jnp.ndarray:
        """
        Embeds a  of the intervention (procedures and inputs) \
        and demographics into a fixed vector.
        """

        inp_emb = self.f_inp_emb(self.f_inp_agg(inp))
        proc_emb = self.f_proc_emb(proc)
        return self.f_int_emb(jnp.hstack([inp_emb, proc_emb, demo_e]))

    @eqx.filter_jit
    def embed_dx(self, x: jnp.ndarray) -> jnp.ndarray:
        """Embeds the discharge codes history into a fixed vector."""
        return self.f_dx_emb(x)

    @eqx.filter_jit
    def _embed_admission(self, demo: jnp.ndarray, dx_history_vec: jnp.ndarray,
                         segmented_inp: jnp.ndarray,
                         segmented_proc: jnp.ndarray) -> EmbeddedInAdmission:
        """ Embeds an admission into fixed vectors as described above."""

        def _embed_segment(inp, proc):
            return self._embed_segment(inp, proc, demo_e)

        dx_emb = self.embed_dx(dx_history_vec)
        demo_e = self._embed_demo(demo)
        inp_proc_demo = jax.vmap(_embed_segment)(segmented_inp, segmented_proc)
        return EmbeddedInAdmission(dx0=dx_emb, inp_proc_demo=inp_proc_demo)

    def embed_admission(self, static_info: StaticInfo, admission: Admission):
        demo = static_info.demographic_vector(admission.admission_dates[0])
        return self._embed_admission(demo, admission.dx_codes_history.vec,
                                     admission.interventions.segmented_input,
                                     admission.interventions.segmented_proc)
