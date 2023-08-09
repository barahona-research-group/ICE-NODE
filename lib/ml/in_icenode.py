"""."""
from __future__ import annotations
from typing import (Any, Dict, List, Callable)
import zipfile
import logging
import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jrandom
import jax.tree_util as jtu
import equinox as eqx

from ..utils import model_params_scaler, tqdm_constructor, write_config
from ..ehr import (Patients, Patient, Admission, InpatientObservables,
                   StaticInfo, AdmissionPrediction, Predictions,
                   MIMIC4ICUDatasetScheme, AggregateRepresentation,
                   InpatientInput, CodesVector, DemographicVectorConfig)

from .base_models import (NeuralODE, ObsStateUpdate, NeuralODE_JAX)
from ..utils import translate_path


class EmbeddedAdmission(eqx.Module):
    state_dx_e0: jnp.ndarray
    int_e: jnp.ndarray


class InICENODEDimensions(eqx.Module):
    state_m: int = 15
    state_dx_e: int = 30
    state_obs_e: int = 25
    input_e: int = 10
    proc_e: int = 10
    demo_e: int = 5
    int_e: int = 15

    def to_dict(self) -> Dict[str, int]:
        return self.__dict__


class InpatientEmbedding(eqx.Module):
    """
    Embeds an inpatient admission into fixed vectors:
        - Embdedded discharge codes history.
        - A sequence of embedded vectors each fusing the input, procedure \
            and demographic information.
    """
    f_dx_emb: Callable
    f_dem_emb: Callable
    f_inp_agg: Callable
    f_inp_emb: Callable
    f_proc_emb: Callable
    f_int_emb: Callable

    def __init__(self, scheme: MIMIC4ICUDatasetScheme,
                 demographic_vector_config: DemographicVectorConfig,
                 dims: InICENODEDimensions, key: "jax.random.PRNGKey"):
        super().__init__()
        (dx_emb_key, inp_agg_key, inp_emb_key, proc_emb_key, dem_emb_key,
         int_emb_key) = jrandom.split(key, 6)

        self.f_dx_emb = eqx.nn.MLP(len(scheme.dx_target),
                                   dims.state_dx_e,
                                   dims.state_dx_e * 5,
                                   depth=1,
                                   key=dx_emb_key)
        self.f_inp_agg = AggregateRepresentation(scheme.int_input_source,
                                                 scheme.int_input_target,
                                                 inp_agg_key, 'jax')
        self.f_inp_emb = eqx.nn.MLP(len(scheme.int_input_target),
                                    dims.input_e,
                                    dims.input_e * 5,
                                    depth=1,
                                    key=inp_emb_key)
        self.f_proc_emb = eqx.nn.MLP(len(scheme.int_proc_target),
                                     dims.proc_e,
                                     dims.proc_e * 5,
                                     depth=1,
                                     key=proc_emb_key)
        demo_input_size = scheme.demographic_vector_size(
            demographic_vector_config)
        self.f_dem_emb = eqx.nn.MLP(demo_input_size,
                                    dims.demo_e,
                                    dims.demo_e * 5,
                                    depth=1,
                                    key=dem_emb_key)
        self.f_int_emb = eqx.nn.MLP(dims.input_e + dims.proc_e + dims.demo_e,
                                    dims.int_e,
                                    dims.int_e * 5,
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

        inp_emb = self.f_inp_emb(inp)
        proc_emb = self.f_proc_emb(proc)
        return self.f_int_emb(jnp.hstack([inp_emb, proc_emb, demo_e]))

    @eqx.filter_jit
    def embed_dx(self, x: jnp.ndarray) -> jnp.ndarray:
        """Embeds the discharge codes history into a fixed vector."""
        return self.f_dx_emb(x)

    def embed_admission(self, static_info: StaticInfo,
                        admission: Admission) -> EmbeddedAdmission:
        """ Embeds an admission into fixed vectors as described above."""
        dx_emb = self.embed_dx(admission.dx_codes_history.vec)

        demo = static_info.demographic_vector(admission.admission_dates[0])
        demo_e = self._embed_demo(demo)

        int_ = admission.interventions.segment_input(self.f_inp_agg)
        int_e = jnp.vstack([
            self._embed_segment(inp, proc, demo_e)
            for inp, proc in zip(int_.segmented_input, int_.segmented_proc)
        ])
        return EmbeddedAdmission(state_dx_e0=dx_emb, int_e=int_e)

    def __call__(self, inpatient: Patient) -> List[EmbeddedAdmission]:
        """
        Embeds all the admissions of an inpatient into fixed vectors as \
        described above.
        """
        return [
            self.embed_admission(inpatient.static_info, admission)
            for admission in inpatient.admissions
        ]


class InICENODE(eqx.Module):
    """
    The InICENODE model. It is composed of the following components:
        - f_emb: Embedding function.
        - f_obs_dec: Observation decoder.
        - f_dx_dec: Discharge codes decoder.
        - f_dyn: Dynamics function.
        - f_update: Update function.
    """
    f_emb: Callable[[Admission], EmbeddedAdmission]
    f_obs_dec: Callable
    f_dx_dec: Callable
    f_dyn: Callable
    f_update: Callable

    scheme: MIMIC4ICUDatasetScheme = eqx.static_field()
    dims: InICENODEDimensions = eqx.static_field()
    demographic_vector_config: DemographicVectorConfig = eqx.static_field()

    def __init__(self, scheme: MIMIC4ICUDatasetScheme,
                 demographic_vector_config: DemographicVectorConfig,
                 dims: InICENODEDimensions, key: "jax.random.PRNGKey"):
        super().__init__()
        self.dims = dims
        self.demographic_vector_config = demographic_vector_config
        self.scheme = scheme
        (emb_key, obs_dec_key, dx_dec_key, dyn_key,
         update_key) = jrandom.split(key, 5)
        self.f_emb = InpatientEmbedding(
            scheme=scheme,
            demographic_vector_config=demographic_vector_config,
            dims=dims,
            key=emb_key)
        self.f_obs_dec = eqx.nn.MLP(dims.state_obs_e,
                                    len(scheme.obs),
                                    dims.state_obs_e * 5,
                                    depth=1,
                                    key=obs_dec_key)
        self.f_dx_dec = eqx.nn.MLP(dims.state_dx_e,
                                   len(scheme.outcome),
                                   dims.state_dx_e * 5,
                                   depth=1,
                                   key=dx_dec_key)
        dyn_state_size = dims.state_obs_e + dims.state_dx_e + dims.state_m
        dyn_input_size = dyn_state_size + dims.int_e

        f_dyn = eqx.nn.MLP(in_size=dyn_input_size,
                           out_size=dyn_state_size,
                           activation=jnn.tanh,
                           depth=3,
                           width_size=dyn_state_size * 5,
                           key=dyn_key)
        f_dyn = model_params_scaler(f_dyn, 1e-3, eqx.is_inexact_array)
        self.f_dyn = NeuralODE_JAX(f_dyn, timescale=1.0)
        self.f_update = ObsStateUpdate(dyn_state_size,
                                       len(scheme.obs),
                                       key=update_key)

    @eqx.filter_jit
    def join_state(self, mem, obs, dx):
        if mem is None:
            mem = jnp.zeros((self.dims.state_m, ))
        if obs is None:
            obs = jnp.zeros((self.dims.state_obs_e, ))
        return jnp.hstack((mem, obs, dx))

    @eqx.filter_jit
    def split_state(self, state: jnp.ndarray):
        s1 = self.dims.state_m
        s2 = self.dims.state_m + self.dims.state_obs_e
        return jnp.hsplit(state, (s1, s2))

    def step_segment(self, state: jnp.ndarray, int_e: jnp.ndarray,
                     obs: InpatientObservables, t0: float, t1: float):

        def try_integrate(delta, state, int_e):
            if float(jax.block_until_ready(jnp.abs(delta))) > 1 / 3600.0:
                return self.f_dyn(delta, state, args=dict(control=int_e))[-1]
            else:
                logging.debug(f"Time diff is less than 1 second: {t_obs - t}")
                return state

        preds = []
        t = t0
        for t_obs, val, mask in zip(obs.time, obs.value, obs.mask):
            # if time-diff is more than 1 seconds, we integrate.
            state = try_integrate(t_obs - t, state, int_e)
            _, obs_e, _ = self.split_state(state)
            pred_obs = self.f_obs_dec(obs_e)
            state = self.f_update(state, pred_obs, val, mask)
            t = t_obs
            preds.append(pred_obs)

        state = try_integrate(t1 - t, state, int_e)

        if len(preds) > 0:
            pred_obs_val = jnp.vstack(preds)
        else:
            pred_obs_val = jnp.empty_like(obs.value)

        return state, InpatientObservables(obs.time, pred_obs_val, obs.mask)

    def __call__(self, admission: Admission,
                 embedded_admission: EmbeddedAdmission) -> AdmissionPrediction:
        state = self.join_state(None, None, embedded_admission.state_dx_e0)
        int_e = embedded_admission.int_e
        obs = admission.observables
        pred_obs_l = []
        t0 = admission.interventions.t0
        t1 = admission.interventions.t1
        for i in range(len(int_e)):
            t = t0[i], t1[i]
            state, pred_obs = self.step_segment(state, int_e[i], obs[i], *t)
            pred_obs_l.append(pred_obs)
        pred_dx = CodesVector(self.f_dx_dec(self.split_state(state)[2]),
                              admission.outcome.scheme)
        return AdmissionPrediction(admission=admission,
                                   outcome=pred_dx,
                                   observables=pred_obs_l)

    def batch_predict(self,
                      inpatients: Patients,
                      leave_pbar: bool = False) -> Predictions:
        total_int_days = inpatients.interval_days()

        inpatients_emb = {
            i: self.f_emb(subject)
            for i, subject in tqdm_constructor(inpatients.subjects.items(),
                                               desc="Embedding",
                                               unit='subject',
                                               leave=leave_pbar)
        }

        r_bar = '| {n:.2f}/{total:.2f} [{elapsed}<{remaining}, ' '{rate_fmt}{postfix}]'
        bar_format = '{l_bar}{bar}' + r_bar
        with tqdm_constructor(total=total_int_days,
                              bar_format=bar_format,
                              unit='odeint-days',
                              leave=leave_pbar) as pbar:
            results = Predictions()
            for i, subject_id in enumerate(inpatients.subjects.keys()):
                inpatient = inpatients.subjects[subject_id]
                embedded_admissions = inpatients_emb[subject_id]
                for adm, adm_e in zip(inpatient.admissions,
                                      embedded_admissions):
                    results.add(subject_id=subject_id,
                                prediction=self(adm, adm_e))
                    pbar.update(adm.interval_days)
                    pbar.set_description(f"Subject {i+1}/{len(inpatients)}")
            return results.filter_nans()

    @staticmethod
    def emb_dyn_partition(pytree: InICENODE):
        """
        Separate the dynamics parameters from the embedding parameters.
        Thanks to Patrick Kidger for the clever function of eqx.partition.
        """
        dyn_leaves = jtu.tree_leaves(pytree.f_dyn)
        dyn_predicate = lambda _t: any(_t is t for t in dyn_leaves)
        dyn_tree, emb_tree = eqx.partition(pytree, dyn_predicate)
        return emb_tree, dyn_tree

    @staticmethod
    def emb_dyn_merge(emb_tree, dyn_tree):
        return eqx.combine(emb_tree, dyn_tree)

    @classmethod
    def from_config(cls, conf: Dict[str, Any], scheme: MIMIC4ICUDatasetScheme,
                    demographic_vector_config: DemographicVectorConfig,
                    key: "jax.random.PRNGKey"):
        dims = InICENODEDimensions(**conf)
        return cls(dims=dims,
                   scheme=scheme,
                   demographic_vector_config=demographic_vector_config,
                   key=key)

    def load_params(self, params_file):
        """
        Load the parameters in `params_file` filepath and
        return as PyTree Object.
        """
        with open(translate_path(params_file), 'rb') as file_rsc:
            return eqx.tree_deserialise_leaves(file_rsc, self)

    def write_params(self, params_file):
        """
        Store the parameters (PyTree object) into a new file
        given by `params_file`.
        """
        with open(translate_path(params_file), 'wb') as file_rsc:
            eqx.tree_serialise_leaves(file_rsc, self)

    def load_params_from_archive(self, zipfile_fname: str, params_fname: str):

        with zipfile.ZipFile(translate_path(zipfile_fname),
                             compression=zipfile.ZIP_STORED,
                             mode="r") as archive:
            with archive.open(params_fname, "r") as zip_member:
                return eqx.tree_deserialise_leaves(zip_member, self)

    def weights(self):
        has_weight = lambda leaf: hasattr(leaf, 'weight')
        # Valid for eqx.nn.MLP and ml.base_models.GRUDynamics
        return tuple(x.weight
                     for x in jtu.tree_leaves(self, is_leaf=has_weight)
                     if has_weight(x))
