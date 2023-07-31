"""."""
from __future__ import annotations
from functools import partial
from collections import namedtuple
from datetime import date
from typing import (Any, Dict, List, TYPE_CHECKING, Callable, Union, Tuple,
                    Optional)
import re

from absl import logging
import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jrandom
import jax.tree_util as jtu
import equinox as eqx
import optuna

from ..utils import model_params_scaler
from ..ehr import (Inpatients, Inpatient, InpatientAdmission,
                   InpatientStaticInfo, BatchPredictedRisks,
                   MIMIC4ICUDatasetScheme, AggregateRepresentation)

from .base_models import (GRUDynamics, NeuralODE, ObsStateUpdate)
from abc import ABC, abstractmethod, ABCMeta
import zipfile

from ..utils import translate_path


class EmbeddedAdmission(eqx.Module):
    state_dx_e0: jnp.ndarray
    int_e: List[jnp.ndarray]


class InICENODEDimensions(eqx.Module):
    state_m: int = 15
    state_dx_e: int = 30
    state_obs_e: int = 25
    input_e: int = 10
    proc_e: int = 10
    demo_e: int = 5
    int_e: int = 15


class InpatientEmbedding(eqx.Module):
    f_dx_emb: Callable  # TODO: state_e(0) = f_dxemb(dx_history)
    f_dem_emb: Callable  # TODO: two layers embeddings for demographics. d=5
    f_inp_agg: Callable  # TODO: aggregator (learnable for weighted sum nns)
    f_inp_emb: Callable  # TODO: int_e = f_inpemb(input_e) d=10
    f_proc_emb: Callable  # TODO: two layers embeddings for procedures. d=10
    f_int_emb: Callable  # TODO: two layers embeddings f_int_emb(inp_e, proc_e, demo_e). d=15

    def __init__(self, scheme: MIMIC4ICUDatasetScheme,
                 dims: InICENODEDimensions, prng_key: "jax.random.PRNGKey"):
        super().__init__()
        (dx_emb_key, inp_agg_key, inp_emb_key, proc_emb_key, dem_emb_key,
         int_emb_key) = jrandom.split(prng_key, 6)

        self.f_dx_emb = eqx.nn.MLP(len(scheme.dx_target),
                                   dims.state_dx_e,
                                   dims.state_dx_e * 5,
                                   depth=1,
                                   key=dx_emb_key)
        self.f_inp_agg = AggregateRepresentation(scheme.int_input_source,
                                                 scheme.int_input_target)
        self.f_inp_emb = eqx.nn.MLP(len(scheme.int_input_target),
                                    dims.input_e,
                                    dims.input_e * 5,
                                    depth=1,
                                    key=inp_emb_key)
        self.f_proc_emb = eqx.nn.MLP(len(scheme.proc_target),
                                     dims.proc_e,
                                     dims.proc_e * 5,
                                     depth=1,
                                     key=proc_emb_key)

        # demo-dims: age(1) + gender(2) + target_ethnicity.
        self.f_dem_emb = eqx.nn.MLP(1 + 2 + len(scheme.eth_target),
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
    def embed_segment(self, inp: jnp.ndarray, proc: jnp.ndarray,
                      demo: jnp.ndarray) -> jnp.ndarray:
        inp_emb = self.f_inp_emb(inp)
        proc_emb = self.f_proc_emb(proc)
        demo_emb = self.f_dem_emb(demo)
        return self.f_int_emb(jnp.hstack([inp_emb, proc_emb, demo_emb]))

    @eqx.filter_jit
    def embed_demo(self, static_info: InpatientStaticInfo,
                   admission_date: date) -> jnp.ndarray:
        demo_vec = static_info.demographic_vector(admission_date)
        return self.f_dem_emb(demo_vec)

    @eqx.filter_jit
    def embed_dx(self, admission: InpatientAdmission) -> jnp.ndarray:
        return self.f_dx_emb(admission.dx_history.vec)

    def embed_admission(self, static_info: InpatientStaticInfo,
                        admission: InpatientAdmission) -> EmbeddedAdmission:
        demo_emb = self.embed_demo(static_info, admission.admission_dates[0])
        dx_emb = self.f_dx_emb(admission)
        int_ = admission.interventions.segment_input(self.f_inp_agg)
        int_inp = int_.segmented_input
        int_proc = int_.segmented_proc
        int_e = [
            self.embed_segment(inp, proc, demo_emb)
            for inp, proc in zip(int_inp, int_proc)
        ]
        return EmbeddedAdmission(state_dx_e0=dx_emb, int_e=int_e)

    def __call__(self, inpatient: Inpatient) -> List[EmbeddedAdmission]:
        return [
            self.embed_admission(inpatient.static_info, admission)
            for admission in inpatient.admissions
        ]


class InICENODE(eqx.Module, metaclass=ABCMeta):
    f_emb: Callable[[
        InpatientAdmission
    ], EmbeddedAdmission]  # TODO: two layers int_e = f_int(input_e, proc_e)
    f_obs_dec: Callable  # TODO: predictor f_obsdec(state) // loss = l2
    f_dx_dec: Callable  # TODO: predictor f_dec(state) // lo
    f_dyn: Callable
    f_update: Callable

    scheme: MIMIC4ICUDatasetScheme = eqx.static_field()
    dims: InICENODEDimensions = eqx.static_field()

    def __init__(self, scheme: MIMIC4ICUDatasetScheme,
                 dims: InICENODEDimensions, prng_key: "jax.random.PRNGKey"):
        super().__init__()
        self.dims = dims
        self.scheme = scheme
        (emb_key, obs_dec_key, dx_dec_key, dyn_key,
         update_key) = jrandom.split(prng_key, 5)
        self.f_emb = InpatientEmbedding(scheme, dims, emb_key)
        self.f_obs_dec = eqx.nn.MLP(dims.state_obs_e,
                                    len(scheme.obs),
                                    dims.state_obs_e * 5,
                                    depth=1,
                                    key=obs_dec_key)
        self.f_dx_dec = eqx.nn.MLP(dims.state_dx_e,
                                   len(scheme.dx_target),
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
        f_dyn = model_params_scaler(f_dyn, 1e-5, eqx.is_inexact_array)
        self.f_dyn = NeuralODE(f_dyn, timescale=24.0)
        self.f_update = ObsStateUpdate(dyn_state_size,
                                       len(scheme.obs),
                                       key=update_key)

    def join_state(self, mem, obs, dx):
        if mem is None:
            mem = jnp.zeros((self.dims.state_m, ))
        return jnp.stack((mem, obs, dx), axis=-1)

    def split_state_emb(self, state: jnp.ndarray):
        return jnp.split(state, (self.dims.state_m, self.dims.state_obs_e),
                         axis=-1)

    def __call__(self, admission: InpatientAdmission,
                 embedded_admission: EmbeddedAdmission):
        pass

    def batch_predict(self, inpatients: Inpatients, subjects_batch: List[str],
                      args):
        results = BatchPredictedRisks()
        inpatients_emb = {i: self.f_emb(inpatients[i]) for i in subjects_batch}
        for i in subjects_batch:
            inpatient = inpatients[i]
            embedded_admissions = inpatients_emb[i]
            for adm, adm_e in zip(inpatient.admissions, embedded_admissions):
                results.add(subject_id=i,
                            admission=adm,
                            prediction=self(adm, adm_e))
        return results

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
    def from_config(cls, conf: Dict[str, Any], inpatients: Inpatients,
                    train_split: List[int], key: "jax.random.PRNGKey"):
        # decoder_input_size = cls.decoder_input_size(conf)
        # emb_models = embeddings_from_conf(conf["emb"], subject_interface,
        # train_split, decoder_input_size)
        # control_size = subject_interface.control_dim
        # return cls(**emb_models,
        # **conf["model"],
        # outcome=subject_interface.outcome_extractor,
        # control_size=control_size,
        # key=key)
        pass

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

    @eqx.filter_jit
    def _integrate_state(self, subject_state: SubjectState,
                         int_time: jnp.ndarray, ctrl: jnp.ndarray,
                         args: Dict[str, Any]):
        args = dict(control=ctrl, **args)
        out = self.ode_dyn(int_time, subject_state.state, args)
        if args.get('tay_reg', 0) > 0:
            state, r, traj = out[0][-1], out[1][-1].squeeze(), out[0]
        else:
            state, r, traj = out[-1], 0.0, out

        return SubjectState(state, subject_state.time + int_time), r, traj

    def _update_state(self, subject_state: SubjectState,
                      adm_info: Admission_JAX,
                      dx_emb: jnp.ndarray) -> jnp.ndarray:
        state, dx_emb_hat = self.split_state_emb(subject_state)
        state = self._jitted_update(state, dx_emb_hat, dx_emb)
        return SubjectState(self.join_state_emb(state, dx_emb),
                            jnp.array(adm_info.admission_time + adm_info.los))

    def __call__(self,
                 subject_interface: Subject_JAX,
                 subjects_batch: List[int],
                 args=dict()):
        dx_for_emb = subject_interface.dx_batch_history_vec(subjects_batch)
        dx_G = self.dx_emb.compute_embeddings_mat(dx_for_emb)
        risk_prediction = self.init_predictions()
        dyn_loss = []
        max_adms = max(
            len(subject_interface[subj_i]) for subj_i in subjects_batch)
        logging.info(f'max_adms: {max_adms}')
        for subj_i in subjects_batch:
            adms = subject_interface[subj_i]
            emb_seq = self.dx_embed(dx_G, adms)
            ctrl_seq = [
                subject_interface.subject_control(subj_i, adm.admission_date)
                for adm in adms
            ]
            subject_state = SubjectState(
                self.join_state_emb(None, emb_seq[0]),
                jnp.array(adms[0].admission_time + adms[0].los))

            for adm, emb, ctrl in zip(adms[1:], emb_seq[1:], ctrl_seq[:-1]):

                # Discharge-to-discharge time.
                d2d_time = jnp.array(
                    self._time_diff(adm.admission_time + adm.los,
                                    subject_state.time))

                # Integrate until next discharge
                subject_state, r, traj = self._integrate_state(
                    subject_state, d2d_time, ctrl, args)
                dyn_loss.append(r)

                dec_dx = self._decode(subject_state)

                risk_prediction.add(subject_id=subj_i,
                                    admission=adm,
                                    prediction=dec_dx,
                                    trajectory=traj)

                if args.get('return_embeddings', False):
                    risk_prediction.set_subject_embeddings(
                        subject_id=subj_i, embeddings=subject_state.state)

                # Update state at discharge
                subject_state = self._update_state(subject_state, adm, emb)

        return {'predictions': risk_prediction, 'dyn_loss': sum(dyn_loss)}

    @classmethod
    def sample_model_config(cls, trial: optuna.Trial):
        return {
            'ode_dyn':
            trial.suggest_categorical('ode_dyn',
                                      ['mlp1', 'mlp2', 'mlp3', 'gru']),
            'ode_init_var':
            trial.suggest_float('ode_i', 1e-8, 1e1, log=True),
            'state_size':
            trial.suggest_int('s', 10, 100, 10),
            'timescale':
            7
        }
