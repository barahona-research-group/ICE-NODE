"""."""
from __future__ import annotations
from functools import partial
from collections import namedtuple
from typing import (Any, Dict, List, TYPE_CHECKING, Callable, Union, Tuple,
                    Optional)
import re

from absl import logging
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import equinox as eqx
import optuna

from ..utils import model_params_scaler
from ..ehr import (Inpatients, Inpatient, InpatientAdmission,
                   BatchPredictedRisks)

from .base_models import (GRUDynamics, NeuralODE, StateUpdate)
from abc import ABC, abstractmethod, ABCMeta
import zipfile

from ..utils import translate_path


class EmbeddedAdmission(eqx.Module):
    state_e0: jnp.ndarray
    int_e: List[jnp.ndarray]


class AdmissionEmbedding(eqx.Module):
    f_dx_emb: Callable  # TODO: state_e(0) = f_dxemb(dx_history)
    f_inp_agg: Callable  # TODO: aggregator (learnable for weighted sum nns)
    f_inp_emb: Callable  # TODO: int_e = f_inpemb(input_e) d=10
    f_proc_emb: Callable  # TODO: two layers embeddings for procedures. d=10
    f_dem_emb: Callable  # TODO: two layers embeddings for demographics. d=5
    f_int_emb: Callable  # TODO: two layers embeddings f_int_emb(inp_e, proc_e, demo_e). d=15


class InICENODE(eqx.Module, metaclass=ABCMeta):
    f_emb: Callable[[
        InpatientAdmission
    ], EmbeddedAdmission]  # TODO: two layers int_e = f_int(input_e, proc_e)
    f_obs_dec: Callable  # TODO: predictor f_obsdec(state) // loss = l2
    f_dx_dec: Callable  # TODO: predictor f_dec(state) // lo
    f_dyn: Callable
    f_update: Callable

    state_size: int
    control_size: int

    @abstractmethod
    def __call__(self, inpatients: Inpatients, subjects_batch: List[str],
                 args):
        pass

    def subject_embeddings(self, subject_interface: Inpatients,
                           batch: List[str]):
        out = self(subject_interface, batch, dict(return_embeddings=True))
        return {i: out['predictions'].get_subject_embeddings(i) for i in batch}

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
        decoder_input_size = cls.decoder_input_size(conf)
        emb_models = embeddings_from_conf(conf["emb"], subject_interface,
                                          train_split, decoder_input_size)
        control_size = subject_interface.control_dim
        return cls(**emb_models,
                   **conf["model"],
                   outcome=subject_interface.outcome_extractor,
                   control_size=control_size,
                   key=key)

    def load_params(self, params_file):
        """
        Load the parameters in `params_file` filepath and return as PyTree Object.
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


def ode_dyn(label, state_size, embeddings_size, control_size, key):
    if 'mlp' in label:
        nlayers = int(re.findall(r'\d+\b', label)[0])
        label = 'mlp'
    else:
        nlayers = 0

    dyn_state_size = state_size + embeddings_size
    input_size = state_size + embeddings_size + control_size

    nn_kwargs = {
        'mlp':
        dict(activation=jax.nn.tanh,
             final_activation=jax.nn.tanh,
             depth=nlayers - 1,
             width_size=input_size,
             in_size=input_size,
             out_size=dyn_state_size),
        'gru':
        dict(input_size=input_size, state_size=dyn_state_size)
    }

    if label == 'mlp':
        return eqx.nn.MLP(**nn_kwargs['mlp'], key=key)
    if label == 'gru':
        return GRUDynamics(**nn_kwargs['gru'], key=key)

    raise RuntimeError(f'Unexpected dynamics label: {label}')


class SubjectState(eqx.Module):
    state: jnp.ndarray
    time: jnp.ndarray  # shape ()


class ICENODE(AbstractModel):
    ode_dyn: eqx.Module
    f_update: eqx.Module

    ode_init_var: float = eqx.static_field()
    timescale: float = eqx.static_field()

    @staticmethod
    def decoder_input_size(expt_config):
        return expt_config["emb"]["dx"]["embeddings_size"]

    @property
    def dyn_state_size(self):
        return self.state_size + self.dx_emb.embeddings_size

    def __init__(self, ode_dyn_label: str, ode_init_var: float,
                 timescale: float, key: "jax.random.PRNGKey", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timescale = timescale
        self.ode_init_var = ode_init_var
        key1, key2 = jrandom.split(key, 2)
        ode_dyn_f = ode_dyn(ode_dyn_label,
                            state_size=self.state_size,
                            embeddings_size=self.dx_emb.embeddings_size,
                            control_size=self.control_size,
                            key=key1)
        ode_dyn_f = model_params_scaler(ode_dyn_f, ode_init_var,
                                        eqx.is_inexact_array)

        self.ode_dyn = NeuralODE_JAX(ode_dyn_f, timescale=timescale)

        self.f_update = StateUpdate(
            state_size=self.state_size,
            embeddings_size=self.dx_emb.embeddings_size,
            key=key2)

    def weights(self):
        has_weight = lambda leaf: hasattr(leaf, 'weight')
        # Valid for eqx.nn.MLP and ml.base_models.GRUDynamics
        return tuple(x.weight
                     for x in jtu.tree_leaves(self, is_leaf=has_weight)
                     if has_weight(x))

    def join_state_emb(self, state, emb):
        if state is None:
            state = jnp.zeros((self.state_size, ))
        return jnp.hstack((state, emb))

    def split_state_emb(self, state: SubjectState):
        return jnp.split(state.state, (self.state_size, ), axis=-1)

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

    @eqx.filter_jit
    def _jitted_update(self, *args):
        return self.f_update(*args)

    def _update_state(self, subject_state: SubjectState,
                      adm_info: Admission_JAX,
                      dx_emb: jnp.ndarray) -> jnp.ndarray:
        state, dx_emb_hat = self.split_state_emb(subject_state)
        state = self._jitted_update(state, dx_emb_hat, dx_emb)
        return SubjectState(self.join_state_emb(state, dx_emb),
                            jnp.array(adm_info.admission_time + adm_info.los))

    @eqx.filter_jit
    def _decode(self, subject_state: SubjectState):
        _, dx_emb_hat = self.split_state_emb(subject_state)
        return self.dx_dec(dx_emb_hat)

    @eqx.filter_jit
    def _decode_trajectory(self, traj: SubjectState):
        _, dx_emb_hat = self.split_state_emb(traj)
        return jax.vmap(self.dx_dec)(dx_emb_hat)

    @staticmethod
    def _time_diff(t1, t2):
        """
        This static method is created to simplify creating a variant of
        ICE-NODE (i.e. ICE-NODE_UNIFORM) that integrates with a
        fixed-time interval. So only this method that needs to be overriden.
        """
        return t1 - t2

    def init_predictions(self):
        return BatchPredictedRisks()

    def dx_embed(self, dx_G: jnp.ndarray, adms: List[Admission_JAX]):
        return [self.dx_emb.encode(dx_G, adm.dx_vec) for adm in adms]

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


class ICENODE_UNIFORM(ICENODE):
    @staticmethod
    def _time_diff(t1, t2):
        return 7.0


class ICENODE_ZERO(ICENODE_UNIFORM):
    @eqx.filter_jit
    def _integrate_state(self, subject_state, int_time, ctrl, args):
        return SubjectState(subject_state.state,
                            subject_state.time + int_time), 0.0, None


class AICE(ICENODE):
    in_mix: jnp.ndarray
    out_mix: jnp.ndarray

    @staticmethod
    def decoder_input_size(expt_config):
        return expt_config["emb"]["dx"]["embeddings_size"]

    @property
    def dyn_state_size(self):
        return self.state_size + self.dx_emb.embeddings_size

    def __init__(self, key: "jax.random.PRNGKey", *args, **kwargs):
        key1, key2, key3 = jax.random.split(key, 3)
        super().__init__(*args, **kwargs, key=key1)

        self.out_mix = jnp.zeros((self.dx_dec.output_size, ),
                                 dtype=jnp.float32)
        self.in_mix = jnp.zeros((self.dx_emb.input_size, ), dtype=jnp.float32)
