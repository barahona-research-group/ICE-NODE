"""."""
from __future__ import annotations
from functools import partial
from collections import namedtuple
from typing import Any, Dict, List
import re

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import equinox as eqx
import optuna

from ..utils import model_params_scaler
from ..ehr import Subject_JAX, Admission_JAX, BatchPredictedRisks

from .base_models import (GRUDynamics, NeuralODE, StateUpdate, NeuralODE_JAX)
from .abstract_model import AbstractModel


class SubjectState(eqx.Module):
    state: jnp.ndarray
    time: jnp.ndarray  # shape ()


class NJODE(AbstractModel):
    ode_dyn: eqx.Module
    f_update: eqx.Module

    ode_init_var: float = eqx.static_field()
    timescale: float = eqx.static_field()

    @property
    def dyn_state_size(self):
        return self.state_size + self.dx_emb.embeddings_size

    def __init__(self, ode_nlayers: str, ode_init_var: float, timescale: float,
                 key: "jax.random.PRNGKey", *args, **kwargs):
        embeddings_size = kwargs['dx_emb'].embeddings_size
        kwargs['state_size'] = embeddings_size
        super().__init__(*args, **kwargs)
        self.timescale = timescale
        self.ode_init_var = ode_init_var

        dyn_state_size = embeddings_size
        input_size = dyn_state_size + self.control_size + kwargs[
            'dx_emb'].input_size + 1

        key1, key2 = jrandom.split(key, 2)
        ode_dyn_f = eqx.nn.MLP(activation=jax.nn.tanh,
                               final_activation=jax.nn.tanh,
                               depth=ode_nlayers - 1,
                               width_size=input_size,
                               in_size=input_size,
                               out_size=dyn_state_size,
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

    @eqx.filter_jit
    def _integrate_state(self, subject_state: SubjectState,
                         int_time: jnp.ndarray, ctrl: jnp.ndarray,
                         args: Dict[str, Any]):
        args = dict(control=ctrl, **args)
        args['sampling_rate'] = args.get('sampling_rate', self.timestep)
        out = self.ode_dyn(int_time, subject_state.state, args)
        if args.get('tay_reg', 0) > 0:
            state, r, traj = out[0][-1], out[1][-1].squeeze(), out[0]
        else:
            state, r, traj = out[-1], 0.0, out

        return SubjectState(state, subject_state.time + int_time), r, traj

    @staticmethod
    def _time_diff(t1, t2):
        """
        This static method is created to simplify creating a variant of
        ICE-NODE (i.e. ICE-NODE_UNIFORM) that integrates with a
        fixed-time interval. So only this method that needs to be overriden.
        """
        return t1 - t2

    def __call__(self,
                 subject_interface: Subject_JAX,
                 subjects_batch: List[int],
                 args=dict()):
        dx_for_emb = subject_interface.dx_batch_history_vec(subjects_batch)
        dx_G = self.dx_emb.compute_embeddings_mat(dx_for_emb)
        f_emb = partial(self.dx_emb.encode, dx_G)
        risk_prediction = BatchPredictedRisks()
        dyn_loss = []

        for subj_i in subjects_batch:
            adms = subject_interface[subj_i]
            emb_seq = [f_emb(adm.dx_vec) for adm in adms]
            static_seq = [
                subject_interface.subject_control(subj_i, adm.admission_date)
                for adm in adms
            ]
            subject_state = SubjectState(
                emb_seq[0], jnp.array(adms[0].admission_time + adms[0].los))

            for prev_adm, adm, emb, static in zip(adms[:-1], adms[1:],
                                                  emb_seq[1:],
                                                  static_seq[:-1]):
                last_disch_time = jnp.array(prev_adm.admission_time +
                                            prev_adm.los)
                current_disch_time = jnp.array(adm.admission_time + adm.los)

                # Discharge-to-discharge time.
                integration_time = self._time_diff(current_disch_time,
                                                   last_disch_time)

                # ODE control content: static features, previous dx, time.
                ctrl = jnp.hstack(
                    (static, prev_adm.dx_vec, subject_state.time))

                # Integrate until next discharge.
                subject_state, r, traj = self._integrate_state(
                    subject_state, integration_time, ctrl, args)
                dyn_loss.append(r)

                dec_dx = self.dx_dec(subject_state.state)
                risk_prediction.add(subject_id=subj_i,
                                    admission=adm,
                                    prediction=dec_dx,
                                    trajectory=traj)
                # NJ Loss:
                # xi = adm.dx_vec
                # yi = self.dx_dec(subject_state.state)
                # yi' = self.dx_dec(emb)
                if args.get('return_embeddings', False):
                    risk_prediction.set_subject_embeddings(
                        subject_id=subj_i, embeddings=subject_state.state)


                # Jump
                subject_state = SubjectState(emb, current_disch_time)
        return {'predictions': risk_prediction, 'dyn_loss': sum(dyn_loss)}

    @classmethod
    def sample_model_config(cls, trial: optuna.Trial):
        return {
            'ode_dyn':
            trial.suggest_categorical('ode_dyn',
                                      ['mlp1', 'mlp2', 'mlp3', 'gru']),
            'ode_init_var':
            trial.suggest_float('ode_i', 1e-8, 1e1, log=True),
            'timescale':
            7
        }
