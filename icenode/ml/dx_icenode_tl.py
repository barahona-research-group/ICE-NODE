"""."""
from __future__ import annotations
from functools import partial
from collections import namedtuple
from typing import Any, Dict, List
import re

import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import optuna

from ..metric import balanced_focal_bce, BatchPredictedRisks
from ..utils import model_params_scaler
from ..ehr import Subject_JAX, Admission_JAX

from .base_models import (GRUDynamics, NeuralODE, StateUpdate)
from .abstract_trainer import AbstractTrainer
from .abstract_model import AbstractModel


def ode_dyn(label, state_size, embeddings_size, control_size, key):
    if 'mlp' in label:
        depth = int(re.findall(r'\d+\b', label)[0])
        label = 'mlp'
    else:
        depth = 0

    dyn_state_size = state_size + embeddings_size
    input_size = state_size + embeddings_size + control_size

    nn_kwargs = {
        'mlp':
        dict(activation=jax.nn.tanh,
             depth=depth,
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


SubjectState = namedtuple('SubjectState', ['state', 'time'])


class ICENODE(AbstractModel):
    ode_dyn: eqx.Module
    f_update: eqx.Module

    ode_init_var: float = eqx.static_field()
    timescale: float = eqx.static_field()

    def __init__(self, ode_dyn_label: str, ode_init_var: float,
                 timescale: float, key: "jax.random.PRNGKey", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timescale = timescale
        key1, key2 = jrandom.split(key, 2)
        ode_dyn_f = ode_dyn(ode_dyn_label,
                            state_size=self.state_size,
                            embeddings_size=self.dx_emb.embeddings_size,
                            control_size=self.control_size,
                            key=key1)
        ode_dyn_f = model_params_scaler(ode_dyn_f, ode_init_var,
                                        eqx.is_inexact_array)

        self.ode_dyn = NeuralODE(ode_dyn_f, timescale=timescale)

        self.f_update = StateUpdate(
            state_size=self.state_size,
            embeddings_size=self.dx_emb.embeddings_size,
            key=key2)

    def join_state_emb(self, state, emb):
        if state is None:
            state = jnp.zeros((self.state_size, ))
        return jnp.hstack((state, emb))

    def split_state_emb(self, state: SubjectState):
        return jnp.split(state.state, (self.state_size, ), axis=-1)

    @eqx.filter_jit
    def _integrate_state(self, subject_state: SubjectState, int_time: float,
                         ctrl: jnp.ndarray, args: Dict[str, Any]):
        args = dict(control=ctrl, n_samples=2, **args)
        out = self.ode_dyn(subject_state.state, int_time, args)
        if args.get('tay_reg', 0) > 0:
            state, r, traj = out[0][-1], out[1][-1], out[0]
        else:
            state, r, traj = out[-1], 0.0, out

        if args.get('sampling_rate', None) is None:
            traj = None

        return SubjectState(state, subject_state.time + int_time), r, traj

    @eqx.filter_jit
    def _update_state(self, subject_state: SubjectState,
                      adm_info: Admission_JAX,
                      dx_emb: jnp.ndarray) -> jnp.ndarray:
        state, dx_emb_hat = self.split_state_emb(subject_state)
        state = self.f_update(state, dx_emb_hat, dx_emb)
        return SubjectState(self.join_state_emb(state, dx_emb),
                            adm_info.admission_time + adm_info.los)

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

    def __call__(self,
                 subject_interface: Subject_JAX,
                 subjects_batch: List[int],
                 args=dict()):

        dx_G = self.dx_emb.compute_embeddings_mat()
        f_emb = partial(self.dx_emb, dx_G)
        risk_prediction = BatchPredictedRisks()
        dyn_loss = []

        for subj_i in subjects_batch:
            adms = subject_interface[subj_i]
            emb_seq = [f_emb(adm.dx_vec) for adm in adms]
            ctrl_seq = [
                subject_interface.subject_control(subj_i, adm.admission_date)
                for adm in adms
            ]
            subject_state = SubjectState(self.join_state_emb(None, emb_seq[0]),
                                         adms[0].admission_time + adms[0].los)

            for adm, emb, ctrl in zip(adms[1:], emb_seq[1:], ctrl_seq[1:]):

                # Discharge-to-discharge time.
                d2d_time = self._time_diff(adm.time + adm.los,
                                           subject_state.time)

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

        return {'predictions': risk_prediction, 'dyn_loss': dyn_loss}

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


class Trainer(AbstractTrainer):
    tay_reg: int = 3

    @classmethod
    def odeint_time(cls, predictions: BatchPredictedRisks):
        int_time = 0
        for subj_id, preds in predictions.items():
            adms = [preds[idx] for idx in sorted(preds)]
            # Integration time from the first discharge (adm[0].(length of
            # stay)) to last discarge (adm[-1].time + adm[-1].(length of stay)
            int_time += adms[-1].time + adms[-1].los - adms[0].los
        return int_time

    @classmethod
    def dx_loss(y: jnp.ndarray, dx_logits: jnp.ndarray):
        return balanced_focal_bce(y, dx_logits)

    def reg_loss(self, model: AbstractModel, subject_interface: Subject_JAX,
                 batch: List[int]):
        args = dict(tay_reg=self.tay_reg)
        res = model(subject_interface, batch, args)
        preds = res['predictions']
        l = preds.prediction_loss(self.dx_loss)

        integration_weeks = self.odeint_time(preds) / 7
        l1_loss = model.l1()
        l2_loss = model.l2()
        dyn_loss = res['dyn_loss'] / integration_weeks
        l1_alpha = self.reg_hyperparams['L_l1']
        l2_alpha = self.reg_hyperparams['L_l2']
        dyn_alpha = self.reg_hyperparams['L_dyn']

        loss = l + (l1_alpha * l1_loss) + (l2_alpha * l2_loss) + (dyn_alpha *
                                                                  dyn_loss)

        return loss, ({
            'dx_loss': l,
            'loss': loss,
            'l1_loss': l1_loss,
            'l2_loss': l2_loss,
        }, preds)

    @classmethod
    def sample_reg_hyperparams(cls, trial: optuna.Trial):
        return {
            'L_l1': 0,  #trial.suggest_float('l1', 1e-8, 5e-3, log=True),
            'L_l2': 0,  # trial.suggest_float('l2', 1e-8, 5e-3, log=True),
            'L_dyn': 1e3  # trial.suggest_float('L_dyn', 1e-6, 1, log=True)
        }

    @classmethod
    def sample_training_config(cls, trial: optuna.Trial):
        return {
            'epochs': 60,
            'batch_size': 2**trial.suggest_int('Bexp', 1, 8),
            #trial.suggest_int('B', 2, 27, 5),
            'optimizer': 'adam',
            #trial.suggest_categorical('opt', ['adam', 'adamax']),
            'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
            'decay_rate': trial.suggest_float('dr', 1e-1, 9e-1),
            'reg_hyperparams': cls.sample_reg_hyperparams(trial)
        }


Trainer.register_trainer(ICENODE)
