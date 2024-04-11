"""."""
from __future__ import annotations

from typing import List, Callable, Tuple, Optional

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom

from .base_models import (StateUpdate, NeuralODE_JAX)
from .model import (DischargeSummaryModel, ModelConfig, ModelRegularisation,
                    Precomputes)
from ..ehr import (Patient, DemographicVectorConfig,
                   DatasetScheme, CodesVector)
from ..utils import model_params_scaler


class ICENODEConfig(ModelConfig):
    mem: int = 15


class ICENODERegularisation(ModelRegularisation):
    L_taylor: float = 0.0
    taylor_order: int = 0


#     @classmethod
#     def sample_model_config(cls, trial: optuna.Trial):
#         return {'state_size': trial.suggest_int('s', 10, 100, 10)}


class ICENODE(DischargeSummaryModel):
    _f_dyn: Callable
    _f_update: Callable
    config: ICENODEConfig = eqx.static_field()

    def __init__(self, config: ICENODEConfig, schemes: Tuple[DatasetScheme],
                 demographic_vector_config: DemographicVectorConfig,
                 key: "jax.random.PRNGKey"):
        self._assert_demo_dim(config, schemes[1], demographic_vector_config)
        (emb_key, dx_dec_key, dyn_key, up_key) = jrandom.split(key, 4)

        f_emb = OutpatientEmbedding(
            schemes=schemes,
            demographic_vector_config=demographic_vector_config,
            config=config.emb,
            key=emb_key)
        f_dx_dec = eqx.nn.MLP(config.emb.dx,
                              len(schemes[1].outcome),
                              config.emb.dx * 5,
                              depth=1,
                              key=dx_dec_key)

        dyn_state_size = config.emb.dx + config.mem
        dyn_input_size = dyn_state_size + config.emb.demo
        f_dyn = eqx.nn.MLP(in_size=dyn_input_size,
                           out_size=dyn_state_size,
                           activation=jnn.tanh,
                           depth=2,
                           width_size=dyn_state_size * 5,
                           key=dyn_key)
        ode_dyn_f = model_params_scaler(f_dyn, 1e-3, eqx.is_inexact_array)

        self._f_dyn = NeuralODE_JAX(ode_dyn_f, timescale=1.0)

        self._f_update = StateUpdate(state_size=config.mem,
                                     embeddings_size=config.emb.dx,
                                     key=up_key)

        super().__init__(config=config, _f_emb=f_emb, _f_dx_dec=f_dx_dec)

    @property
    def dyn_params_list(self):
        return self.params_list(self._f_dyn)

    def join_state_emb(self, state, emb):
        if state is None:
            state = jnp.zeros((self.config.mem,))
        return jnp.hstack((state, emb))

    def split_state_emb(self, state: jnp.ndarray):
        return jnp.hsplit(state, (self.config.mem,))

    @eqx.filter_jit
    def _integrate(self, state, delta, ctrl):
        second = jnp.array(1 / (3600.0 * 24.0))
        delta = jnp.where((delta < second) & (delta >= 0.0), second, delta)
        return self._f_dyn(delta, state, args=dict(control=ctrl))[-1]

    @eqx.filter_jit
    def _integrate_reg(self, state, delta, ctrl, taylor_order):
        second = jnp.array(1 / (3600.0 * 24.0))
        delta = jnp.where((delta < second) & (delta >= 0.0), second, delta)
        state_v, reg_v = self._f_dyn(delta,
                                     state,
                                     args=dict(control=ctrl,
                                               tay_reg=taylor_order))
        return state_v[-1], reg_v[-1]

    @eqx.filter_jit
    def _update(self, *args):
        return self._f_update(*args)

    @eqx.filter_jit
    def _decode(self, dx_e: jnp.ndarray):
        return self._f_dx_dec(dx_e)

    @staticmethod
    def _time_diff(t1, t2):
        """
        This static method is created to simplify creating a variant of
        ICE-NODE (i.e. ICE-NODE_UNIFORM) that integrates with a
        fixed-time interval. So only this method that needs to be overriden.
        """
        return t1 - t2

    @staticmethod
    def regularisation_effect(reg: ICENODERegularisation):
        return reg is not None and reg.L_taylor > 0.0 and reg.taylor_order > 0

    def __call__(self, patient: Patient,
                 embedded_admissions: List[EmbeddedOutAdmission],
                 precomputes: Precomputes,
                 regularisation: Optional[ICENODERegularisation],
                 store_embeddings: bool):
        adms = patient.admissions
        state = self.join_state_emb(None, embedded_admissions[0].dx)
        t0_date = adms[0].admission_dates[0]
        preds = []
        for i in range(1, len(adms)):
            adm = adms[i]
            # Integrate
            # days between first admission and last discharge
            t0 = adms[i - 1].days_since(t0_date)[1]
            # days between first admission and current discharge
            t1 = adms[i].days_since(t0_date)[1]
            # days between last discharge and current discharge
            dt = self._time_diff(t1, t0)
            delta_disch2disch = jnp.array(dt)

            demo_e = embedded_admissions[i - 1].demo
            if self.regularisation_effect(regularisation):
                state, reg = self._integrate_reg(state, delta_disch2disch,
                                                 demo_e,
                                                 regularisation.taylor_order)
                reg = {'L_taylor': reg / delta_disch2disch}
            else:
                state = self._integrate(state, delta_disch2disch, demo_e)
                reg = None

            mem, dx_e_hat = self.split_state_emb(state)

            # Predict
            dx_hat = CodesVector(self._decode(dx_e_hat), adm.outcome.scheme)

            # Update
            dx_e = embedded_admissions[i].dx
            mem = self._update(mem, dx_e_hat, dx_e)
            state = self.join_state_emb(mem, dx_e)
            #
            # if store_embeddings:
            #     trajectory = PatientTrajectory(time=t1, state=state)
            #     preds.append(
            #         AdmissionPrediction(admission=adm,
            #                             outcome=dx_hat,
            #                             trajectory=trajectory,
            #                             associative_regularisation=reg))
            # else:
            #     preds.append(
            #         AdmissionPrediction(admission=adm,
            #                             outcome=dx_hat,
            #                             associative_regularisation=reg))
        return preds


class ICENODE_UNIFORM(ICENODE):

    @staticmethod
    def _time_diff(t1, t2):
        return 7.0


class ICENODE_ZERO(ICENODE_UNIFORM):

    @eqx.filter_jit
    def _integrate(self, state, int_time, ctrl):
        return state


class GRUConfig(ModelConfig):
    pass


class GRU(DischargeSummaryModel):
    _f_update: Callable
    config: GRUConfig = eqx.static_field()

    def __init__(self, config: GRUConfig, schemes: Tuple[DatasetScheme],
                 demographic_vector_config: DemographicVectorConfig,
                 key: "jax.random.PRNGKey"):
        self._assert_demo_dim(config, schemes[1], demographic_vector_config)
        (emb_key, dx_dec_key, up_key) = jrandom.split(key, 3)
        f_emb = OutpatientEmbedding(
            schemes=schemes,
            demographic_vector_config=demographic_vector_config,
            config=config.emb,
            key=emb_key)
        f_dx_dec = eqx.nn.MLP(config.emb.dx,
                              len(schemes[1].outcome),
                              config.emb.dx * 5,
                              depth=1,
                              key=dx_dec_key)

        self._f_update = eqx.nn.GRUCell(config.emb.dx + config.emb.demo,
                                        config.emb.dx,
                                        use_bias=True,
                                        key=up_key)

        super().__init__(config=config, _f_emb=f_emb, _f_dx_dec=f_dx_dec)

    @property
    def dyn_params_list(self):
        return self.params_list(self._f_update)

    @eqx.filter_jit
    def _update(self, mem: jnp.ndarray, dx_e_prev: jnp.ndarray,
                demo: jnp.ndarray):
        x = jnp.hstack((dx_e_prev, demo))
        return self._f_update(x, mem)

    @eqx.filter_jit
    def _decode(self, dx_e_hat: jnp.ndarray):
        return self._f_dx_dec(dx_e_hat)

    def __call__(self, patient: Patient,
                 embedded_admissions: List[EmbeddedOutAdmission],
                 precomputes: Precomputes, regularisation: ModelRegularisation,
                 store_embeddings: bool):
        adms = patient.admissions
        state = jnp.zeros((self.config.emb.dx,))
        preds = []
        for i in range(1, len(adms)):
            adm = adms[i]
            demo = embedded_admissions[i - 1].demo
            dx_e_prev = embedded_admissions[i - 1].dx
            # Step
            state = self._update(state, dx_e_prev, demo)
            # Predict
            dx_hat = CodesVector(self._decode(state), adm.outcome.scheme)
            #
            # if store_embeddings:
            #     tdisch = adm.days_since(adms[0].admission_dates[0])[1]
            #     trajectory = PatientTrajectory(time=tdisch, state=state)
            #     preds.append(
            #         AdmissionPrediction(admission=adm,
            #                             outcome=dx_hat,
            #                             trajectory=trajectory))
            # else:
            #     preds.append(AdmissionPrediction(admission=adm,
            #                                      outcome=dx_hat))
            #
            # preds.append(AdmissionPrediction(admission=adm, outcome=dx_hat))
        return preds


class RETAINConfig(ModelConfig):
    mem_a: int = eqx.static_field(default=45)
    mem_b: int = eqx.static_field(default=45)

    # @staticmethod
    # def sample_model_config(trial: optuna.Trial):
    #     sa = trial.suggest_int('sa', 100, 350, 50)
    #     sb = trial.suggest_int('sb', 100, 350, 50)
    #     return {'state_size': (sa, sb)}


class RETAIN(DischargeSummaryModel):
    _f_gru_a: Callable
    _f_gru_b: Callable
    _f_att_a: Callable
    _f_att_b: Callable
    config: RETAINConfig = eqx.static_field()

    def __init__(self, config: RETAINConfig, schemes: Tuple[DatasetScheme],
                 demographic_vector_config: DemographicVectorConfig,
                 key: "jax.random.PRNGKey"):
        self._assert_demo_dim(config, schemes[1], demographic_vector_config)
        k1, k2, k3, k4, k5, k6 = jrandom.split(key, 6)

        f_emb = OutpatientEmbedding(
            schemes=schemes,
            demographic_vector_config=demographic_vector_config,
            config=config.emb,
            key=k1)
        f_dx_dec = eqx.nn.MLP(config.emb.dx,
                              len(schemes[1].outcome),
                              config.emb.dx * 5,
                              depth=1,
                              key=k2)
        self._f_gru_a = eqx.nn.GRUCell(config.emb.dx + config.emb.demo,
                                       config.mem_a,
                                       use_bias=True,
                                       key=k3)
        self._f_gru_b = eqx.nn.GRUCell(config.emb.dx + config.emb.demo,
                                       config.mem_b,
                                       use_bias=True,
                                       key=k4)

        self._f_att_a = eqx.nn.Linear(config.mem_a, 1, use_bias=True, key=k5)
        self._f_att_b = eqx.nn.Linear(config.mem_b,
                                      config.emb.dx,
                                      use_bias=True,
                                      key=k6)

        super().__init__(config=config, _f_emb=f_emb, _f_dx_dec=f_dx_dec)

    @property
    def dyn_params_list(self):
        return self.params_list(
            (self._f_gru_a, self._f_gru_b, self._f_att_a, self._f_att_b))

    @eqx.filter_jit
    def _gru_a(self, x, state):
        return self._f_gru_a(x, state)

    @eqx.filter_jit
    def _gru_b(self, x, state):
        return self._f_gru_b(x, state)

    @eqx.filter_jit
    def _att_a(self, x):
        return self._f_att_a(x)

    @eqx.filter_jit
    def _att_b(self, x):
        return self._f_att_b(x)

    @eqx.filter_jit
    def _dx_dec(self, x):
        return self._f_dx_dec(x)

    def __call__(self, patient: Patient,
                 embedded_admissions: List[EmbeddedOutAdmission],
                 precomputes: Precomputes, regularisation: ModelRegularisation,
                 store_embeddings: bool):
        adms = patient.admissions
        state_a0 = jnp.zeros(self.config.mem_a)
        state_b0 = jnp.zeros(self.config.mem_b)
        preds = []

        # step 1 @RETAIN paper

        # v1, v2, ..., vT
        # Merge controls with embeddings
        cv_seq = [
            jnp.hstack([adm.demo, adm.dx]) for adm in embedded_admissions
        ]

        hsplit_idx = self.config.emb.demo

        for i in range(1, len(adms)):
            # e: i, ..., 1
            e_seq = []

            # beta: i, ..., 1
            b_seq = []

            state_a = state_a0
            state_b = state_b0
            for j in reversed(range(i)):
                # step 2 @RETAIN paper
                state_a = self._gru_a(cv_seq[j], state_a)
                e_j = self._att_a(state_a)
                # After the for-loop apply softmax on e_seq to get
                # alpha_seq

                e_seq.append(e_j)

                # step 3 @RETAIN paper
                h_j = state_b = self._gru_b(cv_seq[j], state_b)
                b_j = self._att_b(h_j)

                b_seq.append(jnp.tanh(b_j))

            # alpha: i, ..., 1
            a_seq = jax.nn.softmax(jnp.hstack(e_seq))

            # step 4 @RETAIN paper

            # v_i, ..., v_1
            v_context = cv_seq[:i][::-1]
            v_context = [jnp.hsplit(v, [hsplit_idx])[1] for v in v_context]
            c_context = sum(a * (b * v)
                            for a, b, v in zip(a_seq, b_seq, v_context))

            # step 5 @RETAIN paper
            logits = CodesVector(self._dx_dec(c_context),
                                 adms[i].outcome.scheme)

            # if store_embeddings:
            #     tdisch = adms[i].days_since(adms[0].admission_dates[0])[1]
            #     trajectory = PatientTrajectory(time=tdisch, state=c_context)
            #     preds.append(
            #         AdmissionPrediction(admission=adms[i],
            #                             outcome=logits,
            #                             trajectory=trajectory))
            # else:
            #     preds.append(
            #         AdmissionPrediction(admission=adms[i], outcome=logits))

        return preds
