from functools import partial
from typing import (Any, Callable, Dict, Iterable, List, Optional, Tuple, Set)

from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp

import optuna

from .metrics import (l2_squared, l1_absolute, softmax_logits_bce)
from .utils import wrap_module
from .jax_interface import (DiagnosisJAXInterface, create_patient_interface)
from .models import (MLPDynamics, ResDynamics, GRUDynamics, NeuralODE,
                     EmbeddingsDecoder_Logits, StateUpdate)
from .abstract_model import AbstractModel
from .gram import AbstractEmbeddingsLayer


class ICENODE(AbstractModel):

    def __init__(self, subject_interface: DiagnosisJAXInterface,
                 diag_emb: AbstractEmbeddingsLayer, ode_dyn: str,
                 ode_with_bias: bool, ode_init_var: float, state_size: int):

        self.subject_interface = subject_interface
        self.diag_emb = diag_emb
        self.dimensions = {
            'diag_emb': diag_emb.embeddings_dim,
            'diag_out': len(subject_interface.diag_ccs_idx),
            'state': state_size
        }
        if ode_dyn == 'gru':
            ode_dyn_cls = GRUDynamics
        elif ode_dyn == 'res':
            ode_dyn_cls = ResDynamics
        elif ode_dyn == 'mlp':
            ode_dyn_cls = MLPDynamics
        else:
            raise RuntimeError(f"Unrecognized dynamics class: {ode_dyn}")
        state_emb_size = self.dimensions['diag_emb'] + state_size

        f_n_ode_init, f_n_ode = hk.without_apply_rng(
            hk.transform(
                wrap_module(NeuralODE,
                            ode_dyn_cls=ode_dyn_cls,
                            state_size=state_emb_size,
                            depth=3,
                            timescale=1,
                            with_bias=ode_with_bias,
                            init_var=ode_init_var,
                            name='f_n_ode',
                            tay_reg=3)))
        f_n_ode = jax.jit(f_n_ode, static_argnums=(1, 2))
        self.f_n_ode = (
            lambda params, *args: f_n_ode(params['f_n_ode'], *args))

        f_update_init, f_update = hk.without_apply_rng(
            hk.transform(
                wrap_module(StateUpdate,
                            state_size=state_size,
                            embeddings_size=self.dimensions['diag_emb'],
                            name='f_update')))
        f_update = jax.jit(f_update)
        self.f_update = (
            lambda params, *args: f_update(params['f_update'], *args))

        f_dec_init, f_dec = hk.without_apply_rng(
            hk.transform(
                wrap_module(EmbeddingsDecoder_Logits,
                            n_layers=2,
                            embeddings_size=self.dimensions['diag_emb'],
                            diag_size=self.dimensions['diag_out'],
                            name='f_dec')))
        f_dec = jax.jit(f_dec)
        self.f_dec = (lambda params, *args: f_dec(params['f_dec'], *args))

        self.initializers = {
            'f_n_ode': f_n_ode_init,
            'f_update': f_update_init,
            'f_dec': f_dec_init
        }
        self.init_data = self._initialization_data()

    def join_state_emb(self, state, emb):
        if state is None:
            state = jnp.zeros((self.dimensions['state'], ))
        return jnp.hstack((state, emb))

    def split_state_emb(self, state_emb):
        return jnp.split(state_emb, (self.dimensions['state'], ))

    def init_params(self, rng_key):
        return {
            "diag_emb": self.diag_emb.init_params(rng_key),
            **{
                label: init(rng_key, *self.init_data[label])
                for label, init in self.initializers.items()
            }
        }

    def state_size(self):
        return self.dimensions['state']

    def diag_out_index(self) -> List[str]:
        index2code = {
            i: c
            for c, i in self.subject_interface.diag_ccs_idx.items()
        }
        return list(map(index2code.get, range(len(index2code))))

    def _initialization_data(self):
        """
        Creates data for initializing each of the
        modules based on the shapes of init_data.
        """
        emb = jnp.zeros(self.dimensions['diag_emb'])
        state = jnp.zeros(self.dimensions['state'])
        state_emb = jnp.hstack((state, emb))
        ode_ctrl = jnp.array([])
        return {
            "f_n_ode": [2, True, state_emb, 0.1, ode_ctrl],
            "f_update": [state, emb, emb],
            "f_dec": [emb],
        }

    def _extract_nth_admission(self, params: Any, subjects_batch: List[int],
                               n: int) -> Dict[str, Dict[int, jnp.ndarray]]:
        diag_G = self.diag_emb.compute_embeddings_mat(params["diag_emb"])

        adms = self.subject_interface.nth_admission_batch(n, subjects_batch)
        if len(adms) == 0:
            return None

        diag_emb = {
            i: self.diag_emb.encode(diag_G, v['diag_ccs_vec'])
            for i, v in adms.items() if v['diag_ccs_vec'] is not None
        }
        diag_out = {
            i: v['diag_ccs_vec']
            for i, v in adms.items() if v['diag_ccs_vec'] is not None
        }
        los = {i: v['los'] for i, v in adms.items()}
        adm_id = {i: v['admission_id'] for i, v in adms.items()}
        adm_time = {i: v['time'] for i, v in adms.items()}

        return {
            'time': adm_time,
            'los': los,
            'diag_emb': diag_emb,
            'diag_out': diag_out,
            'admission_id': adm_id
        }

    def _f_n_ode(self, params, count_nfe, state_e, t):
        c = jnp.array([])
        h_r_nfe = {
            i: self.f_n_ode(params, 2, count_nfe, state_e[i], t[i], c)
            for i in t
        }
        nfe = {i: n for i, (h, r, n) in h_r_nfe.items()}
        nfe_sum = sum(nfe.values())
        drdt = jnp.sum(sum(r for (h, r, n) in h_r_nfe.values()))

        state_e = {i: h[-1, :] for i, (h, r, n) in h_r_nfe.items()}
        return state_e, (drdt, nfe, nfe_sum)

    def _f_update(self, params: Any, state_e: Dict[int, jnp.ndarray],
                  emb: jnp.ndarray) -> jnp.ndarray:
        new_state = {}
        for i in emb:
            emb_nominal = emb[i]
            state, emb_pred = self.split_state_emb(state_e[i])
            state = self.f_update(params, state, emb_pred, emb_nominal)
            new_state[i] = self.join_state_emb(state, emb_nominal)
        return new_state

    def _f_dec(self, params: Any, state_e: Dict[int, jnp.ndarray]):
        emb = {i: self.split_state_emb(state_e[i])[1] for i in state_e}
        return {i: self.f_dec(params, emb[i]) for i in emb}

    def _diag_loss(self, diag: Dict[int, jnp.ndarray],
                   dec_diag: Dict[int, jnp.ndarray]):
        l = {i: softmax_logits_bce(diag[i], dec_diag[i]) for i in diag.keys()}
        return sum(l.values()) / len(l)

    def __call__(self,
                 params: Any,
                 subjects_batch: List[int],
                 count_nfe: bool = False):
        nth_adm = partial(self._extract_nth_admission, params, subjects_batch)
        nn_ode = partial(self._f_n_ode, params, count_nfe)
        nn_update = partial(self._f_update, params)
        nn_decode = partial(self._f_dec, params)
        diag_loss = self._diag_loss

        adm0 = nth_adm(0)
        subject_state = {
            i: {
                'state_e': self.join_state_emb(None, adm0['diag_emb'][i]),
                'time': adm0['time'][i] + adm0['los'][i]
            }
            for i in adm0['admission_id']
        }

        total_nfe = 0

        prediction_losses = []

        adm_counts = []
        diag_detectability = {i: {} for i in subjects_batch}
        odeint_weeks = 0.0
        dyn_losses = []

        for n in self.subject_interface.n_support[1:]:
            adm_n = nth_adm(n)
            if adm_n is None:
                break
            adm_id = adm_n['admission_id']
            adm_los = adm_n['los']  # length of stay
            adm_time = adm_n['time']
            emb = adm_n['diag_emb']
            diag = adm_n['diag_out']

            # To normalize the prediction loss by the number of patients
            adm_counts.append(len(adm_id))

            state_e = {i: subject_state[i]['state_e'] for i in adm_id}

            d2d_time = {
                i: adm_time[i] + adm_los[i] - subject_state[i]['time']
                for i in adm_id
            }

            # Integrate until next discharge
            state_e, (r, nfe, nfe_sum) = nn_ode(state_e, d2d_time)
            dec_diag = nn_decode(state_e)

            for subject_id in state_e.keys():
                diag_detectability[subject_id][n] = {
                    'admission_id': adm_id[subject_id],
                    'nfe': nfe[subject_id],
                    'time': adm_time[subject_id] + adm_los[subject_id],
                    'los': adm_los[subject_id],
                    'diag_true': diag[subject_id],
                    'pre_logits': dec_diag[subject_id]
                }

            odeint_weeks += sum(d2d_time.values()) / 7
            dyn_losses.append(r)
            prediction_losses.append(diag_loss(diag, dec_diag))
            total_nfe += nfe_sum

            # Update state at discharge
            state_e = nn_update(state_e, emb)

            # Update the states:
            for subject_id, new_state in state_e.items():
                subject_state[subject_id] = {
                    'time': adm_time[subject_id] + adm_los[subject_id],
                    'state_e': new_state
                }

        prediction_loss = jnp.average(prediction_losses, weights=adm_counts)

        ret = {
            'prediction_loss': prediction_loss,
            'dyn_loss': jnp.sum(sum(dyn_losses)),
            'odeint_weeks': odeint_weeks,
            'admissions_count': sum(adm_counts),
            'nfe': total_nfe,
            'diag_detectability': diag_detectability
        }

        return ret

    def detailed_loss(self, loss_mixing, params, res):
        prediction_loss = res['prediction_loss']
        dyn_loss = res['dyn_loss']
        dyn_alpha = loss_mixing['L_dyn'] / (res['odeint_weeks'] + 1e-10)
        loss = prediction_loss + dyn_alpha * dyn_loss
        return {
            'loss': loss,
            'prediction_loss': prediction_loss,
            'dyn_loss': dyn_loss
        }

    def eval_stats(self, res):
        nfe = res['nfe']
        return {
            'admissions_count': res['admissions_count'],
            'nfe_per_week': nfe / (res['odeint_weeks'] + 1e-10),
            'nfex1000': nfe / 1000
        }

    @staticmethod
    def create_patient_interface(mimic_dir, data_tag: str):
        return create_patient_interface(mimic_dir, data_tag=data_tag)

    @classmethod
    def create_model(cls, config, patient_interface, train_ids,
                     pretrained_components):
        diag_emb = cls.create_embedding(
            emb_config=config['emb']['diag'],
            emb_kind=config['emb']['kind'],
            patient_interface=patient_interface,
            train_ids=train_ids,
            pretrained_components=pretrained_components)
        return cls(subject_interface=patient_interface,
                   diag_emb=diag_emb,
                   **config['model'])

    @staticmethod
    def _sample_ode_training_config(trial: optuna.Trial, epochs):
        config = AbstractModel._sample_training_config(trial, epochs)
        config['loss_mixing']['L_dyn'] = trial.suggest_float('L_dyn',
                                                             1e-6,
                                                             1e3,
                                                             log=True)
        return config

    @staticmethod
    def sample_training_config(trial: optuna.Trial):
        return ICENODE._sample_ode_training_config(trial, epochs=20)

    @staticmethod
    def _sample_ode_model_config(trial: optuna.Trial):
        model_params = {
            'ode_dyn': trial.suggest_categorical('ode_dyn', ['mlp', 'gru']),
            'ode_with_bias': False,
            'ode_init_var': trial.suggest_float('ode_i', 1e-15, 1e-2,
                                                log=True),
            'state_size': trial.suggest_int('s', 10, 100, 10),
        }
        return model_params

    @classmethod
    def sample_model_config(cls, trial: optuna.Trial):
        return cls._sample_ode_model_config(trial)


if __name__ == '__main__':
    from .hpo_utils import capture_args, run_trials
    run_trials(model_cls=ICENODE, **capture_args())
