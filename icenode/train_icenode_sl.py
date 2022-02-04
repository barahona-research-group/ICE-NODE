from functools import partial
from typing import (Any, Callable, Dict, Iterable, List, Optional, Set)

import jax
import jax.numpy as jnp

import optuna

from .metrics import (softmax_logits_bce)
from .jax_interface import (DiagnosisJAXInterface)
from .gram import AbstractEmbeddingsLayer
from .train_icenode_tl import ICENODE as ICENODE_TL


class ICENODE(ICENODE_TL):

    def __init__(self, subject_interface: DiagnosisJAXInterface,
                 diag_emb: AbstractEmbeddingsLayer, ode_dyn: str,
                 ode_with_bias: bool, ode_init_var: float, loss_half_life: int,
                 state_size: int):
        super().__init__(subject_interface=subject_interface,
                         diag_emb=diag_emb,
                         ode_dyn=ode_dyn,
                         ode_with_bias=ode_with_bias,
                         ode_init_var=ode_init_var,
                         state_size=state_size)

        self.trajectory_samples = 3
        self.loss_half_life = loss_half_life

    def split_state_emb_seq(self, seq):
        # seq.shape: (time_samples, state_size + embeddings_size)
        return jnp.split(seq, (self.dimensions['state'], ), axis=1)

    def _f_n_ode(self, params, count_nfe, state_e, t):
        c = jnp.array([])
        h_r_nfe = {
            i: self.f_n_ode(params, self.trajectory_samples + 1, count_nfe,
                            state_e[i], t[i], c)
            for i in t
        }
        nfe = {i: n for i, (h, r, n) in h_r_nfe.items()}
        nfe_sum = sum(nfe.values())
        drdt = jnp.sum(sum(r for (h, r, n) in h_r_nfe.values()))

        state_seq = {i: h[1:, :] for i, (h, r, n) in h_r_nfe.items()}
        return state_seq, (drdt, nfe, nfe_sum)

    def _f_update(self, params: Any, state_seq: Dict[int, jnp.ndarray],
                  emb: jnp.ndarray) -> jnp.ndarray:
        new_state = {}
        for i in emb:
            emb_nominal = emb[i]
            state, emb_pred = self.split_state_emb_seq(state_seq[i])
            # state.shape: (timesamples, state_size)
            # emb_pred.shape: (timesamples, embeddings_size)
            state = self.f_update(params, state[-1], emb_pred[-1], emb_nominal)
            new_state[i] = self.join_state_emb(state, emb_nominal)
        return new_state

    def _f_dec(self, params: Any, state_e: Dict[int, jnp.ndarray]):
        emb = {i: self.split_state_emb_seq(state_e[i])[1] for i in state_e}
        # each (emb_i in emb).shape: (timesamples, emebddings_size)

        # each (out_i in output).shape: (timesamples, diag_size)
        return {i: jax.vmap(partial(self.f_dec, params))(emb[i]) for i in emb}

    def _diag_loss(self, diag: Dict[int, jnp.ndarray],
                   dec_diag_seq: Dict[int, jnp.ndarray], t: Dict[int, float]):
        loss_vals = {}
        for i in diag:
            t_seq = jnp.linspace(0.0, t[i], self.trajectory_samples + 1)[1:]
            t_diff = (t_seq[-1] - t_seq) / self.loss_half_life
            loss_seq = [
                softmax_logits_bce(diag[i], dec_diag_seq[i][j])
                for j in range(self.trajectory_samples)
            ]
            loss_vals[i] = sum(l * jnp.power(0.5, dt)
                               for (l, dt) in zip(loss_seq, t_diff))

        return sum(loss_vals.values()) / len(loss_vals)

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

        dyn_losses = []
        total_nfe = 0

        prediction_losses = []

        adm_counts = []
        diag_detectability = {i: {} for i in subjects_batch}
        odeint_weeks = 0.0

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

            state_di = {i: subject_state[i]['state_e'] for i in adm_id}

            d2d_time = {
                i: adm_time[i] + adm_los[i] - subject_state[i]['time']
                for i in adm_id
            }

            # Integrate until next discharge
            state_seq, (drdt, nfe, nfe_sum) = nn_ode(state_di, d2d_time)
            dec_diag_seq = nn_decode(state_seq)

            for subject_id in state_seq.keys():
                diag_detectability[subject_id][n] = {
                    'admission_id': adm_id[subject_id],
                    'nfe': nfe[subject_id],
                    'time': adm_time[subject_id],
                    'los': adm_los[subject_id],
                    'diag_true': diag[subject_id],
                    'pre_logits': dec_diag_seq[subject_id][-1, :]
                }

            odeint_weeks += sum(d2d_time.values()) / 7
            prediction_losses.append(diag_loss(diag, dec_diag_seq, d2d_time))
            dyn_losses.append(drdt)
            total_nfe += nfe_sum

            # Update state at discharge
            state_dj = nn_update(state_seq, emb)

            # Update the states:
            for subject_id, new_state in state_dj.items():
                subject_state[subject_id] = {
                    'time': adm_time[subject_id] + adm_los[subject_id],
                    'state_e': new_state
                }

        prediction_loss = jnp.average(prediction_losses, weights=adm_counts)

        ret = {
            'prediction_loss': prediction_loss,
            'odeint_weeks': odeint_weeks,
            'admissions_count': sum(adm_counts),
            'nfe': total_nfe,
            'diag_detectability': diag_detectability
        }

        return ret

    @staticmethod
    def _sample_ode_model_config(trial: optuna.Trial):
        model_params = {
            'ode_dyn': trial.suggest_categorical('ode_dyn', ['mlp', 'gru']),
            'ode_with_bias': False,
            'loss_half_life': trial.suggest_int('lt0.5', 1, 2e2, log=True),
            'ode_init_var': trial.suggest_float('ode_i', 1e-15, 1e-2,
                                                log=True),
            'state_size': trial.suggest_int('s', 10, 100, 10),
        }
        return model_params


if __name__ == '__main__':
    from .hpo_utils import capture_args, run_trials
    run_trials(model_cls=ICENODE, **capture_args())
