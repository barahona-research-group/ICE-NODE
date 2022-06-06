from functools import partial
from typing import (Any, Callable, Dict, Iterable, List, Optional, Set)

import jax
import jax.numpy as jnp

import optuna

from .metrics import (balanced_focal_bce)
from .train_icenode_tl import ICENODE as ICENODE_TL


class ICENODE(ICENODE_TL):

    def __init__(self, loss_half_life: int, **kwargs):
        super().__init__(**kwargs)
        self.trajectory_samples = 3
        self.lambd = jnp.log(2) / (loss_half_life / self.timescale)

    def split_state_emb_seq(self, seq):
        # seq.shape: (time_samples, state_size + embeddings_size)
        return jnp.split(seq, (self.dimensions['state'], ), axis=1)

    def _f_n_ode(self, params, count_nfe, state_e, t):
        h_r_nfe = {
            i: self.f_n_ode(params['f_n_ode'], self.trajectory_samples + 1,
                            False, state_e[i], t[i])
            for i in t
        }
        if count_nfe:
            n = {
                i: self.f_n_ode(params['f_n_ode'], 2, True, state_e[i],
                                t[i])[-1]
                for i in t
            }
        else:
            n = {i: 0 for i in t}

        r = {i: r[-1] for i, (_, r, _) in h_r_nfe.items()}
        state_seq = {i: h[1:, :] for i, (h, _, _) in h_r_nfe.items()}
        return state_seq, r, n

    def _f_update(self, params: Any, state_seq: Dict[int, jnp.ndarray],
                  emb: jnp.ndarray) -> jnp.ndarray:
        new_state = {}
        for i in emb:
            emb_nominal = emb[i]
            state, emb_pred = self.split_state_emb_seq(state_seq[i])
            # state.shape: (timesamples, state_size)
            # emb_pred.shape: (timesamples, embeddings_size)
            state = self.f_update(params['f_update'], state[-1], emb_pred[-1],
                                  emb_nominal)
            new_state[i] = self.join_state_emb(state, emb_nominal)
        return new_state

    def _f_dec(self, params: Any, state_e: Dict[int, jnp.ndarray]):
        emb = {i: self.split_state_emb_seq(state_e[i])[1] for i in state_e}
        # each (emb_i in emb).shape: (timesamples, emebddings_size)

        # each (out_i in output).shape: (timesamples, diag_size)
        return {
            i: jax.vmap(partial(self.f_dec, params['f_dec']))(emb[i])
            for i in emb
        }

    def _diag_sampled_loss(self, diag: Dict[int, jnp.ndarray],
                           dec_diag_seq: Dict[int, jnp.ndarray],
                           t: Dict[int, float]):
        loss_vals = []
        for i in diag:
            ti = t[i] / self.timescale

            t_seq = jnp.linspace(0.0, ti, self.trajectory_samples + 1)[1:]
            t_diff = jax.lax.stop_gradient(self.lambd * (t_seq - ti))
            loss_seq = [
                balanced_focal_bce(diag[i], dec_diag_seq[i][j])
                for j in range(self.trajectory_samples)
            ]
            loss_vals.append(
                sum(l * jnp.exp(dt) for (l, dt) in zip(loss_seq, t_diff)))
        return sum(loss_vals) / len(loss_vals)

    def __call__(self,
                 params: Any,
                 subjects_batch: List[int],
                 count_nfe: bool = False):
        nth_adm = partial(self._extract_nth_admission, params, subjects_batch)
        nn_ode = partial(self._f_n_ode, params, count_nfe)
        nn_update = partial(self._f_update, params)
        nn_decode = partial(self._f_dec, params)
        diag_loss = self._diag_sampled_loss

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
        odeint_time = []
        dyn_loss = 0

        for n in self.subject_interface.n_support[1:]:
            adm_n = nth_adm(n)
            if adm_n is None:
                break
            adm_id = adm_n['admission_id']
            adm_los = adm_n['los']  # length of stay
            adm_time = adm_n['time']
            emb = adm_n['diag_emb']
            diag = adm_n['diag_out']

            adm_counts.append(len(adm_id))

            state_e = {i: subject_state[i]['state_e'] for i in adm_id}

            d2d_time = {
                i: adm_time[i] + adm_los[i] - subject_state[i]['time']
                for i in adm_id
            }

            # Integrate until next discharge
            state_seq, r, nfe = nn_ode(state_e, d2d_time)
            dec_diag_seq = nn_decode(state_seq)

            for subject_id in state_seq.keys():
                diag_detectability[subject_id][n] = {
                    'admission_id': adm_id[subject_id],
                    'nfe': nfe[subject_id],
                    'time': adm_time[subject_id],
                    'los': adm_los[subject_id],
                    'true_diag': diag[subject_id],
                    'pred_logits': dec_diag_seq[subject_id][-1, :]
                }

            odeint_time.append(sum(d2d_time.values()))
            dyn_loss += sum(r.values())

            pred_loss_avg = diag_loss(diag, dec_diag_seq, d2d_time)
            prediction_losses.append(pred_loss_avg)

            total_nfe += sum(nfe.values())

            # Update state at discharge
            state_e = nn_update(state_seq, emb)

            # Update the states:
            for subject_id, new_state in state_e.items():
                subject_state[subject_id] = {
                    'time': adm_time[subject_id] + adm_los[subject_id],
                    'state_e': new_state
                }

        prediction_loss = jnp.average(jnp.array(prediction_losses),
                                      weights=jnp.array(adm_counts))

        ret = {
            'prediction_loss': prediction_loss,
            'dyn_loss': dyn_loss,
            'odeint_weeks': sum(odeint_time) / 7.0,
            'admissions_count': sum(adm_counts),
            'nfe': total_nfe,
            'diag_detectability': diag_detectability
        }

        return ret

    @classmethod
    def sample_model_config(cls, trial: optuna.Trial):
        return {
            'loss_half_life': trial.suggest_int('lt0.5', 7, 7 * 1e2, log=True),
            **ICENODE_TL.sample_model_config(trial)
        }


if __name__ == '__main__':
    from .hpo_utils import capture_args, run_trials
    run_trials(model_cls=ICENODE, **capture_args())
