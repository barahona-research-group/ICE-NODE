from functools import partial
from typing import (Any, Callable, Dict, Iterable, List, Optional, Set)

from absl import logging
import jax
import jax.numpy as jnp
import haiku as hk

import optuna

from .metrics import (balanced_focal_bce)
from .jax_interface import (DiagnosisJAXInterface)
from .gram import AbstractEmbeddingsLayer
from .train_icenode_mtl import ICENODE as ICENODE_MTL


class ICENODE(ICENODE_MTL):

    def __init__(self, subject_interface: DiagnosisJAXInterface,
                 diag_emb: AbstractEmbeddingsLayer, ode_dyn: str,
                 ode_with_bias: bool, ode_init_var: float, loss_half_life: int,
                 memory_size: int, timescale: float):
        super().__init__(subject_interface=subject_interface,
                         diag_emb=diag_emb,
                         ode_dyn=ode_dyn,
                         ode_with_bias=ode_with_bias,
                         ode_init_var=ode_init_var,
                         memory_size=memory_size,
                         timescale=timescale)
        self.timescale = timescale
        self.trajectory_samples = 3
        self.lambd = jnp.log(2) / (loss_half_life / timescale)

    def _f_n_ode(self, params, count_nfe, emb, t, mem):
        h_r_nfe = {
            i: self.f_n_ode(params['f_n_ode'], self.trajectory_samples + 1,
                            False, emb[i], t[i], mem[i])
            for i in t
        }
        n = self._f_n_ode_nfe(params, count_nfe, emb, t, mem)
        r = {i: r[-1] for i, (_, r, _) in h_r_nfe.items()}
        emb_seq = {i: h[1:, :] for i, (h, _, _) in h_r_nfe.items()}
        return emb_seq, r, n

    def _f_dec(self, params: Any, emb: Dict[int, jnp.ndarray]):
        return {
            i: jax.vmap(partial(self.f_dec, params['f_dec']))(e)
            for i, e in emb.items()
        }

    def _diag_loss(self, diag: Dict[int, jnp.ndarray],
                   dec_diag_seq: Dict[int, jnp.ndarray], t: Dict[int, float]):
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
        diag_loss = self._diag_loss

        adm0 = nth_adm(0)
        rnn0 = nn_update(adm0['diag_emb'])
        subject_state = {
            i: {
                'rnn': rnn0[i][1],
                'mem': rnn0[i][0],
                'emb': adm0['diag_emb'][i],
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

            d2d_time = {
                i: adm_time[i] + adm_los[i] - subject_state[i]['time']
                for i in adm_id
            }

            emb = {i: subject_state[i]['emb'] for i in adm_id}
            mem = {i: subject_state[i]['mem'] for i in adm_id}
            rnn = {i: subject_state[i]['rnn'] for i in adm_id}

            # Integrate until next discharge
            emb_seq, r, nfe = nn_ode(emb, d2d_time, mem)
            dec_diag_seq = nn_decode(emb_seq)

            for subject_id in adm_id:
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

            pred_loss = diag_loss(diag, dec_diag_seq, d2d_time)
            prediction_losses.append(pred_loss)

            total_nfe += sum(nfe.values())

            # Update state at discharge
            true_emb = adm_n['diag_emb']
            rnn = nn_update(true_emb, rnn)

            # Update the states:
            for subject_id in emb:
                subject_state[subject_id] = {
                    'time': adm_time[subject_id] + adm_los[subject_id],
                    'emb': emb_seq[subject_id][-1, :],
                    'mem': rnn[subject_id][0],
                    'rnn': rnn[subject_id][1]
                }

        prediction_loss = jnp.average(prediction_losses, weights=adm_counts)

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
            **ICENODE_MTL.sample_model_config(trial)
        }


if __name__ == '__main__':
    from .hpo_utils import capture_args, run_trials
    run_trials(model_cls=ICENODE, **capture_args())
