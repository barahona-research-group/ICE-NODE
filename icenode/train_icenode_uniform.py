from functools import partial
from typing import (Any, Callable, Dict, Iterable, List, Optional, Set)

import jax
import jax.numpy as jnp

from .train_icenode_tl import ICENODE as ICENODE_TL


class ICENODE(ICENODE_TL):

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

            d2d_time = {i: 7.0 for i in adm_id}

            # Integrate until next discharge
            state_e, r, nfe = nn_ode(state_e, d2d_time)
            dec_diag = nn_decode(state_e)

            for subject_id in state_e.keys():
                diag_detectability[subject_id][n] = {
                    'admission_id': adm_id[subject_id],
                    'nfe': nfe[subject_id],
                    'time': adm_time[subject_id] + adm_los[subject_id],
                    'los': adm_los[subject_id],
                    'R/T': r[subject_id] / d2d_time[subject_id],
                    'true_diag': diag[subject_id],
                    'pred_logits': dec_diag[subject_id]
                }

            odeint_time.append(sum(d2d_time.values()))
            dyn_loss += sum(r.values())

            pred_loss = diag_loss(diag, dec_diag)
            prediction_losses.append(pred_loss)

            total_nfe += sum(nfe.values())

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
            'dyn_loss': dyn_loss,
            'odeint_weeks': sum(odeint_time) / 7.0,
            'admissions_count': sum(adm_counts),
            'nfe': total_nfe,
            'diag_detectability': diag_detectability
        }

        return ret


if __name__ == '__main__':
    from .hpo_utils import capture_args, run_trials
    run_trials(model_cls=ICENODE, **capture_args())
