from functools import partial
from typing import (Any, Callable, Dict, Iterable, List, Optional, Set)

import jax
import jax.numpy as jnp

from .dx_icenode_tl import ICENODE as ICENODE_TL
from .risk import BatchPredictedRisks


class ICENODE(ICENODE_TL):
    def __call__(self,
                 params: Any,
                 subjects_batch: List[int],
                 count_nfe: bool = False):
        batch = self.subject_interface.batch_nth_admission(subjects_batch)
        nth_adm = partial(self._extract_nth_admission, params, batch)
        nn_ode = partial(self._f_n_ode, params, count_nfe)
        nn_update = partial(self._f_update, params)
        nn_decode = partial(self._f_dec, params)
        dx_loss = self._dx_loss

        adm0 = nth_adm(0)
        subject_state = {
            i: {
                'state_e': self.join_state_emb(None, adm0['dx_emb'][i]),
                'time': adm0['admission_time'][i] + adm0['los'][i]
            }
            for i in adm0['admission_id']
        }
        total_nfe = 0

        prediction_losses = []

        adm_counts = []
        risk_prediction = BatchPredictedRisks()
        odeint_time = []
        dyn_loss = 0

        for n in sorted(batch)[1:]:
            adm_n = nth_adm(n)
            if adm_n is None:
                break
            adm_id = adm_n['admission_id']
            adm_los = adm_n['los']  # length of stay
            adm_time = adm_n['admission_time']
            emb = adm_n['dx_emb']
            dx = adm_n['dx_out']

            adm_counts.append(len(adm_id))

            state_e = {i: subject_state[i]['state_e'] for i in adm_id}

            d2d_time = {i: 7.0 for i in adm_id}

            # Integrate until next discharge
            state_e, r, nfe = nn_ode(state_e, d2d_time)
            dec_dx = nn_decode(state_e)

            for subject_id in state_e.keys():
                risk_prediction.add(subject_id=subject_id,
                                    admission_id=adm_id[subject_id],
                                    ground_truth=dx[subject_id],
                                    prediction=dec_dx[subject_id],
                                    index=n,
                                    time=adm_time[subject_id] +
                                    adm_los[subject_id],
                                    los=adm_los[subject_id],
                                    nfe=nfe[subject_id])

            odeint_time.append(sum(d2d_time.values()))
            dyn_loss += sum(r.values())

            pred_loss = dx_loss(dx, dec_dx)
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

        prediction_loss = jnp.average(jnp.array(prediction_losses),
                                      weights=jnp.array(adm_counts))

        ret = {
            'prediction_loss': prediction_loss,
            'dyn_loss': dyn_loss,
            'odeint_weeks': sum(odeint_time) / 7.0,
            'admissions_count': sum(adm_counts),
            'nfe': total_nfe,
            'risk_prediction': risk_prediction
        }

        return ret


if __name__ == '__main__':
    from ..hyperopt.hpo_utils import capture_args, run_trials
    run_trials(model_cls=ICENODE, **capture_args())
