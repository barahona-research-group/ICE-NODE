from functools import partial
from typing import (Any, Callable, Dict, List, Optional, Tuple)

from absl import logging
import jax
import jax.numpy as jnp
import optuna

from .jax_interface import DiagnosisJAXInterface
from .gram import AbstractEmbeddingsLayer
from .train_icenode import ICENODE
from .abstract_model import AbstractModel


class ICENODE3(ICENODE):

    def __init__(self, subject_interface: DiagnosisJAXInterface,
                 diag_emb: AbstractEmbeddingsLayer, ode_dyn: str,
                 ode_depth: int, ode_with_bias: bool, ode_init_var: float,
                 ode_timescale: float, los_sample_rate: int,
                 tay_reg: Optional[int], state_size: int, init_depth: bool,
                 diag_loss: Callable[[jnp.ndarray, jnp.ndarray], float]):
        super().__init__(subject_interface=subject_interface,
                         diag_emb=diag_emb,
                         ode_dyn=ode_dyn,
                         ode_depth=ode_depth,
                         ode_with_bias=ode_with_bias,
                         ode_init_var=ode_init_var,
                         ode_timescale=ode_timescale,
                         los_sample_rate=los_sample_rate,
                         tay_reg=tay_reg,
                         state_size=state_size,
                         init_depth=init_depth,
                         diag_loss=diag_loss)

        self.d2d_sample_rate = 2 * los_sample_rate

    def _f_n_ode_d2d(self, params, count_nfe, h, t):
        null = jnp.array([])
        c = {i: null for i in h}

        h_r_nfe = {
            i: self.f_n_ode(params['f_n_ode'], self.d2d_sample_rate, count_nfe,
                            h[i], t[i], c[i])
            for i in t.keys()
        }
        nfe = {i: n for i, (h, r, n) in h_r_nfe.items()}
        nfe_sum = sum(nfe.values())
        drdt = jnp.sum(sum(r for (h, r, n) in h_r_nfe.values()))

        h_samples = {i: h for i, (h, r, n) in h_r_nfe.items()}
        h_final = {i: hi[-1, :] for i, hi in h_samples.items()}
        return h_samples, h_final, (drdt, nfe, nfe_sum)

    def _f_dec_d2d(
        self, params: Any, state_samples: Dict[int, jnp.ndarray]
    ) -> Tuple[Dict[int, jnp.ndarray], Dict[int, jnp.ndarray]]:

        dec_out = {
            i: jax.vmap(partial(self.f_dec, params['f_dec']))(state)
            for i, state in state_samples.items()
        }
        emb = {i: jnp.mean(g, axis=0) for i, (g, _) in dec_out.items()}
        out = {i: jnp.mean(o, axis=0) for i, (_, o) in dec_out.items()}

        return emb, out

    def __call__(self,
                 params: Any,
                 subjects_batch: List[int],
                 count_nfe: bool = False):
        nth_adm = partial(self._extract_nth_admission, params, subjects_batch)
        # nODE: within an admission (LoS: Legth of Stay)
        nn_ode_d2d = partial(self._f_n_ode_d2d, params, count_nfe)

        nn_update = partial(self._f_update, params)

        # For decoding over samples within a single admission
        nn_decode_d2d = partial(self._f_dec_d2d, params)

        nn_init = partial(self._f_init, params)
        diag_loss = self._diag_loss
        subject_state = {}
        dyn_losses = []
        total_nfe = 0

        prediction_losses = []
        update_losses = []

        adm_counts = []

        diag_detectability = {i: {} for i in subjects_batch}
        odeint_weeks = 0.0

        for n in self.subject_interface.n_support:
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

            update_loss = 0.0
            if n == 0:
                # Initialize all states for first admission (no predictions)
                state0 = nn_init(emb)
                subject_state.update(state0)
                state0 = {i: subject_state[i]['state'] for i in adm_id}
                # Integrate until first discharge
                state_samples, state, (drdt, _,
                                       nfe_sum) = nn_ode_d2d(state0, adm_los)
                dec_emb, dec_diag = nn_decode_d2d(state_samples)

                odeint_weeks += sum(adm_los.values()) / 7

            else:
                state = {i: subject_state[i]['state'] for i in adm_id}

                d2d_time = {
                    i: adm_time[i] + adm_los[i] - subject_state[i]['time']
                    for i in adm_id
                }

                # Integrate until next discharge
                state_samples, state, (drdt, nfe,
                                       nfe_sum) = nn_ode_d2d(state, d2d_time)
                dec_emb, dec_diag = nn_decode_d2d(state_samples)

                for subject_id in state.keys():
                    diag_detectability[subject_id][n] = {
                        'admission_id': adm_id[subject_id],
                        'nfe': nfe[subject_id],
                        'time': adm_time[subject_id],
                        'diag_true': diag[subject_id],
                        'pre_logits': dec_diag[subject_id]
                    }
                prediction_losses.append(diag_loss(diag, dec_diag))

                odeint_weeks += sum(d2d_time.values()) / 7

            dyn_losses.append(drdt)
            total_nfe += nfe_sum
            # Update state at discharge
            state, update_loss = nn_update(state, emb, diag, dec_emb)
            update_losses.append(update_loss)

            # Update the states:
            for subject_id, new_state in state.items():
                subject_state[subject_id] = {
                    'time': adm_time[subject_id] + adm_los[subject_id],
                    'state': new_state
                }

        prediction_loss = jnp.average(prediction_losses,
                                      weights=adm_counts[1:])
        update_loss = jnp.average(update_losses, weights=adm_counts)

        ret = {
            'prediction_loss': prediction_loss,
            'update_loss': update_loss,
            'los_loss': 0.0,
            'dyn_loss': jnp.sum(sum(dyn_losses)),
            'odeint_weeks': odeint_weeks,
            'admissions_count': sum(adm_counts),
            'nfe': total_nfe,
            'diag_detectability': diag_detectability
        }

        return ret

    @staticmethod
    def _sample_ode_training_config(trial: optuna.Trial, epochs):
        config = AbstractModel._sample_training_config(trial, epochs)
        config['diag_loss'] = 'softmax'

        config['loss_mixing'] = {
            'L_pred': trial.suggest_float('L_pred', 1e-4, 1, log=True),
            'L_los': 0.0,
            'L_update': trial.suggest_float('L_update', 1e-4, 1, log=True),
            'L_dyn': trial.suggest_float('L_dyn', 1e-3, 1e3, log=True),
            **config['loss_mixing']
        }

        return config

    @staticmethod
    def sample_training_config(trial: optuna.Trial):
        return ICENODE3._sample_ode_training_config(trial, epochs=10)


if __name__ == '__main__':
    from .hpo_utils import capture_args, run_trials
    run_trials(model_cls=ICENODE3, **capture_args())
