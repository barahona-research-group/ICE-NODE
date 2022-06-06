from functools import partial
from typing import (Any, Callable, Dict, Iterable, List, Optional, Set)

import jax
import jax.numpy as jnp
import haiku as hk
import optuna

from .metrics import (softmax_logits_bce)
from .jax_interface import (DiagnosisJAXInterface)
from .gram import AbstractEmbeddingsLayer
from .models import NeuralODE_IL, GRUDynamics, ResDynamics, MLPDynamics
from .train_icenode_tl import ICENODE as ICENODE_TL
from .utils import wrap_module
from .abstract_model import AbstractModel


class ICENODE(ICENODE_TL):

    def __init__(self, subject_interface: DiagnosisJAXInterface,
                 diag_emb: AbstractEmbeddingsLayer, ode_dyn: str,
                 ode_with_bias: bool, ode_init_var: float, loss_half_life: int,
                 state_size: int, timescale: float):
        super().__init__(subject_interface=subject_interface,
                         diag_emb=diag_emb,
                         ode_dyn=ode_dyn,
                         ode_with_bias=ode_with_bias,
                         ode_init_var=ode_init_var,
                         state_size=state_size,
                         timescale=timescale)

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
                wrap_module(NeuralODE_IL,
                            ode_dyn_cls=ode_dyn_cls,
                            state_size=state_emb_size,
                            depth=1,
                            loss_f=self.diag_loss,
                            with_bias=ode_with_bias,
                            init_var=ode_init_var,
                            timescale=timescale,
                            name='f_n_ode_il',
                            tay_reg=3)))
        self.f_n_ode = jax.jit(f_n_ode, static_argnums=(1, ))

        self.timescale = timescale
        self.loss_half_life = loss_half_life / timescale

        self.initializers['f_n_ode'] = f_n_ode_init

    def init_params(self, rng_key):
        emb = jnp.zeros(self.dimensions['diag_emb'])
        state = jnp.zeros(self.dimensions['state'])
        diag = jnp.zeros(self.dimensions['diag_out'])
        state_emb = self.join_state_emb(state, emb)

        dec_params = self.initializers['f_dec'](rng_key, emb)
        # lambda_param = self.initializers['f_lambda'](rng_key)

        loss_args = (0.1, diag, dec_params)
        ode_params = self.initializers['f_n_ode'](rng_key, True, state_emb,
                                                  0.1, None, *loss_args)
        update_params = self.initializers['f_update'](rng_key, state, emb, emb)

        return {
            "diag_emb": self.diag_emb.init_params(rng_key),
            'f_n_ode': ode_params,
            'f_dec': dec_params,
            'f_update': update_params
        }

    @partial(jax.jit, static_argnums=(0, ))
    def diag_loss(self, h: jnp.ndarray, dhdt: jnp.ndarray, ti: float,
                  tf: float, y: jnp.ndarray, f_dec_params: Any):
        _, e = self.split_state_emb(h)
        y_hat = self.f_dec(f_dec_params, e)
        dldy = softmax_logits_bce(y, y_hat)
        lambd = jnp.log(2) / self.loss_half_life
        return dldy * jax.lax.stop_gradient(jnp.exp(-lambd * jnp.abs(ti - tf)))

    def _f_n_ode(self, params, count_nfe, state_e, t, diag):
        # s: Integrated state
        # l: Integrated loss
        # r: Integrated dynamics penalty
        # n: Number of dynamics function calls
        s_l_r_n = {
            i: self.f_n_ode(params['f_n_ode'], False, state_e[i], t[i], None,
                            t[i] / self.timescale, diag[i], params['f_dec'])
            for i in t
        }

        if count_nfe:
            n = {
                i: self.f_n_ode(params['f_n_ode'], True, state_e[i], t[i],
                                None, t[i] / self.timescale, diag[i],
                                params['f_dec'])[-1]
                for i in t
            }
        else:
            n = {i: 0 for i in t}

        state_e = {i: s for i, (s, _, _, _) in s_l_r_n.items()}
        l = [s_l_r_n[i][1] for i in sorted(t.keys())]
        w = [1 / t[i] for i in sorted(t.keys())]
        l_norm = jnp.average(jnp.array(l), weights=jnp.array(w))
        l_avg = sum(l) / len(l)

        r = jnp.sum(sum(r for (_, _, r, _) in s_l_r_n.values()))
        dec_diag = {
            i: self.f_dec(params['f_dec'],
                          self.split_state_emb(se)[1])
            for i, se in state_e.items()
        }
        return state_e, (l_norm, l_avg), r, n, sum(n.values()), dec_diag

    def __call__(self,
                 params: Any,
                 subjects_batch: List[int],
                 count_nfe: bool = False,
                 interval_norm: bool = False):
        nth_adm = partial(self._extract_nth_admission, params, subjects_batch)
        nn_ode = partial(self._f_n_ode, params, count_nfe)
        nn_update = partial(self._f_update, params)

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

            # To normalize the prediction loss by the number of patients
            adm_counts.append(len(adm_id))

            state_di = {i: subject_state[i]['state_e'] for i in adm_id}

            d2d_time = {
                i: adm_time[i] + adm_los[i] - subject_state[i]['time']
                for i in adm_id
            }

            # Integrate until next discharge
            state_dj, (l_norm, l_avg), r, nfe, nfe_sum, dec_diag = nn_ode(
                state_di, d2d_time, diag)

            for subject_id in state_dj.keys():
                diag_detectability[subject_id][n] = {
                    'admission_id': adm_id[subject_id],
                    'nfe': nfe[subject_id],
                    'time': adm_time[subject_id],
                    'true_diag': diag[subject_id],
                    'pred_logits': dec_diag[subject_id]
                }

            odeint_time.append(sum(d2d_time.values()))
            prediction_losses.append(l_avg)

            dyn_loss += r
            total_nfe += nfe_sum

            # Update state at discharge
            state_dj = nn_update(state_dj, emb)

            # Update the states:
            for subject_id, new_state in state_dj.items():
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
