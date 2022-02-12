from functools import partial
from typing import (Any, Dict, Iterable, List)

import haiku as hk
import jax
from jax import lax
import jax.numpy as jnp
import numpy as onp

import optuna

from .jax_interface import DiagnosisJAXInterface
from .metrics import (softmax_logits_bce)
from .utils import wrap_module
from .models import (MLPDynamics, ResDynamics, GRUDynamics, NeuralODE,
                     EmbeddingsDecoder_Logits)
from .gram import AbstractEmbeddingsLayer
from .train_icenode_tl import ICENODE as ICENODE_TL


class ICENODE(ICENODE_TL):

    def __init__(self, subject_interface: DiagnosisJAXInterface,
                 diag_emb: AbstractEmbeddingsLayer, ode_dyn: str,
                 ode_with_bias: bool, ode_init_var: float, memory_size: int,
                 timescale: float):

        self.subject_interface = subject_interface
        self.diag_emb = diag_emb
        self.dimensions = {
            'diag_emb': diag_emb.embeddings_dim,
            'diag_out': len(subject_interface.diag_ccs_idx),
            'memory': memory_size
        }
        if ode_dyn == 'gru':
            ode_dyn_cls = GRUDynamics
        elif ode_dyn == 'res':
            ode_dyn_cls = ResDynamics
        elif ode_dyn == 'mlp':
            ode_dyn_cls = MLPDynamics
        else:
            raise RuntimeError(f"Unrecognized dynamics class: {ode_dyn}")

        f_n_ode_init, f_n_ode = hk.without_apply_rng(
            hk.transform(
                wrap_module(NeuralODE,
                            ode_dyn_cls=ode_dyn_cls,
                            state_size=self.dimensions['diag_emb'],
                            depth=1,
                            timescale=timescale,
                            with_bias=ode_with_bias,
                            init_var=ode_init_var,
                            name='f_n_ode',
                            tay_reg=3)))
        self.f_n_ode = jax.jit(f_n_ode, static_argnums=(1, 2))

        f_update_init, f_update = hk.without_apply_rng(
            hk.transform(
                wrap_module(hk.LSTM, hidden_size=memory_size,
                            name='f_update')))
        self.f_update = jax.jit(f_update)

        f_dec_init, f_dec = hk.without_apply_rng(
            hk.transform(
                wrap_module(EmbeddingsDecoder_Logits,
                            n_layers=2,
                            embeddings_size=self.dimensions['diag_emb'],
                            diag_size=self.dimensions['diag_out'],
                            name='f_dec')))
        self.f_dec = jax.jit(f_dec)

        self.initializers = {
            'f_n_ode': f_n_ode_init,
            'f_update': f_update_init,
            'f_dec': f_dec_init
        }
        self.init_data = self._initialization_data()

    def init_params(self, prng_seed=0):
        rng_key = jax.random.PRNGKey(prng_seed)
        return {
            "diag_emb": self.diag_emb.init_params(rng_key),
            **{
                label: init(rng_key, *self.init_data[label])
                for label, init in self.initializers.items()
            }
        }

    def _initialization_data(self):
        """
        Creates data for initializing each of the
        modules based on the shapes of init_data.
        """
        emb = jnp.zeros(self.dimensions['diag_emb'])
        memory = jnp.zeros(self.dimensions['memory'])
        lstm_state = hk.LSTMState(hidden=memory, cell=memory)
        return {
            "f_n_ode": [2, True, emb, 0.1, memory],
            "f_update": [emb, lstm_state],
            "f_dec": [emb],
        }

    def _f_update(self,
                  params: Any,
                  emb: jnp.ndarray,
                  lstm_state: Dict[int, jnp.ndarray] = None) -> jnp.ndarray:
        if lstm_state is None:
            mem0 = jnp.zeros(self.dimensions['memory'])
            state0 = hk.LSTMState(mem0, mem0)
            lstm_state = {i: state0 for i in emb}
        return {
            i: self.f_update(params['f_update'], emb[i], lstm_state[i])
            for i in emb
        }

    def _f_dec(self, params: Any, emb: Dict[int, jnp.ndarray]):
        return {i: self.f_dec(params['f_dec'], e) for i, e in emb.items()}

    def _diag_loss(self, diag: Dict[int, jnp.ndarray],
                   dec_diag: Dict[int, jnp.ndarray]):
        l = [
            softmax_logits_bce(diag[i], dec_diag[i])
            for i in sorted(diag.keys())
        ]
        return sum(l) / len(l)

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
        lstm0 = nn_update(adm0['diag_emb'])
        subject_state = {
            i: {
                'lstm': lstm0[i][1],
                'mem': lstm0[i][0],
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

            adm_counts.append(len(adm_id))

            emb = {i: subject_state[i]['emb'] for i in adm_id}
            mem = {i: subject_state[i]['mem'] for i in adm_id}
            lstm = {i: subject_state[i]['lstm'] for i in adm_id}

            d2d_time = {
                i: adm_time[i] + adm_los[i] - subject_state[i]['time']
                for i in adm_id
            }

            # Integrate until next discharge
            emb, r, nfe = nn_ode(emb, d2d_time, mem)
            pred_diag = nn_decode(emb)
            odeint_time.append(sum(d2d_time.values()))
            dyn_loss += sum(r.values())

            true_diag = adm_n['diag_out']
            prediction_losses.append(diag_loss(true_diag, pred_diag))

            total_nfe += sum(nfe.values())

            # Update memory at discharge
            true_emb = adm_n['diag_emb']
            lstm = nn_update(true_emb, lstm)

            for subject_id in emb.keys():
                diag_detectability[subject_id][n] = {
                    'admission_id': adm_id[subject_id],
                    'nfe': nfe[subject_id],
                    'time': adm_time[subject_id] + adm_los[subject_id],
                    'los': adm_los[subject_id],
                    'R/T': r[subject_id] / d2d_time[subject_id],
                    'true_diag': true_diag[subject_id],
                    'pred_logits': pred_diag[subject_id]
                }

            # Update the states:
            for subject_id in emb:
                subject_state[subject_id] = {
                    'time': adm_time[subject_id] + adm_los[subject_id],
                    'emb': emb[subject_id],
                    'mem': lstm[subject_id][0],
                    'lstm': lstm[subject_id][1]
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

    def _f_n_ode_trajectory(self, params, sampling_rate, emb, t_offset, t,
                            mem):
        nn_decode = partial(self._f_dec, params)

        def timesamples(tf, dt):
            return jnp.linspace(0, tf - tf % dt,
                                round((tf - tf % dt) / dt + 1))

        def odeint_samples(carry, t_ij):
            current_emb, current_mem = carry
            h, _, _ = self.f_n_ode(params['f_n_ode'], 2, False, current_emb,
                                   sampling_rate, current_mem)
            next_emb = h[-1, :]
            return (next_emb, current_mem), next_emb

        trajectory_samples = {}
        new_emb = {}
        for i, ti in t.items():
            t_samples = timesamples(ti, sampling_rate) + t_offset[i]
            current_emb, e_samples = lax.scan(odeint_samples, emb[i],
                                              t_samples[1:], emb[i])
            new_emb[i] = current_emb
            # diagnostic samples
            d_samples = jax.vmap(partial(self.f_dec,
                                         params['f_dec']))(e_samples)
            # convert from logits to probs.
            d_samples = jax.vmap(jax.nn.softmax)(d_samples)
            trajectory_samples[i] = {
                't': t_samples,
                'e': e_samples,
                'd': d_samples
            }

        return new_emb, trajectory_samples

    def sample_trajectory(self, model_state, batch: List[int],
                          sample_rate: float):
        params = self.get_params(model_state)
        nth_adm = partial(self._extract_nth_admission, params, batch)
        nn_ode = partial(self._f_n_ode_trajectory, params, sample_rate)
        nn_update = partial(self._f_update, params)

        adm0 = nth_adm(0)
        mem0 = nn_update(None, adm0['diag_emb'])
        subject_state = {
            i: {
                'mem': mem0[i],
                'emb': adm0['diag_emb'][i],
                'time': adm0['time'][i] + adm0['los'][i]
            }
            for i in adm0['admission_id']
        }

        trajectory = {
            i: {
                't': [],
                'e': [],
                'd': [],
                'tp10': []
            }
            for i in batch
        }

        for n in self.subject_interface.n_support[1:]:
            adm_n = nth_adm(n)
            if adm_n is None:
                break
            adm_id = adm_n['admission_id']
            adm_los = adm_n['los']  # length of stay
            adm_time = adm_n['time']
            true_emb = adm_n['diag_emb']
            true_diag = adm_n['diag_out']

            emb = {i: subject_state[i]['emb'] for i in adm_id}
            mem = {i: subject_state[i]['mem'] for i in adm_id}

            d2d_time = {
                i: adm_time[i] + adm_los[i] - subject_state[i]['time']
                for i in adm_id
            }

            offset = {i: subject_state[i]['time'] for i in adm_id}

            # Integrate until next discharge
            emb, traj_n = nn_ode(emb, offset, d2d_time, mem)

            for subject_id, traj_ni in traj_n.items():
                for symbol in ('t', 'e', 'd'):
                    trajectory[subject_id][symbol].append(traj_ni[symbol])

                # For the last timestamp, get sorted indices for the predictions
                top10_idx = jnp.argsort(-traj_ni['d'][-1, :])[:10]
                pos = onp.zeros_like(true_diag[subject_id])
                pos[[top10_idx]] = 1
                tp = (pos == true_diag[subject_id]) * 1
                t = traj_ni['t']

                trajectory[subject_id]['tp10'].append(jnp.vstack([tp] *
                                                                 len(t)))

            # Update state at discharge
            mem = nn_update(mem, true_emb)

            # Update the states:
            for subject_id in emb:
                subject_state[subject_id] = {
                    'time': adm_time[subject_id] + adm_los[subject_id],
                    'emb': emb[subject_id],
                    'mem': mem[subject_id]
                }

        for i, traj_i in trajectory.items():
            for symbol in ('t', 's', 'e', 'd', 'd1d', 'd2d', 'tp10'):
                traj_i[symbol] = jnp.concatenate(traj_i[symbol], axis=0)

        return trajectory

    @classmethod
    def sample_model_config(cls, trial: optuna.Trial):
        return {
            'ode_dyn': trial.suggest_categorical('ode_dyn', ['mlp', 'gru']),
            'ode_with_bias': False,
            'ode_init_var': trial.suggest_float('ode_i', 1e-9, 1, log=True),
            'memory_size': trial.suggest_int('s', 10, 100, 10),
            'timescale': 7
        }


if __name__ == '__main__':
    from .hpo_utils import capture_args, run_trials
    run_trials(model_cls=ICENODE, **capture_args())
