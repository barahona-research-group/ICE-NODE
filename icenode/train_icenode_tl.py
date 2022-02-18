from functools import partial
from typing import (Any, Dict, Iterable, List, Optional, Set)

import haiku as hk
import jax
from jax import lax
import jax.numpy as jnp
import numpy as onp

import optuna

from .metrics import (balanced_focal_bce, admissions_auc_scores)
from .utils import wrap_module
from .jax_interface import (DiagnosisJAXInterface, create_patient_interface)
from .models import (MLPDynamics, ResDynamics, GRUDynamics, NeuralODE,
                     EmbeddingsDecoder_Logits, StateUpdate)
from .abstract_model import AbstractModel
from .gram import AbstractEmbeddingsLayer


class ICENODE(AbstractModel):

    def __init__(self, subject_interface: DiagnosisJAXInterface,
                 diag_emb: AbstractEmbeddingsLayer, ode_dyn: str,
                 ode_with_bias: bool, ode_init_var: float, state_size: int,
                 timescale: float):

        self.subject_interface = subject_interface
        self.diag_emb = diag_emb
        self.timescale = timescale
        self.dimensions = {
            'diag_emb': diag_emb.embeddings_dim,
            'diag_out': len(subject_interface.diag_ccs_idx),
            'state': state_size
        }
        depth = 2
        if ode_dyn == 'gru':
            ode_dyn_cls = GRUDynamics
        elif ode_dyn == 'res':
            ode_dyn_cls = ResDynamics
        elif ode_dyn == 'mlp2':
            ode_dyn_cls = MLPDynamics
            depth = 2
        elif ode_dyn == 'mlp3':
            ode_dyn_cls = MLPDynamics
            depth = 3
        else:
            raise RuntimeError(f"Unrecognized dynamics class: {ode_dyn}")
        state_emb_size = self.dimensions['diag_emb'] + state_size

        f_n_ode_init, f_n_ode = hk.without_apply_rng(
            hk.transform(
                wrap_module(NeuralODE,
                            ode_dyn_cls=ode_dyn_cls,
                            state_size=state_emb_size,
                            depth=depth,
                            timescale=timescale,
                            with_bias=ode_with_bias,
                            init_var=ode_init_var,
                            name='f_n_ode',
                            tay_reg=3)))
        self.f_n_ode = jax.jit(f_n_ode, static_argnums=(1, 2))

        f_update_init, f_update = hk.without_apply_rng(
            hk.transform(
                wrap_module(StateUpdate,
                            state_size=state_size,
                            embeddings_size=self.dimensions['diag_emb'],
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

    def join_state_emb(self, state, emb):
        if state is None:
            state = jnp.zeros((self.dimensions['state'], ))
        return jnp.hstack((state, emb))

    def split_state_emb(self, state_emb):
        return jnp.split(state_emb, (self.dimensions['state'], ))

    def init_params(self, prng_seed=0):
        rng_key = jax.random.PRNGKey(prng_seed)
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
        return {
            "f_n_ode": [2, True, state_emb, 0.1],
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

    def _f_n_ode_nfe(self, params, count_nfe, emb, t, c):
        if count_nfe:
            return {
                i: self.f_n_ode(params['f_n_ode'], 2, True, emb[i], t[i],
                                c.get(i))[-1]
                for i in t
            }
        else:
            return {i: 0 for i in t}

    def _f_n_ode(self, params, count_nfe, state, t, c={}):
        s_r_nfe = {
            i: self.f_n_ode(params['f_n_ode'], 2, False, state[i], t[i],
                            c.get(i))
            for i in t
        }
        n = self._f_n_ode_nfe(params, count_nfe, state, t, c)
        r = {i: r.squeeze() for i, (_, r, _) in s_r_nfe.items()}
        s = {i: s[-1, :] for i, (s, _, _) in s_r_nfe.items()}
        return s, r, n

    def _f_update(self, params: Any, state_e: Dict[int, jnp.ndarray],
                  emb: jnp.ndarray) -> jnp.ndarray:
        new_state = {}
        for i in emb:
            emb_nominal = emb[i]
            state, emb_pred = self.split_state_emb(state_e[i])
            state = self.f_update(params['f_update'], state, emb_pred,
                                  emb_nominal)
            new_state[i] = self.join_state_emb(state, emb_nominal)
        return new_state

    def _f_dec(self, params: Any, state_e: Dict[int, jnp.ndarray]):
        emb = {i: self.split_state_emb(state_e[i])[1] for i in state_e}
        return {i: self.f_dec(params['f_dec'], emb[i]) for i in emb}

    def _diag_loss(self, diag: Dict[int, jnp.ndarray],
                   dec_diag: Dict[int, jnp.ndarray]):
        l = [
            balanced_focal_bce(diag[i], dec_diag[i])
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

    def _f_n_ode_trajectory(self, params, sampling_rate, state_e, t_offset, t):
        nn_decode = partial(self._f_dec, params)

        def timesamples(tf, dt):
            return jnp.linspace(0, tf - tf % dt,
                                round((tf - tf % dt) / dt + 1))

        def odeint_samples(current_state, t_ij):
            h, _, _ = self.f_n_ode(params['f_n_ode'], 2, False, current_state,
                                   sampling_rate)
            next_state = h[-1, :]
            return next_state, next_state

        trajectory_samples = {}
        new_se = {}
        for i, ti in t.items():
            t_samples = timesamples(ti, sampling_rate) + t_offset[i]
            current_se, se_samples = lax.scan(odeint_samples, state_e[i],
                                              t_samples[1:])
            new_se[i] = current_se
            s, e = self.split_state_emb(state_e[i])
            s_samples = [s]
            e_samples = [e]
            for se_sample in se_samples:
                s, e = self.split_state_emb(se_sample)
                s_samples.append(s)
                e_samples.append(e)

            # state samples
            s_samples = jnp.vstack(s_samples)
            # embedding samples
            e_samples = jnp.vstack(e_samples)
            # diagnostic samples
            d_samples = jax.vmap(partial(self.f_dec,
                                         params['f_dec']))(e_samples)
            # convert from logits to probs.
            d_samples = jax.vmap(jax.nn.sigmoid)(d_samples)

            # 1st-order derivative of d_samples
            grad = jax.vmap(lambda v: jnp.gradient(v, sampling_rate))
            d1d_samples = grad(d_samples)

            # 2nd-order derivative of d_samples
            d2d_samples = grad(d1d_samples)

            trajectory_samples[i] = {
                't': t_samples,
                's': s_samples,
                'e': e_samples,
                'd': d_samples,
                'd1d': d1d_samples,
                'd2d': d2d_samples
            }

        return new_se, trajectory_samples

    def sample_trajectory(self, model_state, batch: List[int],
                          sample_rate: float):
        params = self.get_params(model_state)
        nth_adm = partial(self._extract_nth_admission, params, batch)
        nn_ode = partial(self._f_n_ode_trajectory, params, sample_rate)
        nn_update = partial(self._f_update, params)

        adm0 = nth_adm(0)
        subject_state = {
            i: {
                'state_e': self.join_state_emb(None, adm0['diag_emb'][i]),
                'time': adm0['time'][i] + adm0['los'][i]
            }
            for i in adm0['admission_id']
        }

        trajectory = {
            i: {
                't': [],
                'e': [],
                'd': [],
                'tp10': [],
                'fp10': [],
                's': [],
                'd1d': [],
                'd2d': []
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
            emb = adm_n['diag_emb']
            diag = adm_n['diag_out']

            state_e = {i: subject_state[i]['state_e'] for i in adm_id}

            d2d_time = {
                i: adm_time[i] + adm_los[i] - subject_state[i]['time']
                for i in adm_id
            }

            offset = {i: subject_state[i]['time'] for i in adm_id}

            # Integrate until next discharge
            state_e, traj_n = nn_ode(state_e, offset, d2d_time)

            for subject_id, traj_ni in traj_n.items():
                for symbol in ('t', 's', 'e', 'd', 'd1d', 'd2d'):
                    trajectory[subject_id][symbol].append(traj_ni[symbol])

                # For the last timestamp, get sorted indices for the predictions
                top10_idx = jnp.argsort(-traj_ni['d'][-1, :])[:10]
                pos = onp.zeros_like(diag[subject_id])
                pos[[top10_idx]] = 1
                tp = (pos == diag[subject_id]) * 1
                fp = (pos != diag[subject_id]) * 1
                trajectory[subject_id]['tp10'].append(tp)
                trajectory[subject_id]['fp10'].append(fp)

            # Update state at discharge
            state_e = nn_update(state_e, emb)

            # Update the states:
            for subject_id, new_state in state_e.items():
                subject_state[subject_id] = {
                    'time': adm_time[subject_id] + adm_los[subject_id],
                    'state_e': new_state
                }

        for i, traj_i in trajectory.items():
            for symbol in ('t', 's', 'e', 'd', 'd1d', 'd2d', 'tp10', 'fp10'):
                traj_i[symbol] = jnp.concatenate(traj_i[symbol], axis=0)

        return trajectory

    def detailed_loss(self, loss_mixing, params, res):
        prediction_loss = res['prediction_loss']
        dyn_loss = res['dyn_loss']
        dyn_alpha = loss_mixing['L_dyn']

        loss = prediction_loss + (dyn_alpha * dyn_loss) / res['odeint_weeks']
        return {
            'loss': loss,
            'prediction_loss': res['prediction_loss'],
            'dyn_loss': dyn_loss,
            'admissions_count': res['admissions_count'],
            'odeint_weeks': res['odeint_weeks']
        }

    def eval_stats(self, res):
        nfe = res['nfe']
        return {
            'admissions_count': res['admissions_count'],
            'nfe_per_week': nfe / res['odeint_weeks'],
            'Kfe': nfe / 1000
        }

    def eval(self, model_state: Any, batch: List[int]) -> Dict[str, float]:
        loss_mixing = model_state[-1]
        params = self.get_params(model_state)
        res = self(params, batch, count_nfe=True)

        return {
            'loss': self.detailed_loss(loss_mixing, params, res),
            'stats': self.eval_stats(res),
            'diag_detectability': res['diag_detectability']
        }

    def admissions_auc_scores(self, model_state: Any, batch: List[int]):
        params = self.get_params(model_state)
        res = self(params, batch, count_nfe=True)
        return admissions_auc_scores(res['diag_detectability'])

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

    @classmethod
    def sample_training_config(cls, trial: optuna.Trial):
        return {
            'epochs': 60,
            'batch_size': 2**trial.suggest_int('Bexp', 1, 8),
            #trial.suggest_int('B', 2, 27, 5),
            'optimizer': 'adam',
            #trial.suggest_categorical('opt', ['adam', 'adamax']),
            'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
            'decay_rate': trial.suggest_float('dr', 1e-1, 9e-1),
            'loss_mixing': {
                'L_l1': 0,  #trial.suggest_float('l1', 1e-8, 5e-3, log=True),
                'L_l2': 0,  # trial.suggest_float('l2', 1e-8, 5e-3, log=True),
                'L_dyn': 1e3  # trial.suggest_float('L_dyn', 1e-6, 1, log=True)
            }
        }

    @classmethod
    def sample_model_config(cls, trial: optuna.Trial):
        return {
            'ode_dyn':
            trial.suggest_categorical('ode_dyn', ['mlp2', 'mlp3']),
            'ode_with_bias':
            False,
            'ode_init_var':
            trial.suggest_float('ode_i', 1e-9, 1e1, log=True),
            'state_size':
            trial.suggest_int('s', 10, 100, 10),
            'timescale':
            7
        }


if __name__ == '__main__':
    from .hpo_utils import capture_args, run_trials
    run_trials(model_cls=ICENODE, **capture_args())
