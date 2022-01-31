from functools import partial
from typing import (Any, Callable, Dict, Iterable, List, Optional, Tuple, Set)

from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp

import optuna

from .metrics import (l2_squared, l1_absolute)
from .utils import wrap_module
from .jax_interface import (DiagnosisJAXInterface, create_patient_interface)
from .models import (MLPDynamics, ResDynamics, GRUDynamics, NeuralODE,
                     DiagnosesUpdate, EmbeddingsDecoder)
from .inn_models import InvertibleLayers

from .abstract_model import AbstractModel
from .gram import AbstractEmbeddingsLayer


@jax.jit
def loss(y: jnp.ndarray, p: jnp.ndarray):
    return -jnp.mean(y * jnp.log(p + 1e-10) + (1 - y) * jnp.log(1 - p + 1e-10))


class ICENODE_I(AbstractModel):

    def __init__(self, subject_interface: DiagnosisJAXInterface,
                 diag_emb: AbstractEmbeddingsLayer, ode_dyn: str,
                 ode_with_bias: bool, ode_init_var: float,
                 ode_timescale: float, tay_reg: Optional[int], state_size: int,
                 diag_loss: Callable[[jnp.ndarray, jnp.ndarray], float]):

        self.subject_interface = subject_interface
        self.diag_emb = diag_emb
        self.tay_reg = tay_reg
        self.diag_loss = diag_loss
        self.dimensions = {
            'diag_emb': diag_emb.embeddings_dim,
            'diag_in': len(subject_interface.diag_ccs_idx),
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

        f_n_ode_init, f_n_ode = hk.without_apply_rng(
            hk.transform(
                wrap_module(NeuralODE,
                            ode_dyn_cls=ode_dyn_cls,
                            state_size=state_size,
                            depth=2,
                            timescale=ode_timescale,
                            with_bias=ode_with_bias,
                            init_var=ode_init_var,
                            name='f_n_ode',
                            tay_reg=tay_reg)))
        f_n_ode = jax.jit(f_n_ode, static_argnums=(1, 2))
        self.f_n_ode = (
            lambda params, *args: f_n_ode(params['f_n_ode'], *args))

        f_update_init, f_update = hk.without_apply_rng(
            hk.transform(
                wrap_module(DiagnosesUpdate,
                            state_size=state_size,
                            name='f_update')))
        f_update = jax.jit(f_update)
        self.f_update = (
            lambda params, *args: f_update(params['f_update'], *args))

        f_dec1_init, f_dec1 = hk.without_apply_rng(
            hk.transform(
                wrap_module(InvertibleLayers,
                            n_layers=8,
                            size=state_size,
                            name='f_dec1')))
        f_dec1 = jax.jit(f_dec1, static_argnums=(1, ))
        self.f_dec1 = lambda params, h: f_dec1(params['f_dec1'], True, h)
        self.f_enc1 = lambda params, e: f_dec1(params['f_dec1'], False, e)

        f_dec2_init, f_dec2 = hk.without_apply_rng(
            hk.transform(
                wrap_module(EmbeddingsDecoder,
                            n_layers=2,
                            embeddings_size=self.dimensions['diag_emb'],
                            diag_size=self.dimensions['diag_out'],
                            name='f_dec2')))
        f_dec2 = jax.jit(f_dec2)
        self.f_dec2 = lambda params, *args: f_dec2(params['f_dec2'], *args)

        self.initializers = {
            'f_n_ode': f_n_ode_init,
            'f_update': f_update_init,
            'f_dec2': f_dec2_init,
            'f_dec1': f_dec1_init
        }

        self.init_data = self._initialization_data()

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
        diag_emb_ = jnp.zeros(self.dimensions['diag_emb'])
        state = jnp.zeros(self.dimensions['state'])
        ode_ctrl = jnp.array([])
        return {
            "f_n_ode": [2, True, state, 0.1, ode_ctrl],
            "f_update": [state, diag_emb_],
            "f_dec1": [True, state],
            "f_dec2": [diag_emb_]
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

    def _f_n_ode(self, params, count_nfe, h, t):
        null = jnp.array([])
        c = {i: null for i in h}

        h_r_nfe = {
            i: self.f_n_ode(params, 2, count_nfe, h[i], t[i], c[i])
            for i in t.keys()
        }
        nfe = {i: n for i, (h, r, n) in h_r_nfe.items()}
        nfe_sum = sum(nfe.values())
        drdt = jnp.sum(sum(r for (h, r, n) in h_r_nfe.values()))

        h_final = {i: h[-1, :] for i, (h, r, n) in h_r_nfe.items()}
        return h_final, (drdt, nfe, nfe_sum)

    def _f_update(self, params: Any, state: Dict[int, jnp.ndarray],
                  nominal_state: Dict[int, jnp.ndarray]) -> jnp.ndarray:
        delta1 = {i: nominal_state[i] - state[i] for i in nominal_state}

        updated_state = {
            i: self.f_update(params, state[i], d)
            for i, d in delta1.items()
        }

        delta2 = {i: nominal_state[i] - updated_state[i] for i in state}

        trajectory_loss = sum(
            jnp.mean(d1**2) + jnp.mean(d2**2)
            for d1, d2 in zip(delta1.values(), delta2.values()))

        return updated_state, trajectory_loss

    def _f_dec(self, params: Any, state: Dict[int, jnp.ndarray]):
        emb = {i: self.f_dec1(params, state[i]) for i in state}
        diag = {i: self.f_dec2(params, emb[i]) for i in state}
        return emb, diag

    def _diag_loss(self, diag_true: Dict[int, jnp.ndarray],
                   diag_predicted: Dict[int, jnp.ndarray]):
        loss = {
            i: self.diag_loss(diag_true[i], diag_predicted[i])
            for i in diag_predicted.keys()
        }
        if loss:
            return sum(loss.values()) / len(loss)
        else:
            return 0.0

    def __call__(self,
                 params: Any,
                 subjects_batch: List[int],
                 count_nfe: bool = False):
        nth_adm = partial(self._extract_nth_admission, params, subjects_batch)
        nn_ode = partial(self._f_n_ode, params, count_nfe)
        nn_update = partial(self._f_update, params)
        nn_decode = partial(self._f_dec, params)

        diag_loss = self._diag_loss
        subject_state = {}
        dyn_losses = []
        total_nfe = 0

        prediction_losses = []
        trajectory_losses = []

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

            h_n = {i: self.f_enc1(params, emb[i]) for i in emb}

            # To normalize the prediction loss by the number of patients
            adm_counts.append(len(adm_id))

            update_loss = 0.0
            if n == 0:
                subject_state = {i: {'time': 0, 'state': h_n[i]} for i in h_n}
                state0 = {i: subject_state[i]['state'] for i in adm_id}
                # Integrate until first discharge
                state, (drdt, _, nfe_sum) = nn_ode(state0, adm_los)
                _, dec_diag = nn_decode(state)
                odeint_weeks += sum(adm_los.values()) / 7

            else:
                state = {i: subject_state[i]['state'] for i in adm_id}

                d2d_time = {
                    i: adm_time[i] + adm_los[i] - subject_state[i]['time']
                    for i in adm_id
                }

                # Integrate until next discharge
                state, (drdt, nfe, nfe_sum) = nn_ode(state, d2d_time)
                dec_emb, dec_diag = nn_decode(state)

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
            state, trajectory_loss = nn_update(state, h_n)
            trajectory_losses.append(trajectory_loss)

            # Update the states:
            for subject_id, new_state in state.items():
                subject_state[subject_id] = {
                    'time': adm_time[subject_id] + adm_los[subject_id],
                    'state': new_state
                }

        prediction_loss = jnp.average(prediction_losses,
                                      weights=adm_counts[1:])
        trajectory_loss = jnp.average(trajectory_losses, weights=adm_counts)

        ret = {
            'prediction_loss': prediction_loss,
            'trajectory_loss': trajectory_loss,
            'dyn_loss': jnp.sum(sum(dyn_losses)),
            'odeint_weeks': odeint_weeks,
            'admissions_count': sum(adm_counts),
            'nfe': total_nfe,
            'diag_detectability': diag_detectability
        }

        return ret

    def detailed_loss(self, loss_mixing, params, res):
        prediction_loss = res['prediction_loss']
        trajectory_loss = res['trajectory_loss']
        l1_loss = l1_absolute(params)
        l2_loss = l2_squared(params)
        dyn_loss = res['dyn_loss']
        pred_alpha = loss_mixing['L_pred']
        traj_alpha = loss_mixing['L_traj']

        l1_alpha = loss_mixing['L_l1']
        l2_alpha = loss_mixing['L_l2']
        dyn_alpha = loss_mixing['L_dyn'] / (res['odeint_weeks'] + 1e-10)

        diag_loss = (pred_alpha * prediction_loss +
                     traj_alpha * trajectory_loss)

        loss = diag_loss + (l1_alpha * l1_loss) + (l2_alpha * l2_loss) + (
            dyn_alpha * dyn_loss)

        return {
            'prediction_loss': prediction_loss,
            'trajectory_loss': trajectory_loss,
            'diag_loss': diag_loss,
            'loss': loss,
            'l1_loss': l1_loss,
            'l2_loss': l2_loss,
            'dyn_loss': dyn_loss,
            'dyn_loss_per_week': dyn_loss / (res['odeint_weeks'] + 1e-10)
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

        # Equate embeddings size with state size.
        config['emb']['diag']['embeddings_dim'] = config['model']['state_size']

        diag_emb = cls.create_embedding(
            emb_config=config['emb']['diag'],
            emb_kind=config['emb']['kind'],
            patient_interface=patient_interface,
            train_ids=train_ids,
            pretrained_components=pretrained_components)
        diag_loss = loss
        return cls(subject_interface=patient_interface,
                   diag_emb=diag_emb,
                   **config['model'],
                   diag_loss=diag_loss)

    @staticmethod
    def _sample_ode_training_config(trial: optuna.Trial, epochs):
        config = AbstractModel._sample_training_config(trial, epochs)
        config['loss_mixing'] = {
            'L_pred': trial.suggest_float('L_pred', 1e-4, 1, log=True),
            'L_traj': trial.suggest_float('L_traj', 1e-4, 1, log=True),
            'L_dyn': trial.suggest_float('L_dyn', 1e-3, 1e3, log=True),
            **config['loss_mixing']
        }

        return config

    @staticmethod
    def sample_training_config(trial: optuna.Trial):
        return ICENODE_I._sample_ode_training_config(trial, epochs=20)

    @staticmethod
    def _sample_ode_model_config(trial: optuna.Trial):
        model_params = {
            'ode_dyn': trial.suggest_categorical('ode_dyn', ['mlp', 'gru']),
            'ode_with_bias': False,
            'ode_init_var': 1e-2,
            'ode_timescale': trial.suggest_float('ode_ts', 1, 1e2, log=True),
            'state_size': trial.suggest_int('s', 30, 300, 30),
            'tay_reg': trial.suggest_categorical('tay', [0, 2, 3, 4]),
        }
        return model_params

    @staticmethod
    def sample_model_config(trial: optuna.Trial):
        return ICENODE_I._sample_ode_model_config(trial)


if __name__ == '__main__':
    from .hpo_utils import capture_args, run_trials
    run_trials(model_cls=ICENODE_I, **capture_args())
