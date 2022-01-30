from functools import partial
from typing import (Any, Callable, Dict, Iterable, List, Optional, Tuple, Set)

from absl import logging
import jax
import jax.numpy as jnp

import optuna

from .metrics import (l2_squared, l1_absolute)
from .jax_interface import (DiagnosisJAXInterface, create_patient_interface)
from .abstract_model import AbstractModel
from .gram import AbstractEmbeddingsLayer
from .train_icenode import ICENODE


class ICENODE2(ICENODE):
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

    def _f_n_ode_los(self, params, count_nfe, h, t):
        null = jnp.array([])
        c = {i: null for i in h}

        if any(ti <= 0 for ti in t.values()):
            logging.error('Zero time ODEINT')

        h_r_nfe = {
            i: self.f_n_ode(params['f_n_ode'], self.los_sample_rate,
                            count_nfe, h[i], t[i], c[i])
            for i in t.keys()
        }

        h = {i: h for i, (h, r, n) in h_r_nfe.items()}
        h_final = {i: hi[-1,:] for i, hi in h.items()}
        return h, h_final

    def _f_dec_los(
        self, params: Any, state: Dict[int, jnp.ndarray]
    ) -> Tuple[Dict[int, jnp.ndarray], Dict[int, jnp.ndarray]]:

        dec_out = {i: jax.vmap(partial(self.f_dec, params['f_dec']))(state[i]) for i in state}
        emb = {i: jnp.mean(g, axis=0) for i, (g, _) in dec_out.items()}
        out = {i: jnp.max(o, axis=0) for i, (_, o) in dec_out.items()}

        return emb, out



    def __call__(self,
                 params: Any,
                 subjects_batch: List[int],
                 count_nfe: bool = False):
        nth_adm = partial(self._extract_nth_admission, params, subjects_batch)
        # nODE between two admissions
        nn_ode = partial(self._f_n_ode, params, count_nfe)
        # nODE: within an admission (LoS: Legth of Stay)
        nn_ode_los = partial(self._f_n_ode_los, params, count_nfe)

        nn_update = partial(self._f_update, params)

        # For decoding just before admissions
        nn_decode = partial(self._f_dec, params)

        # For decoding over samples within a single admission
        nn_decode_los = partial(self._f_dec_los, params)

        nn_init = partial(self._f_init, params)
        diag_loss = self._diag_loss
        subject_state = dict()
        dyn_losses = []
        total_nfe = 0

        prediction_losses = []
        update_losses = []
        # For within LoS prediction loss after odeint.
        los_losses = []

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
            los_loss = 0.0
            if n == 0:
                # Initialize all states for first admission (no predictions)
                state_0 = nn_init(emb)
                subject_state.update(state_0)
                assert all(
                    i in subject_state for i in subjects_batch
                ), "Expected all subjects have at least one admission"
                state = {i: subject_state[i]['state'] for i in adm_id}
            else:
                # Between-admissions integration (and predictions)

                # time-difference betweel last discharge and current admission
                delta_days = {
                    i: (adm_time[i] - subject_state[i]['time'])
                    for i in adm_id
                }
                state = {i: subject_state[i]['state'] for i in adm_id}

                odeint_weeks += sum(delta_days.values()) / 7
                ################## ODEINT BETWEEN ADMS #####################
                state, dyn_loss, (nfe, nfe_sum) = nn_ode(state, delta_days)
                dyn_losses.append(dyn_loss)
                total_nfe += nfe_sum
                ########## DIAG LOSS #########################
                dec_emb, dec_diag = nn_decode(state)
                prediction_losses.append(diag_loss(diag, dec_diag))

                for subject_id in state.keys():
                    diag_detectability[subject_id][n] = {
                        'admission_id': adm_id[subject_id],
                        'nfe': nfe[subject_id],
                        'time': adm_time[subject_id],
                        'diag_true': diag[subject_id],
                        'pre_logits': dec_diag[subject_id]
                    }

            # Within-admission integration for the LoS
            ################## ODEINT BETWEEN ADMS #####################
            state_los, state = nn_ode_los(state, adm_los)
            dec_emb, dec_diag = nn_decode_los(state_los)
            los_loss += diag_loss(diag, dec_diag)


            state, update_loss = nn_update(state, emb, diag, dec_emb)
            update_losses.append(update_loss)
            los_losses.append(los_loss)

            # Update the states:
            for subject_id, new_state in state.items():
                subject_state[subject_id] = {
                    'time':
                    adm_time[subject_id] + adm_los[subject_id],
                    'state':
                    new_state
                }

        prediction_loss = jnp.average(prediction_losses,
                                      weights=adm_counts[1:])
        update_loss = jnp.average(update_losses, weights=adm_counts)
        los_loss = jnp.average(los_losses, weights=adm_counts)

        ret = {
            'prediction_loss': prediction_loss,
            'update_loss': update_loss,
            'los_loss': los_loss,
            'dyn_loss': jnp.sum(sum(dyn_losses)),
            'odeint_weeks': odeint_weeks,
            'admissions_count': sum(adm_counts),
            'nfe': total_nfe,
            'diag_detectability': diag_detectability
        }

        return ret

    def detailed_loss(self, loss_mixing, params, res):
        prediction_loss = res['prediction_loss']
        update_loss = res['update_loss']
        los_loss = res['los_loss']
        l1_loss = l1_absolute(params)
        l2_loss = l2_squared(params)
        dyn_loss = res['dyn_loss']
        pred_alpha = loss_mixing['L_pred']
        los_alpha = loss_mixing['L_los']
        update_alpha = loss_mixing['L_update']

        l1_alpha = loss_mixing['L_l1']
        l2_alpha = loss_mixing['L_l2']
        dyn_alpha = loss_mixing['L_dyn'] / (res['odeint_weeks'] + 1e-10)

        diag_loss = (pred_alpha * prediction_loss + update_alpha * update_loss +los_alpha * los_loss)

        loss = diag_loss + (l1_alpha * l1_loss) + (l2_alpha * l2_loss) + (dyn_alpha * dyn_loss)

        return {
            'prediction_loss': prediction_loss,
            'update_loss': update_loss,
            'los_loss': los_loss,
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

        diag_emb = cls.create_embedding(
            emb_config=config['emb']['diag'],
            emb_kind=config['emb']['kind'],
            patient_interface=patient_interface,
            train_ids=train_ids,
            pretrained_components=pretrained_components)

        diag_loss = cls.select_loss(config['training']['diag_loss'],
                                    patient_interface, train_ids)
        return cls(subject_interface=patient_interface,
                   diag_emb=diag_emb,
                   **config['model'],
                   diag_loss=diag_loss)

    @staticmethod
    def _sample_ode_training_config(trial: optuna.Trial, epochs):
        config = AbstractModel._sample_training_config(trial, epochs)
        # UNDO
        config['diag_loss'] = 'softmax'
        # trial.suggest_categorical('dx_loss', ['balanced_bce', 'softmax'])
        # config['diag_loss'] = trial.suggest_categorical(
        #     'dx_loss', ['balanced_focal', 'bce', 'softmax', 'balanced_bce'])

        config['loss_mixing'] = {
            'L_pred': trial.suggest_float('L_pred', 1e-4, 1, log=True),
            'L_los': trial.suggest_float('L_los', 1e-4, 1, log=True),
            'L_update': trial.suggest_float('L_update', 1e-4, 1, log=True),
            'L_dyn': trial.suggest_float('L_dyn', 1e-3, 1e3, log=True),
            **config['loss_mixing']
        }

        return config

    @staticmethod
    def sample_training_config(trial: optuna.Trial):
        return ICENODE._sample_ode_training_config(trial, epochs=10)

    @staticmethod
    def _sample_ode_model_config(trial: optuna.Trial):
        model_params = {
            'ode_dyn': trial.suggest_categorical(
                'ode_dyn', ['mlp', 'gru', 'res'
                            ]),  # Add depth conditional to 'mlp' or 'res'
            'ode_with_bias': False, # trial.suggest_categorical('ode_b', [True, False]),
            'ode_init_var': trial.suggest_float('ode_iv', 1e-4, 1e-1, log=True),
            'ode_timescale': trial.suggest_float('ode_ts', 1, 1e2, log=True),
            'los_sample_rate': trial.suggest_int('los_f', 1, 8),
            'state_size': trial.suggest_int('s', 30, 300, 30),
            'init_depth': trial.suggest_int('init_d', 1, 4),
            'tay_reg': trial.suggest_categorical('tay', [0, 2, 3, 4]),
        }
        if model_params['ode_dyn'] == 'gru':
            model_params['ode_depth'] = 0
        else:
            model_params['ode_depth'] = trial.suggest_int('ode_d', 1, 4)

        return model_params

    @staticmethod
    def sample_model_config(trial: optuna.Trial):
        return ICENODE._sample_ode_model_config(trial)


if __name__ == '__main__':
    from .hpo_utils import capture_args, run_trials
    run_trials(model_cls=ICENODE, **capture_args())
