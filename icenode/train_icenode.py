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
                     DiagnosesUpdate, StateDiagnosesDecoder, StateInitializer,
                     DiagnosticSamplesCombine)
from .abstract_model import AbstractModel
from .gram import AbstractEmbeddingsLayer


class ICENODE(AbstractModel):

    def __init__(self, subject_interface: DiagnosisJAXInterface,
                 diag_emb: AbstractEmbeddingsLayer, ode_dyn: str,
                 ode_depth: int, ode_with_bias: bool, ode_init_var: float,
                 ode_timescale: float, trajectory_sample_rate: int,
                 diag_seq_combiner: str, tay_reg: Optional[int],
                 state_size: int, init_depth: bool,
                 diag_loss: Callable[[jnp.ndarray, jnp.ndarray], float]):

        self.subject_interface = subject_interface
        self.diag_emb = diag_emb
        self.tay_reg = tay_reg
        self.diag_loss = diag_loss
        self.trajectory_sample_rate = trajectory_sample_rate
        self.dimensions = {
            'diag_emb': diag_emb.embeddings_dim,
            'diag_in': len(subject_interface.diag_ccs_idx),
            'diag_out': len(subject_interface.diag_ccs_idx),
            'state': state_size,
            'ode_depth': ode_depth,
            'init_depth': init_depth
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
                            depth=ode_depth,
                            timescale=ode_timescale,
                            with_bias=ode_with_bias,
                            init_var=ode_init_var,
                            name='f_n_ode',
                            tay_reg=tay_reg)))
        self.f_n_ode = jax.jit(f_n_ode, static_argnums=(1, 2))

        f_update_init, f_update = hk.without_apply_rng(
            hk.transform(
                wrap_module(DiagnosesUpdate,
                            state_size=state_size,
                            name='f_update')))
        self.f_update = jax.jit(f_update)

        f_dec_init, f_dec = hk.without_apply_rng(
            hk.transform(
                wrap_module(StateDiagnosesDecoder,
                            n_layers_d1=3,
                            n_layers_d2=2,
                            embeddings_size=self.dimensions['diag_emb'],
                            diag_size=self.dimensions['diag_out'],
                            name='f_dec')))
        self.f_dec = jax.jit(f_dec)

        f_init_init, f_init = hk.without_apply_rng(
            hk.transform(
                wrap_module(StateInitializer,
                            hidden_size=self.dimensions['diag_emb'],
                            state_size=state_size,
                            depth=init_depth,
                            name='f_init')))
        self.f_init = jax.jit(f_init)

        self.initializers = {
            'f_n_ode': f_n_ode_init,
            'f_update': f_update_init,
            'f_dec': f_dec_init,
            'f_init': f_init_init
        }

        self.init_data = self._initialization_data()

        if diag_seq_combiner == 'last':
            self._f_dec_seq = self._f_dec_last_combine
        elif diag_seq_combiner == 'mean':
            self._f_dec_seq = self._f_dec_mean_combine
        elif diag_seq_combiner == 'max':
            self._f_dec_seq = self._f_dec_max_combine
        elif diag_seq_combiner == 'att':
            self._f_dec_seq = self._f_dec_att_combine

            f_combine_init, f_combine = hk.without_apply_rng(
                hk.transform(
                    wrap_module(DiagnosticSamplesCombine,
                                embeddings_size=self.dimensions['diag_emb'],
                                name='f_combine')))
            self.f_combine = jax.jit(f_combine)

            self.initializers['f_combine'] = f_combine_init
            self.init_data['f_combine'] = [
                jnp.zeros((3, self.dimensions['diag_emb'])),
                jnp.zeros((3, self.dimensions['diag_out']))
            ]

        else:
            raise ValueError(
                f'Unrecognized combiner label: {diag_seq_combiner}')

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
            "f_dec": [state],
            "f_init": [diag_emb_]
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
            i: self.f_n_ode(params['f_n_ode'], self.trajectory_sample_rate,
                            count_nfe, h[i], t[i], c[i])
            for i in t.keys()
        }
        nfe = {i: n for i, (h, r, n) in h_r_nfe.items()}
        nfe_sum = sum(nfe.values())
        drdt = jnp.sum(sum(r for (h, r, n) in h_r_nfe.values()))

        h_samples = {i: h for i, (h, r, n) in h_r_nfe.items()}
        h_final = {i: hi[-1, :] for i, hi in h_samples.items()}
        return h_samples, h_final, (drdt, nfe, nfe_sum)

    def _f_update(self, params: Any, state: Dict[int, jnp.ndarray],
                  diag_emb: jnp.ndarray, diag_out: jnp.ndarray,
                  decoded_emb: jnp.ndarray) -> jnp.ndarray:
        delta_emb = self._emb_error(diag_emb, decoded_emb)
        updated_state = {
            i: self.f_update(params['f_update'], state[i], delta)
            for i, delta in delta_emb.items()
        }
        _, post_diag_out = self._f_dec(params, updated_state)
        update_loss = self._diag_loss(diag_out, post_diag_out)

        return updated_state, update_loss

    def _f_dec(self, params: Any, state: Dict[int, jnp.ndarray]):
        dec = {i: self.f_dec(params['f_dec'], state[i]) for i in state}
        emb = {i: e for i, (e, d) in dec.items()}
        diag = {i: d for i, (e, d) in dec.items()}
        return emb, diag

    def _f_dec_att_combine(
        self, params: Any, state_seq: Dict[int, jnp.ndarray]
    ) -> Tuple[Dict[int, jnp.ndarray], Dict[int, jnp.ndarray]]:

        dec_seq = {
            i: jax.vmap(partial(self.f_dec, params['f_dec']))(state)
            for i, state in state_seq.items()
        }
        dec = {
            i: self.f_combine(params['f_combine'], e_seq, d_seq)
            for i, (e_seq, d_seq) in dec_seq.items()
        }
        emb = {i: e for i, (e, d) in dec.items()}
        diag = {i: d for i, (e, d) in dec.items()}
        return emb, diag

    def _f_dec_mean_combine(
        self, params: Any, state_seq: Dict[int, jnp.ndarray]
    ) -> Tuple[Dict[int, jnp.ndarray], Dict[int, jnp.ndarray]]:

        dec_seq = {
            i: jax.vmap(partial(self.f_dec, params['f_dec']))(state)
            for i, state in state_seq.items()
        }

        emb = {i: jnp.mean(e_seq, axis=0) for i, (e_seq, _) in dec_seq.items()}

        diag = {
            i: jnp.mean(d_seq, axis=0)
            for i, (_, d_seq) in dec_seq.items()
        }

        return emb, diag

    def _f_dec_max_combine(
        self, params: Any, state_seq: Dict[int, jnp.ndarray]
    ) -> Tuple[Dict[int, jnp.ndarray], Dict[int, jnp.ndarray]]:

        dec_seq = {
            i: jax.vmap(partial(self.f_dec, params['f_dec']))(state)
            for i, state in state_seq.items()
        }

        emb = {i: jnp.mean(e_seq, axis=0) for i, (e_seq, _) in dec_seq.items()}

        diag = {i: jnp.max(d_seq, axis=0) for i, (_, d_seq) in dec_seq.items()}

        return emb, diag

    def _f_dec_last_combine(
        self, params: Any, state_seq: Dict[int, jnp.ndarray]
    ) -> Tuple[Dict[int, jnp.ndarray], Dict[int, jnp.ndarray]]:
        dec = {
            i: self.f_dec(params['f_dec'], state_seq[i][-1, :])
            for i in state_seq
        }
        emb = {i: e for i, (e, d) in dec.items()}
        diag = {i: d for i, (e, d) in dec.items()}

        return emb, diag

    def _f_init(self, params, diag_emb):
        return {
            i: {
                'time': 0,
                'state': self.f_init(params['f_init'], diag_emb[i])
            }
            for i in diag_emb
        }

    @staticmethod
    def _emb_error(emb_true, emb_predicted):
        error_emb = {
            i: emb_true[i] - emb_predicted[i]
            for i in emb_predicted.keys()
        }
        return error_emb

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

        # For decoding over samples
        nn_decode_seq = partial(self._f_dec_seq, params)

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
                state_seq, state, (drdt, _, nfe_sum) = nn_ode(state0, adm_los)
                dec_emb, dec_diag = nn_decode_seq(state_seq)

                odeint_weeks += sum(adm_los.values()) / 7

            else:
                state = {i: subject_state[i]['state'] for i in adm_id}

                d2d_time = {
                    i: adm_time[i] + adm_los[i] - subject_state[i]['time']
                    for i in adm_id
                }

                # Integrate until next discharge
                state_seq, state, (drdt, nfe,
                                   nfe_sum) = nn_ode(state, d2d_time)
                dec_emb, dec_diag = nn_decode_seq(state_seq)

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
        l1_loss = l1_absolute(params)
        l2_loss = l2_squared(params)
        dyn_loss = res['dyn_loss']
        pred_alpha = loss_mixing['L_pred']
        update_alpha = loss_mixing['L_update']

        l1_alpha = loss_mixing['L_l1']
        l2_alpha = loss_mixing['L_l2']
        dyn_alpha = loss_mixing['L_dyn'] / (res['odeint_weeks'] + 1e-10)

        diag_loss = (pred_alpha * prediction_loss + update_alpha * update_loss)

        loss = diag_loss + (l1_alpha * l1_loss) + (l2_alpha * l2_loss) + (
            dyn_alpha * dyn_loss)

        return {
            'prediction_loss': prediction_loss,
            'update_loss': update_loss,
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
            'L_update': trial.suggest_float('L_update', 1e-4, 1, log=True),
            'L_dyn': trial.suggest_float('L_dyn', 1e-3, 1e3, log=True),
            **config['loss_mixing']
        }

        return config

    @staticmethod
    def sample_training_config(trial: optuna.Trial):
        return ICENODE._sample_ode_training_config(trial, epochs=15)

    @staticmethod
    def _sample_ode_model_config(trial: optuna.Trial):
        model_params = {
            'ode_dyn':
            trial.suggest_categorical(
                'ode_dyn', ['mlp', 'gru', 'res'
                            ]),  # Add depth conditional to 'mlp' or 'res'
            'ode_with_bias':
            False,  # trial.suggest_categorical('ode_b', [True, False]),
            'ode_init_var':
            trial.suggest_float('ode_iv', 1e-4, 1e-1, log=True),
            'ode_timescale':
            trial.suggest_float('ode_ts', 1, 1e1, log=True),
            'trajectory_sample_rate':
            trial.suggest_int('los_f', 2, 25),
            'diag_seq_combiner':
            trial.suggest_categorical('comb', ['last', 'mean', 'max', 'att']),
            'state_size':
            trial.suggest_int('s', 30, 300, 30),
            'init_depth':
            3,  # trial.suggest_int('init_d', 2, 5),
            'tay_reg':
            trial.suggest_categorical('tay', [0, 2, 3, 4]),
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
