from absl import logging
from functools import partial
from typing import (Any, Callable, Dict, Iterable, List, Optional, Tuple, Set)

import haiku as hk
import jax
import jax.numpy as jnp

import optuna

from .metrics import (bce, softmax_loss, balanced_focal_bce, weighted_bce,
                      l2_squared, l1_absolute)
from .utils import wrap_module, load_config, load_params

from .jax_interface import (SubjectJAXInterface, create_patient_interface)
from .gram import DAGGRAM
from .models import (MLPDynamics, ResDynamics, GRUDynamics, NeuralODE,
                     DiagnosesUpdate, StateDiagnosesDecoder, StateInitializer)
from .abstract_model import AbstractModel
from .glove import glove_representation


class SNONETDiag(AbstractModel):
    def __init__(self, subject_interface: SubjectJAXInterface,
                 diag_gram: DAGGRAM, ode_dyn: str, ode_depth: int,
                 ode_with_bias: bool, ode_init_var: float,
                 ode_timescale: float, tay_reg: Optional[int], state_size: int,
                 init_depth: bool,
                 diag_loss: Callable[[jnp.ndarray, jnp.ndarray],
                                     float], max_odeint_days: int):

        self.subject_interface = subject_interface
        self.diag_gram = diag_gram
        self.tay_reg = tay_reg
        self.max_odeint_days = max_odeint_days
        self.diag_loss = diag_loss
        self.dimensions = {
            'diag_gram': diag_gram.embeddings_dim,
            'diag_in': len(subject_interface.diag_multi_ccs_idx),
            'diag_out': len(subject_interface.diag_multi_ccs_idx),
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
        self.f_n_ode = jax.jit(f_n_ode, static_argnums=(1, ))

        f_update_init, f_update = hk.without_apply_rng(
            hk.transform(
                wrap_module(DiagnosesUpdate,
                            state_size=state_size,
                            name='f_update')))
        self.f_update = jax.jit(f_update)

        f_dec_init, f_dec = hk.without_apply_rng(
            hk.transform(
                wrap_module(StateDiagnosesDecoder,
                            hidden_size=self.dimensions['diag_gram'],
                            gram_size=self.dimensions['diag_gram'],
                            output_size=self.dimensions['diag_out'],
                            name='f_dec')))
        self.f_dec = jax.jit(f_dec)

        f_init_init, f_init = hk.without_apply_rng(
            hk.transform(
                wrap_module(StateInitializer,
                            hidden_size=self.dimensions['diag_gram'],
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

    def init_params(self, rng_key):
        init_data = self._initialization_data()
        return {
            "diag_gram": self.diag_gram.init_params(rng_key),
            **{
                label: init(rng_key, *init_data[label])
                for label, init in self.initializers.items()
            }
        }

    def state_size(self):
        return self.dimensions['state']

    def diag_out_index(self) -> List[str]:
        index2code = {
            i: c
            for c, i in self.subject_interface.diag_single_ccs_idx.items()
        }
        return list(map(index2code.get, range(len(index2code))))

    def _initialization_data(self):
        """
        Creates data for initializing each of the
        modules based on the shapes of init_data.
        """
        diag_gram_ = jnp.zeros(self.dimensions['diag_gram'])
        state = jnp.zeros(self.dimensions['state'])
        ode_ctrl = jnp.array([])
        return {
            "f_n_ode": [True, state, 0.1, ode_ctrl],
            "f_update": [state, diag_gram_],
            "f_dec": [state],
            "f_init": [diag_gram_]
        }

    def _extract_nth_points(self, params: Any, subjects_batch: List[int],
                            n: int) -> Dict[str, Dict[int, jnp.ndarray]]:
        diag_G = self.diag_gram.compute_embedding_mat(params["diag_gram"])

        points = self.subject_interface.nth_points_batch(n, subjects_batch)
        if len(points) == 0:
            return None

        diag_gram = {
            i: self.diag_gram.encode(diag_G, v['diag_multi_ccs_vec'])
            for i, v in points.items() if v['diag_multi_ccs_vec'] is not None
        }
        diag_out = {
            i: v['diag_multi_ccs_vec']
            for i, v in points.items() if v['diag_multi_ccs_vec'] is not None
        }
        days_ahead = {i: v['days_ahead'] for i, v in points.items()}

        return {
            'ode_control': None,
            'days_ahead': days_ahead,
            'diag_gram': diag_gram,
            'diag_out': diag_out
        }

    def _f_n_ode(self, params, count_nfe, h, t, c):
        if c is None:
            null = jnp.array([])
            c = {i: null for i in h}

        h_r_nfe = {
            i: self.f_n_ode(params['f_n_ode'], count_nfe, h[i], t[i], c[i])
            for i in h.keys()
        }

        nfe = sum(n for h, r, n in h_r_nfe.values())
        r1 = jnp.sum(sum(r for (h, r, n) in h_r_nfe.values()))
        h1 = {i: h for i, (h, r, n) in h_r_nfe.items()}
        return h1, r1, nfe

    def _f_update(self, params: Any, state: Dict[int, jnp.ndarray],
                  diag_gram_error: jnp.ndarray) -> jnp.ndarray:
        updated_state = {
            i: self.f_update(params['f_update'], state[i], gram_error)
            for i, gram_error in diag_gram_error.items()
        }

        return updated_state

    def _f_dec(
        self, params: Any, state: Dict[int, jnp.ndarray], selection: Set[int]
    ) -> Tuple[Dict[int, jnp.ndarray], Dict[int, jnp.ndarray]]:
        gram_out = {
            i: self.f_dec(params['f_dec'], state[i])
            for i in selection
        }
        gram = {i: g for i, (g, _) in gram_out.items()}
        out = {i: o for i, (_, o) in gram_out.items()}

        return gram, out

    def _f_init(self, params, points_n, subjects: Iterable[int],
                days_ahead: Dict[int, int]):
        diag_gram = points_n['diag_gram']

        def _state_init(subject_id):
            return {
                'value': self.f_init(params['f_init'], diag_gram[subject_id]),
                'days': days_ahead[subject_id]
            }

        return {i: _state_init(i) for i in (set(diag_gram) & set(subjects))}

    @staticmethod
    def _gram_error(gram_true, gram_predicted):
        error_gram = {
            i: gram_true[i] - gram_predicted[i]
            for i in gram_predicted.keys()
        }
        return error_gram

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
        nth_points = partial(self._extract_nth_points, params, subjects_batch)
        nn_ode = partial(self._f_n_ode, params, count_nfe)
        nn_update = partial(self._f_update, params)
        nn_decode = partial(self._f_dec, params)
        nn_init = partial(self._f_init, params)
        diag_loss = self._diag_loss
        subject_state = dict()
        dyn_loss = []
        nfe = []

        prejump_diag_loss = []
        postjump_diag_loss = []
        diag_weights = []

        diag_detectability = {i: {} for i in subjects_batch}
        all_points_count = 0
        predictable_count = 0
        odeint_weeks = 0.0

        for n in self.subject_interface.n_support:
            points_n = nth_points(n)

            if points_n is None:
                continue
            all_points_count += len(points_n)

            days_ahead = points_n['days_ahead']
            diag_gram = points_n['diag_gram']
            diag_out = points_n['diag_out']
            ode_c = points_n['ode_control']
            delta_days = {
                i: (days_ahead[i] - subject_state[i]['days'])
                for i in (set(subject_state) & set(days_ahead))
                if days_ahead[i] -
                subject_state[i]['days'] <= self.max_odeint_days
            }

            state_i = {i: subject_state[i]['value'] for i in delta_days}

            # From points returned at index n:
            # - Consider for odeint:
            #   1. subjects that already have a previous state, and
            #   2. days difference doesn't exceed maximum days.
            # - For returned subjects with days difference exceeding the
            # threshold, reset their previous state.
            # - Initialize new states for subjects that have diagnosis codes
            #   that has not been previously initialized or has been reset.

            # Reset subjects state with long gaps
            reset_subjects = set(days_ahead) - set(delta_days)
            map(lambda k: subject_state.pop(k, None), reset_subjects)

            state_0 = nn_init(points_n, reset_subjects, days_ahead)
            subject_state.update(state_0)
            # This intersection ensures only prediction for:
            # 1. cases that are integrable (i.e. with previous state), and
            # 2. cases that have diagnosis at index n.
            predictable_cases = set(delta_days).intersection(diag_out.keys())
            predictable_count += len(predictable_cases)

            # No. of "Predictable" diagnostic points
            diag_weights.append(len(predictable_cases))
            '''
            Idea: scale days_forward to weeks_forward.
            This can:
                1. Improve the numerical stability and accuracy of numerical integration.
                2. Force the ode_dyn model to learn weekly dynamics, which is a suitable time scale for cancer development.
            '''
            odeint_weeks += sum(delta_days.values()) / 7
            ################## ODEINT #####################
            state_j, _dyn_loss, _nfe = nn_ode(state_i, delta_days, ode_c)
            dyn_loss.append(_dyn_loss)
            nfe.append(_nfe)
            ########## PRE-JUMP DAG LOSS #########################
            pre_diag_gram, pre_diag_out = nn_decode(state_j, predictable_cases)
            pre_diag_loss = diag_loss(diag_out, pre_diag_out)
            pre_diag_gram_error = self._gram_error(diag_gram, pre_diag_gram)
            prejump_diag_loss.append(pre_diag_loss)
            ############## GRU BAYES ####################
            # Using GRUObservationCell to update h.
            state_j_updated = nn_update(state_j, pre_diag_gram_error)
            state_j.update(state_j_updated)
            # Update the states:
            for subject_id, new_state in state_j.items():
                subject_state[subject_id] = {
                    'days': days_ahead[subject_id],
                    'value': new_state
                }

            ############### POST-JUNP DAG LOSS ########################
            _, post_diag_out = nn_decode(state_j_updated, predictable_cases)

            post_diag_loss = diag_loss(diag_out, post_diag_out)
            postjump_diag_loss.append(post_diag_loss)

            for subject_id in post_diag_out.keys():
                diag_detectability[subject_id][n] = {
                    'days_ahead': days_ahead[subject_id],
                    'diag_true': diag_out[subject_id],
                    'pre_logits': pre_diag_out[subject_id],
                    'post_logits': post_diag_out[subject_id]
                }

        prejump_diag_loss = jnp.average(prejump_diag_loss,
                                        weights=diag_weights)
        postjump_diag_loss = jnp.average(postjump_diag_loss,
                                         weights=diag_weights)
        ret = {
            'prejump_diag_loss': prejump_diag_loss,
            'postjump_diag_loss': postjump_diag_loss,
            'dyn_loss': jnp.sum(sum(dyn_loss)),
            'odeint_weeks': odeint_weeks,
            'all_points_count': all_points_count,
            'predictable_count': predictable_count,
            'nfe': sum(nfe),
            'diag_detectability': diag_detectability
        }

        return ret

    def detailed_loss(self, loss_mixing, params, res):
        prejump_diag_loss = res['prejump_diag_loss']
        postjump_diag_loss = res['postjump_diag_loss']
        l1_loss = l1_absolute(params)
        l2_loss = l2_squared(params)
        dyn_loss = res['dyn_loss']
        diag_alpha = loss_mixing['L_diag']
        l1_alpha = loss_mixing['L_l1']
        l2_alpha = loss_mixing['L_l2']
        dyn_alpha = loss_mixing['L_dyn'] / (res['odeint_weeks'])

        diag_loss = (1 - diag_alpha
                     ) * prejump_diag_loss + diag_alpha * postjump_diag_loss
        loss = diag_loss + (l1_alpha * l1_loss) + (l2_alpha * l2_loss) + (
            dyn_alpha * dyn_loss)

        return {
            'prejump_diag_loss': prejump_diag_loss,
            'postjump_diag_loss': postjump_diag_loss,
            'diag_loss': diag_loss,
            'loss': loss,
            'l1_loss': l1_loss,
            'l2_loss': l2_loss,
            'dyn_loss': dyn_loss,
            'dyn_loss_per_week': dyn_loss / res['odeint_weeks']
        }

    def eval_stats(self, res):
        nfe = res['nfe']
        return {
            'all_points_count': res['all_points_count'],
            'predictable_count': res['predictable_count'],
            'nfe_per_week': nfe / res['odeint_weeks'],
            'nfex1000': nfe / 1000
        }

    @staticmethod
    def create_patient_interface(mimic_dir):
        return create_patient_interface(mimic_dir, ignore_tests=True)

    @classmethod
    def select_loss(cls, loss_label: str, patient_interface, train_ids):
        if loss_label == 'balanced_focal':
            return lambda t, p: balanced_focal_bce(t, p, gamma=2, beta=0.999)
        elif loss_label == 'softmax':
            return softmax_loss
        elif loss_label == 'bce':
            return bce
        elif loss_label == 'balanced_bce':
            codes_dist = patient_interface.diag_multi_ccs_frequency_vec(
                train_ids)
            weights = codes_dist.sum() / (codes_dist + 1e-1) * len(codes_dist)
            return lambda t, logits: weighted_bce(t, logits, weights)
        else:
            raise ValueError(f'Unrecognized diag_loss: {loss_label}')

    @classmethod
    def create_model(cls, config, patient_interface, train_ids,
                     pretrained_components):
        pretrained_components = load_config(pretrained_components)
        gram_component = pretrained_components['gram']['diag']['params_file']
        diag_gram_pretrained_params = load_params(gram_component)['diag_gram']

        diag_gram = DAGGRAM(
            ccs_dag=patient_interface.dag,
            code2index=patient_interface.diag_multi_ccs_idx,
            basic_embeddings=None,
            ancestors_mat=patient_interface.diag_multi_ccs_ancestors_mat,
            frozen_params=diag_gram_pretrained_params,
            **config['gram']['diag'])

        diag_loss = cls.select_loss(config['training']['diag_loss'],
                                    patient_interface, train_ids)

        return cls(subject_interface=patient_interface,
                   diag_gram=diag_gram,
                   **config['model'],
                   tay_reg=config['training']['tay_reg'],
                   diag_loss=diag_loss)

    @staticmethod
    def _sample_ode_training_config(trial: optuna.Trial, epochs):
        config = AbstractModel._sample_training_config(trial, epochs)
        config['tay_reg'] = trial.suggest_categorical('tay', [0, 2, 3])
        # UNDO
        config['diag_loss'] = trial.suggest_categorical(
            'dx_loss', ['balanced_bce', 'softmax'])
        # trial.suggest_categorical('dx_loss', ['balanced_focal', 'bce', 'softmax', 'balanced_bce'])

        config['loss_mixing'] = {
            'L_diag':
            trial.suggest_float('L_dx', 1e-4, 1, log=True),
            'L_dyn':
            trial.suggest_float('L_dyn', 1e-3, 1e3, log=True)
            if config['tay_reg'] > 0 else 0.0,
            **config['loss_mixing']
        }

        return config

    @staticmethod
    def sample_training_config(trial: optuna.Trial):
        return SNONETDiag._sample_ode_training_config(trial, epochs=10)

    @staticmethod
    def _sample_ode_model_config(trial: optuna.Trial):
        model_params = {
            'ode_dyn': trial.suggest_categorical(
                'ode_dyn', ['mlp', 'gru', 'res'
                            ]),  # Add depth conditional to 'mlp' or 'res'
            'ode_with_bias': trial.suggest_categorical('ode_b', [True, False]),
            'ode_init_var': trial.suggest_float('ode_iv', 1e-5, 1, log=True),
            'ode_timescale': trial.suggest_float('ode_ts', 1, 1e4, log=True),
            'state_size': trial.suggest_int('s', 100, 350, 50),
            'init_depth': trial.suggest_int('init_d', 1, 4),
            'max_odeint_days':
            # UNDO
            10 * 360  #trial.suggest_int('mx_ode_ds', 8 * 7, 16 * 7, 7)
        }
        if model_params['ode_dyn'] == 'gru':
            model_params['ode_depth'] = 0
        else:
            model_params['ode_depth'] = trial.suggest_int('ode_d', 1, 4)

        return model_params

    @staticmethod
    def sample_model_config(trial: optuna.Trial):
        return SNONETDiag._sample_ode_model_config(trial)

    @classmethod
    def sample_experiment_config(cls, trial: optuna.Trial,
                                 pretrained_components: str):
        pretrained_components = load_config(pretrained_components)
        # Configurations used for pretraining GRAM.
        # The configuration will be used to initialize GloVe and GRAM models.
        gram_component = pretrained_components['gram']['diag']['config_file']
        gram_component = load_config(gram_component)

        return {
            'pretrained_components': pretrained_components,
            'glove': None,
            'gram': gram_component['gram'],
            'model': cls.sample_model_config(trial),
            'training': cls.sample_training_config(trial)
        }


if __name__ == '__main__':
    from .hpo_utils import capture_args, run_trials
    run_trials(model_cls=SNONETDiag, **capture_args())
