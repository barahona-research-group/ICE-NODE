from functools import partial
from typing import (Any, Callable, Dict, Iterable, List, Optional, Tuple)

import haiku as hk
import jax
import jax.numpy as jnp

import optuna

from .jax_interface import (SubjectJAXInterface, create_patient_interface)
from .gram import DAGGRAM
from .models import NumericObsModel, GRUBayes
from .metrics import (l2_squared, l1_absolute, numeric_error, lognormal_loss,
                      compute_KL_loss)
from .utils import wrap_module
from .train_snonet_lite import SNONETLite


class SNONET(SNONETLite):
    def __init__(self, subject_interface: SubjectJAXInterface,
                 diag_gram: DAGGRAM, proc_gram: DAGGRAM, ode_dyn: str,
                 ode_depth: int, ode_with_bias: bool, ode_init_var: float,
                 ode_timescale: float, tay_reg: Optional[int], state_size: int,
                 numeric_hidden_size: int, init_depth: bool,
                 diag_loss: Callable[[jnp.ndarray, jnp.ndarray],
                                     float], max_odeint_days: int):
        super().__init__(subject_interface=subject_interface,
                         diag_gram=diag_gram,
                         proc_gram=proc_gram,
                         ode_dyn=ode_dyn,
                         ode_depth=ode_depth,
                         ode_with_bias=ode_with_bias,
                         ode_init_var=ode_init_var,
                         ode_timescale=ode_timescale,
                         tay_reg=tay_reg,
                         state_size=state_size,
                         init_depth=init_depth,
                         diag_loss=diag_loss,
                         max_odeint_days=max_odeint_days)

        self.dimensions['numeric'] = len(subject_interface.test_idx)
        self.dimensions['numeric_hidden'] = numeric_hidden_size

        f_update_init, f_update = hk.without_apply_rng(
            hk.transform(
                wrap_module(GRUBayes, state_size=state_size, name='f_update')))
        self.f_update = jax.jit(f_update)
        self.initializers['f_update'] = f_update_init

        f_num_init, f_num = hk.without_apply_rng(
            hk.transform(
                wrap_module(NumericObsModel,
                            numeric_size=self.dimensions['numeric'],
                            numeric_hidden_size=numeric_hidden_size,
                            name='f_numeric')))

        self.f_num = jax.jit(f_num)
        self.initializers['f_num'] = f_num_init

    def numeric_index(self) -> List[str]:
        index2test = {i: t for t, i in self.subject_interface.test_idx.items()}
        return list(map(index2test.get, range(len(index2test))))

    def _initialization_data(self):
        """
        Creates data for initializing each of the
        modules based on the shapes of init_data.
        """
        numeric = jnp.zeros(self.dimensions['numeric'])
        diag_gram_ = jnp.zeros(self.dimensions['diag_gram'])
        state = jnp.zeros(self.dimensions['state'])
        ode_ctrl = jnp.zeros(self.dimensions['ode_control'])
        state_in = jnp.zeros(self.dimensions['state_init'])
        return {
            "f_n_ode": [True, state, 0.1, ode_ctrl],
            "f_update": [state, numeric, numeric, diag_gram_],
            "f_num": [state],
            "f_dec": [state],
            "f_init": [state_in]
        }

    def _extract_nth_points(self, params, subjects_batch: List[int],
                            n: int) -> Dict[str, Dict[int, jnp.ndarray]]:
        diag_G = self.diag_gram.compute_embedding_mat(params["diag_gram"])
        proc_G = self.proc_gram.compute_embedding_mat(params["proc_gram"])

        points = self.subject_interface.nth_points_batch(n, subjects_batch)
        if len(points) == 0:
            return None

        diag_gram = {
            i: self.diag_gram.encode(diag_G, v['diag_multi_ccs_vec'])
            for i, v in points.items() if v['diag_multi_ccs_vec'] is not None
        }
        proc_gram = {
            i: self.proc_gram.encode(proc_G, v['proc_multi_ccs_vec'])
            for i, v in points.items() if v['proc_multi_ccs_vec'] is not None
        }
        diag_out = {
            i: v['diag_single_ccs_vec']
            for i, v in points.items() if v['diag_single_ccs_vec'] is not None
        }
        days_ahead = {i: v['days_ahead'] for i, v in points.items()}
        numeric = {
            i: v['tests'][0]
            for i, v in points.items() if v['tests'] is not None
        }
        mask = {
            i: v['tests'][1]
            for i, v in points.items() if v['tests'] is not None
        }
        age = {i: v['age'] for i, v in points.items()}

        def _ode_control(subject_id):
            zero_proc_gram = jnp.zeros(self.dimensions['proc_gram'])
            d = {
                'proc_gram': proc_gram.get(subject_id, zero_proc_gram),
                'age': points[subject_id]['age'],
                'static': self.subject_interface.subject_static(subject_id)
            }
            return jnp.hstack(map(d.get, self.ode_control_passes))

        ode_control = {i: _ode_control(i) for i in points.keys()}

        return {
            'age': age,
            'days_ahead': days_ahead,
            'diag_gram': diag_gram,
            'numeric': numeric,
            'mask': mask,
            'ode_control': ode_control,
            'diag_out': diag_out
        }

    def _f_num(
        self, params: Any, state: Dict[str,
                                       jnp.ndarray], selection: Iterable[int]
    ) -> Tuple[Dict[int, jnp.ndarray], Dict[int, jnp.ndarray]]:

        mean_logvar = {
            i: self.f_num(params['f_num'], state[i])
            for i in selection
        }

        mean = {i: m for i, (m, _) in mean_logvar.items()}
        logvar = {i: lv for i, (_, lv) in mean_logvar.items()}
        return mean, logvar

    def _f_update(self, params: Any, state: Dict[int, jnp.ndarray],
                  numeric_error: Dict[int, jnp.ndarray],
                  numeric_mask: Dict[int, jnp.ndarray],
                  diag_gram_error: jnp.ndarray) -> jnp.ndarray:
        zero_gram_error = jnp.zeros(self.dimensions['diag_gram'])
        zero_numeric = jnp.zeros(self.dimensions['numeric'])

        updated_state = {
            i: self.f_update(params['f_update'], state[i],
                             numeric_error.get(i, zero_numeric),
                             numeric_mask.get(i, zero_numeric),
                             diag_gram_error.get(i, zero_gram_error))
            for i in (set(numeric_error) | set(diag_gram_error))
        }

        return updated_state

    @staticmethod
    def _numeric_error(mean_true, mean_predicted, logvar_predicted):
        error_num = {
            i: numeric_error(mean_true[i], mean_predicted[i],
                             logvar_predicted[i])
            for i in mean_predicted.keys()
        }
        return error_num

    @staticmethod
    def _lognormal_loss(mask: Dict[int, jnp.ndarray],
                        normal_error: Dict[int, jnp.ndarray],
                        logvar: Dict[int, jnp.ndarray]):
        loss = {
            i: lognormal_loss(mask[i], normal_error[i], logvar[i])
            for i in normal_error.keys()
        }
        if loss:
            return sum(loss.values()) / len(loss)
        else:
            return 0.0

    @staticmethod
    def _kl_loss(mean_true: Dict[int, jnp.ndarray], mask: Dict[int,
                                                               jnp.ndarray],
                 mean_predicted: Dict[int, jnp.ndarray],
                 logvar_predicted: Dict[int, jnp.ndarray]):
        loss = {
            i: compute_KL_loss(mean, mask[i], mean_predicted[i],
                               logvar_predicted[i])
            for i, mean in mean_predicted.items()
        }
        if loss:
            return sum(loss.values()) / len(loss)
        else:
            return 0.0

    def __call__(self, params: Any, subjects_batch: List[int],
                 count_nfe: bool):
        nth_points = partial(self._extract_nth_points, params, subjects_batch)
        nn_num = partial(self._f_num, params)
        nn_ode = partial(self._f_n_ode, params, count_nfe)
        nn_update = partial(self._f_update, params)  # state, e, m, diag_error
        nn_decode = partial(self._f_dec, params)
        nn_init = partial(self._f_init, params)
        diag_loss = self._diag_loss

        subject_state = {}
        dyn_loss = []
        nfe = []

        prejump_diag_loss = []
        postjump_diag_loss = []
        prejump_num_loss = []
        postjump_num_loss = []

        num_weights = []
        diag_weights = []

        diag_detectability = {i: dict() for i in subjects_batch}
        all_points_count = 0
        diag_count = 0
        num_count = 0
        odeint_weeks = 0.0

        for n in self.subject_interface.n_support:
            points_n = nth_points(n)

            if points_n is None:
                continue

            all_points_count += len(points_n)

            days_ahead = points_n['days_ahead']
            diag_gram = points_n['diag_gram']
            diag_out = points_n['diag_out']
            numeric = points_n['numeric']
            mask = points_n['mask']
            delta_days = {
                i: (days_ahead[i] - subject_state[i]['days'])
                for i in (set(subject_state) & set(days_ahead))
                if days_ahead[i] -
                subject_state[i]['days'] <= self.max_odeint_days
            }

            state_i = {i: subject_state[i]['value'] for i in delta_days}

            # Reset subjects state with long gaps
            reset_subjects = set(days_ahead) - set(delta_days)
            map(lambda k: subject_state.pop(k, None), reset_subjects)

            state_0 = nn_init(points_n, reset_subjects, days_ahead)
            subject_state.update(state_0)
            # This intersection ensures only prediction for:
            # 1. cases that are integrable (i.e. with previous state), and
            # 2. cases that have diagnosis at index n.
            diag_cases = set(delta_days).intersection(diag_out.keys())
            diag_count += len(diag_cases)
            # No. of "Predictable" diagnostic points
            diag_weights.append(len(diag_cases))

            num_cases = set(delta_days).intersection(numeric.keys())
            num_count += len(num_cases)
            num_weights.append(sum(mask[i].sum() for i in num_cases) + 1e-10)

            odeint_weeks += sum(delta_days.values()) / 7.0
            ################## ODEINT #####################
            state_j, _dyn_loss, _nfe = nn_ode(state_i, delta_days,
                                              points_n['ode_control'])
            dyn_loss.append(_dyn_loss)
            nfe.append(_nfe)
            ########## PRE-JUMP NUM LOSS ########################
            pre_mean, pre_logvar = nn_num(state_j, num_cases)
            error = self._numeric_error(numeric, pre_mean, pre_logvar)

            prejump_num_loss.append(
                self._lognormal_loss(mask, error, pre_logvar))

            ########## PRE-JUMP DAG LOSS #########################
            pre_diag_gram, pre_diag_out = nn_decode(state_j, diag_cases)
            pre_diag_loss = diag_loss(diag_out, pre_diag_out)
            pre_diag_gram_error = self._gram_error(diag_gram, pre_diag_gram)
            prejump_diag_loss.append(pre_diag_loss)

            ############## GRU BAYES ####################
            # Using GRUObservationCell to update h.
            state_j_updated = nn_update(state_j, error, mask,
                                        pre_diag_gram_error)
            state_j.update(state_j_updated)
            # Update the states:
            for subject_id, new_state in state_j.items():
                subject_state[subject_id] = {
                    'days': days_ahead[subject_id],
                    'value': new_state
                }
            ################ POST-JUMP NUM LOSS ####################
            post_mean, post_logvar = nn_num(state_j_updated, num_cases)
            postjump_num_loss.append(
                self._kl_loss(numeric, mask, post_mean, post_logvar))

            ############### POST-JUNP DAG LOSS ########################
            _, post_diag_out = nn_decode(state_j_updated, diag_cases)

            post_diag_loss = diag_loss(diag_out, post_diag_out)
            postjump_diag_loss.append(post_diag_loss)

            for subject_id in post_diag_out.keys():
                diag_detectability[subject_id][n] = {
                    'days_ahead': days_ahead[subject_id],
                    'diag_true': diag_out[subject_id],
                    'pre_logits': pre_diag_out[subject_id],
                    'post_logits': post_diag_out[subject_id]
                }

        prejump_num_loss = jnp.average(prejump_num_loss, weights=num_weights)
        postjump_num_loss = jnp.average(postjump_num_loss, weights=num_weights)
        prejump_diag_loss = jnp.average(prejump_diag_loss,
                                        weights=diag_weights)
        postjump_diag_loss = jnp.average(postjump_diag_loss,
                                         weights=diag_weights)
        return {
            'prejump_num_loss': prejump_num_loss,
            'postjump_num_loss': postjump_num_loss,
            'prejump_diag_loss': prejump_diag_loss,
            'postjump_diag_loss': postjump_diag_loss,
            'dyn_loss': jnp.sum(sum(dyn_loss)),
            'odeint_weeks': odeint_weeks,
            'all_points_count': all_points_count,
            'diag_count': diag_count,
            'num_count': num_count,
            'nfe': sum(nfe),
            'diag_detectability': diag_detectability,
        }

    def detailed_loss(self, loss_mixing, params, res):
        prejump_num_loss = res['prejump_num_loss']
        postjump_num_loss = res['postjump_num_loss']
        prejump_diag_loss = res['prejump_diag_loss']
        postjump_diag_loss = res['postjump_diag_loss']
        l1_loss = l1_absolute(params)
        l2_loss = l2_squared(params)
        dyn_loss = res['dyn_loss']
        num_alpha = loss_mixing['L_num']
        diag_alpha = loss_mixing['L_diag']
        ode_alpha = loss_mixing['L_ode']
        l1_alpha = loss_mixing['L_l1']
        l2_alpha = loss_mixing['L_l2']
        dyn_alpha = loss_mixing['L_dyn'] / (res['odeint_weeks'])

        num_loss = (
            1 - num_alpha) * prejump_num_loss + num_alpha * postjump_num_loss
        diag_loss = (1 - diag_alpha
                     ) * prejump_diag_loss + diag_alpha * postjump_diag_loss
        ode_loss = (1 - ode_alpha) * diag_loss + ode_alpha * num_loss
        loss = ode_loss + (l1_alpha * l1_loss) + (l2_alpha * l2_loss) + (
            dyn_alpha * dyn_loss)
        return {
            'prejump_num_loss': prejump_num_loss,
            'postjump_num_loss': postjump_num_loss,
            'prejump_diag_loss': prejump_diag_loss,
            'postjump_diag_loss': postjump_diag_loss,
            'num_loss': num_loss,
            'diag_loss': diag_loss,
            'ode_loss': ode_loss,
            'l1_loss': l1_loss,
            'l2_loss': l2_loss,
            'dyn_loss': dyn_loss,
            'dyn_loss_per_week': dyn_loss / res['odeint_weeks'],
            'loss': loss
        }

    def eval_stats(self, res):
        nfe = res['nfe']
        return {
            'all_points_count': res['all_points_count'],
            'diag_count': res['diag_count'],
            'num_count': res['num_count'],
            'nfe/week': nfe / res['odeint_weeks'],
            'nfex1000': nfe / 1000
        }

    @staticmethod
    def create_patient_interface(mimic_dir: str):
        return create_patient_interface(mimic_dir)

    @staticmethod
    def sample_training_config(trial: optuna.Trial):
        config = SNONET._sample_ode_training_config(trial, epochs=2)
        config['loss_mixing'] = {
            **config['loss_mixing'], 'L_num':
            trial.suggest_float('L_num', 1e-4, 1, log=True),
            'L_ode':
            trial.suggest_float('L_ode', 1e-5, 1, log=True)
        }
        return config

    @staticmethod
    def sample_model_config(trial: optuna.Trial):
        config = SNONETLite._sample_ode_model_config(trial)
        config['numeric_hidden_size'] = trial.suggest_int(
            'num_h_size', 100, 350, 50)
        return config


if __name__ == '__main__':
    from .hpo_utils import capture_args, run_trials
    run_trials(model_cls=SNONET, **capture_args())
