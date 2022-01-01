from functools import partial
from typing import (Any, Callable, Dict, Iterable, List, Optional, Tuple, Set)

import jax.numpy as jnp
import optuna

from .jax_interface import SubjectJAXInterface
from .gram import DAGGRAM
from .train_snonet_diag import SNONETDiag
from .glove import glove_representation


class SNONETDiagProc(SNONETDiag):
    def __init__(self, subject_interface: SubjectJAXInterface,
                 diag_gram: DAGGRAM, proc_gram: DAGGRAM, ode_dyn: str,
                 ode_depth: int, tay_reg: Optional[int], state_size: int,
                 init_depth: bool, bias: bool,
                 diag_loss: Callable[[jnp.ndarray, jnp.ndarray], float],
                 max_odeint_days: int, **init_kwargs):
        super().__init__(subject_interface=subject_interface,
                         diag_gram=diag_gram,
                         ode_dyn=ode_dyn,
                         ode_depth=ode_depth,
                         tay_reg=tay_reg,
                         state_size=state_size,
                         init_depth=init_depth,
                         bias=bias,
                         diag_loss=diag_loss,
                         max_odeint_days=max_odeint_days)

        self.proc_gram = proc_gram

        self.dimensions['proc_gram'] = proc_gram.basic_embeddings_dim
        self.dimensions['proc_dag'] = len(subject_interface.proc_multi_ccs_idx)

        self.ode_control_passes = ['proc_gram']
        self.state_init_passes = ['diag_gram']

        self.dimensions.update({
            'ode_control':
            sum(map(self.dimensions.get, self.ode_control_passes)),
            'state_init':
            sum(map(self.dimensions.get, self.state_init_passes))
        })

    def init_params(self, rng_key):
        init_data = self._initialization_data()
        return {
            "diag_gram": self.diag_gram.init_params(rng_key),
            "proc_gram": self.proc_gram.init_params(rng_key),
            **{
                label: init(rng_key, *init_data[label])
                for label, init in self.initializers.items()
            }
        }

    def _initialization_data(self):
        """
        Creates data for initializing each of the
        modules based on the shapes of init_data.
        """
        diag_gram_ = jnp.zeros(self.dimensions['diag_gram'])
        state = jnp.zeros(self.dimensions['state'])
        ode_ctrl = jnp.zeros(self.dimensions['ode_control'])
        state_in = jnp.zeros(self.dimensions['state_init'])
        return {
            "f_n_ode": [True, state, 0.1, ode_ctrl],
            "f_update": [state, diag_gram_],
            "f_dec": [state],
            "f_init": [state_in]
        }

    def _extract_nth_points(self, params: Any, subjects_batch: List[int],
                            diag_G: jnp.ndarray, proc_G: jnp.ndarray,
                            n: int) -> Dict[str, Dict[int, jnp.ndarray]]:

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
            'ode_control': ode_control,
            'diag_out': diag_out
        }

    def _f_n_ode(self, params, count_nfe, h, t, c):
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

    def _generate_embedding_mats(self, params):
        diag = self.diag_gram.compute_embedding_mat(params["diag_gram"])
        proc = self.proc_gram.compute_embedding_mat(params["proc_gram"])
        return diag, proc

    def _f_init(self, params, diag_gram, subjects: Iterable[int],
                days_ahead: Dict[int, int]):
        def _state_init(subject_id):
            return {
                'value': self.f_init(params['f_init'], diag_gram[subject_id]),
                'days': days_ahead[subject_id]
            }

        return {i: _state_init(i) for i in (set(diag_gram) & set(subjects))}

    def __call__(self,
                 params: Any,
                 subjects_batch: List[int],
                 return_path: bool = False,
                 count_nfe: bool = False):

        diag_G, proc_G = self._generate_embedding_mats(params)
        nth_points = partial(self._extract_nth_points, params, subjects_batch,
                             diag_G, proc_G)
        nn_ode = partial(self._f_n_ode, params, count_nfe)
        nn_update = partial(self._f_update, params)
        nn_decode = partial(self._f_dec, params)
        nn_init = partial(self._f_init, params)
        diag_loss = partial(self._diag_loss, self.diag_loss)

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
            delta_weeks = {
                i: (days_ahead[i] - subject_state[i]['days']) / 7.0
                for i in (set(subject_state) & set(days_ahead))
                if days_ahead[i] -
                subject_state[i]['days'] <= self.max_odeint_days
            }

            state_i = {i: subject_state[i]['value'] for i in delta_weeks}

            # From points returned at index n:
            # - Consider for odeint:
            #   1. subjects that already have a previous state, and
            #   2. days difference doesn't exceed maximum days.
            # - For returned subjects with days difference exceeding the
            # threshold, reset their previous state.
            # - Initialize new states for subjects that have diagnosis codes
            #   that has not been previously initialized or has been reset.

            # Reset subjects state with long gaps
            reset_subjects = set(days_ahead) - set(delta_weeks)
            map(lambda k: subject_state.pop(k, None), reset_subjects)

            state_0 = nn_init(diag_gram, reset_subjects, days_ahead)
            subject_state.update(state_0)
            # This intersection ensures only prediction for:
            # 1. cases that are integrable (i.e. with previous state), and
            # 2. cases that have diagnosis at index n.
            predictable_cases = set(delta_weeks).intersection(diag_out.keys())
            predictable_count += len(predictable_cases)

            # No. of "Predictable" diagnostic points
            diag_weights.append(len(predictable_cases))
            '''
            Idea: scale days_forward to weeks_forward.
            This can:
                1. Improve the numerical stability and accuracy of numerical integration.
                2. Force the ode_dyn model to learn weekly dynamics, which is a suitable time scale for cancer development.
            '''
            odeint_weeks += sum(delta_weeks.values())
            ################## ODEINT #####################
            state_j, _dyn_loss, _nfe = nn_ode(state_i, delta_weeks,
                                              points_n['ode_control'])
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
        return {
            'prejump_diag_loss': prejump_diag_loss,
            'postjump_diag_loss': postjump_diag_loss,
            'dyn_loss': jnp.sum(sum(dyn_loss)),
            'odeint_weeks': odeint_weeks,
            'all_points_count': all_points_count,
            'predictable_count': predictable_count,
            'nfe': sum(nfe),
            'diag_detectability': diag_detectability
        }

    @classmethod
    def create_model(cls, config, patient_interface, train_ids):
        diag_glove, proc_glove = glove_representation(
            diag_idx=patient_interface.diag_multi_ccs_idx,
            proc_idx=patient_interface.proc_multi_ccs_idx,
            ccs_dag=patient_interface.dag,
            subjects=[patient_interface.subjects[i] for i in train_ids],
            **config['glove'])

        diag_gram = DAGGRAM(
            ccs_dag=patient_interface.dag,
            code2index=patient_interface.diag_multi_ccs_idx,
            basic_embeddings=diag_glove,
            ancestors_mat=patient_interface.diag_multi_ccs_ancestors_mat,
            **config['gram']['diag'])

        proc_gram = DAGGRAM(
            ccs_dag=patient_interface.dag,
            code2index=patient_interface.proc_multi_ccs_idx,
            basic_embeddings=proc_glove,
            ancestors_mat=patient_interface.proc_multi_ccs_ancestors_mat,
            **config['gram']['proc'])

        return cls(subject_interface=patient_interface,
                   diag_gram=diag_gram,
                   proc_gram=proc_gram,
                   **config['model'],
                   tay_reg=config['training']['tay_reg'],
                   diag_loss=config['training']['diag_loss'])

    @staticmethod
    def sample_gram_config(trial: optuna.Trial):
        return {
            'diag': DAGGRAM.sample_model_config('dx', trial),
            'proc': DAGGRAM.sample_model_config('pr', trial)
        }

if __name__ == '__main__':
    from .hpo_utils import capture_args, run_trials
    kwargs = {'model_class': SNONETDiagProc, **capture_args()}
    run_trials(**kwargs)
