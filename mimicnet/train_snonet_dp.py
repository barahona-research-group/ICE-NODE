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
                 ode_depth: int, ode_with_bias: bool, ode_init_var: float,
                 ode_timescale: float, tay_reg: int, state_size: int,
                 init_depth: bool,
                 diag_loss: Callable[[jnp.ndarray, jnp.ndarray], float],
                 max_odeint_days: int, **init_kwargs):
        super().__init__(subject_interface=subject_interface,
                         diag_gram=diag_gram,
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
    run_trials(model_cls=SNONETDiagProc, **capture_args())
