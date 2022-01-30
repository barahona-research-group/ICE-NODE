from functools import partial
from typing import (Any, Callable, Dict, Iterable, List, Optional, Tuple, Set)

import jax.numpy as jnp

from .jax_interface import SubjectJAXInterface
from .gram import DAGGRAM
from .train_snonet_diag import SNONETDiag


class SNONETDiagStat(SNONETDiag):
    def __init__(self, subject_interface: SubjectJAXInterface,
                 diag_gram: DAGGRAM, ode_dyn: str, ode_depth: int,
                 ode_with_bias: bool, ode_init_var: float,
                 ode_timescale: float, tay_reg: Optional[int], state_size: int,
                 init_depth: bool, diag_loss: str, max_odeint_days: int):
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

        self.dimensions['age'] = 1
        self.dimensions['static'] = len(subject_interface.static_idx)

        self.ode_control_passes = ['age', 'static']
        self.state_init_passes = ['age', 'static', 'diag_gram']

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
        age = {i: v['age'] for i, v in points.items()}

        def _ode_control(subject_id):
            d = {
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

    def _f_init(self, params, points_n, subjects: Iterable[int],
                days_ahead: Dict[int, int]):
        diag_gram = points_n['diag_gram']
        age = points_n['age']

        def _state_init(subject_id):
            d = {
                'diag_gram': diag_gram[subject_id],
                'age': age[subject_id],
                'static': self.subject_interface.subject_static(subject_id)
            }
            state_input = jnp.hstack(map(d.get, self.state_init_passes))
            return {
                'value': self.f_init(params['f_init'], state_input),
                'days': days_ahead[subject_id]
            }

        return {i: _state_init(i) for i in (set(diag_gram) & set(subjects))}


if __name__ == '__main__':
    from .hpo_utils import capture_args, run_trials
    run_trials(model_cls=SNONETDiagStat, **capture_args())
