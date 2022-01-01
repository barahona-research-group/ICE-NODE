from functools import partial
from typing import (Any, Callable, Dict, Iterable, List, Optional, Tuple, Set)

import jax.numpy as jnp

from .jax_interface import SubjectJAXInterface
from .gram import DAGGRAM
from .train_snonet_dp import SNONETDiagProc


class SNONETLite(SNONETDiagProc):
    def __init__(self, subject_interface: SubjectJAXInterface,
                 diag_gram: DAGGRAM, proc_gram: DAGGRAM, ode_dyn: str,
                 ode_depth: int, ode_with_bias: bool, ode_init_var: float,
                 ode_timescale: float, tay_reg: Optional[int], state_size: int,
                 init_depth: bool,
                 diag_loss: Callable[[jnp.ndarray, jnp.ndarray], float],
                 max_odeint_days: int, **init_kwargs):
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

        self.dimensions['age'] = 1
        self.dimensions['static'] = len(subject_interface.static_idx)

        self.ode_control_passes = ['age', 'static', 'proc_gram']
        self.state_init_passes = ['age', 'static', 'diag_gram']

        self.dimensions.update({
            'ode_control':
            sum(map(self.dimensions.get, self.ode_control_passes)),
            'state_init':
            sum(map(self.dimensions.get, self.state_init_passes))
        })

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

    def _f_init(self, params, points_n, subjects: Iterable[int],
                days_ahead: Dict[int, int]):
        diag_gram = points_n['diag_gram']
        age = points_n['diag_gram']

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
    run_trials(model_cls=SNONETLite, **capture_args())
