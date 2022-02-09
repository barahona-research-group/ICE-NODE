from typing import (Any, Callable, Dict, Iterable, List, Optional)

import jax
import jax.numpy as jnp
import optuna

from .utils import tree_map
from .metrics import l2_squared
from .jax_interface import (DiagnosisJAXInterface)
from .train_icenode_tl import ICENODE as ICENODE_TL
from .gram import AbstractEmbeddingsLayer


class ICENODE(ICENODE_TL):

    def __init__(self, subject_interface: DiagnosisJAXInterface,
                 diag_emb: AbstractEmbeddingsLayer, ode_dyn: str,
                 ode_with_bias: bool, ode_init_var: float, state_size: int,
                 timescale: float):
        super().__init__(subject_interface=subject_interface,
                         diag_emb=diag_emb,
                         ode_dyn=ode_dyn,
                         ode_with_bias=ode_with_bias,
                         ode_init_var=ode_init_var,
                         state_size=state_size,
                         timescale=timescale)

    def step_optimizer(self, step, model_state, batch):
        opt_state, opt_update, get_params, loss_, loss_mixing = model_state
        params = get_params(opt_state)
        grads = jax.grad(loss_)(params,
                                batch,
                                count_nfe=False,
                                interval_norm=False)
        grads1 = {'f_n_ode': grads['f_n_ode']}
        grads2 = {
            'f_dec': grads['f_dec'],
            'f_update': grads['f_update'],
            'diag_emb': grads['diag_emb']
        }

        grads1_norm = l2_squared(grads1)
        grads2_norm = l2_squared(grads2)
        rescale1 = jnp.sqrt(1. + grads2_norm / grads1_norm)
        rescale2 = jnp.sqrt(1. + grads1_norm / grads2_norm)

        grads1 = tree_map(lambda g: g * rescale1, grads1)
        grads2 = tree_map(lambda g: g * rescale2, grads2)

        opt_state = opt_update(step, {**grads1, **grads2}, opt_state)
        return opt_state, opt_update, get_params, loss_, loss_mixing

    @classmethod
    def sample_training_config(cls, trial: optuna.Trial):
        return {
            'epochs': 20,
            'batch_size': trial.suggest_int('B', 2, 27, 5),
            'optimizer': 'adam',
            'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
            'decay_rate': trial.suggest_float('dr', 1e-1, 9e-1),
            'loss_mixing': {
                'L_l1': 0,  #trial.suggest_float('l1', 1e-8, 5e-3, log=True),
                'L_l2': 0,  # trial.suggest_float('l2', 1e-8, 5e-3, log=True),
                'L_dyn': 0  # trial.suggest_float('L_dyn', 1e-6, 1, log=True)
            }
        }

    @classmethod
    def sample_model_config(cls, trial: optuna.Trial):
        return {
            'ode_dyn': 'mlp',
            'ode_with_bias': False,
            'ode_init_var': trial.suggest_float('ode_i', 1e-10, 1e-1,
                                                log=True),
            'state_size': trial.suggest_int('s', 10, 100, 10),
            'timescale': 60
        }


if __name__ == '__main__':
    from .hpo_utils import capture_args, run_trials
    run_trials(model_cls=ICENODE, **capture_args())
