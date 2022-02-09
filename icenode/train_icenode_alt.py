from typing import (Any, Callable, Dict, Iterable, List, Optional, Set)

import jax
import jax.numpy as jnp
import optuna

from .jax_interface import (DiagnosisJAXInterface)
from .gram import AbstractEmbeddingsLayer
from .train_icenode_2lr import ICENODE as ICENODE_2LR


class ICENODE(ICENODE_2LR):

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

    @classmethod
    def smooth_square_sin(cls, i, period, eps=0.5):
        a = (i / period) * jnp.pi
        return jnp.sin(a) / jnp.sqrt(jnp.sin(a)**2 + eps)

    @classmethod
    def sin_lr(cls, lr1, lr2, period):

        def lr1_fn(i):
            return lr1 * (1 + cls.smooth_square_sin(i, period)) / 2 + 1e-3

        def lr2_fn(i):
            return lr2 * (1 - cls.smooth_square_sin(i, period)) / 2 + 1e-3

        return lr1_fn, lr2_fn

    def init_optimizer(self, config, params):
        c = config['training']
        opt_cls = self.optimizer_class(c['optimizer'])
        lr1, lr2 = self.sin_lr(c['lr1'], c['lr2'], 33)

        # Alternating updates
        lr1_alt = lambda i: 0 if i % 2 == 0 and i > 5 else lr1(i)
        lr2_alt = lambda i: 0 if i % 2 == 1 and i > 5 else lr2(i)

        opt_init, opt_update, get_params = opt_cls(step_size=lr1_alt)
        opt_state = opt_init({'f_n_ode': params['f_n_ode']})
        opt1 = (opt_state, opt_update, get_params)

        opt_init, opt_update, get_params = opt_cls(step_size=lr2_alt)
        opt_state = opt_init({
            'f_dec': params['f_dec'],
            'diag_emb': params['diag_emb'],
            'f_update': params['f_update']
        })
        opt2 = (opt_state, opt_update, get_params)
        return opt1, opt2

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

    @classmethod
    def sample_training_config(cls, trial: optuna.Trial):
        return {
            'epochs': 20,
            'batch_size': trial.suggest_int('B', 2, 27, 5),
            'optimizer': 'adam',
            'lr1': trial.suggest_float('lr1', 1e-5, 1e-2, log=True),
            'lr2': trial.suggest_float('lr2', 1e-5, 1e-2, log=True),
            'loss_mixing': {
                'L_l1': 0,  #trial.suggest_float('l1', 1e-8, 5e-3, log=True),
                'L_l2': 0,  # trial.suggest_float('l2', 1e-8, 5e-3, log=True),
                'L_dyn': 0  # trial.suggest_float('L_dyn', 1e-6, 1, log=True)
            }
        }


if __name__ == '__main__':
    from .hpo_utils import capture_args, run_trials
    run_trials(model_cls=ICENODE, **capture_args())
