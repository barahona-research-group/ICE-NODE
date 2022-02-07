from functools import partial
from typing import (Any, Callable, Dict, Iterable, List, Optional, Set)

import jax
from jax.experimental import optimizers

import optuna

from .jax_interface import (DiagnosisJAXInterface)
from .train_icenode_sl import ICENODE as ICENODE_SL
from .gram import AbstractEmbeddingsLayer


class ICENODE(ICENODE_SL):

    def __init__(self, subject_interface: DiagnosisJAXInterface,
                 diag_emb: AbstractEmbeddingsLayer, ode_dyn: str,
                 ode_with_bias: bool, ode_init_var: float,
                 loss_half_life: float, state_size: int, timescale: float):
        super().__init__(subject_interface=subject_interface,
                         diag_emb=diag_emb,
                         ode_dyn=ode_dyn,
                         ode_with_bias=ode_with_bias,
                         ode_init_var=ode_init_var,
                         loss_half_life=loss_half_life,
                         state_size=state_size,
                         timescale=timescale)

    def init_optimizer(self, config, params):
        opt_cls = self.optimizer_class(config['training']['optimizer'])
        lr1 = config['training']['lr']
        lr2 = 10 * lr1
        opt_init, opt_update, get_params = opt_cls(step_size=lr1)
        opt_state = opt_init({'f_n_ode': params['f_n_ode']})
        opt1 = (opt_state, opt_update, get_params)

        opt_init, opt_update, get_params = opt_cls(step_size=lr2)
        opt_state = opt_init({
            'f_dec': params['f_dec'],
            'diag_emb': params['diag_emb'],
            'f_update': params['f_update']
        })
        opt2 = (opt_state, opt_update, get_params)
        return opt1, opt2

    def init_with_params(self, config: Dict[str, Any], params: Any):
        opt1, opt2 = self.init_optimizer(config, params)
        loss_mixing = config['training']['loss_mixing']
        loss_ = partial(self.loss, loss_mixing)
        return opt1, opt2, loss_, loss_mixing

    def get_params(self, model_state):
        opt1, opt2, _, _ = model_state
        opt1_state, _, get_params1 = opt1
        opt2_state, _, get_params2 = opt2
        return {**get_params1(opt1_state), **get_params2(opt2_state)}

    def loss(self, loss_mixing: Dict[str, float], params: Any,
             batch: List[int], **kwargs) -> float:
        res = self(params, batch, **kwargs)
        detailed = self.detailed_loss(loss_mixing, params, res)
        return detailed['loss'], detailed

    def step_optimizer(self, step, model_state, batch):
        opt1, opt2, loss_, loss_mixing = model_state
        opt1_state, opt1_update, get_params1 = opt1
        opt2_state, opt2_update, get_params2 = opt2

        params = self.get_params(model_state)
        grads, detailed = jax.grad(loss_, has_aux=True)(params,
                                                        batch,
                                                        count_nfe=False,
                                                        interval_norm=False)

        grads1 = {'f_n_ode': grads['f_n_ode']}
        grads2 = {
            'f_dec': grads['f_dec'],
            'f_update': grads['f_update'],
            'diag_emb': grads['diag_emb']
        }

        opt1_state = opt1_update(step, grads1, opt1_state)
        opt2_state = opt2_update(step, grads2, opt2_state)

        opt1 = (opt1_state, opt1_update, get_params1)
        opt2 = (opt2_state, opt2_update, get_params2)
        return opt1, opt2, loss_, loss_mixing

    @classmethod
    def _sample_training_config(cls, trial: optuna.Trial, epochs):
        l_mixing = {
            'L_l1': 0,  #trial.suggest_float('l1', 1e-8, 5e-3, log=True),
            'L_l2': 0  # trial.suggest_float('l2', 1e-8, 5e-3, log=True),
        }

        return {
            'epochs': epochs,
            'batch_size': trial.suggest_int('B', 2, 27, 5),
            'loss_half_life': trial.suggest_int('lt0.5', 7, 2e2, log=True),
            # UNDO/TODO
            'optimizer': 'adamax',
            'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
            'loss_mixing': l_mixing
        }


if __name__ == '__main__':
    from .hpo_utils import capture_args, run_trials
    run_trials(model_cls=ICENODE, **capture_args())
