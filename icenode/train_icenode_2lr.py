from functools import partial
from typing import (Any, Callable, Dict, Iterable, List, Optional, Set)

import jax
from jax.experimental import optimizers

import optuna

from .jax_interface import (DiagnosisJAXInterface)
from .train_icenode_tl import ICENODE as ICENODE_TL
from .gram import AbstractEmbeddingsLayer


class ICENODE(ICENODE_TL):

    @classmethod
    def init_optimizer(cls, config, params):
        c = config['training']
        opt_cls = cls.optimizer_class(c['optimizer'])
        lr1 = cls.lr_schedule(c['lr1'], c['decay_rate1'])
        lr2 = cls.lr_schedule(c['lr2'], c['decay_rate2'])

        opt_init, opt_update, get_params = opt_cls(step_size=lr1)
        opt_state = opt_init({'f_n_ode': params['f_n_ode']})
        opt1 = (opt_state, opt_update, get_params)

        opt_init, opt_update, get_params = opt_cls(step_size=lr2)
        opt_state = opt_init({
            'f_dec': params['f_dec'],
            'diag_emb': params['diag_emb'],
            'f_update': params.get('f_update')
        })
        opt2 = (opt_state, opt_update, get_params)
        return opt1, opt2

    def init_with_params(self, config: Dict[str, Any], params: Any):
        opt1, opt2 = self.init_optimizer(config, params)
        loss_mixing = config['training']['loss_mixing']
        loss_ = partial(self.loss, loss_mixing)
        return opt1, opt2, loss_, loss_mixing

    @classmethod
    def get_params(cls, model_state):
        opt1, opt2, _, _ = model_state
        opt1_state, _, get_params1 = opt1
        opt2_state, _, get_params2 = opt2
        return {**get_params1(opt1_state), **get_params2(opt2_state)}

    @classmethod
    def step_optimizer(cls, step, model_state, batch):
        opt1, opt2, loss_, loss_mixing = model_state
        opt1_state, opt1_update, get_params1 = opt1
        opt2_state, opt2_update, get_params2 = opt2

        params = cls.get_params(model_state)
        grads = jax.grad(loss_)(params, batch)

        grads1 = {'f_n_ode': grads['f_n_ode']}
        grads2 = {
            'f_dec': grads['f_dec'],
            'f_update': grads.get('f_update'),
            'diag_emb': grads['diag_emb']
        }

        opt1_state = opt1_update(step, grads1, opt1_state)
        opt2_state = opt2_update(step, grads2, opt2_state)

        opt1 = (opt1_state, opt1_update, get_params1)
        opt2 = (opt2_state, opt2_update, get_params2)
        return opt1, opt2, loss_, loss_mixing

    @classmethod
    def sample_training_config(cls, trial: optuna.Trial):
        return {
            'epochs': 25,
            'batch_size': 2**trial.suggest_int('Bexp', 1, 7),
            'optimizer': 'adam',
            'lr1': trial.suggest_float('lr1', 1e-5, 1e-2, log=True),
            'lr2': trial.suggest_float('lr2', 1e-5, 1e-2, log=True),
            'decay_rate1': trial.suggest_float('dr1', 1e-1, 9e-1),
            'decay_rate2': trial.suggest_float('dr2', 1e-1, 9e-1),
            'loss_mixing': {
                'L_l1': 0,  #trial.suggest_float('l1', 1e-8, 5e-3, log=True),
                'L_l2': 0,  # trial.suggest_float('l2', 1e-8, 5e-3, log=True),
                'L_dyn': 0  # trial.suggest_float('L_dyn', 1e-6, 1, log=True)
            }
        }


if __name__ == '__main__':
    from .hpo_utils import capture_args, run_trials
    run_trials(model_cls=ICENODE, **capture_args())
