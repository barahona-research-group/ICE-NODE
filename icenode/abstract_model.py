from typing import Dict, List, Any, Optional, Iterable
from functools import partial

import jax
from jax.experimental import optimizers
import optuna
from .utils import (load_config, load_params, parameters_size, tree_hasnan,
                    write_params)
from .gram import (FrozenGRAM, SemiFrozenGRAM, TunableGRAM, GloVeGRAM,
                   MatrixEmbeddings, OrthogonalGRAM)
from .metrics import (bce, softmax_logits_bce, balanced_focal_bce,
                      weighted_bce, admissions_auc_scores)


class ImplementationException(Exception):
    pass


class AbstractModel:

    def __call__(self, params: Any, subjects_batch: List[int], **kwargs):
        raise ImplementationException('Should be overriden')

    def detailed_loss(self, loss_mixing, params, res):
        raise ImplementationException('Should be overriden')

    def eval_stats(self, res):
        raise ImplementationException('Should be overriden')

    def eval(self, opt_obj: Any, batch: List[int]) -> Dict[str, float]:
        loss_mixing = opt_obj[-1]
        params = self.get_params(opt_obj)
        res = self(params, batch, count_nfe=True)

        return {
            'loss': self.detailed_loss(loss_mixing, params, res),
            'stats': self.eval_stats(res),
            'diag_detectability': res['diag_detectability']
        }

    def admissions_auc_scores(self, params: Any, batch: List[int]):
        res = self(params, batch)
        return admissions_auc_scores(res['diag_detectability'], 'pre')

    def loss(self, loss_mixing: Dict[str, float], params: Any,
             batch: List[int], **kwargs) -> float:
        res = self(params, batch, **kwargs)
        return self.detailed_loss(loss_mixing, params, res)['loss']

    def init_params(self, prng_seed: int = 0):
        raise ImplementationException('Should be ovreriden')

    def init_optimizer(self, config, params):
        lr = config['training']['lr']
        if config['training']['optimizer'] == 'adam':
            optimizer = optimizers.adam
        else:
            optimizer = optimizers.sgd

        opt_init, opt_update, get_params = optimizer(step_size=lr)
        opt_state = opt_init(params)
        return opt_state, opt_update, get_params

    def step_optimizer(self, step, opt_object, batch):
        opt_state, opt_update, get_params, loss_, loss_mixing = opt_object
        params = get_params(opt_state)
        grads = jax.grad(loss_)(params, batch)
        opt_state = opt_update(step, grads, opt_state)
        return opt_state, opt_update, get_params, loss_, loss_mixing

    def init(self, config: Dict[str, Any], prng_seed: int = 0):
        params = self.init_params(prng_seed)
        opt_state, opt_update, get_params = self.init_optimizer(config, params)

        loss_mixing = config['training']['loss_mixing']
        loss_ = partial(self.loss, loss_mixing)

        return opt_state, opt_update, get_params, loss_, loss_mixing

    def get_params(self, opt_object):
        opt_state, _, get_params, _, _ = opt_object
        return get_params(opt_state)

    def parameters_size(self, opt_object):
        params = self.get_params(opt_object)
        return parameters_size(params)

    def hasnan(self, opt_obj):
        params = self.get_params(opt_obj)
        return tree_hasnan(params)

    def write_params(self, opt_obj, fname):
        params = self.get_params(opt_obj)
        write_params(params, fname)

    @classmethod
    def create_embedding(cls, emb_config, emb_kind, patient_interface,
                         train_ids, pretrained_components):
        if emb_kind == 'matrix':
            input_dim = len(patient_interface.diag_ccs_idx)
            return MatrixEmbeddings(input_dim=input_dim, **emb_config)

        if emb_kind == 'orthogonal_gram':
            return OrthogonalGRAM('diag',
                                  patient_interface=patient_interface,
                                  **emb_config)
        if emb_kind == 'glove_gram':
            return GloVeGRAM(category='diag',
                             patient_interface=patient_interface,
                             train_ids=train_ids,
                             **emb_config)

        if emb_kind in ('semi_frozen_gram', 'frozen_gram', 'tunable_gram'):
            pretrained_components = load_config(pretrained_components)
            emb_component = pretrained_components['emb']['diag']['params_file']
            emb_params = load_params(emb_component)['diag_emb']
            if emb_kind == 'semi_frozen_gram':
                return SemiFrozenGRAM(initial_params=emb_params, **emb_config)
            elif emb_kind == 'frozen_gram':
                return FrozenGRAM(initial_params=emb_params, **emb_config)
            else:
                return TunableGRAM(initial_params=emb_params, **emb_config)
        else:
            raise RuntimeError(f'Unrecognized Embedding kind {emb_kind}')

    @staticmethod
    def code_partitions(patient_interface, train_ids):
        return patient_interface.diag_ccs_by_percentiles(20, train_ids)

    @classmethod
    def create_model(cls, config, patient_interface, train_ids,
                     pretrained_components):
        raise ImplementationException('Should be overriden')

    @classmethod
    def select_loss(cls, loss_label: str, patient_interface, train_ids):
        if loss_label == 'balanced_focal':
            return lambda t, p: balanced_focal_bce(t, p, gamma=2, beta=0.999)
        elif loss_label == 'softmax_logits_bce':
            return softmax_logits_bce
        elif loss_label == 'bce':
            return bce
        elif loss_label == 'balanced_bce':
            codes_dist = patient_interface.diag_ccs_frequency_vec(train_ids)
            weights = codes_dist.sum() / (codes_dist + 1e-1) * len(codes_dist)
            return lambda t, logits: weighted_bce(t, logits, weights)
        else:
            raise ValueError(f'Unrecognized diag_loss: {loss_label}')

    @staticmethod
    def _sample_training_config(trial: optuna.Trial, epochs):
        l_mixing = {
            'L_l1': 0,  #trial.suggest_float('l1', 1e-8, 5e-3, log=True),
            'L_l2': 0  # trial.suggest_float('l2', 1e-8, 5e-3, log=True),
        }

        return {
            'epochs': epochs,
            'batch_size': trial.suggest_int('B', 2, 27, 5),
            # UNDO/TODO
            'optimizer': 'adam',
            # 'optimizer': trial.suggest_categorical('opt', ['adam', 'sgd']),
            'lr': trial.suggest_float('lr', 5e-5, 5e-3, log=True),
            'loss_mixing': l_mixing
        }

    @staticmethod
    def sample_embeddings_config(trial: optuna.Trial,
                                 emb_kind: str,
                                 pretrained_components: Optional[Any] = None):
        if emb_kind == 'matrix':
            emb_config = MatrixEmbeddings.sample_model_config('dx', trial)
        elif emb_kind == 'orthogonal_gram':
            emb_config = OrthogonalGRAM.sample_model_config('dx', trial)
        elif emb_kind == 'glove_gram':
            emb_config = GloVeGRAM.sample_model_config('dx', trial)
        elif emb_kind in ('semi_frozen_gram', 'frozen_gram', 'tunable_gram'):
            pretrained_components = load_config(pretrained_components)
            gram_component = pretrained_components['emb']['diag'][
                'config_file']
            gram_component = load_config(gram_component)

            emb_config = gram_component['emb']['diag']
        else:
            raise RuntimeError(f'Unrecognized Embedding kind {emb_kind}')

        return {'diag': emb_config, 'kind': emb_kind}

    @staticmethod
    def sample_training_config(trial: optuna.Trial):
        raise ImplementationException('Should be overriden')

    @staticmethod
    def sample_model_config(trial: optuna.Trial):
        return {'state_size': trial.suggest_int('s', 100, 350, 50)}

    @classmethod
    def sample_experiment_config(cls, trial: optuna.Trial, emb_kind: str,
                                 pretrained_components: str):
        return {
            'emb':
            cls.sample_embeddings_config(
                trial, emb_kind, pretrained_components=pretrained_components),
            'model':
            cls.sample_model_config(trial),
            'training':
            cls.sample_training_config(trial)
        }
