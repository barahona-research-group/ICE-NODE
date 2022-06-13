"""Abstract class for predictive EHR models."""

from typing import Dict, List, Any, Optional
from functools import partial
from absl import logging
import jax
from jax.example_libraries import optimizers
import optuna

from ..utils import (load_config, load_params, parameters_size, tree_hasnan,
                     tree_lognan, write_params, OOPError)
from ..embeddings.gram import (FrozenGRAM, SemiFrozenGRAM, TunableGRAM,
                               GloVeGRAM, MatrixEmbeddings, OrthogonalGRAM)
from ..metric.common_metrics import (bce, softmax_logits_bce,
                                     balanced_focal_bce, weighted_bce,
                                     admissions_auc_scores, codes_auc_scores)
from ..ehr_model.jax_interface import create_patient_interface
from ..ehr_model.ccs_dag import ccs_dag

from .trainer import minibatch_trainer


class AbstractModel:

    def __call__(self, params: Any, subjects_batch: List[int], **kwargs):
        raise OOPError('Should be overriden')

    def detailed_loss(self, loss_mixing, params, res):
        raise OOPError('Should be overriden')

    def eval_stats(self, res):
        raise OOPError('Should be overriden')

    def eval(self, opt_obj: Any, batch: List[int]) -> Dict[str, float]:
        loss_mixing = opt_obj[-1]
        params = self.get_params(opt_obj)
        res = self(params, batch)

        return {
            'loss': self.detailed_loss(loss_mixing, params, res),
            'stats': self.eval_stats(res),
            'risk_prediction': res['risk_prediction']
        }

    def admissions_auc_scores(self, model_state: Any, batch: List[int]):
        params = self.get_params(model_state)
        res = self(params, batch)
        return admissions_auc_scores(res['risk_prediction'])

    def codes_auc_scores(self, model_state: Any, batch: List[int]):
        params = self.get_params(model_state)
        res = self(params, batch)
        return codes_auc_scores(res['risk_prediction'])

    def loss(self, loss_mixing: Dict[str, float], params: Any,
             batch: List[int], **kwargs) -> float:
        res = self(params, batch, **kwargs)
        return self.detailed_loss(loss_mixing, params, res)['loss']

    def init_params(self, prng_seed: int = 0):
        raise OOPError('Should be ovreriden')

    @staticmethod
    def optimizer_class(label: str):
        if label == 'adam':
            return optimizers.adam
        if label == 'sgd':
            return optimizers.sgd
        if label == 'adamax':
            return optimizers.adamax

    @staticmethod
    def lr_schedule(lr, decay_rate):
        if decay_rate is None:
            return lr
        return optimizers.exponential_decay(lr,
                                            decay_steps=50,
                                            decay_rate=decay_rate)

    @classmethod
    def init_optimizer(cls, config, params):
        c = config['training']
        lr = cls.lr_schedule(c['lr'], c['decay_rate'])
        opt_cls = cls.optimizer_class(c['optimizer'])
        opt_init, opt_update, get_params = opt_cls(step_size=lr)
        opt_state = opt_init(params)
        return opt_state, opt_update, get_params

    @classmethod
    def step_optimizer(cls, step, model_state, batch):
        opt_state, opt_update, get_params, loss_, loss_mixing = model_state
        params = get_params(opt_state)
        grads = jax.grad(loss_)(params, batch)
        opt_state = opt_update(step, grads, opt_state)
        return opt_state, opt_update, get_params, loss_, loss_mixing

    def init_with_params(self, config: Dict[str, Any], params: Any):
        opt_state, opt_update, get_params = self.init_optimizer(config, params)
        loss_mixing = config['training']['loss_mixing']
        loss_ = partial(self.loss, loss_mixing)
        return opt_state, opt_update, get_params, loss_, loss_mixing

    def init(self, config: Dict[str, Any], prng_seed: int = 0):
        params = self.init_params(prng_seed)
        return self.init_with_params(config, params)

    @classmethod
    def get_params(cls, opt_object):
        opt_state, _, get_params, _, _ = opt_object
        return get_params(opt_state)

    @classmethod
    def parameters_size(cls, opt_object):
        params = cls.get_params(opt_object)
        return parameters_size(params)

    @classmethod
    def hasnan(cls, opt_obj):
        params = cls.get_params(opt_obj)
        if tree_hasnan(params):
            logging.warning(f'params with NaN: {tree_lognan(params)}')
            return True
        return False

    @classmethod
    def write_params(cls, opt_obj, fname):
        params = cls.get_params(opt_obj)
        write_params(params, fname)

    @classmethod
    def create_embedding(cls, emb_config, emb_kind, patient_interface,
                         train_ids, pretrained_components):
        if emb_kind == 'matrix':
            input_dim = len(ccs_dag.dx_ccs_idx)
            return MatrixEmbeddings(input_dim=input_dim, **emb_config)

        if emb_kind == 'orthogonal_gram':
            return OrthogonalGRAM('dx',
                                  patient_interface=patient_interface,
                                  **emb_config)
        if emb_kind == 'glove_gram':
            return GloVeGRAM(category='dx',
                             patient_interface=patient_interface,
                             train_ids=train_ids,
                             **emb_config)

        if emb_kind in ('semi_frozen_gram', 'frozen_gram', 'tunable_gram'):
            pretrained_components = load_config(pretrained_components)
            emb_component = pretrained_components['emb']['dx']['params_file']
            emb_params = load_params(emb_component)['dx_emb']
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
        return patient_interface.dx_flatccs_by_percentiles(20, train_ids)

    @classmethod
    def create_model(cls, config, patient_interface, train_ids,
                     pretrained_components):
        raise OOPError('Should be overriden')

    @classmethod
    def select_loss(cls, loss_label: str, patient_interface, train_ids):
        if loss_label == 'balanced_focal':
            return lambda t, p: balanced_focal_bce(t, p, gamma=2, beta=0.999)
        elif loss_label == 'softmax_logits_bce':
            return softmax_logits_bce
        elif loss_label == 'bce':
            return bce
        elif loss_label == 'balanced_bce':
            codes_dist = patient_interface.dx_flatccs_frequency_vec(train_ids)
            weights = codes_dist.sum() / (codes_dist + 1e-1) * len(codes_dist)
            return lambda t, logits: weighted_bce(t, logits, weights)
        else:
            raise ValueError(f'Unrecognized dx_loss: {loss_label}')

    @classmethod
    def sample_training_config(cls, trial: optuna.Trial):
        l_mixing = {
            'L_l1': 0,  #trial.suggest_float('l1', 1e-8, 5e-3, log=True),
            'L_l2': 0  # trial.suggest_float('l2', 1e-8, 5e-3, log=True),
        }

        return {
            'epochs': 10,
            'batch_size': trial.suggest_int('B', 2, 27, 5),
            'optimizer': 'adam',
            'lr': trial.suggest_float('lr', 5e-5, 5e-3, log=True),
            'decay_rate': None,
            'loss_mixing': l_mixing
        }

    @classmethod
    def sample_embeddings_config(cls,
                                 trial: optuna.Trial,
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
            gram_component = pretrained_components['emb']['dx']['config_file']
            gram_component = load_config(gram_component)

            emb_config = gram_component['emb']['dx']
        else:
            raise RuntimeError(f'Unrecognized Embedding kind {emb_kind}')

        return {'dx': emb_config, 'kind': emb_kind}

    @classmethod
    def sample_model_config(cls, trial: optuna.Trial):
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

    @staticmethod
    def create_patient_interface(mimic_dir, data_tag: str):
        return create_patient_interface(mimic_dir, data_tag=data_tag)

    @staticmethod
    def get_trainer():
        return minibatch_trainer
