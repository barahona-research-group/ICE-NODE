from absl import logging
import pickle
from datetime import datetime
from functools import partial
from typing import (AbstractSet, Any, Callable, Dict, Iterable, List, Mapping,
                    Optional, Tuple, Union, Set)

import pandas as pd
import haiku as hk
import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.tree_util import tree_map

from tqdm import tqdm

from .jax_interface import (SubjectJAXInterface, create_patient_interface)
from .gram import DAGGRAM
from .models import (MLPDynamics, ResDynamics, GRUDynamics, NeuralODE,
                     GRUBayes, NumericObsModel, StateDiagnosesDecoder,
                     StateInitializer)
from .metrics import (jit_sigmoid, bce, balanced_focal_bce, l2_squared,
                      l1_absolute, numeric_error, lognormal_loss,
                      compute_KL_loss, evaluation_table, EvalFlag)
from .utils import (parameters_size, tree_hasnan, tree_lognan, wrap_module,
                    load_config)


class BatchedApply:
    @staticmethod
    def _numeric_error(mean_true, mean_predicted, logvar_predicted):
        error_num = {
            i: numeric_error(mean_true[i], mean_predicted[i],
                             logvar_predicted[i])
            for i in mean_predicted.keys()
        }
        return error_num

    @staticmethod
    def _lognormal_loss(mask: Dict[int, jnp.ndarray],
                        normal_error: Dict[int, jnp.ndarray],
                        logvar: Dict[int, jnp.ndarray]):
        loss = {
            i: lognormal_loss(mask[i], normal_error[i], logvar[i])
            for i in normal_error.keys()
        }
        if loss:
            return sum(loss.values()) / len(loss)
        else:
            return 0.0

    @staticmethod
    def _kl_loss(mean_true: Dict[int, jnp.ndarray], mask: Dict[int,
                                                               jnp.ndarray],
                 mean_predicted: Dict[int, jnp.ndarray],
                 logvar_predicted: Dict[int, jnp.ndarray]):
        loss = {
            i: compute_KL_loss(mean, mask[i], mean_predicted[i],
                               logvar_predicted[i])
            for i, mean in mean_predicted.items()
        }
        if loss:
            return sum(loss.values()) / len(loss)
        else:
            return 0.0

    @staticmethod
    def _gram_error(gram_true, gram_predicted):
        error_gram = {
            i: gram_true[i] - jit_sigmoid(gram_predicted[i])
            for i in gram_predicted.keys()
        }
        return error_gram

    @staticmethod
    def _diag_loss(loss_apply, diag_true: Dict[int, jnp.ndarray],
                   diag_predicted: Dict[int, jnp.ndarray]):
        loss = {
            i: loss_apply(diag_true[i], diag_predicted[i])
            for i in diag_predicted.keys()
        }
        if loss:
            return sum(loss.values()) / len(loss)
        else:
            return 0.0


class SNONET(BatchedApply):
    def __init__(self, subject_interface: SubjectJAXInterface,
                 diag_gram: DAGGRAM, proc_gram: DAGGRAM, ode_dyn: str,
                 ode_depth: int, tay_reg: Optional[int], state_size: int,
                 numeric_hidden_size: int, init_depth: bool, bias: bool,
                 diag_loss: Callable[[jnp.ndarray, jnp.ndarray], float],
                 max_odeint_days: int, **init_kwargs):

        self.subject_interface = subject_interface
        self.diag_gram = diag_gram
        self.proc_gram = proc_gram
        self.tay_reg = tay_reg
        self.max_odeint_days = max_odeint_days

        if diag_loss == 'balanced_focal':
            self.diag_loss = lambda t, p: balanced_focal_bce(
                t, p, gamma=2, beta=0.999)
        elif diag_loss == 'bce':
            self.diag_loss = bce

        self.dimensions = {
            'age': 1,
            'static': len(subject_interface.static_idx),
            'numeric': len(subject_interface.test_idx),
            'proc_gram': proc_gram.basic_embeddings_dim,
            'diag_gram': diag_gram.basic_embeddings_dim,
            'proc_dag': len(subject_interface.proc_multi_ccs_idx),
            'diag_dag': len(subject_interface.diag_multi_ccs_idx),
            'diag_out': len(subject_interface.diag_single_ccs_idx),
            'state': state_size,
            'numeric_hidden': numeric_hidden_size,
            'ode_depth': ode_depth,
            'init_depth': init_depth
        }

        self.ode_control_passes = ['age', 'static', 'proc_gram']
        self.state_init_passes = ['age', 'static', 'diag_gram']

        self.dimensions.update({
            'ode_control':
            sum(map(self.dimensions.get, self.ode_control_passes)),
            'state_init':
            sum(map(self.dimensions.get, self.state_init_passes))
        })
        """
        Constructs the GRU-ODE-Bayes model with the given dimensions.
        """
        init_kwargs = {
            "with_bias": bias,
            "w_init": hk.initializers.RandomNormal(mean=0, stddev=1e-3),
            "b_init": jnp.zeros
        }
        if ode_dyn == 'gru':
            ode_dyn_cls = GRUDynamics
        elif ode_dyn == 'res':
            ode_dyn_cls = ResDynamics
        elif ode_dyn == 'mlp':
            ode_dyn_cls = MLPDynamics
        else:
            raise RuntimeError(f"Unrecognized dynamics class: {ode_dyn}")

        f_n_ode_init, f_n_ode = hk.without_apply_rng(
            hk.transform(
                wrap_module(NeuralODE,
                            ode_dyn_cls=ode_dyn_cls,
                            state_size=state_size,
                            depth=ode_depth,
                            with_quad_augmentation=False,
                            name='n_ode',
                            tay_reg=tay_reg,
                            **init_kwargs)))
        self.f_n_ode = jax.jit(f_n_ode, static_argnums=(1, ))

        f_update_init, f_update = hk.without_apply_rng(
            hk.transform(
                wrap_module(GRUBayes,
                            state_size=state_size,
                            with_quad_augmentation=False,
                            name='gru_bayes',
                            **init_kwargs)))
        self.f_update = jax.jit(f_update)

        f_num_init, f_num = hk.without_apply_rng(
            hk.transform(
                wrap_module(NumericObsModel,
                            numeric_size=self.dimensions['numeric'],
                            numeric_hidden_size=numeric_hidden_size,
                            with_quad_augmentation=False,
                            name='f_numeric',
                            **init_kwargs)))

        self.f_num = jax.jit(f_num)

        f_dec_init, f_dec = hk.without_apply_rng(
            hk.transform(
                wrap_module(StateDiagnosesDecoder,
                            hidden_size=self.dimensions['diag_gram'],
                            gram_size=self.dimensions['diag_gram'],
                            output_size=self.dimensions['diag_out'],
                            name='f_dec')))
        self.f_dec = jax.jit(f_dec)

        f_state_init_init, f_state_init = hk.without_apply_rng(
            hk.transform(
                wrap_module(StateInitializer,
                            hidden_size=self.dimensions['diag_gram'],
                            state_size=state_size,
                            depth=init_depth,
                            name='f_init')))
        self.f_state_init = jax.jit(f_state_init)

        self.initializers = {
            'f_n_ode': f_n_ode_init,
            'f_update': f_update_init,
            'f_num': f_num_init,
            'f_dec': f_dec_init,
            'f_state_init': f_state_init_init
        }

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

    def state_size(self):
        return self.dimensions['state']

    def diag_out_index(self) -> List[str]:
        index2code = {
            i: c
            for c, i in self.subject_interface.diag_single_ccs_idx.items()
        }
        return list(map(index2code.get, range(len(index2code))))

    def numeric_index(self) -> List[str]:
        index2test = {i: t for t, i in self.subject_interface.test_idx.items()}
        return list(map(index2test.get, range(len(index2test))))

    def _initialization_data(self):
        """
        Creates data for initializing each of the
        modules based on the shapes of init_data.
        """
        numeric = jnp.zeros(self.dimensions['numeric'])
        diag_gram_ = jnp.zeros(self.dimensions['diag_gram'])
        state = jnp.zeros(self.dimensions['state'])
        ode_ctrl = jnp.zeros(self.dimensions['ode_control'])
        state_in = jnp.zeros(self.dimensions['state_init'])
        return {
            "f_n_ode": [True, state, 0.1, ode_ctrl],
            "f_update": [state, numeric, numeric, diag_gram_],
            "f_num": [state],
            "f_dec": [state],
            "f_state_init": [state_in]
        }

    def _extract_nth_points(self, params, subjects_batch: List[int],
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
        numeric = {
            i: v['tests'][0]
            for i, v in points.items() if v['tests'] is not None
        }
        mask = {
            i: v['tests'][1]
            for i, v in points.items() if v['tests'] is not None
        }
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
            'numeric': numeric,
            'mask': mask,
            'ode_control': ode_control,
            'diag_out': diag_out
        }

    def _f_num(
        self, params: Any, state: Dict[str,
                                       jnp.ndarray], selection: Iterable[int]
    ) -> Tuple[Dict[int, jnp.ndarray], Dict[int, jnp.ndarray]]:

        mean_logvar = {
            i: self.f_num(params['f_num'], state[i])
            for i in selection
        }

        mean = {i: m for i, (m, _) in mean_logvar.items()}
        logvar = {i: lv for i, (_, lv) in mean_logvar.items()}
        return mean, logvar

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
                  numeric_error: Dict[int, jnp.ndarray],
                  numeric_mask: Dict[int, jnp.ndarray],
                  diag_gram_error: jnp.ndarray) -> jnp.ndarray:
        zero_gram_error = jnp.zeros(self.dimensions['diag_gram'])
        zero_numeric = jnp.zeros(self.dimensions['numeric'])

        updated_state = {
            i: self.f_update(params['f_update'], state[i],
                             numeric_error.get(i, zero_numeric),
                             numeric_mask.get(i, zero_numeric),
                             diag_gram_error.get(i, zero_gram_error))
            for i in (set(numeric_error) | set(diag_gram_error))
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

    def _f_init(self, params, diag_gram, age, subjects: Iterable[int],
                days_ahead: Dict[int, int]):
        def _state_init(subject_id):
            d = {
                'diag_gram': diag_gram[subject_id],
                'age': age[subject_id],
                'static': self.subject_interface.subject_static(subject_id)
            }
            state_input = jnp.hstack(map(d.get, self.state_init_passes))
            return {
                'value': self.f_state_init(params['f_state_init'],
                                           state_input),
                'days': days_ahead[subject_id]
            }

        return {i: _state_init(i) for i in (set(diag_gram) & set(subjects))}

    def __call__(self, params: Any, subjects_batch: List[int],
                 count_nfe: bool):

        diag_G, proc_G = self._generate_embedding_mats(params)

        nth_points = partial(self._extract_nth_points, params, subjects_batch,
                             diag_G, proc_G)
        nn_num = partial(self._f_num, params)
        nn_ode = partial(self._f_n_ode, params, count_nfe)
        nn_update = partial(self._f_update, params)  # state, e, m, diag_error
        nn_decode = partial(self._f_dec, params)
        nn_init = partial(self._f_init, params)
        diag_loss = partial(self._diag_loss, self.diag_loss)

        subject_state = {}
        dyn_loss = []
        nfe = []

        prejump_diag_loss = []
        postjump_diag_loss = []
        prejump_num_loss = []
        postjump_num_loss = []

        num_weights = []
        diag_weights = []

        diag_detectability = {i: dict() for i in subjects_batch}
        all_points_count = 0
        diag_count = 0
        num_count = 0
        odeint_weeks = 0.0

        for n in self.subject_interface.n_support:
            points_n = nth_points(n)

            if points_n is None:
                continue

            all_points_count += len(points_n)

            numeric = points_n['numeric']
            mask = points_n['mask']
            days_ahead = points_n['days_ahead']
            age = points_n['age']
            diag_gram = points_n['diag_gram']
            diag_out = points_n['diag_out']
            delta_weeks = {
                i: (days_ahead[i] - subject_state[i]['days']) / 7.0
                for i in (set(subject_state) & set(days_ahead))
                if days_ahead[i] -
                subject_state[i]['days'] <= self.max_odeint_days
            }

            state_i = {i: subject_state[i]['value'] for i in delta_weeks}

            # Reset subjects state with long gaps
            reset_subjects = set(days_ahead) - set(delta_weeks)
            map(lambda k: subject_state.pop(k, None), reset_subjects)

            state_0 = nn_init(diag_gram, age, reset_subjects, days_ahead)
            subject_state.update(state_0)
            # This intersection ensures only prediction for:
            # 1. cases that are integrable (i.e. with previous state), and
            # 2. cases that have diagnosis at index n.
            diag_cases = set(delta_weeks).intersection(diag_out.keys())
            diag_count += len(diag_cases)
            # No. of "Predictable" diagnostic points
            diag_weights.append(len(diag_cases))

            num_cases = set(delta_weeks).intersection(numeric.keys())
            num_count = len(num_cases)
            num_weights.append(sum(mask[i].sum() for i in num_cases) + 1e-10)

            odeint_weeks += sum(delta_weeks.values())
            ################## ODEINT #####################
            state_j, _dyn_loss, _nfe = nn_ode(state_i, delta_weeks,
                                              points_n['ode_control'])
            dyn_loss.append(_dyn_loss)
            nfe.append(_nfe)
            ########## PRE-JUMP NUM LOSS ########################
            pre_mean, pre_logvar = nn_num(state_j, num_cases)
            error = self._numeric_error(numeric, pre_mean, pre_logvar)

            prejump_num_loss.append(
                self._lognormal_loss(mask, error, pre_logvar))

            ########## PRE-JUMP DAG LOSS #########################
            pre_diag_gram, pre_diag_out = nn_decode(state_j, diag_cases)
            pre_diag_loss = diag_loss(diag_out, pre_diag_out)
            pre_diag_gram_error = self._gram_error(diag_gram, pre_diag_gram)
            prejump_diag_loss.append(pre_diag_loss)

            ############## GRU BAYES ####################
            # Using GRUObservationCell to update h.
            state_j_updated = nn_update(state_j, error, mask,
                                        pre_diag_gram_error)
            state_j.update(state_j_updated)
            # Update the states:
            for subject_id, new_state in state_j.items():
                subject_state[subject_id] = {
                    'days': days_ahead[subject_id],
                    'value': new_state
                }
            ################ POST-JUMP NUM LOSS ####################
            post_mean, post_logvar = nn_num(state_j_updated, num_cases)
            postjump_num_loss.append(
                self._kl_loss(numeric, mask, post_mean, post_logvar))

            ############### POST-JUNP DAG LOSS ########################
            _, post_diag_out = nn_decode(state_j_updated, diag_cases)

            post_diag_loss = diag_loss(diag_out, post_diag_out)
            postjump_diag_loss.append(post_diag_loss)

            for subject_id in post_diag_out.keys():
                diag_detectability[subject_id][n] = {
                    'days_ahead': days_ahead[subject_id],
                    'diag_true': diag_out[subject_id],
                    'pre_logits': pre_diag_out[subject_id],
                    'post_logits': post_diag_out[subject_id]
                }

        prejump_num_loss = jnp.average(prejump_num_loss, weights=num_weights)
        postjump_num_loss = jnp.average(postjump_num_loss, weights=num_weights)
        prejump_diag_loss = jnp.average(prejump_diag_loss,
                                        weights=diag_weights)
        postjump_diag_loss = jnp.average(postjump_diag_loss,
                                         weights=diag_weights)
        return {
            'prejump_num_loss': prejump_num_loss,
            'postjump_num_loss': postjump_num_loss,
            'prejump_diag_loss': prejump_diag_loss,
            'postjump_diag_loss': postjump_diag_loss,
            'dyn_loss': jnp.sum(sum(dyn_loss)),
            'odeint_weeks': odeint_weeks,
            'all_points_count': all_points_count,
            'diag_count': diag_count,
            'num_count': num_count,
            'nfe': sum(nfe),
            'diag_detectability': diag_detectability,
        }


def eval_fn(ode_model: SNONET, loss_mixing: Dict[str, float],
            params: optimizers.Params, batch: List[int]) -> Dict[str, float]:
    res = ode_model(params, batch, count_nfe=True)

    prejump_num_loss = res['prejump_num_loss']
    postjump_num_loss = res['postjump_num_loss']
    prejump_diag_loss = res['prejump_diag_loss']
    postjump_diag_loss = res['postjump_diag_loss']
    l1_loss = l1_absolute(params)
    l2_loss = l2_squared(params)
    dyn_loss = res['dyn_loss']
    num_alpha = loss_mixing['num_alpha']
    diag_alpha = loss_mixing['diag_alpha']
    ode_alpha = loss_mixing['ode_alpha']
    l1_alpha = loss_mixing['l1_reg']
    l2_alpha = loss_mixing['l2_reg']
    dyn_alpha = loss_mixing['dyn_reg'] / (res['odeint_weeks'])

    num_loss = (1 -
                num_alpha) * prejump_num_loss + num_alpha * postjump_num_loss
    diag_loss = (
        1 - diag_alpha) * prejump_diag_loss + diag_alpha * postjump_diag_loss
    ode_loss = (1 - ode_alpha) * diag_loss + ode_alpha * num_loss
    loss = ode_loss + (l1_alpha * l1_loss) + (l2_alpha *
                                              l2_loss) + (dyn_alpha * dyn_loss)
    nfe = res['nfe']
    return {
        'loss': {
            'prejump_num_loss': prejump_num_loss,
            'postjump_num_loss': postjump_num_loss,
            'prejump_diag_loss': prejump_diag_loss,
            'postjump_diag_loss': postjump_diag_loss,
            'num_loss': num_loss,
            'diag_loss': diag_loss,
            'ode_loss': ode_loss,
            'l1_loss': l1_loss,
            'l2_loss': l2_loss,
            'dyn_loss': dyn_loss,
            'dyn_loss_per_week': dyn_loss / res['odeint_weeks'],
            'loss': loss
        },
        'stats': {
            'all_points_count': res['all_points_count'],
            'diag_count': res['diag_count'],
            'num_count': res['num_count'],
            'nfe/week': nfe / res['odeint_weeks'],
            'nfex1000': nfe / 1000
        },
        'diag_detectability': res['diag_detectability']
    }


def loss_fn(ode_model: SNONET, loss_mixing: Dict[str, float],
            params: optimizers.Params, batch: List[int]):
    res = ode_model(params, batch, count_nfe=False)

    prejump_num_loss = res['prejump_num_loss']
    postjump_num_loss = res['postjump_num_loss']
    prejump_diag_loss = res['prejump_diag_loss']
    postjump_diag_loss = res['postjump_diag_loss']
    l1_loss = l1_absolute(params)
    l2_loss = l2_squared(params)
    dyn_loss = res['dyn_loss'] / (res['odeint_weeks'])
    num_alpha = loss_mixing['num_alpha']
    diag_alpha = loss_mixing['diag_alpha']
    ode_alpha = loss_mixing['ode_alpha']
    l1_alpha = loss_mixing['l1_reg']
    l2_alpha = loss_mixing['l2_reg']
    dyn_alpha = loss_mixing['dyn_reg']

    num_loss = (1 -
                num_alpha) * prejump_num_loss + num_alpha * postjump_num_loss
    diag_loss = (
        1 - diag_alpha) * prejump_diag_loss + diag_alpha * postjump_diag_loss
    ode_loss = (1 - ode_alpha) * diag_loss + ode_alpha * num_loss

    loss = ode_loss + (l1_alpha * l1_loss) + (l2_alpha *
                                              l2_loss) + (dyn_alpha * dyn_loss)

    return loss


def train_ehr(
        subject_interface: SubjectJAXInterface,
        diag_gram: DAGGRAM,
        proc_gram: DAGGRAM,
        rng: Any,
        # Model configurations
        model_config: Dict[str, Any],
        # Training configurations
        train_validation_split: float,
        batch_size: int,
        epochs: int,
        lr: float,
        diag_loss: str,
        tay_reg: Optional[int],
        loss_mixing: Dict[str, float],
        eval_freq: int,
        save_freq: int,
        output_dir: str):

    prng_key = jax.random.PRNGKey(rng.randint(0, 100))
    ode_model = SNONET(subject_interface=subject_interface,
                       diag_gram=diag_gram,
                       proc_gram=proc_gram,
                       **model_config,
                       tay_reg=tay_reg,
                       diag_loss=diag_loss)

    params = ode_model.init_params(prng_key)

    logging.info(f'#params: {parameters_size(params)}')
    logging.info(f'shape(params): {tree_map(jnp.shape, params)}')
    opt_init, opt_update, get_params = optimizers.adam(step_size=lr)

    loss = partial(loss_fn, ode_model, loss_mixing)
    eval_ = partial(eval_fn, ode_model, loss_mixing)

    def update(
            step: int, batch: Iterable[int],
            opt_state: optimizers.OptimizerState) -> optimizers.OptimizerState:
        params = get_params(opt_state)
        grads = jax.grad(loss)(params, batch)
        return opt_update(step, grads, opt_state)

    opt_state = opt_init(params)

    subjects_id = list(subject_interface.subjects.keys())
    rng.shuffle(subjects_id)

    train_ids = subjects_id[:int(train_validation_split * len(subjects_id))]
    valid_ids = subjects_id[int(train_validation_split * len(subjects_id)):]
    batch_size = min(batch_size, len(train_ids))
    val_batch_size = min(batch_size, len(valid_ids))

    codes_by_percentiles = subject_interface.diag_single_ccs_by_percentiles(
        20, train_ids)

    res_val = {}
    res_trn = {}

    iters = int(epochs * len(train_ids) / batch_size)
    val_pbar = tqdm(total=iters)

    for step in range(iters):
        rng.shuffle(train_ids)
        train_batch = train_ids[:batch_size]

        val_pbar.update(1)

        try:
            opt_state = update(step, train_batch, opt_state)
        except (ValueError, FloatingPointError) as e:
            from traceback import format_exception
            tb_str = ''.join(format_exception(None, e, e.__traceback__))
            logging.warning(f'ValueError exception raised: {tb_str}')
            break

        if step % save_freq == 0 and step > 0:
            with open(f'{output_dir}/step{step:03d}.pickle', 'wb') as f:
                pickle.dump(get_params(opt_state), f)

        if step % eval_freq != 0: continue

        rng.shuffle(valid_ids)

        valid_batch = valid_ids  #[:val_batch_size]
        params = get_params(opt_state)
        trn_res = eval_(params, train_batch)
        res_trn[step] = trn_res
        val_res = eval_(params, valid_batch)
        res_val[step] = val_res

        eval_table = evaluation_table(trn_res, val_res,
                                      EvalFlag.POST | EvalFlag.CM,
                                      codes_by_percentiles)
        logging.info('\n' + str(eval_table))

    return {
        'res_val': res_val,
        'res_trn': res_trn,
        'model_params': get_params(opt_state),
        'ode_model': ode_model,
        'trn_ids': train_ids,
        'val_ids': valid_ids
    }


if __name__ == '__main__':
    from pathlib import Path
    import random
    import argparse
    from .glove import glove_representation

    logging.set_verbosity(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--mimic-processed-dir',
                        required=True,
                        help='Absolute path to MIMIC-III processed tables')
    parser.add_argument('-o',
                        '--output-dir',
                        required=True,
                        help='Aboslute path to log intermediate results')

    parser.add_argument(
        '-c',
        '--config',
        required=True,
        help='Absolute path to JSON file of experiment configurations')

    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    mimic_processed_dir = args.mimic_processed_dir
    output_dir = args.output_dir
    cpu = args.cpu
    config_file = args.config

    if cpu:
        jax.config.update('jax_platform_name', 'cpu')
    else:
        jax.config.update('jax_platform_name', 'gpu')

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logging.info('[LOADING] Patients JAX Interface.')
    patient_interface = create_patient_interface(mimic_processed_dir)
    logging.info('[DONE] Patients JAX Interface')

    config = load_config(config_file)

    diag_glove, proc_glove = glove_representation(
        diag_idx=patient_interface.diag_multi_ccs_idx,
        proc_idx=patient_interface.proc_multi_ccs_idx,
        ccs_dag=patient_interface.dag,
        subjects=patient_interface.subjects.values(),
        **config['glove_config'])

    diag_gram = DAGGRAM(
        ccs_dag=patient_interface.dag,
        code2index=patient_interface.diag_multi_ccs_idx,
        basic_embeddings=diag_glove,
        ancestors_mat=patient_interface.diag_multi_ccs_ancestors_mat,
        **config['gram_config']['diag'])

    proc_gram = DAGGRAM(
        ccs_dag=patient_interface.dag,
        code2index=patient_interface.proc_multi_ccs_idx,
        basic_embeddings=proc_glove,
        ancestors_mat=patient_interface.proc_multi_ccs_ancestors_mat,
        **config['gram_config']['proc'])

    res = train_ehr(subject_interface=patient_interface,
                    diag_gram=diag_gram,
                    proc_gram=proc_gram,
                    rng=random.Random(42),
                    model_config=config['model'],
                    **config['training'],
                    train_validation_split=0.8,
                    output_dir=output_dir,
                    eval_freq=10,
                    save_freq=100)
