from absl import logging
import pickle
from datetime import datetime
from functools import partial
from typing import (Any, Callable, Dict, Iterable, List, Optional, Tuple, Set)

import pandas as pd
import haiku as hk
import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.tree_util import tree_map

from tqdm import tqdm

from .metrics import (bce, balanced_focal_bce, l2_squared, l1_absolute,
                      evaluation_table, EvalFlag)
from .utils import (parameters_size, wrap_module, load_config)

from .jax_interface import (SubjectJAXInterface, create_patient_interface,
                            Ignore)
from .gram import DAGGRAM
from .models import (MLPDynamics, ResDynamics, GRUDynamics, NeuralODE,
                     DiagnosesUpdate, StateDiagnosesDecoder, StateInitializer)
from .train_snonet import BatchedApply


class SNONETLite(BatchedApply):
    def __init__(self, subject_interface: SubjectJAXInterface,
                 diag_gram: DAGGRAM, proc_gram: DAGGRAM, ode_dyn: str,
                 ode_depth: int, tay_reg: Optional[int], state_size: int,
                 init_depth: bool, bias: bool,
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
            'proc_gram': proc_gram.basic_embeddings_dim,
            'diag_gram': diag_gram.basic_embeddings_dim,
            'proc_dag': len(subject_interface.proc_multi_ccs_idx),
            'diag_dag': len(subject_interface.diag_multi_ccs_idx),
            'diag_out': len(subject_interface.diag_single_ccs_idx),
            'state': state_size,
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
                            name='f_n_ode',
                            tay_reg=tay_reg,
                            **init_kwargs)))
        self.f_n_ode = jax.jit(f_n_ode, static_argnums=(1, ))

        f_update_init, f_update = hk.without_apply_rng(
            hk.transform(
                wrap_module(DiagnosesUpdate,
                            state_size=state_size,
                            with_quad_augmentation=False,
                            name='f_update',
                            **init_kwargs)))
        self.f_update = jax.jit(f_update)

        f_dec_init, f_dec = hk.without_apply_rng(
            hk.transform(
                wrap_module(StateDiagnosesDecoder,
                            hidden_size=self.dimensions['diag_gram'],
                            gram_size=self.dimensions['diag_gram'],
                            output_size=self.dimensions['diag_out'],
                            name='f_dec')))
        self.f_dec = jax.jit(f_dec)

        f_init_init, f_init = hk.without_apply_rng(
            hk.transform(
                wrap_module(StateInitializer,
                            hidden_size=self.dimensions['diag_gram'],
                            state_size=state_size,
                            depth=init_depth,
                            name='f_init')))
        self.f_init = jax.jit(f_init)

        self.initializers = {
            'f_n_ode': f_n_ode_init,
            'f_update': f_update_init,
            'f_dec': f_dec_init,
            'f_init': f_init_init
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
                  diag_gram_error: jnp.ndarray) -> jnp.ndarray:
        updated_state = {
            i: self.f_update(params['f_update'], state[i], gram_error)
            for i, gram_error in diag_gram_error.items()
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
                'value': self.f_init(params['f_init'], state_input),
                'days': days_ahead[subject_id]
            }

        return {i: _state_init(i) for i in (set(diag_gram) & set(subjects))}

    def __call__(self,
                 params: Any,
                 subjects_batch: List[int],
                 return_path: bool = False,
                 count_nfe: bool = False):

        diag_G, proc_G = self._generate_embedding_mats(params)
        nth_points = partial(self._extract_nth_points, params, subjects_batch,
                             diag_G, proc_G)
        nn_ode = partial(self._f_n_ode, params, count_nfe)
        nn_update = partial(self._f_update, params)
        nn_decode = partial(self._f_dec, params)
        nn_init = partial(self._f_init, params)
        diag_loss = partial(self._diag_loss, self.diag_loss)

        subject_state = dict()
        dyn_loss = []
        nfe = []

        prejump_diag_loss = []
        postjump_diag_loss = []
        diag_weights = []

        diag_detectability = {i: {} for i in subjects_batch}
        all_points_count = 0
        predictable_count = 0
        odeint_weeks = 0.0

        for n in self.subject_interface.n_support:
            points_n = nth_points(n)

            if points_n is None:
                continue
            all_points_count += len(points_n)

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

            # From points returned at index n:
            # - Consider for odeint:
            #   1. subjects that already have a previous state, and
            #   2. days difference doesn't exceed maximum days.
            # - For returned subjects with days difference exceeding the
            # threshold, reset their previous state.
            # - Initialize new states for subjects that have diagnosis codes
            #   that has not been previously initialized or has been reset.

            # Reset subjects state with long gaps
            reset_subjects = set(days_ahead) - set(delta_weeks)
            map(lambda k: subject_state.pop(k, None), reset_subjects)

            state_0 = nn_init(diag_gram, age, reset_subjects, days_ahead)
            subject_state.update(state_0)
            # This intersection ensures only prediction for:
            # 1. cases that are integrable (i.e. with previous state), and
            # 2. cases that have diagnosis at index n.
            predictable_cases = set(delta_weeks).intersection(diag_out.keys())
            predictable_count += len(predictable_cases)

            # No. of "Predictable" diagnostic points
            diag_weights.append(len(predictable_cases))
            '''
            Idea: scale days_forward to weeks_forward.
            This can:
                1. Improve the numerical stability and accuracy of numerical integration.
                2. Force the ode_dyn model to learn weekly dynamics, which is a suitable time scale for cancer development.
            '''
            odeint_weeks += sum(delta_weeks.values())
            ################## ODEINT #####################
            state_j, _dyn_loss, _nfe = nn_ode(state_i, delta_weeks,
                                              points_n['ode_control'])
            dyn_loss.append(_dyn_loss)
            nfe.append(_nfe)
            ########## PRE-JUMP DAG LOSS #########################
            pre_diag_gram, pre_diag_out = nn_decode(state_j, predictable_cases)
            pre_diag_loss = diag_loss(diag_out, pre_diag_out)
            pre_diag_gram_error = self._gram_error(diag_gram, pre_diag_gram)
            prejump_diag_loss.append(pre_diag_loss)
            ############## GRU BAYES ####################
            # Using GRUObservationCell to update h.
            state_j_updated = nn_update(state_j, pre_diag_gram_error)
            state_j.update(state_j_updated)
            # Update the states:
            for subject_id, new_state in state_j.items():
                subject_state[subject_id] = {
                    'days': days_ahead[subject_id],
                    'value': new_state
                }

            ############### POST-JUNP DAG LOSS ########################
            _, post_diag_out = nn_decode(state_j_updated, predictable_cases)

            post_diag_loss = diag_loss(diag_out, post_diag_out)
            postjump_diag_loss.append(post_diag_loss)

            for subject_id in post_diag_out.keys():
                diag_detectability[subject_id][n] = {
                    'days_ahead': days_ahead[subject_id],
                    'diag_true': diag_out[subject_id],
                    'pre_logits': pre_diag_out[subject_id],
                    'post_logits': post_diag_out[subject_id]
                }

        prejump_diag_loss = jnp.average(prejump_diag_loss,
                                        weights=diag_weights)
        postjump_diag_loss = jnp.average(postjump_diag_loss,
                                         weights=diag_weights)
        ret = {
            'prejump_diag_loss': prejump_diag_loss,
            'postjump_diag_loss': postjump_diag_loss,
            'dyn_loss': jnp.sum(sum(dyn_loss)),
            'odeint_weeks': odeint_weeks,
            'all_points_count': all_points_count,
            'predictable_count': predictable_count,
            'nfe': sum(nfe),
            'diag_detectability': diag_detectability
        }

        return ret


def eval_fn(ode_model: SNONETLite, loss_mixing: Dict[str, float],
            params: optimizers.Params, batch: List[int]) -> Dict[str, float]:
    res = ode_model(params, batch, count_nfe=True)
    prejump_diag_loss = res['prejump_diag_loss']
    postjump_diag_loss = res['postjump_diag_loss']
    l1_loss = l1_absolute(params)
    l2_loss = l2_squared(params)
    dyn_loss = res['dyn_loss']
    diag_alpha = loss_mixing['L_diag']
    l1_alpha = loss_mixing['L_l1']
    l2_alpha = loss_mixing['L_l2']
    dyn_alpha = loss_mixing['L_dyn'] / (res['odeint_weeks'])

    diag_loss = (
        1 - diag_alpha) * prejump_diag_loss + diag_alpha * postjump_diag_loss
    loss = diag_loss + (l1_alpha * l1_loss) + (l2_alpha * l2_loss) + (
        dyn_alpha * dyn_loss)
    nfe = res['nfe']
    return {
        'loss': {
            'prejump_diag_loss': prejump_diag_loss,
            'postjump_diag_loss': postjump_diag_loss,
            'diag_loss': diag_loss,
            'loss': loss,
            'l1_loss': l1_loss,
            'l2_loss': l2_loss,
            'dyn_loss': dyn_loss,
            'dyn_loss/week': dyn_loss / res['odeint_weeks']
        },
        'stats': {
            'all_points_count': res['all_points_count'],
            'predictable_count': res['predictable_count'],
            'nfe/week': nfe / res['odeint_weeks'],
            'nfex1000': nfe / 1000
        },
        'diag_detectability': res['diag_detectability']
    }


def loss_fn(ode_model: SNONETLite, loss_mixing: Dict[str, float],
            params: optimizers.Params, batch: List[int]) -> float:
    res = ode_model(params, batch, count_nfe=False)

    prejump_diag_loss = res['prejump_diag_loss']
    postjump_diag_loss = res['postjump_diag_loss']
    l1_loss = l1_absolute(params)
    l2_loss = l2_squared(params)
    dyn_loss = res['dyn_loss'] / (res['odeint_weeks'])
    diag_alpha = loss_mixing['L_diag']
    l1_alpha = loss_mixing['L_l1']
    l2_alpha = loss_mixing['L_l2']
    dyn_alpha = loss_mixing['L_dyn']

    diag_loss = (
        1 - diag_alpha) * prejump_diag_loss + diag_alpha * postjump_diag_loss
    loss = diag_loss + (l1_alpha * l1_loss) + (l2_alpha * l2_loss) + (
        dyn_alpha * dyn_loss)

    return loss
