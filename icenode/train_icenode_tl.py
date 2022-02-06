from functools import partial
from typing import (Any, Callable, Dict, Iterable, List, Optional, Tuple, Set)

from absl import logging
import haiku as hk
import jax
from jax.experimental import optimizers
import jax.numpy as jnp

import optuna

from .metrics import (l2_squared, l1_absolute, softmax_logits_bce)
from .utils import wrap_module, tree_map
from .jax_interface import (DiagnosisJAXInterface, create_patient_interface)
from .models import (MLPDynamics, ResDynamics, GRUDynamics, NeuralODE,
                     EmbeddingsDecoder_Logits, StateUpdate)
from .abstract_model import AbstractModel
from .gram import AbstractEmbeddingsLayer


class ICENODE(AbstractModel):

    def __init__(self, subject_interface: DiagnosisJAXInterface,
                 diag_emb: AbstractEmbeddingsLayer, ode_dyn: str,
                 ode_with_bias: bool, ode_init_var: float, state_size: int,
                 timescale: float):

        self.subject_interface = subject_interface
        self.diag_emb = diag_emb
        self.dimensions = {
            'diag_emb': diag_emb.embeddings_dim,
            'diag_out': len(subject_interface.diag_ccs_idx),
            'state': state_size
        }
        if ode_dyn == 'gru':
            ode_dyn_cls = GRUDynamics
        elif ode_dyn == 'res':
            ode_dyn_cls = ResDynamics
        elif ode_dyn == 'mlp':
            ode_dyn_cls = MLPDynamics
        else:
            raise RuntimeError(f"Unrecognized dynamics class: {ode_dyn}")
        state_emb_size = self.dimensions['diag_emb'] + state_size

        f_n_ode_init, f_n_ode = hk.without_apply_rng(
            hk.transform(
                wrap_module(NeuralODE,
                            ode_dyn_cls=ode_dyn_cls,
                            state_size=state_emb_size,
                            depth=1,
                            timescale=timescale,
                            with_bias=ode_with_bias,
                            init_var=ode_init_var,
                            name='f_n_ode',
                            tay_reg=3)))
        self.f_n_ode = jax.jit(f_n_ode, static_argnums=(1, 2))

        f_update_init, f_update = hk.without_apply_rng(
            hk.transform(
                wrap_module(StateUpdate,
                            state_size=state_size,
                            embeddings_size=self.dimensions['diag_emb'],
                            name='f_update')))
        self.f_update = jax.jit(f_update)

        f_dec_init, f_dec = hk.without_apply_rng(
            hk.transform(
                wrap_module(EmbeddingsDecoder_Logits,
                            n_layers=2,
                            embeddings_size=self.dimensions['diag_emb'],
                            diag_size=self.dimensions['diag_out'],
                            name='f_dec')))
        self.f_dec = jax.jit(f_dec)

        self.initializers = {
            'f_n_ode': f_n_ode_init,
            'f_update': f_update_init,
            'f_dec': f_dec_init
        }
        self.init_data = self._initialization_data()

    def join_state_emb(self, state, emb):
        if state is None:
            state = jnp.zeros((self.dimensions['state'], ))
        return jnp.hstack((state, emb))

    def split_state_emb(self, state_emb):
        return jnp.split(state_emb, (self.dimensions['state'], ))

    def init_params(self, prng_seed=0):
        rng_key = jax.random.PRNGKey(prng_seed)
        return {
            "diag_emb": self.diag_emb.init_params(rng_key),
            **{
                label: init(rng_key, *self.init_data[label])
                for label, init in self.initializers.items()
            }
        }

    def state_size(self):
        return self.dimensions['state']

    def diag_out_index(self) -> List[str]:
        index2code = {
            i: c
            for c, i in self.subject_interface.diag_ccs_idx.items()
        }
        return list(map(index2code.get, range(len(index2code))))

    def _initialization_data(self):
        """
        Creates data for initializing each of the
        modules based on the shapes of init_data.
        """
        emb = jnp.zeros(self.dimensions['diag_emb'])
        state = jnp.zeros(self.dimensions['state'])
        state_emb = jnp.hstack((state, emb))
        return {
            "f_n_ode": [2, True, state_emb, 0.1],
            "f_update": [state, emb, emb],
            "f_dec": [emb],
        }

    def _extract_nth_admission(self, params: Any, subjects_batch: List[int],
                               n: int) -> Dict[str, Dict[int, jnp.ndarray]]:
        diag_G = self.diag_emb.compute_embeddings_mat(params["diag_emb"])

        adms = self.subject_interface.nth_admission_batch(n, subjects_batch)
        if len(adms) == 0:
            return None

        diag_emb = {
            i: self.diag_emb.encode(diag_G, v['diag_ccs_vec'])
            for i, v in adms.items() if v['diag_ccs_vec'] is not None
        }
        diag_out = {
            i: v['diag_ccs_vec']
            for i, v in adms.items() if v['diag_ccs_vec'] is not None
        }
        los = {i: v['los'] for i, v in adms.items()}
        adm_id = {i: v['admission_id'] for i, v in adms.items()}
        adm_time = {i: v['time'] for i, v in adms.items()}

        return {
            'time': adm_time,
            'los': los,
            'diag_emb': diag_emb,
            'diag_out': diag_out,
            'admission_id': adm_id
        }

    def _f_n_ode(self, params, count_nfe, state_e, t):
        h_r_nfe = {
            i: self.f_n_ode(params['f_n_ode'], 2, False, state_e[i], t[i])
            for i in t
        }
        if count_nfe:
            n = {
                i: self.f_n_ode(params['f_n_ode'], 2, True, state_e[i],
                                t[i])[-1]
                for i in t
            }
        else:
            n = {i: 0 for i in t}
        r = {i: r.squeeze() for i, (_, r, _) in h_r_nfe.items()}
        state_e = {i: h[-1, :] for i, (h, _, _) in h_r_nfe.items()}
        return state_e, r, n

    def _f_update(self, params: Any, state_e: Dict[int, jnp.ndarray],
                  emb: jnp.ndarray) -> jnp.ndarray:
        new_state = {}
        for i in emb:
            emb_nominal = emb[i]
            state, emb_pred = self.split_state_emb(state_e[i])
            state = self.f_update(params['f_update'], state, emb_pred,
                                  emb_nominal)
            new_state[i] = self.join_state_emb(state, emb_nominal)
        return new_state

    def _f_dec(self, params: Any, state_e: Dict[int, jnp.ndarray]):
        emb = {i: self.split_state_emb(state_e[i])[1] for i in state_e}
        return {i: self.f_dec(params['f_dec'], emb[i]) for i in emb}

    def _diag_loss(self, diag: Dict[int, jnp.ndarray],
                   dec_diag: Dict[int, jnp.ndarray], t: Dict[int,
                                                             jnp.ndarray]):
        T = [1 / ti for ti in sorted(t.keys())]
        l = [
            softmax_logits_bce(diag[i], dec_diag[i])
            for i in sorted(diag.keys())
        ]
        l_norm = jnp.average(l, weights=T)
        return l_norm, sum(l) / len(l)

    def __call__(self,
                 params: Any,
                 subjects_batch: List[int],
                 count_nfe: bool,
                 interval_norm: bool = False):
        nth_adm = partial(self._extract_nth_admission, params, subjects_batch)
        nn_ode = partial(self._f_n_ode, params, count_nfe)
        nn_update = partial(self._f_update, params)
        nn_decode = partial(self._f_dec, params)
        diag_loss = self._diag_loss

        adm0 = nth_adm(0)
        subject_state = {
            i: {
                'state_e': self.join_state_emb(None, adm0['diag_emb'][i]),
                'time': adm0['time'][i] + adm0['los'][i]
            }
            for i in adm0['admission_id']
        }

        total_nfe = 0

        prediction_losses = []

        adm_counts = []
        diag_detectability = {i: {} for i in subjects_batch}
        odeint_time = []
        dyn_loss = 0

        for n in self.subject_interface.n_support[1:]:
            adm_n = nth_adm(n)
            if adm_n is None:
                break
            adm_id = adm_n['admission_id']
            adm_los = adm_n['los']  # length of stay
            adm_time = adm_n['time']
            emb = adm_n['diag_emb']
            diag = adm_n['diag_out']

            adm_counts.append(len(adm_id))

            state_e = {i: subject_state[i]['state_e'] for i in adm_id}

            d2d_time = {
                i: adm_time[i] + adm_los[i] - subject_state[i]['time']
                for i in adm_id
            }

            # Integrate until next discharge
            state_e, r, nfe = nn_ode(state_e, d2d_time)
            dec_diag = nn_decode(state_e)

            for subject_id in state_e.keys():
                diag_detectability[subject_id][n] = {
                    'admission_id': adm_id[subject_id],
                    'nfe': nfe[subject_id],
                    'time': adm_time[subject_id] + adm_los[subject_id],
                    'los': adm_los[subject_id],
                    'diag_true': diag[subject_id],
                    'pre_logits': dec_diag[subject_id]
                }

            odeint_time.append(sum(d2d_time.values()))
            dyn_loss += sum(r.values())

            pred_loss_norm, pred_loss_avg = diag_loss(diag, dec_diag, d2d_time)
            if interval_norm:
                prediction_losses.append(pred_loss_norm)
            else:
                prediction_losses.append(pred_loss_avg)

            total_nfe += sum(nfe.values())

            # Update state at discharge
            state_e = nn_update(state_e, emb)

            # Update the states:
            for subject_id, new_state in state_e.items():
                subject_state[subject_id] = {
                    'time': adm_time[subject_id] + adm_los[subject_id],
                    'state_e': new_state
                }

        prediction_loss = jnp.average(prediction_losses, weights=adm_counts)

        ret = {
            'prediction_loss': prediction_loss,
            'dyn_loss': dyn_loss,
            'odeint_weeks': sum(odeint_time) / 7.0,
            'admissions_count': sum(adm_counts),
            'nfe': total_nfe,
            'diag_detectability': diag_detectability
        }

        return ret

    def detailed_loss(self, loss_mixing, params, res):
        prediction_loss = res['prediction_loss']
        dyn_loss = res['dyn_loss']
        dyn_alpha = loss_mixing['L_dyn']

        loss = prediction_loss + (dyn_alpha * dyn_loss) / res['odeint_weeks']
        return {
            'loss': loss,
            'prediction_loss': res['prediction_loss'],
            'dyn_loss': dyn_loss,
            'admissions_count': res['admissions_count'],
            'odeint_weeks': res['odeint_weeks']
        }

    def eval_stats(self, res):
        nfe = res['nfe']
        return {
            'admissions_count': res['admissions_count'],
            'nfe_per_week': nfe / res['odeint_weeks'],
            'Kfe': nfe / 1000
        }

    def init_optimizer(self, config, params):
        lr = config['training']['lr']
        opt_init, opt_update, get_params = optimizers.adam(step_size=lr)
        opt_state = opt_init({'f_n_ode': params['f_n_ode']})
        opt1 = (opt_state, opt_update, get_params)

        opt_init, opt_update, get_params = optimizers.adam(step_size=lr)
        opt_state = opt_init({
            'f_dec': params['f_dec'],
            'diag_emb': params['diag_emb'],
            'f_update': params['f_update']
        })
        opt2 = (opt_state, opt_update, get_params)
        return opt1, opt2

    def init(self, config: Dict[str, Any], prng_seed: int = 0):
        params = self.init_params(prng_seed)
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

        grads1 = tree_map(lambda g: g / detailed['odeint_weeks'],
                          {'f_n_ode': grads['f_n_ode']})
        grads2 = tree_map(
            lambda g: g / detailed['admissions_count'], {
                'f_dec': grads['f_dec'],
                'f_update': grads['f_update'],
                'diag_emb': grads['diag_emb']
            })

        opt1_state = opt1_update(step, grads1, opt1_state)
        opt2_state = opt2_update(step, grads2, opt2_state)

        opt1 = (opt1_state, opt1_update, get_params1)
        opt2 = (opt2_state, opt2_update, get_params2)
        return opt1, opt2, loss_, loss_mixing

    @staticmethod
    def create_patient_interface(mimic_dir, data_tag: str):
        return create_patient_interface(mimic_dir, data_tag=data_tag)

    @classmethod
    def create_model(cls, config, patient_interface, train_ids,
                     pretrained_components):
        diag_emb = cls.create_embedding(
            emb_config=config['emb']['diag'],
            emb_kind=config['emb']['kind'],
            patient_interface=patient_interface,
            train_ids=train_ids,
            pretrained_components=pretrained_components)
        return cls(subject_interface=patient_interface,
                   diag_emb=diag_emb,
                   **config['model'])

    @staticmethod
    def _sample_ode_training_config(trial: optuna.Trial, epochs):
        config = AbstractModel._sample_training_config(trial, epochs)
        config['loss_mixing'][
            'L_dyn'] = 0  # trial.suggest_float('L_dyn', 1e-6, 1, log=True)
        return config

    @staticmethod
    def sample_training_config(trial: optuna.Trial):
        return ICENODE._sample_ode_training_config(trial, epochs=20)

    @staticmethod
    def _sample_ode_model_config(trial: optuna.Trial):
        model_params = {
            'ode_dyn': trial.suggest_categorical('ode_dyn', ['mlp', 'gru']),
            'ode_with_bias': False,
            'ode_init_var': trial.suggest_float('ode_i', 1e-10, 1e-1,
                                                log=True),
            'state_size': trial.suggest_int('s', 10, 100, 10),
            'timescale': 7
        }
        return model_params

    @classmethod
    def sample_model_config(cls, trial: optuna.Trial):
        return cls._sample_ode_model_config(trial)


if __name__ == '__main__':
    from .hpo_utils import capture_args, run_trials
    from jax import config
    config.update('jax_debug_nans', True)
    run_trials(model_cls=ICENODE, **capture_args())
