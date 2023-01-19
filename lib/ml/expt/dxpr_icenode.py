from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

import haiku as hk
import jax
import jax.numpy as jnp
from absl import logging

if TYPE_CHECKING:
    import optuna

from ... import utils
from ... import ehr
from ... import embeddings as E

from .base_models import (StateIntervenedUpdate)

if TYPE_CHECKING:
    import optuna

from ..dx_icenode_2lr import ICENODE as DX_ICENODE


class ICENODE(DX_ICENODE):

    def __init__(self, pr_emb: E.AbstractEmbeddingsLayer, **kwargs):
        DX_ICENODE.__init__(self, **kwargs)

        self.pr_emb = pr_emb
        self.dimensions['pr_emb'] = pr_emb.embeddings_dim

        f_update_init, f_update = hk.without_apply_rng(
            hk.transform(
                utils.wrap_module(StateIntervenedUpdate,
                                  state_size=self.dimensions['state'],
                                  name='f_update')))
        self.f_update = jax.jit(f_update)
        self.initializers['f_update'] = f_update_init

    def _initialization_data(self):
        """
        Creates data for initializing each of the
        modules based on the shapes of init_data.
        """
        ret = super(ICENODE, self)._initialization_data()
        dx_emb = jnp.zeros(self.dimensions['dx_emb'])
        pr_emb = jnp.zeros(self.dimensions['pr_emb'])
        state = jnp.zeros(self.dimensions['state'])
        ret['f_update'] = [state, dx_emb, dx_emb, pr_emb]

        return ret

    def init_params(self, prng_seed=0):
        ret = super(ICENODE, self).init_params(prng_seed=prng_seed)
        ret['pr_emb'] = self.pr_emb.init_params(prng_seed)
        logging.warning(f'initialized params')
        return ret

    def _extract_nth_admission(self, params: Any,
                               batch: Dict[int, Dict[int, ehr.Admission_JAX]],
                               n: int) -> Dict[str, Dict[int, jnp.ndarray]]:
        ret = super(ICENODE, self)._extract_nth_admission(params=params,
                                                          batch=batch,
                                                          n=n)
        adms = batch[n]
        pr_G = self.pr_emb.compute_embeddings_mat(params["pr_emb"])
        ret['pr_emb'] = {
            i: self.pr_emb.encode(pr_G, adm.pr_vec)
            for i, adm in adms.items()
        }
        return ret

    def _f_update(self, params: Any, state_e: Dict[int, jnp.ndarray],
                  **kwargs) -> jnp.ndarray:
        new_state = {}
        for i in state_e.keys():
            dx_emb = kwargs['dx_emb'][i]
            pr_emb = kwargs['pr_emb'][i]
            state, dx_emb_hat = self.split_state_emb(state_e[i])
            state = self.f_update(params['f_update'], state, dx_emb_hat,
                                  dx_emb, pr_emb)
            new_state[i] = self.join_state_emb(state, dx_emb)
        return new_state

    @classmethod
    def sample_embeddings_config(cls, trial: optuna.Trial, emb_kind: str):
        if emb_kind == 'matrix':
            dx_emb_config = E.MatrixEmbeddings.sample_model_config('dx', trial)
            pr_emb_config = E.MatrixEmbeddings.sample_model_config('pr', trial)
        elif emb_kind == 'gram':
            dx_emb_config = E.GRAM.sample_model_config('dx', trial)
            pr_emb_config = E.GRAM.sample_model_config('pr', trial)
        else:
            raise RuntimeError(f'Unrecognized Embedding kind {emb_kind}')

        return {'dx': dx_emb_config, 'pr': pr_emb_config, 'kind': emb_kind}

    @classmethod
    def create_embedding(cls, category, emb_config, emb_kind,
                         subject_interface, train_ids):
        if emb_kind == 'matrix':
            if category == 'dx':
                input_dim = subject_interface.dx_dim
            else:
                input_dim = subject_interface.pr_dim
            return E.MatrixEmbeddings(input_dim=input_dim, **emb_config)

        if emb_kind == 'gram':
            return E.CachedGRAM(category=category,
                                subject_interface=subject_interface,
                                train_ids=train_ids,
                                **emb_config)
        else:
            raise RuntimeError(f'Unrecognized Embedding kind {emb_kind}')

    @classmethod
    def create_model(cls, config, subject_interface, train_ids):
        dx_emb = cls.create_embedding(category='dx',
                                      emb_config=config['emb']['dx'],
                                      emb_kind=config['emb']['kind'],
                                      subject_interface=subject_interface,
                                      train_ids=train_ids)
        pr_emb = cls.create_embedding(category='pr',
                                      emb_config=config['emb']['pr'],
                                      emb_kind=config['emb']['kind'],
                                      subject_interface=subject_interface,
                                      train_ids=train_ids)
        return cls(subject_interface=subject_interface,
                   dx_emb=dx_emb,
                   pr_emb=pr_emb,
                   **config['model'])
