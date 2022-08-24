from __future__ import annotations

from functools import partial
from typing import Any, Dict, TYPE_CHECKING

import haiku as hk
import jax
import jax.numpy as jnp
from jax import lax

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
        self.init_data = self._initialization_data()

    def init_params(self, prng_seed=0):
        ret = super(ICENODE, self).init_params(prng_seed)
        rng_key = jax.random.PRNGKey(prng_seed)
        ret['pr_emb'] = self.pr_emb.init_params(rng_key)
        return ret

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

    def _extract_nth_admission(self, params: Any,
                               batch: Dict[int, Dict[int, ehr.Admission_JAX]],
                               n: int) -> Dict[str, Dict[int, jnp.ndarray]]:
        ret = super(ICENODE, self)._extract_nth_admission(params=params,
                                                          batch=batch,
                                                          n=n)
        adms = batch[n]
        pr_G = self.dx_emb.compute_embeddings_mat(params["dx_emb"])
        pr_emb = {
            i: self.pr_emb.encode(pr_G, adm.pr_vec)
            for i, adm in adms.items()
        }
        ret['pr_emb'] = pr_emb
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
