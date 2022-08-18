from __future__ import annotations

from functools import partial
from typing import Any, Dict, TYPE_CHECKING

import haiku as hk
import jax
from jax import lax

if TYPE_CHECKING:
    import optuna

from .. import utils
from .. import ehr
from .. import embeddings as E

from .base_models import (StateIntervenedUpdate)

if TYPE_CHECKING:
    import optuna

from .dx_icenode_2lr import ICENODE as DX_ICENODE

class ICENODE(DX_ICENODE):

    def __init__(self, pr_emb: E.AbstractEmbeddingsLayer, **kwargs):
        DX_ICENODE.__init__(self, **kwargs)

        f_update_init, f_update = hk.without_apply_rng(
            hk.transform(
                utils.wrap_module(StateIntervenedUpdate,
                                  state_size=self.dimensions['state'],
                                  embeddings_size=self.dimensions['dx_emb'],
                                  name='f_update')))
        self.f_update = jax.jit(f_update)
        self.initializers['f_update'] = f_update_init
        self.init_data = self._initialization_data()


