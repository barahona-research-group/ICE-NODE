from typing import Optional, Callable, Any, Dict, Tuple
from functools import partial

import jax
import jax.numpy as jnp
import haiku as hk

from .train_icenode import ICENODE
from .jax_interface import DiagnosisJAXInterface
from .gram import AbstractEmbeddingsLayer
from .utils import wrap_module
from .models import DiagnosticSamplesCombine


class ICENODE_A(ICENODE):

    def __init__(self, subject_interface: DiagnosisJAXInterface,
                 diag_emb: AbstractEmbeddingsLayer, ode_dyn: str,
                 ode_depth: int, ode_with_bias: bool, ode_init_var: float,
                 ode_timescale: float, trajectory_sample_rate: int,
                 tay_reg: Optional[int], state_size: int, init_depth: bool,
                 diag_loss: Callable[[jnp.ndarray, jnp.ndarray], float]):

        super().__init__(subject_interface=subject_interface,
                         diag_emb=diag_emb,
                         ode_dyn=ode_dyn,
                         ode_depth=ode_depth,
                         ode_with_bias=ode_with_bias,
                         ode_init_var=ode_init_var,
                         ode_timescale=ode_timescale,
                         trajectory_sample_rate=trajectory_sample_rate,
                         tay_reg=tay_reg,
                         state_size=state_size,
                         init_depth=init_depth,
                         diag_loss=diag_loss)
        f_combine_init, f_combine = hk.without_apply_rng(
            hk.transform(
                wrap_module(DiagnosticSamplesCombine,
                            embeddings_size=self.dimensions['diag_emb'],
                            name='f_combine')))
        self.f_combine = jax.jit(f_combine)

        self.initializers['f_combine'] = f_combine_init
        self.init_data['f_combine'] = [
            jnp.zeros((3, self.dimensions['diag_emb'])),
            jnp.zeros((3, self.dimensions['diag_out']))
        ]

    def _f_dec_seq(
        self, params: Any, state_seq: Dict[int, jnp.ndarray]
    ) -> Tuple[Dict[int, jnp.ndarray], Dict[int, jnp.ndarray]]:

        dec_seq = {
            i: jax.vmap(partial(self.f_dec, params['f_dec']))(state)
            for i, state in state_seq.items()
        }
        dec = {
            i: self.f_combine(params['f_combine'], e_seq, d_seq)
            for i, (e_seq, d_seq) in dec_seq.items()
        }
        emb = {i: e for i, (e, d) in dec.items()}
        diag = {i: d for i, (e, d) in dec.items()}
        return emb, diag


if __name__ == '__main__':
    from .hpo_utils import capture_args, run_trials
    run_trials(model_cls=ICENODE_A, **capture_args())
