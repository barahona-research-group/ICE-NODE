from typing import Optional, Callable, Any, Dict, Tuple
from functools import partial

import jax
import jax.numpy as jnp

from .train_icenode import ICENODE
from .jax_interface import DiagnosisJAXInterface
from .gram import AbstractEmbeddingsLayer


class ICENODE_X(ICENODE):

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

    def _f_dec_seq(
        self, params: Any, state_seq: Dict[int, jnp.ndarray]
    ) -> Tuple[Dict[int, jnp.ndarray], Dict[int, jnp.ndarray]]:

        dec_seq = {
            i: jax.vmap(partial(self.f_dec, params['f_dec']))(state)
            for i, state in state_seq.items()
        }

        emb = {i: jnp.mean(e_seq, axis=0) for i, (e_seq, _) in dec_seq.items()}

        diag = {i: jnp.max(d_seq, axis=0) for i, (_, d_seq) in dec_seq.items()}

        return emb, diag


if __name__ == '__main__':
    from .hpo_utils import capture_args, run_trials
    run_trials(model_cls=ICENODE_X, **capture_args())
