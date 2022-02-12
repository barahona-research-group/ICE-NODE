from .jax_interface import (DiagnosisJAXInterface)
from .train_icenode_msl import ICENODE as ICENODE_MSL
from .train_icenode_m2lr import ICENODE as ICENODE_M2LR
from .gram import AbstractEmbeddingsLayer


class ICENODE(ICENODE_M2LR, ICENODE_MSL):

    def __init__(self, subject_interface: DiagnosisJAXInterface,
                 diag_emb: AbstractEmbeddingsLayer, ode_dyn: str,
                 ode_with_bias: bool, ode_init_var: float,
                 loss_half_life: float, memory_size: int, timescale: float):
        ICENODE_MSL.__init__(self,
                             subject_interface=subject_interface,
                             diag_emb=diag_emb,
                             ode_dyn=ode_dyn,
                             ode_with_bias=ode_with_bias,
                             ode_init_var=ode_init_var,
                             loss_half_life=loss_half_life,
                             memory_size=memory_size,
                             timescale=timescale)


if __name__ == '__main__':
    from .hpo_utils import capture_args, run_trials
    run_trials(model_cls=ICENODE, **capture_args())
