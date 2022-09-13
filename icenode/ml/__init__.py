# All modules below imported to execute the registration.
from .abstract import AbstractModel
from .dx_gru import GRU
from .dx_icenode_tl import ICENODE as ICENODE
from .dx_icenode_2lr import ICENODE as ICENODE_2LR

from .dx_icenode_uniform import ICENODE as ICENODE_UNIFORM
from .dx_icenode_uniform2lr import ICENODE as ICENODE_UNIFORM_2LR
from .dx_retain import RETAIN
from .dx_window_logreg import (WindowLogReg as WLR, WindowLogReg_Sklearn as
                               WLR_SK, logreg_loss_multinomial_mode)

from .expt.dxpr_icenode import ICENODE as PR_ICENODE

from .trainer import (AbstractReporter, MinibatchLogger, EvaluationDiskWriter,
                      ParamsDiskWriter, ConfigDiskWriter)

GRU.register_model('dx_gru')
ICENODE.register_model('dx_icenode')
ICENODE_2LR.register_model('dx_icenode_2lr')
ICENODE_UNIFORM.register_model('dx_icenode_uniform')
ICENODE_UNIFORM_2LR.register_model('dx_icenode_uniform2lr')
RETAIN.register_model('dx_retain')
WLR.register_model('dx_window_logreg')
WLR_SK.register_model('dx_window_logreg_sklearn')

PR_ICENODE.register_model('dxpr_icenode')

model_cls = AbstractModel.model_cls
