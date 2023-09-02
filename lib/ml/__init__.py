import sys
import inspect
from .model import (AbstractModel, ModelDimensions, model_classes,
                    model_dim_classes)
from .dx_models import (ICENODE, ICENODE_UNIFORM, ICENODE_ZERO, GRU, RETAIN,
                        ICENODEDimensions, GRUDimensions, RETAINDimensions)
from .in_models import (InICENODE, InICENODEDimensions)
from . import dx_models
from . import in_models

from .embeddings import (InpatientEmbedding, InpatientEmbeddingDimensions,
                         EmbeddedAdmission, OutpatientEmbedding,
                         OutpatientEmbeddingDimensions,
                         PatientEmbeddingDimensions, PatientEmbedding)
from .trainer import (Trainer, OptimizerConfig, TrainerReporting, InTrainer,
                      WarmupConfig)

for m in [dx_models, in_models]:
    model_classes.update({
        name: clas
        for name, clas in inspect.getmembers(m, inspect.isclass)
        if issubclass(clas, AbstractModel)
    })

    model_dim_classes.update({
        name: clas
        for name, clas in inspect.getmembers(m, inspect.isclass)
        if issubclass(clas, ModelDimensions)
    })

