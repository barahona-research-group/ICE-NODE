import inspect

from .model import (AbstractModel, ModelDimensions)
from .dx_models import (ICENODE, ICENODE_UNIFORM, ICENODE_ZERO, GRU, RETAIN,
                        ICENODEDimensions, GRUDimensions, RETAINDimensions)
from .in_models import (InICENODE, InICENODEDimensions)

from .embeddings import (InpatientEmbedding, InpatientEmbeddingDimensions,
                         EmbeddedAdmission, OutpatientEmbedding,
                         OutpatientEmbeddingDimensions,
                         PatientEmbeddingDimensions, PatientEmbedding)
from .trainer import (Trainer, OptimizerConfig, TrainerReporting, InTrainer,
                      WarmupConfig)

from . import model
from . import dx_models
from . import in_models

for m in [model, dx_models, in_models]:
    for name, model_class in inspect.getmembers(m, inspect.isclass):
        if issubclass(model_class, AbstractModel):
            model_class.register()
