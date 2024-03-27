import inspect

from .base import Config, Module, VxData
from .ehr import tvx_concepts, dataset, coding_scheme, transformations, tvx_transformations, tvx_ehr
from .ehr.example_datasets import mimic3, cprd, mimiciv, mimiciv_aki
from .ehr.example_schemes import icd
from .metric import stat
from .ml import dx_models
from .ml import embeddings
from .ml import evaluation
from .ml import experiment
from .ml import in_models
from .ml import model
from .ml import trainer

modules = [
    model, dx_models, in_models, embeddings, trainer,
    experiment, evaluation, stat,
    # schemes
    icd,
    # ehr
    dataset, coding_scheme, transformations, mimic3, cprd, mimiciv, mimiciv_aki,
    # tvx_ehr
    tvx_ehr, tvx_concepts, tvx_concepts, tvx_transformations
]
for m in modules:
    for name, _class in inspect.getmembers(m, inspect.isclass):
        if issubclass(_class, (Config, Module, VxData)):
            _class.register()
