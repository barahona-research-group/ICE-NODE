import inspect

from .base import Config, Module
from .ehr import concepts
from .ehr import dataset
from .ehr import interface
from .ehr.example_datasets import mimic3, cprd, mimiciv, mimiciv_aki
from .metric import stat
from .ml import dx_models
from .ml import embeddings
from .ml import evaluation
from .ml import experiment
from .ml import in_models
from .ml import in_models_modular
from .ml import model
from .ml import trainer

modules = [
    model, dx_models, in_models, in_models_modular, embeddings, trainer,
    experiment, evaluation, interface, stat, concepts, dataset, mimic3, cprd, mimiciv, mimiciv_aki
]
for m in modules:
    for name, _class in inspect.getmembers(m, inspect.isclass):
        if issubclass(_class, Config):
            _class.register()

        if issubclass(_class, Module):
            _class.register()
