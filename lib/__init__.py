import inspect

from .base import Config, Module

from .ml import model
from .ml import dx_models
from .ml import in_models
from .ml import in_models_modular
from .ml import embeddings
from .ml import trainer
from .ml import experiment
from .ml import evaluation

from .ehr import _dataset_cprd, _dataset_mimic3, _dataset_mimic4, dataset
from .ehr import interface
from .ehr import concepts

from .metric import stat

modules = [
    model, dx_models, in_models, in_models_modular, embeddings, trainer,
    experiment, evaluation, interface, stat, concepts, dataset, _dataset_mimic3,
    _dataset_mimic4, _dataset_cprd
]
for m in modules:
    for name, _class in inspect.getmembers(m, inspect.isclass):
        if issubclass(_class, Config):
            _class.register()

        if issubclass(_class, Module):
            _class.register()
