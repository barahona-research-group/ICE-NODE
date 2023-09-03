import inspect

from .base import AbstractConfig

from .ml import model
from .ml import dx_models
from .ml import in_models
from .ml import embeddings
from .ml import trainer

from .ehr import dataset
from .ehr import interface
from .ehr import concepts

modules = [
    model, dx_models, in_models, embeddings, trainer, dataset, interface,
    concepts
]
for m in modules:
    for name, conf_class in inspect.getmembers(m, inspect.isclass):
        if issubclass(conf_class, AbstractConfig):
            conf_class.register()
