import json
from abc import ABCMeta
from collections.abc import Iterable
from copy import deepcopy
from typing import Dict, Any, ClassVar, Union

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from .utils import load_config


class Config(eqx.Module):
    """
    Config class for managing configuration data.

    Registers config classes, converts between config objects and dicts,
    updates configs, and provides utility functions for handling configuration data.

    Used as a base class for defining new config classes that can leverage these
    utilities.

    Attributes:

        _class_registry: Class registry mapping config class names to classes.

    Methods:
        as_dict() - Serialize config to a dict without type information.

        to_dict() - Serialize config to a dict with type information.

        from_dict() - Deserialize a dict back to a Config, using registry to pick subclass.

        register() - Register a Config subclass in the registry.

        copy() - Return a deep copy of the Config.

        path_update() - Update a nested config value specified by a path string.

        update() - Update config by merging with another Config or kwargs.

        get() - Get a config value by path, with default.
    """
    _class_registry: ClassVar[Dict[str, "Config"]] = {}

    @staticmethod
    def _is_config(x):
        return issubclass(x.__class__, Config)

    @staticmethod
    def _is_config_dict(x):
        return isinstance(x, dict) and "_type" in x

    @staticmethod
    def _addiction(x):
        if issubclass(x.__class__, Config):
            return {k: v for k, v in x.__dict__.items() if not k.startswith("_")}
        else:
            return x

    @staticmethod
    def _typed_addiction(x):
        if issubclass(x.__class__, Config):
            return {k: v for k, v in x.__dict__.items() if not k.startswith("_")} | {
                "_type": x.__class__.__name__
            }
        else:
            return x

    def as_dict(self) -> Dict[str, Any]:
        def _as_dict(x):
            dicted = self._addiction(x)
            if isinstance(dicted, Iterable) and not isinstance(dicted, str):
                return jtu.tree_map(_as_dict, dicted, is_leaf=self._is_config)
            else:
                return dicted

        return json.loads(json.dumps(_as_dict(self), default=vars))

    def to_dict(self):
        def _to_dict(x):
            dicted = self._typed_addiction(x)
            if isinstance(dicted, Iterable) and not isinstance(dicted, str):
                return jtu.tree_map(_to_dict, dicted, is_leaf=self._is_config)
            else:
                return dicted

        return json.loads(json.dumps(_to_dict(self), default=vars))

    def __len__(self):
        return len(self.as_dict())

    @classmethod
    def from_dict(cls, config: Dict[str, Any], config_class=None, **kwargs) -> "Config":
        def _concretise(x):
            if cls._is_config_dict(x):
                return cls.from_dict(x, **kwargs)
            elif isinstance(x, Iterable) and not isinstance(x, str):
                return jtu.tree_map(_concretise, x, is_leaf=cls._is_config_dict)
            else:
                return x

        config_class = cls._class_registry.get(config.pop("_type"), config_class)
        kwargs = {k: kwargs[k] for k in set(kwargs) & set(config)}
        return config_class(**({k: _concretise(v) for k, v in config.items()} | kwargs))

    @classmethod
    def register(cls):
        cls._class_registry[cls.__name__] = cls

    def copy(self) -> "Config":
        return deepcopy(self)

    def path_update(self, path, value):
        nesting = path.split(".")

        def _get(x):
            for n in nesting:
                x = getattr(x, n)
            return x

        _type = type(_get(self))

        return eqx.tree_at(_get, self, _type(value))

    def update(self, other=None, **kwargs):
        if other is not None:
            updated = self
            common_attrs = set(self.as_dict().keys()) & set(other.as_dict().keys())
            for attr in common_attrs:
                updated = eqx.tree_at(
                    lambda x: getattr(x, attr), updated, getattr(other, attr)
                )
            return updated

        updated = self.to_dict()
        updated.update({k: kwargs[k] for k in set(kwargs) & set(updated)})
        return Config.from_dict(updated, config_class=self.__class__)

    def get(self, path, default=None):
        x = self
        for n in path.split("."):
            x = getattr(x, n, default)
        return x


class Module(eqx.Module, metaclass=ABCMeta):
    """
    Base class for all modules with a config.

    Attributes:
        config: Config object.
        _class_registry: Class registry mapping module class names to classes.

    Methods:
        register() - Register a Module subclass in the registry.

        export_config() - Export config to a dict.

        external_argnames() - Return list of external arguments.

        from_config() - Create a Module from a config.

        copy() - Return a deep copy of the Module.

        path_update() - Update a nested config value specified by a path string.

        update() - Update config by merging with another Config or kwargs.

        get() - Get a config value by path, with default.
    """
    config: Config
    _class_registry: ClassVar[Dict[str, 'Module']] = {}

    def __init__(self,
                 config: Config = None,
                 config_path: str = None,
                 **kwargs):
        super().__init__()
        if config_path is not None:
            config = Config.from_dict(load_config(config_path))
        conf_kwargs = kwargs.copy()
        conf_kwargs = {
            k: conf_kwargs.pop(k)
            for k in set(config.as_dict()) & set(kwargs)
        }
        self.config = config.update(**conf_kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def register(cls):
        cls._class_registry[cls.__name__] = cls

    @classmethod
    def external_argnames(cls):
        return []

    def export_config(self):
        return self.config.to_dict()

    @classmethod
    def import_module(cls,
                      config: Union[Dict[str, Any], Config],
                      classname=None,
                      **kwargs):
        if issubclass(type(config), Config):
            config = config.copy()
            module_class = cls._class_registry[classname or cls.__name__]
            return module_class(config=config, **kwargs)
        config = deepcopy(config)

        external_kwargs = {
            k: kwargs.pop(k)
            for k in set(config.get('external_argnames', []))
                     & set(kwargs.keys())
        }
        module_class = cls._class_registry[config.get('classname', classname)]
        if isinstance(config['config'], dict):
            config = Config.from_dict(config['config']).update(**kwargs)
        else:
            config = config['config'].update(**kwargs)
        return module_class(config=config, **external_kwargs)

    def export_module(self):
        return self.export_module_class(self.config)

    @classmethod
    def export_module_class(cls, config: Config):
        conf = {'config': config.to_dict(), 'classname': cls.__name__}
        if len(cls.external_argnames()) > 0:
            conf['external_argnames'] = cls.external_argnames()
        return conf


class Data(eqx.Module):
    """
    Data class that inherits from eqx.Module.

    Methods:

        to_cpu() - Copy arrays in module to CPU.

        to_device() - Copy arrays in module to device.
    """
    def to_cpu(self):
        arrs, others = eqx.partition(self, eqx.is_array)
        arrs = jtu.tree_map(lambda a: np.array(a), arrs)
        return eqx.combine(arrs, others)

    def to_device(self):
        arrs, others = eqx.partition(self, eqx.is_array)
        arrs = jtu.tree_map(lambda a: jnp.array(a), arrs)
        return eqx.combine(arrs, others)


Config.register()
Module.register()
