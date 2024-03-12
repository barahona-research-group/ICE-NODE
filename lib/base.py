import json
from abc import ABCMeta
from typing import Dict, Any, ClassVar, Union, Type, Callable

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)


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
    _class_registry: ClassVar[Dict[str, Type["Config"]]] = {}

    @staticmethod
    def _map_hierarchical_config(unit_config_map: Callable[["Config"], Dict[str, Any]], x: Any) -> Any:
        if isinstance(x, Config):
            x = unit_config_map(x)
        if isinstance(x, dict):
            return {k: Config._map_hierarchical_config(unit_config_map, v) for k, v in x.items()}
        elif isinstance(x, list):
            return [Config._map_hierarchical_config(unit_config_map, v) for v in x]
        elif isinstance(x, tuple):
            return tuple(Config._map_hierarchical_config(unit_config_map, v) for v in x)
        else:
            return x

    def equals(self, other: 'Config') -> bool:
        return other.as_dict() == self.as_dict()

    def __eq__(self, other: 'Config') -> bool:
        return self.equals(other)

    @staticmethod
    def as_normal_dict(x: 'Config') -> Dict[str, Any]:
        return {k: v for k, v in x.__dict__.items() if not k.startswith("_")}

    @staticmethod
    def as_typed_dict(x: 'Config') -> Dict[str, Any]:
        return Config.as_normal_dict(x) | {"_type": x.__class__.__name__}

    @staticmethod
    def _is_typed_dict(x) -> bool:
        return isinstance(x, dict) and "_type" in x

    @staticmethod
    def map_config_to_dict(unit_map: Callable[[Union['Config', Any]], Dict[str, Any]], x) -> Dict[str, Any]:
        return json.loads(json.dumps(Config._map_hierarchical_config(unit_map, x), cls=NumpyEncoder))

    def as_dict(self) -> Dict[str, Any]:
        return self.map_config_to_dict(Config.as_normal_dict, self)

    def to_dict(self) -> Dict[str, Any]:
        return self.map_config_to_dict(Config.as_typed_dict, self)

    def update(self, other: Union['Config', Dict[str, Any]]) -> 'Config':
        if isinstance(other, Config):
            other = other.as_dict()
        return Config.from_dict(self.to_dict() | other)

    def __len__(self) -> int:
        return len(self.as_dict())

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "Config":
        def _map_dict_to_config(x):
            if cls._is_typed_dict(x):
                config_class = cls._class_registry[x.pop("_type")]
                config_kwargs = {k: _map_dict_to_config(v) for k, v in x.items()}
                return config_class(**config_kwargs)
            elif isinstance(x, dict):
                return {k: _map_dict_to_config(v) for k, v in x.items()}
            elif isinstance(x, list):
                return [_map_dict_to_config(v) for v in x]
            elif isinstance(x, tuple):
                return tuple(_map_dict_to_config(v) for v in x)
            else:
                return x

        return _map_dict_to_config(config)

    @classmethod
    def register(cls):
        cls._class_registry[cls.__name__] = cls

    def path_update(self, path, value):
        nesting = path.split(".")

        def _get(x):
            for n in nesting:
                x = getattr(x, n)
            return x

        _type = type(_get(self))

        return eqx.tree_at(_get, self, _type(value))


class FlatConfig(Config):

    def __post_init__(self):
        assert all(not isinstance(v, Config) for v in self.__dict__.values()), \
            "FlatConfig cannot contain nested Configs."


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
    _class_registry: ClassVar[Dict[str, Type['Module']]] = {}

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
                      **external_kwargs):
        if issubclass(type(config), Config):
            module_class = cls._class_registry[classname or cls.__name__]
            return module_class(config=config, **external_kwargs)

        if len(external_kwargs) > 0:
            assert set(config['external_argnames']) == set(external_kwargs.keys()), \
                "External kwargs do not match external argnames."

        module_class = cls._class_registry[config.get('classname', classname)]
        if isinstance(config['config'], dict):
            config = Config.from_dict(config['config'])
        else:
            config = config['config']

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
