from typing import Dict, Any, ClassVar, Union
from abc import ABCMeta
from copy import deepcopy
import equinox as eqx

from .utils import load_config


class Config(eqx.Module):
    _class_registry: ClassVar[Dict[str, 'Config']] = {}

    def as_dict(self) -> Dict[str, Any]:
        return {
            k: v.as_dict() if issubclass(v.__class__, Config) else v
            for k, v in self.__dict__.items() if not k.startswith('_')
        }

    def to_dict(self):
        return {
            k: v.to_dict() if issubclass(v.__class__, Config) else v
            for k, v in self.__dict__.items() if not k.startswith('_')
        } | {
            '_type': self.__class__.__name__
        }

    def __len__(self):
        return len(self.as_dict())

    @classmethod
    def from_dict(cls,
                  config: Dict[str, Any],
                  config_class=None,
                  **kwargs) -> 'Config':
        config_class = cls._class_registry.get(config.pop('_type'),
                                               config_class)
        return config_class(**({
            k:
            cls.from_dict(v) if isinstance(v, dict) and '_type' in v else v
            for k, v in config.items()
        } | kwargs))

    @classmethod
    def register(cls):
        cls._class_registry[cls.__name__] = cls

    def copy(self) -> 'Config':
        return deepcopy(self)

    def update(self, other=None, **kwargs):
        if other is not None:
            updated = self
            common_attrs = set(self.as_dict().keys()) & set(
                other.as_dict().keys())
            for attr in common_attrs:
                updated = eqx.tree_at(lambda x: getattr(x, attr), updated,
                                      getattr(other, attr))
            return updated

        updated = self.to_dict()
        updated.update(kwargs)
        return Config.from_dict(updated, config_class=self.__class__)


class Module(eqx.Module, metaclass=ABCMeta):
    config: Config
    _class_registry: ClassVar[Dict[str, 'Module']] = {}

    def __init__(self,
                 config: Config = None,
                 config_path: str = None,
                 **kwargs):
        super().__init__()
        if config_path is not None:
            config = Config.from_dict(load_config(config_path))
        self.config = config.update(**kwargs)

    @classmethod
    def register(cls):
        cls._class_registry[cls.__name__] = cls

    @classmethod
    def external_argnames(cls):
        return []

    @classmethod
    def from_config(cls,
                    config: Union[Dict[str, Any], Config],
                    classname=None,
                    **kwargs):
        if issubclass(type(config), Config):
            module_class = cls._class_registry[classname or cls.__name__]
            return module_class(config=config, **kwargs)

        external_kwargs = {
            k: kwargs.pop(k)
            for k in set(config.get('external_argnames', []))
            & set(kwargs.keys())
        }
        module_class = cls._class_registry[config.get('classname', classname)]
        config = Config.from_dict(config['config']).update(**kwargs)
        return module_class(config=config, **external_kwargs)

    def export_config(self):
        conf = {
            'config': self.config.to_dict(),
            'classname': self.__class__.__name__
        }
        if len(self.external_argnames()) > 0:
            conf['external_argnames'] = self.external_argnames()

        for attr, attr_v in self.__dict__.items():
            if issubclass(type(attr_v), Module):
                conf[attr] = attr_v.export_config()

        return conf


class Data(eqx.Module):
    pass


Config.register()
Module.register()
