from typing import Dict, Any
import equinox as eqx
from copy import deepcopy


class AbstractConfig(eqx.Module):

    def to_dict(self) -> Dict[str, Any]:
        return {
            k: v.to_config() if issubclass(v.__class__, AbstractConfig) else v
            for k, v in self.__dict__.items() if not k.startswith('_')
        } | {
            '_type': self.__class__.__name__
        }

    def __len__(self):
        return len(tuple(k for k in self.__dict__ if not k.startswith('_')))

    @staticmethod
    def from_dict(config: Dict[str, Any]) -> 'AbstractConfig':
        clas = configuration_class_registry[config.pop('_type')]
        return clas(
            **{
                k:
                AbstractConfig.
                from_dict(v) if isinstance(v, dict) and '_type' in v else v
                for k, v in config.items()
            })

    @classmethod
    def register(cls):
        configuration_class_registry[cls.__name__] = cls

    def copy(self) -> 'AbstractConfig':
        return deepcopy(self)

    def update(self, **kwargs):
        updated = self.to_dict()
        updated.update(kwargs)
        return AbstractConfig.from_dict(**updated)


configuration_class_registry = {}
AbstractConfig.register()
