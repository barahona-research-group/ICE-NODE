from typing import Dict, Any
import equinox as eqx
from copy import deepcopy


class AbstractConfig(eqx.Module):

    def as_dict(self) -> Dict[str, Any]:
        return {
            k: v.as_dict() if issubclass(v.__class__, AbstractConfig) else v
            for k, v in self.__dict__.items() if not k.startswith('_')
        }

    def to_dict(self):
        return {
            k: v.to_dict() if issubclass(v.__class__, AbstractConfig) else v
            for k, v in self.__dict__.items() if not k.startswith('_')
        } | {
            '_type': self.__class__.__name__
        }

    def __len__(self):
        return len(self.as_dict())

    @staticmethod
    def from_dict(config: Dict[str, Any], **kwargs) -> 'AbstractConfig':
        clas = configuration_class_registry[config.pop('_type')]
        return clas(**({
            k:
            AbstractConfig.
            from_dict(v) if isinstance(v, dict) and '_type' in v else v
            for k, v in config.items()
        } | kwargs))

    @classmethod
    def register(cls):
        configuration_class_registry[cls.__name__] = cls

    def copy(self) -> 'AbstractConfig':
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
        return AbstractConfig.from_dict(updated)


configuration_class_registry = {}
AbstractConfig.register()
