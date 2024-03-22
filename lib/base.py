import dataclasses
import json
from abc import ABCMeta
from typing import Dict, Any, ClassVar, Union, Type, Callable, Tuple

# TODO: update to Python 3.11, then use typing.Self
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pandas as pd
import tables as tb


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
        return {field.name: getattr(x, field.name) for field in dataclasses.fields(x) if not field.name.startswith("_")}

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
                config_class = cls.config_class(x.pop("_type"))
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

    @staticmethod
    def config_class(label: str):
        return Config._class_registry[label]

    @classmethod
    def register(cls):
        Config._class_registry[cls.__name__] = cls

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
        Module._class_registry[cls.__name__] = cls

    @classmethod
    def external_argnames(cls):
        return []

    def export_config(self):
        return self.config.to_dict()

    @staticmethod
    def module_class(label: str):
        return Module._class_registry[label]

    @classmethod
    def import_module(cls,
                      config: Union[Dict[str, Any], Config],
                      classname=None,
                      **external_kwargs):
        if issubclass(type(config), Config):
            module_class = cls.module_class(classname or cls.__name__)
            return module_class(config=config, **external_kwargs)

        if len(external_kwargs) > 0:
            assert set(config['external_argnames']) == set(external_kwargs.keys()), \
                "External kwargs do not match external argnames."

        module_class = cls.module_class(config.get('classname', classname))
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


Array = Union[np.ndarray, jnp.ndarray, jax.Array]


def equal_arrays(a: Array, b: Array):
    _np = np if isinstance(a, np.ndarray) else jnp
    is_nan = _np.isnan(a) & _np.isnan(b)
    return _np.array_equal(a[~is_nan], b[~is_nan], equal_nan=False)


class VxData(eqx.Module):
    """
    VxData class represents vectorized data object, which inherits from eqx.Module.

    Methods:

        to_cpu() - Copy arrays in module to CPU.

        to_device() - Copy arrays in module to device.
    """
    _class_registry: ClassVar[Dict[str, Type["VxData"]]] = {}

    def __post_init__(self):
        unsupported_field_types = set()
        for f in self.fields:
            obj = type(getattr(self, f))
            if isinstance(getattr(self, f), VxDataIterableField):
                if not all(isinstance(i, VxDataItem) for i in getattr(self, f)):
                    unsupported_field_types.add((f'{f}[i]', obj))
            elif not isinstance(getattr(self, f), VxDataField):
                unsupported_field_types.add((f, obj))
        unsupported_items_str = ', '.join(map(lambda p: f"{p[0]} ({p[1]})", unsupported_field_types))
        assert len(unsupported_field_types) == 0, \
            f"VxData object contains unsupported type(s): {unsupported_items_str}."

    @classmethod
    def register(cls):
        VxData._class_registry[cls.__name__] = cls

    @staticmethod
    def data_class(label: str):
        return VxData._class_registry[label]

    def to_cpu(self):
        arrs, others = eqx.partition(self, eqx.is_array)
        arrs = jtu.tree_map(lambda a: np.array(a), arrs)
        return eqx.combine(arrs, others)

    def to_device(self):
        arrs, others = eqx.partition(self, eqx.is_array)
        arrs = jtu.tree_map(lambda a: jnp.array(a), arrs)
        return eqx.combine(arrs, others)

    def __len__(self) -> int:
        raise NotImplementedError

    @property
    def fields(self) -> Tuple[str, ...]:
        return tuple(k.name for k in dataclasses.fields(self) if getattr(self, k.name) is not None)

    @property
    def str_attributes(self) -> Tuple[str, ...]:
        return tuple(k for k in self.fields if isinstance(getattr(self, k), str))

    @property
    def array_attributes(self) -> Tuple[str, ...]:
        return tuple(k for k in self.fields if isinstance(getattr(self, k), Array))

    @property
    def data_attributes(self) -> Tuple[str, ...]:
        return tuple(k for k in self.fields if isinstance(getattr(self, k), VxData))

    @property
    def timestamp_attributes(self) -> Tuple[str, ...]:
        return tuple(k for k in self.fields if isinstance(getattr(self, k), pd.Timestamp))

    @property
    def iterable_attributes(self) -> Tuple[str, ...]:
        return tuple(k for k in self.fields if isinstance(getattr(self, k), (list, tuple)))

    @staticmethod
    def _store_array_to_hdf(group: tb.Group, name: str, array: Array) -> None:
        if len(array) == 0:
            group._v_file.create_array(group, name, obj=array)
        else:
            group._v_file.create_carray(group, name, obj=array)

    @staticmethod
    def _store_timestamp_to_hdf(group: tb.Group, name: str, date: pd.Timestamp) -> None:
        group._v_file.create_array(group, name, obj=date.value)

    @staticmethod
    def _load_timestamp_from_hdf(group: tb.Group, name: str) -> pd.Timestamp:
        return pd.Timestamp(group[name].read())

    def to_hdf_group(self, group: tb.Group) -> None:
        h5file = group._v_file

        h5file.create_array(group, 'classname', obj=self.__class__.__name__.encode('utf-8'))

        for attr in self.str_attributes:
            h5file.create_array(group, attr, obj=getattr(self, attr).encode('utf-8'))
        for attr in self.timestamp_attributes:
            self._store_timestamp_to_hdf(group, f'_x_timestamp_{attr}', getattr(self, attr))
        for attr in self.array_attributes:
            self._store_array_to_hdf(group, attr, getattr(self, attr))
        for attr in self.data_attributes:
            attr_group = h5file.create_group(group, attr)
            getattr(self, attr).to_hdf_group(attr_group)

        for attr in self.iterable_attributes:
            iterable = getattr(self, attr)
            if isinstance(iterable, list):
                group_name = f'_x_list_{attr}'
            elif isinstance(iterable, tuple):
                group_name = f'_x_tuple_{attr}'
            else:
                raise TypeError(f"Unsupported type {type(iterable)} for attribute {attr}")

            attr_group = h5file.create_group(group, group_name)
            for i, item in enumerate(iterable):
                if isinstance(item, Array):
                    self._store_array_to_hdf(attr_group, str(i), item)
                elif isinstance(item, VxData):
                    item.to_hdf_group(h5file.create_group(attr_group, str(i)))
                else:
                    raise TypeError(f"Unsupported type {type(item)} for attribute {attr}")

    @staticmethod
    def deserialize_iterable(group: tb.Group, iterable_class: Type['VxDataIterableField']) -> 'VxDataIterableField':
        sequence = [str(i) for i in range(group._v_nchildren)]
        leaves = {k: group[k].read() for k in group._v_leaves}
        leaves = {k: v.decode('utf-8') if isinstance(v, bytes) else v for k, v in leaves.items()}
        groups = {k: group[k] for k in group._v_groups}
        items = []
        for i in sequence:
            if i in leaves:
                items.append(leaves[i])
            elif i in groups:
                items.append(VxData.from_hdf_group(groups[i]))
        return iterable_class(items)

    @staticmethod
    def from_hdf_group(group: tb.Group) -> 'VxData':
        classname = group['classname'].read().decode('utf-8')
        cls = VxData.data_class(classname)
        data = {k: group[k].read() for k in group._v_leaves if
                not k.startswith('_x_timestamp_') and not k == 'classname'}
        data = {k: v.decode('utf-8') if isinstance(v, bytes) else v for k, v in data.items()}
        data |= {k.split('_x_timestamp_')[1]: VxData._load_timestamp_from_hdf(group, k) for k in group._v_leaves if
                 k.startswith('_x_timestamp_')}

        groups = {k: group[k] for k in group._v_groups}
        if len(groups) > 0:
            list_groups = {k.split('_x_list_')[1]: g for k, g in groups.items() if k.startswith('_x_list_')}
            tuple_groups = {k.split('_x_tuple_')[1]: g for k, g in groups.items() if k.startswith('_x_tuple_')}
            data_attrs = set(k for k in groups.keys() if not k.startswith(('_x_list_', '_x_tuple_')))
            data |= {k: VxData.from_hdf_group(g) for k, g in groups.items() if k in data_attrs}
            data |= {k: VxData.deserialize_iterable(g, list) for k, g in list_groups.items()}
            data |= {k: VxData.deserialize_iterable(g, tuple) for k, g in tuple_groups.items()}
        return cls(**data)

    @staticmethod
    def equal_attributes(a: 'VxData', b: 'VxData', attributes: Tuple[str, ...]) -> bool:
        # Light checks first.
        if len(a) != len(b):
            return False

        # Some more light checks first.
        for k in attributes:
            a_k = getattr(a, k)
            b_k = getattr(b, k)
            if type(a_k) is not type(b_k):
                return False
            if isinstance(a_k, Array) and (a_k.shape != b_k.shape or a_k.dtype != b_k.dtype):
                return False

        # Now the heavy checks.
        for k in attributes:
            a_k = getattr(a, k)
            b_k = getattr(b, k)

            if a_k is not None:
                if hasattr(a_k, 'equals'):
                    if not a_k.equals(b_k):
                        return False
                elif isinstance(a_k, (list, tuple)):
                    if len(a_k) != len(b_k):
                        return False
                    if any(not x.equals(y) for x, y in zip(a_k, b_k)):
                        return False
                elif isinstance(a_k, Array):
                    if not equal_arrays(a_k, b_k):
                        return False
                elif isinstance(a_k, (pd.Timestamp, str)):
                    if a_k != b_k:
                        return False
                else:
                    raise TypeError(f"Unsupported type {type(a_k)} for attribute {k}")
        return True

    @property
    def comparable_attribute_names(self) -> Tuple[str, ...]:
        """
        Returns the names of all attributes that can be compared for equality. Sorted by type considering
        the lighter comparisons first.
        """
        return self.timestamp_attributes + self.str_attributes + \
            self.array_attributes + self.data_attributes + self.iterable_attributes

    def equals(self, other: 'VxData') -> bool:
        """
        Compares two Data objects for equality.

        Args:
            other (VxData): the other Data object to compare.

        Returns:
            bool: whether the two Data objects are equal.
        """
        attribute_names = self.comparable_attribute_names

        return type(self) == type(
            other) and attribute_names == other.comparable_attribute_names and self.equal_attributes(self, other,
                                                                                                     attribute_names)


VxDataItem = Union[VxData, Array]
VxDataIterableField = Union[list, tuple]
VxDataField = Union[VxDataItem, pd.Timestamp, str, VxDataIterableField]

Config.register()
Module.register()
