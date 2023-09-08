"""."""
from typing import Dict, List, Tuple
import inspect
import unittest
import equinox as eqx
import jax.tree_util as jtu
import os
from lib import Config
from lib.utils import write_config, load_config
from lib.ehr.coding_scheme import MIMICObservables
from lib.ehr import (load_dataset_config, load_dataset_scheme,
                     DemographicVectorConfig, CPRDDemographicVectorConfig,
                     InterfaceConfig, LeadingObservableConfig)
from lib.ml import ExperimentConfig
import lib.ehr as ehr
import lib.ml as ml

from lib.metric import stat


class DummyConfig(Config):
    x: int = 1
    y: int = 2


DummyConfig.register()


class NestedConfig(Config):
    a: Dict[str, DummyConfig]
    b: DummyConfig
    c: List[DummyConfig]


NestedConfig.register()


def gen_nested_configs():
    a = {'a1': DummyConfig(3, 3), 'a2': DummyConfig(4, 4)}
    b = DummyConfig(0, 1)
    c = [DummyConfig(4, 1), DummyConfig(5, 6)]
    return [b, *c, NestedConfig(a, b, c)]


def gen_dataset_configs():
    return [
        load_dataset_config('M3'),
        load_dataset_config('M4'),
        load_dataset_config('CPRD'),
        load_dataset_config('M4ICU'),
        load_dataset_config('M3CV')
    ]


def gen_ml_configs():
    ml_configs = []
    for name, conf_class in inspect.getmembers(ml, inspect.isclass):
        if issubclass(conf_class, Config) and conf_class != ExperimentConfig:
            ml_configs.append(conf_class())
    return ml_configs


def gen_lead_config():
    lead = ehr.concepts.LeadingObservableConfig(index=0,
                                                leading_hours=[24, 48, 72],
                                                scheme=MIMICObservables(),
                                                window_aggregate='max')
    return lead


def gen_interface_config():
    dataset_scheme = load_dataset_scheme('M3')
    target_conf = dataset_scheme.make_target_scheme_config(
        outcome='dx_icd9_filter_v3_groups')
    interface_conf = InterfaceConfig(
        demographic_vector=DemographicVectorConfig(),
        leading_observable=gen_lead_config(),
        dataset_scheme=dataset_scheme,
        scheme=target_conf)
    return interface_conf


def gen_demo_configs():
    demo1 = DemographicVectorConfig()
    demo2 = CPRDDemographicVectorConfig()
    return [demo1, demo2]


def gen_concept_configs():
    return [gen_lead_config()] + gen_demo_configs()


def gen_stat_configs():
    conf = []
    for name, conf_class in inspect.getmembers(stat, inspect.isclass):
        if issubclass(conf_class, Config):
            conf.append(conf_class())
    return conf


class ConfigTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """."""
        cls.all_configs = gen_dataset_configs() + gen_ml_configs() + \
            gen_concept_configs() + [gen_interface_config()] + \
            gen_stat_configs() + gen_nested_configs()

    def test_registration(self):
        """."""
        for conf in self.all_configs:
            name = type(conf).__name__
            self.assertIn(name, Config._class_registry)
            self.assertEqual(Config._class_registry[name], type(conf))

    def test_to_dict(self):
        """."""
        for conf in self.all_configs:
            conf_dict = conf.to_dict()
            self.assertIsInstance(conf_dict, dict)
            self.assertEqual(conf_dict['_type'], type(conf).__name__)
            # No nodes with AbstractConfig remains in dict.
            leaves = jtu.tree_leaves(
                eqx.filter(conf_dict, lambda x: issubclass(type(x), Config)))
            self.assertCountEqual(leaves, [])

    def test_from_dict(self):
        for conf in self.all_configs:
            conf_dict = conf.to_dict()
            conf_ = Config.from_dict(conf_dict)
            self.assertIsInstance(conf_, type(conf))
            self.assertEqual(conf_, conf)

    def test_serialisation(self):
        for conf in self.all_configs:
            write_config(conf.to_dict(), 'test.unit.base.json')
            conf_dict = load_config('test.unit.base.json')
            os.remove('test.unit.base.json')
            conf_ = Config.from_dict(conf_dict)
            self.assertEqual(conf_, conf)

    def test_path_update(self):
        c1 = DummyConfig(3, 6)
        c1_updated = c1.path_update('x', 4)
        self.assertEqual(c1_updated, DummyConfig(4, 6))

        c2 = NestedConfig(a={'n': DummyConfig(3, 3)},
                          b=DummyConfig(0, 10),
                          c=[DummyConfig(4, 1),
                             DummyConfig(5, 6)])

        c3 = NestedConfig(a={'n': DummyConfig(3, 3)},
                          b=DummyConfig(0, 20),
                          c=[DummyConfig(4, 1),
                             DummyConfig(5, 6)])

        c2_updated = c2.path_update('b.y', 20)
        self.assertEqual(c2_updated, c3)
