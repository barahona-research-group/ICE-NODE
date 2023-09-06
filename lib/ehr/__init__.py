import os
from pathlib import Path
from .dataset import Dataset, DatasetScheme
from .ds_mimic3 import MIMIC3Dataset
from .ds_mimic4 import MIMIC4Dataset, MIMIC4ICUDataset
from .ds_cprd import CPRDDataset

from .coding_scheme import (AbstractScheme, OutcomeExtractor, Gender,
                            Ethnicity)
from .concepts import (Admission, Patient, InpatientObservables,
                       InpatientInterventions, AggregateRepresentation,
                       StaticInfo, InpatientInput, CodesVector,
                       DemographicVectorConfig, LeadingObservableConfig,
                       CPRDDemographicVectorConfig)
from .interface import (AdmissionPrediction, Predictions, Patients,
                        InterfaceConfig)
from ..utils import load_config
from ..base import Config, Module

_DIR = os.path.dirname(__file__)
_PROJECT_DIR = Path(_DIR).parent.parent.absolute()
_META_DIR = os.path.join(_PROJECT_DIR, 'datasets_meta')

_default_config_files = {
    'M3': f'{_META_DIR}/mimic3_meta.json',
    'M3CV': f'{_META_DIR}/mimic3cv_meta.json',
    'M4': f'{_META_DIR}/mimic4_meta.json',
    'CPRD': f'{_META_DIR}/cprd_meta.json',
    'M4ICU': f'{_META_DIR}/mimic4icu_meta.json',
}


def load_dataset_scheme(tag) -> DatasetScheme:
    conf = load_config(_default_config_files[tag])
    config = Config.from_dict(conf['config']['scheme'])
    scheme_class = Module._class_registry[conf['config']['scheme_classname']]
    return scheme_class(config)


def load_dataset_config(tag: str = None,
                        config: 'Config' = None,
                        **init_kwargs):
    if config is not None:
        tag = config.tag
    else:
        config = load_config(_default_config_files[tag])["config"]
        config = Config.from_dict(config)
        config = config.update(**init_kwargs)
    return config


def load_dataset(tag: str = None, config: 'Config' = None, **init_kwargs):
    config = load_dataset_config(tag=tag, config=config, **init_kwargs)
    if tag is None:
        tag = config.tag
    if tag in ('M3', 'M3CV'):
        return MIMIC3Dataset(config)
    if tag == 'M4':
        return MIMIC4Dataset(config)
    if tag == 'CPRD':
        return CPRDDataset(config)
    if tag == 'M4ICU':
        return MIMIC4ICUDataset(config)
