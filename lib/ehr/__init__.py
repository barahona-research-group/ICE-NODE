import os
from typing import Optional

from lib.ehr.example_datasets.mimic3 import MIMIC3Dataset
from lib.ehr.example_schemes.cprd import setup_cprd
from lib.ehr.example_schemes.icd import setup_icd
from .coding_scheme import (CodingScheme, OutcomeExtractor, CodingSchemeConfig, FlatScheme, HierarchicalScheme,
                            OutcomeExtractorConfig)
from .concepts import (Admission, Patient, InpatientObservables,
                       InpatientInterventions, AggregateRepresentation,
                       StaticInfo, InpatientInput, CodesVector,
                       DemographicVectorConfig, LeadingObservableExtractorConfig,
                       LeadingObservableExtractor,
                       CPRDDemographicVectorConfig)
from .dataset import Dataset, DatasetScheme, DatasetConfig, DatasetSchemeConfig
from .interface import (AdmissionPrediction, Predictions, Patients,
                        InterfaceConfig, PatientTrajectory, TrajectoryConfig)


def resources_dir(subdir: Optional[str] = None) -> str:
    if subdir is not None:
        return os.path.join(os.path.dirname(__file__), "resources", subdir)
    return os.path.join(os.path.dirname(__file__), "resources")


setup_icd()
setup_cprd()
