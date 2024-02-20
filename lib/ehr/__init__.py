from .coding_scheme import (CodingScheme, OutcomeExtractor, CodingSchemeConfig, FlatScheme, HierarchicalScheme,
                            OutcomeExtractorConfig, resources_dir)
from .concepts import (Admission, Patient, InpatientObservables,
                       InpatientInterventions, AggregateRepresentation,
                       StaticInfo, InpatientInput, CodesVector,
                       DemographicVectorConfig, LeadingObservableExtractorConfig,
                       LeadingObservableExtractor,
                       CPRDDemographicVectorConfig)
from .dataset import Dataset, DatasetScheme, DatasetConfig, DatasetSchemeConfig
from .example_datasets.mimic3 import MIMIC3Dataset
from .example_schemes.cprd import setup_cprd
from .example_schemes.icd import setup_icd
from .interface import (AdmissionPrediction, Predictions, Patients,
                        InterfaceConfig, PatientTrajectory, TrajectoryConfig)

setup_icd()
setup_cprd()


def load_dataset():
    pass
