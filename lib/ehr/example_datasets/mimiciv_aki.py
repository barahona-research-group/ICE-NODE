from typing import Literal

from lib.ehr.dataset import AbstractDatasetPipeline
from lib.ehr.example_datasets.mimiciv import MIMICIVDatasetSchemeConfig, \
    MIMICIVDataset, MIMICIVDatasetConfig
from lib.ehr.transformations import SetIndex, CastTimestamps, \
    SelectSubjectsWithObservation, ProcessOverlappingAdmissions, FilterSubjectsNegativeAdmissionLengths, \
    FilterClampTimestampsToAdmissionInterval, FilterUnsupportedCodes, ICUInputRateUnitConversion, \
    FilterInvalidInputRatesSubjects, SetAdmissionRelativeTimes, ValidatedDatasetPipeline


class AKIMIMICIVDatasetSchemeConfig(MIMICIVDatasetSchemeConfig):
    name_prefix: str = 'mimiciv.aki_study'
    resources_dir: str = 'mimiciv/aki_study'


class AKIMIMICIVDatasetConfig(MIMICIVDatasetConfig):
    scheme: AKIMIMICIVDatasetSchemeConfig = AKIMIMICIVDatasetSchemeConfig()
    overlapping_admissions: Literal["merge", "remove"] = "merge"
    filter_subjects_with_observation: str = 'renal_aki.aki_binary'


class AKIMIMICIVDataset(MIMICIVDataset):

    @classmethod
    def _setup_pipeline(cls, config: AKIMIMICIVDatasetConfig) -> AbstractDatasetPipeline:
        pconfig = config.pipeline
        pipeline = [
            SetIndex(),
            SelectSubjectsWithObservation(),
            CastTimestamps(),
            ProcessOverlappingAdmissions(),
            FilterSubjectsNegativeAdmissionLengths(),
            FilterClampTimestampsToAdmissionInterval(),
            FilterUnsupportedCodes(),
            ICUInputRateUnitConversion(),
            FilterInvalidInputRatesSubjects(),
            SetAdmissionRelativeTimes()
        ]
        return ValidatedDatasetPipeline(config=pconfig, transformations=pipeline)
