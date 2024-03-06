from lib.ehr.dataset import AbstractDatasetPipeline
from lib.ehr.example_datasets.mimiciv import MIMICIVDatasetSchemeConfig, \
    MIMICIVDataset, MIMICIVDatasetConfig
from lib.ehr.transformations import SetIndex, CastTimestamps, SetCodeIntegerIndices, \
    SelectSubjectsWithObservation, ProcessOverlappingAdmissions, FilterSubjectsNegativeAdmissionLengths, \
    FilterClampTimestampsToAdmissionInterval, FilterUnsupportedCodes, ICUInputRateUnitConversion, \
    FilterInvalidInputRatesSubjects, SetAdmissionRelativeTimes, DatasetPipeline


class AKIMIMICIVDatasetSchemeConfig(MIMICIVDatasetSchemeConfig):
    name_prefix: str = 'mimiciv.aki_study'
    resources_dir: str = 'mimiciv/aki_study'


class AKIMIMICIVDatasetConfig(MIMICIVDatasetConfig):
    scheme: AKIMIMICIVDatasetSchemeConfig = AKIMIMICIVDatasetSchemeConfig()


class AKIMIMICIVDataset(MIMICIVDataset):

    @classmethod
    def _setup_core_pipeline(cls, config: AKIMIMICIVDatasetConfig) -> AbstractDatasetPipeline:
        pconfig = config.pipeline
        conversion_table = cls.icu_inputs_uom_normalization(config.tables.icu_inputs,
                                                            config.scheme.icu_inputs_uom_normalization_table)
        pipeline = [
            SetIndex(),
            SelectSubjectsWithObservation(name='select_with_aki_info',
                                          code='renal_aki.aki_binary'),
            CastTimestamps(),
            ProcessOverlappingAdmissions(merge=pconfig.overlap_merge),
            FilterSubjectsNegativeAdmissionLengths(),
            FilterClampTimestampsToAdmissionInterval(),
            FilterUnsupportedCodes(),
            ICUInputRateUnitConversion(conversion_table=conversion_table),
            FilterInvalidInputRatesSubjects(),
            SetCodeIntegerIndices(),
            SetAdmissionRelativeTimes()
        ]
        return DatasetPipeline(config=pconfig, transformations=pipeline)
