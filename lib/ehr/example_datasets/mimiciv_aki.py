from dataclasses import field
from typing import Literal, Final, Optional, Callable

from lib import Config
from lib.ehr import TVxEHR, TVxEHRConfig, DemographicVectorConfig, LeadingObservableExtractorConfig
from lib.ehr.dataset import AbstractDatasetPipeline
from lib.ehr.example_datasets.mimiciv import MIMICIVDatasetSchemeConfig, \
    MIMICIVDataset, MIMICIVDatasetConfig
from lib.ehr.transformations import SetIndex, CastTimestamps, \
    SelectSubjectsWithObservation, ProcessOverlappingAdmissions, FilterSubjectsNegativeAdmissionLengths, \
    FilterClampTimestampsToAdmissionInterval, FilterUnsupportedCodes, ICUInputRateUnitConversion, \
    FilterInvalidInputRatesSubjects, SetAdmissionRelativeTimes, ValidatedDatasetPipeline
from lib.ehr.tvx_ehr import TVxEHRSchemeConfig, TVxEHRSampleConfig, TVxEHRSplitsConfig, \
    DatasetNumericalProcessorsConfig, AbstractTVxPipeline
from lib.ehr.tvx_transformations import SampleSubjects, ObsIQROutlierRemover, RandomSplits, ObsAdaptiveScaler, \
    InputScaler, ObsTimeBinning, TVxConcepts, InterventionSegmentation, ExcludeShortAdmissions, \
    LeadingObservableExtraction

OBSERVABLE_AKI_TARGET_CODE: Final[str] = 'renal_aki.aki_binary'


class AKIMIMICIVDatasetSchemeConfig(MIMICIVDatasetSchemeConfig):
    name_prefix: str = 'mimiciv.aki_study'
    resources_dir: str = 'mimiciv/aki_study'


DEFAULT_AKI_MIMICIV_DATASET_SCHEME_CONFIG: Final[AKIMIMICIVDatasetSchemeConfig] = AKIMIMICIVDatasetSchemeConfig()


class AKIMIMICIVDatasetConfig(MIMICIVDatasetConfig):
    scheme: AKIMIMICIVDatasetSchemeConfig = field(default_factory=lambda: DEFAULT_AKI_MIMICIV_DATASET_SCHEME_CONFIG)
    overlapping_admissions: Literal["merge", "remove"] = "merge"
    filter_subjects_with_observation: str = field(default_factory=lambda: OBSERVABLE_AKI_TARGET_CODE)


DEFAULT_AKI_MIMICIV_DATASET_CONFIG: Final[AKIMIMICIVDatasetConfig] = AKIMIMICIVDatasetConfig()


class AKIMIMICIVDataset(MIMICIVDataset):

    @classmethod
    def _setup_pipeline(cls, config: AKIMIMICIVDatasetConfig) -> AbstractDatasetPipeline:
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
        return ValidatedDatasetPipeline(transformations=pipeline)


DEFAULT_AKI_DEMOGRAPHIC: Final[DemographicVectorConfig] = DemographicVectorConfig(age=True,
                                                                                  gender=True,
                                                                                  ethnicity=True)
DEFAULT_AKI_LEAD_EXTRACTION_FACTORY: Final[
    Callable[[], LeadingObservableExtractorConfig]] = lambda: LeadingObservableExtractorConfig(
    observable_code=OBSERVABLE_AKI_TARGET_CODE,
    scheme=DEFAULT_AKI_MIMICIV_DATASET_SCHEME_CONFIG.obs,
    leading_hours=[6., 12., 24., 48., 72.],  # hours
    entry_neglect_window=6.,  # hours
    minimum_acquisitions=2,  # number of observables acquisitions.
    recovery_window=12.)  # hours


class TVxAKIMIMICIVDatasetSchemeConfig(TVxEHRSchemeConfig):
    @staticmethod
    def from_mimiciv_dataset_scheme_config(config: AKIMIMICIVDatasetSchemeConfig) -> 'TVxAKIMIMICIVDatasetSchemeConfig':
        return TVxAKIMIMICIVDatasetSchemeConfig(
            gender=config.gender,
            ethnicity=config.propose_target_scheme_name(config.suffixes.ethnicity),
            dx_discharge='dx_icd9',
            obs=config.obs,
            icu_inputs=config.propose_target_scheme_name(config.suffixes.icu_inputs),
            icu_procedures=config.propose_target_scheme_name(config.suffixes.icu_procedures),
            hosp_procedures=config.propose_target_scheme_name(config.suffixes.hosp_procedures),
            outcome='dx_icd9_v1')


DEFAULT_TVX_AKI_MIMICIV_DATASET_SCHEME_CONFIG: Final[TVxAKIMIMICIVDatasetSchemeConfig] = \
    TVxAKIMIMICIVDatasetSchemeConfig.from_mimiciv_dataset_scheme_config(DEFAULT_AKI_MIMICIV_DATASET_SCHEME_CONFIG)

DEFAULT_TVX_AKI_SPLITS: Final[TVxEHRSplitsConfig] = TVxEHRSplitsConfig(split_quantiles=[0.6, 0.7, 0.8], seed=0,
                                                                       discount_first_admission=False,
                                                                       balance='admissions')


class TVxAKIMIMICIVDatasetConfig(TVxEHRConfig):
    scheme: TVxAKIMIMICIVDatasetSchemeConfig = field(
        default_factory=lambda: DEFAULT_TVX_AKI_MIMICIV_DATASET_SCHEME_CONFIG,
        kw_only=True)
    demographic: DemographicVectorConfig = field(default_factory=lambda: DEFAULT_AKI_DEMOGRAPHIC, kw_only=True)
    leading_observable: LeadingObservableExtractorConfig = field(default_factory=DEFAULT_AKI_LEAD_EXTRACTION_FACTORY,
                                                                 kw_only=True)
    sample: Optional[TVxEHRSampleConfig] = field(default=None, kw_only=True)
    splits: Optional[TVxEHRSplitsConfig] = field(default_factory=lambda: DEFAULT_TVX_AKI_SPLITS, kw_only=True)
    numerical_processors: DatasetNumericalProcessorsConfig = field(default_factory=DatasetNumericalProcessorsConfig)
    interventions: bool = True
    observables: bool = True
    time_binning: Optional[float] = None
    interventions_segmentation: bool = True


class TVxAKIMIMICIVDataset(TVxEHR):
    config: TVxAKIMIMICIVDatasetConfig = field(kw_only=True)
    dataset: AKIMIMICIVDataset = field(kw_only=True)

    @classmethod
    def _setup_pipeline(cls, config: Config) -> AbstractTVxPipeline:
        pipeline = [
            SampleSubjects(),
            RandomSplits(),
            ObsIQROutlierRemover(),
            ObsAdaptiveScaler(),
            InputScaler(),
            TVxConcepts(),
            ExcludeShortAdmissions(),
            ObsTimeBinning(),
            LeadingObservableExtraction(),
            InterventionSegmentation(),
        ]
        return AbstractTVxPipeline(transformations=pipeline)
