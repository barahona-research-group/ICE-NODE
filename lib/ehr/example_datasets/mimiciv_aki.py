from dataclasses import field
from typing import Literal, Final, Optional, List, Callable

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
    InputScaler, ObsTimeBinning, TVxConcepts, InterventionSegmentation, ExcludeShortAdmissions

OBSERVABLE_AKI_TARGET_CODE: Final[str] = 'renal_aki.aki_binary'


class AKIMIMICIVDatasetSchemeConfig(MIMICIVDatasetSchemeConfig):
    name_prefix: str = 'mimiciv.aki_study'
    resources_dir: str = 'mimiciv/aki_study'


DEFAULT_AKI_MIMICIV_DATASET_SCHEME_CONFIG: Final[AKIMIMICIVDatasetSchemeConfig] = AKIMIMICIVDatasetSchemeConfig()


class AKIMIMICIVDatasetConfig(MIMICIVDatasetConfig):
    scheme: AKIMIMICIVDatasetSchemeConfig = DEFAULT_AKI_MIMICIV_DATASET_SCHEME_CONFIG
    overlapping_admissions: Literal["merge", "remove"] = "merge"
    filter_subjects_with_observation: str = OBSERVABLE_AKI_TARGET_CODE


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
    entry_neglect_window=4.,  # hours
    minimum_acquisitions=2,  # number of observables acquisitions.
    recovery_window=12.)  # hours


class TVxAKIMIMICIVDatasetSchemeConfig(TVxEHRSchemeConfig):
    @staticmethod
    def from_mimiciv_dataset_scheme_config(config: AKIMIMICIVDatasetSchemeConfig) -> 'TVxAKIMIMICIVDatasetSchemeConfig':
        return TVxAKIMIMICIVDatasetSchemeConfig(
            gender=config.gender,
            ethnicity=config.propose_target_scheme_name('ethnicity'),
            dx_discharge='dx_icd9',
            obs=config.obs,
            icu_inputs=config.propose_target_scheme_name('icu_inputs'),
            icu_procedures=config.propose_target_scheme_name('icu_procedures'),
            hosp_procedures=config.propose_target_scheme_name('hosp_procedures'),
            outcome='dx_icd9_v1')


DEFAULT_TVX_AKI_MIMICIV_DATASET_SCHEME_CONFIG: Final[TVxAKIMIMICIVDatasetSchemeConfig] = \
    TVxAKIMIMICIVDatasetSchemeConfig.from_mimiciv_dataset_scheme_config(DEFAULT_AKI_MIMICIV_DATASET_SCHEME_CONFIG)


class TVxAKIMIMICIVDatasetConfig(TVxEHRConfig):
    scheme: TVxAKIMIMICIVDatasetSchemeConfig = field(default=DEFAULT_TVX_AKI_MIMICIV_DATASET_SCHEME_CONFIG,
                                                     kw_only=True)
    demographic: DemographicVectorConfig = field(default=DEFAULT_AKI_DEMOGRAPHIC, kw_only=True)
    leading_observable: LeadingObservableExtractorConfig = field(default_factory=DEFAULT_AKI_LEAD_EXTRACTION_FACTORY,
                                                                 kw_only=True)
    sample: Optional[TVxEHRSampleConfig] = field(default=None, kw_only=True)
    splits: Optional[TVxEHRSplitsConfig] = field(default=None, kw_only=True)
    numerical_processors: DatasetNumericalProcessorsConfig = DatasetNumericalProcessorsConfig()
    interventions: bool = True
    observables: bool = True
    time_binning: Optional[float] = None
    interventions_segmentation: bool = False

    @staticmethod
    def compile_default_from_arguments(
            seed: int = 0,
            extract_interventions: bool = True,
            extract_observables: bool = True,
            interventions_segmentation: bool = True,
            admission_minimum_los: Optional[float] = 12.0,
            time_binning: Optional[float] = None,
            sample_n_subjects: Optional[int] = None,
            sample_offset: Optional[int] = None,
            split_balance_discount_first_admission: Optional[bool] = None,
            split_balance: Optional[str] = None, split_quantiles: Optional[List[float]] = None):
        sample_kwargs = dict(n_subjects=sample_n_subjects, offset=sample_offset, seed=seed)
        sample_kwargs = {k: v for k, v in sample_kwargs.items() if v is not None}
        split_kwargs = dict(balance_discount_first_admission=split_balance_discount_first_admission,
                            balance=split_balance, quantiles=split_quantiles, seed=seed)
        split_kwargs = {k: v for k, v in split_kwargs.items() if v is not None}
        sample = TVxEHRSampleConfig(**sample_kwargs) if len(sample_kwargs) > 0 else None
        splits = TVxEHRSplitsConfig(**split_kwargs) if len(split_kwargs) > 0 else None

        return TVxAKIMIMICIVDatasetConfig(
            sample=sample,
            splits=splits,
            interventions=extract_interventions,
            observables=extract_observables,
            interventions_segmentation=interventions_segmentation,
            time_binning=time_binning,
            admission_minimum_los=admission_minimum_los)


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
            LeadingObservableExtractorConfig(),
            InterventionSegmentation(),
        ]
        return AbstractTVxPipeline(transformations=pipeline)
