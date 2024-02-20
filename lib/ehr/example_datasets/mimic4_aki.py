import pandas as pd

from lib.ehr import DatasetConfig
from lib.ehr import resources_dir
from lib.ehr.dataset import AbstractDatasetPipeline, Dataset, AbstractDatasetPipelineConfig, DatasetTables
from lib.ehr.example_datasets.mimic4 import MIMICIVSQL, MIMICIVSQLConfig, MIMICIVDatasetScheme
from lib.ehr.pipeline import SetIndex, CastTimestamps, SetCodeIntegerIndices, \
    SelectSubjectsWithObservation, ProcessOverlappingAdmissions, FilterSubjectsNegativeAdmissionLengths, \
    FilterClampTimestampsToAdmissionInterval, FilterUnsupportedCodes, ICUInputRateUnitConversion, \
    FilterInvalidInputRatesSubjects, SetAdmissionRelativeTimes


class MIMIC4DatasetPipelineConfig(AbstractDatasetPipelineConfig):
    overlap_merge: bool


class MIMIC4DatasetConfig(DatasetConfig):
    pipeline: MIMIC4DatasetPipelineConfig
    sql: MIMICIVSQLConfig
    resources_dir: str = "mimic4_aki_study"
    scheme_prefix: str = "mimic4_aki"


class MIMIC4Dataset(Dataset):

    def __init__(self, config: MIMIC4DatasetConfig, tables: DatasetTables):
        super().__init__(config, tables)

    @staticmethod
    def load_icu_inputs_conversion_table(config: MIMIC4DatasetConfig) -> pd.DataFrame:
        return pd.read_csv(resources_dir(config.resources_dir) / "icu_inputs_conversion_table.csv")

    @staticmethod
    def load_dataset_scheme(config: MIMIC4DatasetConfig) -> MIMICIVDatasetScheme:
        sql = MIMICIVSQL(config.sql)
        load_df = lambda path: pd.read_csv(resources_dir(config.resources_dir) / "scheme" / path)
        return sql.dataset_scheme_from_selection(
            name_prefix=config.scheme_prefix,
            ethnicity=load_df("ethnicity.csv"),
            gender=load_df("gender.csv"),
            dx_discharge=load_df("dx_discharge.csv"),
            obs=load_df("obs.csv"),
            icu_procedures=load_df("icu_procedures.csv"),
            icu_inputs=load_df("icu_inputs.csv"),
            hosp_procedures=load_df("hosp_procedures.csv")
        )

    @classmethod
    def load_tables(cls, config: MIMIC4DatasetConfig) -> DatasetTables:
        sql = MIMICIVSQL(config.sql)
        return sql.load_tables(cls.load_dataset_scheme(config))

    def _setup_core_pipeline(cls, config: MIMIC4DatasetConfig) -> AbstractDatasetPipeline:
        pconfig = config.pipeline
        pipeline = [
            SetIndex(),
            SelectSubjectsWithObservation(name='select_with_aki_info',
                                          code='renal_aki.aki_binary'),
            CastTimestamps(),
            ProcessOverlappingAdmissions(merge=pconfig.overlap_merge),
            FilterSubjectsNegativeAdmissionLengths(),
            FilterClampTimestampsToAdmissionInterval(),
            FilterUnsupportedCodes(),
            ICUInputRateUnitConversion(conversion_table=cls.load_icu_inputs_conversion_table()),
            FilterInvalidInputRatesSubjects(),
            SetCodeIntegerIndices(),
            SetAdmissionRelativeTimes()
        ]
        return AbstractDatasetPipeline(transformations=pipeline)
