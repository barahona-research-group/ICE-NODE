from typing import Optional, Tuple

import pandas as pd

from lib.ehr import resources_dir
from lib.ehr.coding_scheme import CodeMap, CodeMapConfig, Ethnicity, CodingScheme, CodingSchemeConfig
from lib.ehr.dataset import AbstractDatasetPipeline, Dataset, AbstractDatasetPipelineConfig, DatasetTables, \
    DatasetSchemeConfig
from lib.ehr.example_datasets.mimic4 import MIMICIVSQLTablesInterface, MIMICIVSQLTablesConfig, MIMICIVDatasetScheme, \
    MIMICIVSQLConfig, MIMICIVDatasetSchemeMapsFiles
from lib.ehr.pipeline import SetIndex, CastTimestamps, SetCodeIntegerIndices, \
    SelectSubjectsWithObservation, ProcessOverlappingAdmissions, FilterSubjectsNegativeAdmissionLengths, \
    FilterClampTimestampsToAdmissionInterval, FilterUnsupportedCodes, ICUInputRateUnitConversion, \
    FilterInvalidInputRatesSubjects, SetAdmissionRelativeTimes


class MIMICIVDatasetPipelineConfig(AbstractDatasetPipelineConfig):
    overlap_merge: bool


class MIMICIVDatasetSchemeConfig(DatasetSchemeConfig):
    prefix: str = 'mimic4_study_aki'
    map_files: MIMICIVDatasetSchemeMapsFiles = MIMICIVDatasetSchemeMapsFiles()
    icu_inputs_uom_normalization: Optional[Tuple[str]] = ("uom_normalization", "icu_inputs.csv")


class MIMICIVDatasetConfig(MIMICIVSQLConfig):
    pipeline: MIMICIVDatasetPipelineConfig
    tables: MIMICIVSQLTablesConfig
    scheme: MIMICIVDatasetSchemeConfig


class MIMICIVDataset(Dataset):
    scheme: MIMICIVDatasetScheme

    def __init__(self, config: MIMICIVDatasetConfig, tables: DatasetTables):
        self.scheme = self.load_dataset_scheme(config)
        self.load_maps(config, self.scheme)
        super().__init__(config, tables)

    @staticmethod
    def load_icu_inputs_conversion_table(config: MIMICIVDatasetConfig) -> pd.DataFrame:
        df = pd.read_csv(resources_dir(config.scheme.prefix, *config.scheme.icu_inputs_uom_normalization), dtype=str)
        c_icu_inputs = config.tables.icu_inputs
        c_universal_uom = c_icu_inputs.derived_universal_unit
        c_code = c_icu_inputs.code_alias
        c_unit = c_icu_inputs.amount_unit_alias
        c_normalization = c_icu_inputs.derived_unit_normalization_factor
        df = df.astype({c_normalization: float})

        columns = [c_code, c_unit, c_normalization]
        assert all(c in df.columns for c in columns), \
            f"Columns {columns} not found in icu_inputs.csv"

        if c_universal_uom not in df.columns:
            df[c_universal_uom] = ''
            for (code, uom), uom_df in df.groupby([c_code, c_unit]):
                # Select the first unit associated with 1.0 as a normalization factor.
                index = uom_df[uom_df[c_normalization] == 1.0].first_valid_index()
                assert index is not None, f"No unit associated with 1.0 normalization factor for code {code} and unit {uom}"
                df.loc[uom_df.index, c_universal_uom] = uom_df.loc[index, c_unit]
        return df

    @staticmethod
    def load_dataset_scheme(config: MIMICIVDatasetConfig) -> MIMICIVDatasetScheme:
        sql = MIMICIVSQLTablesInterface(config.tables)
        return sql.dataset_scheme_from_selection(config=config.scheme)

    @classmethod
    def load_tables(cls, config: MIMICIVDatasetConfig) -> DatasetTables:
        sql = MIMICIVSQLTablesInterface(config.tables)
        return sql.load_tables(cls.load_dataset_scheme(config))

    def _setup_core_pipeline(cls, config: MIMICIVDatasetConfig) -> AbstractDatasetPipeline:
        pconfig = config.pipeline
        conversion_table = cls.load_icu_inputs_conversion_table(config)
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
        return AbstractDatasetPipeline(transformations=pipeline)


