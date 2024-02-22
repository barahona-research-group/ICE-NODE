from dataclasses import field

import pandas as pd

from lib.ehr.dataset import AbstractDatasetPipeline, Dataset, AbstractDatasetPipelineConfig, DatasetTables
from lib.ehr.example_datasets.mimic4 import MIMICIVSQLTablesInterface, MIMICIVSQLTablesConfig, MIMICIVDatasetScheme, \
    MIMICIVSQLConfig, MIMICIVDatasetSchemeConfig
from lib.ehr.pipeline import SetIndex, CastTimestamps, SetCodeIntegerIndices, \
    SelectSubjectsWithObservation, ProcessOverlappingAdmissions, FilterSubjectsNegativeAdmissionLengths, \
    FilterClampTimestampsToAdmissionInterval, FilterUnsupportedCodes, ICUInputRateUnitConversion, \
    FilterInvalidInputRatesSubjects, SetAdmissionRelativeTimes, DatasetPipeline


class MIMICIVDatasetPipelineConfig(AbstractDatasetPipelineConfig):
    overlap_merge: bool = True


class AKIMIMICIVDatasetSchemeConfig(MIMICIVDatasetSchemeConfig):
    gender: str = 'mf_gender'
    ethnicity: str = 'ethnicity'
    dx_discharge: str = 'dx_mixed_icd'
    obs: str = 'obs'
    icu_inputs: str = 'icu_inputs'
    icu_procedures: str = 'icu_procedures'
    hosp_procedures: str = 'pr_mixed_icd'
    name_prefix: str = 'mimic4_aki_study'
    resources_dir: str = 'mimic4_aki_study'


class AKIMIMICIVDatasetConfig(MIMICIVSQLConfig):
    tables: MIMICIVSQLTablesConfig = field(default_factory=MIMICIVSQLTablesConfig, kw_only=True)
    pipeline: MIMICIVDatasetPipelineConfig = MIMICIVDatasetPipelineConfig()
    scheme: AKIMIMICIVDatasetSchemeConfig = AKIMIMICIVDatasetSchemeConfig()


class MIMICIVDataset(Dataset):
    scheme: MIMICIVDatasetScheme

    @staticmethod
    def icu_inputs_uom_normalization(config: AKIMIMICIVDatasetConfig) -> pd.DataFrame:
        c_icu_inputs = config.tables.icu_inputs
        c_universal_uom = c_icu_inputs.derived_universal_unit
        c_code = c_icu_inputs.code_alias
        c_unit = c_icu_inputs.amount_unit_alias
        c_normalization = c_icu_inputs.derived_unit_normalization_factor
        df = config.scheme.icu_inputs_uom_normalization_table.astype({c_normalization: float})

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
    def load_dataset_scheme(config: AKIMIMICIVDatasetConfig) -> MIMICIVDatasetScheme:
        sql = MIMICIVSQLTablesInterface(config.tables)
        return sql.dataset_scheme_from_selection(config=config.scheme)

    @classmethod
    def load_tables(cls, config: AKIMIMICIVDatasetConfig, scheme: MIMICIVDatasetScheme) -> DatasetTables:
        sql = MIMICIVSQLTablesInterface(config.tables)
        return sql.load_tables(scheme)

    @classmethod
    def _setup_core_pipeline(cls, config: AKIMIMICIVDatasetConfig) -> AbstractDatasetPipeline:
        pconfig = config.pipeline
        conversion_table = cls.icu_inputs_uom_normalization(config)
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
        return DatasetPipeline(transformations=pipeline)
