from dataclasses import field
from typing import List

import equinox as eqx
import numpy as np
import pandas as pd
import pytest

from lib.ehr.dataset import Report, Dataset, RatedInputTableConfig, DatasetSchemeConfig, DatasetConfig, \
    AbstractDatasetPipeline
from lib.ehr.transformations import ICUInputRateUnitConversion, SetIndex, FilterInvalidInputRatesSubjects
from test.ehr.conftest import NaiveDataset, MockMIMICIVDataset


class TestUnitConversionAndFilterInvalidInputRates:

    @pytest.fixture
    def fixed_dataset(self, mimiciv_dataset: MockMIMICIVDataset) -> Dataset:
        return ICUInputRateUnitConversion.apply(mimiciv_dataset, Report())[0]

    @pytest.fixture
    def icu_inputs_unfixed(self, mimiciv_dataset: MockMIMICIVDataset):
        return mimiciv_dataset.tables.icu_inputs

    @pytest.fixture
    def icu_inputs_fixed(self, fixed_dataset: Dataset):
        return fixed_dataset.tables.icu_inputs

    @pytest.fixture
    def derived_icu_inputs_cols(self, icu_input_derived_unit_normalization_factor: str,
                                icu_input_derived_universal_unit: str,
                                icu_input_derived_normalized_amount: str,
                                icu_input_derived_normalized_amount_per_hour: str):
        return [icu_input_derived_unit_normalization_factor, icu_input_derived_universal_unit,
                icu_input_derived_normalized_amount, icu_input_derived_normalized_amount_per_hour]

    def test_icu_input_rate_unit_conversion(self,
                                            icu_inputs_fixed: pd.DataFrame,
                                            icu_inputs_unfixed: pd.DataFrame,
                                            unit_converter_table: pd.DataFrame,
                                            derived_icu_inputs_cols: List[str],
                                            icu_input_code_alias: str,
                                            icu_input_amount_alias: str,
                                            icu_input_amount_unit_alias: str,
                                            icu_input_derived_universal_unit: str,
                                            icu_input_derived_unit_normalization_factor: str,
                                            icu_input_derived_normalized_amount: str):
        assert all(c not in icu_inputs_unfixed.columns for c in derived_icu_inputs_cols)
        assert all(c in icu_inputs_fixed.columns for c in derived_icu_inputs_cols)

        # For every (code, unit) pair, a unique normalization factor and universal unit is assigned.
        for (code, unit), inputs_df in icu_inputs_fixed.groupby([icu_input_code_alias, icu_input_amount_unit_alias]):
            ctable = unit_converter_table[(unit_converter_table[icu_input_code_alias] == code)]
            ctable = ctable[ctable[icu_input_amount_unit_alias] == unit]
            norm_factor = ctable[icu_input_derived_unit_normalization_factor].iloc[0]
            universal_unit = ctable[icu_input_derived_universal_unit].iloc[0]

            assert inputs_df[icu_input_derived_universal_unit].unique() == universal_unit
            assert inputs_df[icu_input_derived_unit_normalization_factor].unique() == norm_factor
            assert inputs_df[icu_input_derived_normalized_amount].equals(
                inputs_df[icu_input_amount_alias] * norm_factor)


    @pytest.fixture
    def nan_inputs_dataset(self, fixed_dataset: pd.DataFrame, admission_id_alias: str,
                           icu_input_derived_normalized_amount_per_hour: str):
        icu_inputs = fixed_dataset.tables.icu_inputs.copy()
        admission_id = icu_inputs.iloc[0][admission_id_alias]
        icu_inputs.loc[
            icu_inputs[admission_id_alias] == admission_id, icu_input_derived_normalized_amount_per_hour] = np.nan
        return eqx.tree_at(lambda x: x.tables.icu_inputs, fixed_dataset, icu_inputs)

    @pytest.fixture
    def filtered_dataset(self, nan_inputs_dataset: Dataset):
        return FilterInvalidInputRatesSubjects.apply(nan_inputs_dataset, Report())[0]

    def test_filter_invalid_input_rates_subjects(self, nan_inputs_dataset: Dataset,
                                                 filtered_dataset: Dataset,
                                                 admission_id_alias: str,
                                                 subject_id_alias: str,
                                                 icu_input_derived_normalized_amount_per_hour: str):
        icu_inputs0 = nan_inputs_dataset.tables.icu_inputs
        admissions0 = nan_inputs_dataset.tables.admissions
        static0 = nan_inputs_dataset.tables.static

        icu_inputs1 = filtered_dataset.tables.icu_inputs
        admissions1 = filtered_dataset.tables.admissions
        static1 = filtered_dataset.tables.static

        admission_id = icu_inputs0.iloc[0][admission_id_alias]
        subject_id = static0[static0.index == admissions0.loc[admission_id, subject_id_alias]].index[0]
        subject_admissions = admissions0[admissions0[subject_id_alias] == subject_id]

        assert subject_id not in static1.index
        assert not subject_admissions.index.isin(admissions1.index).all()
        assert not subject_admissions.index.isin(icu_inputs1[admission_id_alias]).all()
        assert icu_inputs0[icu_input_derived_normalized_amount_per_hour].isna().any()
        assert not icu_inputs1[icu_input_derived_normalized_amount_per_hour].isna().any()
