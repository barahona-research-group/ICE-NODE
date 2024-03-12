from dataclasses import field
from typing import List

import equinox as eqx
import numpy as np
import pandas as pd
import pytest

from lib.ehr.dataset import Report, Dataset, RatedInputTableConfig, DatasetSchemeConfig, DatasetConfig, \
    AbstractDatasetPipeline
from lib.ehr.transformations import ICUInputRateUnitConversion, SetIndex, FilterInvalidInputRatesSubjects
from test.ehr.conftest import NaiveDataset


class MockDatasetSchemeConfig(DatasetSchemeConfig):
    _icu_inputs_uom_normalization_table: pd.DataFrame = field(kw_only=True)

    @property
    def icu_inputs_uom_normalization_table(self) -> pd.DataFrame:
        return self._icu_inputs_uom_normalization_table

    @classmethod
    def _setup_pipeline(cls, config: DatasetConfig) -> AbstractDatasetPipeline:
        return AbstractDatasetPipeline(transformations=[SetIndex()])


class MockMIMICIVDataset(NaiveDataset):
    @staticmethod
    def icu_inputs_uom_normalization(icu_inputs_config: RatedInputTableConfig,
                                     icu_inputs_uom_normalization_table: pd.DataFrame) -> pd.DataFrame:
        return icu_inputs_uom_normalization_table


@pytest.fixture
def unit_converter_table(dataset_config, dataset_tables):
    if 'icu_inputs' not in dataset_tables.tables_dict or len(dataset_tables.icu_inputs) == 0:
        pytest.skip("No ICU inputs in dataset.")
    c_code = dataset_config.tables.icu_inputs.code_alias
    c_amount_unit = dataset_config.tables.icu_inputs.amount_unit_alias
    c_norm_factor = dataset_config.tables.icu_inputs.derived_unit_normalization_factor
    c_universal_unit = dataset_config.tables.icu_inputs.derived_universal_unit
    icu_inputs = dataset_tables.icu_inputs

    table = pd.DataFrame(columns=[c_code, c_amount_unit],
                         data=[(code, unit) for code, unit in
                               icu_inputs.groupby([c_code, c_amount_unit]).groups.keys()])

    for code, df in table.groupby(c_code):
        units = df[c_amount_unit].unique()
        universal_unit = np.random.choice(units, size=1)[0]
        norm_factor = 1
        if len(units) > 1:
            norm_factor = np.random.choice([1e-3, 100, 10, 1e3], size=len(units))
            norm_factor = np.where(units == universal_unit, 1, norm_factor)
        table.loc[df.index, c_norm_factor] = norm_factor
        table.loc[df.index, c_universal_unit] = universal_unit
    return table


@pytest.fixture
def mimiciv_dataset_scheme_config(ethnicity_scheme_name: str,
                                  gender_scheme_name: str,
                                  dx_scheme_name: str,
                                  icu_proc_scheme_name: str,
                                  icu_inputs_scheme_name: str,
                                  observation_scheme_name: str,
                                  hosp_proc_scheme_name: str,
                                  unit_converter_table) -> MockDatasetSchemeConfig:
    return MockDatasetSchemeConfig(
        ethnicity=ethnicity_scheme_name,
        gender=gender_scheme_name,
        dx_discharge=dx_scheme_name,
        icu_procedures=icu_proc_scheme_name,
        icu_inputs=icu_inputs_scheme_name,
        obs=observation_scheme_name,
        hosp_procedures=hosp_proc_scheme_name,
        _icu_inputs_uom_normalization_table=unit_converter_table)


@pytest.fixture
def mimiciv_dataset_config(mimiciv_dataset_scheme_config, dataset_tables_config):
    return DatasetConfig(scheme=mimiciv_dataset_scheme_config, tables=dataset_tables_config)


@pytest.fixture
def mimiciv_dataset(mimiciv_dataset_config, dataset_tables) -> MockMIMICIVDataset:
    ds = MockMIMICIVDataset(config=dataset_config)
    return eqx.tree_at(lambda x: x.tables, ds, dataset_tables,
                       is_leaf=lambda x: x is None).execute_pipeline()


class TestUnitConversion:

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


class TestFilterInvalidInputRatesSubjects:
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
