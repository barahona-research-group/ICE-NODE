from typing import Type, List

import equinox as eqx
import numpy as np
import pytest

from lib.ehr import Dataset
from lib.ehr.tvx_ehr import TrainableTransformation, TVxEHR, TVxReport, TVxEHRSampleConfig, ScalerConfig, \
    DatasetNumericalProcessorsConfig, ScalersConfig, OutlierRemoversConfig, IQROutlierRemoverConfig, \
    DatasetNumericalProcessors
from lib.ehr.tvx_transformations import SampleSubjects, CodedValueScaler, ObsAdaptiveScaler, InputScaler, \
    ObsIQROutlierRemover


@pytest.fixture
def multi_subjects_ehr(tvx_ehr: TVxEHR):
    if len(tvx_ehr.dataset.tables.static) <= 1:
        pytest.skip("Only one subject in dataset.")
    if len(tvx_ehr.dataset.tables.admissions) == 0:
        pytest.skip("No admissions table found in dataset. The sampling will result on empty dataset.")

    return tvx_ehr


@pytest.fixture
def large_ehr(multi_subjects_ehr: TVxEHR):
    if len(multi_subjects_ehr.dataset.tables.static) <= 5:
        pytest.skip("Only one subject in dataset.")
    if len(multi_subjects_ehr.dataset.tables.admissions) == 0:
        pytest.skip("No admissions table found in dataset. The sampling will result on empty dataset.")

    return multi_subjects_ehr


class TestSampleSubjects:

    @pytest.fixture(params=[(1, 3), (111, 9)])
    def sampled_tvx_ehr(self, multi_subjects_ehr: TVxEHR, request):
        seed, offset = request.param
        n_subjects = len(multi_subjects_ehr.dataset.tables.static) // 5
        sample = TVxEHRSampleConfig(seed=seed, n_subjects=n_subjects, offset=offset)
        multi_subjects_ehr = eqx.tree_at(lambda x: x.config.sample, multi_subjects_ehr, sample,
                                         is_leaf=lambda x: x is None)
        return SampleSubjects.apply(multi_subjects_ehr, TVxReport())[0]

    def test_sample_subjects(self, multi_subjects_ehr: TVxEHR, sampled_tvx_ehr: TVxEHR):
        original_subjects = multi_subjects_ehr.dataset.tables.static.index
        sampled_subjects = sampled_tvx_ehr.dataset.tables.static.index
        assert len(sampled_subjects) == len(original_subjects) // 5
        assert len(set(sampled_subjects)) == len(sampled_subjects)
        assert set(sampled_subjects).issubset(set(original_subjects))


class TestTrainableTransformer:

    @pytest.fixture(params=['obs', 'icu_inputs'])
    def scalable_table_name(self, request):
        return request.param

    @pytest.fixture(params=[True, False])
    def use_float16(self, request):
        return request.param

    @pytest.fixture
    def scaler_class(self, scalable_table_name) -> TrainableTransformation:
        return {'obs': ObsAdaptiveScaler, 'icu_inputs': InputScaler}[scalable_table_name]

    @pytest.fixture(params=[True, False])
    def numerical_processor_config(self, request) -> DatasetNumericalProcessorsConfig:
        null = request.param
        if null:
            return DatasetNumericalProcessorsConfig()
        scalers_conf = ScalersConfig(obs=ScalerConfig(use_float16=True),
                                     icu_inputs=ScalerConfig(use_float16=True))
        outliers_conf = OutlierRemoversConfig(obs=IQROutlierRemoverConfig())
        return DatasetNumericalProcessorsConfig(scalers_conf, outliers_conf)

    @pytest.fixture
    def large_scalable_splitted_ehr(self, large_ehr: TVxEHR, scalable_table_name: str, use_float16: bool):
        if len(getattr(large_ehr.dataset.tables, scalable_table_name)) == 0:
            pytest.skip(f"No {scalable_table_name} table found in dataset.")
        subjects = large_ehr.dataset.tables.static.index.tolist()
        large_ehr = eqx.tree_at(lambda x: x.splits, large_ehr, (tuple(subjects),),
                                is_leaf=lambda x: x is None)

        large_ehr = eqx.tree_at(
            lambda x: getattr(x.config.numerical_processors.scalers, scalable_table_name),
            large_ehr, ScalerConfig(use_float16=use_float16),
            is_leaf=lambda x: x is None)
        return large_ehr

    @pytest.fixture
    def scaled_ehr(self, large_scalable_splitted_ehr, scaler_class: Type[TrainableTransformation]):
        return scaler_class.apply(large_scalable_splitted_ehr, TVxReport())[0]

    def test_trainable_transformer(self, large_scalable_splitted_ehr: TVxEHR,
                                   scaled_ehr: TVxEHR,
                                   scalable_table_name: str,
                                   use_float16: bool):
        assert getattr(large_scalable_splitted_ehr.numerical_processors.scalers, scalable_table_name) is None
        scaler = getattr(scaled_ehr.numerical_processors.scalers, scalable_table_name)
        assert isinstance(scaler, CodedValueScaler)
        assert scaler.table_getter(scaled_ehr.dataset) is getattr(scaled_ehr.dataset.tables, scalable_table_name)
        assert scaler.table_getter(large_scalable_splitted_ehr.dataset) is getattr(
            large_scalable_splitted_ehr.dataset.tables,
            scalable_table_name)
        assert scaler.config.use_float16 == use_float16

        table0 = scaler.table_getter(large_scalable_splitted_ehr.dataset)
        table1 = scaler.table_getter(scaled_ehr.dataset)
        assert scaler.value_column in table1.columns
        assert scaler.code_column in table1.columns
        assert table1 is not table0
        if use_float16:
            assert table1[scaler.value_column].dtype == np.float16
        else:
            assert table1[scaler.value_column].dtype == table0[scaler.value_column].dtype

    @pytest.fixture
    def processed_ehr(self, large_scalable_splitted_ehr: TVxEHR, numerical_processor_config: DatasetNumericalProcessorsConfig) -> TVxEHR:
        large_scalable_splitted_ehr = eqx.tree_at(lambda x: x.config.numerical_processors, large_scalable_splitted_ehr, numerical_processor_config)
        return large_scalable_splitted_ehr.execute_external_transformations([ObsIQROutlierRemover(), InputScaler(), ObsAdaptiveScaler()])

    @pytest.fixture
    def fitted_numerical_processors(self, processed_ehr: TVxEHR) -> DatasetNumericalProcessors:
        return processed_ehr.numerical_processors

    def test_numerical_processors_serialization(self, fitted_numerical_processors: DatasetNumericalProcessors,
                                                tmpdir: str):
        path = f'{tmpdir}/numerical_processors.h5'
        fitted_numerical_processors.save(path, key='numerical_processors')
        loaded_numerical_processors = DatasetNumericalProcessors.load(path, key='numerical_processors')
        assert fitted_numerical_processors.equals(loaded_numerical_processors)



# def test_obs_minmax_scaler(int_indexed_dataset: Dataset):
#     assert False
#
#
# def test_obs_adaptive_scaler(int_indexed_dataset: Dataset):
#     assert False
#
#
# def test_obs_iqr_outlier_remover(indexed_dataset: Dataset):
#     assert False


@pytest.mark.parametrize('splits', [[0.5], [0.2, 0.5, 0.7], [0.1, 0.2, 0.3, 0.4, 0.5]])
def test_random_splits(indexed_dataset: Dataset, splits: List[float]):
    # The logic of splits already tested in test.ehr.dataset.test_dataset.
    # Maybe assert that functions are called with the correct arguments.
    pass
