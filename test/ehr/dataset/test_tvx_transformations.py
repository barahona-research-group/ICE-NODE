from typing import Tuple, Type

import pytest

from lib import Config
from lib.ehr import Dataset
from lib.ehr.tvx_ehr import TrainableTransformation
from lib.ehr.tvx_transformations import SampleSubjects, CodedValueScaler, ObsAdaptiveScaler, InputScaler


@pytest.mark.parametrize('seed', [0, 1])
def test_sample_subjects(indexed_dataset: Dataset, seed: int):
    indexed_dataset = indexed_dataset.execute_pipeline()

    if len(indexed_dataset.tables.static) <= 1:
        pytest.skip("Only one subject in dataset.")
    if len(indexed_dataset.tables.admissions) == 0:
        pytest.skip("No admissions table found in dataset. The sampling will result on empty dataset.")

    n_subjects = len(indexed_dataset.tables.static)
    n_sample = int(n_subjects / 5)

    for offset in range(n_subjects - n_sample):
        sampled_dataset, aux = SampleSubjects(n_subjects=n_sample, seed=seed, offset=offset)(indexed_dataset, {})
        # assert serialization/deserialization of aux reports
        assert all(isinstance(v.as_dict(), dict) for v in aux['report'])
        assert all([Config.from_dict(v.to_dict()).equals(v) for v in aux['report']])

        sampled_subjects = sampled_dataset.subject_ids
        assert len(sampled_subjects) == n_sample
        assert len(set(sampled_subjects)) == n_sample
        assert set(sampled_subjects).issubset(set(indexed_dataset.subject_ids))



@pytest.mark.parametrize('fit_only', [True, False])
@pytest.mark.parametrize('use_float16', [True, False])
@pytest.mark.parametrize('scaler', [('obs', ObsAdaptiveScaler), ('icu_inputs', InputScaler)])
def test_trainable_transformer(preprocessed_dataset: Dataset, use_float16: bool, fit_only: bool,
                               scaler: Tuple[str, Type[TrainableTransformation]]):
    if len(preprocessed_dataset.tables.static) < 5 or len(getattr(preprocessed_dataset.tables, scaler[0])) == 0:
        pytest.skip("Not enough subjects in dataset or no data to scale.")
    table_name, scaler_class = scaler
    scaler_name = f'{table_name}_scaler'

    with pytest.raises(AssertionError):
        scaler_class(use_float16=use_float16, transformer_key=scaler_name,
                     fit_only=fit_only, splits_key='splits',
                     training_split_index=0)(preprocessed_dataset, {})

    aux = {'splits': [preprocessed_dataset.subject_ids[:3], preprocessed_dataset.subject_ids[3:]]}
    transformer = scaler_class(use_float16=use_float16, transformer_key=scaler_name,
                               splits_key='splits',
                               fit_only=fit_only,
                               training_split_index=0)

    assert isinstance(transformer, TrainableTransformation)
    assert transformer.fit_only == fit_only
    assert transformer.transformer_key == scaler_name

    scaled_ds, aux = transformer(preprocessed_dataset, aux)
    scaler = aux[scaler_name]
    assert scaler is not None
    assert isinstance(scaler, CodedValueScaler)
    assert scaler.table(scaled_ds) is getattr(scaled_ds.tables, table_name)
    assert scaler.table(preprocessed_dataset) is getattr(preprocessed_dataset.tables, table_name)
    assert scaler.use_float16 == use_float16

    table0 = scaler.table(preprocessed_dataset)
    table1 = scaler.table(scaled_ds)
    c_value = scaler.value_column(scaled_ds)
    c_code = scaler.code_column(scaled_ds)
    assert c_value in table1.columns
    assert c_code in table1.columns
    if fit_only:
        assert table1 is table0
        assert table1[c_value].dtype == scaler.original_dtype
    else:
        assert table1 is not table0

        if use_float16:
            assert table1[c_value].dtype == np.float16
        else:
            assert table1[c_value].dtype == table0[c_value].dtype

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
