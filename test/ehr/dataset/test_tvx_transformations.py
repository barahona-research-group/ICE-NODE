from typing import Type, List, Dict

import equinox as eqx
import numpy as np
import pytest

from lib.ehr import Dataset, CodesVector, InpatientInput, InpatientObservables, DemographicVectorConfig, StaticInfo
from lib.ehr.tvx_ehr import TrainableTransformation, TVxEHR, TVxReport, TVxEHRSampleConfig, ScalerConfig, \
    DatasetNumericalProcessorsConfig, ScalersConfig, OutlierRemoversConfig, IQROutlierRemoverConfig, \
    DatasetNumericalProcessors
from lib.ehr.tvx_transformations import SampleSubjects, CodedValueScaler, ObsAdaptiveScaler, InputScaler, \
    ObsIQROutlierRemover, TVxConcepts


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
    def processed_ehr(self, large_scalable_splitted_ehr: TVxEHR,
                      numerical_processor_config: DatasetNumericalProcessorsConfig) -> TVxEHR:
        large_scalable_splitted_ehr = eqx.tree_at(lambda x: x.config.numerical_processors, large_scalable_splitted_ehr,
                                                  numerical_processor_config)
        return large_scalable_splitted_ehr.execute_external_transformations(
            [ObsIQROutlierRemover(), InputScaler(), ObsAdaptiveScaler()])

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


class TestTVxConcepts:
    @pytest.fixture(params=['gender', 'age', 'ethnicity'])
    def tvx_ehr_demographic_config(self, request) -> DemographicVectorConfig:
        config = DemographicVectorConfig(False, False, False)
        return eqx.tree_at(lambda x: getattr(x, request.param), config, True)

    @pytest.fixture
    def tvx_ehr_configured_demographic(self, large_ehr: TVxEHR,
                                       tvx_ehr_demographic_config: DemographicVectorConfig) -> TVxEHR:
        return eqx.tree_at(lambda x: x.config.demographic, large_ehr, tvx_ehr_demographic_config)

    @pytest.fixture
    def tvx_concepts_static(self, tvx_ehr_configured_demographic: TVxEHR) -> Dict[str, StaticInfo]:
        return TVxConcepts._static_info(tvx_ehr_configured_demographic, TVxReport())[0]

    def test_tvx_concepts_static(self, large_ehr: TVxEHR,
                                 tvx_concepts_static: Dict[str, StaticInfo],
                                 tvx_ehr_demographic_config: DemographicVectorConfig):
        assert len(tvx_concepts_static) == len(large_ehr.dataset.tables.static)
        for subject_id, static_info in tvx_concepts_static.items():
            if tvx_ehr_demographic_config.gender:
                assert static_info.gender is not None
                assert len(static_info.gender) == len(large_ehr.scheme.gender)
                assert static_info.gender.vec.sum() == 1
                assert static_info.gender.scheme == large_ehr.scheme.gender.name
            else:
                assert static_info.gender is None

            if tvx_ehr_demographic_config.age:
                assert static_info.date_of_birth is not None
                assert static_info.age(static_info.date_of_birth) == 0.0
            else:
                assert static_info.date_of_birth is None

            if tvx_ehr_demographic_config.ethnicity:
                assert static_info.ethnicity is not None
                assert len(static_info.ethnicity) == len(large_ehr.scheme.ethnicity)
                assert static_info.ethnicity.vec.sum() == 1
                assert static_info.ethnicity.scheme == large_ehr.scheme.ethnicity.name

    @pytest.fixture
    def tvx_ehr_with_dx(self, large_ehr: TVxEHR) -> TVxEHR:
        if len(large_ehr.dataset.tables.dx_discharge) == 0:
            pytest.skip("No diagnoses table found in dataset.")
        n = len(large_ehr.dataset.tables.admissions)
        c_admission_id = large_ehr.dataset.config.tables.admissions.admission_id_alias
        random_admission_id = large_ehr.dataset.tables.admissions.index[n // 2]
        dx_discharge = large_ehr.dataset.tables.dx_discharge
        dx_discharge = dx_discharge[dx_discharge[c_admission_id] != random_admission_id]
        return eqx.tree_at(lambda x: x.dataset.tables.dx_discharge, large_ehr, dx_discharge)

    @pytest.fixture
    def admission_dx_codes(self, tvx_ehr_with_dx: TVxEHR) -> Dict[str, CodesVector]:
        return TVxConcepts._dx_discharge(tvx_ehr_with_dx)

    def test_admission_dx_codes(self, tvx_ehr_with_dx: TVxEHR, admission_dx_codes: Dict[str, CodesVector]):
        c_admission_id = tvx_ehr_with_dx.dataset.config.tables.admissions.admission_id_alias
        assert len(admission_dx_codes) == tvx_ehr_with_dx.dataset.tables.dx_discharge[c_admission_id].nunique()
        for admission_id, codes in admission_dx_codes.items():
            assert codes.vec.sum() > 0
            assert codes.vec.dtype == bool
            assert codes.scheme == tvx_ehr_with_dx.scheme.dx_discharge.name
            assert len(codes) == len(tvx_ehr_with_dx.scheme.dx_discharge)

    @pytest.fixture
    def admission_dx_history_codes(self, tvx_ehr_with_dx: TVxEHR,
                                   admission_dx_codes: Dict[str, CodesVector]) -> Dict[str, CodesVector]:
        return TVxConcepts._dx_discharge_history(tvx_ehr_with_dx, admission_dx_codes)

    def test_admission_dx_history_codes(self, tvx_ehr_with_dx: TVxEHR,
                                        admission_dx_codes: Dict[str, CodesVector],
                                        admission_dx_history_codes: Dict[str, CodesVector]):
        assert len(admission_dx_history_codes) == len(tvx_ehr_with_dx.dataset.tables.admissions)
        # because of the random admission_id we removed from dx_discharge.
        assert len(admission_dx_history_codes) == len(admission_dx_codes) + 1
        for subject_id, admission_ids in tvx_ehr_with_dx.subjects_sorted_admission_ids.items():
            assert admission_dx_history_codes[admission_ids[0]].vec.sum() == 0

            accumulation = np.zeros(len(tvx_ehr_with_dx.scheme.dx_discharge), dtype=bool)
            for i, admission_id in enumerate(admission_ids):
                assert admission_id in admission_dx_history_codes
                history = admission_dx_history_codes[admission_id]
                assert history.vec.dtype == bool
                assert history.scheme == tvx_ehr_with_dx.scheme.dx_discharge.name
                assert len(history) == len(tvx_ehr_with_dx.scheme.dx_discharge)
                assert (history.vec == accumulation).all()
                if admission_id in admission_dx_codes:
                    accumulation |= admission_dx_codes[admission_id].vec

    @pytest.fixture
    def admission_outcome(self, tvx_ehr_with_dx: TVxEHR,
                          admission_dx_codes: Dict[str, CodesVector]) -> Dict[str, CodesVector]:
        return TVxConcepts._outcome(tvx_ehr_with_dx, admission_dx_codes)

    def test_admission_outcome(self, tvx_ehr_with_dx: TVxEHR,
                               admission_dx_codes: Dict[str, CodesVector],
                               admission_outcome: Dict[str, CodesVector]):
        assert len(admission_outcome) == len(admission_dx_codes)
        for admission_id, outcome in admission_outcome.items():
            assert outcome.scheme == tvx_ehr_with_dx.scheme.outcome.name
            assert len(outcome) == len(tvx_ehr_with_dx.scheme.outcome)

    @pytest.fixture
    def tvx_ehr_with_icu_inputs(self, large_ehr: TVxEHR) -> TVxEHR:
        if len(large_ehr.dataset.tables.icu_inputs) == 0:
            pytest.skip("No icu_inputs table found in dataset.")
        return large_ehr

    @pytest.fixture
    def admission_icu_inputs(self, tvx_ehr_with_icu_inputs: TVxEHR) -> Dict[str, InpatientInput]:
        return TVxConcepts._icu_inputs(tvx_ehr_with_icu_inputs)

    def test_admission_icu_inputs(self, tvx_ehr_with_icu_inputs: TVxEHR,
                                  admission_icu_inputs: Dict[str, InpatientInput]):
        icu_inputs = tvx_ehr_with_icu_inputs.dataset.tables.icu_inputs
        c_admission_id = tvx_ehr_with_icu_inputs.dataset.config.tables.admissions.admission_id_alias
        assert len(admission_icu_inputs) == len(tvx_ehr_with_icu_inputs.dataset.tables.admissions)
        assert sum(len(inputs.starttime) for inputs in admission_icu_inputs.values()) == len(icu_inputs)

        for admission_id, admission_inputs_df in icu_inputs.groupby(c_admission_id):
            tvx_inputs = admission_icu_inputs[admission_id]
            assert len(tvx_inputs.starttime) == len(admission_inputs_df)
            assert all(tvx_inputs.code_index < len(tvx_ehr_with_icu_inputs.dataset.scheme.icu_inputs))

    @pytest.fixture
    def tvx_ehr_with_obs(self, large_ehr: TVxEHR) -> TVxEHR:
        if len(large_ehr.dataset.tables.obs) == 0:
            pytest.skip("No observations table found in dataset.")
        return large_ehr



    @pytest.fixture
    def admission_obs(self, tvx_ehr_with_obs: TVxEHR) -> Dict[str, InpatientObservables]:
        return TVxConcepts._observables(tvx_ehr_with_obs, TVxReport())[0]

    def test_admission_obs(self, tvx_ehr_with_obs: TVxEHR, admission_obs: Dict[str, InpatientObservables]):
        obs_df = tvx_ehr_with_obs.dataset.tables.obs
        c_admission_id = tvx_ehr_with_obs.dataset.config.tables.admissions.admission_id_alias
        assert len(admission_obs) == len(tvx_ehr_with_obs.dataset.tables.admissions)
        assert sum(obs.mask.sum() for obs in admission_obs.values()) == len(obs_df)

        for admission_id, admission_obs_df in obs_df.groupby(c_admission_id):
            tvx_obs = admission_obs[admission_id]
            assert tvx_obs.mask.sum() == len(admission_obs_df)
            assert tvx_obs.value.shape[1] == len(tvx_ehr_with_obs.dataset.scheme.obs)

    @pytest.fixture
    def tvx_ehr_with_hosp_procedures(self, large_ehr: TVxEHR) -> TVxEHR:
        if len(large_ehr.dataset.tables.hosp_procedures) == 0:
            pytest.skip("No hospital procedures table found in dataset.")
        return large_ehr

    @pytest.fixture
    def admission_hosp_procedures(self, tvx_ehr_with_hosp_procedures: TVxEHR) -> Dict[str, InpatientInput]:
        return TVxConcepts._hosp_procedures(tvx_ehr_with_hosp_procedures)

    def test_admission_hosp_procedures(self, tvx_ehr_with_hosp_procedures: TVxEHR,
                                  admission_hosp_procedures: Dict[str, InpatientInput]):
        hosp_procedures = tvx_ehr_with_hosp_procedures.dataset.tables.hosp_procedures
        c_admission_id = tvx_ehr_with_hosp_procedures.dataset.config.tables.admissions.admission_id_alias
        assert len(admission_hosp_procedures) == len(tvx_ehr_with_hosp_procedures.dataset.tables.admissions)
        assert sum(len(proc.starttime) for proc in admission_hosp_procedures.values()) == len(hosp_procedures)

        for admission_id, admission_hosp_procedures_df in hosp_procedures.groupby(c_admission_id):
            tvx_hosp_proc = admission_hosp_procedures[admission_id]
            assert len(tvx_hosp_proc.starttime) == len(admission_hosp_procedures_df)
            assert all(tvx_hosp_proc.code_index < len(tvx_ehr_with_hosp_procedures.dataset.scheme.hosp_procedures))

    @pytest.fixture
    def tvx_ehr_with_icu_procedures(self, large_ehr: TVxEHR) -> TVxEHR:
        if len(large_ehr.dataset.tables.icu_procedures) == 0:
            pytest.skip("No icu procedures table found in dataset.")
        return large_ehr

    @pytest.fixture
    def admission_icu_procedures(self, tvx_ehr_with_icu_procedures: TVxEHR) -> Dict[str, InpatientInput]:
        return TVxConcepts._icu_procedures(tvx_ehr_with_icu_procedures)

    def test_admission_icu_procedures(self, tvx_ehr_with_icu_procedures: TVxEHR,
                                       admission_icu_procedures: Dict[str, InpatientInput]):
        icu_procedures = tvx_ehr_with_icu_procedures.dataset.tables.icu_procedures
        c_admission_id = tvx_ehr_with_icu_procedures.dataset.config.tables.admissions.admission_id_alias
        assert len(admission_icu_procedures) == len(tvx_ehr_with_icu_procedures.dataset.tables.admissions)
        assert sum(len(proc.starttime) for proc in admission_icu_procedures.values()) == len(icu_procedures)

        for admission_id, admission_icu_procedures_df in icu_procedures.groupby(c_admission_id):
            tvx_icu_proc = admission_icu_procedures[admission_id]
            assert len(tvx_icu_proc.starttime) == len(admission_icu_procedures_df)
            assert all(tvx_icu_proc.code_index < len(tvx_ehr_with_icu_procedures.dataset.scheme.icu_procedures))


    @pytest.fixture(params=['interventions', 'observables'])
    def tvx_ehr_conf_concept(self, large_ehr: TVxEHR, request) -> TVxEHR:
        concept_name = request.param
        conf = large_ehr.config
        for cname in ('interventions', 'observables'):
            conf = eqx.tree_at(lambda x: getattr(x, cname), conf,
                               concept_name == cname)
        return eqx.tree_at(lambda x: x.config, large_ehr, conf)

    @pytest.fixture
    def tvx_ehr_concept(self, tvx_ehr_conf_concept: TVxEHR):
        return tvx_ehr_conf_concept.execute_external_transformations([TVxConcepts()])[0]
