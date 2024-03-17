from typing import Type, List, Dict

import equinox as eqx
import numpy as np
import pytest

from lib.ehr import Dataset, CodesVector, InpatientInput, InpatientObservables, DemographicVectorConfig, StaticInfo, \
    InpatientInterventions
from lib.ehr.tvx_concepts import SegmentedPatient, Patient, SegmentedAdmission, Admission, \
    SegmentedInpatientInterventions, LeadingObservableExtractorConfig
from lib.ehr.tvx_ehr import TrainableTransformation, TVxEHR, TVxReport, TVxEHRSampleConfig, ScalerConfig, \
    DatasetNumericalProcessorsConfig, ScalersConfig, OutlierRemoversConfig, IQROutlierRemoverConfig, \
    DatasetNumericalProcessors, SegmentedTVxEHR
from lib.ehr.tvx_transformations import SampleSubjects, CodedValueScaler, ObsAdaptiveScaler, InputScaler, \
    ObsIQROutlierRemover, TVxConcepts, InterventionSegmentation, ObsTimeBinning, LeadingObservableExtraction, \
    ExcludeShortAdmissions
from test.ehr.conftest import BINARY_OBSERVATION_CODE_INDEX, MAX_STAY_DAYS


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


def test_serialization_multi_subjects(multi_subjects_ehr: TVxEHR, tmpdir: str):
    path = f'{tmpdir}/multi_subjects_ehr'
    multi_subjects_ehr.save(path)
    loaded_ehr = TVxEHR.load(path)
    assert multi_subjects_ehr.equals(loaded_ehr)


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

    def test_ehr_serialization(self, multi_subjects_ehr: TVxEHR, sampled_tvx_ehr: TVxEHR, tmpdir: str):
        path1 = f'{tmpdir}/multi_subjects_ehr'
        path2 = f'{tmpdir}/sampled_tvx_ehr'

        multi_subjects_ehr.save(path1)
        loaded_multi_subjects_ehr = TVxEHR.load(path1)

        sampled_tvx_ehr.save(path2)
        loaded_sampled_tvx_ehr = TVxEHR.load(path2)

        assert not multi_subjects_ehr.equals(sampled_tvx_ehr)
        assert not multi_subjects_ehr.equals(loaded_sampled_tvx_ehr)
        assert not loaded_multi_subjects_ehr.equals(sampled_tvx_ehr)
        assert not loaded_multi_subjects_ehr.equals(loaded_sampled_tvx_ehr)
        assert sampled_tvx_ehr.equals(loaded_sampled_tvx_ehr)


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
    def large_scalable_split_ehr(self, large_ehr: TVxEHR, scalable_table_name: str, use_float16: bool):
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
    def scaled_ehr(self, large_scalable_split_ehr, scaler_class: Type[TrainableTransformation]):
        return scaler_class.apply(large_scalable_split_ehr, TVxReport())[0]

    def test_trainable_transformer(self, large_scalable_split_ehr: TVxEHR,
                                   scaled_ehr: TVxEHR,
                                   scalable_table_name: str,
                                   use_float16: bool):
        assert getattr(large_scalable_split_ehr.numerical_processors.scalers, scalable_table_name) is None
        scaler = getattr(scaled_ehr.numerical_processors.scalers, scalable_table_name)
        assert isinstance(scaler, CodedValueScaler)
        assert scaler.table_getter(scaled_ehr.dataset) is getattr(scaled_ehr.dataset.tables, scalable_table_name)
        assert scaler.table_getter(large_scalable_split_ehr.dataset) is getattr(
            large_scalable_split_ehr.dataset.tables,
            scalable_table_name)
        assert scaler.config.use_float16 == use_float16

        table0 = scaler.table_getter(large_scalable_split_ehr.dataset)
        table1 = scaler.table_getter(scaled_ehr.dataset)
        assert scaler.value_column in table1.columns
        assert scaler.code_column in table1.columns
        assert table1 is not table0
        if use_float16:
            assert table1[scaler.value_column].dtype == np.float16
        else:
            assert table1[scaler.value_column].dtype == table0[scaler.value_column].dtype

    @pytest.fixture
    def processed_ehr(self, large_scalable_split_ehr: TVxEHR,
                      numerical_processor_config: DatasetNumericalProcessorsConfig) -> TVxEHR:
        large_scalable_split_ehr = eqx.tree_at(lambda x: x.config.numerical_processors, large_scalable_split_ehr,
                                               numerical_processor_config)
        return large_scalable_split_ehr.execute_external_transformations(
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

    def test_ehr_serialization(self, large_ehr: TVxEHR, large_scalable_split_ehr: TVxEHR, processed_ehr: TVxEHR,
                               tmpdir: str):
        assert not large_ehr.equals(large_scalable_split_ehr)
        assert not large_ehr.equals(processed_ehr)
        assert not large_scalable_split_ehr.equals(processed_ehr)

        split_path = f'{tmpdir}/split_ehr'
        processed_path = f'{tmpdir}/processed_ehr'
        large_scalable_split_ehr.save(split_path)
        loaded_split_ehr = TVxEHR.load(split_path)
        assert large_scalable_split_ehr.equals(loaded_split_ehr)

        processed_ehr.save(processed_path)
        loaded_processed_ehr = TVxEHR.load(processed_path)
        assert processed_ehr.equals(loaded_processed_ehr)


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
        assert set(admission_dx_codes.keys()) == set(tvx_ehr_with_dx.admission_ids)
        for admission_id, codes in admission_dx_codes.items():
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
        assert set(admission_dx_history_codes.keys()) == set(tvx_ehr_with_dx.admission_ids)
        assert set(admission_dx_history_codes.keys()) == set(admission_dx_codes.keys())
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
        assert set(admission_outcome.keys()) == set(admission_dx_codes.keys())
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
        assert set(admission_icu_inputs.keys()).issubset(set(tvx_ehr_with_icu_inputs.admission_ids))
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
        assert set(admission_obs.keys()) == set(tvx_ehr_with_obs.admission_ids)
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
        assert set(admission_hosp_procedures.keys()).issubset(set(tvx_ehr_with_hosp_procedures.admission_ids))
        assert sum(len(proc.starttime) for proc in admission_hosp_procedures.values() if proc is not None) == len(
            hosp_procedures)

        for admission_id, admission_hosp_procedures_df in hosp_procedures.groupby(c_admission_id):
            tvx_hosp_proc = admission_hosp_procedures[admission_id]
            assert len(tvx_hosp_proc.starttime) == len(admission_hosp_procedures_df)
            assert all(tvx_hosp_proc.code_index < len(tvx_ehr_with_hosp_procedures.scheme.hosp_procedures))

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
        assert set(admission_icu_procedures.keys()).issubset(set(tvx_ehr_with_icu_procedures.admission_ids))
        assert sum(len(proc.starttime) for proc in admission_icu_procedures.values() if proc is not None) == len(
            icu_procedures)

        for admission_id, admission_icu_procedures_df in icu_procedures.groupby(c_admission_id):
            tvx_icu_proc = admission_icu_procedures[admission_id]
            assert len(tvx_icu_proc.starttime) == len(admission_icu_procedures_df)
            assert all(tvx_icu_proc.code_index < len(tvx_ehr_with_icu_procedures.scheme.icu_procedures))

    @pytest.fixture
    def tvx_ehr_with_all_interventions(self, large_ehr: TVxEHR) -> TVxEHR:
        if len(large_ehr.dataset.tables.icu_procedures) == 0:
            pytest.skip("No icu procedures table found in dataset.")
        if len(large_ehr.dataset.tables.hosp_procedures) == 0:
            pytest.skip("No hospital procedures table found in dataset.")
        if len(large_ehr.dataset.tables.icu_inputs) == 0:
            pytest.skip("No icu inputs table found in dataset.")
        return large_ehr

    @pytest.fixture
    def admission_interventions(self, tvx_ehr_with_all_interventions: TVxEHR) -> Dict[str, InpatientInterventions]:
        return TVxConcepts._interventions(tvx_ehr_with_all_interventions, TVxReport())[0]

    def test_admission_interventions(self, tvx_ehr_with_all_interventions: TVxEHR,
                                     admission_interventions: Dict[str, InpatientInterventions]):
        assert set(admission_interventions.keys()) == set(tvx_ehr_with_all_interventions.admission_ids)
        for attr in ('icu_procedures', 'hosp_procedures', 'icu_inputs'):
            table = getattr(tvx_ehr_with_all_interventions.dataset.tables, attr)
            interventions = {admission_id: getattr(v, attr) for admission_id, v in admission_interventions.items()}
            assert sum(len(v.starttime) for v in interventions.values() if v is not None) == len(table)

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
        return tvx_ehr_conf_concept.execute_external_transformations([TVxConcepts()])

    def test_tvx_ehr_concept(self, tvx_ehr_concept: TVxEHR):
        assert set(tvx_ehr_concept.subjects.keys()) == set(tvx_ehr_concept.dataset.tables.static.index)
        if tvx_ehr_concept.config.interventions:
            assert all(adm.interventions is not None for patient in tvx_ehr_concept.subjects.values() for adm in
                       patient.admissions)
        else:
            assert all(adm.interventions is None for patient in tvx_ehr_concept.subjects.values() for adm in
                       patient.admissions)
        if tvx_ehr_concept.config.observables:
            assert all(adm.observables is not None for patient in tvx_ehr_concept.subjects.values() for adm in
                       patient.admissions)
        else:
            assert all(
                adm.observables is None for patient in tvx_ehr_concept.subjects.values() for adm in patient.admissions)

        for subject_id, sorted_admission_ids in tvx_ehr_concept.subjects_sorted_admission_ids.items():
            assert [adm.admission_id for adm in tvx_ehr_concept.subjects[subject_id].admissions] == sorted_admission_ids

    @pytest.fixture
    def mutated_ehr_concept(self, tvx_ehr_concept: TVxEHR, tvx_ehr_conf_concept: TVxEHR):
        if tvx_ehr_conf_concept.config.interventions:
            s0 = tvx_ehr_concept.subject_ids[0]
            a0 = tvx_ehr_concept.subjects[s0].admissions[0]
            a1 = eqx.tree_at(lambda x: x.interventions.icu_inputs.rate, a0,
                             a0.interventions.icu_inputs.rate + 0.1)
            return eqx.tree_at(lambda x: x.subjects[s0].admissions[0], tvx_ehr_concept, a1)
        else:
            s0 = tvx_ehr_concept.subject_ids[0]
            a0 = tvx_ehr_concept.subjects[s0].admissions[0]
            a1 = eqx.tree_at(lambda x: x.observables.value, a0, a0.observables.value + 0.1)
            return eqx.tree_at(lambda x: x.subjects[s0].admissions[0], tvx_ehr_concept, a1)

    def test_ehr_serialization(self, tvx_ehr_conf_concept: TVxEHR, tvx_ehr_concept: TVxEHR,
                               mutated_ehr_concept: TVxEHR,
                               tmpdir: str):
        ehr_list = [tvx_ehr_conf_concept, tvx_ehr_concept, mutated_ehr_concept]
        for i, ehr_i in enumerate(ehr_list):
            path = f'{tmpdir}/tvx_ehr_{i}'
            ehr_i.save(path)
            loaded_ehr_i = TVxEHR.load(path)
            for j in range(i + 1, len(ehr_list)):
                ehr_j = ehr_list[j]
                assert loaded_ehr_i.equals(ehr_j) == (i == j)


class TestInterventionSegmentation:
    @pytest.fixture
    def tvx_ehr_concept(self, large_ehr: TVxEHR):
        large_ehr = eqx.tree_at(lambda x: x.config.interventions, large_ehr, True)
        large_ehr = eqx.tree_at(lambda x: x.config.observables, large_ehr, True)
        return large_ehr.execute_external_transformations([TVxConcepts()])

    @pytest.fixture
    def tvx_ehr_segmented(self, tvx_ehr_concept: TVxEHR) -> SegmentedTVxEHR:
        return tvx_ehr_concept.execute_external_transformations([InterventionSegmentation()])

    def test_segmentation(self, tvx_ehr_concept: TVxEHR, tvx_ehr_segmented: SegmentedTVxEHR):
        first_patient = next(iter(tvx_ehr_concept.subjects.values()))
        first_segmented_patient = next(iter(tvx_ehr_segmented.subjects.values()))
        assert isinstance(first_patient, Patient)
        assert isinstance(first_segmented_patient, SegmentedPatient)
        assert isinstance(first_patient.admissions[0], Admission)
        assert isinstance(first_segmented_patient.admissions[0], SegmentedAdmission)
        assert isinstance(first_patient.admissions[0].observables, InpatientObservables)
        assert isinstance(first_segmented_patient.admissions[0].observables, list)
        assert isinstance(first_segmented_patient.admissions[0].observables[0], InpatientObservables)
        assert isinstance(first_patient.admissions[0].interventions, InpatientInterventions)
        assert isinstance(first_segmented_patient.admissions[0].interventions, SegmentedInpatientInterventions)

    @pytest.fixture(
        params=['mutate_obs', 'intervention_time', 'mutate_icu_proc', 'mutate_hosp_proc', 'mutate_icu_input'])
    def mutated_ehr_concept(self, tvx_ehr_segmented: SegmentedTVxEHR, request):
        s0 = tvx_ehr_segmented.subject_ids[0]
        a0 = tvx_ehr_segmented.subjects[s0].admissions[0]
        if request.param == 'mutate_obs':
            a1 = eqx.tree_at(lambda x: x.observables[0].time, a0, a0.observables[0].time + 0.001)
        elif request.param == 'intervention_time':
            a1 = eqx.tree_at(lambda x: x.interventions.time, a0, a0.interventions.time + 1e-6)
        elif request.param == 'mutate_icu_proc':
            if a0.interventions.icu_procedures is None:
                pytest.skip("No icu procedures in admission.")
            a1 = eqx.tree_at(lambda x: x.interventions.icu_procedures, a0,
                             ~a0.interventions.icu_procedures[0])

        elif request.param == 'mutate_hosp_proc':
            if a0.interventions.hosp_procedures is None:
                pytest.skip("No hospital procedures in admission.")
            a1 = eqx.tree_at(lambda x: x.interventions.hosp_procedures, a0,
                             ~a0.interventions.hosp_procedures[0])
        elif request.param == 'mutate_icu_input':
            if a0.interventions.icu_inputs is None:
                pytest.skip("No icu inputs in admission.")
            a1 = eqx.tree_at(lambda x: x.interventions.icu_inputs, a0, a0.interventions.icu_inputs[0] + 0.1)
        else:
            raise ValueError(f"Invalid param: {request.param}")

        return eqx.tree_at(lambda x: x.subjects[s0].admissions[0], tvx_ehr_segmented, a1)

    def test_ehr_serialization(self, tvx_ehr_concept: TVxEHR, tvx_ehr_segmented: SegmentedTVxEHR,
                               mutated_ehr_concept: SegmentedTVxEHR,
                               tmpdir: str):
        ehr_list = [tvx_ehr_concept, tvx_ehr_segmented, mutated_ehr_concept]
        for i, ehr_i in enumerate(ehr_list):
            path = f'{tmpdir}/tvx_ehr_{i}'
            ehr_i.save(path)
            loaded_ehr_i = TVxEHR.load(path)
            for j in range(i + 1, len(ehr_list)):
                ehr_j = ehr_list[j]
                assert loaded_ehr_i.equals(ehr_j) == (i == j)


class TestObsTimeBinning:
    @pytest.fixture
    def tvx_ehr_concept(self, large_ehr: TVxEHR):
        large_ehr = eqx.tree_at(lambda x: x.config.interventions, large_ehr, False)
        large_ehr = eqx.tree_at(lambda x: x.config.observables, large_ehr, True)
        return large_ehr.execute_external_transformations([TVxConcepts()])

    @pytest.fixture
    def tvx_ehr_binned(self, tvx_ehr_concept: TVxEHR) -> TVxEHR:
        tvx_ehr_concept = eqx.tree_at(lambda x: x.config.time_binning, tvx_ehr_concept, 12.0,
                                      is_leaf=lambda x: x is None)
        return tvx_ehr_concept.execute_external_transformations([ObsTimeBinning()])

    def test_binning(self, tvx_ehr_concept: TVxEHR, tvx_ehr_binned: TVxEHR):
        assert all((np.diff(o.time) == tvx_ehr_binned.config.time_binning).all() for o in tvx_ehr_binned.iter_obs() if
                   len(o) > 1)


class TestLeadExtraction:
    @pytest.fixture
    def tvx_ehr_concept(self, large_ehr: TVxEHR):
        large_ehr = eqx.tree_at(lambda x: x.config.interventions, large_ehr, False)
        large_ehr = eqx.tree_at(lambda x: x.config.observables, large_ehr, True)
        obs_scheme = large_ehr.scheme.obs
        lead_config = LeadingObservableExtractorConfig(leading_hours=[1.0, 2.0],
                                                       scheme=obs_scheme.name,
                                                       entry_neglect_window=0.0,
                                                       recovery_window=0.0,
                                                       minimum_acquisitions=0,
                                                       observable_code=obs_scheme.index2code[
                                                           BINARY_OBSERVATION_CODE_INDEX])
        large_ehr = eqx.tree_at(lambda x: x.config.leading_observable, large_ehr, lead_config,
                                is_leaf=lambda x: x is None)
        return large_ehr.execute_external_transformations([TVxConcepts()])

    @pytest.fixture
    def tvx_ehr_lead(self, tvx_ehr_concept: TVxEHR) -> TVxEHR:

        return tvx_ehr_concept.execute_external_transformations([LeadingObservableExtraction()])

    def test_lead(self, tvx_ehr_concept: TVxEHR, tvx_ehr_lead: TVxEHR):
        first_patient0 = next(iter(tvx_ehr_concept.subjects.values()))
        first_patient1 = next(iter(tvx_ehr_lead.subjects.values()))
        assert isinstance(first_patient0, Patient)
        assert isinstance(first_patient1, Patient)
        assert isinstance(first_patient0.admissions[0], Admission)
        assert isinstance(first_patient1.admissions[0], Admission)
        assert isinstance(first_patient0.admissions[0].observables, InpatientObservables)
        assert isinstance(first_patient1.admissions[0].observables, InpatientObservables)
        assert first_patient0.admissions[0].leading_observable is None
        assert isinstance(first_patient1.admissions[0].leading_observable, InpatientObservables)

        for subject_id, patient in tvx_ehr_lead.subjects.items():
            for admission in patient.admissions:
                assert admission.leading_observable is not None
                assert len(admission.leading_observable) == len(admission.observables)
                assert admission.leading_observable.value.shape[1] == len(
                    tvx_ehr_lead.config.leading_observable.leading_hours)

    def test_ehr_serialization(self, tvx_ehr_concept: TVxEHR, tvx_ehr_lead: TVxEHR,
                               tmpdir: str):
        ehr_list = [tvx_ehr_concept, tvx_ehr_lead]
        for i, ehr_i in enumerate(ehr_list):
            path = f'{tmpdir}/tvx_ehr_{i}'
            ehr_i.save(path)
            loaded_ehr_i = TVxEHR.load(path)
            for j in range(i + 1, len(ehr_list)):
                ehr_j = ehr_list[j]
                assert loaded_ehr_i.equals(ehr_j) == (i == j)


class TestExcludeShortAdmissions:
    @pytest.fixture
    def tvx_ehr_concept(self, large_ehr: TVxEHR):
        large_ehr = eqx.tree_at(lambda x: x.config.interventions, large_ehr, False)
        large_ehr = eqx.tree_at(lambda x: x.config.observables, large_ehr, False)
        large_ehr = eqx.tree_at(lambda x: x.config.admission_minimum_los, large_ehr, MAX_STAY_DAYS * 24 / 2,
                                is_leaf=lambda x: x is None)
        return large_ehr.execute_external_transformations([TVxConcepts()])

    @pytest.fixture
    def tvx_ehr_filtered(self, tvx_ehr_concept: TVxEHR) -> TVxEHR:
        return tvx_ehr_concept.execute_external_transformations([ExcludeShortAdmissions()])

    def test_filter(self, tvx_ehr_concept: TVxEHR, tvx_ehr_filtered: TVxEHR):
        assert sum(len(s.admissions) for s in tvx_ehr_concept.subjects.values()) > sum(
            len(s.admissions) for s in tvx_ehr_filtered.subjects.values())
        assert all(adm.interval_hours >= tvx_ehr_concept.config.admission_minimum_los for s in
                   tvx_ehr_filtered.subjects.values() for adm in s.admissions)
