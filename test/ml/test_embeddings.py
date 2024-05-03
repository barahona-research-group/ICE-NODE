from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from lib.ehr.coding_scheme import GroupingData, CodingScheme, ReducedCodeMapN1, CodingSchemesManager, NumericScheme
from lib.ml.embeddings import GroupEmbedding, InterventionsEmbeddingsConfig, InterventionsEmbeddings, \
    AdmissionGenericEmbeddingsConfig, AdmissionEmbedding, EmbeddedAdmission




@pytest.fixture
def icu_procedures_target_size(icu_proc_scheme):
    return len(icu_proc_scheme)


@pytest.fixture
def hosp_procedures_target_size(hosp_proc_scheme):
    return len(hosp_proc_scheme)


@pytest.fixture(params=[0, 55])
def icu_inputs_sample(icu_inputs_grouping_data, request):
    seed = request.param
    n, _ = icu_inputs_grouping_data.scheme_size
    return jrandom.uniform(jrandom.PRNGKey(seed), shape=(n,))


@pytest.fixture
def icu_procedures_sample(icu_procedures_target_size):
    return jrandom. binomial(jrandom.PRNGKey(0), n=1, p=0.5, shape=(icu_procedures_target_size,) )


@pytest.fixture
def hosp_procedures_sample(hosp_procedures_target_size):
    return jrandom. binomial(jrandom.PRNGKey(0), n=1, p=0.5, shape=(hosp_procedures_target_size,))


def test_grouping_data(icu_inputs_grouping_data: GroupingData,
                       icu_inputs_scheme: CodingScheme, icu_inputs_target_scheme: CodingScheme,
                       icu_inputs_map: ReducedCodeMapN1):
    M = len(icu_inputs_scheme)
    N = len(icu_inputs_target_scheme)
    N_ = len(icu_inputs_map.reduced_groups)
    assert icu_inputs_grouping_data.scheme_size.tolist() == [M, N]
    assert len(icu_inputs_grouping_data.permute) == M
    assert len(icu_inputs_grouping_data.size) == N_
    assert len(icu_inputs_grouping_data.split) == N_
    assert icu_inputs_grouping_data.size.sum() == M
    assert len(icu_inputs_grouping_data.aggregation) == N_


class TestGroupEmbedding:
    @pytest.fixture(params=[1, 5, 100])
    def embeddings_size(self, request):
        return request.param

    @pytest.fixture(params=[0, 55])
    def icu_inputs_embedding(self, icu_inputs_grouping_data, request, embeddings_size):
        seed = request.param
        return GroupEmbedding(icu_inputs_grouping_data, embeddings_size, jrandom.PRNGKey(seed))

    @pytest.fixture
    def group_indexes(self, icu_inputs_scheme_manager: CodingSchemesManager,
                      icu_inputs_scheme: CodingScheme, icu_inputs_target_scheme: CodingScheme,
                      icu_inputs_map: ReducedCodeMapN1) -> Tuple[
        Tuple[int, ...], ...]:
        return tuple(tuple(map(icu_inputs_scheme.index.get, group)) for group in
                     icu_inputs_map.groups(icu_inputs_scheme_manager))

    @pytest.fixture
    def split_source_array(self, icu_inputs_sample, icu_inputs_embedding):
        return icu_inputs_embedding.split_source_array(icu_inputs_sample)

    @pytest.mark.usefixtures('jax_cpu_execution')
    @pytest.mark.serial_test
    def test_split_source_array(self, icu_inputs_sample, group_indexes, split_source_array):
        for array, indexes in zip(split_source_array, group_indexes):
            assert array.shape == (len(indexes),)
            assert jnp.array_equal(array, icu_inputs_sample[jnp.array(indexes)])

    @pytest.mark.usefixtures('jax_cpu_execution')
    @pytest.mark.serial_test
    def test_apply(self, icu_inputs_sample, embeddings_size, icu_inputs_embedding):
        y = icu_inputs_embedding(icu_inputs_sample)
        assert y.shape == (embeddings_size,)


class TestInterventionsEmbeddings:
    @pytest.fixture
    def interventions_embeddings_config(self):
        return InterventionsEmbeddingsConfig(icu_inputs=10, icu_procedures=5, hosp_procedures=10, interventions=25)

    @pytest.fixture
    def interventions_embedding(self, icu_inputs_grouping_data, interventions_embeddings_config,
                                icu_procedures_target_size,
                                hosp_procedures_target_size):
        return InterventionsEmbeddings(config=interventions_embeddings_config,
                                       icu_inputs_grouping=icu_inputs_grouping_data,
                                       icu_procedures_size=icu_procedures_target_size,
                                       hosp_procedures_size=hosp_procedures_target_size,
                                       key=jrandom.PRNGKey(0))

    @pytest.mark.usefixtures('jax_cpu_execution')
    @pytest.mark.serial_test
    def test_apply(self, interventions_embedding, icu_inputs_sample, icu_procedures_sample, hosp_procedures_sample,
                   interventions_embeddings_config):
        v = interventions_embedding(icu_inputs_sample, icu_procedures_sample, hosp_procedures_sample)
        assert v.shape == (interventions_embeddings_config.interventions,)


class TestAdmissionEmbedding:
    @pytest.fixture
    def admission_embedding_config(self, inpatient_interventions) -> AdmissionGenericEmbeddingsConfig:
        interventions = InterventionsEmbeddingsConfig(
            icu_inputs=10 if inpatient_interventions.icu_inputs is not None else None,
            hosp_procedures=10 if inpatient_interventions.hosp_procedures is not None else None,
            icu_procedures=5 if inpatient_interventions.icu_procedures is not None else None,
            interventions=5)

        return AdmissionGenericEmbeddingsConfig(dx_codes=10,
                                                interventions=interventions,
                                                demographic=0,
                                                observables=13)

    @pytest.fixture
    def observables_size(self, observation_scheme: NumericScheme) -> int:
        return len(observation_scheme)

    @pytest.fixture
    def dx_codes_size(self, dx_scheme: CodingScheme) -> int:
        return len(dx_scheme)

    @pytest.fixture
    def admission_embedding(self, admission_embedding_config, dx_codes_size, icu_inputs_grouping_data,
                            icu_procedures_target_size,
                            hosp_procedures_target_size,
                            observables_size) -> AdmissionEmbedding:
        return AdmissionEmbedding(config=admission_embedding_config, dx_codes_size=dx_codes_size,
                                  icu_inputs_grouping=icu_inputs_grouping_data,
                                  icu_procedures_size=icu_procedures_target_size,
                                  hosp_procedures_size=hosp_procedures_target_size,
                                  demographic_size=0,
                                  observables_size=observables_size,
                                  key=jrandom.PRNGKey(0))

    @pytest.fixture
    def embedded_admission(self, admission_embedding, segmented_admission) -> EmbeddedAdmission:
        return admission_embedding(segmented_admission, None)

    @pytest.mark.usefixtures('jax_cpu_execution')
    @pytest.mark.serial_test
    def test_embedded_admission(self, embedded_admission, segmented_admission, admission_embedding_config):

        if embedded_admission.dx_codes is not None or admission_embedding_config.dx_codes:
            assert embedded_admission.dx_codes.shape == (admission_embedding_config.dx_codes,)
        if embedded_admission.dx_codes_history is not None or admission_embedding_config.dx_codes:
            assert embedded_admission.dx_codes_history.shape == (admission_embedding_config.dx_codes,)
        if embedded_admission.interventions is not None or admission_embedding_config.interventions:
            assert embedded_admission.interventions.ndim == 2
            assert embedded_admission.interventions.shape[1] == admission_embedding_config.interventions.interventions
        if embedded_admission.demographic is not None or admission_embedding_config.demographic:
            assert embedded_admission.demographic.shape == (admission_embedding_config.demographic,)
