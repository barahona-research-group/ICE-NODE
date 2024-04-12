import pytest

from lib.ehr.coding_scheme import GroupingData


@pytest.fixture
def grouping_data(dataset_scheme_manager, tvx_ehr_scheme_config, dataset_scheme_config) -> GroupingData:
    icu_inputs_map = dataset_scheme_manager.map[(dataset_scheme_config.icu_inputs, tvx_ehr_scheme_config.icu_inputs)]
    return icu_inputs_map.grouping_data


def test_grouping_data(grouping_data: GroupingData,
                       tvx_ehr_scheme_config,
                       dataset_scheme_config,
                       dataset_scheme_manager):
    m = dataset_scheme_manager.map[(dataset_scheme_config.icu_inputs, tvx_ehr_scheme_config.icu_inputs)]
    M = len(dataset_scheme_manager.scheme[dataset_scheme_config.icu_inputs])
    N = len(dataset_scheme_manager.scheme[tvx_ehr_scheme_config.icu_inputs])
    N_ = len(m.reduced_groups)
    assert grouping_data.scheme_size.tolist() == [M, N]
    assert len(grouping_data.permute) == M
    assert len(grouping_data.size) == N_
    assert len(grouping_data.split) == N_
    assert grouping_data.size.sum() == M
    assert len(grouping_data.aggregation) == N_
