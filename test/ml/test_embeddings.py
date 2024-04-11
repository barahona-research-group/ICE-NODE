import pytest

from lib.ehr.coding_scheme import GroupingData


@pytest.fixture
def grouping_data(dataset_scheme_manager, tvx_ehr_scheme_config, dataset_scheme_config) -> GroupingData:
    icu_inputs_map = dataset_scheme_manager.map[(dataset_scheme_config.icu_inputs, tvx_ehr_scheme_config.icu_inputs)]
    return icu_inputs_map.grouping_data


# def test_grouping_data(grouping_data: GroupingData, tvx_ehr_scheme_config, dataset_scheme_config,
#                        dataset_scheme_manager):
#     print('xxx')
#     assert len(grouping_data.aggregation) == len(dataset_scheme_manager.scheme[tvx_ehr_scheme_config.icu_inputs])
