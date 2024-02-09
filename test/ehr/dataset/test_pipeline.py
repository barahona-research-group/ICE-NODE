from typing import List
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from lib.ehr.pipeline import DatasetTransformation, DatasetPipeline, SampleSubjects, CastTimestamps, \
    FilterUnsupportedCodes, SetAdmissionRelativeTimes, SetCodeIntegerIndices, SetIndex, ProcessOverlappingAdmissions, \
    FilterClampTimestampsToAdmissionInterval, FilterInvalidInputRatesSubjects, ICUInputRateUnitConversion, \
    ObsIQROutlierRemover, ObsAdaptiveScaler, InputScaler


class TestDatasetTransformation:

    @pytest.mark.parametrize('cls, params', [
        (SampleSubjects, {'n_subjects': 10, 'seed': 0, 'offset': 5}),
        (CastTimestamps, {}),
        (FilterUnsupportedCodes, {}),
        (SetAdmissionRelativeTimes, {}),
        (SetCodeIntegerIndices, {}),
        (SetIndex, {}),
        (ProcessOverlappingAdmissions, {'merge': False}),
        (FilterClampTimestampsToAdmissionInterval, {}),
        (FilterInvalidInputRatesSubjects, {}),
        (ICUInputRateUnitConversion, {'conversion_table': pd.DataFrame()}),
        (ObsIQROutlierRemover, {'outlier_q1': 0.0, 'outlier_q2': 0.0,
                                'outlier_iqr_scale': 0.0, 'outlier_z1': 0.0,
                                'outlier_z2': 0.0, 'transformer_key': 'key'})])
    def test_additional_parameters(self, cls, params):
        # Test that additional_parameters returns dict without name
        assert cls(name='test', **params).additional_parameters == params

    def test_report(self):
        # Test that report adds item to aux['report']
        transformation = DatasetTransformation()
        aux = {}
        transformation.report(aux, foo='bar')
        self.assertEqual(len(aux['report']), 1)
        self.assertEqual(aux['report'][0].foo, 'bar')

    def test_synchronize_index(self):
        # Test synchronize_index drops rows and updates count
        transformation = DatasetTransformation()
        dataset = MockDataset()
        indexed_table = MockTable()
        indexed_table.columns = ['id', 'val']
        indexed_table.index = [1, 2, 3]
        target_table = MockTable()
        target_table.columns = ['id', 'val', 'index']
        target_table.index = [1, 2, 3, 4]
        dataset.tables = MockTables({'indexed': indexed_table,
                                     'target': target_table})
        aux = {}

        dataset, aux = transformation.synchronize_index(dataset, 'indexed', 'id', aux)

        self.assertEqual(len(target_table), 3)
        self.assertEqual(aux['report'][0].before, 4)
        self.assertEqual(aux['report'][0].after, 3)

    def test_filter_no_admission_subjects(self):
        # Test filter drops rows not in admissions
        transformation = DatasetTransformation()
        dataset = MockDataset()
        static = MockTable()
        static.index = [1, 2, 3, 4]
        admissions = MockTable()
        admissions.index = [1, 2, 3]
        dataset.tables = MockTables({'static': static,
                                     'admissions': admissions})
        dataset.config.tables.static.subject_id_alias = 'index'
        aux = {}

        dataset, aux = transformation.filter_no_admission_subjects(dataset, aux)

        self.assertEqual(len(static), 3)
        self.assertEqual(aux['report'][0].before, 4)
        self.assertEqual(aux['report'][0].after, 3)


class MockDataset:
    pass


class MockTables:
    def __init__(self, tables):
        self.tables_dict = tables


class MockTable:
    pass


class TestDatasetPipeline(unittest.TestCase):

    def test_pipeline(self):
        # Create mock transformations
        transformation1 = MagicMock(spec=DatasetTransformation)
        transformation2 = MagicMock(spec=DatasetTransformation)
        transformation3 = MagicMock(spec=DatasetTransformation)

        # Create mock dataset
        dataset = MagicMock()

        # Create pipeline with the mock transformations
        pipeline = DatasetPipeline(transformations=[transformation1, transformation2, transformation3])

        # Call the pipeline
        result_dataset, result_auxiliary = pipeline(dataset)

        # Assert that the transformations were called in the correct order
        transformation1.assert_called_with(dataset, {'report': []})
        transformation2.assert_called_with(transformation1.return_value, transformation1.return_value)
        transformation3.assert_called_with(transformation2.return_value, transformation2.return_value)

        # Assert that the result dataset and auxiliary are returned
        self.assertEqual(result_dataset, transformation3.return_value)
        self.assertEqual(result_auxiliary, transformation3.return_value)

        self.fail()


class TestSampleSubjects(unittest.TestCase):

    def test_call(self):
        # Create mock dataset
        dataset = MagicMock()
        dataset.tables.static.index.names = ['subject_id']
        dataset.config.tables.static.subject_id_alias = 'subject_id'

        # Create mock auxiliary
        aux = {}

        # Create instance of SampleSubjects transformation
        transformation = SampleSubjects()
        transformation.n_subjects = 10
        transformation.seed = 123
        transformation.offset = 0

        # Call the transformation
        result_dataset, result_aux = transformation(dataset, aux)

        # Assert that the index name is correct
        self.assertIn('subject_id', dataset.tables.static.index.names)

        # Assert that the report is updated correctly
        self.assertEqual(len(aux['report']), 1)
        self.assertEqual(aux['report'][0].table, 'static')
        self.assertEqual(aux['report'][0].column, 'subject_id')
        self.assertEqual(aux['report'][0].before, len(dataset.tables.static))
        self.assertEqual(aux['report'][0].after, len(result_dataset.tables.static))
        self.assertEqual(aux['report'][0].value_type, 'count')
        self.assertEqual(aux['report'][0].operation, 'sample')

        # Assert that the dataset is updated correctly
        self.assertEqual(result_dataset, transformation.synchronize_subjects.return_value)

        self.fail()


class TestCastTimestamps(unittest.TestCase):

    def test_call_skips_table_with_no_time_cols(self):
        dataset = MagicMock()
        dataset.config.tables.table1.time_cols = []

        transformation = CastTimestamps()
        result, aux = transformation(dataset, {})

        dataset.tables.table1.assert_not_called()
        self.fail()

    def test_call_skips_column_already_datetime(self):
        dataset = MagicMock()
        table = MagicMock()
        table.columns = ['col1']
        table.col1.dtype = 'datetime64[ns]'
        dataset.config.tables.table1.time_cols = ['col1']
        dataset.tables.table1 = table

        transformation = CastTimestamps()
        result, aux = transformation(dataset, {})

        table.col1.assert_not_called()

        self.fail()

    def test_call_casts_column(self):
        dataset = MagicMock()
        table = MagicMock()
        table.columns = ['col1']
        table.col1.dtype = 'object'
        dataset.config.tables.table1.time_cols = ['col1']
        dataset.tables.table1 = table

        transformation = CastTimestamps()
        result, aux = transformation(dataset, {})

        table.col1.assert_called_with(pd.to_datetime, errors='raise')
        self.fail()


class TestFilterUnsupportedCodes(unittest.TestCase):

    def test_filter_unsupported_codes(self):
        dataset = MagicMock()
        dataset.tables.table1.code_column = 'code'
        dataset.scheme.table1.codes = ['A', 'B']
        dataset.tables.table1.code = ['A', 'C', 'B']

        aux = {}

        filter = FilterUnsupportedCodes()
        result, aux = filter(dataset, aux)

        dataset.tables.table1.assert_called_with(['A', 'B'])
        self.assertEqual(len(aux['report']), 1)
        self.assertEqual(aux['report'][0]['before'], 3)
        self.assertEqual(aux['report'][0]['after'], 2)
        self.fail()

    def test_multiple_tables(self):
        dataset = MagicMock()
        dataset.tables.table1.code_column = 'code1'
        dataset.tables.table2.code_column = 'code2'
        dataset.scheme.table1.codes = ['A', 'B']
        dataset.scheme.table2.codes = ['X', 'Y']
        dataset.tables.table1.code1 = ['A', 'C', 'B']
        dataset.tables.table2.code2 = ['X', 'Z', 'Y']

        aux = {}

        filter = FilterUnsupportedCodes()
        result, aux = filter(dataset, aux)

        dataset.tables.table1.assert_called_with(['A', 'B'])
        dataset.tables.table2.assert_called_with(['X', 'Y'])
        self.assertEqual(len(aux['report']), 2)
        self.fail()


class TestSetAdmissionRelativeTimes(unittest.TestCase):

    def test_temporal_admission_linked_table(self):
        dataset = MagicMock()
        dataset.config.tables.table1 = MagicMock(spec=TimestampedTableConfig)
        self.assertTrue(SetAdmissionRelativeTimes.temporal_admission_linked_table(dataset, 'table1'))

        dataset.config.tables.table2 = MagicMock(spec=IntervalBasedTableConfig)
        self.assertTrue(SetAdmissionRelativeTimes.temporal_admission_linked_table(dataset, 'table2'))

        dataset.config.tables.table3 = MagicMock(spec=AdmissionLinkedTableConfig)
        self.assertFalse(SetAdmissionRelativeTimes.temporal_admission_linked_table(dataset, 'table3'))
        self.fail()

    def test_call(self):
        dataset = MagicMock()
        dataset.config.tables.time_cols = {'table1': ['time1'],
                                           'table2': ['time2', 'time3']}
        dataset.config.tables.admissions.admission_time_alias = 'admittime'
        dataset.config.tables.admissions.admission_id_alias = 'admission_id'

        admissions = MagicMock()
        admissions.__getitem__.return_value = ['admittime']

        tables_dict = {'table1': MagicMock(),
                       'table2': MagicMock()}
        dataset.tables.admissions = admissions
        dataset.tables.tables_dict = tables_dict

        transformation = SetAdmissionRelativeTimes()
        result_dataset, result_aux = transformation(dataset, {})

        # Assert time columns converted
        for table_name in ['table1', 'table2']:
            table = tables_dict[table_name]
            for time_col in dataset.config.tables.time_cols[table_name]:
                table.__getitem__.assert_called_with(time_col)

        # Assert report updated
        self.assertEqual(len(result_aux['report']), 3)
        self.fail()


class TestSetCodeIntegerIndices(unittest.TestCase):

    def test_call(self):
        dataset = MagicMock()
        dataset.tables.table1.code_column = 'code'
        dataset.scheme.table1.index = [1, 2, 3]
        dataset.tables.table1['code'] = [1, 2, np.nan]

        aux = {}

        transformation = SetCodeIntegerIndices()
        result_dataset, result_aux = transformation(dataset, aux)

        dataset.tables.table1.assert_called_with('code', dataset.scheme.table1.index)
        self.assertEqual(result_dataset.tables.table1['code'].dtype, np.int64)
        self.assertEqual(len(result_dataset.tables.table1), 2)
        self.assertEqual(result_aux['report'][0]['before'], 3)
        self.assertEqual(result_aux['report'][0]['after'], 2)
        self.fail()

    def test_multiple_tables(self):
        dataset = MagicMock()
        dataset.tables.table1.code_column = 'code1'
        dataset.tables.table2.code_column = 'code2'
        dataset.scheme.table1.index = [1, 2, 3]
        dataset.scheme.table2.index = [4, 5, 6]
        dataset.tables.table1['code1'] = [1, 2, np.nan]
        dataset.tables.table2['code2'] = [4, np.nan, 6]

        aux = {}

        transformation = SetCodeIntegerIndices()
        result_dataset, result_aux = transformation(dataset, aux)

        dataset.tables.table1.assert_called_with('code1', dataset.scheme.table1.index)
        self.assertEqual(result_dataset.tables.table1['code1'].dtype, np.int64)

        dataset.tables.table2.assert_called_with('code2', dataset.scheme.table2.index)
        self.assertEqual(result_dataset.tables.table2['code2'].dtype, np.int64)

        self.assertEqual(len(result_aux['report']), 4)
        self.fail()


class TestSetIndex(unittest.TestCase):

    def test_set_index(self):
        dataset = MagicMock()
        dataset.tables.table1.index.name = 'old_index'
        dataset.config.tables.indices = {'table1': 'new_index'}

        aux = {'report': []}

        set_index = SetIndex()
        result_dataset, result_aux = set_index(dataset, aux)

        dataset.tables.table1.set_index.assert_called_with('new_index')
        self.assertEqual(result_aux['report'][0]['table'], 'table1')
        self.assertEqual(result_aux['report'][0]['column'], 'new_index')
        self.assertEqual(result_aux['report'][0]['before'], 'old_index')
        self.assertEqual(result_aux['report'][0]['after'], dataset.tables.table1.index.name)
        self.fail()

    def test_multiple_tables(self):
        dataset = MagicMock()
        dataset.tables.table1.index.name = 'old_index1'
        dataset.tables.table2.index.name = 'old_index2'
        dataset.config.tables.indices = {'table1': 'new_index1',
                                         'table2': 'new_index2'}

        aux = {'report': []}

        set_index = SetIndex()
        result_dataset, result_aux = set_index(dataset, aux)

        dataset.tables.table1.set_index.assert_called_with('new_index1')
        dataset.tables.table2.set_index.assert_called_with('new_index2')

        self.assertEqual(len(result_aux['report']), 2)
        self.assertEqual(result_aux['report'][0]['table'], 'table1')
        self.assertEqual(result_aux['report'][1]['table'], 'table2')
        self.fail()

    def test_no_index_change(self):
        dataset = MagicMock()
        dataset.tables.table1.index.name = 'current_index'
        dataset.config.tables.indices = {'table1': 'current_index'}

        aux = {'report': []}

        set_index = SetIndex()
        result_dataset, result_aux = set_index(dataset, aux)

        dataset.tables.table1.set_index.assert_not_called()
        self.assertEqual(len(result_aux['report']), 0)
        self.fail()


class TestMergeOverlappingAdmissions(unittest.TestCase):

    def test_merge_overlapping_admissions(self):
        # Test merging overlapping admissions
        admissions = MockTable()
        admissions.loc[1, 'dischtime'] = 10
        admissions.loc[2, 'dischtime'] = 15
        admissions.loc[3, 'dischtime'] = 20

        sub2sup = {2: 1, 3: 1}

        process = ProcessOverlappingAdmissions()
        dataset = MockDataset(admissions)
        aux = {}

        dataset, aux = process._merge_overlapping_admissions(dataset, aux, sub2sup)

        # Assert discharge time was extended
        self.assertEqual(admissions.loc[1, 'dischtime'], 20)

        # Assert sub admissions were removed
        self.assertEqual(len(admissions), 1)
        self.fail()

    def test_no_overlap(self):
        # Test no merge when no overlap
        admissions = MockTable()
        admissions.loc[1, 'dischtime'] = 10
        admissions.loc[2, 'dischtime'] = 15

        sub2sup = {2: 1}

        process = ProcessOverlappingAdmissions()
        dataset = MockDataset(admissions)
        aux = {}

        dataset, aux = process._merge_overlapping_admissions(dataset, aux, sub2sup)

        # Assert no change to discharge times
        self.assertEqual(admissions.loc[1, 'dischtime'], 10)
        self.assertEqual(admissions.loc[2, 'dischtime'], 15)

        # Assert no admissions removed
        self.assertEqual(len(admissions), 2)
        self.fail()

    def _generate_admission_from_pattern(self, pattern: List[str]) -> pd.DataFrame:
        if len(pattern) == 0:
            return pd.DataFrame(columns=['admittime', 'dischtime'])
        random_monotonic_positive_integers = np.random.randint(1, 50, size=len(pattern)).cumsum()
        sequence_dates = list(
            map(lambda x: pd.Timestamp.today() + pd.Timedelta(days=x), random_monotonic_positive_integers))
        event_times = dict(zip(pattern, sequence_dates))
        admittimes = {k: v for k, v in event_times.items() if k.startswith('A')}
        dischtimes = {k.replace('D', 'A'): v for k, v in event_times.items() if k.startswith('D')}
        admissions = pd.DataFrame(index=admittimes.keys())
        admissions['admittime'] = admissions.index.map(admittimes)
        admissions['dischtime'] = admissions.index.map(dischtimes)
        # shuffle the rows shuffled.
        return admissions.sample(frac=1).reset_index(drop=True)

    @parameterized.expand([
        # TODO: check https://hypothesis.readthedocs.io/en/latest/quickstart.html to automate generation of test cases.
        # Below are a list of tuples of the form (admission_pattern, expected_super_sub).
        # The admission pattern is a sequence of admission/discharge.
        # In the database it is possible to have overlapping admissions, so we merge them.
        # A1 and D1 represent the admission and discharge time of a particular admission record in the database.
        # A2 and D2 means the same and that A2 happened after A1 (and not necessarily after D1).
        (['A1', 'A2', 'A3', 'D1', 'D3', 'D2'], {'A1': ['A2', 'A3']}),
        (['A1', 'A2', 'A3', 'D1', 'D2', 'D3'], {'A1': ['A2', 'A3']}),
        (['A1', 'A2', 'A3', 'D2', 'D1', 'D3'], {'A1': ['A2', 'A3']}),
        (['A1', 'A2', 'A3', 'D2', 'D3', 'D1'], {'A1': ['A2', 'A3']}),
        (['A1', 'A2', 'A3', 'D3', 'D1', 'D2'], {'A1': ['A2', 'A3']}),
        (['A1', 'A2', 'A3', 'D3', 'D2', 'D1'], {'A1': ['A2', 'A3']}),
        ##
        (['A1', 'A2', 'D1', 'A3', 'D3', 'D2'], {'A1': ['A2', 'A3']}),
        (['A1', 'A2', 'D1', 'A3', 'D2', 'D3'], {'A1': ['A2', 'A3']}),
        ##
        (['A1', 'A2', 'D1', 'D2', 'A3', 'D3'], {'A1': ['A2']}),
        (['A1', 'A2', 'D1', 'D2'], {'A1': ['A2']}),
        ##
        (['A1', 'A2', 'D2', 'A3', 'D1', 'D3'], {'A1': ['A2', 'A3']}),
        (['A1', 'A2', 'D2', 'A3', 'D3', 'D1'], {'A1': ['A2', 'A3']}),
        (['A1', 'A2', 'D2', 'D1', 'A3', 'D3'], {'A1': ['A2']}),
        ##
        (['A1', 'D1', 'A2', 'A3', 'D2', 'D3'], {'A2': ['A3']}),
        (['A1', 'D1', 'A2', 'A3', 'D3', 'D2'], {'A2': ['A3']}),
        (['A1', 'D1', 'A2', 'D2', 'A3', 'D3'], {}),
        ##
        (['A1', 'D1'], {}),
        ([], {}),
        (['A1', 'D1', 'A2', 'A3', 'D3', 'A4', 'D2', 'D4'], {'A2': ['A3', 'A4']}),
        (['A1', 'A2', 'D2', 'D1', 'A3', 'A4', 'D3', 'D4'], {'A1': ['A2'], 'A3': ['A4']}),
    ])
    def test_overlapping_cases(self, admission_pattern, expected_super_sub):
        admissions = self._generate_admission_from_pattern(admission_pattern)
        sub2sup = ProcessOverlappingAdmissions._collect_overlaps(admissions, 'admittime', 'dischtime')
        self.assertEqual(sub2sup, expected_super_sub)

    def test_mapping_admission_ids(self):
        # Test mapping admission ids
        admissions = MockTable()
        admissions.loc[1, 'dischtime'] = 10
        admissions.loc[2, 'dischtime'] = 15
        admissions.loc[3, 'dischtime'] = 20

        sub2sup = {2: 1, 3: 1}

        process = ProcessOverlappingAdmissions()
        dataset = MockDataset(admissions)
        aux = {}

        dataset, aux = process._mapping_admission_ids(dataset, aux, sub2sup)

        # Assert sub admissions were mapped
        self.assertEqual(admissions.loc[2, 'dischtime'], 1)
        self.assertEqual(admissions.loc[3, 'dischtime'], 1)

        # Assert no admissions removed
        self.assertEqual(len(admissions), 3)
        self.fail()


class TestFilterClampTimestamps(unittest.TestCase):

    def test_filter_timestamped_tables(self):
        # Test filter drops rows outside admission interval

        # Arrange
        transformation = FilterClampTimestampsToAdmissionInterval()
        dataset = MagicMock()
        table = pd.DataFrame({'time': [1, 2, 3, 4], 'val': [10, 20, 30, 40]})
        dataset.tables.table = table
        dataset.config.tables.table.time_alias = 'time'
        admissions = pd.DataFrame({'admittime': [2], 'dischtime': [3]})
        dataset.tables.admissions = admissions

        # Act
        result, aux = transformation._filter_timestamped_tables(dataset, {})

        # Assert
        expected = pd.DataFrame({'time': [2, 3], 'val': [20, 30]})
        self.assertTrue(result.tables.table.equals(expected))
        self.assertEqual(len(aux['report']), 1)
        self.assertEqual(aux['report'][0]['before'], 4)
        self.assertEqual(aux['report'][0]['after'], 2)
        self.fail()

    def test_clamp_interval_tables(self):
        # Test clamping intervals to admission

        # Arrange
        transformation = FilterClampTimestampsToAdmissionInterval()
        dataset = MagicMock()
        table = pd.DataFrame({'start': [1, 2, 4], 'end': [3, 5, 6]})
        dataset.tables.table = table
        admissions = pd.DataFrame({'admittime': [2], 'dischtime': [4]})
        dataset.tables.admissions = admissions

        # Act
        result, aux = transformation._filter_interval_based_tables(dataset, {})

        # Assert
        expected = pd.DataFrame({'start': [2, 2, 4], 'end': [3, 4, 4]})
        self.assertTrue(result.tables.table.equals(expected))
        self.assertEqual(aux['report'][0]['before'], None)
        self.assertEqual(aux['report'][0]['after'], 2)
        self.fail()

    def test_call(self):
        # Test overall call

        # Arrange
        dataset = MagicMock()
        # Populate dataset

        # Act
        result, aux = FilterClampTimestampsToAdmissionInterval()(dataset, {})

        # Assert
        # Validate dataset and aux
        self.fail()


class TestFilterInvalidInputRatesSubjects(unittest.TestCase):

    def test_filter_subjects_with_nan_rates(self):
        dataset = MagicMock()
        dataset.tables.icu_inputs.index = [1, 2, 3]
        dataset.tables.icu_inputs['derived_normalized_amount_per_hour'] = [np.nan, 1.0, 2.0]

        dataset.tables.admissions.index = [1, 2]
        dataset.tables.admissions['admission_id'] = [1, 2]
        dataset.tables.admissions['subject_id'] = [100, 200]

        dataset.tables.static.index = [100, 200, 300]

        aux = {}

        filter_invalid = FilterInvalidInputRatesSubjects()
        dataset, aux = filter_invalid(dataset, aux)

        self.assertEqual(len(dataset.tables.static), 1)
        self.assertEqual(dataset.tables.static.index[0], 300)
        self.fail()

    def test_no_nan_rates(self):
        dataset = MagicMock()
        dataset.tables.icu_inputs['derived_normalized_amount_per_hour'] = [1.0, 2.0, 3.0]

        aux = {}

        filter_invalid = FilterInvalidInputRatesSubjects()
        dataset, aux = filter_invalid(dataset, aux)

        self.assertEqual(len(dataset.tables.static), 3)
        self.fail()


class TestICUInputRateUnitConversion(unittest.TestCase):

    def test_columns(self):
        # Validate merged dataframe contains expected columns
        transformation = ICUInputRateUnitConversion()
        dataset = MagicMock()
        dataset.tables.icu_inputs.columns = ['code', 'amount']
        dataset.config.tables.icu_inputs.code_alias = 'code'
        dataset.config.tables.icu_inputs.amount_alias = 'amount'

        df = transformation(dataset)

        expected_columns = ['code', 'amount', 'amount_per_hour', 'normalized_amount_per_hour']
        self.assertEqual(df.columns, expected_columns)

    def test_rate_calculation(self):
        # Test derived rate columns are calculated correctly
        transformation = ICUInputRateUnitConversion()
        dataset = MagicMock()
        dataset.tables.icu_inputs = MagicMock()
        dataset.tables.icu_inputs.code = ['A', 'B']
        dataset.tables.icu_inputs.amount = [10, 20]
        dataset.tables.icu_inputs.start_time = [0, 10]
        dataset.tables.icu_inputs.end_time = [10, 20]

        df = transformation(dataset)

        # Amount per hour should be amount / delta hours
        expected_rates = [1, 2]
        self.assertEqual(df.amount_per_hour.tolist(), expected_rates)
        self.fail()

    def test_dataset_update(self):
        # Validate dataset is updated with new columns
        transformation = ICUInputRateUnitConversion()
        dataset = MagicMock()
        dataset.tables.icu_inputs.columns = ['code', 'amount']

        df = transformation(dataset)

        dataset.tree_at.assert_called_with(dataset.tables.icu_inputs, df)
        self.fail()


class TestObsIQROutlierRemover(unittest.TestCase):

    def test_call(self):
        remover = ObsIQROutlierRemover()
        dataset = MagicMock()
        aux = {}

        # Call with fit_only=True
        remover.fit_only = True
        dataset, aux = remover(dataset, aux)
        self.assertEqual(aux[remover.transformer_key], remover)

        # Call with fit_only=False
        remover.fit_only = False
        dataset, aux = remover(dataset, aux)
        self.assertEqual(len(aux['report']), 1)
        self.assertEqual(aux['report'][0].table, 'obs')
        self.fail()


class TestObsAdaptiveScaler(unittest.TestCase):

    def test_call_fits_scaler(self):
        dataset = MagicMock()
        aux = {}

        scaler = ObsAdaptiveScaler()
        scaler(dataset, aux)

        self.assertTrue(scaler.fit_called)
        self.fail()

    def test_call_transforms_data(self):
        dataset = MagicMock()
        dataset.tables.obs.dtype = float

        aux = {}

        scaler = ObsAdaptiveScaler()
        scaler.fit = MagicMock()
        scaler.transform = MagicMock()

        scaler(dataset, aux)

        scaler.transform.assert_called_with(dataset)
        self.fail()

    def test_call_casts_float16(self):
        dataset = MagicMock()
        dataset.tables.obs.dtype = np.float64

        aux = {}

        scaler = ObsAdaptiveScaler(use_float16=True)
        scaler.fit = MagicMock()
        scaler.transform = MagicMock()

        scaler(dataset, aux)

        self.assertEqual(dataset.tables.obs.dtype, np.float16)
        self.fail()

    def test_call_keeps_int_dtype(self):
        dataset = MagicMock()
        dataset.tables.obs.dtype = np.int32

        aux = {}

        scaler = ObsAdaptiveScaler()
        scaler.fit = MagicMock()
        scaler.transform = MagicMock()

        scaler(dataset, aux)

        self.assertEqual(dataset.tables.obs.dtype, np.int32)
        self.fail()


class TestInputScaler(unittest.TestCase):

    def test_scaling_float16(self):
        scaler = InputScaler()
        scaler.use_float16 = True

        dataset = MagicMock()
        dataset.tables.icu_inputs.code_alias = 'code'
        dataset.tables.icu_inputs.derived_normalized_amount_per_hour = 'value'
        dataset.tables.icu_inputs.value = pd.Series([1, 2, 3], dtype=np.float32)

        aux = {}

        dataset, aux = scaler(dataset, aux)

        self.assertEqual(dataset.tables.icu_inputs.value.dtype, np.float16)
        self.fail()

    def test_scaling_no_float16(self):
        scaler = InputScaler()
        scaler.use_float16 = False

        dataset = MagicMock()
        dataset.tables.icu_inputs.code_alias = 'code'
        dataset.tables.icu_inputs.derived_normalized_amount_per_hour = 'value'
        dataset.tables.icu_inputs.value = pd.Series([1, 2, 3], dtype=np.float32)

        aux = {}

        dataset, aux = scaler(dataset, aux)

        self.assertEqual(dataset.tables.icu_inputs.value.dtype, np.float32)
        self.fail()

    def test_fit(self):
        scaler = InputScaler()

        dataset = MagicMock()
        dataset.config.tables.icu_inputs.code_alias = 'code'
        dataset.config.tables.icu_inputs.derived_normalized_amount_per_hour = 'value'
        dataset.tables.icu_inputs.code = ['a', 'b', 'c']
        dataset.tables.icu_inputs.value = [1, 2, 3]

        aux = {'admission_ids': [1, 2]}

        scaler.fit(dataset, aux['admission_ids'])

        self.assertTrue(hasattr(scaler, 'scaler_'))
        self.fail()

    def test_transform(self):
        scaler = InputScaler()
        scaler.scaler_ = MagicMock()

        dataset = MagicMock()
        dataset.config.tables.icu_inputs.code_alias = 'code'
        dataset.config.tables.icu_inputs.derived_normalized_amount_per_hour = 'value'
        dataset.tables.icu_inputs.code = ['a', 'b', 'c']
        dataset.tables.icu_inputs.value = [1, 2, 3]

        aux = {}

        dataset, aux = scaler(dataset, aux)

        scaler.scaler_.assert_called_with(dataset.tables.icu_inputs)
        self.fail()

    def test_invalid_column(self):
        scaler = InputScaler()

        dataset = MagicMock()
        dataset.config.tables.icu_inputs.code_alias = 'bad_code'
        dataset.config.tables.icu_inputs.derived_normalized_amount_per_hour = 'bad_value'

        aux = {}

        with self.assertRaises(ValueError):
            scaler(dataset, aux)
        self.fail()

    def test_empty_dataset(self):
        scaler = InputScaler()

        dataset = MagicMock()
        dataset.tables = {}

        aux = {}

        dataset, aux = scaler(dataset, aux)

        self.assertEqual(dataset, {})
        self.fail()


# Mocks

class MockDataset:
    def __init__(self, admissions):
        self.tables = MockTables(admissions)


class MockTables:
    def __init__(self, admissions):
        self.admissions = admissions


if __name__ == '__main__':
    unittest.main()
