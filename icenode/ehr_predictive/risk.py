"""Encapsulation of predicted risk for a dictionary of subjects."""


class SubjectPredictedRisk:

    def __init__(self,
                 admission_id,
                 time,
                 prediction,
                 ground_truth=None,
                 **other_attrs):
        self.admission_id = admission_id
        self.time = time
        self.prediction = prediction
        self.ground_truth = ground_truth
        self.other_attrs = other_attrs


class BatchPredictedRisks:

    def __init__(self):
        self.subject_risks = {}

    def add_prediction(self,
                       subject_id,
                       admission_id,
                       time,
                       prediction,
                       ground_truth=None,
                       **other_attrs):
        if subject_id not in self.subject_risks:
            self.subject_risks[subject_id] = {}

        self.subject_risks[subject_id][time] = SubjectPredictedRisk(
            admission_id=admission_id,
            time=time,
            prediction=prediction,
            ground_truth=ground_truth,
            **other_attrs)
