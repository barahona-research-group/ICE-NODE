"""Encapsulation of predicted risk for a dictionary of subjects."""


class SubjectPredictedRisk:
    def __init__(self,
                 admission_id,
                 index,
                 prediction,
                 ground_truth=None,
                 **other_attrs):
        self.admission_id = admission_id
        self.index = index
        self.prediction = prediction
        self.ground_truth = ground_truth
        self.other_attrs = other_attrs


class BatchPredictedRisks:
    def __init__(self):
        self.subject_risks = {}

    def add(self,
            subject_id,
            admission_id,
            index,
            prediction,
            ground_truth=None,
            **other_attrs):
        if subject_id not in self.subject_risks:
            self.subject_risks[subject_id] = {}

        self.subject_risks[subject_id][index] = SubjectPredictedRisk(
            admission_id=admission_id,
            index=index,
            prediction=prediction,
            ground_truth=ground_truth,
            **other_attrs)

    def get_subjects(self):
        return sorted(self.subject_risks)

    def get_risks(self, subject_id):
        risks = self.subject_risks[subject_id]
        return list(map(risks.get, sorted(risks)))
