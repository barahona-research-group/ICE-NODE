"""Encapsulation of predicted risk for a dictionary of subjects."""

import jax.numpy as jnp


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

    def __str__(self):
        return f"""
    adm_id (i: {self.index}): {self.admission_id}\n
    prediction: {self.prediction}\n
    ground_truth: {self.ground_truth}
    """

    def __eq__(self, other):
        id_attrs = lambda x: (x.admission_id, x.index)
        arr_attrs = lambda x: (x.prediction, x.ground_truth)
        return id_attrs(self) == id_attrs(other) and \
            all(map(jnp.array_equal, arr_attrs(self), arr_attrs(other)))


class BatchPredictedRisks(dict):

    def __init__(self):
        self.embeddings = dict()

    def __str__(self):
        subjects_str = []
        for subj_id, _risks in self.items():
            subjects_str.extend([
                f'subject_id:{subj_id}\n{_risk}' for _risk in _risks.values()
            ])
        return '\n========\n'.join(subjects_str)

    def set_subject_embeddings(self, subject_id, embeddings):
        self.embeddings[subject_id] = embeddings

    def get_subject_embeddings(self, subject_id):
        return self.embeddings[subject_id]

    def add(self,
            subject_id,
            admission_id,
            index,
            prediction,
            ground_truth=None,
            **other_attrs):
        if subject_id not in self:
            self[subject_id] = {}

        self[subject_id][index] = SubjectPredictedRisk(
            admission_id=admission_id,
            index=index,
            prediction=prediction,
            ground_truth=ground_truth,
            **other_attrs)

    def get_subjects(self):
        return sorted(self.keys())

    def get_risks(self, subject_id):
        risks = self[subject_id]
        return list(map(risks.get, sorted(risks)))
