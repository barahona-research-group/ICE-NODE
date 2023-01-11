"""Encapsulation of predicted risk for a dictionary of subjects."""

from dataclasses import dataclass
from typing import Optional
import jax.numpy as jnp

from ..ehr import Admission_JAX


@dataclass
class SubjectPredictedRisk:
    admission: Admission_JAX
    prediction: jnp.ndarray
    trajectory: Optional[jnp.ndarray] = None

    def __str__(self):
        return f"""
                adm_id: {self.admission.admission_id}\n
                prediction: {self.prediction}\n
                """


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
            subject_id: int,
            admission: Admission_JAX,
            prediction: jnp.ndarray,
            trajectory: Optional[jnp.ndarray] = None):

        if subject_id not in self:
            self[subject_id] = {}

        self[subject_id][admission.admission_id] = SubjectPredictedRisk(
            admission=admission, prediction=prediction, trajectory=trajectory)

    def get_subjects(self):
        return sorted(self.keys())

    def get_risks(self, subject_id):
        risks = self[subject_id]
        return list(map(risks.get, sorted(risks)))

    def subject_prediction_loss(self, subject_id, loss_f):
        loss = [
            loss_f(r.admission.dx_outcome, r.prediction)
            for r in self[subject_id].values()
        ]
        return sum(loss) / len(loss)

    def prediction_loss(self, loss_f):
        loss = [
            self.subject_prediction_loss(subject_id, loss_f)
            for subject_id in self.keys()
        ]
        return sum(loss) / len(loss)
