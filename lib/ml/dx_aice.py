"""."""
from __future__ import annotations
from functools import partial
from collections import namedtuple
from typing import Any, Dict, List
from dataclasses import dataclass

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import equinox as eqx

from ..ehr import Subject_JAX, Admission_JAX, BatchPredictedRisks

from .dx_icenode import ICENODE



class SubjectState(eqx.Module):
    state: jnp.ndarray
    time: jnp.ndarray  # shape ()


@jax.jit
def balanced_focal_bce(y: jnp.ndarray,
                       logits: jnp.ndarray,
                       w_mask: jnp.ndarray,
                       gamma=2,
                       beta=0.999):
    """
    This loss function employs two concepts:
      - Effective number of sample, to mitigate class imbalance [1].
      - Focal loss, to underweight the easy to classify samples [2].

    The function takes four inputs:
      - The ground-truth `y` is a vector, where each
        element is an integer in :math:`\{0, 1\}`.
      - The predictions `logits`, before applying the Sigmoid function, where
      each element is a float in :math:`(-\infty, \infty)`.
      - `gamma` is the :math:`\gamma` parameter in [2].
      - `beta` is the :math:`\beta` parameter in [1].

    References:
      [1] _Cui et al._, Class-Balanced Loss Based on Effective Number of Samples.
      [2] _Lin et al., Focal Loss for Dense Object Detection.
    """

    n1 = jnp.sum(y, axis=0)
    n0 = len(y) - n1
    # Effective number of samples.
    e1 = (1 - beta**n1) / (1 - beta) + 1e-1
    e0 = (1 - beta**n0) / (1 - beta) + 1e-1

    # Focal weighting
    p = jnn.sigmoid(logits)
    w1 = jnp.power(1 - p, gamma)
    w0 = jnp.power(p, gamma)
    # Note: softplus(-logits) = -log(sigmoid(logits)) = -log(p)
    # Note: softplut(logits) = -log(1 - sigmoid(logits)) = -log(1-p)
    terms = y * (w1 / e1) * jnn.softplus(-logits) + (1 - y) * (
        w0 / e0) * jnn.softplus(logits)
    return jnp.nanmean(terms * w_mask)

@dataclass
class AICEBatchPredictedRisks(BatchPredictedRisks):
    out_mix: jnp.ndarray

    # Balanced, without mask
    def subject_prediction_loss(self, subject_id):
        subject_predictions = self[subject_id]
        x_o = []
        x_h = []
        for adm_idx in subject_predictions.keys():
            x_o.append(subject_predictions[adm_idx].admission.get_outcome())
            x_h.append(subject_predictions[adm_idx].prediction)
        x_o = jnp.vstack(x_o)
        x_hat = jnp.vstack(x_h)
        x_fo = jnp.cumsum(x_o, axis=0) > 0

        alpha = jnn.sigmoid(self.out_mix)
        loss_o = balanced_focal_bce(x_o, x_hat, alpha)
        loss_fo = balanced_focal_bce(x_fo, x_hat, 1 - alpha)
        return loss_o + loss_fo

    
    def prediction_loss(self, ignore=None):
        loss = [
            self.subject_prediction_loss(subject_id)
            for subject_id in self.keys()
        ]
        return jnp.nanmean(jnp.array(loss))


class AICE(ICENODE):
    in_mix: jnp.ndarray
    out_mix: jnp.ndarray

    @staticmethod
    def decoder_input_size(expt_config):
        return expt_config["emb"]["dx"]["embeddings_size"]

    def dx_embed(self, dx_G: jnp.ndarray, adms: List[Admission_JAX]): 
        dx_mat = jnp.vstack([adm.dx_vec for adm in adms])
        dx_mat_fo = jnp.cumsum(dx_mat, axis=0) > 0
        beta = jnn.sigmoid(self.in_mix)
        dx_mat = beta * dx_mat + (1 - beta) * dx_mat_fo
        return  [self.dx_emb.encode(dx_G, dx_vec) for dx_vec in dx_mat]
    
    def init_predictions(self):
        return AICEBatchPredictedRisks(self.out_mix)
    
    @property
    def dyn_state_size(self):
        return self.state_size + self.dx_emb.embeddings_size

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.out_mix = jnp.zeros((self.dx_dec.output_size,), dtype=jnp.float32)
        self.in_mix = jnp.zeros((self.dx_emb.input_size,), dtype=jnp.float32)
