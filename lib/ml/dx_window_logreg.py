"""Logistic Regression EHR predictive model based on diagnostic codes in
previous visits."""
from __future__ import annotations
from typing import Any, List, Dict, TYPE_CHECKING, Optional, Tuple

import numpy as np

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx

if TYPE_CHECKING:
    import optuna

from ..ml import AbstractModelProxMap
from ..ehr import Subject_JAX, BatchPredictedRisks


def prox_elastic_net(x: Any,
                     hyperparams: Optional[Tuple[Any, Any]] = None,
                     scaling: float = 1.0) -> Any:
    # Copied from https://github.com/google/jaxopt/blob/main/jaxopt/_src/prox.py
    r"""Proximal operator for the elastic net.
    .. math::
        \underset{y}{\text{argmin}} ~ \frac{1}{2} ||x - y||_2^2
        + \text{scaling} \cdot \text{hyperparams[0]} \cdot g(y)
    where :math:`g(y) = ||y||_1 + \text{hyperparams[1]} \cdot 0.5 \cdot ||y||_2^2`.
    Args:
        x: input pytree.
        hyperparams: a tuple, where both ``hyperparams[0]`` and ``hyperparams[1]``
        can be either floats or pytrees with the same structure as ``x``.
        scaling: a scaling factor.
    Returns:
        output pytree, with the same structure as ``x``.
    """
    if hyperparams is None:
        hyperparams = (1.0, 1.0)

    prox_l1 = lambda u, lam: jnp.sign(u) * jax.nn.relu(jnp.abs(u) - lam)
    fun = lambda u, lam, gamma: (prox_l1(u, scaling * lam) /
                                 (1.0 + scaling * lam * gamma))
    return jtu.tree_map(fun, x, hyperparams[0], hyperparams[1])


class WindowLogReg(AbstractModelProxMap):
    W: jnp.ndarray
    b: jnp.ndarray

    def __init__(self, input_size, output_size, key: "jax.random.PRNGKey"):
        super().__init__(dx_emb=None,
                         dx_dec=None,
                         state_size=0,
                         control_size=0)

        self.W = 1e-5 * jnp.ones((output_size, input_size), dtype=float)
        self.b = jnp.zeros(output_size, dtype=float)

    @classmethod
    def from_config(cls, conf, subject_interface, train_split, key):
        return cls(subject_interface.dx_dim, subject_interface.outcome_dim,
                   key)

    def weights(self):
        return (self.W, )

    @eqx.filter_jit
    def predict_logits(self, x):
        return self.W @ x + self.b

    @eqx.filter_jit
    def predict_proba(self, x):
        return jax.nn.softmax(self.predict_logits(x))

    def __call__(self, subject_interface: Subject_JAX,
                 subjects_batch: List[int], args):
        risk_prediction = BatchPredictedRisks()
        for subj_id in subjects_batch:
            adms = subject_interface[subj_id]
            X = subject_interface.tabular_features(subjects_batch)
            risks = [self.predict_logits(x) for x in X]
            for (adm, risk) in zip(adms[1:], risks):
                risk_prediction.add(subject_id=subj_id,
                                    admission=adm,
                                    prediction=risk)

        return {'predictions': risk_prediction}

    @staticmethod
    def prox_map():

        def prox_elasticnet(model, hyperparams):
            l1 = hyperparams['L_l1']
            l2 = hyperparams['L_l2']
            hps = (l1, l2 / l1 if l1 > 0 else 0)
            new_W = prox_elastic_net(model.W, hps)
            return eqx.tree_at(lambda m: m.W, model, replace=new_W)

        return prox_elasticnet
