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
from ..ehr import WindowedInterface_JAX, Subject_JAX
from ..metric import BatchPredictedRisks


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

        W = 1e-5 * jnp.ones((output_size, input_size), dtype=float)
        b = jnp.zeros(output_size, dtype=float)

        self.supported_labels = None
        self.n_labels = None

    @classmethod
    def from_config(cls, conf, subject_interface, train_split, key):
        return cls(subject_interface.dx_dim, subject_interface.dx_outcome_dim,
                   key)

    @staticmethod
    def alpha_beta_config(alpha, beta):
        # alpha is for L2-norm, beta is for L1-norm
        return (beta, alpha / (beta + jnp.finfo('float').eps))

    def predict_logits(self, X):
        return self.W @ X + self.b

    def predict_proba(self, X):
        return jax.nn.softmax(self.predict_logits(X))

    def __call__(self, subject_interface: Subject_JAX,
                 subjects_batch: List[int], args):
        subject_interface = WindowedInterface_JAX(subject_interface)
        risk_prediction = BatchPredictedRisks()
        for subj_id in subjects_batch:
            adms = subject_interface[subj_id]
            features = subject_interface.features[subj_id]

            X = np.vstack([feats.dx_features for feats in features[1:]])
            y = np.vstack([adm.dx_outcome for adm in adms[1:]])
            risk = self.predict_logits(X)

            for (adm, pred) in zip(adms[1:], risk):
                risk_prediction.add(subject_id=subj_id,
                                    admission=adm,
                                    prediction=pred)

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
