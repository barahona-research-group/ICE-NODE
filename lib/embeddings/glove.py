"""GloVe for initializing the basic embeddings for the GRAM method"""

import math
from typing import List, Dict, Mapping, Tuple, Callable, Set
from absl import logging
import numpy as np
import numpy.random as nrandom
"""
This implementation follows the following technical blog:
http://www.foldl.me/2014/glove-python/
"""


def run_iter(data, learning_rate=0.05, x_max=100, alpha=0.75):
    """
    Run a single iteration of GloVe training using the given
    cooccurrence data and the previously computed weight vectors /
    biases and accompanying gradient histories.
    `data` is a pre-fetched data / weights list where each element is of
    the form
        (v_main, v_context,
         b_main, b_context,
         gradsq_W_main, gradsq_W_context,
         gradsq_b_main, gradsq_b_context,
         cooccurrence)
    as produced by the `train_glove` function. Each element in this
    tuple is an `ndarray` view into the data structure which contains
    it.
    See the `train_glove` function for information on the shapes of `W`,
    `biases`, `gradient_squared`, `gradient_squared_biases` and how they
    should be initialized.
    The parameters `x_max`, `alpha` define our weighting function when
    computing the cost for two word pairs; see the GloVe paper for more
    details.
    Returns the cost associated with the given weight assignments and
    updates the weights by online AdaGrad in place.
    """

    global_cost = 0

    for (v_main, v_context, b_main, b_context, gradsq_W_main, gradsq_W_context,
         gradsq_b_main, gradsq_b_context, cooccurrence) in data:

        weight = (cooccurrence / x_max)**alpha if cooccurrence < x_max else 1

        # Compute inner component of cost function, which is used in
        # both overall cost calculation and in gradient calculation
        #
        #   $$ J' = w_i^Tw_j + b_i + b_j - log(X_{ij}) $$
        cost_inner = (v_main.dot(v_context) + b_main[0] + b_context[0] -
                      math.log(cooccurrence))

        # Compute cost
        #
        #   $$ J = f(X_{ij}) (J')^2 $$
        cost = weight * (cost_inner**2)

        # Add weighted cost to the global cost tracker
        global_cost += 0.5 * cost

        # Compute gradients for word vector terms.
        #
        # NB: `main_word` is only a view into `W` (not a copy), so our
        # modifications here will affect the global weight matrix;
        # likewise for context_word, biases, etc.
        grad_main = weight * cost_inner * v_context
        grad_context = weight * cost_inner * v_main

        # Compute gradients for bias terms
        grad_bias_main = weight * cost_inner
        grad_bias_context = weight * cost_inner

        # Now perform adaptive updates
        v_main -= (learning_rate * grad_main / np.sqrt(gradsq_W_main))
        v_context -= (learning_rate * grad_context / np.sqrt(gradsq_W_context))

        b_main -= (learning_rate * grad_bias_main / np.sqrt(gradsq_b_main))
        b_context -= (learning_rate * grad_bias_context /
                      np.sqrt(gradsq_b_context))

        # Update squared gradient sums
        gradsq_W_main += np.square(grad_main)
        gradsq_W_context += np.square(grad_context)
        gradsq_b_main += grad_bias_main**2
        gradsq_b_context += grad_bias_context**2

    return global_cost


def train_glove(cooccurrences: np.ndarray,
                vector_size=100,
                iterations=25,
                prng_seed=0,
                **kwargs) -> np.ndarray:
    """
    Train GloVe vectors on the given `cooccurrences`, where
    each element is of the form ((word_i_id, word_j_id): x_ij)
    where `x_ij` is a cooccurrence value $X_{ij}$ as presented in the
    matrix defined by `build_cooccur` and the Pennington et al. (2014)
    paper itself.
    If `iter_callback` is not `None`, the provided function will be
    called after each iteration with the learned `W` matrix so far.
    Keyword arguments are passed on to the iteration step function
    `run_iter`.
    Returns the computed word vector dictionary (code: vector).
    """

    # In case JAX array passed.
    cooccurrences = np.array(cooccurrences)

    size = len(cooccurrences)
    # nx: code space dim
    # ne: embedded space dim
    # (2nx, ne)
    W = (np.random.rand(size * 2, vector_size) - 0.5) / float(vector_size + 1)

    # (2nx, )
    biases = (np.random.rand(size * 2) - 0.5) / float(vector_size + 1)

    # (2nx, ne)
    gradient_squared = np.ones((size * 2, vector_size), dtype=np.float64)

    # (2nx, ne)
    gradient_squared_biases = np.ones(size * 2, dtype=np.float64)

    # i: main code index
    # j: context code index
    indices = list(zip(*cooccurrences.nonzero()))

    # All list elements are views (references) to their corresponding arrays.
    data = [
        (
            # (1, ne), (1, ne)
            W[i],
            W[j + size],
            # (1, 1), (1, 1)
            biases[i:i + 1],
            biases[j + size:j + size + 1],
            # (1, ne), (1, ne)
            gradient_squared[i],
            gradient_squared[j + size],
            # (1, 1), (1, 1)
            gradient_squared_biases[i:i + 1],
            gradient_squared_biases[j + size:j + size + 1],
            # (1, ) <- by value
            cooccurrences[i, j]) for (i, j) in indices
    ]

    prng = nrandom.default_rng(0)

    for i in range(iterations):
        logging.info("\tBeginning iteration %i..", i)
        # We want to iterate over data randomly so as not to unintentionally
        # bias the word vector contents
        prng.shuffle(data)
        cost = run_iter(data, **kwargs)

        logging.info("\t\tDone (cost %f)", cost)

    # GloVe paper Section 4.2: "we choose to use the sum W +W˜ as our word vectors
    W_words = W[:size] + W[size:]

    # (nx, ne)
    return W_words
