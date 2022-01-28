import json
import pickle
import pandas as pd
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_map, tree_leaves
from jax.experimental.optimizers import (pack_optimizer_state,
                                         unpack_optimizer_state)
from .metrics import (top_k_detectability_scores, auc_scores,
                      confusion_matrix_scores)


def parameters_size(pytree):
    leaves, _ = tree_flatten(pytree)
    return sum(jnp.size(x) for x in leaves)


def tree_hasnan(t):
    return any(map(lambda x: jnp.any(jnp.isnan(x)), tree_leaves(t)))


def tree_lognan(t):
    return tree_map(lambda x: jnp.any(jnp.isnan(x)).item(), t)


def array_hasnan(arr):
    return jnp.any(jnp.isnan(arr) | jnp.isinf(arr))


# For haiku-dm modules
def wrap_module(module, *module_args, **module_kwargs):
    """
    Wrap the module in a function to be transformed.
    """
    def wrap(*args, **kwargs):
        """
        Wrapping of module.
        """
        model = module(*module_args, **module_kwargs)
        return model(*args, **kwargs)

    return wrap


def write_params(params, params_file):
    with open(params_file, 'wb') as file_rsc:
        pickle.dump(params, file_rsc, protocol=pickle.HIGHEST_PROTOCOL)


def load_params(params_file):
    with open(params_file, 'rb') as file_rsc:
        return pickle.load(file_rsc)


def load_config(config_file):
    with open(config_file) as json_file:
        return json.load(json_file)


def write_config(data, config_file):
    with open(config_file, 'w') as outfile:
        json.dump(data, outfile, indent=4, sort_keys=True)
