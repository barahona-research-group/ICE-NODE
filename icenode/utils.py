"""Miscalleneous utility functions."""
import os
import contextlib
import json
import pickle
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_map, tree_leaves


class OOPError(Exception):
    """Object Oriented design Exception (e.g. calling abstract method)."""


class Unsupported(Exception):
    """Unsupported operation."""


class LazyDict(dict):

    def __getitem__(self, k):
        v = super().__getitem__(k)
        if callable(v):
            v = v()
            super().__setitem__(k, v)
        return v

    def get(self, k, default=None):
        if k in self:
            return self.__getitem__(k)
        return default


def parameters_size(t):
    """Return the parameters count in a PyTree object."""
    leaves, _ = tree_flatten(t)
    return sum(jnp.size(x) for x in leaves)


def tree_hasnan(t):
    """Retrun True if any paramter in t (PyTree object) is NaN."""
    return any(map(lambda x: jnp.any(jnp.isnan(x)), tree_leaves(t)))


def tree_lognan(t):
    """Returns a PyTree object with the same structure of the
    input (PyTree object). For each leaf (i.e. jax.numpy object) in
    the input, assign True if that leaf has NaN value(s).
    """
    return tree_map(lambda x: jnp.any(jnp.isnan(x)).item(), t)


def array_hasnan(arr):
    """Return True if the input array has NaN value(s)."""
    return jnp.any(jnp.isnan(arr) | jnp.isinf(arr))


# TODO(Asem): Move to Haiku models.
# TODO(Asem): remove any reference to the authors in the documentation.
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
    """
    Store the parameters (PyTree object) into a new file
    given by `params_file`.
    """
    with open(params_file, 'wb') as file_rsc:
        pickle.dump(params, file_rsc, protocol=pickle.HIGHEST_PROTOCOL)


def load_params(params_file):
    """
    Load the parameters in `params_file` filepath and return as PyTree Object.
    """
    with open(params_file, 'rb') as file_rsc:
        return pickle.load(file_rsc)


def load_config(config_file):
    """Load a JSON file from `config_file`."""
    with open(config_file) as json_file:
        return json.load(json_file)


def write_config(data, config_file):
    """Write `data` (dict object) into the given filepath `config_file`."""
    with open(config_file, 'w') as outfile:
        json.dump(data, outfile, indent=4, sort_keys=True)


@contextlib.contextmanager
def modified_environ(*remove, **update):
    """
    Copy from: https://stackoverflow.com/a/34333710
    Temporarily updates the ``os.environ`` dictionary in-place.

    The ``os.environ`` dictionary is updated in-place so that the modification
    is sure to work in all situations.

    :param remove: Environment variables to remove.
    :param update: Dictionary of environment variables and values to add/update.
    """
    env = os.environ
    update = update or {}
    remove = remove or []

    # List of environment variables being updated or removed.
    stomped = (set(update.keys()) | set(remove)) & set(env.keys())
    # Environment variables and values to restore on exit.
    update_after = {k: env[k] for k in stomped}
    # Environment variables and values to remove on exit.
    remove_after = frozenset(k for k in update if k not in env)

    try:
        env.update(update)
        [env.pop(k, None) for k in remove]
        yield
    finally:
        env.update(update_after)
        [env.pop(k) for k in remove_after]
