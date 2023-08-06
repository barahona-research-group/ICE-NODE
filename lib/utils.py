"""Miscalleneous utility functions."""

import os
import contextlib
import zipfile
import json

import numpy as np
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_map, tree_leaves
import equinox as eqx

from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook


def _tqdm_backend():
    """Return the appropriate constructor of tqdm based on the executor
    interpreter, i.e. if it is running on a notebook or not."""
    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            return tqdm_notebook
    except NameError:
        pass
    return tqdm


tqdm_constructor = _tqdm_backend()


def params_size(m):
    """Return the parameters count in a PyTree object."""
    leaves, _ = tree_flatten(eqx.filter(m, eqx.is_inexact_array))
    return sum(jnp.size(x) for x in leaves)


def tree_hasnan(m):
    """Retrun True if any paramter in t (PyTree object) is NaN."""
    t = eqx.filter(m, eqx.is_inexact_array)
    return any(map(lambda x: jnp.any(jnp.isnan(x)), tree_leaves(t)))


def tree_lognan(m):
    """Returns a PyTree object with the same structure of the
    input (PyTree object). For each leaf (i.e. jax.numpy object) in
    the input, assign True if that leaf has NaN value(s).
    """
    t = eqx.filter(m, eqx.is_inexact_array)
    return tree_map(lambda x: jnp.any(jnp.isnan(x)).item(), t)


def tree_add_scalar_mul(tree_x, scalar, tree_y):
    """Compute tree_x + scalar * tree_y."""
    tree_x = eqx.filter(tree_x, eqx.is_inexact_array)
    tree_y = eqx.filter(tree_y, eqx.is_inexact_array)
    return tree_map(lambda x, y: x + scalar * y, tree_x, tree_y)


def model_params_scaler(model, scaler, filter_spec):
    func_model = eqx.filter(model, filter_spec, inverse=True)
    prms_model = eqx.filter(model, filter_spec, inverse=False)
    return eqx.combine(func_model, tree_scalar(prms_model, scaler))


def tree_scalar(tree, scalar):
    """Compute tree_x + scalar * tree_y."""
    return tree_map(lambda x: scalar * x, tree)


def array_hasnan(arr):
    """Return True if the input array has NaN value(s)."""
    return jnp.any(jnp.isnan(arr) | jnp.isinf(arr))


def translate_path(path):
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def append_params_to_zip(model, params_name, zipfile_fname):
    with zipfile.ZipFile(translate_path(zipfile_fname),
                         compression=zipfile.ZIP_STORED,
                         mode="a") as archive:
        with archive.open(params_name, "w") as zip_member:
            eqx.tree_serialise_leaves(zip_member, model)


def zip_members(zipfile_fname):
    with zipfile.ZipFile(translate_path(zipfile_fname)) as archive:
        return archive.namelist()


def load_config(config_file):
    """Load a JSON file from `config_file`."""
    with open(translate_path(config_file)) as json_file:
        return json.load(json_file)


class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray) or isinstance(obj, jnp.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def write_config(data, config_file):
    """Write `data` (dict object) into the given filepath `config_file`."""
    with open(translate_path(config_file), 'w') as outfile:
        json.dump(data, outfile, indent=4, sort_keys=True, cls=NumpyEncoder)


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
