"""Miscalleneous utility functions."""

import contextlib
import json
import os
import zipfile

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_flatten, tree_map, tree_leaves
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
    """
    Parameters:
        m (PyTree object) - The PyTree object to get the parameter count of.

    Returns:
        int - The total number of parameters in the PyTree object m.
    """
    leaves, _ = tree_flatten(eqx.filter(m, eqx.is_inexact_array))
    return sum(jnp.size(x) for x in leaves)


def tree_hasnan(m):
    """Check if a PyTree contains any NaN values.

    Parameters:
        m (PyTree) - The PyTree to check for NaN values.

    Returns:
        bool - True if m contains any NaN values, False otherwise.
    """
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
    """Scale the parameters in a model by a given scaler.

    This scales the parameters (as selected by filter_spec) of the model
    by the given scaler value, while keeping the functional part of the
    model unchanged.

    Parameters:
        model: The model PyTree object.
        scaler: The scalar value to multiply the parameters by.
        filter_spec: The filter specification to select parameters.

    Returns:
        The model PyTree with parameters scaled.
    """
    func_model = eqx.filter(model, filter_spec, inverse=True)
    prms_model = eqx.filter(model, filter_spec, inverse=False)
    return eqx.combine(func_model, tree_scalar(prms_model, scaler))


def tree_scalar(tree, scalar):
    """Multiply all leaf nodes in a PyTree by a scalar value.

    Parameters:
        tree: PyTree object to multiply.
        scalar: Scalar value to multiply.

    Returns:
        PyTree object with all leaf nodes multiplied by the scalar.
    """

    return tree_map(lambda x: scalar * x, tree)


def array_hasnan(arr):
    """Check if a jax numpy array contains any NaN or infinite values.

    Parameters:
        arr: jax numpy array to check.

    Returns: 
        bool: True if arr contains any NaN or infinite values.
    """
    return jnp.any(jnp.isnan(arr) | jnp.isinf(arr))


def translate_path(path):
    """Translate a filesystem path by replacing environment 
    variables with their values.

    Parameters:
        path: Filesystem path to translate.

    Returns:
        Absolute, expanded, translated path.
    """
    assert isinstance(path, str), "Path must be a string."
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def append_params_to_zip(model, params_name, zipfile_fname):
    """Append model parameters to a zip file.

    Appends the parameters from `model` to the zip file at `zipfile_fname`
    under the name `params_name`, using `tree_serialise_leaves` to serialize
    the parameters.

    Parameters:
        model: Model whose parameters to serialize.
        params_name: Name to save parameters under in the zip file.
        zipfile_fname: Path to zip file to append parameters to.
    """
    with zipfile.ZipFile(
            translate_path(zipfile_fname), compression=zipfile.ZIP_STORED, mode="a"
    ) as archive:
        with archive.open(params_name, "w") as zip_member:
            eqx.tree_serialise_leaves(zip_member, model)


def zip_members(zipfile_fname):
    """Return a list of the names of members in the ZIP file.

    Args:
        zipfile_fname: Path to the ZIP file.

    Returns:
        A list of member names in the ZIP file.
    """
    with zipfile.ZipFile(translate_path(zipfile_fname)) as archive:
        return archive.namelist()


def load_config(config_file):
    """Load a JSON file from `config_file`.

    Parameters:
        config_file: Path to the JSON configuration file to load.

    Returns:
        The deserialized JSON contents of the configuration file.
    """
    with open(translate_path(config_file)) as json_file:
        return json.load(json_file)


"""NumpyEncoder class to handle encoding numpy arrays to JSON."""


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) or isinstance(obj, jnp.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def write_config(data, config_file):
    """Write the given data object to the specified config file as JSON.

    Parameters:
        data: The data object to serialize to JSON and write.
        config_file: The path to the config file to write.
    """
    with open(translate_path(config_file), "w") as outfile:
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
