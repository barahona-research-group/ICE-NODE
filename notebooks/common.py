import sys
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('..')
from icenode.ml import ICENODE, ICENODE_UNIFORM, GRU, RETAIN, WLR
from icenode.ehr import MIMICDataset, Subject_JAX, code_scheme
from icenode.utils import load_config, load_params, write_params, write_config

model_cls = {
    'ICE-NODE': ICENODE,
    'ICE-NODE_UNIFORM': ICENODE_UNIFORM,
    'GRU': GRU,
    'RETAIN': RETAIN,
    'WLR': WLR
}


def eval_(model, ids):
    model, state = model
    return model.eval(state, ids)['risk_prediction']


def eval2_(model, ids):
    model, state = model
    return model.eval(state, ids)


def get_model(clf, config, params, interface):
    model = model_cls[clf].create_model(config, interface, [])
    state = model.init_with_params(config, params)
    return model, state


def get_models(clfs, config, params, interface):
    return {
        clf: get_model(clf, config[clf], params[clf], interface)
        for clf in clfs
    }
