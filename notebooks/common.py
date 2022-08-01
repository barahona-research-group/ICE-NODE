import sys
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('..')
from icenode.ml import ICENODE_2LR, ICENODE_UNIFORM_2LR, GRU, RETAIN, WLR
from icenode.ehr import MIMICDataset, Subject_JAX, code_scheme, datasets
from icenode.utils import load_config, load_params, write_params, write_config, modified_environ

model_cls = {
    'ICE-NODE': ICENODE_2LR,
    'ICE-NODE_UNIFORM': ICENODE_UNIFORM_2LR,
    'GRU': GRU,
    'RETAIN': RETAIN,
    'LogReg': WLR
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
