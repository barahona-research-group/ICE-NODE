import sys
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('..')
from icenode.ehr_predictive.dx_icenode_2lr import ICENODE
from icenode.ehr_predictive.dx_icenode_uniform2lr import ICENODE as ICENODE_UNIFORM
from icenode.ehr_predictive.dx_gram import GRAM
from icenode.ehr_predictive.dx_retain import RETAIN
from icenode.ehr_model.ccs_dag import ccs_dag
from icenode.ehr_model.jax_interface import create_patient_interface

from icenode.utils import load_config, load_params, write_params

model_cls = {
    'ICE-NODE': ICENODE,
    'ICE-NODE_UNIFORM': ICENODE_UNIFORM,
    'GRU': GRAM,
    'RETAIN': RETAIN
}


def eval_(model, ids):
    model, state = model
    return model.eval(state, ids)['risk_prediction']


def eval2_(model, ids):
    model, state = model
    return model.eval(state, ids)


def get_model(clf, config, params, interface):
    model = model_cls[clf].create_model(config, interface, [], None)
    state = model.init_with_params(config, params)
    return model, state


def get_models(clfs, config, params, interface):
    return {
        clf: get_model(clf, config[clf], params[clf], interface)
        for clf in clfs
    }
