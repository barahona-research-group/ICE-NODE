import sys
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('..')
from icenode.ml import (ICENODE_2LR, ICENODE_UNIFORM_2LR, GRU, RETAIN, WLR,
                        PR_ICENODE)
from icenode.ehr import (AbstractEHRDataset, ConsistentSchemeEHRDataset,
                         Subject_JAX, code_scheme, datasets)
from icenode.utils import (load_config, load_params, write_params,
                           write_config, modified_environ)

model_cls = {
    'ICE-NODE': ICENODE_2LR,
    'ICE-NODE_UNIFORM': ICENODE_UNIFORM_2LR,
    'GRU': GRU,
    'RETAIN': RETAIN,
    'LogReg': WLR
}

lsr_model_cls = {
    'dx_icenode_M': ICENODE_2LR,
    'dx_icenode_G': ICENODE_2LR,
    'dxpr_icenode_M': PR_ICENODE,
    'dxpr_icenode_G': PR_ICENODE,
    'dx_gru_M': GRU,
    'dx_gru_G': GRU,
    'dx_retain': RETAIN
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


def lsr_get_model(clf, config, params, interface):
    model = lsr_model_cls[clf].create_model(config, interface, [])
    state = model.init_with_params(config, params)
    return model, state


def lsr_get_models(clfs, config, params, interface):
    emb = lambda c: 'G' if '_G' in c else 'M'

    return {
        clf: lsr_get_model(clf, config[clf], params[clf], interface[emb(clf)])
        for clf in clfs
    }
