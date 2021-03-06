"""."""

import sys

sys.path.append(['.', '..'])
import common as C

from icenode.ehr_predictive.trainer import (MinibatchLogger,
                                            EvaluationDiskWriter,
                                            ParamsDiskWriter)


def make_reporters(clfs, clf_dir):
    return {
        clf: [
            MinibatchLogger(),
            EvaluationDiskWriter(output_dir=clf_dir[clf]),
            ParamsDiskWriter(output_dir=clf_dir[clf])
        ]
        for clf in clfs
    }


def init_models(clfs, config, interface, train_ids):
    models = {
        clf: C.model_cls[clf].create_model(config[clf], interface, train_ids)
        for clf in clfs
    }

    return {clf: (models[clf], models[clf].init(config[clf])) for clf in clfs}


def train(model, config, splits, code_groups, reporters):
    model, m_state = model
    trainer = model.get_trainer()
    return trainer(model=model,
                   m_state=m_state,
                   config=config,
                   splits=splits,
                   rng_seed=42,
                   code_frequency_groups=code_groups,
                   reporters=reporters)
