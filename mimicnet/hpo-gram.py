import argparse
import os
from pathlib import Path
import random
import pickle
from typing import (AbstractSet, Any, Callable, Dict, Iterable, List, Mapping,
                    Optional, Tuple, Union)
from functools import partial

import pandas as pd
from absl import logging
from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_map

from jax.experimental import optimizers
import optuna
from optuna.storages import RDBStorage
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from sqlalchemy.pool import NullPool

from .train_gram import (PatientGRAMInterface, tree_hasnan, parameters_size,
                         create_patient_interface, loss_fn,
                         top_k_detectability_scores, top_k_detectability_df,
                         auc_scores, roc_df)

from .concept import Subject
from .dag import CCSDAG
from .jax_interface import SubjectDiagSequenceJAXInterface
from .glove import glove_representation
from .gram import DAGGRAM
from .metrics import (l1_absolute, l2_squared)

logging.set_verbosity(logging.INFO)


def run_trials(study_name: str, store_url: str, num_trials: int,
               mimic_processed_dir: str, output_dir: str, cpu: bool,
               job_id: str):

    storage = RDBStorage(url=store_url, engine_kwargs={'poolclass': NullPool})
    study = optuna.create_study(study_name=study_name,
                                direction="maximize",
                                storage=storage,
                                load_if_exists=True,
                                sampler=TPESampler(),
                                pruner=HyperbandPruner())
    study.set_user_attr('metric', 'auc')

    if cpu:
        jax.config.update('jax_platform_name', 'cpu')
    else:
        jax.config.update('jax_platform_name', 'gpu')

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logging.info('[LOADING] Patients JAX Interface.')
    patient_interface = create_patient_interface(mimic_processed_dir)
    logging.info('[DONE] Patients JAX Interface')

    rng = random.Random(42)
    subjects_id = list(patient_interface.subjects.keys())
    rng.shuffle(subjects_id)

    train_validation_split = 0.8
    train_ids = subjects_id[:int(train_validation_split * len(subjects_id))]
    valid_ids = subjects_id[int(train_validation_split * len(subjects_id)):]
    codes_by_percentiles = patient_interface.diag_single_ccs_by_percentiles(
        20, train_ids)
    eval_freq = 10
    save_freq = 100

    def create_config(trial: optuna.Trial):
        config = {
            'glove_config': {
                'diag_idx':
                patient_interface.diag_multi_ccs_idx,
                'proc_idx':
                patient_interface.proc_multi_ccs_idx,
                'ccs_dag':
                patient_interface.dag,
                'subjects':
                patient_interface.subjects.values(),
                'diag_vector_size':
                trial.suggest_int('diag_vector_size', 100, 250, 50),
                'proc_vector_size':
                50,
                'iterations':
                30,
                'window_size_days':
                2 * 365
            },
            'gram_config': {
                'diag': {
                    'ccs_dag':
                    patient_interface.dag,
                    'code2index':
                    patient_interface.diag_multi_ccs_idx,
                    'attention_method':
                    trial.suggest_categorical('attention_method',
                                              ['tanh', 'l2']),
                    'attention_dim':
                    trial.suggest_int('attention_dim', 100, 250, 50),
                    'ancestors_mat':
                    patient_interface.diag_multi_ccs_ancestors_mat
                }
            },
            'model': {
                'state_size': trial.suggest_int('state_size', 100, 500, 50),
            },
            'training': {
                'batch_size': trial.suggest_int('batch_size', 5, 30, 5),
                'epochs': 10,
                'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
                'loss_mixing': {
                    'l1_reg': trial.suggest_float('l1_reg',
                                                  1e-7,
                                                  1e-1,
                                                  log=True),
                    'l2_reg': trial.suggest_float('l2_reg',
                                                  1e-6,
                                                  1e-1,
                                                  log=True),
                }
            }
        }

        return config

    def objective(trial: optuna.Trial):
        trial.set_user_attr('job_id', job_id)

        trial_dir = os.path.join(output_dir, f'trial_{trial.number:03d}')

        trial.set_user_attr('dir', trial_dir)

        Path(trial_dir).mkdir(parents=True, exist_ok=True)

        logging.info('[LOADING] Sampling & Initializing Models')

        config = create_config(trial)

        logging.info(f'Trial {trial.number} HPs: {trial.params}')
        diag_glove, _ = glove_representation(**config['glove_config'])

        diag_gram = DAGGRAM(**config['gram_config']['diag'],
                            basic_embeddings=diag_glove)

        model = PatientGRAMInterface(subject_interface=patient_interface,
                                     diag_gram=diag_gram,
                                     **config['model'])

        prng_key = jax.random.PRNGKey(rng.randint(0, 100))
        params = model.init_params(prng_key)

        trial.set_user_attr('parameters_size', parameters_size(params))
        logging.info('[DONE] Sampling & Initializing Models')

        lr = config['training']['lr']
        loss_mixing = config['training']['loss_mixing']
        opt_init, opt_update, get_params = optimizers.adam(step_size=lr)
        opt_state = opt_init(params)

        loss = partial(loss_fn, model, loss_mixing)

        def update(
                step: int, batch: Iterable[int],
                opt_state: optimizers.OptimizerState
        ) -> optimizers.OptimizerState:
            params = get_params(opt_state)
            grads, data = jax.grad(loss, has_aux=True)(params, batch)
            return opt_update(step, grads, opt_state), data

        batch_size = config['training']['batch_size']
        batch_size = min(batch_size, len(train_ids))

        epochs = config['training']['epochs']
        iters = int(epochs * len(train_ids) / batch_size)
        trial.set_user_attr('steps', iters)
        val_pbar = tqdm(total=iters)

        for step in range(iters):
            rng.shuffle(train_ids)
            train_batch = train_ids[:batch_size]
            val_pbar.update(1)

            opt_state, res = update(step, train_batch, opt_state)
            if tree_hasnan(get_params(opt_state)):
                trial.set_user_attr('nan', 1)
                return float('nan')

            if step % eval_freq != 0: continue

            params = get_params(opt_state)
            _, trn_res = loss(params, train_batch)
            _, val_res = loss(params, valid_ids)

            losses = pd.DataFrame(index=trn_res['loss'].keys(),
                                  data={
                                      'Training': trn_res['loss'].values(),
                                      'Validation': val_res['loss'].values()
                                  })

            detections_trn = top_k_detectability_scores(
                codes_by_percentiles, trn_res['diag_detectability_df'])
            detections_val = top_k_detectability_scores(
                codes_by_percentiles, val_res['diag_detectability_df'])
            detections_trn_df = pd.DataFrame(
                index=detections_trn['pre'].keys(),
                data={
                    'Trn(pre)': detections_trn['pre'].values(),
                    'Trn(post)': detections_trn['post'].values()
                })

            detections_val_df = pd.DataFrame(
                index=detections_val['pre'].keys(),
                data={
                    'Val(pre)': detections_val['pre'].values(),
                    'Val(post)': detections_val['post'].values()
                })

            auc_trn = auc_scores(trn_res['diag_roc_df'])
            auc_val = auc_scores(val_res['diag_roc_df'])

            auc_df = pd.DataFrame(data={
                'Trn(AUC)': auc_trn,
                'Val(AUC)': auc_val
            })
            df_save_prefix = os.path.join(trial_dir, f'step{step:03d}')

            for name, df in {
                    'losses': losses,
                    'detections_trn_df': detections_trn_df,
                    'detections_val_df': detections_val_df,
                    'auc_df': auc_df
            }.items():
                df.to_csv(f'{df_save_prefix}_{name}.csv')
                logging.info(df)

            auc = auc_val
            trial.report(auc, step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return auc

    study.optimize(objective, n_trials=num_trials)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--mimic-processed-dir',
                        required=True,
                        help='Absolute path to MIMIC-III processed tables')
    parser.add_argument('-o',
                        '--output-dir',
                        required=True,
                        help='Aboslute path to log intermediate results')
    parser.add_argument('-n',
                        '--num-trials',
                        type=int,
                        required=True,
                        help='Number of HPO trials.')

    parser.add_argument('-s',
                        '--store-url',
                        required=True,
                        help='Storage URL, e.g. for PostgresQL database')

    parser.add_argument('--study-name', required=True)

    parser.add_argument('--job-id', required=False)

    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    study_name = args.study_name
    store_url = args.store_url
    num_trials = args.num_trials
    mimic_processed_dir = args.mimic_processed_dir
    output_dir = args.output_dir
    job_id = args.job_id or 'unknown'
    cpu = args.cpu

    run_trials(study_name=study_name,
               store_url=store_url,
               num_trials=num_trials,
               mimic_processed_dir=mimic_processed_dir,
               output_dir=output_dir,
               job_id=job_id,
               cpu=cpu)

### IMPORTANT: https://github.com/optuna/optuna/issues/1647
'''
How each sampler in Optuna treat pruned trials and failed trials is different
depending on each sampler. It is kind for users to document how
pruned or failed trials are processed when suggesting.
I think it is a good idea to give the ..note:: section
for each sampler's docstrings.

For each sampler, the behavior is as follows.
## optuna.samplers.RandomSampler & optuna.samplers.GridSampler & optuna.integration.PyCmaSampler

These samplers treat pruned trials and failed trials samely.
They do not consider any pruned or failed trials. They simply ignore those trials.



## optuna.samplers.TPESampler

This sampler treats pruned trials and failed trials differently.
This sampler simply ignores failed trials. On the other hand, this sampler
considers pruned trials to suggest the next parameters in each iteration.
Concretely, This sampler makes a ranking of completed and pruned trials
based on the pairs of the completed or pruned step and the evaluation value
when completed or pruned. Then, this sampler suggests the next parameters
according to the priority of the trial based on the ranking.


## optuna.samplers.CmaEsSampler & optuna.integration.SkoptSampler

These samplers treat pruned trials and failed trials differently.
This sampler simply ignores failed trials. On the other hand, this sampler
considers pruned trials only when the consider_pruned_trials flag is True.
When consider_pruned_trials = True, these samplers consider that pruned trials'
evaluation values are the evaluation values when pruned.
'''
