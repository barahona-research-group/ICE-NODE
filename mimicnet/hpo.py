import argparse
import os
from pathlib import Path
import random
import pickle
from typing import (AbstractSet, Any, Callable, Dict, Iterable, List, Mapping,
                    Optional, Tuple, Union)

import pandas as pd
from absl import logging
from tqdm import tqdm

import jax
from jax.experimental import optimizers
import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

from .train import train_ehr, PatientGRUODEBayesInterface
from .concept import Subject
from .dag import CCSDAG
from .jax_interface import SubjectJAXInterface
from .glove import glove_representation
from .gram import DAGGRAM
from .metrics import (bce, balanced_focal_bce, l1_absolute, l2_squared,
                      code_detectability_by_percentiles)

jax.config.update('jax_platform_name', 'gpu')
logging.set_verbosity(logging.INFO)

def create_patient_interface(processed_mimic_tables_dir: str):
    static_df = pd.read_csv(f'{processed_mimic_tables_dir}/static_df.csv.gz')
    adm_df = pd.read_csv(f'{processed_mimic_tables_dir}/adm_df.csv.gz')
    diag_df = pd.read_csv(f'{processed_mimic_tables_dir}/diag_df.csv.gz',
                          dtype={'ICD9_CODE': str})
    proc_df = pd.read_csv(f'{processed_mimic_tables_dir}/proc_df.csv.gz',
                          dtype={'ICD9_CODE': str})
    test_df = pd.read_csv(f'{processed_mimic_tables_dir}/test_df.csv.gz')

    # Cast columns of dates to datetime64
    static_df['DOB'] = pd.to_datetime(
        static_df.DOB, infer_datetime_format=True).dt.normalize()
    adm_df['ADMITTIME'] = pd.to_datetime(
        adm_df.ADMITTIME, infer_datetime_format=True).dt.normalize()
    adm_df['DISCHTIME'] = pd.to_datetime(
        adm_df.DISCHTIME, infer_datetime_format=True).dt.normalize()
    test_df['DATE'] = pd.to_datetime(
        test_df.DATE, infer_datetime_format=True).dt.normalize()

    patients = Subject.to_list(static_df, adm_df, diag_df, proc_df, test_df)

    # CCS Knowledge Graph
    k_graph = CCSDAG()

    return SubjectJAXInterface(patients, set(test_df.ITEMID), k_graph)


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
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()

    if args.cpu:
        jax.config.update('jax_platform_name', 'cpu')

    num_trials = args.num_trials
    mimic_processed_dir = args.mimic_processed_dir
    output_dir = args.output_dir
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
        return {
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
                trial.suggest_int('proc_vector_size', 50, 150, 50),
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
                    patient_interface.diag_multi_ccs_ancestors_mat,
                },
                'proc': {
                    'ccs_dag':
                    patient_interface.dag,
                    'code2index':
                    patient_interface.proc_multi_ccs_idx,
                    'attention_method':
                    trial.suggest_categorical('attention_method',
                                              ['tanh', 'l2']),
                    'attention_dim':
                    trial.suggest_int('proc_vector_size', 50, 150, 50),
                    'ancestors_mat':
                    patient_interface.proc_multi_ccs_ancestors_mat,
                }
            },
            'model': {
                'ode_dyn':
                'mlp',  # gru, mlp
                'state_size':
                trial.suggest_int('state_size', 100, 500, 50),
                'numeric_hidden_size':
                trial.suggest_int('numeric_hidden_size', 100, 300, 50),
                'bias':
                True
            },
            'training': {
                'batch_size':
                trial.suggest_int('batch_size', 5, 25, 5),
                'epochs':
                100,
                'lr':
                trial.suggest_float('lr', 1e-5, 1e-2, log=True),
                'diag_loss':
                trial.suggest_categorical('diag_loss',
                                          ['balanced_focal', 'bce']),
                'tay_reg':
                3,  # Order of regularized derivative of the dynamics function (None for disable).
                'loss_mixing': {
                    'num_alpha':
                    trial.suggest_float('num_alpha', 1e-3, 1, log=True),
                    'diag_alpha':
                    trial.suggest_float('diag_alpha', 1e-3, 1, log=True),
                    'ode_alpha':
                    trial.suggest_float('ode_alpha', 1e-5, 1, log=True),
                    'l1_reg':
                    trial.suggest_float('l1_reg', 1e-7, 1e-1, log=True),
                    'l2_reg':
                    trial.suggest_float('l2_reg', 1e-6, 1e-1, log=True),
                    'dyn_reg':
                    trial.suggest_float('dyn_reg', 1e-3, 1e3, log=True)
                }
            }
        }

    def objective(trial: optuna.Trial):
        trial_dir = os.path.join(output_dir, f'trial_{trial.number:03d}')
        Path(trial_dir).mkdir(parents=True, exist_ok=True)

        logging.info('[LOADING] Sampling & Initializing Models')
        config = create_config(trial)

        logging.info(f'Trial {trial.number} HPs: {trial.params}')
        diag_glove, proc_glove = glove_representation(**config['glove_config'])
        diag_gram = DAGGRAM(**config['gram_config']['diag'],
                            basic_embeddings=diag_glove)
        proc_gram = DAGGRAM(**config['gram_config']['proc'],
                            basic_embeddings=proc_glove)

        if config['training']['diag_loss'] == 'balanced_focal':
            diag_loss = lambda t, p: balanced_focal_bce(
                t, p, gamma=2, beta=0.999)
        elif config['training']['diag_loss'] == 'bce':
            diag_loss = bce

        ode_model = PatientGRUODEBayesInterface(
            subject_interface=patient_interface,
            diag_gram=diag_gram,
            proc_gram=proc_gram,
            **config['model'],
            tay_reg=config['training']['tay_reg'],
            diag_loss=diag_loss)

        prng_key = jax.random.PRNGKey(rng.randint(0, 100))
        params = ode_model.init_params(prng_key)

        logging.info('[DONE] Sampling & Initializing Models')

        lr = config['training']['lr']
        loss_mixing = config['training']['loss_mixing']
        opt_init, opt_update, get_params = optimizers.adam(step_size=lr)
        opt_state = opt_init(params)

        def loss_fn(params: optimizers.Params, batch: List[int],
                    iteration_text_callback: Any) -> Dict[str, float]:
            res = ode_model(params,
                            batch,
                            count_nfe=False,
                            iteration_text_callback=iteration_text_callback)

            prejump_num_loss = res['prejump_num_loss']
            postjump_num_loss = res['postjump_num_loss']
            prejump_diag_loss = res['prejump_diag_loss']
            postjump_diag_loss = res['postjump_diag_loss']
            l1_loss = l1_absolute(params)
            l2_loss = l2_squared(params)
            dyn_loss = res['dyn_loss']
            num_alpha = loss_mixing['num_alpha']
            diag_alpha = loss_mixing['diag_alpha']
            ode_alpha = loss_mixing['ode_alpha']
            l1_alpha = loss_mixing['l1_reg'] / (res['points_count'])
            l2_alpha = loss_mixing['l2_reg'] / (2 * res['points_count'])
            dyn_alpha = loss_mixing['dyn_reg'] / (res['odeint_weeks'])

            num_loss = (1 - num_alpha
                        ) * prejump_num_loss + num_alpha * postjump_num_loss
            diag_loss = (
                1 - diag_alpha
            ) * prejump_diag_loss + diag_alpha * postjump_diag_loss
            ode_loss = (1 - ode_alpha) * diag_loss + ode_alpha * num_loss
            loss = ode_loss + (l1_alpha * l1_loss) + (l2_alpha * l2_loss) + (
                dyn_alpha * dyn_loss)
            nfe = res['nfe']
            return loss, {
                'loss': {
                    'prejump_num_loss': prejump_num_loss,
                    'postjump_num_loss': postjump_num_loss,
                    'prejump_diag_loss': prejump_diag_loss,
                    'postjump_diag_loss': postjump_diag_loss,
                    'num_loss': num_loss,
                    'diag_loss': diag_loss,
                    'ode_loss': ode_loss,
                    'l1_loss': l1_loss,
                    'l1_loss_per_point': l1_loss / res['points_count'],
                    'l2_loss': l2_loss,
                    'l2_loss_per_point': l2_loss / res['points_count'],
                    'dyn_loss': dyn_loss,
                    'dyn_loss_per_week': dyn_loss / res['odeint_weeks'],
                    'loss': loss
                },
                'stats': {
                    **{
                        name: score.item()
                        for name, score in res['scores'].items()
                    }, 'points_count':
                    res['points_count'],
                    'odeint_weeks_per_point':
                    res['odeint_weeks'] / res['points_count'],
                    'nfe_per_point':
                    nfe / res['points_count'],
                    'nfe_per_week':
                    nfe / res['odeint_weeks'],
                    'nfex1000':
                    nfe / 1000
                },
                'diag_detectability_df': res['diag_detectability_df']
            }

        def update(step: int, batch: Iterable[int],
                   opt_state: optimizers.OptimizerState,
                   iteration_text_callback: Any) -> optimizers.OptimizerState:
            params = get_params(opt_state)
            grads, data = jax.grad(loss_fn,
                                   has_aux=True)(params, batch,
                                                 iteration_text_callback)
            return opt_update(step, grads, opt_state), data

        batch_size = config['training']['batch_size']
        batch_size = min(batch_size, len(train_ids))
        val_batch_size = min(batch_size, len(valid_ids))

        epochs = config['training']['epochs']
        iters = int(epochs * len(train_ids) / batch_size)
        val_pbar = tqdm(total=iters)

        def update_batch_desc(text):
            val_pbar.set_description(text)

        for step in range(iters):
            rng.shuffle(train_ids)
            train_batch = train_ids[:batch_size]
            val_pbar.update(1)

            opt_state, data = update(step, train_batch, opt_state,
                                     update_batch_desc)

            update_batch_desc('')

            if step % eval_freq == 0:
                params = get_params(opt_state)
                _, trn_res = loss_fn(params, train_batch, update_batch_desc)
                _, val_res = loss_fn(params, valid_ids, update_batch_desc)

                score = val_res['stats']['f1-score']
                trial.report(score, step)

                if trial.should_prune():
                    raise optuna.TrialPruned()

                losses = pd.DataFrame(index=trn_res['loss'].keys(),
                                      data={
                                          'Training': trn_res['loss'].values(),
                                          'Validation':
                                          val_res['loss'].values()
                                      })
                stats = pd.DataFrame(index=trn_res['stats'].keys(),
                                     data={
                                         'Training': trn_res['stats'].values(),
                                         'Valdation':
                                         val_res['stats'].values()
                                     })

                detections_trn = code_detectability_by_percentiles(
                    codes_by_percentiles, trn_res['diag_detectability_df'])
                detections_val = code_detectability_by_percentiles(
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

                df_save_prefix = os.path.join(trial_dir, f'step{step:03d}')

                for name, df in {
                        'losses': losses,
                        'stats': stats,
                        'detections_trn_df': detections_trn_df,
                        'detections_val_df': detections_val_df
                }.items():
                    df.to_csv(f'{df_save_prefix}_{name}.csv', index=False)
                    logging.info(df)

            if step % save_freq == 0 and step > 0:
                save_path = os.path.join(trial_dir,
                                         f'step{step:03d}_params.pickle')
                with open(save_path, 'wb') as f:
                    pickle.dump(get_params(opt_state), f)

        return score

    storage = os.path.join(output_dir, 'study.db')
    study = optuna.create_study(study_name='study',
                                direction="maximize",
                                storage=f'sqlite:///{storage}',
                                load_if_exists=True,
                                sampler=TPESampler(),
                                pruner=HyperbandPruner())
    study.optimize(objective, n_trials=num_trials)
