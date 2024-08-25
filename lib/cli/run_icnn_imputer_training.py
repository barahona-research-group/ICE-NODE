import argparse
import logging
from collections import defaultdict
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import optax
import pandas as pd
from etils.etree import jax
from tqdm import tqdm

from lib.ml.icnn_modules import ProbStackedICNNImputer, ICNNObsDecoder, ProbICNNImputerTrainer, \
    StandardICNNImputerTrainer
from ..utils import append_params_to_zip

PROP_MODELS = ('ICNN_LN', 'ICNN_NLN', 'ICNN_KL', 'ICNN_NKL', 'ICNN_JSD', 'ICNN_NJSD')
DET_MODELS = ('ICNN_MSE', 'ICNN_NMSE')

EXP_DIR = {
    key: f'snapshots_{key.lower()}' for key in PROP_MODELS + DET_MODELS
}


def experiment_data(dataset_path: str):
    obs_val = pd.read_csv(f'{dataset_path}/missingness_vals.csv', index_col=[0])
    obs_mask = pd.read_csv(f'{dataset_path}/missingness_mask.csv', index_col=[0])
    artificial_mask = pd.read_csv(f'{dataset_path}/missingness_artificial_mask.csv', index_col=[0])
    return obs_val, obs_mask, artificial_mask


def experiment_model(exp: str, observables_size: int):
    pmodels = {
        k: ProbStackedICNNImputer(observables_size=observables_size, state_size=0, optimiser_name='lamb',
                                  max_steps=2 ** 9, lr=1e-2,
                                  positivity='softplus', hidden_size_multiplier=2, depth=5, key=jr.PRNGKey(0))
        for k in PROP_MODELS}
    # pmodels['ICNN_LN'] = ProbStackedICNNImputer(observables_size=observables_size, state_size=0, optimiser_name='lamb',
    #                               max_steps=2 ** 9, lr=1e-2,
    #                               positivity='clipped', hidden_size_multiplier=2, depth=5, key=jr.PRNGKey(0))

    dmodels = {k: ICNNObsDecoder(observables_size=observables_size, state_size=0, optimiser_name='lamb',
                                 max_steps=2 ** 9, lr=1e-2,
                                 positivity='softplus', hidden_size_multiplier=3, depth=5, key=jr.PRNGKey(0))
               for k in DET_MODELS}
    return (pmodels | dmodels)[exp]


def experiment_trainer(e: str):
    return {
        'ICNN_LN': ProbICNNImputerTrainer(loss='log_normal', optimiser_name='adam'),
        'ICNN_NLN': ProbICNNImputerTrainer(loss='log_normal', optimiser_name='adam', loss_feature_normalisation=True),
        'ICNN_KL': ProbICNNImputerTrainer(loss='kl_divergence', optimiser_name='adam', ),
        'ICNN_NKL': ProbICNNImputerTrainer(loss='kl_divergence', optimiser_name='adam',
                                           loss_feature_normalisation=True),
        'ICNN_JSD': ProbICNNImputerTrainer(loss='jsd_gaussian', optimiser_name='adam'),
        'ICNN_NJSD': ProbICNNImputerTrainer(loss='jsd_gaussian', optimiser_name='adam',
                                            loss_feature_normalisation=True),
        'ICNN_MSE': StandardICNNImputerTrainer(optimiser_name='adam'),
        'ICNN_NMSE': StandardICNNImputerTrainer(optimiser_name='adam', loss_feature_normalisation=True)
    }[e]


def run_experiment(exp: str, dataset_path: str, experiments_dir: str):
    obs_val, obs_mask, artificial_mask = experiment_data(dataset_path)
    model = experiment_model(exp, obs_val.shape[1])
    trainer = experiment_trainer(exp)

    split_ratio = 0.7
    seed = 0
    indices = jr.permutation(jr.PRNGKey(seed), len(obs_val))
    train_idx = indices[:int(split_ratio * len(indices))]

    obs_val_train = jnp.array(obs_val.iloc[train_idx].to_numpy())
    obs_mask_train = jnp.array(obs_mask.iloc[train_idx].to_numpy())
    art_mask_train = jnp.array(artificial_mask.iloc[train_idx].to_numpy())
    lr = 1e-3
    steps = 10000
    train_batch_size = 256
    model_snapshot_frequency = 100

    optim = optax.lamb(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    data_train = (obs_val_train, obs_mask_train, art_mask_train)

    train_batches = trainer.dataloader(data_train, train_batch_size, key=jr.PRNGKey(0))
    train_history = defaultdict(list)
    progress = tqdm(range(steps))

    experiment_dir = f'{experiments_dir}/{EXP_DIR[exp]}'
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)

    for step, batch_train in zip(progress, train_batches):
        (train_loss, train_aux), model, opt_state = trainer.make_step(model, optim, opt_state, *batch_train)
        train_nsteps = int(sum(train_aux.n_steps) / len(train_aux.n_steps))
        train_history['loss'].append(train_loss)
        train_history['n_opt_steps'].append(train_nsteps)
        if (step % model_snapshot_frequency) == 0 or step == steps - 1:
            append_params_to_zip(model, f'step{step:04d}.eqx', f'{experiment_dir}/params.zip')

        progress.set_description(f"Trn-L: {train_loss:.3f}")

    train_history = pd.DataFrame(train_history).astype(float)
    train_history.to_csv(f'{experiment_dir}/train_history.csv', index=False)


def run_eval(exp: str, dataset_path: str, experiments_dir: str):
    obs_val, obs_mask, artificial_mask = experiment_data(dataset_path)
    model = experiment_model(exp, obs_val.shape[1])
    split_ratio = 0.7
    seed = 0
    indices = jr.permutation(jr.PRNGKey(seed), len(obs_val))
    train_idx = indices[:int(split_ratio * len(indices))]

    obs_val_test = jnp.array(obs_val.iloc[train_idx].to_numpy())
    art_mask_test = jnp.array(artificial_mask.iloc[train_idx].to_numpy())
    experiment_dir = f'{experiments_dir}/{EXP_DIR[exp]}'

    model = model.load_params_from_archive(f'{experiment_dir}/params.zip', 'step9999.eqx')

    if exp in PROP_MODELS:
        with jax.default_device(jax.devices("cpu")[0]):
            obs_test = jnp.where(art_mask_test, obs_val_test, 0.)
            (X_test_imp, X_test_std), _ = eqx.filter_vmap(model.prob_partial_input_optimise)(obs_test, art_mask_test)

        X_test_imp_df = pd.DataFrame(X_test_imp, columns=obs_val.columns)
        X_test_std_df = pd.DataFrame(X_test_std, columns=obs_val.columns)

        X_test_imp_df.to_csv(f'{experiment_dir}/pred_X_test_imp.csv')
        X_test_std_df.to_csv(f'{experiment_dir}/pred_X_test_std.csv')

    elif exp in DET_MODELS:
        with jax.default_device(jax.devices("cpu")[0]):
            obs_test = jnp.where(art_mask_test, obs_val_test, 0.)
            X_test_imp, _ = eqx.filter_vmap(model.partial_input_optimise)(obs_test, art_mask_test)

        X_test_imp_df = pd.DataFrame(X_test_imp, columns=obs_val.columns)
        X_test_imp_df.to_csv(f'{experiment_dir}/pred_X_test_imp.csv')
    else:
        raise ValueError(f"Unknown model {exp}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--experiments-dir', type=str, required=True)
    parser.add_argument('--dataset-path', type=str, required=True)
    args = parser.parse_args()
    logging.warning(args)
    if args.exp.startswith('eval.'):
        exp = args.exp.split('.')[1]
        run_eval(exp, args.dataset_path, args.experiments_dir)
    else:
        run_experiment(args.exp, args.dataset_path, args.experiments_dir)
