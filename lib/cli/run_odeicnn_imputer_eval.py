import argparse
import logging
import os
from collections import defaultdict
from typing import List, Dict, Any

import jax
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError

from lib.utils import translate_path, load_config
from ..base import Config
from ..db.models import create_tables
from ..ehr.tvx_ehr import SegmentedTVxEHR
from ..ml.evaluation import Evaluation
from ..ml.exp_ode_icnn import AutoODEICNN
from ..ml.icnn_modules import ProbICNNImputerTrainer


def predictions_to_dataframe(obs_columns, predictions):
    predictions_df = []
    meta_df = defaultdict(list)
    for subject_id, subject_predictions in predictions.subject_predictions.items():
        for admission_prediction in subject_predictions:
            if admission_prediction.observables is None:
                continue
            time = admission_prediction.observables.time
            obs = admission_prediction.observables.value
            meta_df['subject_id'].extend([subject_id] * len(time))
            meta_df['admission_id'].extend([admission_prediction.admission.admission_id] * len(time))
            meta_df['time_index'].extend(range(len(time)))
            meta_df['time'].extend(time.tolist())
            predictions_df.append(obs)

    predictions_df = pd.DataFrame(np.vstack(predictions_df), columns=obs_columns)
    meta_df = pd.DataFrame(meta_df)
    return predictions_df, meta_df


def imputations_to_dataframe(obs_columns, predictions):
    predictions_df = []
    meta_df = defaultdict(list)
    for subject_id, subject_predictions in predictions.subject_predictions.items():
        for admission_prediction in subject_predictions:
            if admission_prediction.imputed_observables is None:
                continue
            time = admission_prediction.imputed_observables.time
            obs = admission_prediction.imputed_observables.value
            meta_df['subject_id'].extend([subject_id] * len(time))
            meta_df['admission_id'].extend([admission_prediction.admission.admission_id] * len(time))
            meta_df['time_index'].extend(range(len(time)))
            meta_df['time'].extend(time.tolist())
            predictions_df.append(obs)

    predictions_df = pd.DataFrame(np.vstack(predictions_df), columns=obs_columns)
    meta_df = pd.DataFrame(meta_df)
    return predictions_df, meta_df


def load_data(data_dir: str):
    obs_val1 = pd.read_csv(f'{data_dir}/missingness_vals1.csv', index_col=[0])
    obs_mask1 = pd.read_csv(f'{data_dir}/missingness_mask1.csv', index_col=[0])
    meta1 = pd.read_csv(f'{data_dir}/missingness_meta1.csv', index_col=[0])

    masked_tvx1_sample = SegmentedTVxEHR.load(f'{data_dir}/tvx_aki_phantom.h5')
    sampled_masked_obs_val1 = pd.read_csv(f'{data_dir}/missingness_sampled_masked_val1.csv', index_col=[0])
    sampled_masked_obs_mask1 = pd.read_csv(f'{data_dir}/missingness_sampled_masked_mask1.csv', index_col=[0])
    sampled_masked_meta1 = pd.read_csv(f'{data_dir}/missingness_sampled_masked_meta1.csv', index_col=[0])

    id_columns = ['subject_id', 'admission_id', 'time_index']
    sampled_index = sampled_masked_meta1.set_index(id_columns).index
    sampled_meta1_index = meta1.reset_index().set_index(id_columns).loc[sampled_index].reset_index()['index']
    assert sampled_masked_meta1[id_columns].equals(sampled_masked_meta1[id_columns])
    sampled_mask1 = obs_mask1.loc[sampled_meta1_index].reset_index(drop=True)
    sampled_obs_val1 = obs_val1.loc[sampled_meta1_index].reset_index(drop=True)
    prediction_mask = (1 - sampled_masked_obs_mask1) * sampled_mask1
    features = sampled_masked_obs_val1.columns.tolist()
    n_test_censored = pd.Series(prediction_mask.sum(axis=0), index=sampled_mask1.columns)
    vars_n300 = n_test_censored[n_test_censored >= 300].index
    vars_n300, len(vars_n300)
    return {'masked_tvx1_sample': masked_tvx1_sample,
            'sampled_obs_val1': sampled_obs_val1,
            'sampled_masked_obs_val1': sampled_masked_obs_val1,
            'features': features,
            'vars_n300': vars_n300,
            'n_test_censored': n_test_censored}


def gen_perdictions(features: List[str], model: AutoODEICNN, tvx_ehr: SegmentedTVxEHR):
    preds = model.batch_predict(tvx_ehr, training=False)
    odeicnn_imps_df, odeicnn_meta_df = imputations_to_dataframe(features, preds)
    odeicnn_forcs_df, _ = predictions_to_dataframe(features, preds)
    return {'odeicnn_imps_df': odeicnn_imps_df,
            'odeicnn_meta_df': odeicnn_meta_df,
            'odeicnn_forcs_df': odeicnn_forcs_df, }


def gen_model_stats(prediction_mask: pd.DataFrame,
                    sampled_obs_val1: pd.DataFrame,
                    sampled_masked_obs_val1: pd.DataFrame,
                    predicted_values: pd.DataFrame,
                    vars_n300: pd.Series):
    M_ = prediction_mask.to_numpy().astype(bool)
    Z_ = sampled_obs_val1.to_numpy()

    Z_hat_df = predicted_values
    Z_hat = Z_hat_df.to_numpy()

    # Squared-Errors (per instance)
    X_test_se_ = (Z_hat_df - Z_) ** 2
    X_test_se_ = X_test_se_.where(M_, other=np.nan)
    X_test_se_arr = np.array(X_test_se_.to_numpy())

    X_test_se_melt = pd.melt(X_test_se_, value_vars=list(sampled_masked_obs_val1.columns), value_name='SE')
    X_test_se_melt = X_test_se_melt[X_test_se_melt.SE.notnull()]

    # R2/MSE (per feature)
    features_r2_ = jax.vmap(ProbICNNImputerTrainer.r_squared)(Z_.T, Z_hat.T, M_.T)

    mse_ = np.nanmean(X_test_se_arr, axis=0, where=M_)
    features_stats_df_ = pd.DataFrame({'r2': np.array(features_r2_),
                                       'MSE': mse_,
                                       'Feature': X_test_se_.columns})

    # R2/MSE (per model)
    features_stats_300_df = features_stats_df_[features_stats_df_.Feature.isin(vars_n300)]

    all_models_stats_df = pd.DataFrame(
        {'MSE': [np.nanmean(X_test_se_arr, where=M_)],
         'r2': [ProbICNNImputerTrainer.r_squared(Z_, Z_hat, M_).item()],
         'MICRO-AVG(r2)': [ProbICNNImputerTrainer.r_squared_micro_average(Z_, Z_hat, M_).item()],
         'MACRO-AVG(r2)*': [features_stats_300_df['r2'].mean()]})
    return {
        'all_models_X_test_se': X_test_se_melt,
        'all_models_features_stats_df': features_stats_df_,
        'all_models_stats_df': all_models_stats_df
    }


class ImputerEvaluation(Evaluation):

    @property
    def db_url(self) -> str:
        expr_abs_path = os.path.abspath(self.config.experiments_dir)
        return f'sqlite+pysqlite:///{expr_abs_path}/{self.config.db}'

    def evaluate(self, exp: str, snapshot: str, tvx_ehr: Dict[str, Any]) -> Dict[str, float]:
        data = tvx_ehr
        tvx_ehr = data['masked_tvx1_sample']
        model = self.get_experiment(exp).load_model(tvx_ehr, 0)
        model = model.load_params_from_archive(os.path.join(self.experiment_dir[exp], 'params.zip'), snapshot)
        predictions = gen_perdictions(features=data['features'], model=model, tvx_ehr=tvx_ehr)
        results = gen_model_stats(prediction_mask=data['prediction_mask'],
                                  sampled_obs_val1=data['sampled_obs_val1'],
                                  sampled_masked_obs_val1=data['sampled_masked_obs_val1'],
                                  predicted_values=predictions['odeicnn_forcs_df'],
                                  vars_n300=data['vars_n300'])
        stats = results['all_models_stats_df'].iloc[0].to_dict()
        features_stats = pd.melt(results['all_models_features_stats_df'], value_vars=['MSE', 'r2'],
                                 var_name='metric', id_vars=['Feature'], value_name='value')
        features_stats['metric'] = features_stats['Feature'] + '.' + features_stats['metric'].astype(str)
        features_stats = features_stats.set_index('metric')['value'].to_dict()
        return {**stats, **features_stats}

    def start(self, tvx_ehr_path: str):
        logging.info('Database URL: %s', self.db_url)
        engine = create_engine(self.db_url)
        create_tables(engine)
        tvx = load_data(tvx_ehr_path)

        for exp, snapshot in self.generate_experiment_params_pairs():
            try:
                jax.clear_caches()
                jax.clear_backends()
                logging.info(f'Running {exp}, {snapshot}')
                self.run_evaluation(engine, exp=exp, snapshot=snapshot, tvx_ehr=tvx) # noqa
            except IntegrityError as e:
                logging.warning(f'Possible: evaluation already exists: {exp}, {snapshot}')
                logging.warning(e)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--override', type=str, required=False, default="")
    parser.add_argument('--experiments-dir', type=str, required=True)
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--db', type=str, required=True)

    args = parser.parse_args()

    config = load_config(translate_path(args.config))
    config = Config.from_dict(config)

    logging.warning(args)

    config = config.path_update('experiments_dir', translate_path(args.experiments_dir))
    config = config.path_update('db', args.db)

    if args.override is not None and len(args.override) > 0 and args.override != '0':
        splitter = ','
        if ',' in args.override:
            splitter = ','
        elif ';' in args.override:
            splitter = ';'
        elif '&' in args.override:
            splitter = '&'

        for override in args.override.split(splitter):
            key, value = override.split('=')
            config = config.path_update(key, value)

    ImputerEvaluation(config).start(tvx_ehr_path=args.dataset_path)
