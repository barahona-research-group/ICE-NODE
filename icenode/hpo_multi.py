import sys
import subprocess
import argparse
import os
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--model',
                        required=True,
                        help='Model label (snonet, snonet_lite, snonet_ds, ..')

    parser.add_argument('-i',
                        '--mimic-processed-dir',
                        required=True,
                        help='Absolute path to MIMIC-III/IV processed tables')

    parser.add_argument('-o',
                        '--output-dir',
                        required=True,
                        help='Aboslute path to log intermediate results')
    parser.add_argument('-n',
                        '--num-trials',
                        type=int,
                        required=True,
                        help='Number of HPO trials.')

    parser.add_argument(
        '-d',
        '--data-tag',
        required=True,
        help='Data identifier tag (m3 for MIMIC-III or m4 for MIMIC-IV')

    parser.add_argument(
        '-e',
        '--emb',
        required=True,
        help=
        'Embedding method to use (matrix|orthogonal_gram|glove_gram|semi_frozen_gram|frozen_gram|tunable_gram)'
    )
    short_tags = {
        'matrix': 'M',
        'orthogonal_gram': 'O',
        'glove_gram': 'G',
        'semi_frozen_gram': 'S',
        'frozen_gram': 'F',
        'tuneble_gram': 'T'
    }

    parser.add_argument(
        '--optuna-store',
        required=True,
        help='Storage URL for optuna records, e.g. for PostgresQL database')

    parser.add_argument(
        '--mlflow-store',
        required=True,
        help='Storage URL for mlflow records, e.g. for PostgresQL database')

    parser.add_argument('--study-tag', required=True)
    parser.add_argument('--trials-time-limit',
                        type=int,
                        required=True,
                        help='Number of maximum hours for all trials')
    parser.add_argument(
        '--training-time-limit',
        type=int,
        required=True,
        help='Number of maximum hours for training in single trial')

    parser.add_argument('-N',
                        '--num-processes',
                        type=int,
                        required=True,
                        help='Number of parallel processes.')

    parser.add_argument('--job-id', required=False)

    parser.add_argument('--pretrained-components', required=False)

    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    model = args.model
    study_tag = args.study_tag
    optuna_store = args.optuna_store
    mlflow_store = args.mlflow_store
    num_trials = args.num_trials
    mimic_processed_dir = args.mimic_processed_dir
    output_dir = args.output_dir
    cpu = args.cpu
    trials_time_limit = args.trials_time_limit
    training_time_limit = args.training_time_limit
    pretrained_components = args.pretrained_components
    emb = args.emb
    data_tag = args.data_tag

    if num_trials > 0:
        N = args.num_processes
    else:
        N = 1

    job_id = args.job_id or 'unknown'

    study_name = f'{study_tag}{data_tag}_{model}_{short_tags[emb]}'
    output_dir = os.path.join(output_dir, study_name)

    env = dict(os.environ)
    cmd = [
        sys.executable, '-m', f'icenode.train_{model}', '--study-name',
        study_name, '--optuna-store', optuna_store, '--mlflow-store',
        mlflow_store, '--output-dir', output_dir, '--mimic-processed-dir',
        mimic_processed_dir, '--data-tag', data_tag, '--emb', emb,
        '--num-trials', str(num_trials), '--trials-time-limit',
        str(trials_time_limit), '--training-time-limit',
        str(training_time_limit), '--job-id', job_id
    ]
    if pretrained_components:
        cmd.extend(['--pretrained-components', pretrained_components])
    if cpu:
        cmd.append('--cpu')

    procs = [subprocess.Popen(cmd, env=env) for _ in range(N)]

    for proc in procs:
        proc.wait()
