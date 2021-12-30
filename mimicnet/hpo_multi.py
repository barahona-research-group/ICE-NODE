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

    parser.add_argument('--study-tag', required=True)

    parser.add_argument('-N',
                        '--num-processes',
                        type=int,
                        required=True,
                        help='Number of parallel processes.')

    parser.add_argument('--job-id', required=False)

    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    model = args.model
    study_tag = args.study_tag
    store_url = args.store_url
    num_trials = args.num_trials
    mimic_processed_dir = args.mimic_processed_dir
    output_dir = args.output_dir
    cpu = args.cpu
    N = args.num_processes
    job_id = args.job_id or 'unknown'

    study_name = f'{study_tag}_{model}'

    env = dict(os.environ,
               MLFLOW_SQLALCHEMYSTORE_POOL_SIZE="1",
               MLFLOW_SQLALCHEMYSTORE_MAX_OVERFLOW="1",
               MLFLOW_SQLALCHEMYSTORE_NULL_POOL="1")
    cmd = [
        sys.executable, '-m', f'mimicnet.hpo_{model}', '--study-name',
        study_name, '--store-url', store_url, '--output-dir', output_dir,
        '--mimic-processed-dir', mimic_processed_dir, '--num-trials',
        str(num_trials), '--job-id', job_id
    ]
    if cpu:
        cmd.append('--cpu')

    procs = [subprocess.Popen(cmd, env=env) for _ in range(N)]

    for proc in procs:
        proc.wait()
