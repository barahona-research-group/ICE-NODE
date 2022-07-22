"""."""

import sys
import subprocess
import os

from ...cli.cmd_args import get_cmd_parser

if __name__ == '__main__':
    args = get_cmd_parser([
        '--model', '--dataset', '--output-dir', '--num-trials', '--emb',
        '--optuna-store', '--mlflow-store', '--study-tag',
        '--trials-time-limit', '--training-time-limit', '--num-processes',
        '--job-id'
    ]).parse_args()

    model = args.model
    study_tag = args.study_tag
    optuna_store = args.optuna_store
    mlflow_store = args.mlflow_store
    num_trials = args.num_trials
    dataset = args.dataset
    output_dir = args.output_dir
    trials_time_limit = args.trials_time_limit
    training_time_limit = args.training_time_limit
    emb = args.emb
    data_tag = args.data_tag

    if num_trials > 0:
        N = args.num_processes
    else:
        N = 1

    job_id = args.job_id or 'unknown'
    env = dict(os.environ)
    cmd = [
        sys.executable, '-m', 'icenode.hyperopt.optuna_job', '--model', model,
        '--study-tag', study_tag, '--optuna-store', optuna_store,
        '--mlflow-store', mlflow_store, '--output-dir', output_dir,
        '--dataset', dataset, '--data-tag', data_tag, '--emb', emb,
        '--num-trials',
        str(num_trials), '--trials-time-limit',
        str(trials_time_limit), '--training-time-limit',
        str(training_time_limit), '--job-id', job_id
    ]

    procs = [subprocess.Popen(cmd, env=env) for _ in range(N)]

    for proc in procs:
        proc.wait()
