"""."""

import sys
import subprocess
import os

from ...cli.cmd_args import get_cmd_parser, forward_cmd_args
from .optuna_job import cli_args

if __name__ == '__main__':
    args = get_cmd_parser(cli_args + ['--num-processes']).parse_args()

    if args.num_trials > 0:
        N = args.num_processes
    else:
        N = 1

    job_id = args.job_id or 'unknown'
    env = dict(os.environ)
    cmd = [sys.executable, '-m', 'icenode.hyperopt.optuna_job']
    args = forward_cmd_args(args, exclude=['num_processes'])
    procs = [subprocess.Popen(cmd + args, env=env) for _ in range(N)]

    for proc in procs:
        proc.wait()
