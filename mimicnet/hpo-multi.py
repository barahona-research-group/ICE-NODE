import sys
import subprocess
import argparse
import os
from pathlib import Path


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

    parser.add_argument('-N',
                        '--num-processes',
                        type=int,
                        required=True,
                        help='Number of parallel processes.')



    args = parser.parse_args()

    num_trials = args.num_trials
    mimic_processed_dir = args.mimic_processed_dir
    output_dir = args.output_dir
    N = args.N

    procs = []
    for i in range(N):
        proc = subprocess.Popen(f'{os.environ["HOME"]}/anaconda3/envs/mimic3-snonet/bin/python -m mimicnet.hpo -o {output_dir} -i {mimic_processed_dir} -n {n} --cpu')
        procs.append(proc)

    for proc in procs:
        proc.wait()



