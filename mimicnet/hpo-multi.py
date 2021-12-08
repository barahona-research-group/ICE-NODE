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

    parser.add_argument('--cpu', action='store_true')

    parser.add_argument('-s',
                        '--store-url',
                        required=True,
                        help='Storage URL, e.g. for PostgresQL database')

    parser.add_argument('--study-name',
                        required=True)

    parser.add_argument('-N',
                        '--num-processes',
                        type=int,
                        required=True,
                        help='Number of parallel processes.')


    args = parser.parse_args()


    study_name = args.study_name
    store_url = args.store_url
    num_trials = args.num_trials
    mimic_processed_dir = args.mimic_processed_dir
    output_dir = args.output_dir
    cpu= args.cpu
    N = args.num_processes

    procs = []
    for i in range(N):
        proc = subprocess.Popen([sys.executable,
                                 '-m',
                                 'mimicnet.hpo',
                                 '--study-name',
                                 study_name,
                                 '--store-url',
                                 store_url,
                                 '--output-dir',
                                 output_dir,
                                 '--mimic-processed-dir',
                                 mimic_processed_dir,
                                 '--num-trials',
                                 str(num_trials),
                                 '--cpu' if cpu else ''])
        procs.append(proc)


    for proc in procs:
        proc.wait()



