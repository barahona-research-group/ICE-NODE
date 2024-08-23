#!/usr/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=am8520  # required to send email notifcations - please replace <your_username> with your college login name or email address
#SBATCH --output=/vol/bitbucket/am8520/gpu-job%j.out

TERM=vt100 # or TERM=xterm
STORE=/vol/bitbucket/am8520

WORKDIR=${STORE}/gpu_job_${SLURM_JOB_ID}
export JAX_PLATFORM_NAME="gpu"

mkdir -p "$WORKDIR" && cd "$WORKDIR" || exit -1

# Clone repository and checkout to the given tag name.
git clone git@github.com:A-Alaa/ICE-NODE.git $WORKDIR/ICENODE --branch $STUDY_TAG --single-branch  --depth 1

cd $WORKDIR/ICENODE

source /vol/cuda/12.5.0/setup.sh


$HOME/GP/env/icenode-dev/bin/python -m lib.cli.run_icnn_imputer_training \
--exp $EXP \
--experiments-dir $OUTPUT_PATH \
--dataset-path $DATASET_PATH