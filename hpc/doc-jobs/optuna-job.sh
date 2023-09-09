#!/usr/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=am8520  # required to send email notifcations - please replace <your_username> with your college login name or email address
#SBATCH --output=/vol/bitbucket/am8520/gpu-job%j.out

TERM=vt100 # or TERM=xterm
STORE=/vol/bitbucket/am8520

WORKDIR=${STORE}/gpu_job_${SLURM_JOB_ID}
export JAX_PLATFORM_NAME="gpu"
export MLFLOW_STORE="file://${STORE}/mlflow-store"

# Input Environment Variables:
# $STUDY_TAG: Name of the Git branch or tag. If optuna job is executing, the study tag will be used for the optuna study name. Example: v0.2.25

mkdir -p "$WORKDIR" && cd "$WORKDIR" || exit -1

# Clone repository and checkout to the given tag name.
git clone git@github.com:A-Alaa/ICE-NODE.git $WORKDIR/ICENODE --branch $STUDY_TAG --single-branch  --depth 1

cd $WORKDIR/ICENODE

# PostgresQL
source ~/.pgdb-am8520-am8520

source /vol/cuda/11.4.120-cudnn8.2.4/setup.sh




$HOME/GP/env/icenode-dev/bin/python -m icenode.cli.optuna_multi \
--output-dir $OUTPUT_DIR \
--dataset $env_data_tag \
--study-tag $STUDY_TAG \
--emb $env_emb \
--model $MODEL \
--optuna-store $OPTUNA_STORE \
--mlflow-store $MLFLOW_STORE \
--trials-time-limit 120 \
--num-processes 1 \
--num-trials 30 \
--training-time-limit 72 \
--job-id "doc-${slurm_job_id}"


