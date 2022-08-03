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
# $DATA_TAG: Name of the data tag. Example: M3
# $CONFIG: Path (absolute or relative) to JSON config file for the experiment settings. Example: ~/GP/ICE-NODE/optimal_configs/icenode_v1/icenode_2lr.json
# $CONFIG_TAG: Textual tag attributing the config JSON file, to be used for the experiment output directory name. Example: Cv0.2.25
# $MODEL: Model label to be used in the experiment. Example: dx_icenode_2lr

mkdir -p "$WORKDIR" && cd "$WORKDIR" || exit -1

# Clone repository and checkout to the given tag name.
git clone git@github.com:A-Alaa/ICE-NODE.git $WORKDIR/ICENODE --branch $STUDY_TAG --single-branch  --depth 1

cd $WORKDIR/ICENODE

# PostgresQL
source ~/.pgdb-am8520-am8520

source /vol/cuda/11.4.120-cudnn8.2.4/setup.sh

export DATA_DIR="$HOME/GP/ehr-data"
OUTPUT_DIR=""
if [[ "$DATA_TAG" == "M3" ]]; then
  OUTPUT_DIR="$HOME/GP/ehr-data/icenode-m3-exp"
else
  OUTPUT_DIR="$HOME/GP/ehr-data/icenode-m4-exp"
fi


$HOME/GP/env/icenode-dev/bin/python -m icenode.cli.train_app \
--config $CONFIG \
--study-tag $STUDY_TAG \
--config-tag $CONFIG_TAG \
--output-dir $OUTPUT_DIR \
--dataset $DATA_TAG \
--model $MODEL


