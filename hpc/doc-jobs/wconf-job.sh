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
# $CONFIG_PATH: Path (absolute or relative) to JSON config file for the experiment settings. Example: ~/GP/ICE-NODE/optimal_configs/icenode_v1/icenode_2lr.json
# $CACHE_PATH: Path (absolute or relative) to the cache directory. Example: ~/GP/ICE-NODE/cache
# $OUTPUT_PATH: Path (absolute or relative) to the output directory. Example: ~/GP/ICE-NODE/output
# $DATASET_PATH: Path (absolute or relative) to the dataset directory. Example: ~/GP/ehr-data/dataset
# $OVERRIDE: Override existing configuration. Example: model.emb.dx_discharge=10,model.mem=5

mkdir -p "$WORKDIR" && cd "$WORKDIR" || exit -1

# Clone repository and checkout to the given tag name.
git clone git@github.com:A-Alaa/ICE-NODE.git $WORKDIR/ICENODE --branch $STUDY_TAG --single-branch  --depth 1

cd $WORKDIR/ICENODE

# PostgresQL
source ~/.pgdb-am8520-am8520

source /vol/cuda/11.4.120-cudnn8.2.4/setup.sh




$PY_BIN_DIR/python -m lib.cli.run_config \
--config $CONFIG_PATH \
--output-path $OUTPUT_PATH \
--dataset-path $DATASET_PATH \
--cache-path $CACHE_PATH \
--override $OVERRIDE


