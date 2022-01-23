#!/usr/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=am8520  # required to send email notifcations - please replace <your_username> with your college login name or email address
#SBATCH --output=/vol/bitbucket/am8520/gpu-job%j.out

TERM=vt100 # or TERM=xterm
STORE=/vol/bitbucket/am8520

WORKDIR=${STORE}/gpu_job_${SLURM_JOB_ID}


mkdir -p "$WORKDIR" && cd "$WORKDIR" || exit -1

# Clone repository and checkout to the given tag name.
git clone git@github.com:A-Alaa/MIMIC-SNONET.git $WORKDIR/MIMIC-SNONET --branch $STUDY_TAG --single-branch

cd $WORKDIR/MIMIC-SNONET

# PostgresQL
source ~/.pgdb-am8520-am8520

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/vol/bitbucket/am8520/opt/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/vol/bitbucket/am8520/opt/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/vol/bitbucket/am8520/opt/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/vol/bitbucket/am8520/opt/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

source /vol/cuda/11.2.1-cudnn8.1.0.77/setup.sh



export OPTUNA_STORE="postgresql://am8520:dirW3?*4<70HSX@db.doc.ic.ac.uk:5432/am8520"
export MLFLOW_STORE="file://${STORE}/mlflow-store"

# Run program
OUTPUT_DIR=""
DATA_DIR=""

if [[ "$DATA_TAG" == "M3" ]]; then
  OUTPUT_DIR="$STORE/GP/ehr-data/mimicnet-m3-exp"
  DATA_DIR="$STORE/GP/ehr-data/mimic3-transforms"
else
  OUTPUT_DIR="$STORE/GP/ehr-data/mimicnet-m4-exp"
  DATA_DIR="$STORE/GP/ehr-data/mimic4-transforms"
fi

$HOME/anaconda3/envs/mimicnet/bin/python -m mimicnet.hpo_multi \

$STORE/opt/anaconda3/envs/mimic3-snonet/bin/python -m mimicnet.hpo_multi \
--output-dir $OUTPUT_DIR \
--mimic-processed-dir $DATA_DIR \
--study-tag $STUDY_TAG \
--data-tag $DATA_TAG \
--emb $EMB \
--model $MODEL \
--optuna-store $OPTUNA_STORE \
--mlflow-store $MLFLOW_STORE \
--num-trials $NUM_TRIALS \
--trials-time-limit 120 \
--training-time-limit 72 \
--job-id doc-${SLURM_JOB_ID} \
-N 1 \
--pretrained-components mimicnet_configs/pretrained_components_doc.json

