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
git clone git@github.com:A-Alaa/ICE-NODE.git $WORKDIR/ICENODE --branch $STUDY_TAG --single-branch  --depth 1

cd $WORKDIR/ICENODE

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



export MLFLOW_STORE="file://${STORE}/mlflow-store"

OUTPUT_DIR=""
DATA_DIR=""

if [[ "$DATA_TAG" == "M3" ]]; then
  OUTPUT_DIR="$HOME/GP/ehr-data/icenode-m3-exp"
  DATA_DIR="$HOME/GP/ehr-data/mimic3-transforms"
else
  OUTPUT_DIR="$HOME/GP/ehr-data/icenode-m4-exp"
  DATA_DIR="$HOME/GP/ehr-data/mimic4-transforms"
fi




$HOME/GP/env/icenode-env/bin/python -m icenode.ehr_predictive.train_app \
--config $CONFIG \
--study-tag $STUDY_TAG \
--config-tag $CONFIG_TAG \
--output-dir $OUTPUT_DIR \
--mimic-processed-dir $DATA_DIR \
--emb $EMB \
--model $MODEL


