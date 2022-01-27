#!/usr/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=am8520  # required to send email notifcations - please replace <your_username> with your college login name or email address
#SBATCH --output=/vol/bitbucket/am8520/gpu-job%j.out

TERM=vt100 # or TERM=xterm
STORE=/vol/bitbucket/am8520

# Input Environ: STUDY_TAG, DATA_TAG, CONFIG, CONFIG_TAG, MODEL

WORKDIR=${STORE}/gpu_job_${SLURM_JOB_ID}


mkdir -p "$WORKDIR" && cd "$WORKDIR" || exit -1

# Clone repository and checkout to the given tag name.
git clone git@github.com:A-Alaa/MIMIC-SNONET.git $WORKDIR/MIMIC-SNONET --branch $STUDY_TAG --single-branch

cd $WORKDIR/MIMIC-SNONET

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

$STORE/opt/anaconda3/envs/mimic3-snonet/bin/python -m mimicnet.train_config \
--config $CONFIG \
--config-tag $CONFIG_TAG \
--output-dir $OUTPUT_DIR \
--mimic-processed-dir $DATA_DIR \
--data-tag $DATA_TAG \
--model $MODEL \
