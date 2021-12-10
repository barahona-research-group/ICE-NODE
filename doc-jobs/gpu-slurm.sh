#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=am8520  # required to send email notifcations - please replace <your_username> with your college login name or email address

TERM=vt100 # or TERM=xterm

#SBATCH --output=/vol/bitbucket/am8520/gpu-job%j.out


conda activate mimic3-snonet

# Clone repository and checkout to the given tag name.
git clone git@github.com:A-Alaa/MIMIC-SNONET.git --branch $EXP_TAG --single-branch

cd MIMIC-SNONET

# Load modules for any applications

# Run program
python -m mimicnet.hpo-multi \
--output-dir $STORE/GP/ehr-data/mimic3-snonet-exp/$EXP_TAG \
--mimic-processed-dir $STORE/GP/ehr-data/mimic3-transforms \
--study-name $EXP_TAG \
--store-url $EXP_STORE_URL \
--num-trials 10 \
-N 1

