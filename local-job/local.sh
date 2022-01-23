#!/bin/bash

if [[ -v STUDY_TAG ]]; then 
  git clone git@github.com:A-Alaa/MIMIC-SNONET.git --branch $STUDY_TAG --single-branch
  cd MIMIC-SNONET
else
  cp ../mimicnet ../mimicnet_configs . -r
  export STUDY_TAG="debug"
fi

OUTPUT_DIR=""
DATA_DIR=""

if [[ "$DATA_TAG" == "M3" ]]; then
  OUTPUT_DIR="$HOME/GP/ehr-data/mimicnet-m3-exp"
  DATA_DIR="$HOME/GP/ehr-data/mimic3-transforms"
else
  OUTPUT_DIR="$HOME/GP/ehr-data/mimicnet-m4-exp"
  DATA_DIR="$HOME/GP/ehr-data/mimic4-transforms"
fi


OPTUNA_STORE="postgresql://am8520:dirW3?*4<70HSX@db.doc.ic.ac.uk:5432/am8520"
MLFLOW_STORE="file://${HOME}/GP/ehr-data/mlflow-store"

# Run program
python -m mimicnet.hpo_multi \
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
--job-id 0 \
-N 1 \
--pretrained-components mimicnet_configs/pretrained_components_local.json

