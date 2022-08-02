#!/bin/bash

# Input Environment Variables:
# $STUDY_TAG: Name of the Git branch or tag. If optuna job is executing, the study tag will be used for the optuna study name. Example: v0.2.25
# $DATA_TAG: Name of the data tag. Example: M3
# $EMB: Embedding module label. Example: gram

if [[ -v STUDY_TAG ]]; then 
  git clone git@github.com:A-Alaa/ICE-NODE.git --branch $STUDY_TAG --single-branch  --depth 1 ICE-NODE
  cd ICE-NODE
else
  cp ../icenode . -r
  export STUDY_TAG="debug"
fi

export DATA_DIR="$HOME/GP/ehr-data"
OUTPUT_DIR=""
if [[ "$DATA_TAG" == "M3" ]]; then
  OUTPUT_DIR="$HOME/GP/ehr-data/icenode-m3-exp"
else
  OUTPUT_DIR="$HOME/GP/ehr-data/icenode-m4-exp"
fi


export JAX_PLATFORM_NAME="gpu"

MLFLOW_STORE="file://${HOME}/GP/ehr-data/mlflow-store"

$HOME/GP/env/icenode-dev/bin/python -m icenode.cli.optuna_multi \
--output-dir $OUTPUT_DIR \
--dataset $DATA_TAG \
--study-tag $STUDY_TAG \
--emb $EMB \
--model $MODEL \
--optuna-store $OPTUNA_STORE \
--mlflow-store $MLFLOW_STORE \
--trials-time-limit 120 \
--num-processes 1 \
--num-trials 30 \
--training-time-limit 72 \
--job-id 0

