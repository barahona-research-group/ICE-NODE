#!/bin/bash

# Input Environment Variables:
# $STUDY_TAG: Name of the Git branch or tag. If optuna job is executing, the study tag will be used for the optuna study name. Example: v0.2.25

if [[ -v STUDY_TAG ]]; then 
  git clone git@github.com:A-Alaa/ICE-NODE.git --branch $STUDY_TAG --single-branch  --depth 1 ICE-NODE
  cd ICE-NODE
else
  cp ../icenode . -r
  export STUDY_TAG="debug"
fi




export JAX_PLATFORM_NAME="gpu"

MLFLOW_STORE="file://${HOME}/GP/ehr-data/mlflow-store"

$PY_BIN_DIR/python -m icenode.cli.optuna_multi \
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
--job-id 0

