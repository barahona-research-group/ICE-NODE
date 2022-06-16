#!/bin/bash

if [[ -v STUDY_TAG ]]; then 
  git clone git@github.com:A-Alaa/ICE-NODE.git --branch $STUDY_TAG --single-branch  --depth 1 ICE-NODE
  cd ICE-NODE
else
  cp ../icenode . -r
  export STUDY_TAG="debug"
fi

OUTPUT_DIR=""
DATA_DIR=""

if [[ "$DATA_TAG" == "M3" ]]; then
  OUTPUT_DIR="$HOME/GP/ehr-data/icenode-m3-exp"
  DATA_DIR="$HOME/GP/ehr-data/mimic3-transforms"
else
  OUTPUT_DIR="$HOME/GP/ehr-data/icenode-m4-exp"
  DATA_DIR="$HOME/GP/ehr-data/mimic4-transforms"
fi




export JAX_PLATFORM_NAME=gpu

MLFLOW_STORE="file://${HOME}/GP/ehr-data/mlflow-store"

$HOME/GP/env/icenode-env/bin/python -m icenode.ehr_predictive.train_app \
--config $CONFIG \
--study-tag $STUDY_TAG \
--config-tag $CONFIG_TAG \
--output-dir $OUTPUT_DIR \
--mimic-processed-dir $DATA_DIR \
--model $MODEL

