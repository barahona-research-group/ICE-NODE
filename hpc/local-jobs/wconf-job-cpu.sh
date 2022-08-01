#!/bin/bash

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


export JAX_PLATFORM_NAME="cpu"

MLFLOW_STORE="file://${HOME}/GP/ehr-data/mlflow-store"

$HOME/GP/env/icenode-env/bin/python -m cli.train_app \
--config $CONFIG \
--study-tag $STUDY_TAG \
--config-tag $CONFIG_TAG \
--output-dir $OUTPUT_DIR \
--dataset $DATA_TAG \
--model $MODEL

