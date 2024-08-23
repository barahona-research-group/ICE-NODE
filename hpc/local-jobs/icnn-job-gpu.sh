#!/bin/bash

if [[ -v STUDY_TAG ]]; then 
  git clone git@github.com:A-Alaa/ICE-NODE.git --branch $STUDY_TAG --single-branch  --depth 1 ICE-NODE
  cd ICE-NODE
else
  cp ../icenode . -r
  export STUDY_TAG="debug"
fi

export JAX_PLATFORM_NAME="gpu"

$HOME/GP/env/icenode-dev/bin/python -m lib.run_icnn_imputer_training.run_config \
--exp $EXP \
--experiments-dir $OUTPUT_PATH \
--dataset-path $DATASET_PATH
