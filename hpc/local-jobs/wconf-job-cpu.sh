#!/bin/bash

# Input Environment Variables:
# $STUDY_TAG: Name of the Git branch or tag. If optuna job is executing, the study tag will be used for the optuna study name. Example: v0.2.25
# $DATA_TAG: Name of the data tag. Example: M3
# $CONFIG: Path (absolute or relative) to JSON config file for the experiment settings. Example: ~/GP/ICE-NODE/optimal_configs/icenode_v1/icenode_2lr.json
# $CONFIG_TAG: Textual tag attributing the config JSON file, to be used for the experiment output directory name. Example: Cv0.2.25
# $MODEL: Model label to be used in the experiment. Example: dx_icenode_2lr

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

$HOME/GP/env/icenode-dev/bin/python -m icenode.cli.train_app \
--config $CONFIG \
--study-tag $STUDY_TAG \
--config-tag $CONFIG_TAG \
--output-dir $OUTPUT_DIR \
--dataset $DATA_TAG \
--model $MODEL

