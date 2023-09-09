#!/bin/bash

# Input Environment Variables:
# $STUDY_TAG: Name of the Git branch or tag. If optuna job is executing, the study tag will be used for the optuna study name. Example: v0.2.25
# $CONFIG: Path (absolute or relative) to JSON config file for the experiment settings. Example: ~/GP/ICE-NODE/optimal_configs/icenode_v1/icenode_2lr.json
# $CACHE_PATH: Path (absolute or relative) to the cache directory. Example: ~/GP/ICE-NODE/cache
# $OUTPUT_PATH: Path (absolute or relative) to the output directory. Example: ~/GP/ICE-NODE/output
# $DATASET_PATH: Path (absolute or relative) to the dataset directory. Example: ~/GP/ehr-data/dataset
# $OVERRIDE: Override existing configuration. Example: model.emb.dx=10,model.mem=5

if [[ -v STUDY_TAG ]]; then 
  git clone git@github.com:A-Alaa/ICE-NODE.git --branch $STUDY_TAG --single-branch  --depth 1 ICE-NODE
  cd ICE-NODE
else
  cp ../icenode . -r
  export STUDY_TAG="debug"
fi




export JAX_PLATFORM_NAME="cpu"

MLFLOW_STORE="file://${HOME}/GP/ehr-data/mlflow-store"

$HOME/GP/env/icenode-dev/bin/python -m lib.cli.run_config \
--config $env_config_path \
--output-path $OUTPUT_PATH \
--dataset-path $DATASET_PATH \
--cache-path $CACHE_PATH \
--override $OVERRIDE

