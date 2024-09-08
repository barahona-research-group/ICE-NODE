#!/bin/bash

git clone git@github.com:A-Alaa/ICE-NODE.git --branch $STUDY_TAG --single-branch  --depth 1 ICE-NODE

export JAX_PLATFORM_NAME=cpu
export JAX_ENABLE_X64=True

$HOME/GP/env/icenode-dev/bin/python -m lib.cli.run_eval \
--config $CONFIG_PATH \
--experiments-dir $EXPERIMENTS_DIR \
--dataset-path $DATASET_PATH \
--db $DB \
--override $OVERRIDE
