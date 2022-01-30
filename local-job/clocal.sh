#!/bin/bash


# Input Environ: STUDY_TAG, DATA_TAG, CONFIG, CONFIG_TAG, MODEL
if [[ -v STUDY_TAG ]]; then 
  git clone git@github.com:A-Alaa/ICENODE.git --branch $STUDY_TAG --single-branch
  cd ICENODE
else
  cp ../icenode ../icenode_configs . -r
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

# Run program
python -m icenode.train_config \
--config $CONFIG \
--config-tag $CONFIG_TAG \
--output-dir $OUTPUT_DIR \
--mimic-processed-dir $DATA_DIR \
--data-tag $DATA_TAG \
--model $MODEL \
