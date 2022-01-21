if test -n "${STUDY_TAG-}"; then 
  git clone git@github.com:A-Alaa/MIMIC-SNONET.git --branch $STUDY_TAG --single-branch
  cd MIMIC-SNONET
else
  cp ../mimicnet ../mimicnet_configs . -r
  export STUDY_TAG="debug"
fi

OPTUNA_STORE="postgresql://am8520:dirW3?*4<70HSX@db.doc.ic.ac.uk:5432/am8520"
MLFLOW_STORE="file://${HOME}/GP/ehr-data/mlflow-store"

# Run program
python -m mimicnet.hpo_multi \
--output-dir $HOME/GP/ehr-data/mimic3-snonet-exp \
--mimic-processed-dir $HOME/GP/ehr-data/mimic3-transforms \
--study-tag $STUDY_TAG \
--data-tag $DATA_TAG \
--emb $EMB \
--model $MODEL \
--optuna-store $OPTUNA_STORE \
--mlflow-store $MLFLOW_STORE \
--num-trials $NUM_TRIALS \
--trials-time-limit 96 \
--training-time-limit 48 \
--job-id 0 \
-N 1 \
--pretrained-components mimicnet_configs/pretrained_components_local.json

