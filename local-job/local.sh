
if test -n "${STUDY_TAG-}"; then
  git clone git@github.com:A-Alaa/MIMIC-SNONET.git --branch $STUDY_TAG --single-branch
  STUDY_NAME=${STUDY_TAG}_${MODEL}
else
  cp ../mimicnet . -r
  STUDY_NAME=debug_${MODEL}
fi



OPTUNA_STORE="postgresql://am8520:dirW3?*4<70HSX@db.doc.ic.ac.uk:5432/am8520"
MLFLOW_STORE="file://${HOME}/GP/ehr-data/mlflow-store"

which python

# Run program
python -m mimicnet.train_$MODEL \
--output-dir $HOME/GP/ehr-data/mimic3-snonet-exp/$STUDY_NAME \
--mimic-processed-dir $HOME/GP/ehr-data/mimic3-transforms \
--study-name $STUDY_TAG \
--optuna-store $OPTUNA_STORE \
--mlflow-store $MLFLOW_STORE \
--num-trials $NUM_TRIALS \
--trials-time-limit 24 \
--training-time-limit 12 \
--job-id 0 \
--cpu 

