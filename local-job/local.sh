cp ../mimicnet . -r

OPTUNA_STORE="postgresql://am8520:dirW3?*4<70HSX@db.doc.ic.ac.uk:5432/am8520"
MLFLOW_STORE="file://${HOME}/GP/ehr-data/mlflow-store"

# Run program
python -m mimicnet.train_$MODEL \
--output-dir $HOME/GP/ehr-data/mimic3-snonet-exp/debug_${MODEL} \
--mimic-processed-dir $HOME/GP/ehr-data/mimic3-transforms \
--study-name debug_${MODEL} \
--optuna-store $OPTUNA_STORE \
--mlflow-store $MLFLOW_STORE \
--num-trials 1 \
--job-id 0 \
--cpu 

