WORKDIR="$PWD/tmp"

rm -rf $WORKDIR

mkdir -p "$WORKDIR" && cd "$WORKDIR" || exit -1

# Clone repository and checkout to the given tag name.
git clone git@github.com:A-Alaa/MIMIC-SNONET.git $WORKDIR/MIMIC-SNONET --branch $STUDY_TAG --single-branch

cd $WORKDIR/MIMIC-SNONET

OPTUNA_STORE="postgresql://am8520:dirW3?*4<70HSX@db.doc.ic.ac.uk:5432/am8520"
MLFLOW_STORE="file://${HOME}/GP/ehr-data/mlflow-store"

# Run program
python -m mimicnet.train_$MODEL \
--output-dir $HOME/GP/ehr-data/mimic3-snonet-exp/${STUDY_TAG}_${MODEL} \
--mimic-processed-dir $HOME/GP/ehr-data/mimic3-transforms \
--study-tag $STUDY_TAG \
--model $MODEL \
--optuna-store $OPTUNA_STORE \
--mlflow-store $MLFLOW_STORE \
--num-trials 1 \
--job-id 0 \
--cpu 

