#!/bin/bash
set -e

source $HOME/code/CAFA-evaluator/venv/bin/activate

OBOFILE=$HOME/code/APF/Train/go-basic.obo
OBOFILE_PREPPED=$HOME/code/PANDA-3D/$(basename $OBOFILE).prepped
PREDFILE=$HOME/code/PANDA-3D/prediction_filtered_0.6.txt
PREDDIR=$HOME/code/PANDA-3D/predictions
TRUTHFILE=$HOME/code/APF/Train/train_terms.tsv
TRUTHFILE_PREPPED=$HOME/code/PANDA-3D/$(basename $TRUTHFILE).prepped
IAFILE=$HOME/code/APF/IA.txt
RESULTS=$HOME/code/PANDA-3D/results

# prepare obo file
echo "Preparing OBO file..."
python3 $HOME/code/PANDA-3D/prep_obo.py $OBOFILE $OBOFILE_PREPPED

# prepare predictions file
echo "Preparing predictions file..."
cat $PREDFILE | sed -E -e 's/^AF-//g' -e 's/-F1-model_v4//g' > $PREDDIR/$(basename $PREDFILE).prepped

# prepare ground truth file
echo "Preparing ground truth file..."
cat $TRUTHFILE | tail +2 > $TRUTHFILE_PREPPED

# run
mkdir -p $RESULTS

export LD_LIBRARY_PATH=/usr/local/cuda-11.3/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

echo "Running CAFA-eval..."
python3 $HOME/code/CAFA-evaluator/src/cafaeval/__main__.py \
  -out_dir $RESULTS \
  -norm 'pred' \
  -prop 'fill' \
  -th_step 0.05 \
  -max_terms 500 \
  -threads 1 \
  -ia $IAFILE \
  $OBOFILE_PREPPED \
  $PREDDIR \
  $TRUTHFILE_PREPPED

  # $OBOFILE.prepped \
  # $TRUTHFILE.prepped
