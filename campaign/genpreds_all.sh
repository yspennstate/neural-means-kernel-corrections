#!/bin/bash
# Generate val/test/train predictions for every finalised member of every seed.
#
# THREE ENV VARS, ALL MANDATORY. Missing any one fails LATE and quietly:
#   NMKC_DATA       -> the dataset (default code/data is absent)
#   NMKC_RUNS       -> the SEED's runs dir (default code/runs is empty)
#   NMKC_SPLIT_SEED -> THE SEED NUMBER. Each seed has its OWN train/val partition
#                      (campaign/analyze_extras.py:251 sets it from the run name).
#
# The third one is the one that bit. Omitting it writes every seed's predictions
# against split 0, while the KRR arrays the campaign wrote are on split s. Mixing them
# in an ensemble gave equal-weight val ~0.46 against members at ~0.047, and it looked
# CORRECT on seed 0 - because seed 0's split IS split 0. One seed agreeing is not a
# check; the seed whose split differs is the check.
#
# THE INTERPRETER IS DETECTED, NOT ASSUMED - box-01 venvs/main, box-02
# claude-jpl-xl2/venv, box-03 venvs/jpl. The probe covers gen_preds' full import set.
#
# FORCE=1 regenerates files that already exist (needed after the split-seed fix).
set -u
ROOT=/srv/aiwork/nmkc10seed
cd $ROOT/code
export NMKC_DATA=$ROOT/data/structmech
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2

PY=""
for c in /srv/aiwork/venvs/main/bin/python \
         /srv/aiwork/claude-jpl-xl2/venv/bin/python \
         /srv/aiwork/venvs/jpl/bin/python; do
  [ -x "$c" ] || continue
  if "$c" -c "import torch, numpy, json, argparse; import common; import models" >/dev/null 2>&1; then
    PY="$c"; break
  fi
done
if [ -z "$PY" ]; then
  echo "NO INTERPRETER ON $(hostname) SATISFIES gen_preds' IMPORT SET - refusing"
  exit 2
fi
echo "using $PY on $(hostname)"

for sd in $ROOT/seeds/sm_s*/; do
  s=$(basename $sd)
  [ "$s" = "sm_s99" ] && continue
  SEEDNUM=${s#sm_s}
  R=$sd/runs
  [ -d "$R" ] || continue
  for j in $R/*.json; do
    [ -e "$j" ] || continue
    n=$(basename $j .json)
    case "$n" in
      krr*) echo "  $s $n: SKIP (krr preds already on disk, written on split $SEEDNUM)"; continue ;;
    esac
    if [ -f "$R/${n}_predva.npy" ] && [ "${FORCE:-0}" != "1" ]; then
      echo "  $s $n: already done"
      continue
    fi
    echo "  $s $n: generating on split $SEEDNUM..."
    NMKC_SPLIT_SEED=$SEEDNUM NMKC_RUNS=$R nice -n 15 "$PY" gen_preds.py --run "$n" --cpu \
        > $ROOT/logs/genpred_${s}_${n}.log 2>&1
    if [ -f "$R/${n}_predva.npy" ]; then
      echo "      OK  $(grep 'rel-L2' $ROOT/logs/genpred_${s}_${n}.log | tail -1)"
    else
      echo "      FAILED: $(tail -1 $ROOT/logs/genpred_${s}_${n}.log)"
    fi
  done
done
echo "GENPREDS DONE on $(hostname)"
