#!/usr/bin/env bash
# First the seeds-against-architectures analysis on the sixty test-prediction arrays, then
# the exact kernel ridge at the learning-curve sizes (deterministic under the fixed carve,
# so one run per size). Both read-only on the campaign trees; outputs under results/ and
# curve/runs/. The kernel stage holds .gram.lock like every other full-Gram stage here.
set -u
cd "$HOME/nmkc2" || exit 1
source "$HOME/nmkc_venv/bin/activate"
VENV="$HOME/nmkc_venv/bin/python"
export CUDA_VISIBLE_DEVICES="" NMKC_ROOT="$HOME/nmkc2" NMKC_DATA="$HOME/nmkc2/data/structmech"
mkdir -p curve/runs logs results
T=5
export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T NUMEXPR_NUM_THREADS=$T
echo "analysis start $(date -u +%FT%TZ)"
NMKC_THREADS=$T nice -n 12 "$VENV" code/campaign/seedarch.py
echo "SEEDARCH_DONE rc=$? $(date -u +%FT%TZ)"
T=4
export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T NUMEXPR_NUM_THREADS=$T NMKC_RUNS="$HOME/nmkc2/curve/runs"
for n in 2000 3000 5000 9000 13000 17000 20000; do
  N=$((n-1000)); f="curve/runs/krrc_matern52_n${N}.json"
  if [ -s "$f" ]; then echo "have $f"; continue; fi
  echo "=== krr ntrain $n start $(date -u +%T)"
  (cd code && flock "$HOME/nmkc2/.gram.lock" nice -n 12 "$VENV" train_krr.py --ntrain "$n" --lowval 1000 --tag krrc --save_pred 0 > "$HOME/nmkc2/logs/curve_krr_n${n}.log" 2>&1)
  echo "=== krr ntrain $n rc=$? $(date -u +%T) $( [ -s "$f" ] && echo OK || echo MISSING)"
done
echo "CURVE_KRR_DONE $(date -u +%FT%TZ)"
