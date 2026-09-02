#!/usr/bin/env bash
# Five-member variant of the complete-schedule campaign at every seed: the same trained
# artifacts minus the UNet, through the per-pixel affine stack, the residual correction,
# the global convex stack and the split-conformal calibration. Writes hpix5_* / hstk5_*
# beside the six-member files; nothing that exists is touched. Sequential over seeds; the
# Gram stages hold .gram.lock like every other full-Gram stage on this box.
set -u
cd "$HOME/nmkc2" || exit 1
source "$HOME/nmkc_venv/bin/activate"
VENV="$HOME/nmkc_venv/bin/python"
T="${NMKC_THREADS:-12}"
export CUDA_VISIBLE_DEVICES="" NMKC_ROOT="$HOME/nmkc2" NMKC_DATA="$HOME/nmkc2/data/structmech"
export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T NUMEXPR_NUM_THREADS=$T
echo "b5 start $(date -u +%FT%TZ) threads=$T"
for s in 0 1 2 3 4 5 6 7 8 9; do
  RUNS="$HOME/nmkc2/seeds/sm_s$s/runs"
  export NMKC_RUNS="$RUNS" NMKC_SPLIT_SEED=$s NMKC_PIPE_SEED=$s
  M="mlp_s${s}_w1024_d4_n19000_mir,mlpMSE_s${s}_w1024_d4_n19000_mir,mlpR_s${s}_w1024_d4,fno_s${s}_w64_m14_L4_mir"
  echo "=== seed $s start $(date -u +%FT%TZ) ==="
  if [ ! -s "$RUNS/hpix5.json" ]; then
    (cd code && nice -n 12 "$VENV" stack_perpixel.py --members "$M" --krr 1 --tag hpix5) || echo "seed $s hpix5 FAILED"
  fi
  if [ ! -s "$RUNS/hpix5_corr.json" ]; then
    flock "$HOME/nmkc2/.gram.lock" bash -c "cd code && nice -n 12 $VENV campaign/correct_stack.py --tag hpix5" || echo "seed $s corr FAILED"
  fi
  if [ ! -s "$RUNS/hstk5.json" ]; then
    flock "$HOME/nmkc2/.gram.lock" bash -c "cd code && nice -n 12 $VENV stack_correct.py --members $M --krr 1 --tag hstk5" || echo "seed $s hstk5 FAILED"
  fi
  if [ ! -s "$RUNS/hpix5_uq.json" ]; then
    (cd code && nice -n 12 "$VENV" campaign/uq_conformal.py --tag hpix5) || echo "seed $s uq FAILED"
  fi
  echo "=== seed $s done $(date -u +%FT%TZ) ==="
done
echo "B5_DONE $(date -u +%FT%TZ)"
