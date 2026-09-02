#!/usr/bin/env bash
# Second-moment records for the complete-schedule campaign: six members, then the same
# artifacts without the UNet. Read-only on the run directories; writes results/*.json.
set -u
cd "$HOME/nmkc2" || exit 1
source "$HOME/nmkc_venv/bin/activate"
VENV="$HOME/nmkc_venv/bin/python"
T="${NMKC_THREADS:-6}"
export CUDA_VISIBLE_DEVICES="" NMKC_ROOT="$HOME/nmkc2" NMKC_DATA="$HOME/nmkc2/data/structmech"
export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T NUMEXPR_NUM_THREADS=$T
echo "secmom start $(date -u +%FT%TZ)"
NMKC_MEMBERS=mlp,mlpMSE,mlpR,fno,unet,krr NMKC_SECMOM_OUT="$HOME/nmkc2/results/secmom6_seeded.json" \
  nice -n 12 "$VENV" code/campaign/secmom6.py
echo "six-member done $(date -u +%FT%TZ)"
NMKC_MEMBERS=mlp,mlpMSE,mlpR,fno,krr NMKC_SECMOM_OUT="$HOME/nmkc2/results/secmom5c_seeded.json" \
  nice -n 12 "$VENV" code/campaign/secmom6.py
echo "SECMOM_DONE $(date -u +%FT%TZ)"
