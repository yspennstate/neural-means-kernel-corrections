#!/usr/bin/env bash
# Learning curve of the normalized-MSE MLP in the training size, ten seeds, seven sizes,
# under the fixed-carve protocol (the first ntrain rows of the training pool, the last 1000
# of them the validation split), 120 epochs, reflection augmentation and averaging. Each
# (seed, size) job carries its own tag so checkpoints never collide. Writes curve/runs/.
set -u
cd "$HOME/nmkc2" || exit 1
source "$HOME/nmkc_venv/bin/activate"
VENV="$HOME/nmkc_venv/bin/python"
export CUDA_VISIBLE_DEVICES="" NMKC_ROOT="$HOME/nmkc2" NMKC_DATA="$HOME/nmkc2/data/structmech" NMKC_RUNS="$HOME/nmkc2/curve/runs"
mkdir -p curve/runs logs
T=4
export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T NUMEXPR_NUM_THREADS=$T
echo "curve start $(date -u +%FT%TZ)"
one() {
  s=$1; n=$2; N=$((n-1000)); tag="cmse$n"
  f="$HOME/nmkc2/curve/runs/${tag}_s${s}_w1024_d4_n${N}_mir.json"
  if [ -s "$f" ]; then echo "have $f"; return; fi
  echo "=== $tag s$s start $(date -u +%T)"
  (cd code && nice -n 12 "$VENV" train_mlp.py --seed "$s" --mirror 1 --epochs 120 --mse 1 --ntrain "$n" --lowval 1000 --tag "$tag" --threads 4 > "$HOME/nmkc2/logs/curve_${tag}_s${s}.log" 2>&1)
  echo "=== $tag s$s rc=$? $(date -u +%T) $( [ -s "$f" ] && echo OK || echo MISSING)"
}
export -f one; export VENV
# largest sizes first so the long jobs do not trail at the end
for n in 20000 17000 13000 9000 5000 3000 2000; do for s in 0 1 2 3 4 5 6 7 8 9; do echo "$s $n"; done; done \
  | xargs -P 5 -L 1 bash -c 'one $0 $1'
echo "CURVE_MSE_DONE $(date -u +%FT%TZ)"
