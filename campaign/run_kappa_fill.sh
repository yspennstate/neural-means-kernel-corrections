#!/usr/bin/env bash
# Fill the kappa constant (Lemma kappa, Corollary chain) to ten seeds: seeds 3..9, plus seed 0
# repeated as a cross-machine control (kept in results_ctrl/, never collected). Four at a time,
# six threads each, nice 12. eval_kappa serializes its Gram stage on .gram.lock, so the four
# overlap on the rest.
set -u
cd "$HOME/nmkc2" || exit 1
source "$HOME/nmkc_venv/bin/activate"
VENV="$HOME/nmkc_venv/bin/python"
export CUDA_VISIBLE_DEVICES="" NMKC_ROOT="$HOME/nmkc2" NMKC_THREADS=6 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
echo "kappa fill start $(date -u +%FT%TZ)"
run_one() {
  s=$1; tid=$2
  if [ -s "results/$tid.json" ]; then echo "$tid already present, skip"; return; fi
  echo "=== $tid start $(date -u +%FT%TZ) ==="
  NMKC_SEED="$s" TASK_ID="$tid" nice -n 12 "$VENV" code/campaign/eval_kappa.py > "logs/$tid.log" 2>&1
  echo "=== $tid rc=$? $(date -u +%FT%TZ) ($( [ -s results/$tid.json ] && echo OK || echo MISSING))"
}
export -f run_one; export VENV
printf '%s\n' "3 a5_kappa_s3" "4 a5_kappa_s4" "5 a5_kappa_s5" "6 a5_kappa_s6" "7 a5_kappa_s7" "8 a5_kappa_s8" "9 a5_kappa_s9" "0 kappa_ctrl_s0" \
  | xargs -P 4 -L 1 bash -c 'run_one $0 $1'
mv -f results/kappa_ctrl_s0.json results_ctrl/ 2>/dev/null
echo "KAPPA_FILL_DONE $(date -u +%FT%TZ)"
ls -la results/a5_kappa_s*.json results_ctrl/ 2>/dev/null
