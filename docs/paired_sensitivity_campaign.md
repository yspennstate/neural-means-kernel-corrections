# Paired implementation and tuning sensitivity

This campaign follows the September 5 manuscript revision. Its observations
are retrospective sensitivity analyses on the existing benchmark splits.
It does not turn a previously inspected test block into a new evaluation set.

## Comparisons fixed before execution

1. Structural mechanics, seeds 0 through 9: recompute four-fold KRR fields
   with fold-local target means, reproduce the archived pooled-centering
   fields, and retrain both refiner variants on the same host with eight CPU
   threads, the same seed, and the same 100-epoch schedule. Reuse the other
   four neural members and full-data KRR validation/test predictions.
   Rebuild the validation-selected per-pixel stack and its correction. If
   the per-pixel candidate is declined, use the recorded global-stack
   fallback. Recompute the corresponding conformal measurements.
2. Structural mechanics, seeds 0 through 9: freeze the archived per-pixel
   mean and selected correction kernel. Evaluate the full-data KRR channel
   on training inputs, rebuild the refiner and stack at those inputs, and
   measure the propagated difference between the recorded residual labels
   and residuals of the inference-time mean. Compare both corrections
   without kernel retuning or neural retraining.
3. OCO-2, seeds 0, 1, and 2 in each of O2, WCO2, and SCO2: train the three
   original networks once per band/seed, retaining their checkpoints and
   feature arrays. Compare the recorded kernel grid with a larger grid
   on these same frozen features and splits. Apply each grid equally to
   the raw-input, ARD, and three feature kernels. Retain ridge-readout
   controls. Choose kernels and coordinatewise combinations on validation.

The recorded OCO-2 grid is scales `0.5,1,2,4` and nuggets
`1e-8,1e-6,1e-4`. The expanded grid is scales
`0.25,0.5,1,2,4,8,16` and nuggets
`1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3`.
Every failed or nonfinite tuning cell and every selected boundary value is
recorded. Neither grid is claimed to locate a global hyperparameter optimum.

## Execution

For a reconstruction from the public dataset, `krr_oof.py` defaults to
`--target-centering fold-local`. Pass `--target-centering pooled` explicitly
to reproduce the historical target-mean convention. Both modes write the
choice and the field hash to `krr_oof_train.json`. The paired campaign below
uses retained historical runs so that unchanged members need not be trained
again; its executed sources are frozen separately from later documentation
and reproducibility improvements in this repository.

Use Linux and the campaign's locked Python environment. Prepare a new root
with this repository's Python sources in `code/`. The historical root has
`data/structmech/` and `seeds/sm_s0/runs/` through `sm_s9/runs/`.

```sh
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=
nice -n 15 python -B campaign/run_centering_campaign.py \
  --root /path/to/new_campaign --historical-root /path/to/historical_campaign \
  --data /path/to/historical_campaign/data/structmech --threads 8
```

The controller also accepts the verified output of an initial single-seed
`fold_centering_sensitivity.py` run. It checks the seed, source, historical
field hash, and corrected field hash before adopting that output.

Place `correction_mismatch.py`, `jpl_grid_sensitivity.py`, and
`run_paper1_followups.py` in the new root's `diagnostics/` directory. Then:

```sh
nice -n 15 python -B /path/to/new_campaign/diagnostics/run_paper1_followups.py \
  --root /path/to/new_campaign --historical-root /path/to/historical_campaign \
  --jpl-data /path/to/jpl_oco2
```

The follow-up controller waits for successful completion of the centering
campaign. Both controllers share one active-job lock and an eight-thread
ceiling. There are ten centering seed pairs and a fixed list of nineteen
follow-up jobs; neither controller adds jobs at runtime. Each stage requires
eight idle core equivalents, 32 GiB available memory, and 40 GiB free disk
at admission. This does not reserve capacity against uncoordinated users of
the host; live workload monitoring remains necessary.

At 10:47 local time on September 5, additional idle capacity became available.
The waiting follow-up controller was stopped before its first job, and
`run_paper1_followups_parallel.py` started the identical nineteen-job list
in a separate eight-thread lane. The original centering lane and its
experiment sources were unchanged. The total cap became sixteen threads,
with at most one job in each lane. Follow-up admission measures current CPU
use, reserves eight threads for centering even between its stages, and leaves
one core of headroom. The transition is recorded in `scheduling_amendment.json`;
the new controller is retained under `tools/` in the evidence archive.

## Evidence and interpretation

The campaign writes source and input hashes, stage receipts, immutable
per-case errors, chosen branches, and paired seed summaries. Existing
historical artifacts are read through symlinks and never overwritten.
Unreceipted partial output or a changed input stops a resume for inspection.

The centering calculation uses a rank-one identity checked against a
separate solve. The mismatch calculation independently reproduces the
stored KRR, refiner, stack, and correction before reporting a contrast.
Its propagated norm is measured only on the specified query sets; it is
not a bound over all possible inputs. The grid driver retains the
original network-training and ridge routines; the kernel head was checked
against the original on the same numerical inputs, with a known-bad
all-failed grid as a failure control.

Treat repeated seeds and per-case paired errors according to their actual
dependence. In particular, seeds share the benchmark test block. Report
signed differences and uncertainty without selecting seeds, methods, or
claims by favorable test outcomes. No performance result is established
merely by the presence of these scripts.
