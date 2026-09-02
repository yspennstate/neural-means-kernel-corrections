# Reproducing every reported number

The structural-mechanics and OCO-2 campaign ran on CPU-only cloud instances;
development was on a laptop with one GPU, and the trainers will use CUDA if it
is visible, so the device is a property of the machine you run on and not of
the scripts. No run record carries a device field -- do not expect to recover
the device from the artifacts. Kernel solves and error computations are double
precision, network training single precision. Splits are fixed by the seeds
below; every model is selected on the validation split and evaluated once on
the test set.

Seed coverage: ten pipeline seeds back the structural-mechanics members,
stack, correction, second-moment measurement and conformal calibration, in
both structural-mechanics campaigns (the five-member one and the six-member
complete-schedule one), the low-data pipeline, and each of the three OCO-2
bands; ClimSim has five/three/three seeds by training size, and the low-data
kernel-ridge row is a deterministic computation on a fixed split. The
per-member table immediately below is a single seed-0 laptop run kept for
its command lines; its numbers differ from the ten-seed means the paper
reports (for instance FNO 4.70 here against 4.754 +- 0.031 early-stopped and
4.682 +- 0.009 at the complete schedule, MLP 4.86 against 4.836 +- 0.018),
and the paper's values are the campaign ones.

What ships: the per-seed result records (JSON) for every stage of every run,
under `campaign/collected/` and `campaign/{pix4,pix5,ens5,conf5}/`, plus the
scripts that turn them into the paper's tables. The prediction arrays and
network checkpoints are too large to distribute and are NOT included, so
rescoring a new rule on our stored predictions is not possible without
retraining.

## Structural mechanics

Data: the `StructuralMechanics_inputs.npy` / `StructuralMechanics_outputs.npy`
pair from the Caltech record `data.caltech.edu/records/20091` (40000 samples;
the first 20000 are the training pool, the last 20000 the test set). Place them
where `prep_data.py` expects and run it once; it verifies the broadcast
structure exactly and writes the 41-dimensional loads and flattened stress
fields. The validation split is 1000 samples drawn from the training pool by
the fixed permutation in `common.py` (seed 0); the low-data protocol uses the
first 1250 samples with the last 250 as validation.

Members (mean relative L2 on the 20000-sample test set, reflection-averaged):

| member | error | command and configuration |
|--------|------:|---------------------------|
| kernel ridge on loads | 5.19% | `train_krr.py`; Matern-5/2, scale grid {0.5, 0.75, 1, 1.5, 2} x median pairwise distance, nugget grid {1e-8, 1e-6, 1e-4} (scaled by n), exact solve at n = 19000 |
| residual MLP | 4.86% | `train_mlp.py --mirror 1`; width 1024, depth 4, AdamW lr 1e-3, weight decay 1e-5, cosine schedule to 1e-6, 400 epochs, batch 256, reflection augmentation p = 0.5, best-validation checkpoint every 10 epochs |
| MLP, MSE loss | 4.71% | `train_mlp.py --mse 1`; identical except the loss (mean squared error on standardized targets) and 120 epochs |
| kernel-conditioned refiner | 4.73% | `train_mlp_refine.py --epochs 100`; input is the load concatenated with the kernel's predicted field, four-fold out-of-fold on train (`krr_oof.py`), full-train fields at evaluation; 100 epochs, otherwise as the MLP |
| FNO | 4.70% | `train_fno.py --mirror 1`; width 64, 14 modes, 4 spectral layers, batch 256, lr 2e-3, weight decay 1e-6, 200 epochs (best validation at epoch 59) |
| UNet | 4.99% | `train_unet.py --mirror 1`; widths 48/96/192/384 over three scales, batch 256, lr 1.5e-3, weight decay 1e-5; best-validation checkpoint (epoch 50 of a 200-epoch schedule); the campaign value at a complete 100-epoch schedule is 4.740 +- 0.030 over ten seeds |

Pipeline: `gen_preds.py --run <member>` writes train/val/test predictions with
reflection averaging; `stack_correct.py --members <list> --krr 1` fits a global
convex stack and the residual kernel correction (reaching about 4.65%), and
`stack_perpixel.py --members <list> --krr 1 --tag hpix` fits the final
per-pixel affine stack that reaches the reported number. The final surrogate uses
per-coordinate affine stacking (ridge 1e-3, fit on half the validation split
and accepted only because it beat global convex weights on the other half)
followed by the Matern correction of the stacked residual (scale grid
{1, 2, 4} x median distance estimated on 2000 points, nugget grid
{1e-6, 1e-5, 1e-3}, tuned on an 8000-sample subsample, refit on all 19000).
Result: 4.58% after stacking, **4.55%** after correction (`runs/hpix.json`,
`runs/hpix_corr.json`). The low-data pipeline reaches **5.433% +- 0.093%** over ten seeds
(`campaign/collected/*/ld_s*.json`; the single run in `runs/hybLD.json` is
seed 0 at 5.376%).

## OCO-2 radiative-transfer emulation

Data: `dimred_variables_4_mono.jld` and `dimred_data_4_mono.jld` from
`osf.io/u2t8a`, read with h5py, under `data/jpl_oco2/`. Per band the state
dimension is 20 (O2) or 24 (WCO2, SCO2), the reduced radiance has 40
coefficients, and the split is 18000 train / 2000 validation (permutation seed
0) / 2000 test (the file's own test set). The kernel-flow emulator's
predictions on the same test states come from `kf_results_<band>_4_mono.jld`.

One command per band reproduces the comparison table:

    python jpl_pipeline.py --band o2      # likewise wco2, sco2

Configuration, identical across bands: residual MLP width 384, depth 4, AdamW
lr 1e-3, weight decay 1e-5, cosine schedule to 1e-6, 250 epochs, batch 512,
validation every 25 epochs, best checkpoint kept, seed 0. The weighted variant
multiplies the loss residual by s_z (the diagonal radiance metric). The deep
kernel heads standardize the 384-dimensional penultimate features and fit
Matern-5/2 kernel ridge with scale grid {0.5, 1, 2, 4} x the median pairwise
feature distance (6000-point estimate), nugget grid {1e-8, 1e-6, 1e-4}, tuned
against validation on a 6000-sample subsample and refit on all 18000. The
per-coordinate combination picks, for each of the 40 coefficients, the member
with the lowest validation root mean square error; it is the reported model
(on O2 it reaches 4.12% +- 0.05% reduced and 0.0294% +- 0.0020% radiance at
once, over ten seeds at the matched 250-epoch budget).

Results (`runs/jpl_<band>.json`; the kernel-flow rows are computed from the
emulator's own stored predictions):

| band | kernel flow, reduced | ours | kernel flow, radiance | ours |
|------|---------------------:|-----:|----------------------:|-----:|
| O2 | 16.89% | 4.12% +- 0.05% | 0.0448% | 0.0294% +- 0.0020% |
| WCO2 | 24.06% | 16.28% +- 0.06% | 0.0599% | 0.0507% +- 0.0052% |
| SCO2 | 16.14% | 8.08% +- 0.01% | 0.1147% | 0.0584% +- 0.0041% |

Our columns are mean +- standard deviation over ten seeds at a matched
250-epoch budget, aggregated from `campaign/collected/*/oco_<band>_s*.json`
by `campaign/gen_oco_table.py`. A 750-epoch run of the seeded code reaches
3.71% reduced and 0.0174% radiance on O2 and is recorded in
`oco_o2_s900.json`; the table's numbers are at the matched 250-epoch budget.

The seed campaign below replicates the full band pipeline at ten seeds,
adds a network trained in the exact radiance-relative metric (loss computed
through the stored reconstruction, numerator and denominator), and gives the
raw-input kernel rows the identical tuning protocol as the feature heads
(same grid, validation tuning on a 6000-sample subsample, full refit).

## Advection (discontinuous inputs)

This is a supplementary check kept in the repository; it is not part of the
paper. Data: `Advection_inputs.npy` / `Advection_outputs.npy` from the same
Caltech record (200-point binary initial condition to the solution at a later
time; 18000 / 2000 / 20000 split, seed 0). Because the operator is linear, a
ridge-regularized linear member (ridge 1e-2) reaches **11.29%**
(`runs/advection_linear.json`) against the published best of 11.28% (a linear
kernel) and 13.49% for the FNO. Neither the neural means (two seeds, width 512,
200-250 epochs) nor the Matern kernel improves on the linear member: the full
MLP-plus-kernel pipeline reaches **11.35%** (`runs/advection.json`).

## The seed campaign

Every pipeline is replicated at ten seeds, with one seed value threading
every stochastic component: the validation permutation (environment variable
`NMKC_SPLIT_SEED`, read by `common.canonical_split`), member initialization
and batch order, the stacking half-split (`NMKC_PIPE_SEED`), and the kernel
tuning subsamples. Per-seed run directories are isolated through `NMKC_RUNS`
and `NMKC_DATA`, so many seeds share one code tree.

One structural-mechanics seed, end to end (trains six members, stacks,
corrects, calibrates):

    NMKC_ROOT=<root> NMKC_SEED=<s> python campaign/seed_pipeline.py

The first campaign ran that command with its default 200-epoch schedule for
the FNO and the UNet under a 36-hour task limit; the second campaign, on one
40-core host, ran it with `NMKC_EP_FNO=100 NMKC_EP_UNET=100` so that both
convolutional members complete a 100-epoch cosine schedule
(`campaign/run_sm_lane.sh` style lanes; per-seed records under
`campaign/collected/dgx/`). Its companions: `campaign/run_b5.sh` re-runs the
stack, correction, global stack and calibration on the same five members
without the UNet (the paired UNet ablation); `campaign/run_secmom.sh` and
`campaign/secmom6.py` write the second-moment records for the six- and
five-member sets (`campaign/collected/secmom6_seeded.json`,
`secmom5c_seeded.json`); `campaign/run_kappa_fill.sh` completes the
correction-weight constant at ten seeds with a cross-machine control; and
`campaign/runtime_ledger.py` harvests the member costs into
`campaign/collected/dgx/runtime_ledger.csv`. `campaign/seedarch.py` (launched
by `campaign/run_analysis.sh`, which also refits the kernel ridge at the
learning-curve sizes) compares the sixty test-prediction arrays, ten seeds of
six architectures, on a fixed calibration/evaluation split of the test block
and writes `campaign/collected/dgx/seedarch.json`; `campaign/run_curve.sh`
trains the normalized-MSE MLP at seven training sizes and ten seeds under the
fixed-carve protocol (`--ntrain n --lowval 1000`), harvested into
`campaign/collected/dgx/learning_curve.json`.
`campaign/dropone.py` re-solves the pool's convex optima with each
architecture's ten seeds removed, from the released second-moment matrices
(`campaign/collected/dgx/dropone.json`).
`campaign/pool_pipeline.py` runs the pipeline over the pool, each
architecture's ten-seed mean as a member, with every fitted choice made on
the calibration half of the test block and every number read on the
evaluation half (`campaign/collected/dgx/pool_pipeline.json`).
`campaign/verify_floor_and_certificate.py` checks the block floor theorem,
the exchange-rate proposition, the sharpened weight bound with its equality
case, the sharpness and minimax identity of the certified bound and the
signed-stack corollary on synthetic objects and on the released
sixty-predictor matrix. `campaign/collect.py --no-pull` aggregates the
collected records and `campaign/audit_reported_macros.py` recomputes every
macro in `paper/macros.tex` from them.

One OCO-2 band at one seed (three networks including the exact
radiance-metric one, matched raw-kernel rows, deep-kernel heads, both
per-coordinate combinations, per-sample dumps):

    python campaign/jpl_seeded.py --band o2 --seed <s>

One ClimSim scaling point, kernel hyperparameters selected on a held-out
block of the training pool, the test set touched once:

    python campaign/climsim_seeded.py --data train --n <n> --seed <s> --kernel 1

Post-hoc analyses per finished seed directory (tail quantiles and exceedance
curves, the certified fallback rule, the prospective second-moment
scoreboard, error stratified by named state coordinates, the floor bracket
and the certificate constants):

    python campaign/analyze_extras.py --seed-dir <root>/seeds/<dir>

`campaign/verify_synthetic.py` runs the controlled checks of the paper's
theoretical results (exact-norm residual for the certified bound, the
leave-half-out identity, the effective-dimension and Marchenko-Pastur
formulas, the anisotropy rate mechanism); `campaign/eval_kappa.py` measures
the correction-weight norm entering the deployed-surrogate certificate.
Conformal calibration everywhere uses a random subset of the test block that
no selection step touched, evaluated on the disjoint remainder.
