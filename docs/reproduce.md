# Reproducing every reported number

All runs are CPU-only, double precision for kernel solves, single precision for
network training. Splits are fixed by the seeds below; every model is selected
on the validation split and evaluated once on the test set.

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
| kernel-conditioned refiner | 4.73% | `train_mlp_refine.py`; input is the load concatenated with the kernel's predicted field, four-fold out-of-fold on train (`krr_oof.py`), full-train fields at evaluation; 300 epochs, otherwise as the MLP |
| FNO | 4.70% | `train_fno.py --mirror 1`; width 64, 14 modes, 4 spectral layers, batch 256, lr 2e-3, weight decay 1e-6, 200 epochs (best validation at epoch 59) |
| UNet | 4.99% | `train_unet.py --mirror 1`; widths 48/96/192/384 over three scales, batch 256, lr 1.5e-3, weight decay 1e-5; best-validation checkpoint (epoch 50 of a 200-epoch schedule) |

Pipeline: `gen_preds.py --run <member>` writes train/val/test predictions with
reflection averaging; `stack_correct.py --members <list> --krr 1` fits the
stack and the residual kernel correction. The final surrogate uses
per-coordinate affine stacking (ridge 1e-3, fit on half the validation split
and accepted only because it beat global convex weights on the other half)
followed by the Matern correction of the stacked residual (scale grid
{1, 2, 4} x median distance estimated on 2000 points, nugget grid
{1e-6, 1e-5, 1e-3}, tuned on an 8000-sample subsample, refit on all 19000).
Result: 4.58% after stacking, **4.55%** after correction (`runs/hpix.json`,
`runs/hpix_corr.json`). The low-data pipeline reaches **5.38%**
(`runs/hybLD.json`).

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
(on O2 it reaches 3.83% reduced and 0.0267% radiance at once).

Results (`runs/jpl_<band>.json`; the kernel-flow rows are computed from the
emulator's own stored predictions):

| band | kernel flow, reduced | ours | kernel flow, radiance | ours |
|------|---------------------:|-----:|----------------------:|-----:|
| O2 | 16.89% | 3.83% | 0.0448% | 0.0267% |
| WCO2 | 24.06% | 16.12% | 0.0599% | 0.0350% |
| SCO2 | 16.14% | 7.96% | 0.1147% | 0.0432% |

Seed ensembles (seeds 1-3 per class, identical configuration) shift these
numbers by at most a few hundredths of a point; the reported values are the
seed-0 runs, matching the configuration above.

## Advection (discontinuous inputs)

Data: `Advection_inputs.npy` / `Advection_outputs.npy` from the same Caltech
record (200-point binary initial condition to the solution at a later time;
18000 / 2000 / 20000 split, seed 0). Because the operator is linear, a
ridge-regularized linear member (ridge 1e-2) reaches **11.29%** against the
published best of 11.28% (a linear kernel) and 13.49% for the FNO; neural
means (two seeds, width 512, 200-250 epochs, configuration as above) and the
kernel add nothing, which the paper discusses. See `runs/` for the summary.
