# Neural means and kernel corrections for operator learning

Code and paper for a study of how neural networks and exact kernel methods
combine when learning solution operators and physical forward models. The
same small set of components runs on problems that sit at opposite ends of
the neural-versus-kernel spectrum, and the paper's theory says which end a
given problem is on before anything is fit.

The components: a residual network trained directly on the error metric the
application reports; an exact Matern kernel solve, applied either to the raw
input, to the network's residual, or to the network's learned features; and a
per-coordinate combination of members selected on a validation split.

## Results

**OCO-2 radiative-transfer emulation** (Lamminpää et al., AMT 2025; reduced
atmospheric state to radiance spectrum, three instrument bands). The
reference is the kernel-flow emulator of that paper, scored from its own
stored predictions on the same test states.

| band | reduced metric: theirs / ours | radiance metric: theirs / ours |
|------|------------------------------:|-------------------------------:|
| O2 | 16.9% / **3.8%** | 0.045% / **0.027%** |
| WCO2 | 24.1% / **16.1%** | 0.060% / **0.035%** |
| SCO2 | 16.1% / **8.0%** | 0.115% / **0.043%** |

The winner on the reduced metric is an exact kernel head on the flat-trained
network's features; on the radiance metric, the same head on a
radiance-trained network's features. The same kernel scores 40% on the raw
input: its limitation was the features, not the solve.

**Structural mechanics** (de Hoop, Huang, Qian and Stuart; boundary load to
von Mises stress field). The pipeline reaches **4.55%** relative test error
at 20000 training samples, matching the best published architecture
(PARA-Net) and below FNO (4.76%), PCA-Net (4.67%), DeepONet (5.20%) and the
optimal-recovery kernel (5.18%); in the 1250-sample regime it reaches
**5.38%** against a published best of 6.49%. The paper argues from measured
residual correlations, the spatial structure of the shared error, and
flat scaling in the sample size that the published plateau near 4.5% is a
property of this benchmark's data rather than of any architecture.

**Advection with discontinuous inputs** (same source). A linear member
reaches **11.29%** against the published best of 11.28% — the operator is
linear, and the paper uses the problem as the degenerate case of its
regime analysis.

## Layout

```
paper/            LaTeX source, figures, compiled PDF
common.py         structural mechanics: data, splits, metric
prep_data.py      one-time extraction from the distributed arrays
train_*.py        the neural means (MLP, MSE variant, refiner, FNO, UNet, transformer)
train_krr.py      the kernel baseline; krr_oof.py its out-of-fold fields
gen_preds.py      member predictions with reflection averaging
stack_correct.py  stacking and the residual kernel correction
jpl_data.py       OCO-2: data, metrics, the kernel-flow reference
jpl_pipeline.py   OCO-2: the full comparison, one command per band
analyze_corr.py / ensemble_theory.py / ensemble_uq.py / uq_spectra.py
                  the diversity, stacking-identity and uncertainty analyses
figures.py / fig_corr.py / freeze.py
                  paper figures and the macro freeze
runs/             per-run JSON summaries backing every reported number
docs/reproduce.md every number, with its exact command and hyperparameters
```

## Data

None of the datasets is redistributed here. Structural mechanics and
advection come from the Caltech record `data.caltech.edu/records/20091`; the
OCO-2 files come from the OSF project `osf.io/u2t8a` (JLD2 read with h5py).
`docs/reproduce.md` lists sizes, splits and where each file goes.

## Requirements

Python with NumPy, SciPy, PyTorch and h5py. Everything runs on a laptop CPU;
the largest single computation is a Cholesky factorization of a
19000 x 19000 Gram matrix, about a minute in double precision.

## Citing

Please cite the paper in `paper/` together with the dataset sources: de Hoop,
Huang, Qian and Stuart (2022) and Batlle, Darcy, Hosseini and Owhadi (2024)
for the mechanics and PDE benchmarks, and Lamminpää, Susiluoto, Hobbs,
McDuffie, Braverman and Owhadi (2025) for the OCO-2 emulation problem.
