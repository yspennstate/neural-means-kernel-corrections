# Neural means and kernel corrections for operator learning

Code and paper for a study of how neural networks and exact kernel methods
combine when learning solution operators and physical forward models. The
same small set of components runs on problems that sit at opposite ends of
the neural-versus-kernel spectrum, and the paper's theory says which end a
given problem is on from the members' measured residual correlations, before
any stack is fit.

The components: a residual network trained directly on the error metric the
application reports; an exact Matern kernel solve, applied either to the raw
input, to the network's residual, or to the network's learned features; and a
per-coordinate combination of members selected on a validation split.

## Paper and supplement

`paper/main.pdf` is the manuscript: 25 pages of text on the two problems, the
method, and the results, in JMLR format. `paper/supplement.pdf` is the online
supplement it refers to as S1–S8, and holds the material a reader needs only
if they want to check something: the finite-sample bounds behind every fitted
stage and all the proofs, the margin and per-coordinate selection results, the
effective-dimension identity, the ablation and error-localization measurements,
the spectra, the uncertainty-quantification study, the cost accounting, the
uncontrolled survey of the rest of the suite, the mirror-symmetry check, the
OCO-2 input-metric study with the data-scaling and ClimSim results, the
implementation and compute record, and the numerical verification of every
stated result. The two build from one set of sources and one bibliography:

```
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex supplement && bibtex supplement && pdflatex supplement && pdflatex supplement
```

Run the pair twice if you have changed a cross-reference: each document reads
the other's `.aux` through `xr`, so a reference that crosses between them
settles on the second pass. The JMLR style file is included.

## Results

**OCO-2 radiative-transfer emulation** (Lamminpää et al., AMT 2025; reduced
atmospheric state to radiance spectrum, three instrument bands). The
reference is the kernel-flow emulator of that paper, scored from its own
stored predictions on the same test states.

| band | reduced metric: theirs / ours | radiance metric: theirs / ours |
|------|------------------------------:|-------------------------------:|
| O2 | 16.89% / **4.12% +- 0.05%** | 0.0448% / **0.0294% +- 0.0020%** |
| WCO2 | 24.06% / **16.28% +- 0.06%** | 0.0599% / **0.0507% +- 0.0052%** |
| SCO2 | 16.14% / **8.08% +- 0.01%** | 0.1147% / **0.0584% +- 0.0041%** |

Our columns are mean +- standard deviation over ten seeds per band at a
matched 250-epoch budget. On all three bands the "ours" entries are one
model, a per-coordinate combination that beats the emulator on both metrics
at once -- at ten of ten seeds on O2 and SCO2 and nine of ten on WCO2.
Earlier versions of this README reported single-seed numbers from a longer
training budget (O2 3.8% / 0.027%); see docs/reproduce.md for how the two
budgets relate. WCO2 carries the thin margin: the radiance win there is nine
of ten seeds, the exception missing by 0.0015 points. The same kernel scores
40% on the raw input: its limitation was the features, not the solve.

**Structural mechanics** (de Hoop, Huang, Qian and Stuart; boundary load to
von Mises stress field). The pipeline reaches **4.572% +- 0.010%** relative
test error over ten seeds, level with the best published architecture
(PARA-Net, 4.55%) rather than beating it -- the difference is about one
standard error -- and below FNO (4.76%), PCA-Net (4.67%), DeepONet (5.20%)
and the optimal-recovery kernel (5.18%); in the 1250-sample regime it reaches
**5.433% +- 0.093%** over ten seeds against a published best of 6.49%. The paper argues from measured
residual correlations, the spatial structure of the shared error, and
flat scaling in the sample size that the published plateau near 4.5% is a
property of this benchmark's data rather than of any architecture.

**Advection with discontinuous inputs** (same source; a supplementary check
in this repository, not in the paper). A ridge-linear member reaches
**11.29%** (`runs/advection_linear.json`), matching the published
linear-kernel baseline of 11.28%; the operator is linear, so neither the
neural mean nor the Matern kernel improves on it — the full MLP-plus-kernel
pipeline reaches 11.35% (`runs/advection.json`). It is the degenerate,
linear end of the same regime picture.

## Layout

```
paper/            LaTeX source, figures, and both compiled PDFs: main.pdf
                  (the manuscript) and supplement.pdf (the online supplement)
common.py         structural mechanics: data, splits, metric
prep_data.py      one-time extraction from the distributed arrays
train_*.py        the neural means (MLP, MSE variant, refiner, FNO, UNet, transformer)
train_krr.py      the kernel baseline; krr_oof.py its out-of-fold fields
gen_preds.py      member predictions with reflection averaging
stack_correct.py  global stacking and the residual kernel correction
stack_perpixel.py the per-pixel affine stack (the reported pipeline)
jpl_data.py       OCO-2: data, metrics, the kernel-flow reference
jpl_pipeline.py   OCO-2: the full comparison, one command per band
climsim_scaling.py  the ClimSim data-scaling sweep
campaign/         the seed campaign: per-seed pipelines, matched kernel
                  tuning, conformal calibration, verification suite, analyses
analyze_corr.py / ensemble_theory.py / ensemble_uq.py / uq_spectra.py
                  the diversity, stacking-identity and uncertainty analyses
figures.py / fig_corr.py / fig_floor.py / freeze.py
                  paper figures and the macro freeze
runs/             per-run JSON summaries backing every reported number
docs/reproduce.md every number, with its exact command and hyperparameters
```

## Data

None of the datasets is redistributed here. Structural mechanics and
advection come from the Caltech record `data.caltech.edu/records/20091`; the
OCO-2 files come from the OSF project `osf.io/u2t8a` (JLD2 read with h5py);
the ClimSim arrays are the LEAP subsampled low-resolution set
(`huggingface.co/datasets/LEAP/subsampled_low_res`, the four files
`{train,val}_{input,target}.npy`; `campaign/get_climsim.sh` downloads and
shape-checks them). `docs/reproduce.md` lists sizes, splits and where each
file goes.

## Requirements

Python with NumPy, SciPy, PyTorch and h5py. Everything runs on a laptop CPU;
the largest single computation is a Cholesky factorization of a
19000 x 19000 Gram matrix, about a minute in double precision.

## Citing

Please cite the paper in `paper/` together with the dataset sources: de Hoop,
Huang, Qian and Stuart (2022) and Batlle, Darcy, Hosseini and Owhadi (2024)
for the mechanics and PDE benchmarks, and Lamminpää, Susiluoto, Hobbs,
McDuffie, Braverman and Owhadi (2025) for the OCO-2 emulation problem.
