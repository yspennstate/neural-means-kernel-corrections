# Neural means and kernel corrections for operator learning

Code and paper for a study of neural predictors combined with exact kernel regression on structural mechanics and OCO-2 radiative-transfer emulation. The experiments compare residual kernel corrections, kernel readouts on frozen neural features, and validation-selected combinations. Ensemble identities and finite-design kernel bounds help interpret the measured results.

## Paper, supplement, and Overleaf sources

- [Main paper](paper/main.pdf): revised manuscript, 32 pages.
- [Supplement](paper/supplement.pdf): complete proofs and additional experiments, 46 pages. Each result with a deferred proof identifies its location here.
- [Overleaf ZIP](releases/nmkc_overleaf_20260905.zip): complete editable project, figures, both PDFs, and the twenty-review revision record.
- [Publication review](paper/reviews/publication_review.pdf): original votes and the substantive corrections.

The source revision on this branch includes complete main-text proofs and new paired centering, correction-label, and OCO-2 grid experiments. Their [complete evidence archive](campaign/evidence/README.md) supports reconstruction of the new tables from per-case records. The linked PDFs and Overleaf ZIP above still belong to the earlier release while the final PDFs and ten new publication reviews are being prepared; the older twenty-review record is not a vote on this revision. See [the current completion record](docs/paper1_completion_status.md).

Upload the ZIP to Overleaf, choose **pdfLaTeX**, and select **main.tex**. The supplied build configuration compiles the companion supplement and resolves references in both directions. Locally:

```sh
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=output main.tex
```

Select `supplement.tex` instead to display that document as the selected output. Keep `main.pdf` and `supplement.pdf` together under those filenames for links between PDFs. See [paper/README.md](paper/README.md) for provenance, build details, and verification records.

## After-campaign checks

The last paragraph of the section "Seeds against architectures" and two rows
of its table come from six scripts under `campaign/dgx_checks/`, run on the
same seeds, run directories and calibration/evaluation split as the campaign
(the split is the permutation with seed 20260902 used by `seedarch.py`).
The records they wrote are `campaign/collected/dgx/p1_members_eval.json` and
`campaign/collected/dgx/sm_ens_rmt.json`; the rank-400 survey records are
under `campaign/collected/dgx/hidata_p400/`. Paths at the top of each script
(`ROOT`, `P2`, `SM_DATA`, `NMKC_HIDATA`) point at the run and data
directories.

```
# Helmholtz and Navier-Stokes survey cells, output rank 400, exact solve on 19,000 pairs (three seeds each)
python hidata_seeded.py --name Helmholtz    --grid 101 --seed 0 --npca 400 --fit 19000
python hidata_seeded.py --name NavierStokes --grid 64  --seed 0 --npca 400 --fit 19000
# learned kernels for the kernel member (ten seeds): isotropic grid, kernel-flow ARD, empirical-Bayes ARD, additive
python kf_kernels.py --problem structmech --seed 0 --methods val_iso,kf_ard,eb_ard,add_kf
# regularization selectors for the kernel member: validation, GCV, LOO, KARE, MP truncation and shrinkage
python rmt_krr.py --problem structmech --seed 0 --nfit 8000
# kernel-flow-regularized members (weights 0.3 and 1.0), the campaign's own training script
python train_mlp.py --seed 0 --mirror 1 --epochs 400 --kf 0.3 --tag mlpKF03 --threads 4
python train_mlp.py --seed 0 --mirror 1 --epochs 400 --kf 1.0 --tag mlpKF1  --threads 4
# the sixty-member per-pixel stacks with GCV, LOO, per-pixel GCV, MP-edge PCR, Ledoit-Wolf and shrinkage, plus the
# 300- and 120-row calibration regimes
python ens_rmt_dgx.py
# the paragraph's numbers: KF-regularized members from the run records, learned kernels on the evaluation half,
# and the six- and seven-mean stacks at the fixed and the leave-one-out ridge
python p1_members_eval.py
```

Three further computations were added after a review of the manuscript, each
from the stored campaign arrays or the campaign's own scripts; their records
are under `campaign/collected/dgx/`.

```
# the conformal band the theory names, ||e||/P_lambda with P_lambda rebuilt from each seed's correction
# kernel, on the same calibration split as the constant-width and disagreement-scaled bands (Section 4.6)
python campaign/uq_conformal_plam.py --seed 0 --tags hpix5,hpix        # seeds 0..9 -> uq_plam_seeded.json
# the frozen-feature ridge-readout control for the OCO-2 kernel heads: the campaign script rerun with the
# rows ridge_<mode> and combined_plus_ridge (Table of Section 5), ten seeds per band
python campaign/jpl_seeded.py --band o2 --seed 0                        # -> results/oco_o2_s0.json
python campaign/gen_oco_ridge_table.py <results_dir>
# the ClimSim kernel at larger budgets than the campaign's 6000 rows, same protocol (Supplement S7)
python campaign/climsim_seeded.py --data train --n 1000000 --seed 0 --kernel_only 1 --cap 24000
# the OCO-2 alignment diagnostic with the fitted norm tr(alpha^T K alpha), its nugget and scale sensitivity, and
# the test-point power functions of the raw and feature kernels (Section 6.3)
python campaign/dgx_checks/jpl_alignment_check.py --band o2
# the structural-mechanics fitted-norm diagnostic with the deployed correction kernel, normalized by energy (Table S6.2)
python campaign/dgx_checks/sm_norm_check.py --seed 0
# the kernel stage's storage, peak memory, factorization time and query latency at the campaign's shapes (Supplement S9)
python campaign/dgx_checks/cost_check.py --threads 8
# after the second review (3 September 2026):
# the OCO-2 ensemble quantities in the terms of Proposition 6.1 - RMS relative errors, uncentered alignments, the
# two-member admission example and the 840-pair scoreboard scored against the better test member (Section 5.2)
python campaign/dgx_checks/oco_ensemble_recheck.py --root <dir with oco_<band>_s<seed>/member_preds.npz>
# the arrays behind Figure S6.4 from the seeded test-block calibration path, and the figure itself
python campaign/dgx_checks/uq_fig_dump.py 0 && python campaign/fig_uq_seeded.py
```

The records of these checks, the locked Python environment of the campaign host
(`requirements_lock_nmkc_venv.txt`), the SHA-256 checksums of the data files
(`docs/reproduce.md`), the OCO-2 rerun's per-seed member records and per-sample errors
(`campaign/collected/dgx/oco_ridge/`), and the split-conformal records are under
`campaign/collected/dgx/`. The structural-mechanics residual archive (3.84 GB) and the
OCO-2 member prediction arrays (250 MB) are not in this repository: they are retained on the
campaign host and available on request, and are to be deposited under a persistent identifier
with the published version. Until then the release reproduces every table from the per-run
records it carries, and the array-level checks described in the supplement are reproducible only
from the retained archives.

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

Our columns report mean ± standard deviation over ten seeds per band at a matched 250-epoch training schedule. The two metrics rescore stored predictions in the retained 40-component representation; they do not establish full-spectrum or retrieval accuracy. The coordinate combination improves on the source emulator in both ten-seed means on all three bands. It does so at all ten seeds on O2 and SCO2, and nine seeds on WCO2. The flat feature-kernel head is slightly better than the combination on the reduced-error criterion in O2 and WCO2. A separate three-seed comparison expands each kernel search from 12 to 56 candidates. The feature advantage over raw inputs persists, but the combination's mean reduced error increases in every band and radiance changes have mixed signs across seeds. These follow-ups do not replace the historical table above.

**Structural mechanics** (de Hoop, Huang, Qian and Stuart; boundary load to von Mises stress field). The five-member pipeline reaches **4.572% ± 0.010%** mean relative test error over ten seeds. With the completed FNO schedule and a UNet, the six-member pipeline reaches **4.546% ± 0.003%**. This is numerically close to the published PARA-Net score of 4.55%; a scalar literature score without matching predictions or training variability does not establish statistical equivalence. In the 1250-label regime, the pipeline reaches **5.433% ± 0.093%**, compared with a published 6.49%, subject to the protocol differences stated in the paper.

For the fixed pool of sixty measured predictors, the empirical global-convex RMS lower bound is **4.814%**, close to the hindsight optimum of **4.876% RMS**. These are RMS quantities, distinct from the mean relative errors above, and they do not bound new predictors, per-pixel stacks, or kernel-corrected estimators. The experiments do not establish a universal benchmark data floor.

The historical mechanics scores describe foldwise kernel weights with pooled training-target centering, in-sample neural training predictions, and a changed kernel input to the refiner at inference. Ten paired reruns now compare pooled and fold-local target centering, including downstream refiner training and fitting. The mean change is about +0.000011 percentage points, with both signs across seeds. A separate fixed-estimator correction-label check also finds a small mean error change. The public test blocks were inspected across campaigns, so these are retrospective benchmark results.

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

Python with NumPy, SciPy, PyTorch and h5py. The released training and exact-solve workflows support CPU execution. The largest solves factor a 19000 × 19000 Gram matrix in double precision; memory use and runtime depend on hardware and concurrency. See Supplement S9 for measured costs.

## Citing

Please cite the paper in `paper/` together with the dataset sources: de Hoop,
Huang, Qian and Stuart (2022) and Batlle, Darcy, Hosseini and Owhadi (2024)
for the mechanics and PDE benchmarks, and Lamminpää, Susiluoto, Hobbs,
McDuffie, Braverman and Owhadi (2025) for the OCO-2 emulation problem.
