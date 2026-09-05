# Independent review 18 — numerical, reporting, and implementation audit

**Vote on the uploaded version: NO, revise before publication.** This is a repairable paper with a credible empirical contribution. My objection is to demonstrably inconsistent methodological and numerical interpretations, not to the existence of the results, a particular venue, or a demand that the construction itself be new.

## Scope and evidence

I read the original uploaded paper through `tmp/pdfs/original.txt`, checked the relevant original supplement sources (`paper/impl.tex`, `paper/supp_experiments.tex`, and associated sections), inspected the released stacking, prediction-generation, OOF, table-generation and aggregation scripts, ran `campaign/audit_reported_macros.py`, and independently aggregated the thirty original OCO-2 band/seed JSON records under `campaign/collected/box*/`. I did not read other reviewers' reports. I did not retrain the networks or rerun the dense kernel experiments. Record consistency does not independently establish correctness of unavailable prediction arrays or provenance of their training.

The macro audit reports **zero disagreements**. The structural-mechanics headline `4.546 ± 0.003%`, its predecessor `4.572 ± 0.010%`, the low-data `5.433 ± 0.093%`, the sixty-predictor RMS values, and the principal OCO-2 means are genuinely backed by the released records at their printed precision. The manuscript deserves credit for explicitly distinguishing RMS and mean relative error in most of the central floor comparisons, for reporting a weak-band loss, and for including unsuccessful controls.

## Necessary corrections

### 1. The architecture with the largest pool-drop penalty is misidentified

Section 4.6 attributes a `0.021` percentage-point RMS penalty to the “metric-loss MLP with reflection averaging” and calls the relevant architecture the reflection-averaged MLP again when quoting its `4.979%` RMS. The record `campaign/collected/dgx/dropone.json` says these are `mlpR`, the **kernel-conditioned refiner**:

- `drop_mlpR.d_oracle_rms = 0.0002084901998219424`, or `0.0208490` percentage points.
- `only_mlpR.oracle_rms = 0.04978992379409608`, or `4.97899%`.
- The actual metric-loss MLP is `mlp`; its oracle removal penalty is approximately `0.00000325` percentage points, effectively zero.

`gen_preds.py` and `train_mlp_refine.py` make the naming unambiguous: `mlpR` consumes a load plus a KRR field, while `mlp` is the plain network. Replace every architecture attribution in this paragraph, including the quoted correlations, with the correct refiner name. Rename the zero-weight “flat-loss” model consistently as the plain metric-loss MLP. This correction changes the scientific interpretation of the most useful architecture and cannot be left as a cosmetic naming issue.

### 2. The claimed full cross-fitting does not describe the released computation

The discussion after Theorem 6.9 says that “each training row's members are out-of-fold.” The prediction-generation code loads one trained neural checkpoint and evaluates it on `tr`, `va`, and `te`. `campaign/stack_correct_seeded5.py` obtains `krr_oof_train.npy` for the KRR member but the ordinary `*_predtr.npy` files for neural members. Thus the correction is fitted on a hybrid training prediction table: the KRR field is OOF; neural predictions are made by networks fitted on the training set; the refiner additionally receives an OOF field channel. This is not a fully cross-fitted ensemble.

Supplement S9 also documents that the **historical** OOF fields used pooled target centering, whereas the current `krr_oof.py` now uses fitting-fold centering and the rerun is not in this version. That contradicts the strict claim in Section 3.1 that the kernel channel never sees its own target. The current corrected script is therefore not the exact script that generated the displayed historical rows.

Minimum repair: describe the historical predictor table exactly, distinguish it from a fully cross-fitted mean, and state the algebraic bound using the actual discrepancy between that training table and evaluations of the deployed mean. Either regenerate the affected refiner/downstream records with the corrected centering or preserve the historical implementation/configuration and clearly label the reported rows as the pooled-centering version. Do not present a queued rerun as completed. This issue does **not** itself demonstrate test-label leakage or invalidate the fixed-test error computation; the claim that must be removed is exact OOF honesty and exact applicability of the one-mean bound to the shipped correction.

### 3. The claimed simultaneous dominance of every single model is false

The final paragraph of Section 6.2 says the validation-selected combination “improves on every single model on both metrics at once.” Table 6 already contradicts this: on O2 the flat feature head is `4.11%` reduced error while the combination is `4.12%`.

Independent aggregation gives:

| Band | Flat feature head, reduced (%) | Combination, reduced (%) | Combination wins against source emulator, reduced / radiance |
|---|---:|---:|---:|
| O2 | 4.11080 | 4.11510 | 10/10, 10/10 |
| WCO2 | 16.27688 | 16.27925 | 10/10, 9/10 |
| SCO2 | 8.08422 | 8.08365 | 10/10, 10/10 |

The real and useful result is that the combination trades a tiny amount of reduced-coordinate accuracy on O2/WCO2 for substantially better radiance accuracy, while improving on the source emulator's rescored mean on both criteria for each band. Replace universal dominance language accordingly, and ensure “at ten seeds per band” is not read as ten wins on WCO2 radiance.

### 4. Table 1 misstates whose metrics are being reported

Its caption says OCO-2 is scored “in both of that paper's error metrics.” Section 5 explicitly and correctly states that these are two criteria defined **here** on the released reduced representation and that neither is the source paper's full-spectrum/noise acceptance test. Change the caption to match Section 5. Keep the fact that the source emulator's stored predictions were rescored on identical test states; that is a meaningful controlled comparison without claiming to reproduce its original acceptance metrics.

### 5. The statistical comparison contains an arithmetic discrepancy and overstates a tie

Section 4.4 gives per-case relative-error SD `0.0169` over `20000` cases. This yields `100 × 0.0169 / sqrt(20000) = 0.0119501` percentage points, consistent with `0.012`. But applying the stated equal-variance quadrature assumption gives `sqrt(2) × 0.0119501 = 0.0169`, **not `0.020`**. The supplement moreover still says the PCA-Net difference is “real” and describes the assumed comparison as conservative, although the original main text now more carefully calls it a sensitivity calculation.

Minimum repair: use a reproducible sensitivity calculation and remove the implied statistical equivalence. The source predictions and covariance are unavailable; these data support numerical proximity to the published `4.55%` benchmark score, not a formal finding of equivalence or nonseparation. The numerical headline need not change. Synchronize the main and supplement rather than leaving a weaker caveat in one and a stronger inference in the other.

### 6. Method and inference prose must be synchronized with the controls

Several local examples can be repaired without new experiments:

- Section 3 says all networks optimize relative error, although the normalized-MSE network is explicit and central; write “except the stated loss variants.” The conclusion's `4.712%` best network is the MSE-trained network, so it should not be described as trained on the reported metric.
- Section 4.3 calls the FNO at `4.754%` third “behind both residual-MLP variants.” The plain metric-loss MLP is `4.836%`; the models ahead are the normalized-MSE MLP and the kernel-conditioned refiner.
- Supplement S9 initially says the test set is never touched during development, then later explicitly documents repeated test-block inspection during development. Retain the latter accurate historical account. Likewise Section 4.6 fits new weights on a subset of the original test block; distinguish this exploratory pool analysis from the frozen pipeline results.
- The abstract says “replicating every stage at ten seeds,” whereas the representation-norm mechanism is a one-seed diagnostic and some supplemental checks have other coverage. State ten-seed replication of the principal pipelines and identify the one-seed diagnostics directly.
- The introduction says the learned-feature design factor falls “about half”; Section 6.3 reports raw/feature ratio `1.5` at the median scale and `0.99` at the selected scale. Use those conditional values and do not let the introduction imply the selected feature head has a factor-two design improvement.

## Publication judgment after those repairs

**Potentially YES for a carefully scoped empirical and diagnostic paper.** The principal recorded gains survive this audit. The sixty-predictor floor is correctly compared in RMS in Table 5, and the OCO-2 records substantiate improvements over the rescored source emulator with the stated WCO2 exception. I would not demand mesh refinement or a broader search merely to permit publication: remove claims that those unperformed experiments would be needed to establish, retain the documented fixed-benchmark observations, and repair the precise inaccuracies above. A clean rewrite can keep most of the numerical content. Correcting historical OOF provenance and applicability is more consequential than increasing the decimal precision or adding another generic novelty claim.

For the requested final package, the first GitHub footnote should explicitly direct readers to the supplement, and each formal result should have a local proof-location reference. Those editorial changes are necessary for navigation but do not substitute for the corrections above.
