# Independent review 06

**Vote on publishing the attached version: NO.** The paper has a credible empirical contribution and a repairable presentation, but several headline and mathematical interpretation statements contradict its own tables or caveats. I would support a carefully corrected empirical preprint. I would not recommend immediate journal acceptance on this version; the limited scope and largely classical theory need to be stated consistently, and full provenance should remain available to reviewers. No rewrite can guarantee acceptance.

## Scope and evidence

I read the supplied main manuscript and independently examined its OCO-2 experiment code, the original thirty per-seed JSON records, and its aggregation script. Relevant repository files are `jpl_data.py`, `campaign/jpl_seeded.py`, `campaign/gen_oco_table.py`, and `campaign/collected/box*/oco_{band}_s{0..9}.json`. I verified the source paper's reconstruction and evaluation discussion against the official [Lamminpää et al. article](https://amt.copernicus.org/articles/18/673/2025/), particularly Eq. (20) and Sections 4.1–4.3. The underlying HDF5 arrays were not available in the downloaded repository, so I did not independently rerun training or verify numerical orthogonality of the released PCA matrix. The JSON reconciliation is independent arithmetic on supplied summaries, not independent reproduction of experiments.

## What is sound and worth publishing

1. The reduced-representation OCO-2 comparison is scientifically meaningful when labeled as a rescoring experiment. The comparison evaluates both methods on the same public test states and does not purport, in Section 5's careful paragraphs, to reproduce an end-to-end retrieval result.
2. The reconstruction code is mathematically appropriate: it undoes coefficient standardization, applies the stored PCA matrix, and restores the radiance mean. `radiance_error` uses reconstructed targets in its denominator. It does not mistakenly use the norm of weighted coefficients as the reported radiance denominator. The exact-radiance training loss is similarly implemented correctly.
3. Frozen-feature linear ridge is a useful readout control. It strengthens the claim that the reported kernel advantage over the network's trained linear layer is not reproduced by this particular post-hoc linear refit.
4. The seed records reproduce the Table 1 and Table 6 OCO-2 numbers and support the reported paired-bootstrap exception on weak CO2. The large gain on the reduced criterion is not an arithmetic or table-transcription error.
5. The explanations distinguishing mean relative error from RMS relative error, test-case uncertainty from retraining uncertainty, descriptive finite-pool floors from a universal approximation floor, and marginal from conditional conformal coverage are worthwhile. Several passages then undo those distinctions; the repairs below should preserve the careful versions.

## Required corrections

### 1. The headline comparison misidentifies the source metrics

Table 1's caption says the stored OCO-2 source predictions were scored “in both of that paper's error metrics.” Section 5 correctly says the two criteria are defined in the present study and neither is the source acceptance test. These cannot both be true. The source paper reconstructs spectra and additionally evaluates residuals after instrument response and measurement-noise scaling; it also investigates retrieval behavior. The current rescoring excludes PCA truncation, instrument response, and noise scaling.

Replace the caption with: “OCO-2 rows rescore the released predictions of Lamminpää et al. on the same 2,000 public test states using the two reduced-representation criteria defined in Section 5. These are not the source study's instrument-noise or retrieval acceptance tests.” Apply the same scope wherever the introduction currently says “both of that problem's error metrics.”

### 2. The claim of improving every individual model on both metrics is false

Section 6.2 says the combination improves “on every single model on both metrics at once.” Table 6 gives the flat-network Matérn head 4.11% reduced error and the combination 4.12%. The original records give 4.11080% versus 4.11510%, so the combination is worse by 0.00430 percentage points before rounding. It is substantially better in radiance, 0.02940% versus 0.07927%.

Use: “The combination improves on the rescored published emulator in both seed-averaged criteria, while retaining nearly the reduced-coordinate accuracy of the flat-feature kernel head.” State the OCO-2 weak-band seed exception. Avoid implying Pareto dominance over every component.

### 3. Correct why coordinatewise optimization does or does not apply

Section 6.2 first says diagonal evaluation metrics decouple, then says the per-case denominators make the reported relative norms non-separable. The reported mean norms are indeed non-separable, but the explanation identifies the wrong source of coupling. A fixed target-dependent denominator is independent of the selected prediction and preserves separability for squared loss. The square root couples coordinates in a mean of norms.

Let columns of U be the retained orthonormal components, D=diag(s_z), and define

\[
R(z)=\mu_y+U(\mu_z+Dz).
\]

Then the two reported metrics are

\[
E_{red}=N^{-1}\sum_i\frac{\|\hat z_i-z_i\|_2}{\|z_i\|_2},\qquad
E_{rad}=N^{-1}\sum_i\frac{\|D(\hat z_i-z_i)\|_2}{\|R(z_i)\|_2}.
\]

The numerator identity uses U^TU=I. In the row-vector convention of the code, P=U^T. Both offsets enter the radiance denominator and cancel in its error numerator.

For a general squared objective

\[
N^{-1}\sum_i\omega_i\sum_j a_j(\hat z_{ij}-z_{ij})^2,
\]

coordinatewise selection is exact over the Cartesian product of candidate coordinates when all weights are fixed and nonnegative. Ordinary coordinate RMSE minimizes the unweighted squared coefficient objective. Squared relative coefficient loss instead uses omega_i=1/||z_i||^2. Squared relative radiance loss uses omega_i=1/||R(z_i)||^2 and a_j=s_{z,j}^2. Those different sample weights need not produce the same winners. No exact simultaneous optimum follows for the two reported mean-relative-norm criteria. Present coordinate selection as a useful empirical combination procedure with an exact auxiliary squared-loss interpretation.

The code comment “optimal for any diagonal metric” should likewise be narrowed to “optimal for fixed diagonal squared loss without sample reweighting.”

### 4. A minor selector-reconciliation statement is numerically false

Section 5 says weighted and unweighted selectors have identical printed means on every band. At O2, unweighted reduced mean is 4.11510% (4.12% at two decimals), whereas the inverse-squared-target-norm selector gives 4.11455% (4.11%). This is a rounding-boundary difference with no substantive consequence, but the claim is false. Replace it with “their mean errors differ by less than 0.001 percentage points in the reduced criterion, with the radiance means unchanged at the reported precision.” The original JSON values support that statement on all three bands.

### 5. Report the missing ARD radiance summary

Table 6 suppresses the ARD radiance cell because seed-dependent tuning selects two modes. That variability is a property of the declared selection pipeline, exactly as other selected stages have seed variability. Both the aggregate and the modes are informative. The ten original records give **0.20397 ± 0.08652%**, or **0.204 ± 0.087%**. Put this in the cell and retain the caption's explanation of the two selection modes. Do not change the selection rule based on test radiance outcomes.

### 6. Harmonize the feature diagnostic description

The introduction says the design factor falls “by about half.” Section 6.3 reports a raw-to-feature factor of 1.5 at the median scale, and approximate equality at the selected multiplier 4. Thus the former claim is not what the later measurements say. Use the scale-specific description. A ratio of fitted norm lower bounds also does not identify a ratio of unknown target norms; retain its description as a diagnostic and avoid causal claims that the bound has explained a measured error factor.

### 7. Remove invalid inferential shortcuts elsewhere in the main narrative

The structural-mechanics “tie” wording should be “numerically close under the published benchmark scores; a paired comparison or equivalence claim is unavailable.” Absence of the baseline's prediction errors and variance cannot establish a tie, even if the text elsewhere calls its sampling arithmetic a sensitivity calculation. The claimed quadrature value about 0.020 percentage points is also not exactly the quadrature of two 0.012-point components, which is about 0.017.

Section 4.5's statement that a measured weak residual correction proves the shared residual is not a function the Matérn RKHS “can see” is unsupported: one limited grid, one training design, and finite optimization do not establish an approximation impossibility. Say only that the tested corrections recovered little of it. Remove claims that solver discrepancy is established at the spatial locations without a re-solve. The paper acknowledges this caveat elsewhere; consistent wording is enough.

The proof-aware reader will also notice that Corollary 6.4 bounds the optimal convex risk, not every convex ensemble. Its accompanying “whatever convex ensemble one builds” language must refer to the optimum or a verified optimization error. The 4.814% finite-pool lower bound is an RMS-relative-error bound and should never be asserted as a lower bound on the 4.581% mean-relative-error entry.

## Reconciliation of original OCO-2 records

All values below are percentages. Dispersion is sample standard deviation across the original ten seeds.

| Band | Rescored source reduced | Combination reduced | Rescored source radiance | Combination radiance |
|---|---:|---:|---:|---:|
| O2 | 16.8891 | 4.11510 ± 0.04759 | 0.0448 | 0.02940 ± 0.00204 |
| Weak CO2 | 24.0604 | 16.27925 ± 0.05581 | 0.0599 | 0.05073 ± 0.00523 |
| Strong CO2 | 16.1448 | 8.08365 ± 0.01221 | 0.1147 | 0.05837 ± 0.00408 |

Table 1 is consistent with these values. The O2 Table 6 component means also reconcile. Table 7 identifies its own readout-control rerun, so its small weighted-network differences are not intrinsically contradictory; preserve that explanation and distinguish campaign artifacts by identifiers.

The stored paired-bootstrap intervals are negative for combination minus source on both O2 and strong-CO2 metrics at all ten seeds. Weak-CO2 reduced intervals are negative at all ten seeds. Weak-CO2 radiance has one exception, seed 0: its interval is [-0.0012, 0.0042] percentage points and its point estimate is worse by about 0.0015 points. Therefore “every seed on every band on both metrics” would be false, while the existing careful nine-of-ten description is correct.

The 840-pair scoreboard adds correctly: the existence-rule counts total 524/840=62.38%, and the shipped-rule agreement counts total 481/840=57.26%. The manuscript correctly warns that these dependent pairs are not 840 independent Bernoulli trials.

## Publication recommendation after repair

The strongest publishable version is an empirical study of what post-hoc kernels contribute in two specific data regimes, with simple conditional theory used to describe measurements. Keep the OCO-2 numerical advantage, the frozen-readout control, the sixty-predictor finite-pool analysis, and the negative result about the uncertainty ranking. State explicitly that training budget and kernel grids delimit the comparisons. Finish the expanded-grid investigation before claiming performance optimized over kernel scale; alternatively state clearly that the present fixed-grid result is the object studied and remove language about a sweep currently running.

I would vote YES to release that corrected preprint, with its source and records. I would still expect normal journal review of novelty, experimental controls, and reproduction. My NO vote on the supplied PDF is based on concrete correctable inconsistencies, not a belief that the underlying empirical result is worthless or irreparable.
