# Independent review 02 — experimental validity and statistical interpretation

**Vote on publishing the supplied version: NO.** The results could support a useful empirical methods paper after targeted corrections. My vote is against the current manuscript's inconsistent claims and incomplete linkage between the deployed estimator and its guarantees, not against the possibility of publishing this project.

I read the complete 39-page supplied manuscript independently. I did not inspect other reviews. I checked the official Lamminpää et al. article for the baseline's original evaluation and design. After the repository became available locally, I also inspected its correction/stacking/prediction scripts and relevant supplementary sections. I have not independently reproduced the numerical training experiments. Statements below distinguish contradictions demonstrable from the manuscript or code from unresolved experimental risks.

## What is already convincing

The paper has valuable controlled comparisons: ten seeds, common test cases, an explicit raw-input versus feature-kernel tuning budget, readout-ridge controls, paired bootstrap comparisons to stored baseline predictions, separate mean-relative and RMS-relative metrics, and honest discussion of the shifted OCO-2 validation distribution. The sixty-member convex-pool calculation is clearly marked as a finite-pool, empirical RMS bound. The manuscript explicitly identifies the hindsight optimization, discloses the weaker FNO training campaign, reports the constant-width conformal comparator, and admits that the native-space radius is unknown. These are substantial strengths. No direct test-label leakage in the main training procedure is established by the supplied PDF.

## Substantiated corrections needed before publication

### 1. The headline OCO-2 metric attribution contradicts the methods

Table 1's caption says the baseline is scored in both of the source paper's error metrics. Section 5, page 19, correctly explains that the two criteria are defined in this manuscript on the released reduced representation, and excludes truncation error, instrument response and the source acceptance test. Those are different claims. The official source assesses reconstructed radiances against instrument measurement noise and evaluates retrieval behavior; see Sections 4–5 of [Lamminpää et al. (2025)](https://amt.copernicus.org/articles/18/673/2025/).

**Minimum correction:** Name them consistently as this study's standardized-coefficient and retained-PCA radiance criteria. State that the source emulator's stored predictions are rescored. Retain the numerical improvements, but do not imply reproduction of the source paper's complete operational acceptance test.

### 2. A claim of simultaneous dominance over every model is contradicted by Table 6

The paragraph following Proposition 6.7, page 28, says that the coordinate combination improves on every single model on both metrics. On O2, Table 6 gives 4.12% reduced error for the combination and 4.11% for the flat-network feature kernel. The combination therefore does not dominate every component on the reduced metric.

**Minimum correction:** Say that the combination retains approximately the strongest reduced-coordinate accuracy while lowering radiance error, and improves on the stored source baseline on both rescoring criteria. Do not claim simultaneous per-component dominance or optimality.

### 3. A disclosed within-training label leak has not been rerun away

Supplement `paper/impl.tex`, in “Cross-fitted labels and out-of-fold centering,” states that the stored out-of-fold KRR predictions used the pooled target mean over all 19000 training rows before folding. Consequently the held-out fold's targets influenced their own supposed out-of-fold prediction channel. The text explicitly says the code has been corrected but the refiner and downstream rerun is queued and not included. The current `krr_oof.py` confirms fold-local target centering and documents that the released fields used the older pooled mean.

This is **within-training target leakage**, not evidence that test labels entered model training. It directly contradicts the main text's assurance that the kernel channel never sees its own target. Its actual downstream effect has not been measured. The supplement's suggestion that it is small should not be treated as an experimental conclusion.

**Minimum correction:** Either complete and report the corrected reruns, or explicitly retain the current results as those of the actual legacy implementation and withdraw the claim that they evaluate a strictly cross-fitted procedure. The second route is mathematically and empirically defensible: an estimator is permitted to reuse its training labels, provided the reported evaluation uses the appropriate held-out labels and no theorem requiring strict cross-fitting is claimed for it. Describe the estimator as foldwise kernel predictions with pooled training-target centering, make the training-label reuse visible, and distinguish the corrected code from the implementation that generated the numbers. A prose rewrite cannot certify the corrected procedure's accuracy, but it can correctly describe and evaluate the recorded procedure.

### 4. The method and theory describe different correction targets

Section 3.3 defines correction targets as residuals of the stacked mean. The caveat after Theorem 6.9, page 31, instead says that the actual correction uses cross-fitted member predictions and is deployed over a full-data mean. This is a real estimator distinction, not a small technicality. The manuscript admits its mismatch term is unmeasured, but Sections 3, 4.7 and 7 still speak as if the deployed correction carries the displayed bound without that term. The earlier description only clearly specifies out-of-fold kernel channels for the refiner; it does not adequately specify out-of-fold training for every member used in correction residuals.

**Minimum correction:** Reconcile the implementation description from actual code. If cross-fitting is used, specify folds and which members are refit. With a(u)=k(u,X)(K+nλI)^{-1}, D=m_full(X)-m_cf(X), and r_full=G-m_full, the directly applicable statement is

\[
\|G(u)-m_{\rm full}(u)-a(u)R_{\rm cf}\|_2
\le \|r_{\rm full}\|_K\widetilde P_\lambda(u)+\|a(u)D\|_2.
\]

The fitted norm from cross-fitted labels is also not automatically a lower bound on the full-data residual operator's norm. Either measure the mismatch for numerical discussion or explicitly treat the native-space result as a theorem for the ideal same-mean estimator. This repair does not require claiming any unperformed experiment.

Code inspection resolves part of this issue: `stack_perpixel.py` and `campaign/stack_correct_seeded5.py` use `krr_oof_train.npy` only for the kernel member, while neural training predictions come from `*_predtr.npy`. `gen_preds.py` loads the single fitted network and evaluates it on the training rows; there is no network cross-fitting here. The refiner's training evaluation uses its out-of-fold kernel channel, while its query evaluation uses the full-data channel. Thus the assertion that each training row's members are out-of-fold is false. The relevant mismatch includes the kernel and refiner channel changes, not an imagined full cross-fit of all networks.

### 5. The exact conformal law is stated under insufficient hypotheses

Section 6.5 says exchangeability supplies both the coverage interval and exact Beta/Beta-binomial law. Marginal lower coverage follows from exchangeability, but the two-sided rank bound needs a no-ties condition or appropriate randomization, and the exact Beta law requires iid scores with a continuous distribution conditional on the fitted predictor. Exchangeability alone does not yield a Beta mixing law. Moreover, conditional on a realized calibration sample, its population coverage is a fixed number, not a Beta-distributed quantity; it is the distribution of that random coverage over calibration samples that is Beta.

**Minimum correction:** Separate the exchangeability-based lower guarantee from the iid continuous-score calibration law. Say explicitly that the latter is a model-based reference distribution over fresh iid calibration draws, conditional on training. Clarify that a random partition of an already fixed finite test block alone does not establish that population law. The supplement's `supp_theory_conf.tex` does state the stronger iid-continuous and no-ties assumptions; propagate them into the main statement. Coverage results for the originally validation-fitted pipeline must remain separate from the exploratory ensembles that were fitted using the 1000-case test carve.

### 6. The PARA-Net comparison does not establish an inferential tie

The manuscript usefully admits that baseline predictions, variance and covariance are unavailable and calls its uncertainty figure a sensitivity calculation. It nevertheless repeatedly concludes that methods are not separable or tied on the basis of that number. Absence of baseline predictions precludes an equivalence conclusion; imputing the manuscript's per-case variance and zero covariance is not a tested standard error for the difference. The same cases were used, so their unknown covariance matters.

There is also a small arithmetic inconsistency: a per-case standard deviation 0.0169 and 20000 cases gives 0.01195 percentage points; two such terms in quadrature give 0.01690 points, rather than 0.020. Rounding may explain the scale, but the calculation should be internally reproducible.

**Minimum correction:** Use “numerically comparable to the reported 4.55%” and report the 0.004-point difference. Explain that statistical equivalence or significance cannot be assessed from the published scalar score. Keep the stronger paired evidence for comparisons inside this study, conditional on this test block. No baseline retraining is mandatory if claims are narrowed this way.

### 7. Loss-function attribution reverses the observed ranking

The beginning of Section 3 says all networks are trained in the reported metric and that it was better than normalized MSE. Table 4 gives 4.836% for the metric-loss MLP and 4.712% for the normalized-MSE MLP. The discussion then credits networks trained on the reported metric for reaching the 4.712% best-single result. This confounds the loss comparison and gives the wrong practical advice.

**Minimum correction:** Specify the exceptions in the method and state which loss actually produced each reported result. A favorable loss ablation on another setup cannot be generalized over the displayed high-data comparison.

### 8. The shared-residual interpretation still overstates a benchmark floor

Despite good caveats, Section 4.5 says the residual is not a function of the load in any way a Matérn RKHS can see and later describes the remaining error as not being on a path toward zero. A narrow grid search and slowly improving trained pool establish neither assertion. Deterministic discretization and interpolation error change the target function; they do not themselves create irreducible conditional noise when the entire target is a deterministic function of the supplied load. A common architectural or optimization bias can reproduce the observed correlations, localization and spectral concentration.

**Minimum correction:** Call this a shared residual of the evaluated predictor pool. Preserve the empirical localization and finite-pool convex lower bound, but remove universal claims about all surrogates, native spaces, or an irreducible data floor. Refining the solver would be useful future work, but is not required for publishing a carefully limited empirical diagnosis.

## Material limitations that are not demonstrated fatal flaws

1. **Validation reuse:** Section 6.1 correctly admits that the fixed-candidate capacity bounds do not apply to its adaptively selected pipeline. Remove statements elsewhere that fitting is statistically free or safe by those propositions. Reusing validation is not direct test leakage when the final test remains independent.
2. **Exploratory test reuse:** Section 4.6 explicitly fits additional stacks on a 1000-case carve from the benchmark test block and evaluates on the remaining 19000. Those are valid exploratory holdout analyses if their chronology is stated and if their calibrator is not then reused as an untouched conformal sample. Calling both parts untouched by stacking is literally inaccurate after the fitted analyses begin. “Evaluated once per run” does not establish independence of an entire research-development history.
3. **Boundary tuning:** Every feature head selects the largest length-scale and smallest nugget in the grid. The controlled finite-budget comparison remains useful, but it does not establish the optimally tuned advantage of feature kernels. Either include the promised wider grid once actually complete or consistently say “under the stated grid and budget.” The present version should not contain an unfulfilled “is being run” publication claim.
4. **ARD radiance omission:** The caption gives 0.150% at seven seeds and 0.329% at three but declines to report their mean because it reflects selection noise. Other pipeline means explicitly include selection noise. Report this as a bimodal outcome or mean plus dispersion; do not selectively hide the aggregate for this row.
5. **Low-data comparability:** The 1250-sample subset is not known to match the source subset. This is already acknowledged. Retain “reported-score comparison under a matched size budget,” not an exact reproduction of the source protocol.
6. **Uncertainty-cost claim:** A dense triangular solve at each test batch is O(n²) per point and need not be negligible compared with an O(nq) predictor at n=19000 and q=1681. Provide measured total prediction-plus-uncertainty cost or remove “negligible.”
7. **Unreproduced artifacts:** The PDF reports withdrawal of leaked small-n ClimSim points. I cannot verify that remaining scripts or plots exclude them. This is a reproducibility check for the final package, not evidence that its main benchmark results are contaminated.

## Minimum publishable revision

The existing numerical tables can largely survive as reported results of the actual recorded estimator, including its disclosed pooled-target centering. A publishable revision should (i) correct the metric names and false dominance/loss claims, (ii) describe one consistent implemented correction estimator and distinguish legacy numerical results from corrected code, (iii) add the mismatch term or restrict the theorem's applicability, (iv) state the actual conformal assumptions, (v) replace the equivalence claim with a numerical comparison, and (vi) restrict the floor and representation conclusions to the measured finite designs and budgets. A compact split-use table and an explicit metrics table would remove much current ambiguity. Include source, experiment configurations, split identifiers and all proofs in the deliverable, while clearly identifying reported results that this review did not reproduce.

These changes can make a defensible empirical paper. They cannot guarantee journal acceptance; the novelty remains chiefly the controlled diagnostic study rather than a new estimator family.
