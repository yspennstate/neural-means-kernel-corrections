# Independent reviewer 17

**Vote on publishing the uploaded `nmkc_paper_20260903e.pdf` unchanged: NO.**

**Assessment after the corrections below: potentially publishable as an empirical operator-learning study.** The ensemble second-moment identity and block lower bound are valid, useful diagnostics. The principal numerical sixty-member floor checks out. Several surrounding sentences nevertheless turn an optimum bound into a universal bound, a squared-metric optimum into a mean-metric optimum, or a finite-pool diagnostic into a much stronger explanatory claim. These are substantive but repairable issues; they do not require abandoning the experiment.

## Scope and independence

I reviewed the uploaded PDF's extracted text, concentrating on Sections 4.5–4.6 and 6.2, the corresponding supplement arguments, and stored second-moment matrices in `campaign/collected`. I did not read other reviewers' reports or modify the paper. My numerical checks recompute formulas from released summaries; they are not a fresh training run or a raw-prediction reproduction. This is a scientific readiness vote, not a prediction of a particular journal's editorial decision.

## Findings requiring correction

### 1. The mean-metric claim about beating every convex mixture is unsupported

Section 4.6 correctly says the hindsight sixty-member optimization minimizes RMS relative error, giving 4.876% RMS and 4.581% mean relative error, and explicitly acknowledges that the mean-relative-error optimum was not computed. The next paragraph nevertheless calls the deployed pipeline's 4.543% mean error “below the hindsight optimum of every convex mixture of the sixty.” That conclusion is not available: a different convex mixture can have larger RMS and lower mean error.

**Minimal repair:** “The deployed pipeline has lower mean relative error than the RMS-optimal convex mixture evaluated here.” Maintain the statement that the pipeline lies outside the convex candidate class, but do not use that fact to establish a mean-metric optimum that was not solved. This also applies to the discussion's informal “best convex weights” wording where the metric is omitted.

### 2. Corollary 6.4 bounds the best achievable convex risk, not every convex risk

The displayed inequality and its proof are correct. They bound `min_w w^T S w` above by the risk at uniform weights. The following sentence, “whatever convex ensemble one builds sits within that window,” is false. A simplex vertex can have much larger risk. For example, in the stored five-member seed-0 matrix, the largest vertex risk divided by the smallest diagonal is **1.205665**, whereas the displayed upper bracket is **1.037853**. This is an actual-data counterexample, not just a pathological construction.

**Minimal repair:** replace “whatever convex ensemble one builds” with “the optimum over the convex candidate class.” For a statement about the fitted ensemble, compare that particular ensemble's measured risk with the bracket separately.

### 3. The equicorrelation reference value is conflated with the entrywise lower bound

After Corollary 6.4, and in the corresponding supplement paragraph, the text reports the one-sided entrywise bracket and then labels 4.922% a “floor” in a way that identifies it with the corollary. The 4.922% macro is explicitly an **equicorrelation plug-in reference** formed from averaged errors/correlations. It is not the lower bound obtained from the minimum diagonal and minimum off-diagonal entries. Exact equicorrelation is not assumed of these matrices.

Recomputation from the stored annealed five-member subset `secmom5c_seeded.json` gives:

| Diagnostic | Ten-seed mean, RMS percentage |
| --- | ---: |
| Convex optimum | 4.940487 |
| Stored equicorrelation reference | 4.925096 |
| Rigorous entrywise asymptotic lower bound | 4.860786 |
| Stronger finite-five-member lower bound | 4.887049 |

From `secmom6_seeded.json`:

| Diagnostic | Ten-seed mean, RMS percentage |
| --- | ---: |
| Convex optimum | 4.919831 |
| Stored equicorrelation reference | 4.909697 |
| Rigorous entrywise asymptotic lower bound | 4.819795 |
| Stronger finite-six-member lower bound | 4.848514 |

The uploaded paper's first campaign is not identical to the annealed five-member subset, so the final manuscript should recompute the matching campaign values from its intended matrices, rather than blindly substitute my five-member values. The distinction itself is conclusive.

**Minimal repair:** call the averaged-correlation quantity an “equicorrelation reference,” explicitly say it is not a certified lower bound for the measured non-equicorrelated pool, and use the actual entrywise bound when explaining Corollary 6.4. The valid sixty-member bound below can remain the headline rigorous empirical floor.

### 4. Proposition 6.7's near-threshold interpretation is incorrect

The interior derivative formulas are algebraically correct in the nondegenerate range. But the paragraph below them says that near the admission threshold `t-rho` is small and “the rate is large.” For fixed `t>1`, the threshold is `rho=1/t`, and the fractional exchange coefficient has the limit

`(1-rho^2)/(t-rho) -> 1/t`.

There is no divergence. Moreover, since `t>=1`, `(1-rho^2)/(t-rho) <= 1+rho <= 2` throughout the permitted correlation range. When the denominator becomes small as `t` and `rho` both approach one, the numerator also vanishes. A small denominator alone is not a sensitivity argument here.

**Minimal repair:** delete the “near threshold ... rate is large” sentence and retain the measured finite exchange-rate interpretation. Also state positive nonzero member errors when correlations and their ratios are defined, and restrict strict positivity of the derivatives to `-1<rho<1`; at perfect anticorrelation the derivative with respect to `t` vanishes.

### 5. Negligible gain from global signed weights does not isolate the role of signs in pixelwise stacking

Corollary 6.5 and its proof are correct for the global affine constraint `sum w=1`. The empirical near-equality between its signed and convex global minima is useful. But it does not show that the deployed pixelwise affine stack's improvement comes from spatial variation “not their sign.” Pixelwise signed weights, unconstrained affine coefficients, and an intercept enlarge a different candidate class. Global sign ablation does not isolate their effects.

**Minimal repair:** say that allowing signs in the *global* stack buys negligible measured improvement, while the richer pixelwise affine fit improves performance. If causally attributing the improvement to spatial variation alone is important, compare pixelwise convex and signed models with consistent intercept handling. Such an ablation is optional once the causal claim is removed.

### 6. Finite-pool results must remain finite-pool, finite-evaluation statements

Theorem 6.6 is valid under its entrywise hypotheses. Its numerical evaluation is deterministic on the measured evaluation matrix and therefore describes that pool on those cases. It is neither a population confidence lower bound nor a guarantee that future seeds or unseen architectures will preserve the same entrywise minima. The phrase “no number of additional members” needs its condition retained whenever used outside the formal theorem. The sentence attributing the ten-seed-means pipeline's hundredth-point gain to what Theorem 6.6 “allows reseeding to buy” also overreaches: that pipeline has pixelwise affine and kernel stages outside the convex class.

**Minimal repair:** call 4.814% a “lower bound on empirical RMS error for convex mixtures of these sixty stored predictors on the evaluation cases.” State that extension to further members requires preservation of the hypotheses. Present the affine/kernel pipeline's reseeding comparison as a separate empirical observation.

### 7. The Matérn/benchmark-floor explanation is too strong in places

The manuscript is commendably explicit that a mesh-refinement experiment has not been run and that a common approximation bias could explain the shared residual. However, Section 4.5 still says the remaining residual is “not a function of the load in any way a Matérn RKHS on R^41 can see.” Small gains from a selected length-scale/nugget grid do not establish this. A deterministic finite-element solver defines deterministic labels; solver discrepancy against the continuum target is not automatically irreducible error when the surrogate is evaluated against those labels. Highly aligned, spatially structured errors can be shared approximation bias.

**Minimal repair:** “Under the tested designs and tuning grid, the selected Matérn correction explains little of the remaining residual.” Retain solver-discrepancy and approximation-bias explanations as alternatives. Use “empirical plateau for the trained pool” as the factual claim. The proposed refinement experiment is valuable future work, not required to publish this narrower finding.

## Checks that pass

- The normalized residual identity `E ell(f_w,G)^2 = w^T S w` follows directly from linearity when global weights sum to one. It does not require residuals to have zero mean; “second moment” is the correct terminology.
- The two-member criterion follows from the derivative at the better endpoint. The equicorrelated example, entrywise lower bound, optimum upper bracket, and positive-definite signed-weight formula have correct proofs, subject to nonzero-error conventions where division is used.
- The block decomposition in Theorem 6.6 is correct: with architecture masses `W_a`, the entrywise lower bound is `e_min^2 [rho_b + (rho_w-rho_b) sum_a W_a^2 + (1-rho_w)||w||^2]`. Nonnegative coefficients and Cauchy–Schwarz yield the displayed finite-pool lower bound.
- I independently recomputed the sixty-member bound from `dgx/seedarch.json`, using `S_ev`: minimum diagonal **0.002451207023114**, within-architecture coefficient **0.980796276240153**, between-architecture coefficient **0.937796819750730**. The resulting bound is **4.813611379% RMS**, matching 4.814%. The stored hindsight optimum is **4.876132938% RMS**, matching 4.876%.
- Remark 6.2 correctly separates mean relative error and RMS relative error and correctly notes that lower bounds do not transfer downward. The fixes should make the surrounding narrative obey this remark consistently.
- Reporting the hindsight optimum as a descriptive lower-envelope calculation is legitimate when clearly labeled and separated from fitted-and-evaluated performance.

## Requested presentation repairs

The first GitHub footnote currently does not direct readers to the supplement. Add an explicit supplement reference and direct path. Add a proof-location line at each theorem, proposition, lemma, and corollary whose proof is deferred; a single blanket statement at the start of Section 6 does not satisfy the user's request. Retain the supplement and its sources in the Overleaf package so those references resolve.

## Optional improvements

A compact table distinguishing (i) global convex mixtures, (ii) global signed weights summing to one, (iii) pixelwise affine fits with intercepts, and (iv) kernel-corrected pipelines would make the scope of each result much easier to read. An explicit `empirical S` notation in the numerical sections would also prevent population interpretations. Neither requires new experiments.

## Publication judgment

The unchanged version merits **NO** because several central interpretation sentences are mathematically stronger than the theorems and calculations support. The underlying sixty-member result survives direct checking, and these errors are repairable primarily by careful rewriting and consistent metric/candidate-class language. After those repairs, the empirical contribution is plausible for publication, with novelty framed as a replicated comparative diagnostic study rather than as a new ensemble construction or a universal benchmark error floor.
