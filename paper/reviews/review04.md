# Independent reviewer 04

**Vote: NO on publishing version e unchanged.** The ensemble mathematics is largely correct and useful, and I would support publication of a carefully narrowed empirical paper after the specific corrections below. My objections concern actual theorem-to-experiment misidentification and statements outside a theorem's scope, rather than a demand for more architectures or a new theory contribution.

I reviewed the main manuscript, concentrating on Sections 4.5–4.6 and 6.2, the corresponding S8 proof source (`paper/proofs.tex`), the S6 and S10 discussion, and the released second-moment summaries and analysis scripts. This is an independent review; I did not read other reviewers' reports. I independently recomputed the six-member and sixty-member bounds from the released matrices. I did not retrain networks or independently reconstruct their prediction arrays.

## What is mathematically sound

Proposition 6.1's second-moment identity follows exactly from linearity of residuals for weights summing to one. The two-member decision and optimum are correct when the member RMS errors are positive; perfect-member cases need a brief convention. The equicorrelation formula, Corollaries 6.3 and 6.4 as displayed, and the positive-definite signed-stack formula in Corollary 6.5 are correct.

Theorem 6.6 and its S8 proof are correct. Write W_a=sum_k w_ak. Entrywise lower bounds and nonnegative weights give

    w' S w >= ebar² [rho_b + (rho_w-rho_b) sum_a W_a²
                    + (1-rho_w) sum_ak w_ak²].

The assumed ordering makes both coefficients nonnegative, and Cauchy–Schwarz gives sum_a W_a²>=1/M and sum_ak w_ak²>=1/(MK). The equality construction in S8 is positive semidefinite and the uniform weights attain its bound. Neither independence among seeds nor an equicorrelation assumption is required for the inequality. One-sided entrywise bounds are the actual assumptions.

The sixty-member empirical numerical claim survives independently. From `campaign/collected/dgx/seedarch.json`, using its evaluation-set S matrix:

| Quantity | Recomputed value |
|---|---:|
| Smallest diagonal, ebar² | 0.0024512070231 |
| Smallest within-architecture second moment / ebar² | 0.9807962762 |
| Smallest between-architecture second moment / ebar² | 0.9377968198 |
| Theorem 6.6 empirical RMS lower bound | 4.813611379% |
| Stored hindsight-optimal RMS error | 4.876132938% |

Thus the headline 4.814% bound is consistent with the released matrix and theorem. This is a rigorous algebraic bound on the empirical RMS objective of these sixty fixed predictors on these 19000 fixed cases, up to the numerical precision of the stored matrix. It is not a confidence bound for their population risks or a lower bound for untrained predictors.

## Required corrections

### 1. The rigorous floor and the equicorrelation diagnostic have been conflated

The paragraph following Corollary 6.4 says that the optimum 4.940% is against a floor of 4.922%, while the S6 counterpart similarly associates 4.908% with the six-member rigorous floor. Those macros are explicitly defined as equicorrelation diagnostics. `campaign/secmom6.py` computes `floor = mean(member_RMS) * sqrt(mean(centered_pairwise_correlation))`. This is not `sqrt(min(off_diagonal(S)))`, the lower bound furnished by Corollary 6.3. There is no theorem making a floor from these average centered correlations for a heterogeneous finite pool.

Recomputing the actual six-member bounds from `campaign/collected/secmom6_seeded.json` gives:

| Six-member quantity | Mean ± sample SD, percentage points |
|---|---:|
| Corollary 6.3 RMS floor, sqrt(min off-diagonal S) | 4.819795 ± 0.062054 |
| Stronger finite-M lower bound | 4.848514 ± 0.059915 |
| Validation simplex optimum | 4.919831 ± 0.056803 |

The 4.908/4.910 equicorrelation number may be kept as a clearly labelled heuristic reference, but it cannot be presented as the Corollary 6.3/6.4 lower bound. Replace the erroneous sentence and its S6 duplicate with values from the corresponding matrix and split. The sixty-member bound is unaffected.

### 2. Corollary 6.4 does not upper-bound every convex ensemble

The displayed formula bounds the *minimum* over weights, and the S8 proof correctly evaluates the uniform ensemble to establish an upper bound on that minimum. The following sentence, however, claims that 'whatever convex ensemble one builds sits within that window'. That is false. For M=2, ebar=1, rho=0.9, delta_e=delta_rho=0, take S=[[1,.9],[.9,1]]. The bracket for the optimum is [.9,.95], while the convex vertex w=(1,0) has risk 1.

Minimal repair: say the best convex ensemble and the uniform ensemble have the stated upper bound; arbitrary fitted weights require their own risk or an optimization-gap estimate. Do not claim this theorem shows that tuning is harmless for any estimator.

### 3. A same-validation identity is described as prospective prediction

Section 4.5 describes the comparison of predicted RMS and realized mean error as a prospective check across replicates. In `secmom6.py`, however, `disp_factor` is explicitly `real_val / pred_rms`. The reported approximately .938±.003 is therefore the conversion between E1 and E2 on the *same validation data* on which S and the weights were calculated. That is a legitimate dispersion measurement, but it is not independent validation of prediction transfer. The script separately stores `real_test`.

For the six-member records, the actual ratio `real_test / pred_rms` has mean .935734 and SD .011000, appreciably less stable than the displayed same-validation ratio. Moreover, this latter ratio combines metric dispersion with sample transfer and should not itself be called a pure dispersion factor.

Minimal repair: describe the .938 measurement as an in-sample metric diagnostic. If presenting transfer, give validation E2, test E2, and test E1 separately. The already released `real_test` can be reported without pretending it verifies the RMS identity out of sample.

### 4. Several interpretations go beyond Theorem 6.6

After Theorem 6.6 the prose says reseeding divides away seed-to-seed correlation. In its formula only the private term (1-rho_w)/(MK) vanishes with K; the persistent within-architecture term (rho_w-rho_b)/M does not. State this decomposition accurately.

Section 4.6 says that the gain from 4.543% to 4.533% for a per-pixel affine stack plus kernel correction is 'what Theorem 6.6 allows reseeding to buy'. The theorem addresses global convex combinations in RMS error. That pipeline is neither a global convex mixture nor evaluated there in the theorem's metric. Remove the theoretical attribution; the measured gain can stand as empirical evidence.

Explicitly label the floor 4.814% as empirical and restricted to the measured pool. Any claim about arbitrarily many future seeds or architectures must say that the same entrywise lower bounds continue to hold. Observing them for the current pool does not establish that extension.

### 5. Remaining supplement statements contradict the corrected main-text scope

S6 says that the mean error lies below the floor 'by Jensen's inequality whatever the estimator does'. Jensen only states E1<=E2; a lower bound on E2 does not imply E1 is below that lower bound. The same paragraph converts the final pipeline's E1 using dispersion factors measured for a different predictor. The main text's Remark 6.2 explicitly explains why a common conversion must not be reused across predictors. Delete that conversion unless actual E2 values for the final predictors are available.

S6 also says the unconstrained and convex optima coincide at every seed, while the main text and stored signed records correctly acknowledge small negative coordinates and small nonzero gaps. Replace 'coincide' by a measured numerical tolerance and retain the sign information.

### 6. Minor mathematical boundary and wording repairs

Proposition 6.7 claims both partial derivatives are strictly positive throughout the interior-mixture region. At rho=-1 that region can hold, but dV/dt=0 because 1-rho²=0. Add -1<rho<min(t,1/t), or use nonnegative for dV/dt and handle the endpoint. The S8 proof likewise overlooks this endpoint.

Its comment 'near the admission threshold, where t-rho is small' is not true for general t=e2/e1>=1: the threshold is rho=1/t, and t-rho tends to t-1/t, which can be large. Qualify the statement to similarly accurate members (t near 1), which appears to be the intended experimental case.

State that normalized residuals require ||G(u)||>0 almost surely and finite second moments, and handle a perfect member e1=0 separately from formulas dividing by e1e2. These are routine assumption clarifications.

## Publication assessment after repair

The result worth keeping is an empirical operator-learning study with reproducible improvements and transparent limitations. The sixty-member floor is a useful exact diagnostic and its numerical value checks out; it does not need to be advertised as a limitation of the benchmark itself. The main theoretical backbone need not be rewritten. Correct the floor attribution, remove the prospective-validation claim where the comparison is in-sample, enforce the global-convex/RMS scope consistently in both main text and supplement, and make the boundary corrections. Subject to the other domains being checked by their reviewers, those changes would resolve my reasons for voting NO on the current version.
