# Independent reviewer 01

**Vote on publishing the current attached version: NO.**

This is a repairable empirical paper, not a failed research program. Its strongest contribution is the controlled, replicated comparison of kernel corrections on raw inputs and learned features, together with a useful analysis of ensemble residuals. I would reconsider after the specific corrections below. My vote concerns the supplied version, not an assertion that a journal will reject a corrected one.

## Scope of this review

I read the full 39-page attached main manuscript, with special attention to all statements in Section 6 and their use in Sections 4–5. I independently derived the main algebraic claims. After an initial failed web fetch of the supplement, the coordinating agent successfully cloned the repository at commit `50c05ff1cce32d20df446c5c57c1c20027df5875`. I then read `paper/proofs.tex`, the supplementary stage and decision-rule statements, and the conformal statements. This final report incorporates those proofs. I did not rerun the computational campaigns or independently certify all empirical records. The submission package should include the supplement locally.

## Required corrections

### 1. The main conformal summary omits conditions that the supplement correctly supplies

Section 6.5 says that exchangeability alone yields both

\[
1-\alpha\leq\Pr\{Y_*\in C(X_*)\}\leq1-\alpha+1/(m+1)
\]

and an exact nondegenerate Beta law for calibration-conditional coverage. Supplementary Proposition S3.1 correctly restricts the upper bound to almost-surely distinct scores, and Remark S3.2 correctly assumes continuous i.i.d. scores for the Beta interpretation. Thus this is a main/supplement consistency repair, not a false supplementary theorem. The lower marginal bound is the generally valid part. Ties invalidate the main upper bound as written. For a concrete counterexample, let every score be identically zero. Then the usual closed conformal set has coverage one, which exceeds the displayed upper bound for the manuscript's \(m=1000,\alpha=0.1\). The conditional coverage is constant one, not Beta(901,100).

**Minimal repair:** state the lower coverage bound under exchangeability, state the upper bound under almost-sure absence of ties (or a fully specified randomized tie rule), and state the Beta law under conditionally i.i.d. continuous scores for a predictor and scoring function frozen before calibration. Define the case \(k=m+1\) with an infinite threshold. The finite Beta-binomial evaluation-count law can also be obtained from exchangeability of distinct ranks; do not mistakenly claim i.i.d. is logically necessary for every version of that count law. The i.i.d. continuous formulation is a simple sufficient condition for the calibration-conditional Beta interpretation used here.

A primary exposition states the usual continuous i.i.d. Beta benchmark explicitly: [Conformal Prediction via Transported Beta Laws](https://arxiv.org/html/2605.19024v1). The elementary all-ties counterexample above does not depend on this source.

### 2. The E1/E2 distinction is repaired in the theorem section but violated again in the results

Section 4.6 correctly states that the hindsight mixture was optimized for root-mean-square relative error \(E_2\), and explicitly says its reported mean relative error \(E_1=4.581\%\) is not an optimum of its own metric. The next page nevertheless says the deployed \(E_1=4.543\%\) pipeline lies below the hindsight optimum of every convex mixture. That conclusion is not implied: a different convex mixture could have smaller \(E_1\), even though the reported mixture minimizes \(E_2\).

**Minimal repair:** say the deployed pipeline has lower mean relative error than the evaluated hindsight \(E_2\)-optimal convex mixture. Alternatively solve the convex \(E_1\) problem explicitly before claiming it beats every convex mixture. The conditional pool floor remains valid for \(E_2\).

### 3. Proposition 6.7 has an omitted nondegeneracy condition and an incorrect interpretation near the threshold

The derivative formulas are algebraically correct, but both derivatives are declared strictly positive while the hypotheses allow perfect anticorrelation \(\varrho=-1\). For example, take normalized residuals \(\rho_1=Z\), \(\rho_2=-tZ\) with \(t\geq1\) and \(E\|Z\|^2=e_1^2>0\). The optimum is interior, has risk zero, and \(\partial V/\partial t=0\), contrary to the asserted strict positivity.

**Minimal repair:** assume \(e_1>0\), \(-1<\varrho<\min(t,1/t)\); describe the perfect-cancellation boundary separately. Also state positive member errors when defining correlations in Proposition 6.1.

The following sentence, that near the admission threshold \(t-\varrho\) is small and the trade-off rate is large, is generally false. Since \(t\geq1\), the admission threshold is \(\varrho=1/t\), and the coefficient multiplying a fractional accuracy improvement is

\[
\frac{1-\varrho^2}{t-\varrho}\longrightarrow \frac1t,
\]

not infinity. Remove that sentence and retain the exact trade-off formula.

### 4. Theorem 6.9 should state the deployed cross-fitting mismatch explicitly

The manuscript now acknowledges that its kernel is fitted to out-of-fold mean residuals but deployed over a different full-data mean. That caveat is necessary, but the initial method definition still describes residuals of a single mean and the overview continues to associate the theorem with the deployed estimator. Moreover, a cross-fitted mean defined only row by row need not define the global function required by an RKHS assumption.

There is a short exact repair. Let \(r=G-m_{\rm full}\), \(\Delta=m_{\rm full}(X)-m_{\rm cf}(X)\), and \(a(u)=k(u,X)(K+n\lambda I)^{-1}\). Then the actual predictor satisfies

\[
\|G(u)-m_{\rm full}(u)-a(u)[G(X)-m_{\rm cf}(X)]\|_2
\leq \|G-m_{\rm full}\|_K\widetilde P_\lambda(u)+\|a(u)\Delta\|_2.
\]

This preserves the useful theorem and applies pathwise. If the mismatch has not been measured, say so without presenting the first term alone as its error control. The unknown RKHS norm still prevents a numerical error certificate.

### 5. The pullback identity does not prove the stronger expressivity or causal claims

Proposition 6.8 is correct. It does not assert that a learned feature map lowers the norm, or that no rescaling can reproduce a feature-map effect. A feature map can itself be a linear rescaling, and the pullback identity supplies a conditional norm description rather than a direction of improvement for an arbitrary learned map.

The statement that a stationary kernel of fixed smoothness “reaches only its native space” is misleading as an approximation limitation: although each finite kernel fit lies in its native space, universal kernels may approximate targets outside that space on compact sets. Likewise, comparing two lower bounds on unknown target norms cannot prove an ordering of those unknown norms. Several passages already acknowledge the latter correctly; the surrounding causal prose should match them.

**Minimal repair:** confine the claimed direction to the measured finite-design diagnostics and the particular ARD/feature comparisons. Say that native-space error bounds depend on target regularity and that this experiment found a much larger feature benefit than the tested linear metric adaptations. No new theorem is needed.

### 6. Correct numerical rankings and inconsistent labels before publication

From Table 4, the six-member mean-error order is FNO 4.682, normalized-MSE MLP 4.712, refiner 4.730, UNet 4.740, metric-loss MLP 4.836, kernel 5.197. Thus the UNet is fourth of six, not fifth as repeatedly claimed in Sections 4.3, 4.5 and the discussion. The early FNO is third, behind the normalized-MSE MLP and refiner, not behind both residual-MLP variants.

Section 4.6 gives the single-FNO RMS error as 4.937% in prose, whereas Table 5 gives 4.967%. Its deletion narrative refers to both “flat-loss and normalized-MSE networks” in a pool whose stated members do not include a distinct flat-loss network in addition to the metric-loss MLP. Check these against the actual labels and source records rather than guessing replacements.

### 7. Two interpretive claims need narrower wording

First, the discussion following Corollary 6.4 says any convex ensemble sits inside its upper bracket. The displayed bound controls the optimum (and uniform weighting supplies the upper bound), not every badly chosen weight vector. Replace “whatever convex ensemble one builds” by “the optimal convex ensemble.”

Second, deterministic finite-element and interpolation error does not itself create an irreducible Bayes error when the benchmark target is that deterministic solver map. The observed plateau is established for the trained pool. A refined-mesh comparison could support a solver-artifact explanation of the common residual; it would not by itself prove an information-theoretic data floor. Retain the useful mesh-refinement proposal, but call it a diagnostic rather than an experiment that conclusively decides irreducibility.

### 8. Supplementary margin rule has a boundary inconsistency

Proposition S2.1(ii) removes the condition on \(D\) but defines error by \(\operatorname{sign}\widehat D\ne\operatorname{sign}D\). Its proof instead uses the binary mix decision \(\mathbf1\{D<0\}\). These are different at zero. With \(D=0\), \(\widehat D=\pm\sigma\) equally likely and \(\tau=\sigma\), the stipulated sub-Gaussian condition holds, but the sign-mismatch event has probability one, exceeding \(e^{-1/2}\).

**Minimal repair:** use \(\mathbf1\{\widehat D<0\}\ne\mathbf1\{D<0\}\) in the statement, exactly matching the proof and the actual decision. Alternatively retain \(D\ne0\) in part (ii). Also replace the subsequent claim that a sub-Gaussian variance proxy is estimated by empirical differences with the more careful statement already used in the main text: empirical standard deviation measures observed transfer variability but does not certify a sub-Gaussian proxy.

### 9. Two supplementary derivation details should be fixed

The sharpness proof of Theorem 6.9 defines \(r_1=(\|G-m\|_K/\widetilde P_\lambda(u))g_u\) but subsequently writes \(r_1(X)=g_u(X)\). Insert the missing multiplicative factor; the final representer argument remains correct.

The informal rate derivation alternates between \(H^s\) target regularity and Matérn native smoothness \(H^{s+d'/2}\), and calls validation selection cost lower order without a condition such as \(s/r<1/2\) and a specified validation-size regime. Merely calling a proposition informal does not resolve contradictory smoothness conventions. The clean minimal repair is to replace the numbered high-probability rate proposition by a plainly labeled heuristic illustration for exact-rank targets, and omit the unsupported validated-ridge-rate extension. None of the empirical comparisons or rigorous main theorems requires this heuristic.

## What survives the mathematical audit

- Proposition 6.1 is the standard second-moment identity and two-member quadratic minimization; it is correct after excluding undefined zero-error correlations or giving a separate boundary convention.
- Corollaries 6.3–6.4 follow by entrywise comparison and evaluation at uniform weights. The corollary statements are correct; the overstatement is in the prose described above.
- Corollary 6.5 is the constrained positive-definite quadratic optimum and is correct.
- Theorem 6.6 is correct: with architecture masses \(a_m=\sum_s w_{ms}\), entrywise comparison gives \(e^2[\varrho_b+(\varrho_w-\varrho_b)\sum_m a_m^2+(1-\varrho_w)\sum_{m,s}w_{ms}^2]\); Cauchy–Schwarz yields the displayed finite-pool floor. Its empirical use must remain limited to that pool and evaluation distribution.
- Proposition 6.8 is the standard quotient/pullback RKHS identity and is correct.
- Theorems 6.9–6.10 are correct for the single-mean exact-data problem as stated, apart from needing the usual pseudoinverse consistency conventions at singular designs. The exact error representer is \(k_u-\sum_i a_i k_{u_i}\), whose squared norm is \(P_\lambda^2-n\lambda\|a\|^2\). The minimax argument uses opposite residuals in the orthogonal complement of the design span. The ridge excess formula follows by diagonalizing \(K\).
- The supplementary symmetrization, exact half-batch interpolation identity, elementary validation bounds, affine-stack capacity argument, and uniform ridge-weight bound have coherent proofs under their stated boundedness/independence assumptions. They should not regain operational-guarantee wording in introductory prose when the main text correctly says those assumptions do not match all selections in the implemented pipeline.

## Publication assessment after repair

The central numerical findings could support a useful application/methodology paper if the released source data reproduce them. The theory is mainly a classical toolkit and should remain credited as such. The narrow representation comparison, metric-aware coordinate selection, seed replication, and negative uncertainty-ranking result are more persuasive than a claim of a new universal hybrid construction. A corrected manuscript with included proofs, checked table provenance, explicit conformal assumptions, and consistently limited conclusions would be substantially more defensible. I would not promise acceptance or certify empirical records I have not inspected.
