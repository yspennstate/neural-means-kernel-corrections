# Independent review 11 — native-space claims and interpretation

**Vote on the submitted version: NO.**

**Judgment:** The empirical study appears suitable for a publishable, carefully scoped paper, but I would not submit this version unchanged. The kernel pullback, power-function calculation, and noiseless minimax argument are essentially sound. Several nearby sentences nevertheless make stronger statements than those results establish, one theorem's exact-worst-case sentence has a missing multiplicative factor, and the manuscript does not consistently connect the stated residual bound to the correction actually described in its cross-fitting caveat. These are repairable without inventing experiments or replacing the central contribution.

I reviewed the submitted PDF via `tmp/pdfs/original.txt`, its main experimental and theory sections, and the relevant proofs in `source_repo/paper/proofs.tex`, with checks of the supplementary rule and conformal statements. I did not read other reviewers' reports or modify the manuscript. This review does not independently reproduce the large experiments or certify every bibliography entry.

## What survives scrutiny

1. **Proposition 6.8:** The pullback RKHS is the image under composition equipped with the quotient norm. The kernel of the composition map is closed because it is an intersection of kernels of bounded evaluations; hence the minimum-norm representative exists. The bound on the pullback norm of `h composed with phi` is correct. Coordinatewise extension using the sum of squared RKHS norms is also correct.
2. **Theorem 6.9:** For a fixed kernel, design, nugget, and residual function, the error representer is `g_u = k(u,.) - sum_i w_i k(u_i,.)`, with `w=(K+n lambda I)^(-1) k(X,u)`. Its squared norm is exactly `P_lambda(u)^2 - n lambda ||w||_2^2`. Applying Cauchy–Schwarz coordinatewise and summing gives the displayed inequality. The extremizer supported in one output coordinate establishes sharpness after the normalization is corrected below.
3. **Theorem 6.10:** The two opposite residuals vanishing on the design give the correct lower bound `rho P_0(u)` for every estimator from exact residual values. Interpolation attains it. The identity for the regularization penalty is correct: in the eigenbasis of K its nonnegative terms are `(beta_i^2 / mu_i) [n lambda/(mu_i+n lambda)]^2`. The theorem concerns global worst-case recovery on a fixed native-space ball, with exact observations. It is not a data-conditional posterior uncertainty result.
4. **Finite-design norm diagnostics:** The ridge fit has RKHS norm no larger than the norm of any RKHS function furnishing those same noiseless labels. This follows by first orthogonally projecting onto the kernel-section span and then applying the spectral shrinkage factors `mu_i/(mu_i+n lambda)`. Thus the lower-bound reading is mathematically justified for a matching target/residual. Ratios of these lower bounds cannot order the unknown full norms.
5. **Overall empirical scope:** The separation between mean relative error and RMS error, the limitation to the trained predictor pool, the explicit OCO-2 rescoring definition, disclosure of validation reuse, and the refusal to claim an established PDE discretization floor are valuable. The native-space theory need not carry a novel-method claim for the controlled empirical comparison to be worth publishing.

## Mandatory repairs

### 1. Correct the exact-worst-case statement in Theorem 6.9

The theorem says that `tilde P_lambda(u)` is the exact worst-case error over the ball of radius `||G-m||_K`. The correct value is **`||G-m||_K tilde P_lambda(u)`**. Equivalently, `tilde P_lambda(u)` is the exact worst-case error on the unit ball. The displayed inequality itself already has the right factor; the following sentence must match it.

In the supplement's sharpness proof, `r_1=(||G-m||_K/tilde P_lambda(u)) g_u` is correctly defined, but the assertion `r_1(X)=g_u(X)` drops that scaling. Replace it by the scaled equality, or simply say that the estimator is built from `r_1`'s own training values. The zero-power-function case is already handled.

### 2. Confine the minimax claim to interpolation with fixed choices

The introduction says that the power-function bound is sharp and, by Theorem 6.10, minimax; this can be read as asserting minimaxity of the deployed positive-nugget correction. Theorem 6.10 itself correctly says that interpolation is minimax and ridge pays a nonnegative price. State that distinction explicitly in the introduction and discussion. Also replace “the design factor cannot be improved by a better estimator” by a sentence naming `P_0`, since the ridge factor generally can be improved by switching to interpolation in this noiseless worst-case problem.

**Exact counterexample:** Set `n=q=1`, `K=[1]`, query at the training point, and `n lambda=1`. Then `P_0=0`, whereas `tilde P_lambda=1/2`. Interpolation recovers the target exactly; ridge's worst-case error on the radius-rho ball is `rho/2`. This also directly checks the factor missing in item 1.

Keep kernel, feature map, mean, and nugget fixed when interpreting an exact worst case. A bound evaluated pathwise at data-selected choices remains a valid algebraic bound when its membership conditions hold; that does not automatically make the entire data-adaptive selection procedure the fixed-rule minimax estimator described by the theorem.

### 3. Make the residual-training/deployment mismatch explicit

Section 3.3 describes training labels as `G(X)-m(X)`, while the caveat after Theorem 6.9 says they actually use cross-fitted means and deploy a full-data mean. Resolve this against the implementation. If the caveat is accurate, give the following exact bound rather than an unspecified extra term.

Write `r_full=G-m_full`, `Delta_X=m_cf(X)-m_full(X)`, and `w=(K+n lambda I)^(-1)k(X,u)`. Then, for the deployed prediction,

`||G(u)-m_full(u)-w^T[G(X)-m_cf(X)]||_2`

`<= ||r_full||_K tilde P_lambda(u) + ||w^T Delta_X||_2`.

This is algebraic and does not require defining an out-of-fold map away from the training rows. Unless the discrepancy is measured, do not describe Theorem 6.9 alone as the deployed surrogate's numerical bound. Nor is the RKHS norm of a fit to cross-fitted residual labels necessarily a lower bound on `||G-m_full||_K`; those are different labels. The existing disclaimer is helpful but does not cure every application of the theorem elsewhere.

### 4. Remove categorical representation claims that the pullback identity does not imply

The statement that a stationary kernel of fixed smoothness “reaches only its native space” confuses the space containing each fitted function with the closure of the class it can approximate. A Matérn native space on a suitable bounded domain contains smooth functions, and smooth functions can approximate many continuous targets that do not themselves belong to that Sobolev space. Failure of the stated native-space bound is not an impossibility of approximation.

Similarly, “no rescaling of the input can imitate” learned features is false without a restriction on the feature map: take `phi(x)=Ax` with A diagonal. The resulting pulled-back kernel is exactly a rescaled-input kernel. For nonlinear features the tested coordinatewise metric is a smaller family, but the empirical superiority here does not establish a universal impossibility for all input metrics.

Suggested replacement: “The tested feature map changes the kernel's geometry beyond the coordinatewise rescalings evaluated here. The pullback identity provides a conditional mechanism for a smaller target norm when the target factors through these features; it does not establish that factorization, its norm advantage, or a universal separation from raw-input kernels.”

Also soften “This is not an accident of the O2 band; it is what a pulled-back kernel does,” “training drives phi toward exactly the factorization,” and the discussion's claim that the measured fitted-norm drop is *the quantitative reason* residual correction works. The proposition proves none of these empirical causal directions. It permits them and helps interpret diagnostics under explicit conditions.

### 5. Do not infer a true norm reduction from a fitted norm reduction

The main text commendably says the numbers are lower-bound diagnostics, but elsewhere calls the residual norm drop the gain guaranteed by the theorem. Use a conditional: *if* the neural mean reduces the true native-space residual norm, it reduces this fixed-design bound.

**Counterexample showing why the condition matters:** For a strictly positive-definite kernel and a query outside the design, let `g_u` be the interpolation-error representer from the minimax proof. It vanishes at every training input and is nonzero at the query. Set `m=G+A g_u`. Then all training residuals vanish, their fitted RKHS norm is zero, but the true residual norm grows without bound as A grows and the off-design prediction can be arbitrarily poor. Small fitted norm cannot certify either a small full norm or favorable extrapolation.

### 6. Complete proof pointers and small logical qualifications

The requested first-page GitHub footnote should explicitly direct the reader to `paper/supplement.pdf` for the supplement and proofs. Each theorem, proposition, corollary, and substantive numbered result with a supplementary proof should carry a local proof pointer, preferably to the precise supplementary section, rather than relying only on the Section 6 introduction.

Two additional nearby qualifications are required: Section 6.5 must retain the almost-surely-distinct-score condition for the upper conformal coverage bound and the iid continuous-score condition for the Beta/Beta-binomial law; these assumptions appear in the supplement but are omitted from the main displayed summary. The rule for the conformal quantile also needs the usual infinity convention when its requested order statistic exceeds the calibration size. Proposition 6.7's strict derivative inequality excludes perfect anticorrelation: at correlation -1 the derivative with respect to the error ratio is zero, despite an interior optimum. Add `-1<rho<min(t,1/t)` when claiming strict positivity.

## Publication recommendation after revision

I would support submission of a revision that implements these changes and retains the empirical scope already disclosed. The correct contribution is a replicated comparison and diagnosis using familiar mathematical tools. It should not be promoted into a theorem that learned features must beat metrics, a guarantee that accurate neural means reduce native norms, or a minimax certificate for the deployed tuned predictor. No new experiment is demanded by this review if those unsupported claims are withdrawn and the unmatched correction bound is stated honestly.
