# Independent reviewer 05

**Vote on publishing the submitted version: NO.**

The paper is plausibly publishable as a careful empirical study after a focused revision. My vote is against publishing this version unchanged, principally because the main-text claims about its RKHS and conformal guarantees are stronger than the statements actually proved. The central algebraic power-function and minimax results survive this audit. I do not interpret a vote as a prediction of acceptance by a particular journal.

## Scope and independence

I read the complete 39-page submitted manuscript through its extracted text, then independently checked the source and the relevant supplement statements/proofs: Proposition 6.8, Theorems 6.9–6.10, Proposition S3.1, Remark S3.2, and their proofs in S8. I have not reproduced the expensive learning experiments or audited every released numerical record. The source was the supplied PDF and its linked repository, https://github.com/yspennstate/neural-means-kernel-corrections, which was available during this review. This is my own assessment, made without adopting other reviewers' conclusions.

## What is mathematically correct

1. **The feature pullback identity is correct.** With a base RKHS on the feature space and a map phi from the full input domain, the image of the composition map has the stated quotient norm. Its kernel is the pullback kernel. The kernel of the composition map is closed, being the intersection of the nullspaces of bounded evaluations; consequently the minimizing lift exists and the quotient space is complete. The proof would benefit from that one sentence, but there is no substantive obstruction here. Coordinatewise extension to the finite-dimensional output space, with the root sum of squared RKHS norms, is valid.

2. **The sharp regularized error factor in Theorem 6.9 is correct.** Put t=n lambda, a=(K+tI)^(-1)k_u, and g_u=k(u,.)-sum_i a_i k(u_i,.). Then

   ||g_u||_H^2 = k(u,u)-k_u^T(K+tI)^(-1)k_u-t||a||_2^2.

   This is exactly the displayed squared tilde-power function. The vector bound follows by summing squared coordinatewise Cauchy–Schwarz bounds; choosing one residual coordinate proportional to g_u proves sharpness. The result is deterministic and remains valid for a data-dependent mean once that realized mean and its residual are fixed and RKHS membership is assumed. Independence of the mean from the sample is unnecessary for this algebraic assertion.

3. **Theorem 6.10 is correct.** The pair of zero-data residuals plus/minus rho g_u/P_0(u), placed in one output coordinate, proves the minimax lower bound for arbitrary estimators of the noiseless sample. Interpolation attains it. The ridge excess is

   tilde P_lambda(u)^2-P_0(u)^2 = t^2 k_u^T(K+tI)^(-2)K^(-1)k_u >= 0.

   Thus interpolation is minimax over the stated native-space ball, whereas ridge has an explicit positive regularization cost. The result is classical optimal recovery, correctly credited as such. No claim of new general minimax theory should be made.

4. **The supplement's conformal lower bound and proof are correct when its quantile is defined.** Exchangeability gives marginal coverage at least 1-alpha. The supplement correctly conditions the upper bound on distinct scores. The i.i.d., continuous-score version of the Beta/Beta-binomial statement is also correct. In finite-dimensional output space the reported set is a Euclidean ball, not simultaneous coordinatewise intervals; the supplement explicitly says so.

## Required corrections

### 1. Main-text conformal hypotheses are incomplete

Section 6.5 states the two-sided coverage inequality and exact Beta and Beta-binomial laws after saying that coverage needs exchangeability alone. Exchangeability alone gives the lower bound, not all these assertions. An elementary counterexample to the two-sided bound without distinctness is an exact predictor: every score is zero and coverage is one, exceeding 1-alpha+1/(m+1) for the paper's alpha. Similarly, an i.i.d. sample can have atoms, so a random split of i.i.d. inputs does not by itself establish continuity of its nonconformity scores.

**Minimal repair:** Carry the supplement's a.s.-distinct condition into the main-text upper bound. State that the Beta/Beta-binomial comparison additionally assumes conditionally i.i.d. continuous scores for a fixed fitted predictor. Treat the quoted central intervals as that model's reference intervals unless continuity is justified or ties are explicitly randomized. The genuine lower coverage guarantee survives. Say that the Beta distribution is the distribution *across calibration samples* of conditional coverage; once a calibration sample is fixed, its conditional coverage is a number, not a Beta random variable.

The quantile definition also needs the standard endpoint convention: when k=ceil((1-alpha)(m+1))=m+1, set the quantile to infinity. The exact Beta law is only stated for k<=m; the endpoint gives coverage one. This does not alter any reported result at alpha=0.05 or 0.10.

### 2. The fitted mean must match the residual rows for the norm interpretation

Section 6.4 acknowledges that the correction uses out-of-fold mean predictions on training rows but is deployed above a full-data mean. This is good disclosure, but a cross-fitted row matrix is not automatically a globally defined function m_cf in the RKHS setting. The bare product bound does not apply to the shipped predictor, and the norm of the fit to cross-fitted residuals is not automatically a lower bound on ||G-m_full||_K.

**Minimal repair if cross-fitting is what the released implementation uses:** Let M_cf be its out-of-fold row matrix, Delta=m_full(X)-M_cf, r_full=G-m_full, and a_lambda(u)=(K+n lambda I)^(-1)k(X,u). Then the shipped error is

   r_full(u)-a_lambda(u)^T r_full(X)-a_lambda(u)^T Delta,

   and its norm is at most

   ||r_full||_K tilde P_lambda(u)+||a_lambda(u)^T Delta||_2

   <= ||r_full||_K tilde P_lambda(u)+||a_lambda(u)||_2||Delta||_F.

State this explicit bound, and either measure the mismatch or label it unmeasured. If source inspection shows the production residuals actually use in-sample predictions, correct the cross-fitting description instead. Do not assert that the measured fitted norm lower-bounds the full-data residual norm unless the residual rows coincide or the discrepancy is accounted for. The empirical diagnostic can remain, honestly described as the RKHS norm of the fitted function.

### 3. The representation discussion overclaims what pullback proves

Page 29 says a stationary kernel of fixed smoothness “reaches only its native space.” As a claim about approximation, this is false: a sequence of functions in a Matérn RKHS can approximate targets outside that RKHS. What is restricted to native-space targets is the particular norm-based error estimate. A small fitted interpolation norm is likewise only a lower-bound diagnostic and cannot establish that the unknown target norm is small.

The pullback identity supplies a conditional norm upper bound when a factorization G=h composed with phi exists with h in the base RKHS. It does **not** predict that this norm is smaller than the raw-input norm, that neural training finds such a factorization, or that no input rescaling could reproduce the observed advantage. Those are empirical interpretations or hypotheses, not implications of Proposition 6.8. The main text's later diagnostic caveats help but do not erase the categorical statements.

**Minimal repair:** Replace “reaches only its native space” with “the stated native-space guarantee applies only when the target has finite norm in that space.” Describe the measured raw-versus-feature comparison as consistent with a favorable representation, and remove claims that pullback mathematically predicts the sign or magnitude of the measured gap. Preserve the observed 38-fold diagnostic and its scale sensitivity as measurements.

### 4. Small theorem/proof cleanup

- Proposition 6.8 uses an undefined input domain mathcal X, while Section 6 fixes U. Explicitly declare phi:U->Z, k on Z, and require the lifting equality on **all of U**, not just the finite training design X. This avoids confusing the RKHS norm with a finite-design interpolation norm.
- In S8.19 the sharpness construction defines r_1=c g_u but then writes r_1(X)=g_u(X). Insert the same scale c=||G-m||_K/tilde P_lambda(u). The intended proof and theorem are unaffected.
- For lambda=0 with a singular Gram matrix, add the usual statement that k_u belongs to range(K), which follows from positivity of the augmented Gram matrix; this makes the pseudoinverse extension transparent.
- Theorem 6.9 is deterministic. Remove the unnecessary independence phrase for m from its setup and retain the realized-residual RKHS assumption.
- Avoid describing the posterior standard deviation P_lambda itself as the exact worst-case error for positive lambda. The exact factor is tilde P_lambda; the paper's display is correct, but nearby prose should preserve that distinction.

## Overall publication assessment

The manuscript has a coherent empirical question, useful replication, and unusually candid reporting of negative outcomes and conditional guarantees. Its most credible contribution is the controlled comparison and error-correlation diagnosis, not novelty of the elementary RKHS/ensemble identities. It is too categorical in several passages about what those identities establish. The required mathematical revisions are limited and feasible without inventing new experiments. A carefully revised submission could receive **YES for submission/public circulation** from me, conditional on the numerical provenance and protocol checks outside this review. I cannot promise journal acceptance or certify unperformed reproductions.
