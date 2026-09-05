# Independent revision audit 19

**Verdict on the new finite-design rank-truncation result: PASS.**

I read the revised `prop:aniso` in `overleaf/supp_theory_rules.tex`, its complete proof under `proof:prop:aniso` in `overleaf/proofs.tex`, and the revised main theory section. This is a read-only audit; I made no manuscript edits. My original unchanged-publication vote remains recorded in `review19.md`; this report assesses the revised mathematics.

## Rank-truncation proposition

The assumptions and conclusion are now consistent. If `||u||<=R`, `G(u)=g(Au)`, and `g` is Euclidean `L`-Lipschitz, truncated SVD gives

`||G(u)-g(A_r u)|| <= L ||A-A_r||op ||u|| <= LR sigma_(r+1)=epsilon`.

With `phi_r(u)=Sigma_r V_r^T u` and `h(z)=g(U_r z)`, the function `G_r=h o phi_r` has exactly the stipulated feature-space RKHS representation. Applying the deterministic power-function bound to `h` gives the first term. The exact identity

`G(u)-c^T G(X) = G_r(u)-c^T G_r(X) + delta(u)-sum_i c_i delta(u_i)`

gives the added error `epsilon(1+||c||1)`. No additional factor involving output dimension, feature dimension, or sample size is missing: the Lipschitz and residual bounds are already in Euclidean output norm.

I specifically checked the following boundaries:

- For `lambda>0`, the matrix `K+n lambda I` is invertible even if projected inputs repeat or `K` is otherwise singular.
- For `lambda=0`, the stated pseudoinverse convention is valid. Kernel evaluation vectors lie in the range of the positive-semidefinite Gram matrix. The power-function argument remains valid for the compatible `h` labels, while the arbitrary discrepancy in the exact `G(X)` labels is handled by the `epsilon ||c||1` term.
- Different original inputs may have the same projected input and inconsistent exact target values. This does not invalidate the result; the estimator uses pseudoinverse regression in that case and the discrepancy term explicitly covers this situation.
- The proposition states `1<=r<=d`, so it makes no unsupported assertion about an RKHS on a separately defined rank-zero space. If `A` itself has rank zero, the result can still be used with the stated positive feature dimension whenever its other assumptions hold.
- At `r=d`, the convention `sigma_(d+1)=0` gives no truncation term. For lower ranks with vanishing trailing singular values, the same reduction holds.
- The global Lipschitz assumption guarantees both `g(Au)` and `g(A_r u)` are in its domain. The bounded input set controls their discrepancy. RKHS membership is an explicit additional assumption, not inferred from Lipschitz continuity.
- The result makes no sampling-rate or comparative-performance assertion, and the post-result paragraph correctly states what additional controls would be required for such conclusions.

The proof is complete at the stated level. Its direct use of Theorem 6.9 is appropriate, and the statement now contains a local pointer to its supplement proof.

## Main-theory consistency

The revised text implements the substantive repairs from my original review:

- normalized-error expectations and positive target norms are stated;
- the two-member correlation formulas require positive member error, with exact predictors treated separately;
- the coefficient-of-variation identity now explicitly handles zero mean error;
- Corollary 6.4 requires a positive baseline error and correctly bounds the optimal convex risk, not arbitrary fitted weights;
- the exchange-rate proposition excludes perfect anticorrelation from strict monotonicity, explains the boundary, and removes the false threshold-divergence interpretation;
- the margin proposition uses an explicit binary decision and correctly treats ties;
- the single-mean bound is deterministic and no longer unnecessarily requires independence;
- the recorded-label mismatch term is derived explicitly using the deployed mean and training-row means, without asserting that the latter define one global cross-fitted function;
- the text distinguishes fitted RKHS norms from unknown norms of a deployed residual;
- the native-space discussion no longer imposes an incorrect absolute approximation barrier or asserts an automatic favorable pullback comparison;
- the conformal paragraph now distinguishes exchangeability, continuity, predictor independence, the infinite-quantile boundary, and the scope of its exact coverage law;
- Theorem 6.9's stated worst-case error includes the radius factor, consistent with its proof and Theorem 6.10.

I found no remaining blocking mathematical defect in the new result or these repaired statements. This does not constitute an independent rerun of the empirical records, a citation verification, or a prediction of editorial acceptance.

## Minor final wording sent to the coordinating reviewer

At the time of this read, one paragraph in the representation discussion still said the bound “factors the error as” the norm times the power function and called the design factor “set by the Gram spectrum.” Prefer “bounds the error by” and “determined by the kernel, training inputs, and query.” A query's cross-kernel vector is needed; the training spectrum alone does not determine its out-of-sample power function. Likewise, “smaller coefficient norm” should read “smaller fitted RKHS norm” for the quantity `tr(alpha^T K alpha)`. These are local wording fixes and do not change the passed theorem.

**Revised mathematical recommendation:** suitable to retain in the submission after this final wording cleanup. The replacement finite-design theorem is a valid, useful result and is a substantial improvement over the unsupported rate heuristic.
