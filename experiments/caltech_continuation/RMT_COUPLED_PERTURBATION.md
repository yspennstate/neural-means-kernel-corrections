# Covariance changes and prediction changes are coupled

This is a finite algebraic analysis for the next controlled RMT study. It does not establish the cause of the retained Caltech RIE errors, reproduce their magnitude, or change the historical result fields.

Fix one output pixel and a centered calibration design. Let `C` be its positive definite member covariance and `c` its member-target cross-covariance. Write `w=C^{-1}c`. If a changed calculation uses the symmetric matrix `C+E` and vector `c+e`, then, whenever `C+E` is invertible,

```text
changed_w - w = (C+E)^{-1}(e-Ew).
```

This follows by multiplying both sides by `C+E` and using `Cw=c`. If the smallest eigenvalue of C is `sigma>0` and `||E||_2<sigma`, then

```text
||changed_w-w||_2 <= (||e||_2 + ||E||_2 ||w||_2)/(sigma-||E||_2).
```

Indeed, for every unit vector v, `v^T(C+E)v >= sigma-||E||_2`, which bounds the inverse norm. This is a sufficient perturbation bound, not a claim that the historical covariance meets its hypothesis. It gives no conclusion when the denominator is nonpositive.

The coupled numerator matters. In an exact noiseless linear relation, changing a covariance and its cross-covariance consistently can give `e=Ew`, leaving the fitted coefficient unchanged. Cleaning only C keeps `e=0` and generally changes the coefficient. This explains the logical gap in using covariance error alone as a regression guarantee; the separate exact-population-covariance counterexample in `RMT_RESEARCH_CONTROLS.md` already shows that even correcting C perfectly can worsen prediction when c is left at its sample value.

There is also a direct spectral calculation. Suppose `C=V diag(mu_i) V^T` and the cleaner replaces its eigenvalues by positive `xi_i`, with the same orthonormal V and unchanged c. In the noiseless relation `c=C beta`, the cleaned coefficient is

```text
changed_w = V diag(mu_i/xi_i) V^T beta.
```

Thus the relevant multipliers depend on the target's coordinates in the eigenbasis. Their median, or the median of `abs(xi_i/mu_i-1)`, does not measure prediction risk. Ridge's replacements `xi_i=mu_i+rho`, for `rho>0`, have multipliers in `[0,1]`; a general nonlinear cleaner need not. That fact is not a proof that ridge has lower prediction error or that the RIE formula is incorrect.

An exact three-member witness makes the median limitation concrete. Let the predictors be independent signs taking values `(+/-1, +/-1, +/-sqrt(epsilon))`, each of the eight combinations equally likely, and let the target be the third predictor. The mean is zero, `C=diag(1,1,epsilon)`, and `c=(0,0,epsilon)`. Replacing only the third covariance eigenvalue by `2 epsilon` changes the coefficient from `(0,0,1)` to `(0,0,1/2)`. The median relative eigenvalue change is zero, while the relative prediction error is exactly 50% for every case. Independently, the squared-error expectation is `epsilon/4`, the target's second moment is `epsilon`, and their square-root ratio is `1/2`. The relative Frobenius covariance change, `epsilon/sqrt(2+epsilon^2)`, tends to zero as epsilon tends to zero. This witness concerns a chosen spectral modification, not the specific Bun--Bouchaud--Potters cleaner or the retained data.

For the actual per-pixel affine fit, both calculations restore the intercept from their common calibration means. With those means held fixed, prediction differences on evaluation rows are `Z_eval (changed_w-w)`, where Z_eval is centered by the calibration predictor mean. Their mean square equals `delta_w^T M_eval delta_w`, with `M_eval=Z_eval^T Z_eval/n_eval`. This concerns differences between predictions; it is not the whole-field relative-error metric used by the study.

For that whole-field metric, the reverse triangle inequality gives a separate exact bound. For aligned predictions p_i, q_i and nonzero target vectors y_i,

```text
abs(mean_i ||p_i-y_i||/||y_i|| - mean_i ||q_i-y_i||/||y_i||)
  <= mean_i ||p_i-q_i||/||y_i||
  <= sqrt(mean_i ||p_i-q_i||^2) sqrt(mean_i 1/||y_i||^2).
```

The second step is Cauchy--Schwarz. Pixel contributions must be accumulated into each whole-case norm before this metric is averaged. A zero target norm needs an explicit metric policy; the prepared streaming evaluator rejects it.

The next authorized comparison should therefore preserve identical stored predictor values, means, case splits and target mapping; isolate covariance arithmetic precision; record minimum/maximum eigenvalues, floor counts, coefficient norms, the original normal-equation residual `c-C changed_w`, and target-aligned validation error. The original `rie_weights` already centers predictors and targets per pixel and restores an intercept. Centering itself is not a missing feature or a new correction. Parameters and any added stabilization must be selected within training folds, and any reuse of the historical 19,000 evaluation rows must remain explicitly labeled.
