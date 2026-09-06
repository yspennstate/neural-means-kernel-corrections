# Deterministic checks for a fixed finite Gram spectrum

Assume a fixed positive semidefinite `n × n` Gram matrix `K`, a fixed `lambda > 0`, and eigenvalues `mu_i >= 0`. Define `t = n lambda`, `f_i = mu_i/(mu_i+t)`, and `H = K(K+tI)^{-1}`. Then `d(lambda) = tr(H) = sum_i f_i`. These statements concern a fixed matrix and penalty; they do not give degrees of freedom for a procedure that also selects its kernel or penalty using the targets.

Differentiating each term gives

```text
-lambda d'(lambda) = sum_i f_i(1-f_i)
-d log d / d log lambda = 1 - tr(H^2)/tr(H), when tr(H) > 0.
```

Thus the positive local decay exponent is in `[0, 1]`. This is an exact finite identity. It does not require a power-law eigenvalue model. A second derivation uses `H = I-t(K+tI)^{-1}`: differentiating with respect to `t` gives `-t H'(t) = H-H^2`, and taking traces yields the same relation. The spectrum checker compares the identity with a central difference on the logarithmic lambda axis.

For a finite positive definite matrix, `d(lambda)` approaches `n` as `lambda` tends to zero. For sufficiently large lambda, it behaves as `tr(K)/(n lambda)`. Consequently, an intermediate fitted slope such as `-0.46` cannot be a fixed-matrix asymptotic power law at both ends. A rank-window regression and a lambda-window regression are finite diagnostics whose window boundaries must be stated.

Suppose only the top `k < n` ordered eigenvalues and the total trace are retained. Put `B = mu_k > 0`, `r = tr(K)-sum_{i<=k} mu_i`, `m = n-k`, and `D_k = sum_{i<=k} mu_i/(mu_i+t)`. Feasibility requires `0 <= r <= m B`. Concavity of `x/(x+t)` yields

```text
D_k + r/(B+t) <= d(lambda) <= D_k + r/(r/m+t).
```

The upper bound is Jensen's inequality and is attained when all omitted eigenvalues equal `r/m`. A sharper lower bound fills as many omitted eigenvalues as possible to `B`: with `a = floor(r/B)` and `b = r-aB`, it is `D_k + a B/(B+t) + b/(b+t)`. Concavity shows that transferring mass from the smaller to the larger of two unsaturated entries cannot increase the sum; repeating that transfer gives the stated concentrated tail. This is attained by `a` entries equal to `B`, one equal to `b` when nonzero, and the remaining entries zero.

The historical Lanczos artifact uses the looser upper bound `D_k + min(r/t, m)`, which is valid. A tighter finite-count bound is an improvement, not evidence that the historical upper bound was wrong. Numeric intervals derived from rounded Ritz values and floating-point traces are conditional on those inputs; they are not interval-arithmetic certificates of the original Gram solve.

An interval `0 < L <= d <= U` also gives an explicit stopping criterion. The estimate `2 L U/(L+U)` minimizes the worst-case ordinary relative error `|estimate/d-1|` over that interval. Its maximum is `(U-L)/(U+L)`: the endpoint errors are equal, and moving the estimate in either direction increases one of them. Thus a prefix is sufficient for a requested relative tolerance `epsilon` when `(U-L)/(U+L) <= epsilon`. This statement assumes valid bounds; it does not convert unverified floating-point Ritz values into certified bounds. The arithmetic midpoint instead minimizes worst-case absolute error, while the geometric midpoint minimizes worst-case logarithmic error. Those are different objectives.

Trace concentration also differs from effective dimension concentration: a large eigenvalue contributes almost one to `d`, while many small eigenvalues can contribute appreciably in aggregate despite carrying little trace. A top-eigenvalue trace percentage alone cannot substitute for evaluating the regularized spectral sum.

The retained 6,000-row control uses the same standardization and length scale as the full design. The recovered CPU source constructs the subset after drawing the median-distance sample from the same seeded generator and compares its rows against the paper's saved design. Thus it is a principal-submatrix control before normalization. Dividing the full eigenvalues by 19,000 and the subset eigenvalues by 6,000 compares normalized finite spectra; their observed agreement alone proves no limiting spectral law.

Effective dimension also does not determine the largest ridge leverage score. In [Bach's uniform-column Nyström theorem, Section 4.1 and Theorem 1](https://proceedings.mlr.press/v30/Bach13.pdf), the relevant quantity is `n max_i H_ii`, which can exceed `tr(H)`. The stated risk comparison averages over the random columns and response noise in a fixed-design setting. Its sufficient sampling condition should not be replaced by the trace alone. This paragraph records the theorem's scope, not a new assertion about the manuscript's rank choices.

An explicit algebraic witness is `K_A = diag(1,0)` and `K_B = [[1,1],[1,1]]/2`, with `n=2` and `lambda=1/2`. Both have eigenvalues `(1,0)` and `tr(H)=1/2`, but `2 max diag(H)` equals `1` for A and `1/2` for B. Their uniform one-column Nyström behavior differs too. For A, one sampled column reconstructs K and the other is zero; for B either column reconstructs K. With a unit noiseless signal in each matrix's eigenvalue-one direction, full ridge risk `||prediction-signal||^2/2` is `1/8`. The expected one-column risk is `(1/8+1/2)/2=5/16` for A and `1/8` for B. This checks the distinction directly through predictions as well as smoother diagonals. These are generic positive semidefinite matrices, not unit-diagonal replicas of the structural-mechanics kernel or a Caltech performance experiment.
