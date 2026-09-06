# The loss and constraint determine the covariance problem

This is a finite algebraic clarification, not a new Caltech result or a proposed change to the corrected historical driver. The September 5 RMT campaign already records the distinction between squared relative error and mean relative error, including its P47 dispersion comparison and later calibration-only ratio-metric selector. The calculations here make the relevant assumptions explicit.

For one output, let P contain one prediction per row and member, let y be the target, and define E = P - y 1^T. For any weights satisfying 1^T w = 1,

```text
P w - y = E w.
```

This cancellation is exact and requires no independence between target and member errors. With positive row weights a_i and a fitted intercept b, minimizing sum_i a_i (E_i w + b)^2 gives b = -mu_a^T w, where mu_a is the weighted mean error vector. After division by sum_i a_i, the minimized criterion is w^T C_E,a w, the weighted centered error covariance quadratic. Thus the error-covariance formulation is exact for this constrained quadratic fit.

Without the sum-to-one constraint, the residual instead contains the additional term (1^T w - 1)y. An intercept removes a constant mean, not this varying signal term. Also, the centered prediction covariance is generally

```text
Cov(P) = Var(y) 1 1^T + 1 Cov(y,E)^T
         + Cov(y,E) 1^T + Cov(E).
```

The two cross terms cannot be dropped merely by naming E an error. Consequently, moving from unconstrained affine regression on predictions to constrained minimum variance of errors changes the estimator's feasible family. It is not just a cleaner applied to a different matrix.

For whole-field residuals r_i and nonzero target norms, write rho_i = ||r_i|| / ||y_i||. The reported mean relative error L and the squared-relative surrogate Q satisfy

```text
L = mean_i rho_i,
Q = mean_i rho_i^2 = L^2 + Var_i(rho_i),
L <= sqrt(Q).
```

The variance uses divisor n. These identities do not give an ordering of L from an ordering of Q. In particular, reducing dispersion can lower Q while raising L.

An exact six-case, one-output example also permits a fitted intercept. Every target equals 1. Let the two members' residual vectors be

```text
A = (0, 0, 12/5, 0, 0, -12/5),
B = (1, 1, 1, -1, -1, -1).
```

Use weight w on A and 1-w on B, with 0 <= w <= 1. Both vectors have zero mean, so the quadratic-optimal intercept is zero. Their mixture has paired opposite residuals of magnitudes 1-w, 1-w and 1+7w/5. Zero is also a median, hence an optimal intercept for the absolute-error objective. Therefore allowing an intercept does not alter either minimum below:

```text
L(w) = 1 - w/5,
Q(w) = 1 - 2w/5 + 33w^2/25.

w_L = 1,       L(w_L) = 4/5,    Q(w_L) = 48/25;
w_Q = 5/33,    L(w_Q) = 32/33,  Q(w_Q) = 32/33.
```

The Q minimizer follows independently by differentiating the polynomial or completing its square. At w_Q, direct averaging of the three paired magnitudes 28/33, 28/33 and 40/33 verifies both stated losses. This is an objective counterexample, not a sample from any retained scientific corpus. The surrogate optimum can be useful without being the reported-loss optimum.

The constraint nevertheless provides a real stability guarantee for its own quadratic objective. If every feasible w satisfies ||w||_1 <= c, delta = max_ij |S_hat_ij - S_ij|, and the computed w_hat is within epsilon of the estimated minimum, then

```text
|w^T (S_hat-S) w| <= c^2 delta,
Q_S(w_hat) - min_w Q_S(w) <= 2 c^2 delta + epsilon.
```

Expand the first quadratic entry by entry. For the second bound, insert the estimated objective at w_hat and at a true minimizer; the two matrix discrepancies each cost c^2 delta and approximate optimization costs epsilon. This is the deterministic bound of [Fan, Zhang and Yu, Theorem 1](https://fan.princeton.edu/sites/g/files/toruqf5476/files/documents/Portfolio-JASA.pdf), with explicit optimization slack. The simplex has c=1. Neither a covariance estimation rate nor an optimality guarantee for L follows without further assumptions. The six-case example has no covariance estimation error and still separates the two loss optima.

In the helper retained at commit 7f641043219469d66b686205ea11c86d6c59571a, `convex_weights` minimizes the global squared-relative residual quadratic. It has no additional fitted intercept. `select_shrinkage` then scores training-fold-fitted candidate predictions using mean whole-field relative error. This note does not change that protocol or invalidate the independently reviewed fold separation. It prevents interpreting the inner quadratic fit as direct minimization of the outer reported score.

Prior work consulted: the complete brain nodes `rmt_stacking_cleaned_the_signal_gram_2026_09_05`, `rmt_stacking_loss_mismatch_and_rie_in_the_right_metric_2026_09_05`, `long_only_makes_the_cleaner_irrelevant_2026_09_05`, and `rmt_for_the_ensemble_the_ten_hour_block_2026_09_05`. The last node's bytes read on September 6 have SHA256 f6754ed6d1a5960352b7b38be0b26fa85e856662ad722d2d12aa8e9fc2425409. Its later P47/P54/P58/P60 entries govern the research continuation rather than its earlier provisional recipe. Those entries are source leads; this note does not independently authenticate their reported DGX measurements.
