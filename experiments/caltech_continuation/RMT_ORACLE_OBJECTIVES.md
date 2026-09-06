# Which oracle is optimal for which loss?

This note balances the counterexamples in `RMT_RESEARCH_CONTROLS.md` and `RMT_COUPLED_PERTURBATION.md`. Covariance cleaning can be appropriate for a prediction objective under stated conditions. The finite calculations below do not establish those conditions for the retained Caltech ensemble, estimate an implementable cleaner, or supply a new performance result.

Fix a real positive definite matrix C and an orthonormal basis U with columns u_i. Consider estimators restricted to that basis. The covariance estimator `U diag(d_i) U^T` minimizing squared Frobenius distance to C has `d_i=u_i^T C u_i`: after rotating by U, only diagonal terms depend on d. Likewise, the precision estimator minimizing squared Frobenius distance to `C^{-1}` has `a_i=u_i^T C^{-1}u_i`. These a_i generally differ from `1/d_i`. Cauchy--Schwarz gives `d_i a_i >= 1`, with equality when u_i lies in an eigenspace of C.

A different objective gives a different precision oracle. Let c be a random vector whose conditional second moment, given C and U, is R. For `A=U diag(a_i) U^T`, consider the prediction-weighted coefficient discrepancy

```text
J(a) = E[(A c-C^{-1}c)^T C (A c-C^{-1}c) | C,U].
```

This is an expectation over c with C and U fixed; it is not the empirical whole-field relative error. Let `C_U=U^T C U` and `R_U=U^T R U`. Expanding the quadratic gives

```text
J(a) = a^T B a - 2 r^T a + tr(C^{-1}R),
B_ij = (C_U)_ij (R_U)_ij,       r_i=(R_U)_ii.
```

To check the expansion, the first term is `tr(A C A R)`, and both cross terms equal `tr(A R)`. If B is positive definite, the unconstrained optimum solves `B a=r`. If positive a_i are required and this solution is not positive, the optimum must instead be found on the constrained domain. No positivity issue occurs in the isotropic case below.

When `R=I`, B is diagonal with entries d_i and r is the all-ones vector. Consequently, `a_i=1/d_i` is the unique optimum. Equivalently,

```text
J(a) = sum_i d_i (a_i-1/d_i)^2 + tr(C^{-1}) - sum_i 1/d_i.
```

This second derivation proves optimality by completing squares. Inverting the Frobenius-optimal covariance oracle is therefore exactly right for this specified prediction-weighted loss and isotropic second moment. It is not the Frobenius-optimal precision estimator. The distinction is the loss, not a contradiction.

An exact two-dimensional check uses `C=[[2,1],[1,2]]` and `U=I`. The covariance oracle is `2I`; its inverse is `I/2`. The Frobenius-optimal precision oracle is `2I/3`. With `R=I`, the prediction-weighted losses are respectively `1/3` and `4/9`, since `J(aI)=4a^2-4a+4/3`. Their squared Frobenius precision errors go in the opposite order: `5/18` and `2/9`. These values also follow directly from `C^{-1}=[[2,-1],[-1,2]]/3` and its two eigenvalues. Thus even an exact oracle should be named together with its objective.

Direction assumptions change the answer. In the same example, put `R=[[1,rho],[rho,1]]` for `-1<=rho<=1`. Then `B=[[2,rho],[rho,2]]` is positive definite, and the optimum is `A=I/(2+rho)`. At rho=1, c lies in C's eigenvalue-three direction and `A=I/3` has zero discrepancy on that direction. At rho=-1 it lies in the eigenvalue-one direction, and `A=I` does. The isotropic rho=0 case gives `I/2`. This follows both from the two-variable normal equations and by evaluating the corresponding signal direction.

The [Bun--Bouchaud--Potters review, Sections 7.1.2--7.1.3](https://arxiv.org/pdf/1610.08104), likewise makes distribution and independence assumptions when deriving its oracle result for a different, normalized risk objective. That cited result is not the finite J(a) calculation above. It motivates checking the assumptions and the loss before transferring an oracle claim.

For the retained regression study, c is estimated from the same calibration predictions that determine the sample covariance and its eigenbasis. Conditional isotropy has not been established. Its finite-bandwidth RIE also estimates an unobserved oracle rather than knowing C. The original RIE routine already centers each pixel and restores the intercept; that routine remains in the driver pinned at `7f641043219469d66b686205ea11c86d6c59571a`. Honest predictive validation, arithmetic controls and explicit reused-evaluation scope remain necessary. None of these calculations justifies replacing the recorded Caltech error by a theoretical value.
