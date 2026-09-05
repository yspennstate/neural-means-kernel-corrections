# Independent review 19

**Vote on publishing the attached version unchanged: NO.**

**Recommendation:** revise and resubmit. The paper has a credible empirical contribution and most of its central algebraic results are correct. My NO concerns several precise mathematical and explanatory defects, plus claims of applicability that are stronger than their stated assumptions. These are repairable without replacing the reported experiments. I would support submission after the repairs below, subject to the separate audits of numerical provenance and citations.

I independently read the original 39-page manuscript text and its relevant original supplement sources: `theory.tex`, `supp_theory_rules.tex`, `supp_theory_stages.tex`, `supp_theory_conf.tex`, `supp_theory_deff.tex`, and `proofs.tex`. I did not inspect other reviewers' reports. This review evaluates mathematical logic and its interpretation; it does not independently rerun the training campaigns or certify journal acceptance.

## What survives

The second-moment identity, positive-definite signed-stack optimizer, entrywise convex-ensemble lower bound, and seeds-versus-architectures bound all follow by the supplied elementary arguments. The distinction between mean relative error and root-mean-square relative error is valuable and generally handled correctly. The empirical pool lower bound is a bound on that measured pool and metric, not on all possible surrogates.

The feature-pullback identity is mathematically sound. The sharp regularized power-function factor is correct: writing `c=(K+n lambda I)^(-1)k_u`, the evaluation residual representer has squared norm `P_lambda^2-n lambda ||c||^2`. The minimax lower bound follows from opposite functions vanishing on the design, and the positive-nugget excess identity is also correct. The kernel-weight bound follows correctly from the Schur complement and `mu/(mu+t)^2 <= 1/(4t)`.

The finite-candidate and affine-class capacity arguments have the expected structure under their boundedness, independence, and objective-matching assumptions. The manuscript already acknowledges that these assumptions do not hold for every selection in the shipped pipeline, and that observed maxima do not establish population bounds. Those qualifications should be retained and made consistent with the stronger nearby prose.

## Required mathematical repairs

### 1. Proposition 6.7 has a false strict inequality at perfect anticorrelation

The hypothesis allows `rho=-1`, because it only requires `rho < min(t,1/t)`. For `e1=1`, `e2=2`, and perfectly anticorrelated residuals, an interior convex mixture cancels the error exactly. Thus `V=0` for every positive `t`, and `partial V/partial t=0`, contradicting the displayed strict positivity. The derivative formulas themselves are correct.

**Minimal repair:** assume `0<e1<=e2` and `-1<rho<1/t` in this proposition; mention that at `rho=-1` the mixture has zero risk and the error derivative is zero. Alternatively change the first strict inequality to a non-strict one and describe the equality case.

The following interpretation also needs correction. The manuscript says the exchange rate becomes large near the admission threshold because `t-rho` is small. In the actual regime `t>=1`, the fractional coefficient is

`(1-rho^2)/(t-rho) <= 1+rho <= 2`.

For fixed `t>1`, as `rho` approaches the threshold `1/t`, this coefficient tends to `1/t`; there is no asserted blow-up. Replace that sentence by a finite local trade-off statement. The empirical example of accuracy gains being offset by increased alignment remains legitimate.

### 2. The threshold-margin proposition needs an explicit binary tie convention

Proposition S2.1(ii) is stated using `sign(hat D) != sign(D)` and claims no condition on `D`. Its proof instead uses a binary decision that treats `D=0` as the non-mixing class. With the ordinary mathematical sign function these are different statements.

**Counterexample:** let `D=0`, let `hat D` equal `+sigma` or `-sigma` with equal probability, and take `tau=sigma>0`. This centered error is sub-Gaussian with proxy `sigma^2`, but the stated event has probability one, exceeding `exp(-1/2)`.

**Minimal repair:** define `d(x)=1{x<0}` and replace both sign comparisons by `d(hat D)!=d(D)`, assigning ties to “do not mix.” Then the existing one-sided proof is valid. Keep the important distinction between a joint probability bound and accuracy conditional on being in a margin bin. Also specify positive `sigma` or explicitly handle exact estimates when `sigma=0`.

### 3. Normalize only when the denominator and relevant moments exist

Section 6 defines relative errors and normalized residual second moments for arbitrary maps without explicitly excluding `||G(u)||=0` or infinite second moments. Proposition 6.1(i) also divides by `e1 e2`, and Remark 6.2 divides by the mean error.

**Minimal repair:** add a standing assumption `||G(u)||>0` almost surely and finite second moments for the normalized residuals under discussion. Require `0<e1<=e2` for the correlation parameterization. If `e1=0`, the better member is already exact almost surely and cannot be strictly improved. State the coefficient-of-variation identity for positive mean error and handle the identically zero case separately. The numerical applications need no changed values.

### 4. Corollary 6.4 bounds an optimum, not every choice of weights

The display is correct, but “whatever convex ensemble one builds sits within that window” is false. Its upper bound is obtained by evaluating uniform weights; arbitrary poor weights can have much greater error. For example, with `S=diag(1,4)`, `bar e=1`, `bar rho=0`, `delta_e=3`, and `delta_rho=0`, the displayed upper bound is `2.5`, while putting all weight on the second member gives risk `4`.

**Minimal repair:** say that the optimal convex risk lies in the bracket, and that uniform weights attain a risk no larger than its upper endpoint. Any claim about a particular fitted stack needs its actual objective value or an applicable estimation/optimization bound. Add `bar e>0` where dividing by it.

### 5. The anisotropy calculation should cease being presented as a probabilistic rate result

The “informal” label and disclosures are welcome, but the displayed Proposition S2.5 still asserts high-probability rates for validation-selected ridge regression. Its derivation does not establish them:

- The setup assumes unit `H^s` norm, while Step 1 uses the Matérn native space `H^(s+d'/2)` and a pointwise `h^s` bound. Those are different regularity assumptions. Step 3 then again identifies `H^s` with that native norm.
- The sum of lower derivative orders cannot in general be bounded by `sigma_1^s` when `sigma_1<1`; an inhomogeneous bound includes `max(1,sigma_1)^s` and other transformation/domain factors. Absorbing arbitrary singular-value constants does not support identifying a single universal constant benefit `sigma_1^s`.
- A fixed nonsingular `A` cannot retain a fixed effective rank `r<d` as `n` tends to infinity under `sigma_(r+1) <= n^(-1/r)`. An `n`-dependent matrix or a finite-resolution crossover requires uniform constants, which are not supplied.
- Small nonzero trailing singular values do not by themselves license application of an `r`-dimensional zeros lemma with uniform domain constants.
- Step 5 says the validation selection term is lower order than the displayed rates. No validation-size scaling or exponent condition establishes that. An `m_v^(-1/2)` selection term can dominate `n^(-s/r)`, and a finite positive nugget grid need not approach interpolation as `n` grows.
- Comparing two upper bounds does not prove an actual estimator error ratio or a necessary rate advantage; the text sometimes draws that stronger inference despite acknowledging absent lower bounds.

**Minimal repair:** replace the proposition and its alleged derivation by an explicitly conditional fill-distance illustration: if separate uniform interpolation estimates `C_d h_d^nu` and `C_r h_r^nu` are established, substituting random-design fill distances suggests the two exponents. Do not claim those estimates for the validated campaign or identify the actual improvement with the ratio of upper bounds. Retain the measured ARD result and its lack of clear rate improvement; it stands independently.

## Required scope and presentation repairs

1. **Native-space membership is not an absolute approximation barrier.** The sentence “a stationary kernel of fixed smoothness reaches only its native space” is incorrect as a statement about approximation: kernel approximants can converge to functions outside their RKHS. Say instead that the particular native-space bound requires membership and its norm may be unfavorable. A pullback identity permits a more favorable representation but does not predict that every learned feature map decreases the norm or design factor. Change “predicts” to “is consistent with” for the measured direction, and avoid saying no rescaling can reproduce any learned feature benefit.

2. **Write the cross-fitted/full-data mismatch explicitly.** The caveat in Section 6.4 correctly identifies an unresolved difference, but the theorem is still repeatedly described as directly bounding the shipped correction. With `Delta_X=m_full(X)-m_cf(X)` and `r_full=G-m_full`, the shipped error satisfies

   `||G(u)-m_full(u)-hat r_cf(u)|| <= ||r_full||_K tilde P_lambda(u) + ||c_lambda(u)^T Delta_X||`.

   This uses the full deployed mean and does not require extending a fold-specific table into an unspecified global `m_cf`. Do not call the fitted cross-fitted norm a lower bound on the full-data residual norm without an additional argument. Remove the unnecessary independence requirement on `m` from this algebraic theorem's setup; the pathwise proof does not use it.

3. **Carry conformal qualifications into the main text.** The lower coverage bound needs exchangeability; the upper bound needs almost surely distinct scores. The exact Beta/Beta-binomial law needs i.i.d. continuous scores and an order index `k<=m`. I.i.d. continuous inputs alone do not imply continuous scores: a constant error score has ties. Define the conformal quantile as `+infinity` when `k=m+1`, and restrict the Beta formula to its nondegenerate case. These changes leave the reported 90% and 95% calculations intact under their continuity assumption.

4. **Remove contradictory capacity language.** Statements such as “fitting the weights is safe,” “selection cannot meaningfully overfit,” “statistically almost free,” and “the refit winner is covered” are stronger than the qualified, often vacuous bounds actually supplied. State the observed seed stability as empirical evidence and describe the bounds as conditional scaling calculations consistently.

5. **Make proof locations explicit.** Add a proof-location sentence after every displayed main-text proposition, corollary, and theorem whose proof is only in the supplement. The GitHub footnote should specifically point readers to the supplement and its proofs. The supplement should likewise point to its proof section rather than relying on a global statement elsewhere. Label the anisotropy material as an illustration without a claimed proof.

## Overall assessment

The central empirical story is potentially publishable: replication, controlled use of raw versus learned inputs, the distinction between two evaluation metrics, and pool-specific ensemble saturation are worthwhile. The ordinary RKHS and ensemble identities appropriately support that study once their scope is enforced. I recommend a focused revision that removes false boundary cases and overstatements, makes the residual mismatch formula explicit, and replaces the unsupported anisotropy rate proposition by a conditional illustration. This is a NO to the unchanged version, not a conclusion that the project is unsalvageable.
