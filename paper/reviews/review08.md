# Independent publication review 08

**Vote on the uploaded version: NO.**

The mathematical problems I found are repairable and do not refute the principal empirical comparisons. The paper could become a publishable empirical study after a careful revision that narrows the interpretations of its auxiliary theory. The current version still contains false endpoint statements and several claims stronger than their displayed results. I would not publish it unchanged. This vote is about readiness of this manuscript, not a prediction of acceptance at a particular journal.

I read the complete extracted original manuscript and examined `theory.tex`, `supp_theory_rules.tex`, `supp_theory_stages.tex`, `supp_theory_conf.tex`, `supp_theory_deff.tex`, and the corresponding arguments in `proofs.tex`. I did not read other reviewers' reports or modify the manuscript. My assignment was assumptions, zero norms and endpoints, margin rules, native-space interpretation, and consistency between the main paper and supplement; this report does not certify the numerical artifacts or bibliography.

## 1. A false endpoint in Proposition 6.7

The proposition assumes an interior optimal mixture, expressed as rho < min(t,1/t), and concludes that both displayed partial derivatives are strictly positive. The allowed point rho=-1 contradicts the claim about the derivative with respect to t. Take e1=1 and e2=2, with scalar normalized residuals identically 1 and -2. The correlation is -1, the optimal mixture is interior, and its risk is zero. More generally, for every t>0, perfectly anticorrelated residuals have an interior mixture of zero error. Consequently partial V/partial t=0, not >0.

**Minimal repair:** state 0<e1<=e2 and -1<rho<e1/e2 for the strict derivative conclusion. Treat rho=-1 separately as exact cancellation. The derivative formulas themselves are correct.

The paragraph after the proposition also says that near the admission threshold, t-rho is small and the exchange rate is large. The admission threshold is rho=1/t, not rho=t. For t>1, the purported small denominator tends to t-1/t. The fractional-error exchange coefficient has the finite limit

\[
\lim_{\rho\uparrow1/t}\frac{1-\rho^2}{t-\rho}=\frac1t.
\]

Thus the claimed threshold amplification does not follow and is generally false. Delete that sentence or replace it with the exact finite-limit interpretation. This also avoids using the proposition to attribute changes in learned members more specifically than the derivative calculation permits.

## 2. Proposition S2.1 mixes a binary rule with the three-valued sign function

Part (ii) advertises a result with no condition on D and writes the event sign(Dhat) != sign(D). Its proof instead uses the binary decision to mix exactly when D<0, with equality treated as no mixing. Those events differ at D=0.

For a direct counterexample to the displayed sign statement, set D=0 and let Dhat equal +sigma or -sigma with probability one half each. This is a centered sub-Gaussian error with variance proxy sigma^2, because cosh(lambda sigma)<=exp(lambda^2 sigma^2/2). At tau=sigma, the displayed event has probability 1, whereas the proposed upper bound is exp(-1/2). The opening assumption D!=0 and the later phrase "with no condition on D" also conflict.

**Minimal repair:** define a(d)=1{d<0}; replace sign disagreements by a(Dhat)!=a(D). Allow arbitrary real D in the common hypotheses, restrict part (i) to D!=0, and let part (ii) include the boundary. The supplied one-sided argument then works, with consistent strict/non-strict inequalities and an explicit tie convention. State sigma>0, or separately define the trivial noiseless case sigma=0.

The main text correctly distinguishes joint errors from conditional accuracy inside a margin bin. Preserve that correction. However, the supplement says sigma is estimable because cross-block gap differences have "exactly this scale." A pooled empirical standard deviation is not a conditional sub-Gaussian proxy; heterogeneous pair difficulty, bias, test-block estimation noise and validation-to-test distribution shift all matter. Replace this with an empirical scale diagnostic. The main paper already makes much of this qualification, and the supplement should agree.

## 3. Corollary 6.4's prose is stronger than its correct theorem

The display bounds the **optimal** convex risk, using uniform weights to construct an upper bound. Its following paragraph says that "whatever convex ensemble one builds" sits in that window. This is false even in the simplest equicorrelated example. With S=I, ebar=1, rhobar=delta_e=delta_rho=0 and M>=2, the bracket for the minimum is [0,1/M], but a vertex of the simplex has risk 1. Such an S is realized by independent unit-variance scalar residuals and a constant nonzero target.

**Minimal repair:** say that the optimal mixture lies in the window and uniform weights attain its displayed upper bound. A fitted mixture can also be placed there if its achieved objective is explicitly checked or its optimization/statistical excess is added. The theorem and its proof otherwise survive.

The interpretation of Theorem 6.6 should likewise remain conditional on the stated entrywise bounds for the entire pool. Bounds measured on sixty members do not establish what every future architecture must do. The original abstract mostly respects this, by saying "for the pool trained"; retain that qualification throughout.

## 4. Zero target norms, zero member errors, and division by zero

The theory starts with arbitrary measurable G but repeatedly divides by ||G(u)||. State ||G(u)||>0 almost surely wherever relative risk is used. For S to be a finite real matrix, also require E||rho_fm||^2<infinity for the members in the second-moment subsection. These are not consequences of a deterministic solver.

Proposition 6.1(i) defines a correlation by division by e1 e2 without requiring these quantities positive. If e1=0, the best member is already exact almost surely; no strictly improving mixture exists and correlation is undefined. The natural repair is 0<e1<=e2 in the two-member formula, followed by a one-sentence exact-member case. Corollary 6.4 divides by ebar^2 and therefore requires ebar>0, although the non-normalized floor theorem itself permits ebar=0. Remark 6.2 defines a coefficient of variation and E1/E2; its displayed ratio is undefined for an exact predictor. State the ratio for E1>0 and handle the exact case separately.

These are small assumptions, but stating them centrally is preferable to having formally undefined special cases across several propositions.

## 5. Main Section 6.5 omits assumptions retained in the supplement

The main text says exchangeability alone supplies both lower and upper conformal coverage bounds and then gives an exact Beta law. The supplement correctly limits the upper bound to almost surely distinct scores and the Beta/Beta-binomial statement to conditionally i.i.d. scores with a continuous distribution. Those qualifications are mathematically necessary.

For example, if every score is zero, the conformal set covers with probability 1. For alpha=0.1 and m=1000 this exceeds 1-alpha+1/(m+1). Exchangeability still holds. The latent coverage is a point mass at 1, not a Beta distribution. Neither a random split nor continuous input variables alone prove that a score transformation has no atoms.

**Minimal repair:** retain the lower coverage bound under exchangeability; state the distinct-score condition next to the upper bound in the main text; and condition on the fitted predictor when presenting the i.i.d.-continuous Beta law. Describe use of that law on the campaign as relying on those score-distribution conditions. Randomly splitting an i.i.d. block preserves the needed i.i.d. property before conditioning on the realized complete block; a realized fixed benchmark itself does not magically establish a population sampling model.

There is also an unhandled endpoint: k=ceil((m+1)(1-alpha)) can be m+1, but the quantile is defined as the kth smallest of only m scores. For alpha<1/(m+1), it is undefined. Append an additional +infinity score and set q=+infinity when k=m+1, or restrict the finite-quantile statement to k<=m. The Beta law must exclude that endpoint, where the coverage is identically 1.

The conformal object is an output-space Euclidean ball, not a collection of separately calibrated coordinate intervals. The supplement already explains this accurately and should guide the main paper's terminology.

## 6. RKHS pullback is correct; claimed automatic improvement is not

Proposition 6.8 and the quotient-space proof are sound: the minimum norm exists because the functions vanishing on the feature image form a closed subspace, and vector outputs are handled by a direct sum of scalar RKHSs. But the identity contains no comparison forcing the feature norm or power function to decrease relative to a raw-input kernel. A feature map can preserve, improve, destroy or collapse relevant distinctions. For phi(x)=Ax the feature kernel is exactly the corresponding anisotropic input-metric kernel. Hence the blanket statement that no rescaling can imitate the effect is not a consequence of the proposition.

The statement that a stationary kernel "reaches only its native space" also confuses membership of a finite fitted function with the set of targets it can approximate. On a compact interval a cusp such as |x| is outside sufficiently smooth Sobolev native spaces, yet it can be uniformly approximated by smooth functions in those spaces; lack of membership does not imply an approximation barrier. Native-space membership is a sufficient hypothesis for the paper's particular quantitative bound.

**Minimal repair:** say that the tested learned features produced a lower fitted RKHS norm and better errors than the tested raw-input metrics under the specified tuning routine. The pullback identity explains which function-space norm governs a conditional bound, and permits a gain when a suitable factorization is learned; it does not predict a gain for arbitrary features or prove that the untested space of anisotropic kernels cannot match it.

At a positive nugget, tr(alpha^T K alpha) is the norm of a ridge-fitted function, not exactly an interpolant. It is indeed no greater than the target norm when the same exact labels come from a single target in that RKHS: projection followed by the spectral contractions mu/(mu+n lambda) proves this. Call the quantities fitted-function norms, and reserve "interpolant" for lambda=0 or explicitly say approximately interpolating.

## 7. Make the cross-fitting perturbation explicit

Theorem 6.9 is a correct pathwise deterministic inequality for a single residual function r=G-m with exact labels r(X). It does not need independence of m and X; remove the unnecessary independence clause that conflicts with the later correct pathwise explanation. The theorem remains conditional on native-space membership for the realized residual.

The paper acknowledges that the structural correction labels use m_cf(X) while deployment adds the correction to m_full. This is not merely a notation choice. Put

\[
r_{\rm full}=G-m_{\rm full},\quad E=m_{\rm full}(X)-m_{\rm cf}(X),\quad
c(u)=(K+n\lambda I)^{-1}k(X,u).
\]

Since R_cf=r_full(X)+E, the correct deterministic bound is

\[
\|G(u)-m_{\rm full}(u)-c(u)^T R_{\rm cf}\|_2
\le \|r_{\rm full}\|_K\widetilde P_\lambda(u)+\|c(u)^T E\|_2.
\]

This formula makes the existing caveat concrete without pretending its additional term was measured. A rowwise out-of-fold prediction array alone does not define a single global m_cf with a known native norm. Fitted correction norms are not necessarily lower bounds on the norm of G-m_full. As a counterexample, set m_full=G, but m_cf=G-1 on one training row, K=[1]. The ridge correction is nonzero although the full residual function has norm zero.

Two further small fixes: Theorem 6.9 calls Ptilde itself the exact worst-case error on a ball of radius ||G-m||; the error is **radius times Ptilde**, or Ptilde on the unit ball. Its sharpness proof writes r1(X)=g_u(X) after defining r1 as a scaled g_u; restore that scaling factor. The minimax result and its nugget-cost identity otherwise check out, including the P0=0 branch.

## 8. The informal rate proposition should not carry theorem-like claims

Proposition S2.5 is carefully labeled informal, but its derivation still changes the smoothness hypothesis from g in H^s to a Matérn native space H^{s+d/2}, treats nonsingular A as exactly lower rank at the working resolution, and asserts that validation selection costs are lower order than the approximation rates. The last assertion fails, for instance, when a validation error of order m^(-1/2) is compared with n^(-s/r) for m proportional to n and s/r>1/2. Letting lambda tend to zero at fixed design also does not justify an n-dependent rate for a fixed finite nugget grid as the smallest Gram eigenvalue vanishes.

**Minimal repair:** convert this to an explicitly unnumbered heuristic, distinguishing approximation order from Sobolev regularity; restrict its clean rate sketch to exact-rank feature maps, appropriate native-space regularity, interpolation and regular domains. Present the real-data scaling sweep as an empirical result and avoid a theorem about the validation-selected estimator. The main empirical result needs none of these unsupported rate claims.

## 9. Smaller scope inconsistencies to remove

The supplement describes reflection-augmented training as ordinary ERM in the symmetrized predictor class. Convexity supplies an inequality, not that identification. With a singleton predictor f(u)=u, symmetric inputs +/-1, even target G=1 and squared loss, the symmetrized predictor is zero and has risk 1, while augmentation leaves the risk of f at 2. Say augmentation encourages symmetry and that the theorem guarantees the averaged predictor's risk inequality.

Likewise, the tiny difference between globally signed and globally convex optimum weights does not establish that all gains of a per-pixel affine stack come from spatial variation rather than sign. Pixel-dependent signs and an intercept interact with spatial variation; that attribution would need an ablation against per-pixel convex weights with matched intercept handling. Report what was measured, without the exclusive causal attribution.

Several supplement paragraphs still say validation fitting is "safe," cannot meaningfully overfit, or is statistically almost free, although the main text correctly admits candidate dependence and numerically vacuous population bounds. Remove those operational claims consistently. The capacity inequalities are useful as conditional scaling calculations, with population bounds and independent fixed candidates; empirical maxima do not establish their hypotheses.

## Publication path

Retain the two controlled empirical studies, the explicit squared-versus-mean metric distinction, the finite trained-pool residual analysis, and the comparison of conformal scores including the better constant-width ball. Those constitute a coherent, potentially publishable study. Correct the exact algebraic endpoints, propagate the actual coverage assumptions to the main paper, give the cross-fit perturbation term, and recast the RKHS and rate discussion as conditional interpretation rather than a proved representational separation. These repairs can be made in prose and proofs without fabricating new experiments. The empirical findings still require the separate data/code and benchmark audits assigned to other reviewers.
