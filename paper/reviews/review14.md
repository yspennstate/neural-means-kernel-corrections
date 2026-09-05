# Independent review 14

**Vote on publishing the submitted version: NO.**

The paper has a credible empirical contribution and is potentially publishable after focused corrections. I would not release this version as a mathematically checked final manuscript because several formal statements have endpoint or scope errors, and the main text occasionally turns a valid result into a stronger claim its proof does not establish. None of the issues below shows that the principal benchmark scores are false. I did not rerun the training campaigns or independently reconstruct their numerical tables.

I reviewed the complete original main-paper text, the supplement's formal statements, and the proofs in `source_repo/paper`. This review is independent of the other reviewers and concentrates on proof completeness, mapping statements to proofs, and proposition/margin/coverage endpoints.

## What survives

The central second-moment identity is correct, as are the entrywise convex floor, the block floor distinguishing seeds from architectures, the signed-weight minimization under positive definiteness, the pullback RKHS identity, and the power-function/minimax calculations subject to their stated RKHS assumptions. The supplement contains substantive proofs rather than just numerical verifications. The proof of the kernel-weight bound is particularly clean and its two-point cosine-kernel example establishes the claimed universal constant for the stated range.

The paper also now distinguishes mean relative error from RMS relative error, unknown native norms from computable fitted norms, empirical benchmark ordering from paired statistical comparison, and conditional capacity calculations from deployed-estimator guarantees. Those distinctions are necessary and should be retained during revision.

## Corrections required before publication

1. **Proposition S2.1(ii), margin rule: the formal sign event is wrong at zero.** The display claims a bound for `sign(Dhat) != sign(D)` with “no condition on D,” while its proof actually treats the binary decision “mix iff Dhat < 0.” Take D=0 and Dhat uniform on {-1,+1}. The estimation error is centered sub-Gaussian with proxy 1. At tau=1 the displayed sign-error probability is 1, exceeding exp(-1/2). The earlier introductory assumption D != 0 conflicts with the later claim rather than repairing its advertised endpoint. Minimal repair: define b(d)=1{d<0} and replace every error event by b(Dhat) != b(D), including in prose and numerical explanations. Then the proof's D>=0 branch is correct and the factor-one bound survives at D=0. Alternatively restrict both parts explicitly to D!=0. Part (i) should be visibly conditional on D if D is random; an unconditional probability cannot be bounded by an unspecified random right-hand side.

2. **Proposition 6.7: the strict derivative assertion excludes an endpoint that is currently allowed.** For rho=-1 and positive e1,e2, the optimum is interior and the optimal risk is identically zero as t varies. Thus dV/dt=0, contradicting the displayed strict inequality. Add 0<e1<=e2 and -1<rho<min(t,1/t). The displayed derivatives and first-order exchange formula are then correct. Proposition 6.1(i) also needs 0<e1<=e2 before defining rho=S12/(e1e2); if e1=0, the first member is exact and no mixture strictly improves it. Handle that trivial case separately. The interpretation “near the admission threshold, where t-rho is small” should be qualified: with t>=1 the admission threshold is rho=1/t, and t-rho is small there only for nearly equal member errors.

3. **Conformal endpoints and hypotheses need to appear wherever the result is asserted.** In S3 the kth order statistic of m scores is undefined when alpha<1/(m+1), because k=m+1. Append +infinity to the score list or define q=+infinity in that case. For the Beta/Beta-binomial law, restrict k<=m, and state the degenerate full-coverage case separately. The upper coverage bound requires almost surely distinct scores; the main text's Section 6.5 repeats the two-sided bound without that qualification. For example, all scores equal yields coverage 1, which can violate the quoted upper bound. The exact Beta law requires i.i.d. continuous scores conditional on the trained predictor/design, not exchangeability alone. The supplement recognizes this distinction; the main text should state it too. A fixed or random split preserves an existing i.i.d. model, but splitting by itself does not establish it.

4. **Corollary 6.4 bounds the optimum, not every fitted or arbitrary convex ensemble.** Its formula is correct. Its explanatory sentence that “whatever convex ensemble one builds sits within that window” is false. With S=I, M=2, ebar=1, rhobar=0, and both spreads zero, the display gives the correct optimum 1/2, whereas a vertex has risk 1. Replace this wording by “the optimal convex ensemble lies in this window.” If discussing the fitted stack, its proximity to the optimum must remain a measured result or require an optimization/generalization gap.

5. **Theorem 6.9's sharpness prose loses the radius.** The exact worst-case error over a radius rho ball is rho times Ptilde_lambda, not Ptilde_lambda itself. The theorem's first inequality and the subsequent minimax theorem correctly include the norm factor. Change the sentence after the definition of Ptilde to “the exact worst-case error over the radius rho ball is rho Ptilde_lambda,” or refer to the unit ball. In S8.19 the sharpness construction scales g_u but then writes r1(X)=g_u(X); the sample values must carry the same scaling factor. The correct inner-product argument already proves the result, so this is a local repair.

6. **State the deployed cross-fit mismatch bound explicitly.** Section 6.4 acknowledges that correction residuals use out-of-fold predictions while deployment uses full-data predictions. A collection of out-of-fold predictions on training rows is not automatically one global RKHS residual map. The clean statement uses r_full=G-m_full and Delta_X=m_cf(X)-m_full(X):

   ||G(u)-m_full(u)-rhat_cf(u)|| <= ||r_full||_K Ptilde_lambda(u) + ||k(u,X)(K+n lambda I)^(-1) Delta_X||.

   This follows by adding and subtracting the correction built from r_full(X). It avoids inventing a global m_cf. Its second term is not measured in this version, so the paper must continue to withhold a numerical certificate for the shipped estimator. Independence of m from the correction sample is unnecessary for this pathwise algebra and should be removed from the theorem setup if the paragraph continues to say that same-data training does not invalidate the result. Fitted norms based on mismatched residual rows must not be promoted to lower bounds on the full-data residual operator's norm.

7. **Demote or repair the informal anisotropic-rate proposition and its derivation.** Labeling S2.5 “informal” is an improvement, but the attached argument still makes unsupported mathematical transitions: H^s in the statement versus H^(s+d/2) as the Matérn native space in the proof; nonsingular fixed A versus trailing directions treated as exactly zero; a bound proportional to sigma1^s that omits the zeroth-order/max(1,sigma1^s) issue; and a validation-selection penalty asserted to be lower order than both target rates without an assumption on validation size or rate exponents. The exact-rank limit cannot be obtained uniformly from fixed nonsingular A as n grows. Minimal defensible repair: make this an unnumbered heuristic discussion, remove its high-probability KRR-rate assertion and false “lower order” sentence, and retain only the explicit experimental observation that the tested metrics yield a limited gain. A rigorous replacement would need an exact-rank model, correct native smoothness, interpolation assumptions, and explicit rate conditions, which would enlarge the paper unnecessarily.

8. **Remove residual contradictions to the paper's own capacity caveats.** S1.3's claim that selection “cannot meaningfully overfit” and is “statistically almost free” does not follow from an evaluated excess bound of approximately seven percentage points. The main paper admits these constants are vacuous and the candidates reuse validation; the supplement should use the same language. Similarly, Proposition S1.1 proves that test-time averaging reduces risk under exact symmetry, but augmented-data ERM is not generally identical to ERM over the symmetrized predictor class: convexity gives an inequality between their losses. Replace the claimed equivalence with a restrained statement that augmentation encourages the relevant symmetry.

9. **Normalization needs a standing condition.** The relative losses and normalized second moments require ||G(u)||>0 almost surely and finite relevant moments. The experiments may satisfy this, but the formal general setting should say so. The coefficient-of-variation identity in Remark 6.2 needs E1>0; the exact-zero-error case can be declared trivial.

## Proof-location audit

All principal main-paper claims have a proof or direct derivation in the supplied sources. The user requests a local pointer whenever a proof is in the supplement; the generic “proofs in S8” announcement does not fulfill that request on its own. Add a sentence immediately after each applicable statement. These are the original source's proof locations; automatic labels should be used so later revisions do not stale the numbering.

| Main result | Proof location in original supplement |
|---|---|
| Proposition 6.1, second moments | S8.10 |
| Remark 6.2, metric relation | Direct derivation in the main text |
| Corollary 6.3, convex floor | S8.11 |
| Corollary 6.4, floor bracket | S8.6 |
| Corollary 6.5, signed stacks | S8.9 |
| Theorem 6.6, seeds and architectures | S8.12 |
| Proposition 6.7, accuracy/correlation exchange | S8.13 |
| Proposition 6.8, feature pullback | S8.17 |
| Theorem 6.9, power-function bound | S8.19 |
| Theorem 6.10, minimax statement | S8.20 |

| Supplement result | Proof location in original supplement |
|---|---|
| S1.1, symmetry | S8.1 |
| S1.2, kernel-flow identity | S8.2 |
| S1.3, validation stacking | S8.3 |
| S1.4, finite selection | S8.4 |
| S1.5, affine stacking | S8.5 |
| S1.6, composite correction capacity | S8.7 |
| S1.7, kernel weights | S8.8 |
| S2.1, margin | Proof immediately after statement in S2 |
| S2.2, per-coordinate metric | S8.14 |
| S2.3, weighted per-coordinate metric | S8.15 |
| S2.4, empirical coordinate selection | S8.16 |
| S2.5, informal rates | S8.18, explicitly a heuristic derivation |
| S3.1, conformal coverage | S8.21 |
| S3.2, realized coverage law | Derivation within the remark in S3 |
| S5.1, effective dimension | S8.23 |
| S5.2, Marchenko–Pastur expression | S8.22 |

The first-page GitHub footnote currently advertises code, data pointers, and per-run summaries but does not direct the reader to the supplement. Add: “The online supplement, including the proofs, is in `paper/supplement.pdf` in the same repository.” Prefer a clickable direct supplement link as well as the repository link. The later supplementary-material section already gives the direct location, but it does not replace the requested first-page mention.

## Minimum revision and final assessment

A submission-ready revision can preserve the empirical contribution, the principal tables, and the valid algebraic theorems. It needs local theorem/endpoint repairs, explicit proof pointers, the cross-fit mismatch formula, and consistent restraint in the surrounding prose. The unsupported anisotropic asymptotic statement should be demoted rather than expanded into a new research claim. I would reconsider my vote to YES for a carefully corrected empirical-methods paper after verifying those changes; that is a judgment of readiness for submission, not a promise of journal acceptance. The original version receives NO.
