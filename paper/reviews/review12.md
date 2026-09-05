# Independent review 12

**Vote on publication readiness of the submitted version: NO.**

The paper has a credible empirical contribution and is repairable without replacing its experiments. My negative vote concerns two precise mathematical/expository defects and inconsistent descriptions of what the capacity results establish. It does not mean the central benchmark results are false. I read the main paper and inspected the supplement's finite-sample stage bounds, margin and coordinate-selection statements, and their proofs. I did not rerun the expensive training campaigns or consult the other reviews.

## 1. Symmetry theorem is correct; the augmentation consequence is false

In `supp_theory_stages.tex`, immediately after Proposition S1.1 (`prop:sym`), the text says that training on reflection-augmented data is ordinary empirical risk minimization for the symmetrized class. The displayed theorem proves a Jensen inequality, not equality of these two training objectives.

A counterexample with the paper's relative loss is the domain {-1,1}, reflection Sx=-x, identity output action T, target G(x)=1, and predictor f(x)=1+x. Its symmetrization equals G and has zero empirical error. Its reflection-augmented empirical error equals 1. Thus these are different objectives even under exact symmetry.

**Smallest repair:** retain the proposition and its proof. Replace the sentence by: “Reflection-augmented empirical loss upper-bounds the empirical loss of the corresponding symmetrized predictor by convexity. This motivates augmentation, but the proposition alone does not guarantee that training with augmentation improves the fitted predictor.” The test-time population-risk guarantee remains valid under the stated exact equivariance and distribution-invariance hypotheses. The near-mirror data checks are evidence for those hypotheses, not proof of exact equivariance of the distributed solver output.

## 2. Margin proposition has a false tie case

Proposition S2.1(ii) (`prop:margin`) claims its sign-disagreement bound “with no condition on D,” although the preceding setup assumes D≠0. The proof handles D=0 by treating the rule as a binary decision, whereas the statement uses the three-valued mathematical sign function.

For a concrete violation, let D=0 and let D-hat equal +sigma and -sigma with probabilities 1/2 each. The estimation error is centered sub-Gaussian with variance proxy sigma squared. At tau=sigma, sign(D-hat) differs from sign(D)=0 and |D-hat|≥tau with probability 1, exceeding exp(-1/2).

**Preferred repair:** define the actual binary decision explicitly as d(x)=1{x<0}, with ties assigned to “do not mix.” Replace every event involving unequal signs by d(D-hat)≠d(D). For part (i), retain D≠0; part (ii) then correctly includes D=0 by the proof already supplied. Alternatively restrict both parts to D≠0, or introduce a factor two when a genuine sign disagreement at D=0 is intended.

The rest of the one-sided Chernoff proof is correct. The main paper correctly distinguishes a joint error-and-large-margin probability from the conditional error rate within a margin bin.

## 3. Remove guarantees contradicted by the stated assumptions

The detailed bounds are substantially sound in their conditional setting. I checked the simplex covering argument, finite-candidate Hoeffding argument, affine-class Rademacher calculation, and the union over correction configurations. The uniform kernel-weight bound is also justified by the Gram Schur complement and mu/(mu+n lambda)^2 ≤ 1/(4n lambda).

However, their presentation still says:

- `method.tex`: validation stacking “costs nothing” and is “statistically almost free.”
- `theory.tex`: Proposition S1.3 “says fitting the weights is safe.”
- `supp_theory_stages.tex`: the selection “cannot meaningfully overfit,” its excess is capped by seven percentage points after substituting an observed maximum for a population bound, and the later statements “cover” or “close the chain” for the deployed pipeline.

These phrases contradict the paper's own accurate caveat: checkpoints and members reuse validation data; the deployed objective differs from coordinatewise squared error; hyperparameters are tuned at one training size and refit at another; and the observed maxima are not certified population bounds. In fact a bound of seven percentage points cannot demonstrate negligible overfitting for a method whose total error is around five percent.

**Smallest repair:** use “has a finite-dimensional capacity calculation under the stated independence and boundedness assumptions” in place of “safe/almost free/cannot meaningfully overfit.” Call the seven-point substitution an illustrative value conditional on B=0.25 being a genuine uniform population bound. Retitle the correction corollary “capacity bound for a fixed composite correction class,” not “for the deployed corrected surrogate.” State that the optional stack and correction gains are empirical paired findings, not consequences of these bounds.

## 4. Margin scale needs consistent wording in the supplement

The main paper appropriately says the empirical standard deviation 0.099 over 840 dependent pairs does not certify a sub-Gaussian proxy. The supplement nevertheless says sigma “is estimable” from such differences and that those differences “have exactly this scale.” A pooled standard deviation is neither a certified conditional sub-Gaussian variance proxy nor evidence of conditional unbiasedness. Also the OCO-2 validation carve and test set have the documented distribution shift.

**Repair:** say the pooled differences describe the observed transfer scale, while the proposition is conditional on a valid sub-Gaussian proxy and does not establish that proxy for this experiment. Keep the illustrative 4% calculation clearly conditional as the main text already does.

## 5. Coordinate selection: retain correct scope, trim a misleading application

The coordinatewise decomposition and finite-candidate selection inequality are correct for separable squared risks with a common sample-weight function. They do not make the same selector optimal for the two reported means of relative norms, because those metrics involve different denominators and an outer norm. The main paper acknowledges this correctly. The supplement's statements that the practical combination “serves all” metrics and that the deployed estimator is “exactly this estimator” should immediately repeat the fixed-members, independent-validation, squared-objective qualifications, rather than rely on a distant caution.

## Overall assessment and revision vote

The strongest publishable paper here is an empirical investigation of residual corrections versus feature kernels, together with precisely conditional supporting algebra and finite-sample calculations. The manuscript already documents its major empirical limitations unusually explicitly: test-block reuse across seeds, benchmark-only comparison with PARA-Net, OCO-2 rescoring scope, and lack of a refined-mesh experiment establishing a data floor. The repairs above preserve its numbers and main contributions.

**Prospective vote after these repairs: YES for submission as an empirical/methodological paper**, subject to independent verification of the released experiment records and the other mathematical sections. This is not a guarantee of acceptance by a specific journal. Every main-text theorem/proposition/corollary whose proof is external should carry a local “Proof: Supplement S8, …” pointer, and the first GitHub footnote should explicitly direct readers to `paper/supplement.pdf`, as requested.
