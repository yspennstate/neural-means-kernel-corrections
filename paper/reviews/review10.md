# Independent review 10: empirical utility and novelty

**Vote on the submitted version: NO.**  
**Vote on whether the existing results can support a publishable revision without inventing new results: YES.**

The negative vote is for submission in its present wording. It is not a finding that the numerical work lacks publishable value. I would support submission as an empirical operator-learning/application study after the concrete corrections below. I am not assuming JMLR or requiring a new foundational theorem as a condition for publication.

## Material inspected and limits

I independently read the submitted 39-page PDF's extracted text, the corresponding main-paper sources, the relevant supplementary proofs of the pullback identity and optimal-recovery results, and the conformal statement and proof. I did not consult any other review, edit the manuscript, or rerun the training campaign. Numerical conclusions below refer to the reported evidence, not an independent reconstruction of all experiments.

## What is publishable here

1. **A useful controlled empirical comparison.** On OCO-2, the raw-input and learned-feature kernel comparison, including the ridge-readout control, is more informative than merely comparing a neural model to a GP. Keeping the kernel machinery fixed while changing the representation, and checking the head's gain over the network over ten seeds and three bands, isolates a worthwhile practical phenomenon within the tested configuration.
2. **A reproducibility contribution.** The structural-mechanics study separates seed variability, test-block uncertainty, member diversity, stacking and the residual correction. A 19,000-row exact solve and a sixty-predictor campaign provide a substantive empirical record. The low-data reported improvement is materially larger than the very small high-data difference from PARA-Net.
3. **The ensemble analysis has diagnostic utility.** Proposition 6.1 is classical quadratic risk algebra; Theorem 6.6 is an elementary block-structured refinement. Neither needs to be advertised as a major new mathematical discovery. The measured near-saturation of a specified predictor pool and the distinction between adding seeds and adding architectures make these calculations useful in this paper.
4. **The negative uncertainty result is informative.** Reporting that the kernel power function ranks error poorly, and that a constant-width conformal set is narrower, is a strength. It prevents the paper from claiming that kernel-shaped uncertainty is automatically better calibrated or more efficient.
5. **Prior work is mostly positioned honestly.** Neural means, deep kernels, stacking and conformal calibration are acknowledged as established. Wilson et al.'s primary deep-kernel paper explicitly concerns learned input transformations within kernels; it supports the paper's acknowledgement rather than a new-method claim: https://proceedings.mlr.press/v51/wilson16.html .

These contributions can warrant an empirical research submission. Their value does not depend on turning the elementary results into stronger-looking theorem labels.

## Required repairs

### 1. Remove a false general explanation of fixed-kernel limitations

Section 6.3 says that a stationary fixed-smoothness kernel “reaches only its native space” and that rescaling cannot imitate the representation improvement. In the surrounding explanation this confuses membership of a finite fitted predictor with the approximation closure of the method. A fixed kernel can have a dense RKHS on a compact domain and approximate continuous targets that do not themselves have finite native norm. The classical universality distinction is documented in the primary paper by Micchelli, Xu and Zhang: https://jmlr.org/papers/v7/micchelli06a.html .

**Minimal replacement:** “A fixed input representation may make the target expensive to approximate at the available sample size. Learned features change the induced kernel and can give a target a more favorable native-space representation; our finite-design measurements support this interpretation for the tested OCO-2 models.”

The pullback identity gives a conditional factorization statement. It does not prove that a trained feature map decreases the unknown target norm, or that every linear metric fails. The measured ratio of fitted norms compares two lower bounds, as the manuscript itself correctly notes elsewhere. Keep that limitation consistent in the introduction and discussion: prefer “consistent with” to “predicted by” when connecting the identity to the measured raw/feature norm ratio.

### 2. Confine the structural floor to the tested pool and combination class

Theorem 6.6 is mathematically meaningful as an entrywise lower bound for convex mixtures of a specified pool. It supplies no lower bound for arbitrary future architectures, different representation classes, a per-pixel affine stack, or a new residual correction. The paper generally acknowledges this well. Section 7 then calls the remaining error the “last few learnable hundredths” and says it is “not a step on the way to zero”. Those phrases again imply a problem-wide ceiling.

**Minimal repair:** describe “the small further gains observed within the trained pool and tested pipelines”. Keep the finite-element explanation as an untested hypothesis, and retain the proposed mesh-refinement experiment. No new solver experiment is necessary to publish a carefully scoped empirical saturation result.

### 3. Do not transfer the fixed-mean theorem to the deployed correction without its extra term

Section 6.4 acknowledges that training the correction on cross-fitted residuals and deploying it on a full-data mean introduces a discrepancy term not measured here. This is the correct caveat. Section 4.7 and the discussion nevertheless introduce the displayed power-function bound as though it directly bounds the reported corrected predictor.

**Minimal repair:** refer to the fixed-mean reference construction, or explicitly carry the discrepancy term every time the deployed construction is discussed. A theorem on one estimator cannot be used as the mathematical guarantee for another after the mismatch is acknowledged only later. The conformal result can still cover the fixed, fully selected deployed predictor under its own independent-calibration conditions.

### 4. Preserve the OCO-2 metric qualification in the overview table

The main OCO-2 discussion carefully says that two criteria are defined on the released reduced representation and do not establish superiority for end-to-end retrieval. Table 1 instead describes them as both of the source paper's metrics. That wording should be reconciled. Use “the two rescoring criteria defined in Section 5” and retain the same-test-points statement. Avoid claiming an end-to-end better emulator from retained-component rescoring.

### 5. Give each main-text mathematical result a local proof pointer

The global sentence referring readers to Supplement S8 is helpful but does not satisfy the user's requested per-result navigation. Add a short sentence immediately after every proposition, theorem and corollary whose proof is in the supplement, specifying its supplementary subsection when possible. Put the supplement in the opening GitHub footnote too, with the concrete path to `paper/supplement.pdf` and, preferably, the supplement source. Definitions or purely explanatory remarks need no invented proof.

### 6. Small local accuracy corrections

- In the sharpness proof for Theorem 6.9, after setting `r_1 = (||G-m||_K/rho(u)) g_u`, the statement `r_1(X)=g_u(X)` drops the scale factor. Restore it; the argument and theorem remain valid.
- Section 6.1 says “Proposition [stack] says fitting the weights is safe” immediately after stating that the deployed pipeline violates the independent-candidate hypothesis. Replace that with the precise conditional-capacity reading.
- In the main conformal overview, append the distinct-score qualification to the upper coverage bound and the i.i.d./continuous-score qualification to the exact Beta law. The supplement has these qualifications, and they should not vanish when the results are summarized.

## Novelty recommendation

The paper should lead with two replicated case studies and the operational decision being measured: residual correction when a kernel is competitive, learned-feature heads when raw-input kernel performance is poor. Present the mathematical section as established identities and scoped refinements supporting the diagnostics. The minimax interpolation result is classical optimal recovery, and the nugget identity is a useful transparent calculation; neither should be the principal novelty claim. Keep the failed anisotropic-rate prediction explicitly heuristic, ideally without promoting it as a central result.

A theorem-light, evidence-forward positioning would strengthen the submission. Removing supported experiments merely to shorten it is unnecessary, but several repeated explanations of what the theorems do not prove can be replaced by one precise statement and disciplined references.

## Final judgment

The supplied manuscript contains enough useful empirical work to merit publication consideration after revision. My present **NO** is repairable by tightening scientific claims, fixing the small local proof/qualification errors, and making the supplement navigable. I found no reason in this assigned scope to discard the core experiments or demand a wholly new algorithm. This review does not certify that every reported number or every supplementary theorem has been independently verified.
