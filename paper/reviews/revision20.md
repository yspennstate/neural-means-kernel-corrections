# Reviewer 20: bounded audit of the revised manuscript

**Status at this inspection: the revised positioning can support a YES recommendation as retrospective empirical research, after the short corrections below.** This is not a vote for acceptance at JMLR or any particular venue, and it does not certify experiments that were not rerun. My original vote remains NO on the submitted version.

I read revised `main.tex`, `intro.tex`, `method.tex`, `theory.tex`, `experiments.tex`, `discussion.tex`, and `impl.tex`. I did not inspect the concurrently edited supplementary theory and OCO-2 files, did not read other reports, and made no manuscript edits. Local proof pointers are being added separately and are not counted as an outstanding deficiency here.

## Substantive repairs that succeed

The abstract and introduction now disclose pooled target centering and explicitly restrict the empirical claims to the recorded estimator and repeatedly inspected public blocks. This is a legitimate change of claim: it does not make corrected-pipeline claims from unperformed reruns. The mechanics comparison now says numerical proximity to PARA-Net instead of statistical equivalence. OCO-2 headline comparisons correctly refer to the two reduced-representation rescoring criteria and acknowledge the weak-CO2 exception. The strongest finite-pool statement is kept in RMS and restricted to the measured global convex class.

The mixed-label correction formula is correct. With Delta=m_full(X)-M_tr, the actual labels are r_full(X)+Delta, and the error equals the compatible single-mean correction error minus c_lambda(u)^T Delta. The triangle inequality therefore gives exactly the displayed added norm, followed by the valid Frobenius estimate. The statement properly avoids inventing a globally defined cross-fitted function, and it does not call the unmeasured quantity evaluated.

The new perfect-anticorrelation exclusion in Proposition 6.7 repairs the strict derivative claim. The positive e1 condition in Proposition 6.1 repairs its undefined ratios. The refined distinction between empirical second-moment bounds and centered-correlation diagnostics is valuable. The covariance and metric transfer discussion no longer misrepresents within-validation identities as prospective predictions.

The broad overclaims about deterministic solver error and RKHS approximation expressivity are removed in the introduction and discussion. The conformal summary now distinguishes exchangeability, distinct scores, continuous iid scores, the appended infinite score, and study-level adaptation. Those changes make the mathematical facts compatible with retrospective experiments.

The revised sixty-member discussion no longer misidentifies the `mlpR` block as an ordinary metric-loss MLP. I checked that `seedarch.py` and `pool_pipeline.py` use the six block labels `mlp`, `mlpMSE`, `mlpR`, `fno`, `unet`, `krr`, each across ten seeds. The rewritten discussion uses model configurations rather than claiming six distinct architectural families. Its UNet rank of fourth among six is consistent with the displayed mean errors.

## Corrections still needed in the inspected files

1. **Theorem 6.9 still omits the ball radius in its final sentence.** The display is correct, but the prose says P-tilde alone is the exact worst-case error over the ball of radius ||G-m||_K. Replace it with ||G-m||_K P-tilde, or explicitly speak of the unit ball. This is the one direct formal error still visible in the inspected revised theorem.

2. **Residual incompatible pullback interpretation survives in Section 6.3.** The text still says the norm and power-function changes are “the direction Proposition [pullback] predicts” and “This is not an accident of the O2 band; it is what a pulled-back kernel does.” Later it correctly says the identity gives no comparison with the raw-input RKHS. Delete the prediction language and introduce the identity neutrally. It identifies the feature-induced RKHS; it does not predict a favorable norm or power function change.

3. **The final uncertainty paragraph in `impl.tex` still asserts exact applicability automatically.** It says the calibration subset was never touched and “The hypotheses ... therefore hold as stated, with no approximate-exchangeability caveat.” This contradicts the newly qualified main text and the test-block-access paragraph. State that the analysis uses a held-out split for the fixed recorded predictor and compares with the reference law under the stated independence, sampling, and continuity assumptions; prior study-level inspection remains disclosed.

4. **The FNO interruption/checkpoint distinction is misstated.** The opening of `experiments.tex` says the 200-epoch schedule was interrupted between epochs 10 and 70. The implementation discussion records runs reaching epoch 150 and says the *kept checkpoints* lie at epochs 10–70. Say the schedule was interrupted before completion and the retained best-validation checkpoints were at epochs 10–70. The original phrasing was also ambiguous, but the revised one is unequivocally inaccurate.

5. **One refiner comparison remains arithmetically wrong.** `method.tex` says the refiner is within two hundredths of a percentage point of both the normalized-MSE MLP and completed FNO. The displayed means are 4.730, 4.712, and 4.682 respectively. The gaps are 0.018 and 0.048 points. State those values or restrict the two-hundredths claim to the normalized-MSE MLP.

6. **Two small zero-denominator conditions should be explicit.** Corollary 6.4 divides by bar e squared and should assume bar e>0. Remark 6.2 defines a coefficient of variation and E1/E2 and should introduce those ratios when E1>0; for zero error the variance identity remains true and no ratio is needed. These are easy local completeness fixes.

7. **Retained archive language remains internally inconsistent.** The `Released residuals` paragraph says residuals are “what we release,” then correctly says the archives are not part of this release. Rename the paragraph “Retained residual archives” and say the archives are retained and available on request; distinguish public JSON/manifest checks from reconstruction requiring the large arrays. The new availability paragraph already does this well.

8. **Final method scope still calls the symmetry “verified.”** Earlier revised text correctly distinguishes empirical support from exact invariance. Use “the tested reflection symmetry” or refer to the empirical checks. The statement that removing stages “degrades gracefully” is also stronger than the actual validation selection rule; neutral wording that simpler configurations are recovered is sufficient.

## Smaller presentation precision points

The sixty-member table calls its unequal 1000/19000 partition “halves”; use calibration/evaluation subsets. The method calls the posterior-variance triangular solve “negligible extra cost,” which is not an algebraic fact at n=19000: it is O(n squared) per query and needs a timing comparison to justify “negligible.” A neutral statement identifying the additional triangular solve is enough. The introduction table's “Source predictions, rescored” column heading applies to OCO-2 but not quoted mechanics literature numbers; “Comparator” would fit both.

## Recommendation after these fixes

The repaired central claims are narrower than the original ones but remain scientifically useful: a retrospective comparison of exact kernel stages, an analysis of measured ensemble limits, and a coverage-score comparison accompanied by valid conditional theory. I do not see a need to demand new experiments merely to publish these *explicitly retrospective recorded-procedure results*. New experiments remain necessary for claims about a corrected estimator, independent confirmation, universal numerical-data floors, or improved end-to-end OCO-2 retrieval. Once the listed local errors are removed and the supplementary sections are brought into agreement, a YES recommendation for publication in this restricted sense is defensible. That recommendation must not be presented as a guarantee of venue acceptance or an independent rerun of the campaign.

## Final targeted recheck and vote

**Final revised-scope recommendation: YES to submission/publication as retrospective empirical research.** I refreshed the affected root-authored files after the corrections. The missing radius is now present in Theorem 6.9; the mixed-label formula remains correct; the pullback comparison is qualified and its domain is specified; the positive-denominator cases are stated; the refiner and FNO checkpoint descriptions are corrected; and the final implementation coverage paragraph now agrees with the retrospective qualification. I find no remaining material blocker within this bounded assignment.

One minor editorial remnant remains in the paragraph now titled “Retained residual archives”: a line-broken sentence still says “the residuals are what we release” before correctly explaining that the large archives are not included. Replace that fragment with “the residuals are retained in compressed archives.” This is an isolated wording inconsistency, not a reason to withhold the revised-scope YES, because actual availability is explicitly and repeatedly stated nearby. This final vote does not replace the original NO, claim review of the concurrently revised OCO-2/supplement files, or imply guaranteed acceptance at a particular venue.
