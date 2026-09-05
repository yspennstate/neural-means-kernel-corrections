# Independent reviewer 13

**Vote on the submitted `nmkc_paper_20260903e.pdf`: NO, pending a limited but necessary revision.**

**Scope:** I read the submitted main manuscript, checked the relevant supplementary statements and proofs, inspected `jpl_data.py` and `campaign/jpl_seeded.py`, and independently aggregated the thirty original OCO-2 run records in `campaign/collected/box*/oco_<band>_s<seed>.json`. I did not consult other reviewers' reports and did not retrain the networks or independently regenerate the external emulator predictions. My standard is publishability as an empirical research paper, not acceptance at a particular selective venue.

The empirical study has a publishable core. The clear distinction between a residual correction on structural mechanics and a direct-target kernel readout on learned OCO-2 features is useful. The same-kernel representation comparison, six-member coordinate selector, ten-seed replications, and comparisons against the external emulator's stored predictions make a worthwhile controlled case study. The manuscript already acknowledges its most substantial scientific limits: incomplete convergence, restricted hyperparameter grids, validation-to-test shift, differing uncertainty sources, and the absence of an end-to-end retrieval experiment. Nevertheless, several prominent claims still contradict the actual experimental definitions or the released numbers. These should be repaired before publication.

## 1. The combination does not outperform every constituent on both metrics

The last paragraph of Section 6.2 says that the combination improving on every single model on both metrics is an empirical finding. It is not. Independently aggregating the canonical original campaign JSON records gives the following mean percentages:

| Band | Flat feature head, reduced | Combination, reduced | External emulator, reduced | Combination, radiance | External emulator, radiance |
|---|---:|---:|---:|---:|---:|
| O2 | 4.11080 | 4.11510 | 16.88910 | 0.02940 | 0.04480 |
| WCO2 | 16.27688 | 16.27925 | 24.06040 | 0.05073 | 0.05990 |
| SCO2 | 8.08422 | 8.08365 | 16.14480 | 0.05837 | 0.11470 |

The differences are small, but the claim is categorical. O2 and WCO2 have a small reduced-error cost relative to the flat feature head. The combination beats the external kernel-flow baseline on both metrics at 10/10, 9/10, and 10/10 seeds respectively, which does reproduce the manuscript's properly qualified baseline claim. The flat feature head beats its parent network on reduced error at 10/10 seeds on each band, also reproducing the manuscript.

**Minimum fix:** Replace the universal dominance claim by: “The selector improves both rescored criteria relative to the external kernel-flow baseline in the bandwise averages, while retaining nearly the flat feature head's reduced-coordinate accuracy.” Keep the 9/10 WCO2 qualification. Do not characterize the tiny increase relative to the strongest reduced-error constituent as an improvement.

## 2. The external baseline's evaluation criteria must be described consistently

Table 1's caption calls these “both of that paper's error metrics,” and the introduction repeats “both of that problem's error metrics.” Section 5 correctly says they are two rescoring criteria defined in this manuscript on the released reduced representation and do not reproduce the source paper's acceptance test. That discrepancy affects the headline interpretation of the paper, not just notation.

The code reconstructs both predictions and targets from forty retained components, using

`R(z) = (z * s_z + m_z) @ P + m`.

It reports `mean(||R(zhat)-R(z)|| / ||R(z)||)`. This is a valid comparison of the retained-component reconstructions, but it does not include error relative to the original untruncated spectrum, instrument convolution, Jacobian quality, noise-normalized acceptance, or retrieval effects.

**Minimum fix:** Use “our two rescoring criteria on the released reduced representation” in every overview and caption. Name the second metric “retained-component radiance error” at first definition. The current Section 5 limits should remain adjacent to the main comparison.

## 3. Direct-target and residual heads must remain distinct throughout

The implementation is unambiguous: `matern_head` fits `alpha = (K+n*lambda*I)^(-1) Ytr`, even when its inputs are frozen neural features. It replaces the readout; it does not add a kernel fit of the network's residuals. Section 3.3 now explains this correctly. The related-work description, however, calls the feature-space stage the same construction as a post-hoc residual model, which can undo that distinction.

For frozen features the two estimators are

\[
\widehat G_{\mathrm{direct}}(u)=k_\phi(u,X)A^{-1}Y,
\qquad
\widehat G_{\mathrm{residual}}(u)=m(u)+k_\phi(u,X)A^{-1}[Y-m(X)],
\]

with `A=K_phi+n*lambda*I`. Their difference is `m(u)-k_phi(u,X)A^{-1}m(X)`, which need not vanish. The OCO-2 results establish the quality of the direct head under the reported protocol, not superiority of residual correction in feature space or a direct-versus-residual comparison.

**Minimum fix:** Add the direct-target formula beside the residual formula and explicitly identify which experiments use each. Revise related work to say the constructions share frozen learned features but differ in their regression targets. Claim novelty for the comparisons and diagnostics, not these established constructions.

## 4. Give exact loss equations and separate the two reasons selector optimality fails

The script implements three distinct network objectives:

\[
L_{\rm flat}=\mathbb E\frac{\|\widehat z-z\|}{\|z\|},\quad
L_{\rm weighted}=\mathbb E\frac{\|s_z\odot(\widehat z-z)\|}{\|s_z\odot z\|},\quad
L_{\rm radiance}=\mathbb E\frac{\|R(\widehat z)-R(z)\|}{\|R(z)\|}.
\]

With orthonormal PCA rows the last two share a numerator, but their denominators differ. A brief displayed definition is more reliable than repeatedly calling the second loss a metric-weighted loss. It also explains why “train in the exact reported metric” is not established as a universal improvement: at the given training budget, the exact radiance loss can lose to the surrogate-weighted objective.

The selector uses per-coordinate unweighted validation squared errors. Its separability theorem covers squared objectives with a fixed sample-weight function and positive fixed coordinate weights. Sample-dependent denominators do **not**, by themselves, destroy separability of squared relative risk: the supplementary corollary correctly covers this with `a(u)=1/||z||^2`. The square root in the reported mean norm prevents coordinatewise separation; changing from reduced to radiance normalization additionally changes the sample-weight function. Section 6.2 presently attributes nonseparability to denominators without making this distinction.

**Minimum fix:** State these two facts precisely. Restrict simultaneous optimality to squared objectives sharing the same sample weights. Retain the weighted-selector sensitivity analysis and describe its similar observed scores as an empirical result.

## 5. The readout and representation controls support narrower claims than the prose

The ridge readout is a useful control: it uses exactly the frozen features and selects the ridge penalty by validation. But its fitted coefficients minimize unweighted squared training error, whereas the original neural output layer was trained through a relative-norm objective. Therefore the sentence “the gain is the kernel's and not a refitting of the readout” rules out more than this control tests. It rules out the specific validated ridge readout tested, not every linear readout retrained under the network's objective. The same-kernel raw-versus-feature experiment matches the kernel fitting and tuning protocol; it does not match total computation, because only the feature pipeline has supervised neural pretraining.

Furthermore, the pullback identity describes an RKHS but does not guarantee that a learned feature map lowers its target norm or improves error. Neither it nor these experiments proves the broad assertion that no rescaling can produce such an advantage. The main text should not say a stationary kernel “reaches only its native space” as an impossibility claim about learning targets outside that space.

**Minimum fixes:** Say “the feature kernel beats the tested validated ridge readout”; say “matched kernel-fitting and tuning budget”; describe the failure of the tested raw-input metric adaptations as experimental rather than universal. Present the RKHS ratios as finite-design diagnostics, as the later caveats already do. Retain the restricted-grid boundary disclosure and remove any implication that an unfinished extended sweep establishes robustness.

## Recommendation after revision

These repairs preserve the principal numerical results and require no invented experiments. I would support publication of a revised empirical case study that makes them, carries correct proof pointers, and incorporates any independently identified mathematical repairs. I would not insist on proving a universal representation advantage, retraining every baseline, or establishing end-to-end satellite retrieval superiority, provided those are not claimed. My present NO is a request to correct concrete claim-to-evidence mismatches, not a judgment that the work lacks publishable content.
