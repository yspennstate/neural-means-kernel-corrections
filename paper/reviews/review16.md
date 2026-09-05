# Independent reviewer 16

**Vote on publishing the original submitted manuscript: NO, pending targeted revision.**

This is a reparable research submission. I read the complete extracted original manuscript `nmkc_paper_20260903e.pdf`, the conformal statements and rank proof in the supplied supplement sources, the relevant implementation disclosure, and `campaign/uq_conformal_plam.py`. I did not read other reviewers' reports or the revision. My focus was coverage, exchangeability, retrospective data use, and the relation between exact mathematical laws and empirical outputs. The vote concerns the manuscript as written, not the impossibility of publishing this research.

## What is publishable in the contribution

The two-regime empirical comparison is useful, the matched raw-input/feature-kernel comparison and ridge-readout control address a meaningful question, and ten-seed results are more informative than a single leaderboard score. The paper is commendably explicit in several places about squared versus unsquared error, retrospective optimal convex weights, unknown native-space norms, vacuous capacity bounds, and the absence of a refined-mesh experiment. The uncertainty experiment is also useful as a negative result: both the power function and ensemble disagreement rank errors poorly, and a constant-radius conformal set is more efficient here. These findings survive the repairs below.

## Required repair 1: distinguish the deployed-population guarantee from the retrospective diagnostic

The original main text says that exchangeability "holds exactly" (§4.7), that the protocol "provides" exchangeability (§6.5), and that split conformal is the one statement holding for the pipeline "as deployed" (§6.1). The supplement's implementation record simultaneously says that the fixed public test block was read repeatedly during the first campaign, second campaign, ablations, learning curves, hard-case analysis, sixty-predictor analysis, and post-review controls. Its opening assertion that the test set was never touched during development also contradicts that fuller disclosure.

An optimizer's not consuming test labels does not itself establish independence of the ultimately reported estimator and the test block: analyst choices can be informed by earlier readouts. Repeated test access does not automatically invalidate every calculation, and I am not claiming that it proves actual leakage. It does mean the paper has not established the independence needed to attach an unconditional fresh-population guarantee to the completed adaptive research process. Selecting a calibration subset after the decisions does not erase any pre-existing dependence on that block.

Minimal repair: state the theorem conditionally for a predictor and positive score scale frozen before fresh exchangeable calibration/evaluation cases are obtained. Describe the existing public-block experiment as retrospective coverage diagnostics. State directly that no new untouched confirmation set was generated. A publication scoped as this retrospective empirical study need not fabricate or wait for new experiments; a stronger confirmatory claim would require a frozen pipeline and fresh cases.

Suggested sentence: "For a predictor and score scale fixed independently of fresh exchangeable calibration and test cases, Proposition S3.1 gives marginal finite-sample coverage. The public benchmark block was used repeatedly in this study, so the reported coverage measurements are retrospective diagnostics rather than a fresh-data certification of the development process."

## Required repair 2: state the exact-law hypotheses next to each exact-law claim

Proposition S3.1 correctly qualifies the upper coverage bound by almost-surely distinct scores, but the displayed two-sided inequality in main §6.5 omits that qualifier and follows a claim that exchangeability alone suffices. With all scores tied, the reported closed set can cover with probability one, violating the claimed upper bound for ordinary alpha. Keep the lower bound under exchangeability and add the no-ties condition for the upper bound.

The Beta law requires more than finite exchangeability. Conditional on a fixed trained predictor, iid scores with continuous CDF F yield the random conditional coverage

`p_C = F(s_(k)) ~ Beta(k, m+1-k)`.

Conditional on the calibration sample, `J | C ~ Binomial(N,p_C)`; after averaging over calibration, `J ~ BetaBinomial(N,k,m+1-k)`. The supplement essentially gives this correct derivation. The main text should repeat the iid/continuous conditions and say explicitly which law is conditional and which is marginal. A fixed realized conditional probability is a number; its Beta distribution is over repeated calibration samples. Independence and continuity are assumptions here, not consequences merely of choosing a disjoint random split.

The same ten public-block seeds are not ten independent population confirmations: they share cases and closely related predictors. Their mean, spread, and central-interval membership are useful descriptions; do not present them as a goodness-of-fit test of the exact law or as verification of coverage "at every seed."

## An exact finite-block result that can preserve the reported intervals

There is a useful distinction that avoids discarding correct retrospective calculations. Suppose the scores of all `m+N` existing cases are fixed and distinct, and a uniform calibration subset of size m is selected independently of this fixed score vector. Let J be the number of remaining evaluation scores no larger than the kth calibration order statistic. Then, even conditional on the fixed finite block,

`P(J=j) = C(k+j-1,k-1) C(m+N-k-j,m-k) / C(m+N,m)`

for `j=0,...,N`, which equals `BetaBinomial(N,k,m+1-k)`. This follows by conditioning on the total rank `k+j` of the kth calibration score: choose the other `k-1` calibration ranks below it and the `m-k` calibration ranks above it. It is a random-split statement about the existing block and makes no population-generalization claim. It does require a frozen score vector and genuinely independent split; using the same numerical seed to generate both a changing predictor and its split does not by itself supply that conditional randomization argument.

I independently recomputed the paper's central intervals using SciPy's beta-binomial quantiles:

| Nominal coverage | k | Central 95% interval for observed evaluation coverage |
|---|---:|---:|
| 90%, m=1000, N=19000 | 901 | 88.0263% to 91.8368% |
| 95%, m=1000, N=19000 | 951 | 93.5368% to 96.3000% |

The manuscript's rounded intervals are correct. I also checked the finite-block combinatorial formula against the beta-binomial pmf on a small exact-count example; the maximum numerical discrepancy was `2.22e-16`. Thus the numerical intervals should be retained with the appropriate interpretation, not labelled arithmetically wrong.

## Required repair 3: the endpoint quantile convention

The general proposition defines q as the kth of m scores, where `k=ceil((1-alpha)(m+1))`. If `alpha<1/(m+1)`, then `k=m+1` and this statistic is undefined. The implementation uses `min(k,len(cal))-1`, silently replacing it by the largest score. That replacement does not provide the claimed coverage at such alpha.

Minimal repair: define `q=+infinity` when `k=m+1`, and implement that convention for both scaled and constant scores; alternatively restrict the general proposition and CLI to `alpha>=1/(m+1)`. The reported alpha values .10 and .05 are inside the valid range, so this correction does not alter the published numbers.

## Other scope and presentation repairs

1. In the abstract and introduction, distinguish the sharper worst-case factor `Ptilde_lambda` from the GP scale `P_lambda`: Theorem 6.9 makes the former sharp, and Theorem 6.10 makes interpolation minimax at zero nugget. The positive-nugget GP scale itself is not shown minimax.
2. Theorem 6.9 acknowledges that the correction is trained on cross-fitted mean residuals but deployed on a different full-data mean. This is a real limitation and should not disappear in sentences calling the theorem a bound on the deployed estimator. Include the explicit mismatch term or refer consistently to the matched-mean idealization. Conformal prediction, when its independent-data assumptions hold, can still calibrate the actual deployed predictor without an RKHS assumption.
3. Claims based on empirical second moments, especially the 4.814% floor, apply to the specified finite pool and empirical evaluation distribution. They do not independently establish a population or data-generating floor. Much of the original already says this correctly; keep the qualifier wherever a compressed claim is repeated.
4. The initial GitHub footnote should explicitly link to the supplement and say that it contains the proofs. Each theorem/proposition/corollary with its proof elsewhere should carry a local proof-location sentence. The global "proofs in S8" note is helpful but does not satisfy the requested local navigation by itself.

## Publication recommendation after repair

I would support submission of a revised, clearly scoped empirical paper retaining these results and correcting the precise claims above. I would not represent a rewrite alone as establishing a fresh-data validation result, and venue acceptance remains uncertain. The needed changes are predominantly exact hypotheses, data-use disclosure, one quantile edge case, and consistent theorem-to-experiment interpretation; they do not require deleting the empirical core or changing its numerical tables.
