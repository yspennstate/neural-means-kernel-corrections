# Reviewer 04 revision summary

Edited only `overleaf/supp_experiments.tex`; no other Overleaf source changed.

- Recast supplementary experiments as a retrospective study with reused benchmark blocks. Removed pristine-test, prospective-verification, and unconditional publication-baseline significance/equivalence language.
- Distinguished the heuristic equal-error/centered-correlation references from rigorous bounds using uncentered residual second moments. Added the recalculated six-member floors, 4.820 ± 0.062% and finite-six bound 4.849 ± 0.060%, while preserving the independently verified 4.814% sixty-predictor empirical bound.
- Corrected five-member provenance: the 0.980 point is a test quadratic risk at stored deployed global weights, not a test simplex minimum. The six-member 0.972 point remains a validation-matrix optimum. Clarified that Corollary 6.4 does not upper-bound arbitrary ensembles.
- Identified the .938 dispersion comparison as same-validation E1/E2, removed cross-predictor conversions and incorrect Jensen implications, and reported the six-member test-E1/validation-E2 ratio .9357 ± .0110 as a different, mixed quantity.
- Removed exact admission/exchange-rate interpretations of centered-correlation summaries. Retained the corresponding observed matrix optima and error changes.
- Restricted all seed-floor claims to measured pools. Removed claims that ten seeds exhaust reseeding, that small kernel gains prove RKHS irreducibility, or that shared errors identify a data-generating floor. Qualified finite-range scaling fits.
- Described the actual mixed training labels: foldwise kernel fits with pooled training target centering and in-sample neural predictions. Removed native-norm lower-bound claims for the deployed residual from this diagnostic; retained actual fitted-function norm measurements and the table.
- Qualified conformal coverage observations as retrospective split diagnostics. Exact Beta/Beta–binomial statements now explicitly require the conditional i.i.d.-continuous reference model; disjointness alone is not asserted to establish theorem hypotheses.
- Corrected stale signed-optimum equality and uncertainty/significance claims, removed unreported baseline-variance inferences, and qualified regularization-dependent sixty-member affine-fit behavior.
- Checked within-architecture square-root minimum-off-diagonal values directly from the released S matrix, including mlpR = kernel-conditioned refiner. The erroneous 0.021/RMS4.979 architecture-drop narrative is absent from this supplementary revision.

Validation: all original labels, figure paths and table contents were preserved; table/figure environments are balanced. No experiments were rerun. Full manuscript compilation and visual inspection remain with the integrating parent agent.
