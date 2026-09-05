# Revision record 06

Edited only `overleaf/oco2.tex` and `overleaf/supp_oco2.tex`. The readout table body required no numerical change and was left intact. All pre-existing labels are preserved. Three labels were added for the reconstruction and scoring formulas.

Changes implemented:

- Added explicit PCA reconstruction, reduced-relative-error and retained-radiance-relative-error formulas, including the offsets in the radiance denominator.
- Explained that squared loss remains coordinate-separable with fixed case weights, while the square root in mean relative norms couples coordinates; narrowed all selector optimality claims accordingly.
- Filled the omitted ARD radiance entry with **0.204 ± 0.087%**, aggregated from the original ten records, and retained the two-mode selection explanation.
- Corrected the weighted-selector rounding claim to differences below 0.001 percentage points; explicitly reported the O2 combination's tiny reduced-metric loss relative to the flat-feature kernel head.
- Limited ridge conclusions to the tested frozen-feature linear-refit control and clarified that matching kernel grids does not match total neural-pretraining computation.
- Removed a claim that wider-grid runs were currently underway; described the reported grid's boundary selection without promising new experiments.
- Recast all public-benchmark comparisons, margin bins, admission counts, and bootstrap intervals as retrospective diagnostics. Removed claims that a margin of 0.25 supplies a dependable future selection threshold.
- Removed all heuristic constant-versus-rate interpretations. Learning-curve slopes are finite-range descriptions; the new rank-truncation proposition is referenced only as conditional finite-design theory, whose assumptions these sweeps do not verify.
- Replaced architecture-class and infinite-seed floor claims with measured finite-pool second moments and expressly labeled equicorrelation extrapolations.
- Preserved source-versus-combination numerical comparisons and the weak-CO2 seed-0 exception, with its stored paired-bootstrap interval.
- Corrected the supplementary scaling comparison that incorrectly associated a 3.8% result with the main 250-epoch table. The main 4.12% result is now distinguished from the separate 750-epoch 3.71% combination.
- Clarified unequal ClimSim training budgets, exploratory baseline comparison, seed-specific test blocks, and withdrawal of earlier leaked small-sample results. Removed unsupported assertions about all attainable dense-kernel budgets and all canonical ClimSim splits.
- Explicitly stated that this revision retains existing run records and introduces no new training.

Validation: checked all original labels remain, begin/end environment counts match, and brace balances are zero in both edited files. The OCO-2 means, standard deviations, selector differences, and bootstrap exception were independently checked against the thirty original JSON summaries during review06. No model was retrained and no raw prediction array was regenerated. Final integrated LaTeX compilation and rendered-page QA remain the root agent's responsibility.
