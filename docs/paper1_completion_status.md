# Paper 1 completion work, September 5

This branch preserves work in progress toward the completed Paper 1 revision.
The current main branch contains the recovered earlier release. The PDF files
in this branch are still that earlier release; the edited TeX is the working
manuscript and will be rebuilt with the completed sensitivity results.

All ten correction-label checks, ten paired centering seeds, and nine OCO-2
grid comparisons have completed. The fixed queue is empty. The archive was
downloaded and its exact 454-file manifest verified before extraction.

The code now recomputes per-case metrics, conformal ranks and coverage,
checks the exact planned predictor set and case counts, and verifies kernel
selection against recorded validation cells. Complete aggregation passed,
and the final tables, macros and paired-seed figure have been generated.

Editorial checks have corrected the refiner description, optional
kernel-flow implementation scope, cost units, conformal-ball radius
terminology, and the scope of the retained half-precision archives.
The hardware description now distinguishes 20 physical cores from 40
hardware threads. A second matrix optimizer, using active-set solves and
explicit optimality bounds, reproduces the sixty-member hindsight optimum
and all twenty five-/six-member validation optima checked. A separate float64
reconstruction from all sixty saved prediction fields also agrees with the
retained pool floor and optimum to four decimal places in percentage units.
The ten historical pipelines were rescored under plain and trapezoidal grid
norms; those are distinct metrics, not interchangeable comparison tables.

The independent centering checker has recomputed all twenty arms from the
saved 20000-case prediction fields. It checks relative and absolute errors,
the disagreement scale through pairwise member differences, and scalar
summation controls. The largest absolute difference in a per-case relative
error was 4.45e-16 or less. The OCO-2 reconstruction passed for all fourteen
predictors in all eighteen band/seed/grid scenarios. The per-case evidence,
executed source and check records are in `campaign/evidence/`.

An intermediate prose build completed at 32 main-text pages and 46 supplement
pages with no unresolved references or citations. The working manuscript contains
explicit main-text proofs of the ensemble floor, sharp kernel bound and
minimax result, with the singular-Gram argument written out. The final
manuscript is now undergoing a fresh paired build and visual check after
integration. A clean MiKTeX build exposed a bibliography lookup issue in
nested output directories; the portable helper now stages exact bibliography
copies beside each child's auxiliary file.

The completed 34-page article and 51-page supplement have been frozen with
their evidence for ten fresh independent Codex CLI reviews. The panel is in
progress; the twenty older reviews in paper/reviews are not votes on these
bytes. The first returned comments are being integrated into the working
source, which will receive a new paired build and revision review. The frozen
panel packet remains unchanged. The corrections clarify reflection handling,
retrospective test-block fitting and the internal effective-dimension reference,
and make the campaign centering choice and resume provenance explicit.

The new sensitivity sections and generated tables are included in both
roots, with Supplement S11 giving the detailed protocols and full records.
The lecture is also in production. It uses Microsoft Andrew narration and
Latin Modern mathematics, with complete proof chapters and actual benchmark
images. Earlier previews are superseded: a LaTeX-to-SVG import defect hid
fraction bars. The process-local repair has a native notation probe and
rule-removal regression controls. Neither those checks nor the source
compilation establish complete audiovisual approval.
