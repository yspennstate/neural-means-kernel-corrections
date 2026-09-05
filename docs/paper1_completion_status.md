# Paper 1 completion work, September 5

This branch preserves work in progress toward the completed Paper 1 revision.
The current main branch contains the recovered earlier release. The PDF files
in this branch are still that earlier release; the edited TeX is the working
manuscript and will be rebuilt with the completed sensitivity results.

At 16:10 local time (UTC+03), all ten correction-label mismatch probes, all
ten paired centering seeds, and eight of nine OCO-2 grid comparisons had
completed. The SCO2 comparison at seed 2 was running as the final item
in the fixed queue. The centering lane
has finished. The remaining training lane uses eight CPU threads, subject
to measured capacity; no jobs are appended at runtime.

The code now recomputes per-case metrics, conformal ranks and coverage,
checks the exact planned predictor set and case counts, and verifies kernel
selection against recorded validation cells. Full end-to-end aggregation
and final tables await completion of the entire campaign.

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
error was 4.45e-16 or less. The complete OCO-2 reconstruction and final
archive aggregation remain pending.

An intermediate prose build completed at 32 main-text pages and 46 supplement
pages with no unresolved references or citations. The working manuscript contains
explicit main-text proofs of the ensemble floor, sharp kernel bound and
minimax result, with the singular-Gram argument written out. The final
manuscript will receive a fresh build and visual check after integration.

Ten independent publication reviews will inspect the completed manuscript
and its pinned evidence. They have not started. The twenty older reviews in
paper/reviews concern an earlier version and are not votes on this work.

The new sensitivity-section source files are drafts awaiting generated
tables and macros; they are not yet included by main.tex or supplement.tex.
The lecture is also in production. It uses Microsoft Andrew narration and
Latin Modern mathematics, with complete proof chapters and actual benchmark
images. Earlier previews are superseded: a LaTeX-to-SVG import defect hid
fraction bars. The process-local repair has a native notation probe and
rule-removal regression controls. Neither those checks nor the source
compilation establish complete audiovisual approval.
