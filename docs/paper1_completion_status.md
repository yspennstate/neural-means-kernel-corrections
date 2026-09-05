# Paper 1 completion work, September 5

This branch preserves work in progress toward the completed Paper 1 revision.
The current main branch contains the recovered earlier release. The PDF files
in this branch are still that earlier release; the edited TeX is the working
manuscript and will be rebuilt with the completed sensitivity results.

At 11:22 local time (UTC+03), all ten correction-label mismatch probes and
three paired centering seeds had completed. The fourth centering seed and
the first OCO-2 grid comparison were running. The fixed design is ten
centering pairs, ten mismatch probes, and three seeds in each of three OCO-2
bands. The two active lanes use eight CPU threads each, subject to measured
capacity; no jobs are appended to their queues at runtime.

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
and all twenty five-/six-member validation optima checked. This verifies
saved matrices, not their field-array construction.
An intermediate prose build completed at 31 main-text pages and 46 supplement
pages with no unresolved references or citations; later edits and new results
will receive a fresh build and visual check.

Ten independent publication reviews will inspect the completed manuscript
and its pinned evidence. They have not started. The twenty older reviews in
paper/reviews concern an earlier version and are not votes on this work.

The new sensitivity-section source files are drafts awaiting generated
tables and macros; they are not yet included by main.tex or supplement.tex.
The prepared metric check will rescore all sixty retained members and ten
pipelines, compare plain and trapezoidal norms, and reconstruct the pool's
evaluation matrix by streaming float64 residual blocks. It has not run yet.
