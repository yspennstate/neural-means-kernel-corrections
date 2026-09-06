# Neural means and kernel corrections for operator learning

Revised article and supplement, 6 September 2026. The matching files are
`main.pdf` (34 pages) and `supplement.pdf` (52 pages). Keep those filenames
and keep the files together: links between the PDFs use these names. Printed
section references also identify every deferred proof.

## Open in Overleaf

1. Upload `releases/nmkc_overleaf_20260906.zip` from the repository as a new project.
2. Choose **pdfLaTeX** and **main.tex**, then recompile.
3. Choose **supplement.tex** to display the companion document.

The supplied `latexmkrc` builds both documents and resolves their reciprocal
references. No shell-escape setting is required. The new archive contains
the flat paper project and both PDFs; historical review reports and old
build databases are excluded.

For a local installation with latexmk, run from the paper directory:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=output main.tex
```

Replace `main.tex` with `supplement.tex` to select the other root. The helper
creates both named PDFs and the selected `output.pdf`. Use a fresh extraction
when checking reproducibility.

## Contents

| Files | Purpose |
| --- | --- |
| `main.tex`, `supplement.tex`, other `.tex` files | Article and complete supplementary material |
| `refs.bib`, `jmlr2e.sty`, `figs/` | Bibliography, document style, and figures |
| `latexmkrc`, `build_external_documents.pl` | Paired build and external references |
| `main.pdf`, `supplement.pdf` | Matching compiled manuscripts |
| `proof_map.json` | Result-to-proof pointers |

## Correction scope

The manuscript distinguishes the implemented refiner average, which reflects
the supplied kernel field, from reflection of the full composed predictor.
Equivariance of that kernel field is a sufficient condition for the two
averages to agree. It is not established for the recorded fit and is not a
necessary condition for every possible refiner. The supplement proves a
defect bound and supplies an explicit two-point counterexample.

The effective-dimension figure is explicitly a 6,000-row subsample diagnostic.
The Fourier-input provenance, pooled training-target centering, retrospective
selection, and finite-pool RMS interpretation retain their stated scopes.
Final presentation edits label the Marchenko-Pastur experiment as one
Gaussian realization, identify the separate supplement, and display all four
refiner values in the counterexample. They change no recorded benchmark score.

## Source and evidence

Repository: https://github.com/yspennstate/neural-means-kernel-corrections

The base source identified in the article is
`50c05ff1cce32d20df446c5c57c1c20027df5875`. The corrected scientific source
before the final presentation edits was
`47c29c8a32ade3b1131d4c77999973ad171e666a`. These identify earlier states,
not the current release commit.

Per-run summaries, the fixed sensitivity evidence, data checksums, and
reproduction instructions are in the repository. See `docs/reproduce.md`
and `campaign/evidence/README.md` from the repository root. Some array-level
checks require separately retained prediction fields and checkpoints; these
large artifacts are not bundled in the public source package. Recomputing a
summary from retained records is distinct from repeating the original training.

Historical review files remain in the repository as records of their named
versions. They are not journal decisions or automatic approval of changed
source or PDF bytes.
