# Neural means and kernel corrections for operator learning

Final version, 7 September 2026. This directory holds the main article in the
JMLR submission format, the online supplement (Online Appendix 1), their
sources, figures, bibliography, and compiled PDFs.

## Documents

| File | Contents |
| --- | --- |
| `main.tex`, `main.pdf` | The main article. A single self-contained source built with the unmodified `jmlr2e.sty`; its preprint option leaves the editor-assigned publication information unset for initial submission. |
| `supplement.tex`, `supp_*.tex`, `proofs.tex`, `impl.tex`, `verification.tex`, `table_*.tex`, `supplement.pdf` | Online Appendix 1: the full proofs, additional experiments, implementation record, and the paired sensitivity experiments. Sections and results carry the prefix S. |
| `macros.tex`, `sensitivity_macros.tex` | Numerical macros shared by the supplement; the article carries its own copies inline. |
| `refs.bib`, `jmlr2e.sty`, `figs/` | Bibliography, document style, and figure assets. |
| `latexmkrc`, `build_external_documents.pl` | Paired build: compiles both roots and resolves references between them. |
| `figure_sources/` | Numerical inputs, provenance, and the script that regenerates three of the figures. |
| `EXPERIMENTS.md` | Maps each experiment to its implementation and record paths at a pinned repository commit. |
| `proof_map.json` | Result-to-proof pointers. |
| `MANIFEST.sha256` | Checksums of the files in this directory. |

References from the article into the supplement resolve to the S-numbered
results of `supplement.pdf`; the article carries those numbers explicitly, so
it also compiles on its own. References from the supplement back to the
article use the article's consecutive theorem numbering and are resolved by
the paired build.

## Building

With latexmk (TeX Live or MiKTeX), from this directory:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The supplied `latexmkrc` also compiles `supplement.tex` and resolves the
references between the two documents; no shell-escape setting is required.
Temporary files go under `nmkc-build/`. On Overleaf, upload the directory as a
project, choose pdfLaTeX, and set `main.tex` as the main document; select
`supplement.tex` instead to display the supplement. Keep the compiled files
named `main.pdf` and `supplement.pdf` together so links between them continue
to work.

## Figures

All figures required to compile the documents are included. The numerical
inputs and the Python script for the three regenerated figures are in
`figure_sources/`; Python is not required for compilation.

## Code and data

The experimental code and the released benchmark summaries are in the parent
repository, https://github.com/yspennstate/neural-means-kernel-corrections.
Data and code availability is described in the article.
