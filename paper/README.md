# Neural means and kernel corrections for operator learning

Revised manuscript and complete Overleaf project, 5 September 2026.

## Open in Overleaf

1. Choose **New Project → Upload Project** and upload the ZIP.
2. Set the compiler to **pdfLaTeX** and the main document to **main.tex**.
3. Recompile. The supplied `latexmkrc` also builds the supplement and resolves references between the two documents. No shell-escape setting is required.
4. To display the supplement in the editor, select **supplement.tex** as the main document and recompile.

The checked PDFs are `main.pdf` (32 pages) and `supplement.pdf` (46 pages). Keep these filenames and keep both files together when downloading them: links between the PDFs use these names. Some browser PDF viewers do not follow links to another local PDF; the printed section references remain usable.

For a local TeX Live installation with latexmk, run this command from the project directory:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=output main.tex
```

Replace `main.tex` with `supplement.tex` to select the other root. The helper creates both named PDFs as well as the selected `output.pdf`. The archive excludes old build databases and auxiliary files; if reusing an older checkout, start from a fresh extraction or clear generated build files first.

## Contents

| Item | Contents |
| --- | --- |
| `main.tex`, `supplement.tex`, other `.tex` files | Editable article and complete supplementary material |
| `refs.bib`, `jmlr2e.sty`, `figs/` | Bibliography, document style, and figure assets |
| `latexmkrc`, `build_external_documents.pl` | Paired-document build and external references |
| `main.pdf`, `supplement.pdf` | Verified compiled manuscripts |
| `reviews/publication_review.pdf` | Vote tally, main findings, and revision record |
| `reviews/review01.md` through `review20.md` | All twenty original AI-agent reviews |
| `reviews/revision*.md`, `reviews/prose_edit.md` | Targeted follow-up audits and editorial work |
| `verification/` | Numerical audit logs and clean-build report |
| `proof_map.json` | Explicit result-to-proof pointers |
| `MANIFEST.sha256` | Checksums of the packaged files |

## Publication assessment and revision

The coordinating review and all twenty separately spawned reviewing agents voted **NO on publishing the original version unchanged**. The coordinating vote on this revision is **YES for submission as an empirical research paper**. This is a submission recommendation, not a guarantee of acceptance. The twenty original votes are separate AI assessments, not external journal reviews or a statistically independent expert panel. They have not been relabeled as votes on the revision.

The revision corrects theorem hypotheses and endpoint cases, provides an explicit bound for correction-label mismatch, separates rigorous empirical RMS floors from correlation heuristics, and replaces an unsupported anisotropic-rate argument with a proved finite-design rank-truncation result. It describes the actual recorded estimators and limits the empirical claims to the available evidence. The prose has been revised for direct academic exposition. The first GitHub footnote now directs readers to the supplement, and each formal result with a deferred proof identifies its specific proof location.

## Source and numerical provenance

The matching editable source was obtained from:

https://github.com/yspennstate/neural-means-kernel-corrections

Base source commit: `50c05ff1cce32d20df446c5c57c1c20027df5875`.

The original repository PDF is byte-for-byte identical to the uploaded `nmkc_paper_20260903e.pdf`. Its SHA-256 is:

```text
d7952a24060c555e0b86981d8a2eef0365431dbd625f2226ef674f0cbee1a072
```

The revised article, supplement, editable LaTeX sources, figures, and paired-document build files are synchronized with the public repository's `paper/` directory. The commit above identifies the prior version used as the base for this revision. The current online supplement is `paper/supplement.pdf`; it contains the full proofs at the locations cited by each formal result.

The repository's reported-number audit found zero discrepancies against its summary artifacts, and its fifteen algebraic checks passed. The two compile roots were tested from clean source without existing auxiliary files or prebuilt PDFs, using Overleaf's `output` job name. Both builds had no unresolved citations or references, duplicate labels, missing PDF destinations, or overfull boxes. The build report records source checksums and reciprocal PDF links.

No neural training campaigns were rerun for this revision. The authorized Caltech connection failed before authentication with `Network is unreachable`, so no remote sources or new simulation results were obtained there. The retained measurements describe the recorded procedure, including pooled target centering in the foldwise kernel channel. No unperformed corrected-centering rerun is claimed. Large checkpoints and prediction/residual archives are not included; the original source describes them as retained and available on request. The Overleaf build needs none of those training artifacts.
