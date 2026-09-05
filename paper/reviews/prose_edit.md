# Prose revision

Edited only `overleaf/intro.tex`, `overleaf/discussion.tex`, and the abstract in `overleaf/main.tex`, as assigned. The abstract is approximately 212 whitespace-separated words before macro expansion.

The introduction now opens with the scientific choice between residual correction and a learned-feature kernel head, situates both in established work, and introduces the measured contributions. The discussion develops the implications of the results, followed by their limits and the specific experiments that would address them. Repeated audit language such as “recorded,” “this revision,” and “shipped estimator” has been removed from these sections.

All numerical results and macros were preserved. The revisions retain the fixed-pool and RMS scope of the ensemble bound, the retained-PCA scope of OCO-2 rescoring, the absence of a demonstrated benchmark noise floor, pooled-centering and correction-label issues, the conditions for kernel and conformal statements, and the retrospective character of comparisons across repeatedly inspected public test blocks. The supplementary proof pointers remain explicit.

The introduction's GitHub footnote now links directly to the online supplement and directs readers to the accompanying supplement for the current proofs, since the cited repository snapshot predates these changes.

The preamble, funding, declarations, and availability sections were not edited. The existing separate “Supplementary material” paragraph in `main.tex` still says the online supplement is built from the same sources as this manuscript; the parent editor should align that statement with the new footnote if the revised sources have not been uploaded publicly.

This was an editorial pass on the scientifically corrected working files. I did not rerun experiments or certify journal acceptance.
