# Retained RMT source and result scope

This note records limitations identified on 6 September 2026. The original
exploratory source and result record are retained; no corrected experiment is
being substituted for them. A separate source repair and review are pending.

## Exact retained inputs

| File | SHA-256 |
| --- | --- |
| `campaign/dgx_checks/ens_rmt_dgx.py` (LF line endings) | `849b441446b9d5c8b87e637abd583dc5ebea54ef8e8082fc85f350edaf552d2c` |
| `campaign/collected/dgx/sm_ens_rmt.json` | `e028360381e84c2a530093303ebd55c99e6f4b52960cb8813fcbde968a3cd338` |
| `paper/macros.tex` | `a00b3c4c623d91dee143c1c46de558cecb9cb0efac931ecb751a939c0aa2d5a1` |

These hashes identify retained bytes, not an independent rerun of the full
experiment or an attestation of the original execution environment.

## What the manuscript uses

The row "Sixty, per-pixel affine, leave-one-out ridge, fitted" in
`paper/experiments.tex` uses `saPerpixSixtyLoo`. It is the retained
`results.sixty_ridge_loo` value `0.045467826219122015`, multiplied by 100 and
rounded to three decimals: **4.547%**. The script assigns this value before
the subsequent shrinkage block.

Five other macros in the same macro group carry the six-mean LOO result and
the 120- and 300-row calibration ridge results; they are not used in the
current manuscript text. `campaign/audit_reported_macros.py` checks all six
against their recorded fields. The top-level record stores error fractions;
the small-calibration subrecord already stores percentages.

## Identified limitations

The shrinkage block fits global convex weights using all calibration labels,
then reuses those weights while selecting shrinkage in five folds. A held-out
fold's labels can therefore affect its predictions through those weights.
This is not a valid held-out selection procedure. The separate 19,000-case
evaluation block remains distinct: this finding concerns calibration selection,
and does not by itself show that evaluation labels entered training.

The small-calibration loop also stores the mean predicted stress under the
name `pcr_mp`, instead of applying the relative-error function to those
predictions. Its retained values, `12377.9518` and `12378.164`, are not error
percentages. The top-level PCR computation does apply the error function.
No small-calibration PCR value is used in the paper's reported row or its six
RMT-record macros.

Reading the complete retained script shows that the shrinkage block allocates
new fits and outputs without mutating the input prediction or target arrays.
An independent small synthetic execution of the pinned source, with the whole
shrinkage stage omitted, left all 16 remaining top-level result fields exactly
unchanged, including the nested small-calibration results. This is a dependency
control, not a rerun on the original data. Together with the field-to-macro
trace, it supports the narrow conclusion that this shrinkage defect does not
change the paper's reported 4.547% row.

Neither this trace nor a successful macro check validates all RMT methods,
their asymptotic interpretation, or the historical source package as a whole.
The exploratory shrinkage and small-calibration PCR outputs must not be used
as validated method results. Corrected code must be reviewed and any new
experiment must be recorded separately before its results are adopted.
