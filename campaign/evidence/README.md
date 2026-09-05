# Completed implementation-sensitivity evidence

`paper1_completed_evidence_20260905.zip` contains the completed 5 September
2026 follow-up campaign: ten paired mechanics target-centering reruns, ten
fixed-estimator correction-label checks, and three paired OCO-2 seeds in
each of three bands. These are retrospective checks on the existing public
benchmark cases, separate from the historical ten-seed OCO-2 table.

Archive SHA-256:

```text
2cb72d8c84c800113406ceb0f5790000bef62f5c90bccf81cafdb09aed36026f
```

The archive is 27,793,233 bytes and contains an exact manifest for 454 files:
executed source, configuration and input hashes, per-case error arrays,
calibration records, kernel-grid cells, and independent prediction-check
results. The large trained checkpoints and full prediction fields remain in
the retained campaign directories; they are not inside this archive.

Extract into an empty directory, here called `evidence`. From the repository
root, with NumPy, SciPy and Matplotlib installed, run:

```sh
python campaign/aggregate_sensitivity.py --root evidence --out summary.json
python campaign/render_sensitivity.py --root evidence --paper paper --summary summary_with_metric_check.json
```

The first command validates the exact file set and every file hash before
checking run completion, splits, ordering, paired errors, calibration and
selection, and recomputing the aggregates. The second repeats these checks,
adds the independent reconstruction of the sixty-predictor matrix analysis,
and generates the manuscript tables, macros and paired-seed figure. It reads
the historical reference from `campaign/collected/dgx/seedarch.json` and
writes `paper/sensitivity_generation.json` with output hashes.

The committed `campaign/sensitivity_summary_final.json` is the first
command's unchanged output. `campaign/sensitivity_summary_with_metric_check.json`
also contains the matrix reconstruction. Standard deviations describe
variation over seeds on shared test cases; they are not population
confidence intervals. The expanded grid worsens some test results, which
remain in both the records and the paper.

Raw-prediction checks were run against the retained prediction fields before
collection. Their reports are in the archive and their drivers are under
`campaign/`; repeating those checks requires the large fields or a rerun.
This archive directly supports table reconstruction from saved per-case
errors, rather than claiming to bundle the entire training environment.
