# Corrected ensemble-study candidate

The candidate repairs two reporting and selection paths in `ens_rmt_dgx.py`:

- Each shrinkage-validation fold fits its convex weights and chooses its ridge penalty using only that fold's training rows. The held labels enter the score after the predictor has been fitted.
- The small-calibration PCR path computes mean per-case relative error against the evaluation targets before averaging draws and converting to percent. Its old aggregation averaged the prediction matrix itself.

These corrections have been checked on small adversarial inputs. The fold helper also passed an independent deleted-row least-squares check and held-label perturbation check. They have not yet been run on the full retained dataset, so the candidate provides no replacement scientific score.

`sm_ens_rmt.json` belongs to the historical source. The corrected driver uses the distinct output `sm_ens_rmt_corrected_v1.json`, refuses an existing corrected output, and records hashes of its source and `fold_selection.py` at startup. Source hashes are part of provenance; they do not replace a pinned launch command, data hashes, split record, environment, or completion receipt.

The historical source SHA-256 is `849b441446b9d5c8b87e637abd583dc5ebea54ef8e8082fc85f350edaf552d2c` with LF endings (`f2c35624e49899618fc944e9cb19198676008c3b7521f182b6232ddf897d26dc` with CRLF endings). The historical result SHA-256 is `e028360381e84c2a530093303ebd55c99e6f4b52960cb8813fcbde968a3cd338`. Keep that source alongside its result when integrating the corrected candidate.

The original `sixty_ridge_loo` value, printed as 4.547% in the article, is calculated before the shrinkage block. The two affected small-calibration PCR fields and the shrinkage-selection fields do not feed that value. This separation does not establish the correctness of every other estimator in the study.

The inherited PCR threshold `2.858 * median(covariance eigenvalues)` is retained as a heuristic. It is not asserted to be a Marchenko-Pastur edge or a literal Gavish–Donoho singular-value threshold. The legacy `pcr_mp` field names remain for comparison; the corrected output records this qualification.

Run the small tests from this directory with BLAS restricted to one thread:

```text
python -m unittest -v test_fold_selection.py test_rmt_metrics.py
```

Full execution still needs an authorized host, a live resource/hold check, an input-based peak-memory estimate, and a bounded launch. In particular, the historical monolithic loader materializes the full predictor pool and several copies. These small tests do not establish that a full run fits the host's memory limit.
