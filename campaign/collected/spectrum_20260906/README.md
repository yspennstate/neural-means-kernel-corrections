# Structural-mechanics spectrum diagnostic

This is a new, explicitly specified 6000-row subsample calculation. It is
neither a recovered historical array nor the spectrum of the full deployed fit.

`result.json` records the input hashes, split, sampling generators, fixed kernel
parameters, software versions, one-thread BLAS limit and two independent trace
calculations. `design.npz` retains selected raw and standardized loads, the
training mean and scale, training indices, and the median-distance and spectrum
subsample identities. `spectrum.npz` retains all eigenvalues and the plotted
effective-dimension curve. No predictor training or model selection occurred.

The eigenvalue sum gives 103.11885064181605 at lambda = 1e-5. Cholesky solves
give 103.11885064179933 from `6000 - 6000*lambda*trace(A^{-1})`. Their absolute
difference is 1.6726176e-11. The smallest eigenvalue is positive. The leading
eight modes carry 99.0207797681% of the matrix trace; seven carry 98.8986408194%.
This illustrates why trace mass and effective dimension are different summaries.

Regenerate the plot from the retained records:

```bash
python fig_spectrum_diagnostic.py --results campaign/collected/spectrum_20260906
```

Regenerate the diagnostic from the original structural-mechanics input archive
on a research host, in a fresh output directory:

```bash
nice -n 15 python campaign/spectrum_diagnostic.py --data /path/to/structmech --out /path/to/fresh-output
```

The producer uses one BLAS thread and samples CPU pressure before each dense
stage. It waits up to 15 minutes for headroom, then stops if none appears. This
does not alter another job or add work to a shared simulation queue. The original
input hashes are in the result record; `loads.npy` and `idx_train.npy` are needed
for this full regeneration. The saved standardized loads suffice to reconstruct
the diagnostic Gram matrix without the full input archive.
