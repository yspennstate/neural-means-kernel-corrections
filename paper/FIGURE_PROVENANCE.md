# Revised supplementary figures

`figs/spectra.pdf` comes from `fig_spectrum_diagnostic.py` and the complete
numerical record in `campaign/collected/spectrum_20260906`. Its design has 6000
rows sampled from 19000 training rows. The marked nugget is fixed at 1e-5;
neither panel describes the full deployed smoother. `spectra_provenance.json`
binds the current PDF to its inputs. The older `spectra_reference_repair.json`
is explicitly superseded: it records an earlier label-only repair, not the
current figure. The historical spectrum section in `figures.py` is not the
producer of this replacement.

`figs/floor.pdf` comes from `fig_floor.py`. It shows the five rounded pairs in
`runs/fourier_wco2.log` and the log's SiLU error reference. The underlying
predictions and exact moment convention are unavailable, so this figure has no
theorem-admission boundary. Unreproducible architecture-comparison bars have
also been removed. `floor_provenance.json` records the exact log hash, values,
producer and PDF hash. Regeneration reads the archived log; it does not invent
replacement prediction arrays.
