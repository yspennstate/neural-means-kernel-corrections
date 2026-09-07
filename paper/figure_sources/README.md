These files regenerate `figs/floor.pdf`, `figs/scaling_seeded.pdf`, and
`figs/seedspread.pdf` from the recorded numerical values supplied with the
manuscript repository.

Run `python figure_sources/regenerate_figures.py` with Python 3, NumPy, and
Matplotlib installed. LaTeX compilation uses the supplied PDFs and does not
require Python.

`provenance.json` identifies the input records and their SHA-256 checksums.
The plotting script preserves their numerical precision. The Fourier sweep
uses five rounded error/correlation pairs and a reference error from the
source log. The underlying prediction arrays and exact correlation convention
are unavailable, so the scatter plot does not include a theoretical admission
boundary. ClimSim error bars are sample standard deviations over
five seeds at one million training examples and three seeds at each larger
size. The kernel uses 6000 fitting rows at each size. Published reference
values are comparison values reported in the manuscript, with the protocol
limitations described in its captions.
