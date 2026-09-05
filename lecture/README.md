# Neural means and kernel corrections: a visual lecture

Production is in progress. The target is a two-hour lecture, with complete
proofs of the ensemble floor, the sharp kernel bound and minimax recovery.
The final duration will be measured from the finished media, not inferred
from a word count. Paper completion and the bounded Caltech campaign remain
the first priority.

The visual argument proceeds from a loaded elastic body to residual vectors,
shared error directions, kernel projection, indistinguishable functions,
learned input geometry and calibrated prediction balls. Every board has a
specific explanatory visual. Toy geometries are labelled as illustrations;
benchmark images and plots carry their data source and metric.

The narration uses Microsoft `en-US-AndrewMultilingualNeural`, rate `+10%`,
following the owner's September 5 preference. Text is sent to Microsoft's
speech service. Mathematical notation is authored as explicit LaTeX and
rendered with `pdflatex` and the PDF route of `dvisvgm`.

Record narration before rendering. Each audio receipt binds the current text,
voice, settings and actual WAV hash. Changed text or settings invalidate the
recording. Visuals are timed from those recordings. Final equations have no
silent Unicode fallback. Run one render process at a time, with bounded
encoder threads and BelowNormal priority on Windows; read current compute
telemetry before starting.

The lecture sources will be linked to the final manuscript and evidence
snapshot. Current scripts must not describe unfinished sensitivity runs as
completed results.

The first narrated draft is 503.289 seconds at 1280 by 720, with 18 recorded
segments. Its author integrity check verified all 50 frozen inputs, the audio
identities and timeline coverage. This is not a claim of auditory review or
approval of the completed two-hour lecture. Eleven topic chapters are drafted;
the closing synthesis awaits the full sensitivity results.

`preflight_equations.py` compiles all authored equations in one LaTeX document
and measures their boxes before expensive video rendering. Final glyph bounds
and page placement still need visual inspection. `verify_render.py` verifies
the frozen inputs and narration against the actual media and extracts a frame
from every spoken segment for that inspection.

Every long render reads an immutable build snapshot. New renders use a
process-local CPU encoder policy with one encoder thread and an eight-frame
queue; the shared Manim installation is not edited. The policy records the
installed writer's hash. Scientific assets retain their input hashes, case
selection rules, units and reconstruction checks. OCO-2 spectra are benchmark
simulator reconstructions; mechanics images and calibration deciles come from
saved predictions, including unfavorable cases.
