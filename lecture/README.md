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
Latin Modern matches the Computer Modern family used in the manuscript.
The main displayed equations use a fixed font size; long expressions must
be broken into lines instead of being shrunk to fit. Preflight and rendering
import the same LaTeX preamble from `tex_style.py`. A preamble change creates
new TeX cache keys; frozen earlier font assets are retained as evidence.

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
segments. It is superseded: the author found that PDF-derived fraction bars
and other LaTeX rules were lost during SVG import. Earlier source, audio,
timeline and layout checks did not detect this notation defect. Corrected
renders require native-resolution inspection of every board, explicitly
checking fractions, radicals, accents, brackets, and sub/superscripts against
the authored mathematics. No auditory or full audiovisual approval has been
given. All twelve chapters are drafted. The closing chapter uses the completed
sensitivity campaign, with three checked figures and 18 equation segments.
All 252 narration segments are recorded and checked against the current text.
They contain 6,718.992 seconds of speech. Corrected chapter renders are still
in production. `verify_narration.py` also counts the stored PCM frames
independently; neither duration check certifies pronunciation or listening quality.

`preflight_equations.py` compiles all authored equations in one LaTeX document
and measures their boxes before expensive video rendering. Final glyph bounds
and page placement still need visual inspection. `verify_render.py` verifies
the frozen inputs and narration against the actual media and extracts a frame
from every spoken segment for that inspection.

Every long render reads an immutable build snapshot. The process-local encoder
policy supports bounded `libx264` and `h264_nvenc` output with an eight-frame
queue; the shared Manim installation is not edited. Hardware encoding requires
the GPU lease and a fresh responsiveness check. The NVENC adapter is pinned to
the inspected writer bytes and records its in-memory transformation. Its
construction and actual hardware output have been checked. On one fixed
1920×1080 board, delivering and encoding 600 frames took 110.5 seconds with
NVENC and 165.9 seconds with libx264. Both outputs independently decoded to
600 frames and 20 seconds, and their decoded first frames retained the
mathematics and data labels. This comparison excludes scene construction;
it does not establish full-chapter throughput. Settings and limitations are
in `encoder_comparison_20260905.json`. Use `--encoder h264_nvenc` for an
admitted movie render, under the current disjoint GPU workload claim.

The September 6 native writer adds `--writer ffmpeg71`. It uses the exact
owner-installed FFmpeg 7.1 executable, with a hash check; its directory name
is `ffmpeg-7.1.1`, but the binary identifies as `7.1-essentials`. Frozen waits
send one source frame to a native loop, while animations send every frame.
Explicit constant-frame-rate output and FFmpeg concatenation retain exact
frame counts and decode timestamps. A complete narrated-board check produced
2,190 frames and exactly 73 seconds, including all three recordings and all
15 animations; its three decoded end frames were inspected at full resolution.
Subsequent waveform comparison caught a 21 ms delay and a 3 dB level change
in Manim's intermediate AAC conversion. The native writer now encodes the
original mono WAV directly into MP4. The repaired scene passes whole-recording
alignment checks with zero sample offset; the repaired closing chapter also
passes all 18 recordings and measures 13,246 frames / 441.533333 seconds.
`verify_audio_alignment.py` checks every source waveform against the muxed
audio. These checks do not certify pronunciation or listening quality.
Partial movies, logs and receipts are retained without Manim cache eviction.
`--cpus 14,15` selects
two allowed background CPUs; choose only within the owner's current partition
and recheck pressure. The shared Manim environment and owner guard are unchanged.

Scientific assets retain their input hashes, case
selection rules, units and reconstruction checks. OCO-2 spectra are benchmark
simulator reconstructions; mechanics images and calibration deciles come from
saved predictions, including unfavorable cases.

`tex_rules.py` converts the PDF converter's stroke-only straight rules into
filled vector outlines before Manim imports them. It handles both fresh
conversions and copied cache hits. Original frozen cache assets are preserved;
conversion receipts identify the before/after SVG hashes. Unsupported stroke
geometry stops the render. `NotationProbe` supplies a native-resolution check
of the affected notation classes before any corrected chapter is released.
