# Paper 1 and lecture: restart checkpoint, 5 September 2026

## Current checkpoint: September 6, 01:43 Jerusalem

The two votes are still 01 YES and 06 YES. Reviewers 02 and 07 continue in
their existing controllers; no duplicate panel has been launched. Root began
integrating the two returned reviews into the WORKING source, while the
immutable packet stays unchanged. After all ten original reviews, freeze one
corrected packet and obtain independent revision assessments on that packet.

Working manuscript changes: Table 4 now correctly says standalone KRR has
no reflection augmentation or prediction averaging; the test-block paragraph
distinguishes predictor fitting from hindsight optimization/calibration.
Figure S6.3's internal title now says Lemma S5.1. Its original data arrays
were not found in the bounded local/DGX search, so this was an editorial
vector-PDF title repair, not recomputation. At 288 dpi, zero pixels outside
the title changed, and every other text span is unchanged. The corrected
figure was inspected at native resolution. Original PDF is preserved at
.local-verification/spectra_reference_0140/spectra_original.pdf; the public
repair helper and hash receipt document the exact change. Corrected figure
SHA 570cb9121c0cf46e6d840fb8120fdc21e9154e62764e7ed1379774b612f0452d.

seed_pipeline.py now explicitly defaults to historical pooled centering,
passes that argument, separates fold-local directories, and verifies a
configuration/code/data contract before reusing outputs. It checks the OOF
mode, split, producer hash and actual field hash before refiner training.
A per-seed lock serializes concurrent launches. Six bounded regression tests
pass, including altered modes/settings/source/data, unreceipted historical
outputs, a truncated contract and altered OOF provenance. All 17 named source
dependencies exist. No revised training run has been launched or deployed.
The changed TeX and figure still need a fresh paired build and page QA.

Chapter 01 completed (return 0, 710.632 s render). All 316 input hashes,
18 audio waveforms and exact media timing pass: 15,050 frames, 501.666667 s,
480.936 spoken seconds, video SHA
d06b68dbe4dc430db7e90c5e04886c26b8023b2f80627fef9699b377025dba26.
All 18 decoded end frames were inspected at full resolution; no defect found.
The visual receipt is in its author_check directory. Transition/pacing and
auditory review remain pending. Chapter 02 is currently rendering in
20260906T012559_chapter02_final, launcher PID 14788, actual Manim PID 52956,
logs .local-verification/chapter02_0126_*. It has passed animation 80 of 105.
Root must continue chapters 03 through 11 and rerender corrected chapter 12.

Latest verified push before this update: 4042035d8edd7d9fce4da17f48fd73c6bd044cb6.
Root mesh pulse at 01:41:33 renewed claims through 02:26:33 Jerusalem.

## Historical checkpoint: September 6, 01:18 Jerusalem

The overnight order continues through 12:55. Current independent votes are
reviewer 01 YES and reviewer 06 YES, with no blocking findings. ROOT read both
reports and independently verified every cited snapshot hash (10 for 01,
8 for 06). Reviewer 06 recommends correcting the Table 4 KRR symmetrization
caption, making centering explicit in seed_pipeline.py with safe resume
provenance, and correcting the old embedded lemma label. Formal registration
and human review remain PENDING. Do not change the frozen review packet.

Reviewer 02 continues under controller 33120/native 41212. Reviewer 07 is
active under controller 43484/native 55924, runtime 07_10_after06. Both last
reported REVIEWING after recovering from pressure holds. They will advance
through 05 and 10 respectively. Native 44992 for reviewer 06 was still alive
at 01:12 after writing its final report; check cleanup before assuming gone.

Full closing render 20260906T004953_chapter12_final completed with return 0
in 401.604 seconds. Its original MP4 used Manim's faulty intermediate AAC:
21.333 ms delay and approximately -3 dB level. The preserved original video
was copied with its original mono WAV directly into a new MP4 at
lecture/out/chapter12_audio_repair_0110/chapter12_final.mp4, SHA
2f398bc7e45eac1c3fa0b9ad72f095103bca37d815c0359755b4e1315efa29b0.
It has exactly 13,246 frames, 441.533333 seconds, 404.616 spoken seconds.
All 315 frozen inputs and all 18 waveform alignments pass. All 18 decoded
end-of-segment frames were inspected at native resolution. One small visual
defect remains in this movie: Covered crosses the prediction-ball boundary.
The current scene source moves it safely inward; rerender chapter 12 after
this fix before final delivery. Auditory review has not been performed.

Manim's default cache limit deleted 170 temporary partial artifacts from
that completed build. The final media, source manifest, WAV, stills and timing
survive. Current scenes.py disables that eviction to retain every subsequent
partial movie, log and receipt. Do not claim the deleted diagnostics survive.

Chapter 01 is now rendering in 20260906T011010_chapter01_final, launcher
33356 (birth 1788646205.311545), logs .local-verification/chapter01_0110_*.
It uses the new native audio combiner and no-cache-eviction setting, and
actual Manim runs on allowed CPUs 14,15. It does not contain the later
Covered-label position edit, which does not affect chapter 01. Verify this
first complete use of the new combine_to_movie override before continuing.
The other ten opening chapters still need production renders and all QA.

Latest verified remote commit before this update:
1288342b79feb833a12f6c0b38794c728fc9c70a on the completion branch.
Mesh pulse 01:07:56 renewed root claims through 01:52:56 Jerusalem.

## Historical checkpoint: September 6, 00:50 Jerusalem

Continue through 12:55 Jerusalem. The paper and full two-hour lecture are not
finished. One current independent vote has returned: reviewer 01, YES with
two nonblocking editorial observations. Its full report and evidence are in
the canonical reviewer01 leaves. Its ten cited snapshot hashes were checked
again by the parent. The reviewer finished, validated its report, and released
its own claims. Formal Brain registration and human review remain PENDING.

The two observations are an internal figure title in figures.py:89 still
saying Lemma 4.4, and the overbroad optimizer sentence at paper/impl.tex:82.
Do not edit the frozen packet. Integrate all panel findings together, then
have the independent reviewers assess the new pinned revision as needed.

Reviewer 02 is now active under controller PID 33120, native PID 41212,
runtime `02_05_after01`. This controller will run 02 through 05. The previous
01 controller exited on a mesh-claim timeout without killing its reader;
do not restart it. Reviewer 06 is still the detached native PID 44992 in
thread 01a07326-223d-7c03-9897-101e289417ca, runtime `06_10_resume3/reviewer06`.
It is preparing its report. ROOT still must launch 07 through 10 after 06
finishes, because that old controller was detached. No other votes yet.

The corrected narrated-board integration test is COMPLETE:
`lecture/builds/20260906T004304_c12_layers`, return 0, 96.2138 seconds of
render time. Its output is 2190 frames / exactly 73 seconds at 1920x1080,
30 fps. All three recordings fit their boundaries, all 315 frozen inputs
match, and all three decoded end-of-segment frames were inspected at native
resolution. Movie SHA c4427faa5001c2963f1c04e07ed39aee0085e11e0185d61a5db38378a4a1af78.
Audio has not been listened to; waveform/content alignment is the next
independent media check. No positive console-flash certification is claimed.

Use the tested native writer with `--encoder h264_nvenc --writer ffmpeg71
--cpus 14,15`. The required binary is in the owner's ffmpeg-7.1.1 directory
but identifies as 7.1-essentials, SHA
2ce797a0f88d7f067180338fb227f7b1928ea727bd9a4d7a1d022f7c52af71a3.
The process-local adapter sends each still once and repeats it natively;
animations retain every rendered frame. Constant-frame-rate output fixes a
one-frame duration error. FFmpeg concatenation preserves H.264 decode times;
the generic Manim combiner rejected these packets after discarding DTS.
The failed 002317 board render and 0034 pre-launch database failure are
preserved. Do not count either as completed media.

Core pressure explained much of the earlier slowness: at 00:28, background
cores 4 through 9 were at 100%, while allowed background 14 and 15 were idle.
Only this render's birth-verified processes were moved. The next launch pins
its helpers to the requested background CPUs before its first mesh call.
The owner guard remains unchanged. Recheck current pressure before each job.

Full chapter 12 started at 00:49:33, parent PID 56356, logs
`.local-verification/chapter12_final0049_*`. It uses the successful board's
complete TeX cache. Inspect its exact status before any restart. This is the
only owned production render currently launched. After verification, produce
the other eleven chapters through the same pipeline and assemble/measure the
complete film. All 252 recordings and all 252 equation preflights remain valid.

Latest verified remote commit before this update is
4c51c5da8b73f93e66e465cd260db9c417641a51 on the completion branch.

## Historical checkpoint: 23:58 Jerusalem

The overnight order still runs through September 6 at 12:55 Jerusalem. No
independent referee has returned a verdict. Scientific source and frozen packet
remain unchanged. Build 06 also matches build 05's extracted text on all 85 pages.

Referee 01 is running in native thread 01a07306-8a1b-7742-93bb-01f73cdb2df2,
resumed under controller PID 8212 (23:36:50), native PID 46060. Its controller
runtime ends in `01_05_resume2`. The first controller failed on a transient
PermissionError reading compute_state.json; this was not measured CPU distress.
The fixed monitor now holds or pauses safely without discarding native progress.

Referee 06 is running in native thread 01a07326-223d-7c03-9897-101e289417ca,
native PID 44992, birth 1788640004.1754563. Its controller 52288 was deliberately
detached, leaving the native reader alive. ROOT MUST MONITOR this reader directly.
Its output remains under runtime `06_10_resume3/reviewer06`. When it finishes,
validate its actual verdict and packet hash, then launch the fixed controller
for reviewers 07 through 10. Do not launch a duplicate 06. The 01 controller
will handle 02 through 05 after 01. Earlier failed attempts are not votes.

Chapter 12 now has a successful 18-frame sample build:
`lecture/builds/20260905T232406_chapter12_samples`, return 0, 1070.9631 seconds.
Seven of its 18 frames have been inspected at native resolution so far. Its
calibration segment still uses the old pipeline picture; the current source
adds a transition to a calibrated output-space ball, awaiting its own render.
BoardSamples now accepts the existing --board selection for bounded rechecks.
No corrected full chapter movie exists yet.

All 252 equations pass the revised preflight (130-point width, 65-point height).
Native layout uses fixed readable math size and reflows long expressions; it
does not shrink them to fit. The prior 170-point width threshold missed a
5.030-unit overflow at the actual font size. Receipts retain that failed build.
All 252 audio files passed independent RIFF/PCM checks again at 23:49, with
source-bound manifest refreshed after equation-only edits. Spoken duration
remains exactly 6718.992 seconds. The approximately 120-minute assembled
duration remains a prediction until the movie is measured.

Latest verified remote commit before this checkpoint is
08c0eac0ab05fc4f6ed42856203023bd4a88c423. The NVENC benchmark used the actual
PyAV writer; do not describe it as the FFmpeg 7.1 binary. A direct 7.1 writer
probe and final assembly through that exact executable are still required.

## Historical checkpoint: 22:58 Jerusalem

The alternate supplement-root build has now finished successfully: receipt
`final-build-06-supp/completion_receipt.json`, return 0, selected supplement,
same source manifest e20ecd74e50a6f22f41d33d27c028bbc3b331d648a25b0faea58abe547419a0a.
Its output is the 51-page supplement, with the 34-page article also present;
no undefined references or overfull boxes in the three final logs.

Referee 06's FIRST attempt is FAILED, not active: the controller incorrectly
classified a 17.8-second health measurement as actual distress and terminated
the CLI before it emitted any review work (only thread.started/turn.started).
The existing old status.json incorrectly still says REVIEWING. Preserve that
evidence; do not count a vote. The distinction is now fixed: stale/slow sampling
holds new admission, but alone does not classify an existing light reader as
distressed. A 20-second delayed-sample regression control checks this behavior.
A fresh 06–10 controller was launched at 22:56:18, PID 46976, with run-tag
`retry2`, logs `panel06_10_retry2_*`. Referee 01 continues under the first batch.

The 22:47 sample render was also refused for slow sampling; it never drew a
board. A bounded admission loop now resamples for at most ten minutes while
retaining the same headroom limits. New sample attempt: 22:56:17, PID 23324,
logs `chapter12_samples_retry3_*`. Check the live process and receipt before
calling it active or complete. No corrected full chapter movie exists yet.

All audio also passed a separate RIFF/PCM byte-count path, without reusing the
recording contract's WAV parser. `lecture/narration_manifest.json` binds all
252 segments to chapter source hashes, audio/receipt hashes and exact frames.
Total remains 6718.992 seconds (839874/125). Planned transitions and holds add
500.4 seconds; 7219.392 seconds is a prediction, not assembled film duration.

### Earlier recovery detail, 22:48

Continue through September 6 at 12:55 Jerusalem. No reviewer vote has returned.
The immutable packet and source identifiers in the older checkpoint below
remain current. Referee 01 is actually running in the 01–05 controller
(parent 51324, worker 57100); its exact-thread wake completed. The second
controller, 06–10, PID 52004 / birth 1788636698.640, is still verifying the
packet before launching a worker. Do not duplicate either batch. Both use
fresh Codex CLI contexts, with at most one worker active per controller.

Disk pressure has recovered: Windows Search is stopped/disabled, and C: has
about 214 GiB free. No files were deleted by this NMKC agent. The governor is
fresh again, warning about remote reachability and sometimes process pressure.
`lecture/compute_admission.py` supplements the shared state with current CPU,
PSAPI physical/commit memory, visible-window hung status and disk readings.
It preserves a shared distress/throttle veto and all previous numerical
limits. Failure controls passed; the first PSAPI commit limit also agreed
exactly with Windows performance counters. This is not a global guard repair.
The render child still gets verified CPUs 4–5 and BelowNormal before resume.

Chapter 12 narration is COMPLETE: 18 segments, 404.616 seconds. All twelve
chapters total 252 segments / 6718.992 spoken seconds by text-bound WAV checks.
The first chapter 12 sample attempt crashed with 0xc0000005 before producing
a board. Windows Application event 1000 identifies OLEAUT32.dll, PID 52424;
there was no Display 4101 in the queried interval. Cause is NOT yet established.
Its immutable failed build is `lecture/builds/20260905T223334_chapter12_samples`.
The first retry was correctly refused when live sampling took 18.9 seconds.
A second explicit retry started 22:47:09, PID 43572, with Python faulthandler
and unbuffered logs. Inspect `.local-verification/chapter12_samples_retry2_*`
and `lecture/logs/chapter12_samples_render.log` before any further attempt.

The interrupted alternate build 06 was ended by birth-verified exact process
claims. The first cleanup stopped four processes then hit a mesh assert timeout;
the second ended the remaining 46280/51312/51776 and wrote the actual
`final-build-06-supp/author_stop_receipt.json`. No other TeX tree was touched.
Recovery through `resume_pair.py final-build-06-supp supplement` started
22:47:10, PID 43576. It rechecks the source hashes and current health, then
uses a fresh 1800-second wait. Inspect `build06_resume_*` and the build receipt.
Do not resume the dead original parent 28116 or launch a duplicate.

Latest verified remote commit before the next checkpoint push remains
`095affece5eeb70288225a1d4a6c09d1dab389a1`. The live goal API still carries an
old BLOCKED value; actual work is continuing, and the overnight task is not done.

## Historical checkpoint: 22:12 Jerusalem, after the reboot

The overnight order continues through 6 September at 12:55 Jerusalem. Do not
end the task merely because one front is complete. The goal API currently
reports an old BLOCKED state; actual work has continued under the owner's
resumed instruction. That API state is not a completed task.

Latest verified push: `095affece5eeb70288225a1d4a6c09d1dab389a1` on the
completion branch. All Caltech work described below is complete. Do not
restart its queue. The current paired build is
`.local-verification/final-build-05`, with successful `completion_receipt.json`:
34 article pages and 51 supplement pages. All changed pages were inspected
at 144 dpi after pixel comparison with the prior build. Main page 14 and
supplement pages 44–51 changed. The final logs have no unresolved references
or overfull boxes; ordinary class/package warnings remain.

The independent-review target is now COMPLETE and IMMUTABLE under the Brain
audit home at `evidence/nmkc_publication_20260905/final_panel/`:
`packet_97554dd4b58e_final-build-05_v2`. It contains 1,103 payload files,
84,123,589 bytes, and MANIFEST.json SHA-256
`42f48db7bd004625ee4d5d2a42dd7cc70332a12effc9019c6d5f2195358572fe`.
Its scientific source commit is `97554dd4b58ece92675efd1f6faebdbce75d5e80`;
subsequent commits concern the excluded lecture. Previous votes are excluded.
Do not rerun the packet assembler on this completed directory. PDF extracted
text contains some mathematical control characters; use the PDF and TeX for
formula checking. Windows universal-newline conversion caused a false text
mismatch during assembly; exact byte comparison resolved it.

NO independent referee vote has returned. A hidden BelowNormal controller for
reviewers 01–05 was launched at 22:10:49, PID 51324, birth 1788635449.006.
It verifies the packet, then waits for fresh healthy compute readings before
starting a Codex CLI worker. Logs: `.local-verification/panel01_05_*`; runtime
under the final-panel evidence leaf. Inspect it before any restart. Its health
admission wait is bounded to one hour. Reviewers 06–10 are not launched.
Do not count an admission controller as a reviewer or a completed vote.

Current pressure is DISK-ONLY DISTRESS: free C: space fell from about 86 GiB at
21:03 to 23–28 GiB around 22:00. The entire frozen paper packet is only 84 MB.
Guard and restore owners are notified; no other agent's processes or files
were changed. `.local-verification/disk_pause_receipt.json` records the exact
birth-verified processes suspended. Roots 28116 (alternate supplement-root
build) and 33252 (chapter-12 narration) were independently confirmed STOPPED
around 22:00. Resume their recorded live identities after pressure recovery;
if a process died, use its existing validated outputs and bounded launcher.
The alternate build lives in `final-build-06-supp`; its wait timeout includes
wall time, so inspect for timeout/remaining children before resuming it.
The interactive CPU guard heartbeat was stale at 21:09 despite supervisor
restart attempts. Do not call stale telemetry healthy.

Lecture chapter 12 now has six boards and 18 segments, using checked centering
and grid figures. Its 18 equations compile after one line was reflowed.
Twelve closing narration segments were recorded before suspension; validate
actual WAV receipts before counting them. Use C:/Python314/pythonw.exe for
narrate.py: edge_tts is installed in its user site, not the Manim venv. The
existing eleven chapters retain 234 recordings. Full corrected lecture
rendering and audiovisual review remain unfinished.

GPU convention changed by the owner's native 19:38:49 prompt: concurrent useful
GPU work is allowed, with disjoint workload claims. This supersedes the old
card-wide exclusive convention lower in this historical note. NMKC uses
`topic:gpu-workload/MATH-ROSS20/codex-nmkc-resume-20260905`. Preserve CPU masks,
BelowNormal, fresh pressure checks and VRAM limits. The 600-frame real-writer
probe is complete: 110.4747 s NVENC versus 165.8925 s CPU (ratio 1.50163),
with independent ffprobe counts of 600 frames / 20 s / 1920×1080 / 30 fps in
both. Both decoded first frames were inspected. Direct NVML sampling had no
errors; device-wide encoder peak was 95%, which includes concurrent work.
See `lecture/encoder_comparison_20260905.json` and the private `probe_pair_2120`
receipts. This is not a full-chapter speed or audiovisual approval claim.

An additional plain-math beta-binomial summation reproduced the supplement's
central 95% reference intervals: 16725–17449 of 19000 (88.0263–91.8368%) and
17772–18297 of 19000 (93.5368–96.3000%), for m=1000 and ranks 901/951. This is
an author recheck, not an independent panel vote or a validation of sampling
independence in the historical benchmark.

## Historical pre-reboot checkpoint

Saved at 16:32 Jerusalem for the owner's planned reboot. This is an interruption
checkpoint, not a completed-paper or publication verdict.

Work on branch `codex/paper1-completion-20260905`. Read the current project state,
live Brain, mesh digest, and compute policy; reacquire narrow claims before any
mutation. The private `.local-verification/checkpoint.md` preserves transport,
receipts, process identities, and the exact local session restoration record.

## Paper and simulations

**16:36 update:** all planned simulations and the autonomous finalization have
finished. All 18 raw grid scenarios passed. The 27,793,233-byte evidence archive
was downloaded locally; its SHA-256 is
`2cb72d8c84c800113406ceb0f5790000bef62f5c90bccf81cafdb09aed36026f`.
The local exact manifest and every one of its 454 files were verified before
extraction into `.local-verification/evidence-final`. The aggregate validation
completed successfully; its unchanged output is committed as
`campaign/sensitivity_summary_final.json`, SHA-256
`b17dd2b71e2e4f610364e892e668556501094aaf0afdc749f72a59ad41c4c5b0`.
Use that summary and evidence directory for table and figure generation. There
is no remaining simulation queue. Centering and mismatch changes are tiny;
the expanded grid increases the combined model's mean reduced-space error in
all three bands. Preserve these adverse results and the mixed radiance results.
The following 16:30 status is retained only as earlier context; do not relaunch
the campaign or finalizer.

The three main results have complete proofs in the manuscript and supplement.
The latest revision also spells out why evaluation vectors lie in the Gram
matrix range when that matrix is singular. The final independent referee panel
has not yet reviewed this revision.

At 16:30, all ten paired centering reruns and all ten mismatch checks were
complete. Eight of nine OCO grid experiments were complete; `grid_sco2_s2` was
running. Do not append a sweep or duplicate completed seeds.

The remote fixed campaign has an autonomous finalizer, PID 10477 at launch,
started at 16:26:19 Jerusalem. It survives this Windows reboot. Inspect
`finalization_status.json` and `followup_status.json` before taking action.
The finalizer is bounded to 90 minutes, one CPU at nice 19, and launches no
training or GPU work. It waits for the fixed campaign, independently checks raw
grid predictions, validates all 18 expected rows, and collects
`paper1_completed_evidence_20260905.zip`. Its code is
`campaign/finish_campaign.py`, SHA-256
`3a2ba3db4111254bf38ee402fa9c567bdf8e06bacc195f1819f5a84ef770716d`.
It was last verified in `WAITING_FOR_FIXED_CAMPAIGN`.

Never run a second grid checker or finalizer without inspecting its exact
process identity, status, logs and lock directories. A failed or stale lock is
an investigation, not permission to delete and relaunch it.

After `COMPLETE`, download the archive and status, verify the archive hash and
exact manifest, then run the aggregate and rendering tools on the collected
evidence. Integrate the generated macros, the centering/mismatch discussion,
`paper/oco_grid_sensitivity.tex`, and supplement section S11. Correct the old
sentences that say these effects were not measured. Keep the original published
table identity distinct from the new paired reruns; retain adverse grid results.

Build the main paper and supplement together using the existing reciprocal
external-document build helper. Check every PDF page, tables, links and
references. The old 32-page/46-page PDFs are not this final revision. Update the
README and reproducibility files, scan for secrets, commit and push.

Then obtain the owner's requested ten independent YES/NO submission reviews
against one pinned final packet, with reasons and concrete blockers for NO.
No reviewer has been launched for this final panel. Follow the current audit
skill, canonical audit storage and fresh per-agent compute admission. Do not
reuse old votes or count author checks as independent reviews. Resolve material
findings and record which exact revision each verdict concerns.

## Two-hour lecture and Brain

See `lecture/RESUME.md`. Chapters 1 through 11 have Microsoft
AndrewMultilingualNeural narration at +10%, about 105.2 minutes of spoken audio.
Chapter 12 must explain the final results, including failures. The completed
movie's duration, pauses and audiovisual quality still need measurement.

The chapter-1 sample renderer was accidentally terminated by another agent at
about 16:25; the agent acknowledged a broad process-filter error. Nine PNG frames
and complete TeX/SVG cache pairs survive in
`lecture/builds/20260905T153205_chapter01_samples`. Eight frames were inspected at
native resolution; `c01_error03` still needs inspection. This build has no
completed-board manifest. The explicit incomplete-cache recovery flag reuses
validated typesetting, not a completed render or approval.

Use Latin Modern mathematics, fixed readable type size with equation reflow,
and the tested SVG rule repair. Old preview movies are superseded for missing
mathematical rules. The native notation probe and its removal controls passed
author checks, but the formal independent notation audit is not closed.

The process-local NVENC adapter and actual-writer benchmark are ready; this lane
has not opened the GPU encoder. After reboot, inspect the proposed GPU verifier,
check the actual runtime/device, acquire the exclusive GPU claim, and test the
small writer probe before choosing an encoder. Never infer readiness from a
different FFmpeg executable or change the shared environment speculatively.

The latest Brain attachment records picture-first proof explanations, geometry
on every board, actual data with honest axes, Microsoft narration, typography,
and measured rendering. Its latest-100 ledger contains 100 verified submitted
texts, but one historical 51-line pasted attachment is unavailable; do not claim
100 complete prompt bundles. The private video-Brain coordinator has been asked
to preserve the current recipe attachment in its repository snapshot.

Finish and verify the paper first, then complete chapter 12 and the full lecture
using the paper's final pinned evidence. The owner requested both theory and
numerics, complete main proofs, a human academic voice, and a two-hour lecture
built around pictures and geometry.
