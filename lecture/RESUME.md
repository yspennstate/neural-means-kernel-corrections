# Resume the NMKC lecture after an interruption

Do not start a second renderer while the old one is alive. Read the live Brain,
mesh digest and compute policy first. Reacquire the project and any GPU lease;
old process IDs and pre-restart health readings are not current authorization.

The current corrected chapter-1 sample build is
`builds/20260905T153205_chapter01_samples`. Its inputs are frozen. The last
source checkpoint is on branch `codex/paper1-completion-20260905`.
The old 8m23 preview is superseded for missing mathematical rules.

Completed TeX/SVG pairs survive interruption. After confirming the old renderer
has exited, run one hidden, BelowNormal launcher using Python314:

```text
lecture/render_one.py --chapter 01 --quality samples --reuse-tex-from lecture/builds/20260905T153205_chapter01_samples --reuse-incomplete-tex
```

The recovery option validates the source-derived TeX cache keys, complete SVG
XML, and stable bytes while copying. It reuses typesetting only. It does not
declare the interrupted chapter complete, reuse its timing as finished, or
count its uninspected frames as reviewed. New or changed TeX compiles normally.
Inspect every frame from the resulting complete sample build at native size.
Then use that completed build as the cache donor for the narrated render.

The audio in `audio/01` through `audio/11` is already recorded with Microsoft
AndrewMultilingualNeural at +10%. Identity checks reject changed narration or
voice settings. Chapter 12 awaits the final sensitivity results. Never silently
substitute a different narrator.

The NVENC adapter has passed construction checks, but this lane has not yet
opened the encoder. Before a GPU movie, verify the exact installed runtime and
current device after the restart, acquire `topic:gpu/MATH-ROSS20`, and run the
small actual-writer probe. Do not infer CUDA/NVENC readiness from driver
installation or another FFmpeg executable. Record the actual encoder, PID,
device, utilization and output; inspect the decoded mathematical frame.

Caltech simulations run independently of this Windows session. Their fixed
campaign is `/home/yitz/nmkc_paper1_20260905`; inspect its status before launching
anything. At 16:18 Jerusalem, only `grid_sco2_s2` remained. Never relaunch finished
seeds or append a sweep. The paper's final raw-prediction check, evidence
collection, table generation, paired PDF build and ten publication reviews
follow completion. See `.local-verification/checkpoint.md` for current details.
