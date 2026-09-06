"""Write runs/control/queue.txt: the three GPU studies interleaved so every
axis reports early. Each line: `<script> <args>` relative to runs/.

  structmech  train_mlp_scale.py  widths 1024/2048/4096, depths 4/5/6, metric
              and MSE losses, seeds 0-4, one bf16 pair on the paper configuration
  oco2        jpl_scale.py        O2 band, widths 384/768/1536, depths 4/5/6,
              seeds 0-4; the exact feature-kernel head (--head gpu) on the
              paper configuration and the widest one
  kernel      lk_scale.py         kernel-flows metrics at the paper size and at
              2x rows / 4x steps / 2x batch / 2x and 4x rank, both spaces, seeds 0-2
"""
from pathlib import Path

CTRL = Path("C:/Users/owner/GOAL20H_20260906/nmkc_gpu_experiments/runs/control")
CTRL.mkdir(parents=True, exist_ok=True)
core = [(1024, 4), (2048, 4), (4096, 4), (1024, 5), (1024, 6)]
extra = [(2048, 6), (4096, 6), (2048, 5)]
jcore = [(384, 4), (768, 4), (1536, 4), (384, 5), (384, 6)]
lines = []
def sm(w, d, loss, s, amp=None):
    l = f"train_mlp_scale.py --width {w} --depth {d} --loss {loss} --seed {s}"
    lines.append(l + (f" --amp {amp}" if amp else ""))
def jp(w, d, s, head=None, band="o2"):
    lines.append(f"jpl_scale.py --band {band} --width {w} --depth {d} --seed {s}" + (f" --head {head}" if head else ""))
def lk(space, ntr, steps, bs, rank, s):
    lines.append(f"lk_scale.py --space {space} --ntr {ntr} --steps {steps} --bs {bs} --rank {rank} --seed {s}")

# block 1: seed 0 of every axis
for w, d in core:
    sm(w, d, "metric", 0); sm(w, d, "mse", 0)
for w, d in jcore:
    jp(w, d, 0)
lk("state", 6000, 400, 256, 6, 0); lk("features", 6000, 400, 256, 6, 0)
# block 2: remaining seeds of the core, structmech then oco2
for s in range(1, 5):
    for w, d in core:
        sm(w, d, "metric", s)
    for w, d in core:
        sm(w, d, "mse", s)
    for w, d in jcore:
        jp(w, d, s)
# block 3: kernel search at larger sizes (seeds 0-2), then the heads
for s in range(3):
    lk("state", 12000, 400, 256, 6, s); lk("state", 6000, 1600, 256, 6, s); lk("state", 6000, 400, 512, 6, s)
    lk("state", 6000, 400, 256, 12, s); lk("state", 6000, 400, 256, 24, s)
    lk("features", 12000, 400, 256, 6, s); lk("features", 6000, 1600, 256, 6, s); lk("features", 6000, 400, 256, 24, s)
    if s > 0:
        lk("state", 6000, 400, 256, 6, s); lk("features", 6000, 400, 256, 6, s)
for s in range(5):
    jp(384, 4, s, head="gpu"); jp(1536, 4, s, head="gpu")
# block 4: bf16 pair, then the deep-and-wide extras
for s in range(5):
    sm(1024, 4, "metric", s, amp="bf16")
for s in range(5):
    for w, d in extra:
        sm(w, d, "metric", s)
for s in range(5):
    for w, d in extra:
        sm(w, d, "mse", s)
(CTRL / "queue.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(len(lines), "jobs queued")
