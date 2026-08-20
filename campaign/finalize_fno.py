"""Finalise the FNO member from its campaign checkpoint, at every seed on this box.

train_fno.py writes `_ckpt_{tag}_s{seed}.pt` (the best-validation state_dict) every time
validation improves, and only at the very END of training does it load that state back,
score it, and save the named artifact plus its run JSON. The campaign's 36-hour wall clock
landed inside the FNO stage on every seed, so the checkpoints exist and the named
artifacts do not. This performs exactly the tail of train_fno.py: load the checkpoint,
evaluate, write `fno_s{seed}_w64_m14_L4_mir.{pt,json}`.

Nothing is retrained. The reported error is whatever that checkpoint is worth, which is
the honest thing to report and is why the number is measured here rather than assumed.

The model class used is models.FNO2d -- the one gen_preds.py will load the artifact with
-- so any mismatch between the trainer's inline class and the shared one surfaces now,
on load_state_dict(strict=True), rather than silently later.

Config is the campaign's: width 64, modes 14, layers 4, mirror on, 200 epochs
(campaign/seed_pipeline.py:117 and its EP dict).
"""
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = os.environ.get("NMKC_ROOT", "/srv/aiwork/nmkc10seed")
sys.path.insert(0, ROOT + "/code")
os.environ.setdefault("NMKC_DATA", ROOT + "/data/structmech")

WIDTH, MODES, LAYERS, EPOCHS, TAG = 64, 14, 4, 200, "fno"
G = 41

from common import load_arrays, canonical_split, rel_l2  # noqa: E402
import models as M  # noqa: E402

loads, stress = load_arrays()
torch.set_num_threads(int(os.environ.get("NMKC_THREADS", "4")))

for sd in sorted(glob.glob(os.path.join(ROOT, "seeds", "sm_s*"))):
    seed = os.path.basename(sd).replace("sm_s", "")
    if seed == "99":
        continue
    R = os.path.join(sd, "runs")
    ckpt = os.path.join(R, "_ckpt_%s_s%s.pt" % (TAG, seed))
    name = "%s_s%s_w%d_m%d_L%d_mir" % (TAG, seed, WIDTH, MODES, LAYERS)
    if not os.path.exists(ckpt):
        print("s%-2s no checkpoint" % seed, flush=True)
        continue
    if os.path.exists(os.path.join(R, name + ".json")):
        print("s%-2s already finalised" % seed, flush=True)
        continue

    os.environ["NMKC_SPLIT_SEED"] = seed
    tr, va, te = canonical_split(n_val=1000, seed=int(seed))

    # normalisation exactly as train_fno.py forms it, from the training split
    Xtr = torch.from_numpy(loads[tr])
    Ytr = torch.from_numpy(stress[tr])
    mu_y = Ytr.mean(0, keepdim=True)
    sd_y = Ytr.std().item()
    mu_x = Xtr.mean().item()
    sd_x = Xtr.std().item()

    lin = torch.linspace(0, 1, G)
    XX, YY = torch.meshgrid(lin, lin, indexing="ij")
    coords = torch.stack([XX, YY])[None]

    def make_input(xload):
        n = xload.shape[0]
        f2d = ((xload - mu_x) / sd_x)[:, :, None].expand(n, G, G)
        return torch.cat([f2d[:, None], coords.expand(n, 2, G, G)], 1)

    model = M.FNO2d(WIDTH, MODES, LAYERS)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)          # strict: a shape mismatch must raise here
    model.eval()
    nparam = sum(p.numel() for p in model.parameters())

    @torch.no_grad()
    def evaluate(idx, mirror_tta=False, bs=512):
        X = torch.from_numpy(loads[idx])
        Y = stress[idx]
        outs = []
        for k in range(0, X.shape[0], bs):
            xb = X[k:k + bs]
            pr = model(make_input(xb)) * sd_y + mu_y
            if mirror_tta:
                pr2 = model(make_input(torch.flip(xb, dims=[1]))) * sd_y + mu_y
                pr = 0.5 * (pr + torch.flip(pr2, dims=[1]))
            outs.append(pr.float().reshape(len(xb), -1))
        return rel_l2(torch.cat(outs).numpy(), Y.reshape(len(idx), -1))

    e_va = evaluate(va)
    e_te = evaluate(te)
    e_te_tta = evaluate(te, mirror_tta=True)

    torch.save(model.state_dict(), os.path.join(R, name + ".pt"))
    rec = dict(kind="fno",
               args=dict(seed=int(seed), epochs=EPOCHS, width=WIDTH, modes=MODES,
                         layers=LAYERS, mirror=1, final=0, tag=TAG, ntrain=0, lowval=0,
                         kf=0.0, batch=128, lr=1.5e-3, wd=1e-6),
               params=nparam, val=e_va, test=e_te, test_mirror_tta=e_te_tta,
               best_ep=None,
               note="finalised from the campaign checkpoint; training was cut by the "
                    "36h wall clock before train_fno.py reached its own save step")
    with open(os.path.join(R, name + ".json"), "w") as f:
        json.dump(rec, f, indent=1)
    print("s%-2s val %.5f  test %.5f  test+TTA %.5f  params %d  -> %s"
          % (seed, e_va, e_te, e_te_tta, nparam, name), flush=True)
