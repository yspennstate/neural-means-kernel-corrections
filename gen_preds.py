"""Generate train/val/test predictions (with reflection TTA) for a saved model,
so the stack-and-correct step is architecture-agnostic. Handles the plain MLP,
the KRR-field refiner, FNO, the transformer operator (vit), and UNet. Uses the
GPU when available (chunked, guarded), otherwise CPU.

Saves runs/{tag}_predtr.npy, _predva.npy, _predte.npy  (denormalized fields).

Usage: python gen_preds.py --run fnoG_s0_w64_m14_L4_mir
"""
import argparse, json, sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import load_arrays, canonical_split, rel_l2, RUNS, DATA
import models as M

p = argparse.ArgumentParser()
p.add_argument("--run", type=str, required=True)
p.add_argument("--cpu", action="store_true")
p.add_argument("--bs", type=int, default=1024)
args = p.parse_args()
dev = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

cfg = json.load(open(RUNS / f"{args.run}.json"))
kind = cfg["kind"]; a = cfg["args"]
loads, stress = load_arrays()
tr, va, te = canonical_split(n_val=1000, seed=0)
# training-set convention identical to the trainer
t_pool = np.load(DATA / "idx_train.npy")
if a.get("ntrain", 0) > 0:
    tr = t_pool[:a["ntrain"]]
    if a.get("lowval", 0) > 0:
        va = tr[-a["lowval"]:]; tr = tr[:-a["lowval"]]
mu_x = float(loads[tr].mean()); sd_x = float(loads[tr].std())
mu_y = torch.from_numpy(stress[tr].reshape(len(tr), -1).mean(0, keepdims=True)).float().to(dev)
sd_y = float(stress[tr].reshape(len(tr), -1).std())
idx2d = torch.arange(1681, device=dev).reshape(41, 41); MIR = idx2d.flip(0).reshape(-1)

class RefMLP(nn.Module):
    def __init__(s, w, d):
        super().__init__(); s.inp = nn.Linear(41 + 1681, w)
        s.hid = nn.ModuleList([nn.Linear(w, w) for _ in range(d - 1)]); s.out = nn.Linear(w, 1681)
    def forward(s, x):
        h = F.silu(s.inp(x))
        for l in s.hid: h = h + F.silu(l(h))
        return s.out(h)

G = 41
lin = torch.linspace(0, 1, G, device=dev)
XXg, YYg = torch.meshgrid(lin, lin, indexing="ij")
coords = torch.stack([XXg, YYg])[None]           # (1,2,41,41)

if kind == "mlpR":
    model = RefMLP(a["width"], a["depth"])
elif kind == "fno":
    model = M.FNO2d(a["width"], a["modes"], a["layers"])
elif kind == "vit":
    model = M.OpFormer(a["dim"], a["depth"], a["heads"])
elif kind == "unet":
    model = M.UNet(a["width"])
else:
    model = M.MLP(a["width"], a["depth"])
model.load_state_dict(torch.load(RUNS / f"{args.run}.pt", map_location="cpu", weights_only=True))
model.to(dev).eval()

if kind == "mlpR":
    K_tr = np.load(RUNS / a.get("field_tr", "krr_oof_train.npy"))
    K_va = np.load(RUNS / a.get("field_va", "krr_full_matern52_n19000_pred_val.npy"))
    K_te = np.load(RUNS / a.get("field_te", "krr_full_matern52_n19000_pred_test.npy"))

def finp(xn):
    n = xn.shape[0]
    f2d = xn[:, :, None].expand(n, G, G)
    return torch.cat([f2d[:, None], coords.expand(n, 2, G, G)], 1)

@torch.no_grad()
def predict(idx, split):
    outs = []
    for k in range(0, len(idx), args.bs):
        sl = idx[k:k+args.bs]
        x = torch.from_numpy(loads[sl]).float().to(dev)
        xn = (x - mu_x) / sd_x
        if kind == "mlpR":
            kf = {"tr": K_tr, "va": K_va, "te": K_te}[split][k:k+args.bs]
            kf = torch.from_numpy(kf.reshape(len(sl), -1)).float().to(dev)
            inp = torch.cat([xn, (kf - mu_y) / sd_y], 1)
            inpm = torch.cat([torch.flip(xn, [1]), (kf[:, MIR] - mu_y) / sd_y], 1)
            pr = model(inp) * sd_y + mu_y
            pr2 = model(inpm) * sd_y + mu_y
            pr = 0.5 * (pr + pr2[:, MIR])
        elif kind in ("fno", "unet"):
            pr = (model(finp(xn)).reshape(len(sl), -1)) * sd_y + mu_y
            pr2 = (model(finp(torch.flip(xn, [1]))).reshape(len(sl), -1)) * sd_y + mu_y
            pr = 0.5 * (pr + pr2[:, MIR])
        else:                                   # mlp, vit: flat 1681 output
            pr = model(xn) * sd_y + mu_y
            pr2 = model(torch.flip(xn, [1])) * sd_y + mu_y
            pr = 0.5 * (pr + pr2[:, MIR])
        outs.append(pr.float().cpu().numpy())
    return np.concatenate(outs).astype(np.float32)

for split, idx in [("tr", tr), ("va", va), ("te", te)]:
    P = predict(idx, split)
    np.save(RUNS / f"{args.run}_pred{split}.npy", P)
    if split == "te":
        e = rel_l2(P.astype(np.float64), stress[idx].reshape(len(idx), -1).astype(np.float64))
        print(f"{args.run}: test rel-L2 (TTA) = {e:.4f}", flush=True)
print("saved preds for", args.run, flush=True)
