"""4-fold out-of-fold MLP predictions on the 19000-sample train pool, plus a
full-pool MLP prediction on val and test. The OOF train field and the full-pool
val/test fields become an honest (leakage-free) input channel for a refiner.

Saves: mlp_oof_train.npy, mlp_full_pred_val.npy, mlp_full_pred_test.npy
CPU only. Reuses the residual-MLP architecture and metric loss.

Usage: python mlp_oof.py [--epochs 100] [--threads 4]
"""
import argparse, time, sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import load_arrays, canonical_split, rel_l2, RUNS

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--width", type=int, default=1024)
p.add_argument("--depth", type=int, default=4)
p.add_argument("--batch", type=int, default=256)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--threads", type=int, default=4)
args = p.parse_args()
if args.threads > 0: torch.set_num_threads(args.threads)
torch.manual_seed(0); np.random.seed(0)
dev = torch.device("cpu")

loads, stress = load_arrays()
tr, va, te = canonical_split(n_val=1000, seed=0)
idx2d = torch.arange(1681).reshape(41, 41); MIR = idx2d.flip(0).reshape(-1)

class MLP(nn.Module):
    def __init__(s, w, d):
        super().__init__(); s.inp = nn.Linear(41, w)
        s.hid = nn.ModuleList([nn.Linear(w, w) for _ in range(d - 1)]); s.out = nn.Linear(w, 1681)
    def forward(s, x):
        h = F.silu(s.inp(x))
        for l in s.hid: h = h + F.silu(l(h))
        return s.out(h)

def train_predict(fit_idx, pred_idxs, epochs):
    X = torch.from_numpy(loads[fit_idx]); Y = torch.from_numpy(stress[fit_idx]).reshape(len(fit_idx), -1)
    mu_x, sd_x = X.mean(), X.std(); mu_y = Y.mean(0, keepdim=True); sd_y = Y.std()
    model = MLP(args.width, args.depth)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    N = len(fit_idx)
    for ep in range(epochs):
        perm = torch.randperm(N)
        for k in range(0, N, args.batch):
            i = perm[k:k+args.batch]; xb = X[i]; yb = Y[i]
            if torch.rand(()) < 0.5: xb = torch.flip(xb, [1]); yb = yb[:, MIR]
            pred = model((xb - mu_x) / sd_x) * sd_y + mu_y
            loss = (torch.linalg.vector_norm(pred - yb, dim=1) / torch.linalg.vector_norm(yb, dim=1)).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sched.step()
    model.eval()
    outs = []
    with torch.no_grad():
        for idx in pred_idxs:
            Xp = torch.from_numpy(loads[idx])
            pr = model((Xp - mu_x) / sd_x) * sd_y + mu_y
            pr2 = model((torch.flip(Xp, [1]) - mu_x) / sd_x) * sd_y + mu_y
            outs.append((0.5 * (pr + pr2[:, MIR])).numpy().astype(np.float32))
    return outs

t0 = time.time()
# OOF on train
folds = np.array_split(np.random.default_rng(0).permutation(len(tr)), 4)
oof = np.empty((len(tr), 1681), np.float32)
for f, hold in enumerate(folds):
    fit = np.setdiff1d(np.arange(len(tr)), hold)
    (pred_hold,) = train_predict(tr[fit], [tr[hold]], args.epochs)
    oof[hold] = pred_hold
    print(f"fold {f}: oof rel-L2 {rel_l2(pred_hold.astype(np.float64), stress[tr[hold]].reshape(len(hold),-1).astype(np.float64)):.4f} [{time.time()-t0:.0f}s]", flush=True)
np.save(RUNS / "mlp_oof_train.npy", oof)
# full-pool model for val/test
pv, pe = train_predict(tr, [va, te], args.epochs)
np.save(RUNS / "mlp_full_pred_val.npy", pv)
np.save(RUNS / "mlp_full_pred_test.npy", pe)
print(f"full-pool: val {rel_l2(pv.astype(np.float64), stress[va].reshape(len(va),-1).astype(np.float64)):.4f}  "
      f"test {rel_l2(pe.astype(np.float64), stress[te].reshape(len(te),-1).astype(np.float64)):.4f} [{time.time()-t0:.0f}s]", flush=True)
print("saved OOF+full MLP fields", flush=True)
