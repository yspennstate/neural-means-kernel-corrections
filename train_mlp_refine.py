"""MLP refiner: [load (41), KRR-predicted field (1681)] -> stress field.

CPU-friendly replacement for the FNO refiner: train-time KRR channel is
out-of-fold (krr_oof.py), val/test channels are the full-train KRR fit.
Mirror augmentation flips the load and permutes both fields consistently.

Usage: python train_mlp_refine.py [--seed 0] [--epochs 300] [--width 1024]
       [--depth 4] [--tag mlpR]
"""
import argparse, time, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import load_arrays, canonical_split, rel_l2, save_run, RUNS

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=300)
p.add_argument("--width", type=int, default=1024)
p.add_argument("--depth", type=int, default=4)
p.add_argument("--batch", type=int, default=256)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--wd", type=float, default=1e-5)
p.add_argument("--threads", type=int, default=0)
p.add_argument("--field_tr", type=str, default="krr_oof_train.npy")
p.add_argument("--field_va", type=str, default="krr_full_matern52_n19000_pred_val.npy")
p.add_argument("--field_te", type=str, default="krr_full_matern52_n19000_pred_test.npy")
p.add_argument("--tag", type=str, default="mlpR")
args = p.parse_args()

if args.threads > 0:
    torch.set_num_threads(args.threads)
torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", dev, flush=True)

loads, stress = load_arrays()
tr, va, te = canonical_split(n_val=1000, seed=0)
# field channel: OOF predictions on train (no leakage), full-model on val/test
K_tr = np.load(RUNS / args.field_tr)
K_va = np.load(RUNS / args.field_va)
K_te = np.load(RUNS / args.field_te)

Xtr = torch.from_numpy(loads[tr]).to(dev); Ytr = torch.from_numpy(stress[tr]).reshape(len(tr), -1).to(dev)
Xva = torch.from_numpy(loads[va]).to(dev); Yva = torch.from_numpy(stress[va]).reshape(len(va), -1).to(dev)
Xte = torch.from_numpy(loads[te]).to(dev); Yte = torch.from_numpy(stress[te]).reshape(len(te), -1).to(dev)
Ktr = torch.from_numpy(K_tr).to(dev); Kva = torch.from_numpy(K_va).to(dev); Kte = torch.from_numpy(K_te).to(dev)

mu_x, sd_x = Xtr.mean(), Xtr.std()
mu_y = Ytr.mean(0, keepdim=True); sd_y = Ytr.std()

idx2d = torch.arange(1681).reshape(41, 41)
MIR = idx2d.flip(0).reshape(-1).to(dev)

class RefMLP(nn.Module):
    def __init__(self, w, d):
        super().__init__()
        self.inp = nn.Linear(41 + 1681, w)
        self.hid = nn.ModuleList([nn.Linear(w, w) for _ in range(d - 1)])
        self.out = nn.Linear(w, 1681)
    def forward(self, x, features=False):
        h = F.silu(self.inp(x))
        for l in self.hid:
            h = h + F.silu(l(h))
        y = self.out(h)
        return (y, h) if features else y

model = RefMLP(args.width, args.depth).to(dev)
print(f"params: {sum(q.numel() for q in model.parameters())/1e6:.2f}M", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

def make_in(x, kf):
    return torch.cat([(x - mu_x) / sd_x, (kf - mu_y) / sd_y], 1)

def loss_fn(pred_n, ytrue):
    pred = pred_n * sd_y + mu_y
    return (torch.linalg.vector_norm(pred - ytrue, dim=1) / torch.linalg.vector_norm(ytrue, dim=1)).mean()

@torch.no_grad()
def evaluate(X, KF, Y, tta=True, bs=2048):
    model.eval(); outs = []
    for k in range(0, X.shape[0], bs):
        xb, kb = X[k:k+bs], KF[k:k+bs]
        pr = model(make_in(xb, kb)) * sd_y + mu_y
        if tta:
            pr2 = model(make_in(torch.flip(xb, dims=[1]), kb[:, MIR])) * sd_y + mu_y
            pr = 0.5 * (pr + pr2[:, MIR])
        outs.append(pr.float().cpu())
    model.train()
    return rel_l2(torch.cat(outs).numpy(), Y.cpu().numpy())

N = Xtr.shape[0]
best_val, best_state, best_ep = 1e9, None, -1
t0 = time.time()
for ep in range(args.epochs):
    perm = torch.randperm(N, device=dev)
    for k in range(0, N, args.batch):
        idx = perm[k:k+args.batch]
        xb, kb, yb = Xtr[idx], Ktr[idx], Ytr[idx]
        if torch.rand(()) < 0.5:
            xb = torch.flip(xb, dims=[1]); kb = kb[:, MIR]; yb = yb[:, MIR]
        loss = loss_fn(model(make_in(xb, kb)), yb)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    sched.step()
    if (ep + 1) % 10 == 0:
        e_va = evaluate(Xva, Kva, Yva)
        note = ""
        if e_va < best_val:
            best_val, best_ep = e_va, ep
            best_state = {k2: v.detach().clone() for k2, v in model.state_dict().items()}
            torch.save(best_state, RUNS / f"_ckpt_{args.tag}_s{args.seed}.pt")
            note = " *"
        print(f"ep {ep+1:4d}  train {loss.item():.4f}  val {e_va:.4f}  [{time.time()-t0:.0f}s]{note}", flush=True)

model.load_state_dict(best_state)
e_va = evaluate(Xva, Kva, Yva); e_te = evaluate(Xte, Kte, Yte)
e_te_no = evaluate(Xte, Kte, Yte, tta=False)
print(f"FINAL  val {e_va:.4f}  test(TTA) {e_te:.4f}  test(noTTA) {e_te_no:.4f}", flush=True)
name = f"{args.tag}_s{args.seed}_w{args.width}_d{args.depth}"
torch.save(model.state_dict(), RUNS / f"{name}.pt")
save_run(name, dict(kind="mlpR", args=vars(args), val=e_va, test=e_te, test_notta=e_te_no,
                    best_ep=best_ep, minutes=(time.time()-t0)/60))
print("saved", name, flush=True)
