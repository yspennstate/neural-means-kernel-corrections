"""Deep MLP baseline: 41-dim load -> 1681-dim stress field (PARA-Net, modernized).

Small model, trains fast; supports mirror augmentation. Loss = benchmark metric.
Usage: python train_mlp.py [--seed 0] [--epochs 400] [--width 1024] [--depth 4]
       [--mirror 1] [--ntrain 0] [--tag mlp]
"""
import argparse, time, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import load_arrays, canonical_split, rel_l2, save_run, RUNS, DATA

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=400)
p.add_argument("--width", type=int, default=1024)
p.add_argument("--depth", type=int, default=4)
p.add_argument("--batch", type=int, default=256)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--wd", type=float, default=1e-5)
p.add_argument("--mirror", type=int, default=1)
p.add_argument("--ntrain", type=int, default=0)
p.add_argument("--lowval", type=int, default=0)
p.add_argument("--final", type=int, default=0)
p.add_argument("--kf", type=float, default=0.0, help="weight of the Kernel-Flow regularizer on trunk features")
p.add_argument("--threads", type=int, default=0)
p.add_argument("--tag", type=str, default="mlp")
args = p.parse_args()

if args.threads > 0:
    torch.set_num_threads(args.threads)
torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", dev, flush=True)
loads, stress = load_arrays()
tr, va, te = canonical_split(n_val=1000, seed=0)
if args.ntrain > 0:
    tr = np.load(DATA / "idx_train.npy")[:args.ntrain]
    if args.lowval > 0:
        va = tr[-args.lowval:]; tr = tr[:-args.lowval]
if args.final: tr = np.concatenate([tr, va])

Xtr = torch.from_numpy(loads[tr]).to(dev); Ytr = torch.from_numpy(stress[tr]).reshape(len(tr), -1).to(dev)
Xva = torch.from_numpy(loads[va]).to(dev); Yva = torch.from_numpy(stress[va]).reshape(len(va), -1).to(dev)
Xte = torch.from_numpy(loads[te]).to(dev); Yte = torch.from_numpy(stress[te]).reshape(len(te), -1).to(dev)

mu_x, sd_x = Xtr.mean(), Xtr.std()
mu_y = Ytr.mean(0, keepdim=True); sd_y = Ytr.std()

class MLP(nn.Module):
    def __init__(self, w, d):
        super().__init__()
        self.inp = nn.Linear(41, w)
        self.hid = nn.ModuleList([nn.Linear(w, w) for _ in range(d - 1)])
        self.out = nn.Linear(w, 1681)
    def forward(self, x, features=False):
        h = F.silu(self.inp(x))
        for l in self.hid:
            h = h + F.silu(l(h))          # residual trunk
        y = self.out(h)
        return (y, h) if features else y

model = MLP(args.width, args.depth).to(dev)
print(f"params: {sum(q.numel() for q in model.parameters())/1e6:.2f}M", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

MIR = torch.arange(1680, -1, -1, device=dev).reshape(41, 41).flip(0).reshape(-1)
# mirror of flattened field: flip axis0 <-> reverse rows. Build explicit index:
idx2d = torch.arange(1681, device=dev).reshape(41, 41)
MIR = idx2d.flip(0).reshape(-1)

def loss_fn(pred_n, ytrue):
    pred = pred_n * sd_y + mu_y
    return (torch.linalg.vector_norm(pred - ytrue, dim=1) / torch.linalg.vector_norm(ytrue, dim=1)).mean()

log_gamma = torch.zeros((), device=dev, requires_grad=True)
opt_kf = torch.optim.Adam([log_gamma], lr=1e-2) if args.kf > 0 else None

def kf_loss(feats, ytrue_n):
    """Yoo-Owhadi l2 Kernel-Flow loss on trunk features."""
    b = feats.shape[0]; half = b // 2
    per = torch.randperm(b, device=feats.device)
    fb, yb2 = feats[per], ytrue_n[per]
    fc, yc = fb[:half], yb2[:half]
    with torch.no_grad():
        med = torch.cdist(fb, fb).median()
    gamma = torch.exp(log_gamma) / (2 * med * med + 1e-12)
    K_bc = torch.exp(-gamma * torch.cdist(fb, fc).pow(2))
    K_cc = torch.exp(-gamma * torch.cdist(fc, fc).pow(2))
    K_cc = K_cc + 1e-3 * torch.eye(half, device=feats.device)
    pred = K_bc @ torch.cholesky_solve(yc, torch.linalg.cholesky(K_cc))
    return ((yb2 - pred) ** 2).mean()

@torch.no_grad()
def evaluate(X, Y, tta=False):
    model.eval()
    pr = model((X - mu_x) / sd_x) * sd_y + mu_y
    if tta:
        pr2 = model((torch.flip(X, dims=[1]) - mu_x) / sd_x) * sd_y + mu_y
        pr = 0.5 * (pr + pr2[:, MIR])
    model.train()
    return rel_l2(pr.float().cpu().numpy(), Y.cpu().numpy())

N = Xtr.shape[0]
best_val, best_state, best_ep = 1e9, None, -1
t0 = time.time()
for ep in range(args.epochs):
    perm = torch.randperm(N, device=dev)
    for k in range(0, N, args.batch):
        idx = perm[k:k+args.batch]
        xb, yb = Xtr[idx], Ytr[idx]
        if args.mirror and torch.rand(()) < 0.5:
            xb = torch.flip(xb, dims=[1]); yb = yb[:, MIR]
        if args.kf > 0:
            pred, feats = model((xb - mu_x) / sd_x, features=True)
            loss = loss_fn(pred, yb) + args.kf * kf_loss(feats, (yb - mu_y) / sd_y)
        else:
            loss = loss_fn(model((xb - mu_x) / sd_x), yb)
        opt.zero_grad(set_to_none=True)
        if opt_kf is not None: opt_kf.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if opt_kf is not None: opt_kf.step()
    sched.step()
    if (ep + 1) % 10 == 0:
        e_va = evaluate(Xva, Yva)
        note = ""
        if not args.final and e_va < best_val:
            best_val, best_ep = e_va, ep
            best_state = {k2: v.detach().clone() for k2, v in model.state_dict().items()}
            torch.save(best_state, RUNS / f"_ckpt_{args.tag}_s{args.seed}.pt")
            note = " *"
        print(f"ep {ep+1:4d}  train {loss.item():.4f}  val {e_va:.4f}  [{time.time()-t0:.0f}s]{note}", flush=True)

if not args.final and best_state is not None:
    model.load_state_dict(best_state)
e_va = evaluate(Xva, Yva); e_te = evaluate(Xte, Yte); e_tta = evaluate(Xte, Yte, tta=True)
print(f"FINAL  val {e_va:.4f}  test {e_te:.4f}  test+TTA {e_tta:.4f}", flush=True)
name = f"{args.tag}_s{args.seed}_w{args.width}_d{args.depth}_n{N}" + ("_mir" if args.mirror else "")
torch.save(model.state_dict(), RUNS / f"{name}.pt")
save_run(name, dict(kind="mlp", args=vars(args), val=e_va, test=e_te, test_tta=e_tta,
                    best_ep=best_ep, minutes=(time.time()-t0)/60))
print("saved", name, flush=True)
