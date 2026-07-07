"""Transformer operator: encoder over the 41 load samples (tokens), decoder by
cross-attention from 1681 grid-point queries. A ViT-style architecture adapted
to the 1D-load -> 2D-field structure of the benchmark.

Usage: python train_vit.py [--seed 0] [--epochs 300] [--dim 192] [--depth 5]
       [--heads 4] [--mirror 1] [--ntrain 0] [--final 0] [--tag vit]
"""
import argparse, time, sys, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import load_arrays, canonical_split, rel_l2, save_run, RUNS, DATA

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=300)
p.add_argument("--dim", type=int, default=192)
p.add_argument("--depth", type=int, default=5)
p.add_argument("--heads", type=int, default=4)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--lr", type=float, default=8e-4)
p.add_argument("--wd", type=float, default=1e-4)
p.add_argument("--mirror", type=int, default=1)
p.add_argument("--ntrain", type=int, default=0)
p.add_argument("--lowval", type=int, default=0)
p.add_argument("--final", type=int, default=0)
p.add_argument("--threads", type=int, default=0)
p.add_argument("--tag", type=str, default="vit")
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

idx2d = torch.arange(1681, device=dev).reshape(41, 41)
MIR = idx2d.flip(0).reshape(-1)

def fourier_feats(t, nf):
    # t: (...,) in [0,1] -> (..., 2*nf)
    k = torch.arange(1, nf + 1, device=t.device, dtype=t.dtype) * math.pi
    ang = t[..., None] * k
    return torch.cat([torch.sin(ang), torch.cos(ang)], -1)

class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.n1 = nn.LayerNorm(dim); self.att = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
    def forward(self, x):
        h = self.n1(x); x = x + self.att(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))

class XBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.nq = nn.LayerNorm(dim); self.nk = nn.LayerNorm(dim)
        self.att = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
    def forward(self, q, kv):
        h = self.att(self.nq(q), self.nk(kv), self.nk(kv), need_weights=False)[0]
        q = q + h
        return q + self.mlp(self.n2(q))

class OpFormer(nn.Module):
    def __init__(self, dim, depth, heads, nf=10):
        super().__init__()
        self.nf = nf
        self.tok = nn.Linear(1 + 2 * nf, dim)
        self.enc = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.qproj = nn.Linear(4 * nf, dim)
        self.dec1 = XBlock(dim, heads)
        self.dec2 = XBlock(dim, heads)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
        x1 = torch.linspace(0, 1, 41)
        g1, g2 = torch.meshgrid(x1, x1, indexing="ij")
        self.register_buffer("grid", torch.stack([g1.reshape(-1), g2.reshape(-1)], -1))  # (1681,2)
        self.register_buffer("pos1d", x1)
    def encode(self, x):
        n = x.shape[0]
        pe = fourier_feats(self.pos1d, self.nf).expand(n, 41, 2 * self.nf)
        h = self.tok(torch.cat([x[..., None], pe], -1))
        for b in self.enc: h = b(h)
        return h                                            # (n,41,dim)
    def forward(self, x, features=False, qidx=None):
        h = self.encode(x)
        grid = self.grid if qidx is None else self.grid[qidx]
        q = self.qproj(torch.cat([fourier_feats(grid[:, 0], self.nf),
                                  fourier_feats(grid[:, 1], self.nf)], -1))
        q = q[None].expand(x.shape[0], grid.shape[0], -1)
        q = self.dec1(q, h)
        q = self.dec2(q, h)
        y = self.head(q)[..., 0]                            # (n,|q|)
        if features:
            return y, h.mean(1)                             # pooled encoder features
        return y

model = OpFormer(args.dim, args.depth, args.heads).to(dev)
print(f"params: {sum(q.numel() for q in model.parameters())/1e6:.2f}M", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
warm = 10
def lr_lambda(ep):
    if ep < warm: return (ep + 1) / warm
    t = (ep - warm) / max(1, args.epochs - warm)
    return 0.5 * (1 + math.cos(math.pi * t)) * (1 - 1e-3) + 1e-3
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

def loss_fn(pred_n, ytrue, qidx=None):
    mu_q = mu_y if qidx is None else mu_y[:, qidx]
    pred = pred_n * sd_y + mu_q
    yt = ytrue if qidx is None else ytrue[:, qidx]
    return (torch.linalg.vector_norm(pred - yt, dim=1) / torch.linalg.vector_norm(yt, dim=1)).mean()

@torch.no_grad()
def evaluate(X, Y, tta=False, bs=512):
    model.eval(); outs = []
    for k in range(0, X.shape[0], bs):
        xb = X[k:k+bs]
        pr = model((xb - mu_x) / sd_x) * sd_y + mu_y
        if tta:
            pr2 = model((torch.flip(xb, dims=[1]) - mu_x) / sd_x) * sd_y + mu_y
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
        xb, yb = Xtr[idx], Ytr[idx]
        if args.mirror and torch.rand(()) < 0.5:
            xb = torch.flip(xb, dims=[1]); yb = yb[:, MIR]
        qidx = torch.randperm(1681, device=dev)[:512]
        loss = loss_fn(model((xb - mu_x) / sd_x, qidx=qidx), yb, qidx=qidx)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
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
name = f"{args.tag}_s{args.seed}_d{args.dim}x{args.depth}_n{N}" + ("_mir" if args.mirror else "")
torch.save(model.state_dict(), RUNS / f"{name}.pt")
save_run(name, dict(kind="vit", args=vars(args), val=e_va, test=e_te, test_tta=e_tta,
                    best_ep=best_ep, minutes=(time.time()-t0)/60))
print("saved", name, flush=True)
