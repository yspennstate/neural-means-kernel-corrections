"""FNO-2d baseline for the structural mechanics benchmark.

Input: 1-D load broadcast to a 41x41 channel (as shipped in the dataset) plus
coordinate channels. Output: von Mises stress field. Loss = mean relative L2
in original units (the benchmark metric). All tensors live on the GPU.

Usage: python train_fno.py [--seed 0] [--epochs 600] [--width 64] [--modes 14]
       [--layers 4] [--mirror 0] [--final 0] [--tag fno]
"""
import argparse, json, math, time, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import load_arrays, canonical_split, rel_l2, save_run, RUNS, DATA

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=600)
p.add_argument("--width", type=int, default=64)
p.add_argument("--modes", type=int, default=14)
p.add_argument("--layers", type=int, default=4)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--lr", type=float, default=1.5e-3)
p.add_argument("--wd", type=float, default=1e-6)
p.add_argument("--mirror", type=int, default=0, help="1 = train-time mirror augmentation")
p.add_argument("--final", type=int, default=0, help="1 = train on all 20000 (no val selection), fixed schedule")
p.add_argument("--kf", type=float, default=0.0, help="weight of the Kernel-Flow regularizer on pooled features")
p.add_argument("--ntrain", type=int, default=0)
p.add_argument("--lowval", type=int, default=0, help="carve val from inside ntrain (strict low-data protocol)")
p.add_argument("--threads", type=int, default=0)
p.add_argument("--tag", type=str, default="fno")
args = p.parse_args()

if args.threads > 0:
    torch.set_num_threads(args.threads)
torch.manual_seed(args.seed); np.random.seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", dev, flush=True)

loads, stress = load_arrays()
tr, va, te = canonical_split(n_val=1000, seed=0)
if args.ntrain > 0:
    tr = np.load(DATA / "idx_train.npy")[:args.ntrain]
    if args.lowval > 0:
        va = tr[-args.lowval:]; tr = tr[:-args.lowval]
if args.final:
    tr = np.concatenate([tr, va])  # full 20000
G = 41

def to_gpu(idx):
    x = torch.from_numpy(loads[idx]).to(dev)                      # (n,41)
    y = torch.from_numpy(stress[idx]).to(dev)                     # (n,41,41)
    return x, y

Xtr, Ytr = to_gpu(tr); Xva, Yva = to_gpu(va); Xte, Yte = to_gpu(te)

mu_y = Ytr.mean(0, keepdim=True)            # per-pixel mean field
sd_y = Ytr.std().item()                     # global scale
mu_x = Xtr.mean().item(); sd_x = Xtr.std().item()

lin = torch.linspace(0, 1, G, device=dev)
XX, YYc = torch.meshgrid(lin, lin, indexing="ij")
coords = torch.stack([XX, YYc])[None]       # (1,2,41,41)

def make_input(xload):
    n = xload.shape[0]
    f2d = ((xload - mu_x) / sd_x)[:, :, None].expand(n, G, G)     # broadcast along axis 1
    return torch.cat([f2d[:, None], coords.expand(n, 2, G, G)], 1)  # (n,3,41,41)

class SpectralConv2d(nn.Module):
    def __init__(self, cin, cout, m1, m2):
        super().__init__()
        scale = 1.0 / (cin * cout)
        self.m1, self.m2 = m1, m2
        self.w1 = nn.Parameter(scale * torch.randn(cin, cout, m1, m2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.randn(cin, cout, m1, m2, dtype=torch.cfloat))
    def forward(self, x):
        B, C, H, W = x.shape
        xf = torch.fft.rfft2(x)                                   # (B,C,H,W//2+1)
        out = torch.zeros(B, self.w1.shape[1], H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        out[:, :, :self.m1, :self.m2] = torch.einsum("bixy,ioxy->boxy", xf[:, :, :self.m1, :self.m2], self.w1)
        out[:, :, -self.m1:, :self.m2] = torch.einsum("bixy,ioxy->boxy", xf[:, :, -self.m1:, :self.m2], self.w2)
        return torch.fft.irfft2(out, s=(H, W))

class FNO2d(nn.Module):
    def __init__(self, width, modes, layers):
        super().__init__()
        self.lift = nn.Conv2d(3, width, 1)
        self.sp = nn.ModuleList([SpectralConv2d(width, width, modes, modes) for _ in range(layers)])
        self.sk = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.proj1 = nn.Conv2d(width, 128, 1)
        self.proj2 = nn.Conv2d(128, 1, 1)
    def forward(self, x, features=False):
        h = self.lift(x)
        for s, k in zip(self.sp, self.sk):
            h = F.gelu(s(h) + k(h))
        g = F.gelu(self.proj1(h))
        y = self.proj2(g)[:, 0]                                   # (n,41,41)
        if features:
            return y, g.mean(dim=(2, 3))
        return y

model = FNO2d(args.width, args.modes, args.layers).to(dev)
nparam = sum(q.numel() for q in model.parameters())
print(f"params: {nparam/1e6:.2f}M", flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

def loss_fn(pred_norm, ytrue):
    pred = pred_norm * sd_y + mu_y
    num = torch.linalg.vector_norm(pred - ytrue, dim=(1, 2))
    den = torch.linalg.vector_norm(ytrue, dim=(1, 2))
    return (num / den).mean()

log_gamma = torch.zeros((), device=dev, requires_grad=True)
opt_kf = torch.optim.Adam([log_gamma], lr=1e-2) if args.kf > 0 else None

def kf_loss(feats, ytrue_n):
    """Yoo-Owhadi l2 Kernel-Flow loss: predict the batch from a random half via
    the RBF kernel on pooled features; error is the KF regularizer."""
    b = feats.shape[0]; half = b // 2
    per = torch.randperm(b, device=feats.device)
    fb, yb = feats[per], ytrue_n[per].reshape(b, -1)
    fc, yc = fb[:half], yb[:half]
    with torch.no_grad():
        med = torch.cdist(fb, fb).median()
    gamma = torch.exp(log_gamma) / (2 * med * med + 1e-12)
    K_bc = torch.exp(-gamma * torch.cdist(fb, fc).pow(2))
    K_cc = torch.exp(-gamma * torch.cdist(fc, fc).pow(2))
    K_cc = K_cc + 1e-3 * torch.eye(half, device=feats.device)
    pred = K_bc @ torch.cholesky_solve(yc, torch.linalg.cholesky(K_cc))
    return ((yb - pred) ** 2).mean()

@torch.no_grad()
def evaluate(X, Y, bs=1024, mirror_tta=False):
    model.eval()
    outs = []
    for k in range(0, X.shape[0], bs):
        xb = X[k:k+bs]
        pr = model(make_input(xb)) * sd_y + mu_y
        if mirror_tta:
            pr2 = model(make_input(torch.flip(xb, dims=[1]))) * sd_y + mu_y
            pr = 0.5 * (pr + torch.flip(pr2, dims=[1]))
        outs.append(pr.float().cpu())
    model.train()
    return rel_l2(torch.cat(outs).numpy(), Y.cpu().numpy())

N = Xtr.shape[0]
best_val, best_state, best_ep = 1e9, None, -1
t0 = time.time()
for ep in range(args.epochs):
    perm = torch.randperm(N, device=dev)
    tot, nb = 0.0, 0
    for k in range(0, N, args.batch):
        idx = perm[k:k+args.batch]
        xb, yb = Xtr[idx], Ytr[idx]
        if args.mirror and torch.rand(()) < 0.5:
            xb = torch.flip(xb, dims=[1]); yb = torch.flip(yb, dims=[1])
        if args.kf > 0:
            pred, feats = model(make_input(xb), features=True)
            loss = loss_fn(pred, yb) + args.kf * kf_loss(feats, (yb - mu_y) / sd_y)
        else:
            pred = model(make_input(xb))
            loss = loss_fn(pred, yb)
        opt.zero_grad(set_to_none=True)
        if opt_kf is not None: opt_kf.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if opt_kf is not None: opt_kf.step()
        tot += loss.item(); nb += 1
    sched.step()
    if (ep + 1) % 10 == 0 or ep == args.epochs - 1:
        e_va = evaluate(Xva, Yva)
        note = ""
        if not args.final and e_va < best_val:
            best_val, best_ep = e_va, ep
            best_state = {k2: v.detach().clone() for k2, v in model.state_dict().items()}
            torch.save(best_state, RUNS / f"_ckpt_{args.tag}_s{args.seed}.pt")
            note = " *"
        print(f"ep {ep+1:4d}  train {tot/nb:.4f}  val {e_va:.4f}  [{time.time()-t0:.0f}s]{note}", flush=True)

if not args.final and best_state is not None:
    model.load_state_dict(best_state)
e_va = evaluate(Xva, Yva)
e_te = evaluate(Xte, Yte)
e_te_tta = evaluate(Xte, Yte, mirror_tta=True)
print(f"FINAL  val {e_va:.4f}  test {e_te:.4f}  test+mirrorTTA {e_te_tta:.4f}  best_ep {best_ep}", flush=True)

name = f"{args.tag}_s{args.seed}_w{args.width}_m{args.modes}_L{args.layers}" + ("_mir" if args.mirror else "") + ("_final" if args.final else "")
torch.save(model.state_dict(), RUNS / f"{name}.pt")
save_run(name, dict(kind="fno", args=vars(args), params=nparam, val=e_va, test=e_te,
                    test_mirror_tta=e_te_tta, best_ep=best_ep, minutes=(time.time()-t0)/60))
print("saved", name, flush=True)
