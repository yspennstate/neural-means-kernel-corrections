"""UNet baseline for the structural mechanics benchmark.

Same input/output/metric conventions as train_fno.py: input is the broadcast
load plus coordinate channels, output the von Mises stress field, loss the
benchmark relative L2. A conv encoder-decoder gives the ensemble a member with
local multiscale inductive bias, decorrelated from the spectral/MLP members.

Usage: python train_unet.py [--seed 0] [--epochs 200] [--width 48]
       [--mirror 1] [--ntrain 0] [--final 0] [--tag unet]
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
p.add_argument("--epochs", type=int, default=200)
p.add_argument("--width", type=int, default=48)
p.add_argument("--batch", type=int, default=256)
p.add_argument("--lr", type=float, default=1.5e-3)
p.add_argument("--wd", type=float, default=1e-5)
p.add_argument("--mirror", type=int, default=1)
p.add_argument("--ntrain", type=int, default=0)
p.add_argument("--lowval", type=int, default=0)
p.add_argument("--final", type=int, default=0)
p.add_argument("--threads", type=int, default=0)
p.add_argument("--tag", type=str, default="unet")
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
    tr = np.concatenate([tr, va])
G = 41

def to_dev(idx):
    return torch.from_numpy(loads[idx]).to(dev), torch.from_numpy(stress[idx]).to(dev)

Xtr, Ytr = to_dev(tr); Xva, Yva = to_dev(va); Xte, Yte = to_dev(te)
mu_y = Ytr.mean(0, keepdim=True); sd_y = Ytr.std().item()
mu_x = Xtr.mean().item(); sd_x = Xtr.std().item()

lin = torch.linspace(0, 1, G, device=dev)
XX, YYc = torch.meshgrid(lin, lin, indexing="ij")
coords = torch.stack([XX, YYc])[None]

def make_input(xload):
    n = xload.shape[0]
    f2d = ((xload - mu_x) / sd_x)[:, :, None].expand(n, G, G)
    return torch.cat([f2d[:, None], coords.expand(n, 2, G, G)], 1)

class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.g1 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.g2 = nn.GroupNorm(8, cout)
    def forward(self, x):
        x = F.silu(self.g1(self.c1(x)))
        return F.silu(self.g2(self.c2(x)))

class UNet(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.e1 = ConvBlock(3, w)
        self.e2 = ConvBlock(w, 2 * w)
        self.e3 = ConvBlock(2 * w, 4 * w)
        self.bott = ConvBlock(4 * w, 8 * w)
        self.d3 = ConvBlock(8 * w + 4 * w, 4 * w)
        self.d2 = ConvBlock(4 * w + 2 * w, 2 * w)
        self.d1 = ConvBlock(2 * w + w, w)
        self.head = nn.Conv2d(w, 1, 1)
    def forward(self, x, features=False):
        h1 = self.e1(x)                                   # 41
        h2 = self.e2(F.avg_pool2d(h1, 2, ceil_mode=True)) # 21
        h3 = self.e3(F.avg_pool2d(h2, 2, ceil_mode=True)) # 11
        hb = self.bott(F.avg_pool2d(h3, 2, ceil_mode=True))  # 6
        u3 = self.d3(torch.cat([F.interpolate(hb, size=h3.shape[-2:], mode="bilinear", align_corners=False), h3], 1))
        u2 = self.d2(torch.cat([F.interpolate(u3, size=h2.shape[-2:], mode="bilinear", align_corners=False), h2], 1))
        u1 = self.d1(torch.cat([F.interpolate(u2, size=h1.shape[-2:], mode="bilinear", align_corners=False), h1], 1))
        y = self.head(u1)[:, 0]
        if features:
            return y, hb.mean(dim=(2, 3))
        return y

model = UNet(args.width).to(dev)
nparam = sum(q.numel() for q in model.parameters())
print(f"params: {nparam/1e6:.2f}M", flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

def loss_fn(pred_norm, ytrue):
    pred = pred_norm * sd_y + mu_y
    num = torch.linalg.vector_norm(pred - ytrue, dim=(1, 2))
    den = torch.linalg.vector_norm(ytrue, dim=(1, 2))
    return (num / den).mean()

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
        loss = loss_fn(model(make_input(xb)), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
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

name = f"{args.tag}_s{args.seed}_w{args.width}" + ("_mir" if args.mirror else "") + ("_final" if args.final else "")
torch.save(model.state_dict(), RUNS / f"{name}.pt")
save_run(name, dict(kind="unet", args=vars(args), params=nparam, val=e_va, test=e_te,
                    test_mirror_tta=e_te_tta, best_ep=best_ep, minutes=(time.time()-t0)/60))
print("saved", name, flush=True)
