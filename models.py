"""Model definitions shared by training scripts and the hybrid pipeline.
Definitions are identical to those in train_fno.py / train_vit.py / train_mlp.py
so saved state dicts load directly."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    def __init__(self, cin, cout, m1, m2):
        super().__init__()
        scale = 1.0 / (cin * cout)
        self.m1, self.m2 = m1, m2
        self.w1 = nn.Parameter(scale * torch.randn(cin, cout, m1, m2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.randn(cin, cout, m1, m2, dtype=torch.cfloat))
    def forward(self, x):
        B, C, H, W = x.shape
        xf = torch.fft.rfft2(x)
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
        y = self.proj2(g)[:, 0]
        if features:
            return y, g.mean(dim=(2, 3))          # (n,128)
        return y

def fourier_feats(t, nf):
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
        self.register_buffer("grid", torch.stack([g1.reshape(-1), g2.reshape(-1)], -1))
        self.register_buffer("pos1d", x1)
    def encode(self, x):
        n = x.shape[0]
        pe = fourier_feats(self.pos1d, self.nf).expand(n, 41, 2 * self.nf)
        h = self.tok(torch.cat([x[..., None], pe], -1))
        for b in self.enc: h = b(h)
        return h
    def forward(self, x, features=False):
        h = self.encode(x)
        q = self.qproj(torch.cat([fourier_feats(self.grid[:, 0], self.nf),
                                  fourier_feats(self.grid[:, 1], self.nf)], -1))
        q = q[None].expand(x.shape[0], 1681, -1)
        q = self.dec1(q, h)
        q = self.dec2(q, h)
        y = self.head(q)[..., 0]
        if features:
            return y, h.mean(1)
        return y

class MLP(nn.Module):
    def __init__(self, w, d):
        super().__init__()
        self.inp = nn.Linear(41, w)
        self.hid = nn.ModuleList([nn.Linear(w, w) for _ in range(d - 1)])
        self.out = nn.Linear(w, 1681)
    def forward(self, x, features=False):
        h = F.silu(self.inp(x))
        for l in self.hid:
            h = h + F.silu(l(h))
        y = self.out(h)
        return (y, h) if features else y

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
        h1 = self.e1(x)
        h2 = self.e2(F.avg_pool2d(h1, 2, ceil_mode=True))
        h3 = self.e3(F.avg_pool2d(h2, 2, ceil_mode=True))
        hb = self.bott(F.avg_pool2d(h3, 2, ceil_mode=True))
        u3 = self.d3(torch.cat([F.interpolate(hb, size=h3.shape[-2:], mode="bilinear", align_corners=False), h3], 1))
        u2 = self.d2(torch.cat([F.interpolate(u3, size=h2.shape[-2:], mode="bilinear", align_corners=False), h2], 1))
        u1 = self.d1(torch.cat([F.interpolate(u2, size=h1.shape[-2:], mode="bilinear", align_corners=False), h1], 1))
        y = self.head(u1)[:, 0]
        if features:
            return y, hb.mean(dim=(2, 3))
        return y
