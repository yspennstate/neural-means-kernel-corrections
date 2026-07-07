"""Hybrid pipeline: load trained nets, ensemble them (val-fitted convex weights),
then add a kernel-ridge correction fitted to the ensemble residuals, using the
41-dim load (and optionally deep features) as kernel inputs.

Stages reported: single models -> mean ensemble -> stacked ensemble ->
+ residual KRR (input space) -> + residual KRR (deep features).

Usage: python hybrid.py --runs fnoA_s0_...,vit_s0_... [--ntrain 0] [--tag hyb]
"""
import argparse, json, time, sys, pathlib
import numpy as np
import torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import load_arrays, canonical_split, rel_l2, save_run, RUNS, DATA
import models as M

p = argparse.ArgumentParser()
p.add_argument("--runs", type=str, required=True, help="comma-separated run names (json+pt in runs/)")
p.add_argument("--ntrain", type=int, default=0)
p.add_argument("--lowval", type=int, default=0)
p.add_argument("--tag", type=str, default="hyb")
p.add_argument("--krr_sub", type=int, default=8000, help="subset size for KRR hyperparameter tuning")
args = p.parse_args()

dev = torch.device("cpu")  # GPU banned (crash recovery); MLP inference is cheap on CPU
loads, stress = load_arrays()
tr, va, te = canonical_split(n_val=1000, seed=0)
if args.ntrain > 0:
    tr = np.load(DATA / "idx_train.npy")[:args.ntrain]
    if args.lowval > 0:
        va = tr[-args.lowval:]; tr = tr[:-args.lowval]
Ytr = stress[tr].reshape(len(tr), -1).astype(np.float64)
Yva = stress[va].reshape(len(va), -1).astype(np.float64)
Yte = stress[te].reshape(len(te), -1).astype(np.float64)

idx2d = torch.arange(1681).reshape(41, 41)
MIR = idx2d.flip(0).reshape(-1).numpy()

def build_model(run):
    cfg = json.load(open(RUNS / f"{run}.json"))
    a = cfg["args"]
    if cfg["kind"] == "fno":
        m = M.FNO2d(a["width"], a["modes"], a["layers"])
    elif cfg["kind"] == "vit":
        m = M.OpFormer(a["dim"], a["depth"], a["heads"])
    else:
        m = M.MLP(a["width"], a["depth"])
    sd = torch.load(RUNS / f"{run}.pt", map_location="cpu", weights_only=True)
    m.load_state_dict(sd); m.to(dev).eval()
    return m, cfg

# normalization stats must match training: recompute from the training pool
tr_full, va_full, _ = canonical_split(n_val=1000, seed=0)
def stats_for(cfg):
    t = tr_full
    if cfg["args"].get("ntrain", 0) > 0:
        t = np.load(DATA / "idx_train.npy")[:cfg["args"]["ntrain"]]
        if cfg["args"].get("lowval", 0) > 0:
            t = t[:-cfg["args"]["lowval"]]
    if cfg["args"].get("final", 0): t = np.concatenate([t, va_full])
    Xs = loads[t]; Ys = stress[t].reshape(len(t), -1)
    return (float(Xs.mean()), float(Xs.std()),
            torch.from_numpy(Ys.mean(0, keepdims=True)).to(dev), float(Ys.std()))

G = 41
lin = torch.linspace(0, 1, G, device=dev)
XX, YYc = torch.meshgrid(lin, lin, indexing="ij")
coords = torch.stack([XX, YYc])[None]

@torch.no_grad()
def predict(m, cfg, X_np, tta=True, feats=False, bs=512):
    mu_x, sd_x, mu_y, sd_y = stats_for(cfg)
    outs, fts = [], []
    X = torch.from_numpy(X_np).to(dev)
    for k in range(0, len(X), bs):
        xb = X[k:k+bs]
        def run_once(xin):
            xn = (xin - mu_x) / sd_x
            if cfg["kind"] == "fno":
                n = xin.shape[0]
                f2d = xn[:, :, None].expand(n, G, G)
                inp = torch.cat([f2d[:, None], coords.expand(n, 2, G, G)], 1)
                if feats:
                    y, f = m(inp, features=True); return y.reshape(n, -1), f
                return m(inp).reshape(n, -1), None
            else:
                if feats:
                    y, f = m(xn, features=True); return y, f
                return m(xn), None
        y1, f1 = run_once(xb)
        y1 = y1 * sd_y + mu_y
        if tta:
            y2, f2 = run_once(torch.flip(xb, dims=[1]))
            y2 = y2 * sd_y + mu_y
            y1 = 0.5 * (y1 + y2[:, torch.from_numpy(MIR).to(dev)])
            if feats: f1 = 0.5 * (f1 + f2)
        outs.append(y1.float().cpu().numpy())
        if feats: fts.append(f1.float().cpu().numpy())
    Y = np.concatenate(outs).astype(np.float64)
    Fs = np.concatenate(fts) if feats else None
    return Y, Fs

runs = [r.strip() for r in args.runs.split(",")]
P_tr, P_va, P_te, F_tr, F_va, F_te = [], [], [], [], [], []
report = {}
for r in runs:
    m, cfg = build_model(r)
    t0 = time.time()
    ptr, ftr = predict(m, cfg, loads[tr], feats=True)
    pva, fva = predict(m, cfg, loads[va], feats=True)
    pte, fte = predict(m, cfg, loads[te], feats=True)
    P_tr.append(ptr); P_va.append(pva); P_te.append(pte)
    F_tr.append(ftr); F_va.append(fva); F_te.append(fte)
    e = rel_l2(pte, Yte)
    report[r] = dict(test_tta=e, val_tta=rel_l2(pva, Yva))
    print(f"{r}: val {report[r]['val_tta']:.4f}  test {e:.4f}  [{time.time()-t0:.0f}s]", flush=True)
    del m; torch.cuda.empty_cache()

# ---- mean ensemble ----
E_tr = np.mean(P_tr, 0); E_va = np.mean(P_va, 0); E_te = np.mean(P_te, 0)
print(f"MEAN ens: val {rel_l2(E_va, Yva):.4f}  test {rel_l2(E_te, Yte):.4f}", flush=True)
report["mean_ens"] = dict(val=rel_l2(E_va, Yva), test=rel_l2(E_te, Yte))

# ---- stacked ensemble: convex weights minimizing val rel-L2 ----
if len(runs) > 1:
    Pv = torch.tensor(np.stack(P_va), device=dev)          # (m, n, 1681)
    Yv = torch.tensor(Yva, device=dev)
    w = torch.zeros(len(runs), device=dev, requires_grad=True)
    optw = torch.optim.Adam([w], lr=0.05)
    for it in range(400):
        ww = torch.softmax(w, 0)
        pred = torch.einsum("m,mnd->nd", ww, Pv)
        l = (torch.linalg.vector_norm(pred - Yv, dim=1) / torch.linalg.vector_norm(Yv, dim=1)).mean()
        optw.zero_grad(); l.backward(); optw.step()
    ww = torch.softmax(w, 0).detach().cpu().numpy()
    print("stack weights:", np.round(ww, 3), flush=True)
    E_tr = np.einsum("m,mnd->nd", ww, np.stack(P_tr))
    E_va = np.einsum("m,mnd->nd", ww, np.stack(P_va))
    E_te = np.einsum("m,mnd->nd", ww, np.stack(P_te))
    report["stack_ens"] = dict(w=ww.tolist(), val=rel_l2(E_va, Yva), test=rel_l2(E_te, Yte))
    print(f"STACK ens: val {report['stack_ens']['val']:.4f}  test {report['stack_ens']['test']:.4f}", flush=True)

# ---- residual KRR ----
def sqdist(A, B):
    aa = (A * A).sum(1)[:, None]; bb = (B * B).sum(1)[None, :]
    return np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0)

def matern52(D2, s):
    r2 = D2 / (s * s); r = np.sqrt(r2); a = np.sqrt(5.0) * r
    return (1.0 + a + (5.0 / 3.0) * r2) * np.exp(-a)

def krr_stage(Z_tr, Z_va, Z_te, R_tr, base_va, label):
    """Fit KRR Z->R (residuals). Tune on subset against the given validation
    baseline, refit full. Returns corrections."""
    from scipy.linalg import cho_factor, cho_solve
    mu = Z_tr.mean(0); sd = Z_tr.std(0) + 1e-12
    Zt = (Z_tr - mu) / sd; Zv = (Z_va - mu) / sd; Ze = (Z_te - mu) / sd
    nsub = min(args.krr_sub, len(Zt))
    rng = np.random.default_rng(0)
    sub = rng.choice(len(Zt), size=nsub, replace=False)
    D2s = sqdist(Zt[sub], Zt[sub]); D2vs = sqdist(Zv, Zt[sub])
    med = np.sqrt(np.median(D2s[np.triu_indices(nsub, 1)]))
    best = dict(val=1e9)
    Rva_base = rel_l2(base_va, Yva)
    for smult in [0.5, 1.0, 2.0, 4.0]:
        Ks = matern52(D2s, smult * med); Kvs = matern52(D2vs, smult * med)
        for lam in [1e-7, 1e-5, 1e-3]:
            c = cho_factor(Ks + lam * nsub * np.eye(nsub), lower=True, check_finite=False)
            a1 = cho_solve(c, R_tr[sub], check_finite=False)
            e = rel_l2(base_va + Kvs @ a1, Yva)
            if e < best["val"]:
                best = dict(val=e, smult=smult, lam=lam)
    print(f"  [{label}] tune: smult={best['smult']} lam={best['lam']} val={best['val']:.4f} (base {Rva_base:.4f})", flush=True)
    # refit on full train
    D2 = sqdist(Zt, Zt); med_f = med
    K = matern52(D2, best["smult"] * med_f)
    c = cho_factor(K + best["lam"] * len(K) * np.eye(len(K)), lower=True, check_finite=False)
    a_full = cho_solve(c, R_tr, check_finite=False)
    corr_va = matern52(sqdist(Zv, Zt), best["smult"] * med_f) @ a_full
    corr_te = np.zeros_like(E_te)
    CH = 4000
    for k in range(0, len(Ze), CH):
        corr_te[k:k+CH] = matern52(sqdist(Ze[k:k+CH], Zt), best["smult"] * med_f) @ a_full
    corr_tr = K @ a_full
    return corr_tr, corr_va, corr_te, best

R_tr = Ytr - E_tr
t0 = time.time()
c_tr, c_va, c_te, best_in = krr_stage(loads[tr].astype(np.float64), loads[va].astype(np.float64),
                                      loads[te].astype(np.float64), R_tr, E_va, "input-KRR")
E2_va = E_va + c_va; E2_te = E_te + c_te; E2_tr = E_tr + c_tr
report["plus_input_krr"] = dict(**{k: v for k, v in best_in.items() if k != "val"},
                                val=rel_l2(E2_va, Yva), test=rel_l2(E2_te, Yte))
print(f"+input-KRR: val {report['plus_input_krr']['val']:.4f}  test {report['plus_input_krr']['test']:.4f}  [{time.time()-t0:.0f}s]", flush=True)

# ---- deep-feature KRR on remaining residual ----
Feat_tr = np.concatenate(F_tr, 1).astype(np.float64)
Feat_va = np.concatenate(F_va, 1).astype(np.float64)
Feat_te = np.concatenate(F_te, 1).astype(np.float64)
R2_tr = Ytr - E2_tr
t0 = time.time()
d_tr, d_va, d_te, best_ft = krr_stage(Feat_tr, Feat_va, Feat_te, R2_tr, E2_va, "feat-KRR")
E3_va = E2_va + d_va; E3_te = E2_te + d_te; E3_tr = E2_tr + d_tr
report["plus_feat_krr"] = dict(**{k: v for k, v in best_ft.items() if k != "val"},
                               val=rel_l2(E3_va, Yva), test=rel_l2(E3_te, Yte))
print(f"+feat-KRR: val {report['plus_feat_krr']['val']:.4f}  test {report['plus_feat_krr']['test']:.4f}  [{time.time()-t0:.0f}s]", flush=True)

# ---- second input-space pass (finer scale on what remains) ----
R3_tr = Ytr - E3_tr
t0 = time.time()
g_tr, g_va, g_te, best_in2 = krr_stage(loads[tr].astype(np.float64), loads[va].astype(np.float64),
                                       loads[te].astype(np.float64), R3_tr, E3_va, "input-KRR-2")
E4_va = E3_va + g_va; E4_te = E3_te + g_te
report["plus_input_krr2"] = dict(**{k: v for k, v in best_in2.items() if k != "val"},
                                 val=rel_l2(E4_va, Yva), test=rel_l2(E4_te, Yte))
print(f"+input-KRR-2: val {report['plus_input_krr2']['val']:.4f}  test {report['plus_input_krr2']['test']:.4f}  [{time.time()-t0:.0f}s]", flush=True)

# choose final = best val among stages
stage_preds = {"mean_ens": E_te if len(runs) == 1 else None, "stack_ens": E_te,
               "plus_input_krr": E2_te, "plus_feat_krr": E3_te, "plus_input_krr2": E4_te}
stages = {k: v for k, v in report.items() if isinstance(v, dict) and "val" in v and k in stage_preds}
final_stage = min(stages, key=lambda k: stages[k]["val"])
report["final_stage"] = final_stage
report["final_test"] = stages[final_stage]["test"]
print(f"\nFINAL ({final_stage}): test {report['final_test']:.4f}", flush=True)

np.save(RUNS / f"{args.tag}_pred_test.npy", stage_preds[final_stage].astype(np.float32))
# artifacts for UQ / spectra analysis
np.save(RUNS / f"{args.tag}_stack_tr.npy", E_tr.astype(np.float32))
np.save(RUNS / f"{args.tag}_stack_va.npy", E_va.astype(np.float32))
np.save(RUNS / f"{args.tag}_stack_te.npy", E_te.astype(np.float32))
np.save(RUNS / f"{args.tag}_feat_tr.npy", Feat_tr.astype(np.float32))
np.save(RUNS / f"{args.tag}_feat_va.npy", Feat_va.astype(np.float32))
np.save(RUNS / f"{args.tag}_feat_te.npy", Feat_te.astype(np.float32))
np.save(RUNS / f"{args.tag}_members_te.npy", np.stack(P_te).astype(np.float32))  # (M,n,1681) for disagreement UQ
save_run(args.tag, dict(kind="hybrid", runs=runs, report=report, ntrain=len(tr)))
print("saved", args.tag, flush=True)
