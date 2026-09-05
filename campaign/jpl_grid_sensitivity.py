# Paired kernel-grid sensitivity; training and head routines derived from jpl_seeded.py.
"""OCO-2 pipeline at one campaign seed, with matched kernel tuning throughout.

Differences from jpl_pipeline.py:
  - every stochastic choice is seeded by --seed: the train/validation split,
    network initialization and batch order, and the kernel tuning subsample;
  - a third network is trained in the exact radiance-relative metric (the loss
    is computed through the stored PCA reconstruction, numerator and
    denominator, not the diagonally weighted coefficient surrogate);
  - the raw-input kernel rows get the identical protocol as the feature heads:
    the same scale/nugget grid tuned on validation with a 6000-point
    subsample, winner refit on all 18000 points (isotropic and ARD variants);
  - per-sample test errors in both metrics are saved for bootstrap intervals.

Environment: NMKC_ROOT, TASK_ID, NMKC_THREADS. Data under <root>/data/jpl_oco2
via NMKC_JPL_DATA (defaults to the repo layout otherwise).
Rows ridge_<mode> and combined_plus_ridge (added after the campaign) are the
frozen-feature linear-readout control for the kernel heads.

    python campaign/jpl_seeded.py --band o2 --seed 3
"""
import argparse, json, os, pathlib, sys, time, hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import cho_factor, cho_solve

sys.path.insert(0, os.environ.get("NMKC_CODE", str(pathlib.Path(__file__).resolve().parent.parent)))
import jpl_data
from jpl_data import load_band, reconstruction, radiance_error, kernel_flow_predictions, to_radiance

p = argparse.ArgumentParser()
p.add_argument("--band", default="o2")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=250)
p.add_argument("--scales", default="0.5,1,2,4", help="length-scale multipliers of the median distance (kernel heads)")
p.add_argument("--nuggets", default="1e-8,1e-6,1e-4", help="nugget grid of the kernel heads")
p.add_argument("--width", type=int, default=384)
args = p.parse_args()

THREADS = int(os.environ.get("NMKC_THREADS", "5"))
torch.set_num_threads(THREADS)
ROOT = pathlib.Path(os.environ.get("NMKC_ROOT", "."))
TASK_ID = os.environ.get("TASK_ID", f"oco_{args.band}_s{args.seed}")
OUT = ROOT / "seeds" / f"oco_{args.band}_s{args.seed}"
OUT.mkdir(parents=True, exist_ok=True)
if os.environ.get("NMKC_JPL_DATA"):
    jpl_data.DATA = pathlib.Path(os.environ["NMKC_JPL_DATA"])

sp = load_band(args.band, seed=args.seed)
Xtr, Ytr, Xval, Yval, Xte, Yte = (sp[k] for k in ("Xtr", "Ytr", "Xval", "Yval", "Xte", "Yte"))
recon = reconstruction(args.band)
w_z = np.abs(recon["s_z"]); w_z = w_z / w_z.mean()          # legacy numerator weights
s_z = recon["s_z"].astype(np.float64)                        # exact diagonal
P_mat = recon["P"].astype(np.float64)

rel = lambda Pp, T: float(np.mean(np.linalg.norm(Pp - T, axis=1) / np.linalg.norm(T, axis=1)))
rad = lambda Pp, T: radiance_error(Pp, T, recon)
# per-sample radiance denominators (constants of the data, not the model)
r_norm_tr = np.linalg.norm(to_radiance(Ytr, recon), axis=1)
r_norm_va = np.linalg.norm(to_radiance(Yval, recon), axis=1)


class ResidualMLP(nn.Module):
    def __init__(self, d_in, d_out, width, depth=4):
        super().__init__()
        self.inp = nn.Linear(d_in, width)
        self.hidden = nn.ModuleList([nn.Linear(width, width) for _ in range(depth - 1)])
        self.out = nn.Linear(width, d_out)

    def forward(self, x, return_features=False):
        h = F.silu(self.inp(x))
        for layer in self.hidden:
            h = h + F.silu(layer(h))
        return (self.out(h), h) if return_features else self.out(h)


def train(mode):
    """mode: flat | wnum (legacy numerator weights) | radx (exact radiance)."""
    torch.manual_seed(args.seed)
    f32 = lambda a: torch.tensor(np.asarray(a, np.float32))
    xt, yt, xv = f32(Xtr), f32(Ytr), f32(Xval)
    Wt = f32(w_z) if mode == "wnum" else None
    if mode == "radx":
        sz_t, P_t = f32(s_z), f32(P_mat)
        rn_t = f32(r_norm_tr)
    model = ResidualMLP(Xtr.shape[1], Ytr.shape[1], args.width)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    n = len(xt)
    best, best_state = np.inf, None
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        for k in range(0, n, 512):
            i = perm[k:k + 512]
            pred, target = model(xt[i]), yt[i]
            if mode == "flat":
                loss = (torch.linalg.vector_norm(pred - target, dim=1)
                        / torch.linalg.vector_norm(target, dim=1)).mean()
            elif mode == "wnum":
                loss = (torch.linalg.vector_norm((pred - target) * Wt, dim=1)
                        / torch.linalg.vector_norm(target * Wt, dim=1)).mean()
            else:  # radx: exact radiance-relative error through the reconstruction
                diff = ((pred - target) * sz_t) @ P_t
                loss = (torch.linalg.vector_norm(diff, dim=1) / rn_t[i]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (ep + 1) % 25 == 0 or ep == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                pv = model(xv).numpy().astype(np.float64)
            model.train()
            if mode == "flat":
                e = rel(pv, Yval)
            elif mode == "wnum":
                e = rel(pv * w_z, Yval * w_z)
            else:
                e = rad(pv, Yval)
            if e < best:
                best = e
                best_state = {k2: v.clone() for k2, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    torch.save(model.state_dict(), OUT / f"network_{mode}.pt")
    with torch.no_grad():
        preds = [model(f32(Z)).numpy().astype(np.float64) for Z in (Xtr, Xval, Xte)]
        feats = [model(f32(Z), return_features=True)[1].numpy().astype(np.float64)
                 for Z in (Xtr, Xval, Xte)]
    return preds, feats


def sqd(A, B):
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0.0)


def m52(D2, ls):
    a = np.sqrt(5.0) * np.sqrt(D2) / ls
    return (1 + a + (5.0 / 3.0) * (D2 / ls ** 2)) * np.exp(-a)


def matern_head(Ztr, Zva, Zte, err_fn, label, w=None):
    """Exact Matern-5/2 kernel ridge, the single protocol used everywhere:
    inputs standardized by train moments, scale grid --scales (default {0.5,1,2,4}) x median
    pairwise distance (6000-point estimate), nugget grid --nuggets (default {1e-8,1e-6,1e-4}),
    tuned on validation by err_fn over a 6000-sample training subsample,
    winner refit on the full training set. A diagonal metric w is applied
    AFTER standardization: standardization divides by the per-dimension std,
    so any pre-scaling of the inputs cancels exactly and reproduces the
    isotropic row digit for digit (verified live, 2026-08-02). Never pass a
    metric by rescaling the inputs of this function."""
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-9
    Ftr, Fval, Fte = (Ztr - mu) / sd, (Zva - mu) / sd, (Zte - mu) / sd
    if w is not None:
        Ftr, Fval, Fte = Ftr * w, Fval * w, Fte * w
    rng = np.random.default_rng(args.seed)
    sub = rng.choice(len(Ftr), min(6000, len(Ftr)), replace=False)
    med = np.sqrt(np.median(sqd(Ftr[sub], Ftr[sub])[np.triu_indices(len(sub), 1)]))
    best = (np.inf, None)
    cells = []
    D2s, D2vs = sqd(Ftr[sub], Ftr[sub]), sqd(Fval, Ftr[sub])
    for scale in [float(x) for x in args.scales.split(",")]:
        Ks, Kvs = m52(D2s, scale * med), m52(D2vs, scale * med)
        for nug in [float(x) for x in args.nuggets.split(",")]:
            Kr = Ks.copy(); Kr.flat[::len(sub) + 1] += nug * len(sub)
            try:
                c = cho_factor(Kr, lower=True, check_finite=False, overwrite_a=True)
            except np.linalg.LinAlgError as exc:
                cells.append(dict(scale=scale, nugget=nug, status="CHOLESKY_FAILED", error=str(exc)))
                continue
            e = err_fn(Kvs @ cho_solve(c, Ytr[sub], check_finite=False))
            cells.append(dict(scale=scale, nugget=nug, status="OK" if np.isfinite(e) else "NONFINITE", validation=float(e) if np.isfinite(e) else None))
            if np.isfinite(e) and e < best[0]:
                best = (e, (scale, nug))
    if best[1] is None:
        raise RuntimeError("No finite successful kernel candidate: "+label)
    scale, nug = best[1]
    n = len(Ftr)
    # full-Gram refit under the box-wide lock: two concurrent full solves
    # OOM-killed a task on the 31 GB box; one Gram factorization per box
    lk = open(ROOT / ".gram.lock", "w")
    import fcntl
    fcntl.flock(lk, fcntl.LOCK_EX)
    try:
        K = m52(sqd(Ftr, Ftr), scale * med); K.flat[::n + 1] += nug * n
        c = cho_factor(K, lower=True, check_finite=False, overwrite_a=True)
        alpha = cho_solve(c, Ytr, check_finite=False)
        del K
    finally:
        fcntl.flock(lk, fcntl.LOCK_UN)
        lk.close()
    out = []
    for F_ in (Ftr, Fval, Fte):
        pred = np.empty((len(F_), Ytr.shape[1]))
        for k in range(0, len(F_), 4000):
            pred[k:k + 4000] = m52(sqd(F_[k:k + 4000], Ftr), scale * med) @ alpha
        out.append(pred)
    print(f"  {label}: scale {scale} nugget {nug:g} (med {med:.3f})", flush=True)
    return out, dict(scale=scale, nugget=nug, med=float(med), validation=float(best[0]), cells=cells,
                     scale_boundary=scale in (min(map(float,args.scales.split(","))), max(map(float,args.scales.split(",")))),
                     nugget_boundary=nug in (min(map(float,args.nuggets.split(","))), max(map(float,args.nuggets.split(",")))))


def ridge_head(Ztr, Zva, Zte, err_fn, label):
    """Control for the kernel heads: a linear readout refit on the same frozen features.
    Ridge regression with intercept from the standardized last-layer features to the
    targets, the penalty chosen on validation by err_fn from a fixed grid. If this row
    matched the kernel head, the head's gain over the network would be readout refitting;
    if it matches the network, the gain is the kernel."""
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-9
    Ftr, Fval, Fte = (Ztr - mu) / sd, (Zva - mu) / sd, (Zte - mu) / sd
    ym = Ytr.mean(0)
    G = Ftr.T @ Ftr; Bm = Ftr.T @ (Ytr - ym); n = len(Ftr); d = Ftr.shape[1]
    best = (np.inf, None)
    for lam in (1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0):
        W = np.linalg.solve(G + lam * n * np.eye(d), Bm)
        e = err_fn(Fval @ W + ym)
        if e < best[0]:
            best = (e, (lam, W))
    lam, W = best[1]
    print(f"  {label}: ridge {lam:g}", flush=True)
    return [Ftr @ W + ym, Fval @ W + ym, Fte @ W + ym], dict(ridge=lam)



# Paired grid comparison. The candidate sets are fixed before test evaluation.
GRIDS = {
    "recorded_grid": dict(scales="0.5,1,2,4", nuggets="1e-8,1e-6,1e-4"),
    "expanded_grid": dict(scales="0.25,0.5,1,2,4,8,16", nuggets="1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3"),
}
t_all=time.time()
def file_sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

data_hashes={name:hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
             for name,value in zip(("Xtr","Ytr","Xval","Yval","Xte","Yte"),(Xtr,Ytr,Xval,Yval,Xte,Yte))}
identity=dict(band=args.band,seed=args.seed,epochs=args.epochs,width=args.width,threads=THREADS,
              driver_sha256=file_sha(__file__),data_sha256=data_hashes,grids=GRIDS)
identity_path=OUT/"experiment_identity.json"
if identity_path.exists():
    if json.loads(identity_path.read_text(encoding="utf-8"))!=identity:
        raise RuntimeError("Cache belongs to different source, data, or experiment")
else:
    identity_path.write_text(json.dumps(identity,indent=2)+"\n",encoding="utf-8")

networks={};ridge={};ridge_hyper={}
for mode in ("flat","wnum","radx"):
    cache=OUT/f"network_{mode}_arrays.npz";receipt=OUT/f"network_{mode}_arrays.json"
    if cache.exists() or receipt.exists():
        if not cache.exists() or not receipt.exists():raise RuntimeError("Partial feature cache")
        meta=json.loads(receipt.read_text(encoding="utf-8"))
        if meta["sha256"]!=file_sha(cache):raise RuntimeError("Feature cache changed")
        with np.load(cache) as z:
            pm=[z["pred_"+s] for s in ("train","val","test")]
            fm=[z["feature_"+s] for s in ("train","val","test")]
        print("VERIFIED_FEATURE_CACHE",mode,flush=True)
    else:
        t0=time.time();pm,fm=train(mode)
        np.savez(cache,**{"pred_"+s:v for s,v in zip(("train","val","test"),pm)},
                 **{"feature_"+s:v for s,v in zip(("train","val","test"),fm)})
        receipt.write_text(json.dumps(dict(sha256=file_sha(cache),seconds=time.time()-t0),indent=2)+"\n",encoding="utf-8")
        print("NETWORK_TRAINED",mode,round(time.time()-t0,1),flush=True)
    networks[mode]=(pm,fm)
    err_fn=((lambda v:rel(v,Yval)) if mode=="flat" else
            (lambda v:rel(v*w_z,Yval*w_z)) if mode=="wnum" else (lambda v:rad(v,Yval)))
    ridge[mode],ridge_hyper[mode]=ridge_head(*fm,err_fn,f"ridge_{mode}")

xs=(Xtr-Xtr.mean(0))/(Xtr.std(0)+1e-9)
als,*_=np.linalg.lstsq(xs,Ytr,rcond=None)
relevance=np.linalg.norm(als,axis=1);w_ard=relevance/relevance.mean()
kf=kernel_flow_predictions(args.band)
r_true=to_radiance(Yte,recon)
den_red=np.linalg.norm(Yte,axis=1);den_rad=np.linalg.norm(r_true,axis=1)
if np.any(den_red<=0) or np.any(den_rad<=0):raise RuntimeError("Invalid target norm")
all_summaries={};all_errors={}
for scenario,grid in GRIDS.items():
    args.scales=grid["scales"];args.nuggets=grid["nuggets"]
    report_path=OUT/f"{scenario}.json";errors_path=OUT/f"{scenario}_errors.npz"
    if report_path.exists() or errors_path.exists():
        if not report_path.exists() or not errors_path.exists():raise RuntimeError("Partial grid result")
        report=json.loads(report_path.read_text(encoding="utf-8"))
        if report["errors_sha256"]!=file_sha(errors_path):raise RuntimeError("Grid result changed")
        with np.load(errors_path) as z:
            all_errors[scenario]={k:z[k] for k in z.files}
        all_summaries[scenario]=report
        print("VERIFIED_GRID_CACHE",scenario,flush=True);continue
    start_grid=time.time();hyper={};pv={};pt={"kernel_flow":kf}
    for label,weight in (("kernel_raw",None),("kernel_ard",w_ard)):
        pred,h=matern_head(Xtr,Xval,Xte,lambda v:rel(v,Yval),label,w=weight)
        pv[label],pt[label]=pred[1],pred[2];hyper[label]=h
    candidates=[];ridge_candidates=[]
    for mode in ("flat","wnum","radx"):
        pm,fm=networks[mode];mname="mean_"+mode;dname="dkr_"+mode;rname="ridge_"+mode
        pv[mname],pt[mname]=pm[1],pm[2]
        err_fn=((lambda v:rel(v,Yval)) if mode=="flat" else
                (lambda v:rel(v*w_z,Yval*w_z)) if mode=="wnum" else (lambda v:rad(v,Yval)))
        pred,h=matern_head(*fm,err_fn,dname)
        pv[dname],pt[dname]=pred[1],pred[2];hyper[dname]=h
        pv[rname],pt[rname]=ridge[mode][1],ridge[mode][2];hyper[rname]=ridge_hyper[mode]
        candidates.extend((mname,dname));ridge_candidates.append(rname)
    winners={}
    for label,names in (("combined",candidates),("combined_plus_ridge",candidates+ridge_candidates)):
        chosen=np.argmin(np.array([np.mean((pv[name]-Yval)**2,axis=0) for name in names]),axis=0)
        prediction=np.empty_like(Yte)
        for j,index in enumerate(chosen):prediction[:,j]=pt[names[index]][:,j]
        pt[label]=prediction;winners[label]=[names[i] for i in chosen]
    errors={};metrics={}
    for name,pred in pt.items():
        red=np.linalg.norm(pred-Yte,axis=1)/den_red
        raderr=np.linalg.norm(to_radiance(pred,recon)-r_true,axis=1)/den_rad
        if not np.isfinite(red).all() or not np.isfinite(raderr).all():
            raise RuntimeError("Nonfinite final score: "+name)
        errors[name+"_reduced"]=red;errors[name+"_radiance"]=raderr
        metrics[name]=dict(reduced=float(red.mean()),radiance=float(raderr.mean()))
    np.savez(errors_path,**errors)
    np.savez(OUT/f"{scenario}_predictions.npz",Yval=Yval,Yte=Yte,
             **{"val_"+name:v for name,v in pv.items()},**{"test_"+name:v for name,v in pt.items()})
    report=dict(scenario=scenario,metrics=metrics,hyper=hyper,winners=winners,
                seconds=time.time()-start_grid,errors_sha256=file_sha(errors_path))
    report_path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    all_errors[scenario]=errors;all_summaries[scenario]=report
    print("GRID_COMPLETE",scenario,json.dumps(metrics),flush=True)
paired={}
for name in all_errors["recorded_grid"]:
    d=all_errors["expanded_grid"][name]-all_errors["recorded_grid"][name]
    paired[name]=dict(mean=float(d.mean()),p05=float(np.quantile(d,.05)),p95=float(np.quantile(d,.95)),
                      improved_fraction=float(np.mean(d<0)))
out=dict(identity=identity,seconds=time.time()-t_all,results=all_summaries,
         expanded_minus_recorded=paired,
         interpretation="Retrospective validation-selected grid sensitivity; both grids use the same trained networks, split and calibration rows. Values are fractions, not percentages.")
(OUT/"grid_sensitivity.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
print("GRID_SENSITIVITY_COMPLETE",args.band,args.seed,round(time.time()-t_all,1),flush=True)
