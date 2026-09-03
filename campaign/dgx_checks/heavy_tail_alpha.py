"""Experiment 10 of the five-paper plan: heavy-tailed self-regularization (Martin and Mahoney) of the trained
members, read against their test errors.

For every saved checkpoint of a seed (the MLPs, the refiner, the FNO and the UNet of the complete-schedule
campaign): the empirical spectral density of each weight matrix (convolutions reshaped to out x in*k), the
power-law exponent alpha fitted to the tail by the Clauset-Shalizi-Newman maximum-likelihood estimator over
the largest 40 percent of eigenvalues (the weightwatcher convention), the log spectral norm, and the
stable rank; per checkpoint the mean alpha over layers and the alpha-hat metric (alpha weighted by the log
spectral norm). Written next to each member's test error so the correlation across members and seeds can
be read off the collected records.

    python heavy_tail_alpha.py --seed 0   (ROOT ~/nmkc2; writes results/heavy_tail_alpha_s<seed>.json)
"""
import argparse, glob, json, os, pathlib, time
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
ROOT = pathlib.Path(os.environ.get("NMKC_ROOT", os.path.expanduser("~/nmkc2")))
RUNS = ROOT / "seeds" / f"sm_s{args.seed}" / "runs"
t0 = time.time()


def alpha_hill(ev, frac=0.4):
    """Power-law tail exponent by the maximum-likelihood (Hill/CSN) estimator on the top `frac` of eigenvalues."""
    ev = np.sort(ev[ev > 1e-12])
    k = max(int(frac * len(ev)), 5)
    tail = ev[-k:]; xmin = tail[0]
    return float(1.0 + k / np.sum(np.log(tail / xmin)))


out = dict(seed=args.seed, checkpoints={})
for f in sorted(glob.glob(str(RUNS / "_ckpt_*.pt"))):
    name = os.path.basename(f)[6:-3]
    try:
        sd = torch.load(f, map_location="cpu", weights_only=False)
    except Exception as e:
        out["checkpoints"][name] = dict(error=str(e)[:120]); continue
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict): sd = sd["model"]
    if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
    layers = []
    for k, v in sd.items():
        if not torch.is_tensor(v) or v.ndim < 2 or v.numel() < 400: continue
        W = v.detach().double()
        if torch.is_complex(W): W = torch.view_as_real(W).flatten(2).mean(-1)   # spectral weights: real average
        W = W.reshape(W.shape[0], -1).numpy()
        if min(W.shape) < 10: continue
        s = np.linalg.svd(W, compute_uv=False); ev = s ** 2
        layers.append(dict(layer=k, shape=list(W.shape), alpha=alpha_hill(ev), log_spectral_norm=float(np.log10(ev.max())),
                           stable_rank=float(ev.sum() / ev.max())))
    if not layers:
        out["checkpoints"][name] = dict(error="no weight matrices"); continue
    al = np.array([l["alpha"] for l in layers]); ln = np.array([l["log_spectral_norm"] for l in layers])
    out["checkpoints"][name] = dict(n_layers=len(layers), alpha_mean=float(al.mean()), alpha_median=float(np.median(al)),
                                    alpha_hat=float((al * ln).sum() / max(ln.sum(), 1e-12)), log_norm_mean=float(ln.mean()),
                                    stable_rank_mean=float(np.mean([l["stable_rank"] for l in layers])), layers=layers)
    print(f"  {name:12s} layers {len(layers):3d} alpha mean {al.mean():.2f} median {np.median(al):.2f} alpha-hat {out['checkpoints'][name]['alpha_hat']:.2f} [{time.time()-t0:.0f}s]", flush=True)
# the members' test errors, from their run records, for the correlation
errs = {}
for f in glob.glob(str(RUNS / "*.json")):
    try:
        J = json.load(open(f))
    except Exception:
        continue
    if isinstance(J, dict) and "test" in J and isinstance(J["test"], (int, float)):
        errs[os.path.basename(f)[:-5]] = float(J["test"])
out["member_test_errors"] = errs
out["minutes"] = round((time.time() - t0) / 60, 1)
(ROOT / "results").mkdir(exist_ok=True)
json.dump(out, open(ROOT / "results" / f"heavy_tail_alpha_s{args.seed}.json", "w"), indent=1)
print("wrote", ROOT / "results" / f"heavy_tail_alpha_s{args.seed}.json", flush=True)
