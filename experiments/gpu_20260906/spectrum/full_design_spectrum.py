"""Matern-5/2 Gram spectrum and effective dimension of the FULL 19000-row
structural-mechanics training design, with the exact recipe of the paper's
6000-row diagnostic (campaign/spectrum_diagnostic.py, record
campaign/collected/spectrum_20260906/result.json):

  rows            = idx_train[perm[1000:]] with perm = default_rng(0).permutation(20000)
  standardization = training mean / std (+1e-12) of the 19000 rows
  length scale    = 4 x median pairwise distance on a 2000-row subsample
                    drawn by default_rng(0).choice(19000, 2000, replace=False)
  kernel          = Matern 5/2, unit variance
  d_eff(lambda)   = sum_i mu_i / (mu_i + n lambda),  n = number of design rows

Routes (each reported separately, none feeds another):
  cpu64_eig      numpy/scipy float64 Gram, LAPACK eigenvalues (driver evd)
  cpu64_chol     float64 Cholesky of K + n lambda I, d_eff = n - n lambda ||L^-1||_F^2
  gpu32_eig      torch float32 Gram on the card, cuSOLVER eigenvalues (fast, approximate)
  gpu64_chol     torch float64 Cholesky trace on the card, if VRAM allows
Controls:
  the standardized rows at the record's 6000 subsample positions must equal
  the packet's design.npz to machine precision, and the 6000-row d_eff
  recomputed here from that file must reproduce 103.11885064181605.

Usage: python full_design_spectrum.py [--stage all|cpu|gpu] [--threads 4]
Outputs under spectrum/full_design_19000/: result.json, spectrum_full.npz,
design_full.npz, status.json, and the comparison figure.
"""
import argparse, hashlib, json, os, platform, sys, time
from datetime import datetime, timezone
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(name, "4")
import numpy as np
import scipy
from scipy.linalg import cho_factor, eigvalsh, solve_triangular
from threadpoolctl import threadpool_limits, threadpool_info

W = Path("C:/Users/owner/GOAL20H_20260906/nmkc_gpu_experiments")
DATA = W / "data" / "structmech"
REC = W / "records" / "campaign" / "collected" / "spectrum_20260906"
OUT = W / "spectrum" / "full_design_19000"
OUT.mkdir(parents=True, exist_ok=True)
NUGGET = 1e-5
LAM_GRID = np.unique(np.r_[np.logspace(-9, -1, 80), 1e-5])


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def m52_from_r2(r2):
    radial = np.sqrt(5 * r2)
    return (1 + radial + (5 / 3) * r2) * np.exp(-radial)


def gram_blocks(z, ls, block=2000, dtype=np.float64):
    """Float64 Gram matrix assembled in row blocks (bounded temporaries)."""
    n = len(z)
    v = np.sum(z * z, axis=1)
    K = np.empty((n, n), dtype=dtype)
    for s in range(0, n, block):
        e = min(s + block, n)
        d2 = np.maximum(v[s:e, None] + v[None, :] - 2 * z[s:e] @ z.T, 0)
        K[s:e] = m52_from_r2(d2 / ls ** 2)
    return K


def deff(ev, n, lam):
    return float(np.sum(ev / (ev + n * lam)))


class Record:
    def __init__(self, path):
        self.path = path
        self.d = dict(kind="full_design_spectrum_diagnostic", agent="claude-f5-gputrain-dd39",
                      started_at=datetime.now(timezone.utc).isoformat(), source_sha256=sha(Path(__file__)),
                      n_training=19000, n_full_design=19000, split_seed=0, median_subset_size=2000,
                      median_multiplier=4.0, nugget=NUGGET,
                      kernel="Matern 5/2 on training-standardized recorded loads",
                      host=platform.node(), python=platform.python_version(), numpy=np.__version__,
                      scipy=scipy.__version__, stages=[], status="RUNNING")
        if path.exists():
            old = json.loads(path.read_text(encoding="utf-8"))
            for k, v in old.items():
                if k not in ("stages", "status", "started_at"):
                    self.d[k] = v
            self.d["stages"] = old.get("stages", [])
        self.t0 = time.time()

    def stage(self, name, **kw):
        self.d["stages"].append(dict(stage=name, at=datetime.now(timezone.utc).isoformat(),
                                     elapsed_seconds=time.time() - self.t0, **kw))
        self.save()
        print(f"[{time.time()-self.t0:8.1f}s] {name} {kw if kw else ''}", flush=True)

    def save(self):
        self.path.write_text(json.dumps(self.d, indent=2) + "\n", encoding="utf-8")


def load_design(rec):
    paths = {n: DATA / (n + ".npy") for n in ("loads", "idx_train")}
    rec.d["input_files"] = {n: dict(path=str(p), sha256=sha(p)) for n, p in paths.items()}
    loads = np.load(paths["loads"], allow_pickle=False)
    train = np.load(paths["idx_train"], allow_pickle=False)
    assert loads.shape == (40000, 41) and train.shape == (20000,)
    assert np.isfinite(loads).all() and len(np.unique(train)) == 20000
    perm = np.random.default_rng(0).permutation(len(train))
    rows = train[perm[1000:]]
    x = loads[rows].astype(np.float64)
    mean = x.mean(0); scale = x.std(0) + 1e-12
    z = (x - mean) / scale
    rng = np.random.default_rng(0)
    median_positions = rng.choice(len(z), 2000, replace=False)
    zm = z[median_positions]
    vm = np.sum(zm * zm, axis=1)
    dmed = np.maximum(vm[:, None] + vm[None, :] - 2 * zm @ zm.T, 0)
    median = float(np.sqrt(np.median(dmed[np.triu_indices(2000, 1)])))
    positions6000 = rng.choice(len(z), 6000, replace=False)
    ls = 4 * median
    rec.d.update(median_distance=median, length_scale=ls, n_rows=int(len(z)))
    # control: the packet's design file must be reproduced exactly at its positions
    packet = np.load(REC / "design.npz", allow_pickle=False)
    packet_rec = json.loads((REC / "result.json").read_text(encoding="utf-8"))
    ctrl = dict(
        packet_design_sha256=sha(REC / "design.npz"),
        packet_design_sha256_matches_record=(sha(REC / "design.npz") == packet_rec["design_sha256"]),
        training_rows_equal=bool(np.array_equal(packet["training_rows"], rows)),
        subsample_positions_equal=bool(np.array_equal(packet["subsample_positions"], positions6000)),
        median_rows_equal=bool(np.array_equal(packet["median_rows"], rows[median_positions])),
        training_mean_max_abs_diff=float(np.max(np.abs(packet["training_mean"] - mean))),
        training_scale_max_abs_diff=float(np.max(np.abs(packet["training_scale"] - scale))),
        standardized_loads_max_abs_diff=float(np.max(np.abs(packet["standardized_loads"] - z[positions6000]))),
        median_distance_record=packet_rec["median_distance"],
        median_distance_abs_diff=abs(packet_rec["median_distance"] - median),
        length_scale_record=packet_rec["length_scale"])
    rec.d["control_design_reproduction"] = ctrl
    assert ctrl["training_rows_equal"] and ctrl["subsample_positions_equal"] and ctrl["median_rows_equal"]
    assert ctrl["standardized_loads_max_abs_diff"] < 1e-12 and ctrl["median_distance_abs_diff"] < 1e-12
    np.savez_compressed(OUT / "design_full.npz", standardized_loads=z, training_mean=mean, training_scale=scale,
                        training_rows=rows, median_rows=rows[median_positions], length_scale=ls)
    rec.d["design_full_sha256"] = sha(OUT / "design_full.npz")
    return z, ls, positions6000, packet, packet_rec


def cpu_routes(rec, z, ls, positions6000, packet, packet_rec, threads):
    n = len(z)
    with threadpool_limits(threads):
        # ---- control: recompute the 6000-row diagnostic from the packet's own rows
        rec.stage("control 6000-row from packet design.npz")
        Ks = gram_blocks(packet["standardized_loads"].astype(np.float64), ls)
        evs = eigvalsh(Ks, check_finite=True, driver="evd", overwrite_a=True)
        del Ks
        d6 = deff(evs, 6000, NUGGET)
        packet_spec = np.load(REC / "spectrum.npz", allow_pickle=False)
        rec.d["control_6000"] = dict(
            recomputed_effective_dimension=d6,
            record_spectral_effective_dimension=packet_rec["spectral_effective_dimension"],
            abs_diff=abs(d6 - packet_rec["spectral_effective_dimension"]),
            eigenvalue_max_abs_diff_vs_packet=float(np.max(np.abs(np.sort(evs) - np.sort(packet_spec["eigenvalues"])))),
            eigenvalue_max=float(evs[-1]), eigenvalue_min=float(evs[0]), trace=float(evs.sum()),
            top8_trace_fraction=float(np.sort(evs)[::-1][:8].sum() / evs.sum()))
        rec.save()
        # ---- full design, float64 Gram
        rec.stage("full Gram float64 (CPU, blocks)")
        K = gram_blocks(z, ls)
        tr = float(np.trace(K)); fro = float(np.einsum("ij,ij->", K, K))
        rec.d.update(gram_trace=tr, gram_frobenius_squared=fro)
        # ---- independent route first: Cholesky trace of A = K + n lambda I
        rec.stage("cpu64 Cholesky trace route")
        A = K.copy()
        A.flat[::n + 1] += n * NUGGET
        L, lower = cho_factor(A, lower=True, overwrite_a=True, check_finite=False)
        del A
        # solve_triangular(lower=True) reads only the lower triangle (LAPACK dtrtrs UPLO='L'),
        # so the garbage above the diagonal left by cho_factor is never referenced
        inv_fro = 0.0
        blk = 1000
        for s in range(0, n, blk):
            e = min(s + blk, n)
            rhs = np.zeros((n - s, e - s)); rhs[np.arange(e - s), np.arange(e - s)] = 1.0
            X = solve_triangular(L[s:, s:], rhs, lower=True, check_finite=False)
            inv_fro += float(np.einsum("ij,ij->", X, X))
            del X
        del L
        direct = n - n * NUGGET * inv_fro
        rec.d["cpu64_chol"] = dict(trace_inverse=inv_fro, direct_effective_dimension=direct)
        rec.save()
        # ---- eigenvalues
        rec.stage("cpu64 eigenvalues (LAPACK evd)")
        ev = eigvalsh(K, check_finite=True, driver="evd", overwrite_a=True)
        del K
        assert np.isfinite(ev).all()
        ev = np.sort(ev)
        curve = np.array([deff(ev, n, l) for l in LAM_GRID])
        spectral = deff(ev, n, NUGGET)
        top = ev[::-1]
        cum = np.cumsum(top)
        rec.d["cpu64_eig"] = dict(
            eigenvalue_min=float(ev[0]), eigenvalue_max=float(ev[-1]), n_negative=int(np.sum(ev < 0)),
            trace_from_eigenvalues=float(ev.sum()), trace_rel_diff=abs(float(ev.sum()) - tr) / tr,
            frobenius_rel_diff=abs(float(ev @ ev) - fro) / fro,
            spectral_effective_dimension=spectral,
            effective_dimension_same_shift_as_6000=deff(ev, 6000, NUGGET),
            top8_eigenvalues=top[:8].tolist(), top8_eigenvalues_over_n=(top[:8] / n).tolist(),
            top8_trace_fraction=float(cum[7] / ev.sum()),
            modes_for_99_percent_trace=int(np.searchsorted(cum, 0.99 * ev.sum()) + 1),
            modes_for_999_percent_trace=int(np.searchsorted(cum, 0.999 * ev.sum()) + 1),
            trace_check_absolute_difference=abs(direct - spectral))
        rec.d["cpu64_eig"]["agrees_with_chol_1e-8"] = bool(abs(direct - spectral) < 1e-8 * max(1.0, spectral))
        np.savez(OUT / "spectrum_full.npz", eigenvalues=ev, lambda_grid=LAM_GRID, effective_dimension=curve,
                 eigenvalues_6000_control=np.sort(evs))
        rec.d["spectrum_full_sha256"] = sha(OUT / "spectrum_full.npz")
        rec.d["thread_pools_cpu"] = [dict(api=t["internal_api"], threads=t["num_threads"]) for t in threadpool_info()]
        rec.save()


def gpu_routes(rec, z, ls, which=("fp32", "fp64")):
    import torch
    n = len(z)
    if not torch.cuda.is_available():
        rec.d["gpu"] = dict(skipped="no CUDA"); rec.save(); return
    free, total = torch.cuda.mem_get_info()
    rec.stage("gpu: VRAM read", free_mb=free // 2 ** 20, total_mb=total // 2 ** 20)
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    rec.d["gpu"] = dict(device=props.name, torch=torch.__version__, cuda=torch.version.cuda,
                        vram_free_mb_at_start=free // 2 ** 20)

    def gram_torch(dtype, block=2000):
        zt = torch.tensor(z, dtype=dtype, device=dev)
        v = (zt * zt).sum(1)
        K = torch.empty((n, n), dtype=dtype, device=dev)
        for s in range(0, n, block):
            e = min(s + block, n)
            d2 = torch.clamp(v[s:e, None] + v[None, :] - 2 * zt[s:e] @ zt.T, min=0)
            r2 = d2 / ls ** 2
            radial = torch.sqrt(5 * r2)
            K[s:e] = (1 + radial + (5 / 3) * r2) * torch.exp(-radial)
            del d2, r2, radial
        del zt, v
        return K

    # ---- monolithic cuSOLVER routes are DISABLED on this card. One torch.linalg.eigvalsh of
    # the 19000x19000 float32 Gram took the display driver down at 10:00 on 6 Sep 2026 (exit -1,
    # no Python exception, every CUDA context on the card lost). gpu_blocked_routes.py holds the
    # routes that never launch a long kernel (chunked float64 Cholesky, subspace iteration).
    rec.d["gpu32_eig"] = dict(skipped="monolithic cuSOLVER eigvalsh disabled after the 10:00 driver reset; see gpu_blocked_routes.py")
    rec.d["gpu64_chol_monolithic"] = dict(skipped="monolithic torch.linalg.cholesky disabled; see gpu_blocked_routes.py")
    rec.save()


def summarize(rec):
    """Merge the CPU record (result.json) and the GPU record (result_gpu.json), which are
    written by separate processes, into one view before comparing routes."""
    d = dict(rec.d)
    for other in ("result.json", "result_gpu.json"):
        f = OUT / other
        if f != rec.path and f.exists():
            for k, v in json.loads(f.read_text(encoding="utf-8")).items():
                d.setdefault(k, v)
    rec.d["merged_from"] = [str(f.name) for f in (OUT / "result.json", OUT / "result_gpu.json") if f.exists()]
    s = {}
    if "cpu64_eig" in d and "cpu64_chol" in d:
        s["effective_dimension_full_19000"] = d["cpu64_eig"]["spectral_effective_dimension"]
        s["effective_dimension_full_19000_cholesky"] = d["cpu64_chol"]["direct_effective_dimension"]
        s["effective_dimension_6000_record"] = 103.11885064181605
        s["ratio_full_over_6000"] = s["effective_dimension_full_19000"] / 103.11885064181605
    if "gpu32_eig" in d and "spectral_effective_dimension" in d["gpu32_eig"] and "cpu64_eig" in d:
        s["gpu32_minus_cpu64_effective_dimension"] = (d["gpu32_eig"]["spectral_effective_dimension"]
                                                     - d["cpu64_eig"]["spectral_effective_dimension"])
        ev64 = np.load(OUT / "spectrum_full.npz")["eigenvalues"]
        ev32 = np.load(OUT / "eigenvalues_gpu_fp32.npy")
        s["gpu32_vs_cpu64_max_abs_eigenvalue_diff"] = float(np.max(np.abs(ev64 - ev32)))
        s["gpu32_vs_cpu64_top8_max_rel_diff"] = float(np.max(np.abs(ev64[::-1][:8] - ev32[::-1][:8]) / ev64[::-1][:8]))
    if "gpu64_chol" in d and "direct_effective_dimension" in d["gpu64_chol"] and "cpu64_chol" in d:
        s["gpu64_minus_cpu64_cholesky_effective_dimension"] = (d["gpu64_chol"]["direct_effective_dimension"]
                                                              - d["cpu64_chol"]["direct_effective_dimension"])
    rec.d["summary"] = s
    rec.save()
    if s:
        (OUT / "summary.json").write_text(json.dumps(dict(summary=s, cpu64_eig=d.get("cpu64_eig"), cpu64_chol=d.get("cpu64_chol"),
                                                          gpu32_eig=d.get("gpu32_eig"), gpu64_chol=d.get("gpu64_chol"),
                                                          control_6000=d.get("control_6000"), gpu=d.get("gpu"),
                                                          length_scale=d.get("length_scale"), median_distance=d.get("median_distance"),
                                                          input_files=d.get("input_files"), design_full_sha256=d.get("design_full_sha256"),
                                                          spectrum_full_sha256=d.get("spectrum_full_sha256")), indent=2) + "\n", encoding="utf-8")
    return d


def figure(rec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    full = np.load(OUT / "spectrum_full.npz")
    sub = np.load(REC / "spectrum.npz")
    ev_full, lam, curve = full["eigenvalues"], full["lambda_grid"], full["effective_dimension"]
    ev_sub, lam_s, curve_s = sub["eigenvalues"], sub["lambda_grid"], sub["effective_dimension"]
    n, m = len(ev_full), len(ev_sub)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42})
    C = ["#3b5bdb", "#e8590c", "#2b8a3e", "#862e9c", "#495057"]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    ax = axes[0]
    ax.semilogy(np.arange(1, n + 1), np.maximum(ev_full[::-1] / n, 1e-15), color=C[0], lw=1.2, label="full design, $K/19000$")
    ax.semilogy(np.arange(1, m + 1), np.maximum(ev_sub[::-1] / m, 1e-15), color=C[1], lw=1.0, ls="--", label="6000-row subsample, $K_S/6000$")
    ax.set(xlabel="eigenvalue index", ylabel="eigenvalue of $K/n$", title="Matern spectrum, normalized by $n$", xlim=(1, n))
    ax.legend(frameon=False)
    ax = axes[1]
    ax.semilogx(lam, curve, color=C[0], lw=1.3, label="full design ($n=19000$)")
    ax.semilogx(lam_s, curve_s, color=C[1], lw=1.1, ls="--", label="subsample ($n=6000$)")
    d_full = json.loads((OUT / "result.json").read_text(encoding="utf-8"))["cpu64_eig"]["spectral_effective_dimension"]
    ax.axvline(NUGGET, color=C[4], ls=":", lw=0.9)
    ax.plot(NUGGET, d_full, "o", color=C[0], ms=4)
    ax.plot(NUGGET, 103.11885064181605, "s", color=C[1], ms=4)
    ax.annotate(r"$\lambda=10^{-5}$: " + f"{d_full:.1f} (full)\n{103.12:.2f} (subsample)", xy=(NUGGET, d_full),
                xytext=(0.5, 0.72), textcoords="axes fraction", fontsize=8,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=C[4]))
    ax.set(xlabel=r"nugget $\lambda$", ylabel=r"$d_{\rm eff}(\lambda)$", title="Effective dimension of the smoother", ylim=(0, None))
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"spectra_full_vs_subsample.{ext}", metadata={"CreationDate": None, "ModDate": None} if ext == "pdf" else None)
    plt.close(fig)
    rec.d["figure_sha256"] = {ext: sha(OUT / f"spectra_full_vs_subsample.{ext}") for ext in ("pdf", "png")}
    rec.save()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all", choices=["all", "cpu", "gpu", "figure"])
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    rec = Record(OUT / ("result_gpu.json" if a.stage == "gpu" else "result.json"))
    rec.stage("input")
    z, ls, pos, packet, packet_rec = load_design(rec)
    if a.stage in ("all", "cpu"):
        cpu_routes(rec, z, ls, pos, packet, packet_rec, a.threads)
    if a.stage in ("all", "gpu"):
        gpu_routes(rec, z, ls)
    merged = summarize(rec)
    if a.stage in ("all", "figure", "cpu", "gpu") and (OUT / "spectrum_full.npz").exists() and "cpu64_eig" in merged:
        figure(rec)
    rec.d["status"] = "COMPLETE" if "cpu64_eig" in merged else "PARTIAL"
    rec.d["finished_at"] = datetime.now(timezone.utc).isoformat()
    rec.d["elapsed_seconds"] = time.time() - rec.t0
    rec.d["result_sha256_of_source"] = sha(Path(__file__))
    rec.save()
    print(json.dumps(rec.d.get("summary", {}), indent=1), flush=True)
    print("STATUS", rec.d["status"], flush=True)


if __name__ == "__main__":
    main()
