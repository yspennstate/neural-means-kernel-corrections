"""GPU float64 routes for the full-design spectrum that never launch a long
kernel (a single cuSOLVER eigvalsh of the 19000x19000 Gram matrix took the
display driver down at 10:00 on 6 Sep; on a WDDM laptop GPU every kernel must
stay under the ~2 s TDR limit).

  gpu64_chol  in-place blocked Cholesky of A = K + n lambda I in float64 on the
              card, every GEMM/TRSM chunked so one launch does at most
              --chunk-flops (default 6e10, ~0.2 s at 0.3 TFLOPS fp64), then
              ||L^-1||_F^2 by blocked forward substitution; d_eff = n - n lambda ||L^-1||_F^2
  gpu64_top   top-k eigenvalues of K by float64 subspace iteration with
              Rayleigh-Ritz (matmuls only), for the comparison of leading modes
  gpu32_top   the same in float32, to state what fp32 can and cannot deliver

Writes spectrum/full_design_19000/result_gpu.json (fresh) and merges with
result.json into summary.json through full_design_spectrum.summarize.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import full_design_spectrum as fds  # noqa: E402

OUT = fds.OUT; NUGGET = fds.NUGGET
# GPU duty cycle (GPU_DUTY, default 0.8): after every chunk the caller waits for the card and idles
# (1-duty)/duty of the chunk's time, so a utilization sample reads about 100*duty. The owner's
# CrashGuard kills python GPU clients at a sample >= 95; his order is 80 percent.
DUTY = min(max(float(os.environ.get("GPU_DUTY", "0.8")), 0.05), 1.0)


def pace(t_start):
    """Call after a chunk of GPU work that began at perf_counter() == t_start."""
    if DUTY < 1.0:
        torch.cuda.synchronize()
        time.sleep((time.perf_counter() - t_start) * (1.0 - DUTY) / DUTY)
    return time.perf_counter()


def bench_fp64(dev):
    a = torch.randn(3000, 3000, dtype=torch.float64, device=dev)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(3):
        b = a @ a
    torch.cuda.synchronize()
    dt = (time.time() - t) / 3
    return 2 * 3000 ** 3 / dt, dt


def gram_torch(z, ls, dev, dtype, block=2000):
    n = len(z)
    zt = torch.tensor(z, dtype=dtype, device=dev)
    v = (zt * zt).sum(1)
    K = torch.empty((n, n), dtype=dtype, device=dev)
    tp = time.perf_counter()
    for s in range(0, n, block):
        e = min(s + block, n)
        d2 = torch.clamp(v[s:e, None] + v[None, :] - 2 * zt[s:e] @ zt.T, min=0)
        r2 = d2 / ls ** 2
        radial = torch.sqrt(5 * r2)
        K[s:e] = (1 + radial + (5 / 3) * r2) * torch.exp(-radial)
        del d2, r2, radial
        tp = pace(tp)
    del zt, v
    return K


def blocked_cholesky_inplace(A, b, chunk_flops, sync_every=True):
    """Right-looking blocked Cholesky in place (lower). Kernel launches are kept
    short by chunking the trailing update over row groups."""
    n = A.shape[0]
    tp = time.perf_counter()
    for j in range(0, n, b):
        e = min(j + b, n)
        Ljj = torch.linalg.cholesky(A[j:e, j:e])
        A[j:e, j:e] = Ljj
        tp = pace(tp)
        if e < n:
            # panel: A[e:, j:e] = A[e:, j:e] @ inv(Ljj)^T, chunked over rows
            rows = max(1, int(chunk_flops // ((e - j) ** 2)))
            for s in range(e, n, rows):
                t = min(s + rows, n)
                # rows s:t of the panel: X = A[s:t, j:e] Ljj^{-T}, i.e. Ljj X^T = A[s:t, j:e]^T
                A[s:t, j:e] = torch.linalg.solve_triangular(Ljj, A[s:t, j:e].T, upper=False).T
                tp = pace(tp)
            # trailing update A[e:, e:] -= P P^T (lower part suffices), chunked over row groups
            P = A[e:, j:e]
            rows = max(1, int(chunk_flops // (2 * (n - e) * (e - j))))
            for s in range(0, n - e, rows):
                t = min(s + rows, n - e)
                A[e + s:e + t, e:e + t] -= P[s:t] @ P[:t].T
                tp = pace(tp)
            if sync_every:
                torch.cuda.synchronize()
    return A


def inv_frobenius_sq(L, b, chunk_flops):
    """||L^-1||_F^2 by blocked forward substitution on identity column blocks,
    using only the lower triangle of L."""
    n = L.shape[0]
    total = 0.0
    tp = time.perf_counter()
    for s in range(0, n, b):
        e = min(s + b, n)
        m = n - s
        X = torch.zeros((m, e - s), dtype=L.dtype, device=L.device)
        X[:e - s] = torch.linalg.solve_triangular(L[s:e, s:e], torch.eye(e - s, dtype=L.dtype, device=L.device), upper=False)
        # forward substitution over the remaining row blocks
        for r in range(e, n, b):
            q = min(r + b, n)
            # rhs for rows r:q = -L[r:q, s:r] @ X[:r-s], chunked to bound one launch
            rhs = torch.zeros((q - r, e - s), dtype=L.dtype, device=L.device)
            cols = max(1, int(chunk_flops // (2 * (q - r) * (e - s))))
            for c in range(s, r, cols):
                d = min(c + cols, r)
                rhs -= L[r:q, c:d] @ X[c - s:d - s]
                tp = pace(tp)
            X[r - s:q - s] = torch.linalg.solve_triangular(L[r:q, r:q], rhs, upper=False)
            tp = pace(tp)
        total += float((X * X).sum().item())
        torch.cuda.synchronize()
    return total


def subspace_top(K, k, iters, dtype):
    n = K.shape[0]
    g = torch.Generator(device=K.device); g.manual_seed(0)
    Q, _ = torch.linalg.qr(torch.randn(n, k, dtype=dtype, device=K.device, generator=g))
    tp = time.perf_counter()
    for _ in range(iters):
        Z = K @ Q
        Q, _ = torch.linalg.qr(Z)
        tp = pace(tp)
    T = Q.T @ (K @ Q)
    T = 0.5 * (T + T.T)
    ev = torch.linalg.eigvalsh(T)
    return np.sort(ev.double().cpu().numpy())[::-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--block", type=int, default=1000)
    p.add_argument("--chunk-flops", type=float, default=6e10)
    p.add_argument("--topk", type=int, default=32)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--skip-chol", action="store_true")
    p.add_argument("--fp64-top", action="store_true", help="also run the float64 subspace iteration (2.9 GB)")
    p.add_argument("--min-free-mb", type=int, default=1500, help="free VRAM that must remain after my allocation")
    a = p.parse_args()
    def vram_guard(need_bytes):
        free = torch.cuda.mem_get_info()[0]
        if free - need_bytes < a.min_free_mb * 2 ** 20:
            raise SystemExit(f"VRAM guard: free {free//2**20} MB minus need {need_bytes//2**20} MB leaves less than "
                             f"{a.min_free_mb} MB; refusing (a near-full card resets the driver for everyone)")
    rec = fds.Record(OUT / "result_gpu.json")
    rec.d["stages"] = []
    rec.stage("input")
    z, ls, pos, packet, packet_rec = fds.load_design(rec)
    n = len(z)
    dev = torch.device("cuda")
    free, total = torch.cuda.mem_get_info()
    flops, dt = bench_fp64(dev)
    rec.d["gpu"] = dict(device=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
                        vram_free_mb_at_start=free // 2 ** 20, fp64_gemm_tflops=flops / 1e12, fp64_gemm_3000_seconds=dt,
                        chunk_flops=a.chunk_flops, expected_max_kernel_seconds=a.chunk_flops / flops, block=a.block,
                        note="no monolithic cuSOLVER call; every launch chunked below the WDDM TDR limit")
    rec.stage("gpu: bench", fp64_tflops=round(flops / 1e12, 3))
    # ---- top eigenvalues, fp64 and fp32 (matmul-only subspace iteration)
    tops = [(torch.float32, "gpu32_top")] + ([(torch.float64, "gpu64_top")] if a.fp64_top else [])
    for dtype, tag in tops:
        vram_guard(n * n * (8 if dtype == torch.float64 else 4))
        t0 = time.time()
        rec.stage(f"{tag}: Gram + subspace iteration")
        K = gram_torch(z, ls, dev, dtype)
        tr = float(K.diagonal().sum().item())
        fro = float(sum((K[i:i + 1000] * K[i:i + 1000]).sum().item() for i in range(0, n, 1000))) if dtype == torch.float64 else None
        top = subspace_top(K, a.topk, a.iters, dtype)
        del K; torch.cuda.empty_cache()
        rec.d[tag] = dict(seconds=time.time() - t0, gram_trace=tr, gram_frobenius_squared=fro, k=a.topk, iters=a.iters,
                          top_eigenvalues=top[:16].tolist(), top8_eigenvalues_over_n=(top[:8] / n).tolist(),
                          top8_trace_fraction=float(top[:8].sum() / tr), top16_trace_fraction=float(top[:16].sum() / tr))
        rec.save()
    # ---- fp64 blocked Cholesky trace
    if not a.skip_chol:
        t0 = time.time()
        vram_guard(n * n * 8 + 200 * 2 ** 20)
        rec.stage("gpu64_chol: Gram fp64 + blocked Cholesky", vram_free_mb=torch.cuda.mem_get_info()[0] // 2 ** 20)
        A = gram_torch(z, ls, dev, torch.float64)
        A.diagonal().add_(n * NUGGET)
        blocked_cholesky_inplace(A, a.block, a.chunk_flops)
        t1 = time.time()
        rec.stage("gpu64_chol: inverse Frobenius norm", chol_seconds=round(t1 - t0, 1))
        inv_fro = inv_frobenius_sq(A, a.block, a.chunk_flops)
        del A; torch.cuda.empty_cache()
        rec.d["gpu64_chol"] = dict(seconds=time.time() - t0, cholesky_seconds=t1 - t0, trace_inverse=inv_fro,
                                   direct_effective_dimension=n - n * NUGGET * inv_fro, block=a.block)
        rec.save()
    merged = fds.summarize(rec)
    rec.d["status"] = "COMPLETE"
    rec.save()
    print(json.dumps(rec.d.get("summary", {}), indent=1))
    print("gpu64_chol:", json.dumps(rec.d.get("gpu64_chol"), indent=1))
    print("gpu64_top:", json.dumps(rec.d.get("gpu64_top"), indent=1)[:800])


if __name__ == "__main__":
    main()
