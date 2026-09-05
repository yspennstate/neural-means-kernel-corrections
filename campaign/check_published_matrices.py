"""Check released convex-risk records using active-set solves and KKT bounds.

This checks the published matrices, not their construction from field arrays.
It uses no general-purpose optimizer and does not reuse the producer's SLSQP
solution. All error quantities in the JSON output are fractions.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def solve_simplex(matrix):
    S = np.asarray(matrix, dtype=np.float64)
    if S.ndim != 2 or S.shape[0] != S.shape[1] or not np.isfinite(S).all():
        raise ValueError("Invalid matrix")
    if np.max(np.abs(S-S.T)) > 1e-12:
        raise ValueError("Matrix is not symmetric")
    S = (S + S.T) / 2
    eigenvalues = np.linalg.eigvalsh(S)
    if eigenvalues[0] <= 0:
        raise ValueError("This active-set check requires a positive definite matrix")
    n = len(S)
    w = np.zeros(n)
    w[np.argmin(np.diag(S))] = 1
    active = set(np.flatnonzero(w))
    for iteration in range(10*n*n):
        ix = np.array(sorted(active))
        z = np.linalg.solve(S[np.ix_(ix, ix)], np.ones(len(ix)))
        z /= z.sum()
        if np.min(z) < -1e-13:
            direction = z - w[ix]
            candidates = direction < 0
            step = np.min(-w[ix][candidates] / direction[candidates])
            w[ix] += step * direction
            w[np.abs(w) < 1e-13] = 0
            active = set(np.flatnonzero(w))
            continue
        w[:] = 0
        w[ix] = np.maximum(z, 0)
        w /= w.sum()
        Sw = S @ w
        objective = float(w @ Sw)
        entering = int(np.argmin(Sw))
        if Sw[entering] >= objective - 1e-14:
            break
        active.add(entering)
    else:
        raise RuntimeError("Active-set solve did not converge")
    # Convexity gives q(v) >= q(w) + 2(Sw)^T(v-w) for every v.
    # Minimizing that affine expression over the simplex gives this lower
    # bound; q(w) is a feasible upper bound, independently of solver success.
    lower = 2*float(np.min(Sw)) - objective
    gap = objective - lower
    if np.min(w) < 0 or abs(w.sum()-1) > 1e-12 or gap > 2e-12:
        raise ValueError("Simplex/KKT certificate failed")
    return dict(weights=w.tolist(), squared_risk_lower=lower,
                squared_risk_upper=objective, squared_risk_gap=gap,
                rms_lower=math.sqrt(max(0, lower)), rms_upper=math.sqrt(objective),
                smallest_eigenvalue=float(eigenvalues[0]), iterations=iteration+1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Refusing to overwrite a check record")
    source = args.root / "campaign/collected/dgx/seedarch.json"
    record = json.loads(source.read_text(encoding="utf-8"))
    S = np.asarray(record["S_ev"], dtype=np.float64)
    if S.shape != (60, 60) or (record["n_cal"], record["n_ev"]) != (1000, 19000):
        raise ValueError("Unexpected pool design")
    arch = [name.rsplit("_s", 1)[0] for name in record["names"]]
    if len(set(arch)) != 6 or any(arch.count(a) != 10 for a in set(arch)):
        raise ValueError("Unexpected architecture blocks")
    diagonal_min = min(float(S[i,i]) for i in range(60))
    within_min = min(float(S[i,j]) for i in range(60) for j in range(i) if arch[i] == arch[j])
    between_min = min(float(S[i,j]) for i in range(60) for j in range(i) if arch[i] != arch[j])
    if not 0 <= between_min <= within_min <= diagonal_min:
        raise ValueError("Block-floor ordering does not hold")
    # Direct moment form; the producer's presentation uses normalized ratios.
    floor = between_min + (within_min-between_min)/6 + (diagonal_min-within_min)/60
    pool = solve_simplex(S)
    pool.update(empirical_block_floor_rms=math.sqrt(floor), diagonal_min=diagonal_min,
                within_min=within_min, between_min=between_min,
                reported_oracle_rms=record["sixty_convex_oracle_ev"]["e2"])
    if abs(pool["rms_upper"]-pool["reported_oracle_rms"]) > 1e-8:
        raise ValueError("Published pool optimum differs from the independent solve")
    inputs = {str(source.relative_to(args.root)): sha(source)}
    groups = {}
    for name in ("secmom6_seeded.json", "secmom5c_seeded.json"):
        path = args.root / "campaign/collected" / name
        inputs[str(path.relative_to(args.root))] = sha(path)
        rows = json.loads(path.read_text(encoding="utf-8"))
        if sorted(r["seed"] for r in rows) != list(range(10)):
            raise ValueError("Incomplete matrix seed series")
        verified = []
        for r in rows:
            result = solve_simplex(r["S"])
            result.update(seed=r["seed"], reported_rms=r["pred_rms"])
            if abs(result["rms_upper"]-r["pred_rms"]) > 1e-8:
                raise ValueError("Published seed optimum differs from the independent solve")
            verified.append(result)
        groups[name] = verified
    output = dict(schema=1, units="fractions", input_sha256=inputs,
                  driver_sha256=sha(Path(__file__)), pool=pool, seed_matrices=groups,
                  scope="Independent matrix optimization and algebra; not a field-array reconstruction or population certificate")
    args.out.write_text(json.dumps(output, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(pool_rms_interval=[pool["rms_lower"], pool["rms_upper"]],
                         empirical_block_floor=pool["empirical_block_floor_rms"],
                         checked_seed_matrices=sum(map(len, groups.values())))))


if __name__ == "__main__":
    main()
