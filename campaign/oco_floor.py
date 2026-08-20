"""Seed-ensembling floor on the OCO-2 bands, over the full ten-seed campaign.

Proposition~\\ref{prop:secmom} is stated for the squared-metric risk, with
$S_{mk}=\\mathbb E\\,\\langle\\rho_m,\\rho_k\\rangle$ in the relative metric, so
this script forms $S$ over the ten seeded copies of the flat network and reads
the floor off it rather than off an approximation. Three quantities come out:

  * the identity  $\\mathbb E\\,\\ell(f_w)^2=w^\\top Sw$  at uniform weights,
    checked against the directly measured risk of the averaged prediction ---
    this is a control on the whole computation, and it should hold to
    numerical precision;
  * the ten-member value $e^2(\\varrho+(1-\\varrho)/M)$ and the infinite-seed
    floor $e^2\\varrho$ that part (ii) predicts from the measured correlation;
  * the same-metric error of the feature head and of the per-coordinate
    combination, so the question of whether either passes below the floor is
    asked in the metric the proposition uses.

An earlier version of this study measured the correlation from a single pair
of seeds. Here every band has all 45 pairs.

Usage: python campaign/oco_floor.py <resid_dir> [out_json]
"""
import itertools
import json
import os
import sys

import numpy as np

# The a2_* queue lane re-ran part of the O2 sweep; the tables use the original
# lane, so the floor must read the same one. Box of record per band-seed.
LANE = {
    ("o2", 0): "b02", ("o2", 1): "b01", ("o2", 2): "b01", ("o2", 3): "b02",
    ("o2", 4): "b02", ("o2", 5): "b03", ("o2", 6): "b03", ("o2", 7): "b04",
    ("o2", 8): "b04", ("o2", 9): "b04",
    ("wco2", 0): "b01", ("wco2", 1): "b01", ("wco2", 2): "b02", ("wco2", 3): "b02",
    ("wco2", 4): "b03", ("wco2", 5): "b03", ("wco2", 6): "b04", ("wco2", 7): "b04",
    ("wco2", 8): "b04", ("wco2", 9): "b04",
    ("sco2", 0): "b01", ("sco2", 1): "b01", ("sco2", 2): "b01", ("sco2", 3): "b02",
    ("sco2", 4): "b02", ("sco2", 5): "b03", ("sco2", 6): "b03", ("sco2", 7): "b04",
    ("sco2", 8): "b04", ("sco2", 9): "b04",
}


def rms_rel(resid, den):
    """sqrt(E ell^2) for a residual array, in the relative metric."""
    return float(np.sqrt(np.mean((np.linalg.norm(resid, axis=1) / den) ** 2)))


def main():
    root = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    report = {}

    for band in ("o2", "wco2", "sco2"):
        resid, head, comb = {}, {}, {}
        den = None
        for seed in range(10):
            p = os.path.join(root, LANE[(band, seed)], "%s_%d.npz" % (band, seed))
            if not os.path.exists(p):
                print("MISSING %s" % p)
                continue
            z = np.load(p)
            y = z["Yte"].astype(np.float64)
            d = np.linalg.norm(y, axis=1)
            if den is None:
                den = d
            elif not np.allclose(den, d):
                raise SystemExit("%s seed %d: test targets differ between seeds"
                                 % (band, seed))
            resid[seed] = z["resid_flat"].astype(np.float64)
            if "resid_dkr_flat" in z:
                head[seed] = z["resid_dkr_flat"].astype(np.float64)
            if "resid_combined" in z:
                comb[seed] = z["resid_combined"].astype(np.float64)

        seeds = sorted(resid)
        M = len(seeds)
        # S in the relative metric: S_mk = E <rho_m, rho_k> / ||y||^2
        S = np.empty((M, M))
        for i, a in enumerate(seeds):
            for j, b in enumerate(seeds):
                S[i, j] = np.mean((resid[a] * resid[b]).sum(axis=1) / den ** 2)
        e_each = np.sqrt(np.diag(S))
        e = float(np.sqrt(np.mean(np.diag(S))))
        pairs = [S[i, j] / (e_each[i] * e_each[j])
                 for i, j in itertools.combinations(range(M), 2)]
        rho = float(np.mean(pairs))

        w = np.full(M, 1.0 / M)
        identity_lhs = float(np.sqrt(w @ S @ w))
        ens_direct = rms_rel(np.mean([resid[s] for s in seeds], axis=0), den)
        pred_M = e * np.sqrt(rho + (1 - rho) / M)
        floor_inf = e * np.sqrt(rho)

        report[band] = dict(
            n_seeds=M, n_pairs=len(pairs),
            rho_mean=round(rho, 4), rho_sd=round(float(np.std(pairs, ddof=1)), 4),
            rho_min=round(min(pairs), 4), rho_max=round(max(pairs), 4),
            member_rms_pct=round(100 * e, 3),
            ens_uniform_identity_pct=round(100 * identity_lhs, 4),
            ens_uniform_direct_pct=round(100 * ens_direct, 4),
            identity_gap=float(abs(identity_lhs - ens_direct)),
            pred_ten_member_pct=round(100 * pred_M, 3),
            floor_infinite_pct=round(100 * floor_inf, 3),
            head_rms_pct=(round(100 * float(np.mean([rms_rel(head[s], den)
                                                     for s in sorted(head)])), 3)
                          if head else None),
            combined_rms_pct=(round(100 * float(np.mean([rms_rel(comb[s], den)
                                                         for s in sorted(comb)])), 3)
                              if comb else None),
        )
        r = report[band]
        print("%-5s pairs=%d  rho %.4f +- %.4f  member(rms) %.3f%%" %
              (band, r["n_pairs"], r["rho_mean"], r["rho_sd"], r["member_rms_pct"]))
        print("        identity w'Sw %.4f%% vs direct %.4f%%  (gap %.2e)" %
              (r["ens_uniform_identity_pct"], r["ens_uniform_direct_pct"],
               r["identity_gap"]))
        print("        predicted at M=10 %.3f%%   infinite-seed floor %.3f%%" %
              (r["pred_ten_member_pct"], r["floor_infinite_pct"]))
        print("        feature head %s%%   combination %s%%" %
              (r["head_rms_pct"], r["combined_rms_pct"]))

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1, sort_keys=True)
        print("wrote %s" % out_path)


if __name__ == "__main__":
    main()
