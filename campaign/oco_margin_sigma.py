"""Estimate the noise scale of the two-member rule, and test the margin bound.

Proposition~\\ref{prop:margin} is about the error in ESTIMATING the decision
statistic $D=\\varrho-e_1/e_2$: the rule is decided from $\\widehat D$ measured
on validation and is right or wrong according to $D$ on the test block, so the
$\\sigma$ the bound needs is the scale of $\\widehat D-D$, the
validation-to-test transfer.

An earlier version of this script used the within-pair scatter of $\\widehat D$
across seeds instead. That is a different quantity -- it measures how far a
pair's own gap moves under retraining, not how far the estimate of it moves --
and on these data it is about ten times smaller, which would make the bound
appear to be violated by the measured small-margin bins. Both are reported
below so the difference is visible, and the bound is evaluated with the
transfer scale, which is the one the proposition asks for.

Usage: python campaign/oco_margin_sigma.py <score_dir> [out_json]
"""
import collections
import json
import math
import os
import statistics as st
import sys

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
BINS = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.25), (0.25, 99.0)]


def main():
    d = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    per_box = {}
    for box in ("b01", "b02", "b03", "b04"):
        with open(os.path.join(d, box + ".json"), encoding="utf-8") as f:
            per_box[box] = json.load(f)

    rows = []
    for (band, seed), box in sorted(LANE.items()):
        rows.extend(r for r in per_box[box]
                    if r["band"] == band and r["seed"] == seed)
    if not rows:
        raise SystemExit("no rows found for the table's lane")
    if "rho_test" not in rows[0]:
        raise SystemExit("scoreboard predates the test-side statistics; rerun "
                         "campaign/oco_secmom_score.py to record rho_test/ratio_test")

    # The proposition's sigma: how far the validation estimate of D sits from
    # the value of D realized on the test block.
    d_val = [r["rho_val"] - r["ratio_val"] for r in rows]
    d_test = [r["rho_test"] - r["ratio_test"] for r in rows]
    err = [a - b for a, b in zip(d_val, d_test)]
    sigma = st.stdev(err)
    bias = st.mean(err)

    # The other quantity, reported for contrast only.
    by_pair = collections.defaultdict(list)
    for r, dv in zip(rows, d_val):
        by_pair[(r["band"], r["better"], r["weaker"])].append(dv)
    within = [st.stdev(v) for v in by_pair.values() if len(v) > 1]
    sigma_seed = st.median(within)

    print("pairs: %d over %d band-seeds"
          % (len(rows), len({(r["band"], r["seed"]) for r in rows})))
    print("sigma (validation-to-test transfer of D): %.4f   mean shift %+.4f"
          % (sigma, bias))
    print("within-pair scatter of D across seeds:    %.4f   <- NOT the "
          "proposition's sigma" % sigma_seed)
    print()
    print("margin bound exp(-tau^2 / 2 sigma^2) evaluated at each bin's median "
          "observed margin:")
    out = dict(sigma_transfer=round(sigma, 5), sigma_seed_scatter=round(sigma_seed, 5),
               transfer_bias=round(bias, 5), n_pairs=len(rows), bins=[])
    for lo, hi in BINS:
        sel = [r for r in rows if lo <= r["margin"] < hi]
        if not sel:
            continue
        k = sum(r["correct"] for r in sel)
        mid = st.median([r["margin"] for r in sel])
        bound = min(math.exp(-(mid ** 2) / (2 * sigma ** 2)), 1.0)
        guaranteed = 100 * (1 - bound)
        measured = 100 * k / len(sel)
        flag = "" if measured >= guaranteed - 1e-9 else "  <== BELOW THE BOUND"
        print("  [%.2f,%.2f) n=%3d  median margin %.3f  bound %.3f  "
              "guarantees >=%5.1f%%  measured %5.1f%%%s"
              % (lo, hi, len(sel), mid, bound, guaranteed, measured, flag))
        out["bins"].append(dict(lo=lo, hi=hi, n=len(sel),
                                median_margin=round(mid, 4), bound=round(bound, 4),
                                guaranteed_pct=round(guaranteed, 1),
                                measured_pct=round(measured, 1)))
    print()
    print("Note: the 28 pairs of a band-seed share members and the ten seeds of "
          "a band share one test block, so these counts are dependent.")
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        print("wrote %s" % out_path)


if __name__ == "__main__":
    main()
