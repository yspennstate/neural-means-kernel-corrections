"""Merge the per-box prospective scoreboards into the campaign's thirty band-seeds.

Part of the O2 sweep was re-run by a second queue lane, so a band-seed can be
present on two boxes. The tables read one lane; this reads the same one.

Usage: python campaign/oco_secmom_merge.py <score_dir> [out_json]
"""
import collections
import json
import os
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
        p = os.path.join(d, box + ".json")
        with open(p, encoding="utf-8") as f:
            per_box[box] = json.load(f)

    rows, seen = [], set()
    for (band, seed), box in sorted(LANE.items()):
        got = [r for r in per_box[box] if r["band"] == band and r["seed"] == seed]
        if not got:
            print("MISSING %s seed %d on %s" % (band, seed, box))
            continue
        seen.add((band, seed))
        rows.extend(got)

    n = len(rows)
    ok = sum(r["correct"] for r in rows)
    print("band-seeds: %d   pairs: %d" % (len(seen), n))
    print("overall: %d/%d = %.1f%%" % (ok, n, 100 * ok / n))
    print()
    print("by margin |rho - e1/e2| (decided on validation):")
    summary = {"n_band_seeds": len(seen), "n_pairs": n,
               "overall_correct": ok, "overall_pct": round(100 * ok / n, 1),
               "bins": [], "bands": {}, "repeat_misses": []}
    for lo, hi in BINS:
        sel = [r for r in rows if lo <= r["margin"] < hi]
        if not sel:
            continue
        k = sum(r["correct"] for r in sel)
        print("  [%.2f, %.2f)  %4d/%4d = %5.1f%%" % (lo, hi, k, len(sel),
                                                     100 * k / len(sel)))
        summary["bins"].append(dict(lo=lo, hi=hi, correct=k, n=len(sel),
                                    pct=round(100 * k / len(sel), 1)))
    print()
    for band in ("o2", "wco2", "sco2"):
        sel = [r for r in rows if r["band"] == band]
        k = sum(r["correct"] for r in sel)
        print("  %-5s %3d/%3d = %5.1f%%" % (band, k, len(sel), 100 * k / len(sel)))
        summary["bands"][band] = dict(correct=k, n=len(sel),
                                      pct=round(100 * k / len(sel), 1))

    # Does a pair fail on many seeds, or do failures scatter?
    print()
    print("pairs missing on at least 5 of their 10 seeds:")
    miss = collections.Counter()
    tot = collections.Counter()
    for r in rows:
        key = (r["band"], r["better"], r["weaker"])
        tot[key] += 1
        if not r["correct"]:
            miss[key] += 1
    for key, m in sorted(miss.items(), key=lambda kv: -kv[1]):
        if m >= 5:
            band, a, b = key
            mm = [r["margin"] for r in rows
                  if (r["band"], r["better"], r["weaker"]) == key and not r["correct"]]
            print("  %-5s %-12s vs %-12s  %d/%d missed, median margin %.3f"
                  % (band, a, b, m, tot[key], sorted(mm)[len(mm) // 2]))
            summary["repeat_misses"].append(dict(band=band, better=a, weaker=b,
                                                 missed=m, of=tot[key],
                                                 median_margin=round(sorted(mm)[len(mm) // 2], 4)))
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1)
        print("wrote %s" % out_path)


if __name__ == "__main__":
    main()
