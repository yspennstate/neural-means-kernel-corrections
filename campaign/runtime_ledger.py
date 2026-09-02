#!/usr/bin/env python3
"""Runtime ledger for the structmech members: what each model COST and what it BOUGHT.

Every member JSON already carries minutes, threads, params and test error, so this just
harvests them. Wall minutes alone are not comparable across members - the front stages ran
at 16 threads and the current lanes at 10 - so the ledger reports core-minutes too.

Writes results/runtime_ledger.csv and prints a per-member summary. Re-run any time; it
rewrites from whatever JSONs exist, so it fills in as seeds land.
"""
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("NMKC_ROOT", Path.home() / "nmkc2"))
SEEDS = ROOT / "seeds"
OUT = ROOT / "results" / "runtime_ledger.csv"

FIELDS = ["seed", "member", "kind", "epochs", "threads", "params",
          "minutes", "core_minutes", "val", "test", "best_ep", "file"]

SMOKE_SEEDS = {91}          # the 2-epoch full-chain smoke; timings real, numbers not comparable


def member_of(stem):
    """Map a run filename to the pipeline's member name."""
    for tag in ("mlpMSE", "mlpR", "mlp", "fno", "unet", "krr_full", "krr_oof"):
        if stem.startswith(tag):
            return tag
    return stem.split("_")[0]


def rows():
    if not SEEDS.is_dir():
        return
    for seed_dir in sorted(SEEDS.glob("sm_s*")):
        try:
            seed = int(seed_dir.name.replace("sm_s", ""))
        except ValueError:
            continue
        runs = seed_dir / "runs"
        if not runs.is_dir():
            continue
        for jf in sorted(runs.glob("*.json")):
            try:
                with open(jf, encoding="utf-8") as fh:
                    d = json.load(fh)
            except Exception as exc:
                print("skip %s: %s" % (jf.name, exc), file=sys.stderr)
                continue
            args = d.get("args") or {}
            minutes = d.get("minutes")
            threads = args.get("threads")
            core_min = round(minutes * threads, 1) if (minutes and threads) else ""
            yield {
                "seed": seed,
                "member": member_of(jf.stem),
                "kind": d.get("kind", ""),
                "epochs": args.get("epochs", ""),
                "threads": threads if threads is not None else "",
                "params": d.get("params", ""),
                "minutes": round(minutes, 2) if minutes else "",
                "core_minutes": core_min,
                "val": d.get("val", ""),
                "test": d.get("test", ""),
                "best_ep": d.get("best_ep", ""),
                "file": jf.name,
            }


def main():
    data = list(rows())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(data)
    print("wrote %s  (%d rows)" % (OUT, len(data)))

    # The smoke seed runs the whole chain at 2 epochs. Its rows are real timings of a
    # different thing; pooling them with production seeds corrupts every mean.
    timed = [r for r in data
             if r["minutes"] != "" and r["test"] != ""
             and r["seed"] not in SMOKE_SEEDS and r["epochs"] != 2]
    skipped = len(data) - len(timed)
    print("summary over production rows only (%d smoke/untimed rows excluded)" % skipped)
    if not timed:
        print("no timed production rows yet")
        return

    by = {}
    for r in timed:
        by.setdefault(r["member"], []).append(r)

    def mean(vals):
        return sum(vals) / len(vals)

    summary = []
    for m, rs in by.items():
        summary.append(dict(
            member=m,
            n=len(rs),
            epochs=rs[0]["epochs"],
            params=rs[0]["params"],
            minutes=mean([r["minutes"] for r in rs]),
            core_min=mean([r["core_minutes"] for r in rs if r["core_minutes"] != ""] or [0]),
            test=mean([r["test"] for r in rs]),
        ))
    summary.sort(key=lambda s: s["core_min"])
    # KRR carries no thread count in its JSON, so it has no core-minutes. Ranking cost
    # against a zero baseline would invent the ratio, so it sits out of the xcost column.
    priced = [s for s in summary if s["core_min"] > 0]
    base = priced[0]["core_min"] if priced else 1.0

    print()
    print("%-8s %3s %7s %10s %9s %11s %9s %8s" %
          ("member", "n", "epochs", "params", "min", "core-min", "test", "xcost"))
    for s in summary:
        xc = "%6.0fx" % (s["core_min"] / base) if s["core_min"] > 0 else "     - "
        print("%-8s %3d %7s %10s %9.1f %11s %9.6f %8s" %
              (s["member"], s["n"], s["epochs"], s["params"] or "-", s["minutes"],
               ("%.0f" % s["core_min"]) if s["core_min"] > 0 else "n/a", s["test"], xc))
    print("(krr carries no thread count in its JSON - wall minutes only, no core-minutes)")

    if len(priced) >= 2:
        best = min(priced, key=lambda s: s["test"])
        cheap = priced[0]
        print()
        print("cheapest priced member : %s at %.0f core-min, test %.6f" %
              (cheap["member"], cheap["core_min"], cheap["test"]))
        print("most accurate          : %s at %.0f core-min, test %.6f" %
              (best["member"], best["core_min"], best["test"]))
        if best["member"] != cheap["member"]:
            gain = (cheap["test"] - best["test"]) / cheap["test"] * 100.0
            print("=> %.0fx the compute buys %.2f%% of test error" %
                  (best["core_min"] / cheap["core_min"], gain))


if __name__ == "__main__":
    main()
