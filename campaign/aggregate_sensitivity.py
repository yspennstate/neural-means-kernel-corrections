"""Verify a complete evidence package and summarize paired sensitivity runs.

All values are fractions. Seed summaries describe variation across the specified
training procedures on shared benchmark cases; they are not population intervals.
Every reported mean is recomputed from per-case arrays and checked against the
producer's separately stored summary. An incomplete package cannot produce tables.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checked_mean(values, expected, label):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Invalid per-case values: " + label)
    direct = math.fsum(map(float, values)) / len(values)
    if not math.isclose(direct, float(expected), rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f"Summary does not reproduce: {label}: {direct} != {expected}")
    return direct


def stats(values):
    values = list(map(float, values))
    if len(values) < 2 or not all(map(math.isfinite, values)):
        raise ValueError("At least two finite seed values required")
    return dict(n=len(values), mean=statistics.fmean(values),
                sd=statistics.stdev(values), minimum=min(values), maximum=max(values),
                per_seed=values)


def validate_indices(values, expected_size, label):
    if (values.ndim != 1 or len(values) != expected_size
            or not np.issubdtype(values.dtype, np.integer)
            or np.any(values < 0) or len(np.unique(values)) != expected_size):
        raise ValueError("Invalid case indices: " + label)


def aggregate(root):
    root = root.resolve()
    manifest = read(root / "evidence_manifest.json")
    if not manifest["complete"] or manifest["centering_seeds"] != list(range(10)) or manifest["mismatch_seeds"] != list(range(10)) or manifest["grid_seeds"] != [0, 1, 2] or manifest["bands"] != ["o2", "wco2", "sco2"]:
        raise ValueError("Unexpected or incomplete campaign design")
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if root.resolve() not in path.parents or not path.is_file() or sha(path) != expected:
            raise ValueError("Evidence identity failed: " + name)
    # A valid hash for an unrelated file cannot stand in for a consumed record.
    # Reject unhashed evidence as well as changed bytes.
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and p.name != "evidence_manifest.json"}
    if actual != set(manifest["files"]):
        raise ValueError("Evidence file set differs from its manifest")
    for name in ("status.json", "followup_status.json"):
        if read(root / name)["status"] != "COMPLETE":
            raise ValueError("Unfinished controller: " + name)
    centering = []; mismatch = []; folds = []
    for seed in range(10):
        sr = root / "seeds" / f"sm_s{seed}"
        rec = read(sr / "summary.json")
        if rec["seed"] != seed or sha(sr / "paired_errors.npz") != rec["error_archive_sha256"]:
            raise ValueError("Centering seed identity failed")
        with np.load(sr / "paired_errors.npz", allow_pickle=False) as z:
            validate_indices(z["test_indices"], 20000, f"centering {seed}")
            row = dict(seed=seed)
            for arm in ("historical", "pooled", "local"):
                row[arm] = checked_mean(z[arm], rec["metrics"][arm]["mean_relative_l2"], arm)
                rms = math.sqrt(math.fsum(float(v) ** 2 for v in z[arm]) / len(z[arm]))
                if not math.isclose(rms, rec["metrics"][arm]["rms_relative_l2"], rel_tol=1e-10):
                    raise ValueError("RMS does not reproduce")
            delta = z["local"] - z["pooled"]
            if not np.array_equal(delta, z["local_minus_pooled"]):
                raise ValueError("Centering pairing/sign mismatch")
            row["local_minus_pooled"] = checked_mean(delta, rec["local_minus_pooled_mean"], "centering contrast")
            row["local_better_fraction"] = float(np.mean(delta < 0))
            if row["local_better_fraction"] != rec["local_better_fraction"]:
                raise ValueError("Centering win fraction mismatch")
        centering.append(row)
        fold = read(sr / "runs/fold_centering.json")
        if fold["seed"] != seed or len(fold["folds"]) != 4 or fold["ntrain"] != 19000:
            raise ValueError("Fold configuration mismatch")
        folds.extend(fold["folds"])
        mr = root / "mismatch" / f"s{seed}"
        rec = read(mr / "summary.json")
        if rec["seed"] != seed or sha(mr / "paired_errors.npz") != rec["errors_sha256"]:
            raise ValueError("Mismatch seed identity failed")
        row = dict(seed=seed, controls=rec["controls"])
        with np.load(mr / "paired_errors.npz", allow_pickle=False) as z:
            for split, size in (("validation", 1000), ("test", 20000)):
                validate_indices(z[split + "_indices"], size, f"mismatch {split}")
                metric = rec["metrics"][split]
                for field, producer in (("base", "base_mean"), ("historical", "historical_correction_mean"), ("consistent", "consistent_correction_mean")):
                    row[split + "_" + field] = checked_mean(z[split + "_" + field], metric[producer], split + field)
                term = z[split + "_propagated_mismatch"]
                row[split + "_mismatch_mean"] = checked_mean(term, metric["propagated_mismatch"]["mean"], "propagated term")
                for label, value in (("maximum", np.max(term)), ("p95", np.quantile(term, .95))):
                    if not math.isclose(float(value), metric["propagated_mismatch"][label], rel_tol=1e-10, abs_tol=1e-12):
                        raise ValueError("Mismatch distribution summary failed")
                    row[split + "_mismatch_" + label] = float(value)
                row[split + "_consistent_minus_historical"] = checked_mean(
                    z[split + "_consistent"] - z[split + "_historical"],
                    row[split + "_consistent"] - row[split + "_historical"], "mismatch contrast")
            stage = "consistent" if row["validation_consistent"] < row["validation_base"] else "base"
            row["selected_test"] = row["test_" + stage]
            if not math.isclose(row["selected_test"], rec["metrics"]["inference_consistent_selected_test"], abs_tol=1e-12):
                raise ValueError("Mismatch validation selection failed")
        mismatch.append(row)
    grids = {}
    for band in ("o2", "wco2", "sco2"):
        rows = []
        for seed in range(3):
            br = root / "oco_grid" / "seeds" / f"oco_{band}_s{seed}"
            rec = read(br / "grid_sensitivity.json")
            if rec["identity"]["seed"] != seed or rec["identity"]["band"] != band:
                raise ValueError("Grid identity mismatch")
            row = dict(seed=seed, scenarios={})
            errors = {}
            for scenario in ("recorded_grid", "expanded_grid"):
                report = read(br / f"{scenario}.json")
                if report != rec["results"][scenario] or sha(br / f"{scenario}_errors.npz") != report["errors_sha256"]:
                    raise ValueError("Grid report/array identity failed")
                with np.load(br / f"{scenario}_errors.npz", allow_pickle=False) as z:
                    errors[scenario] = {name: z[name] for name in z.files}
                metrics = {}
                for model, scores in report["metrics"].items():
                    metrics[model] = {metric: checked_mean(errors[scenario][model + "_" + metric], expected, model + metric) for metric, expected in scores.items()}
                heads = [h for h in report["hyper"].values() if "cells" in h]
                row["scenarios"][scenario] = dict(metrics=metrics,
                    boundary_heads=sum(h["scale_boundary"] or h["nugget_boundary"] for h in heads),
                    failed_cells=sum(c["status"] != "OK" for h in heads for c in h["cells"]))
            row["expanded_minus_recorded"] = {}
            for name, old in errors["recorded_grid"].items():
                delta = errors["expanded_grid"][name] - old
                row["expanded_minus_recorded"][name] = checked_mean(delta, rec["expanded_minus_recorded"][name]["mean"], name + " grid contrast")
                if name.startswith(("mean_", "ridge_", "kernel_flow_")) and np.any(delta != 0):
                    raise ValueError("A frozen control changed between grids: " + name)
            rows.append(row)
        models = rows[0]["scenarios"]["recorded_grid"]["metrics"]
        grids[band] = dict(rows=rows, aggregate={
            scenario: {model: {metric: stats(r["scenarios"][scenario]["metrics"][model][metric] for r in rows)
                               for metric in ("reduced", "radiance")} for model in models}
            for scenario in ("recorded_grid", "expanded_grid")},
            differences={name: stats(r["expanded_minus_recorded"][name] for r in rows)
                         for name in rows[0]["expanded_minus_recorded"]})
    return dict(schema=1, units="fractions; multiply by 100 for percent or percentage-point differences",
                seed_uncertainty="sample standard deviation on shared benchmark cases, not a population confidence interval",
                evidence_manifest_sha256=sha(root / "evidence_manifest.json"),
                centering=dict(rows=centering, aggregate={key: stats(r[key] for r in centering)
                    for key in ("historical", "pooled", "local", "local_minus_pooled")},
                    field_change_range=[min(f["centering_relative_frobenius"] for f in folds), max(f["centering_relative_frobenius"] for f in folds)],
                    independent_solve_max=max(f["independent_solve_max_absolute_difference"] for f in folds)),
                mismatch=dict(rows=mismatch, aggregate={key: stats(r[key] for r in mismatch)
                    for key in ("test_base", "test_historical", "test_consistent", "selected_test",
                                "test_mismatch_mean", "test_mismatch_p95", "test_mismatch_maximum", "test_consistent_minus_historical")}),
                oco_grids=grids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = aggregate(args.root)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Verified complete evidence and wrote", args.out)


if __name__ == "__main__":
    main()
