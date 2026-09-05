"""Package small, completed sensitivity results without prediction fields.

Run on the campaign host. The ZIP contains the executed source snapshot,
input identities, per-case errors, and stage receipts. Large checkpoints and
field predictions stay in the retained campaign directory. No SSH or account
configuration is part of this utility.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.out.exists():
        raise SystemExit("Refusing to replace an evidence archive")
    for name in ("status.json", "followup_status.json"):
        record = json.loads((root / name).read_text(encoding="utf-8"))
        if record.get("status") != "COMPLETE":
            raise SystemExit("Campaign is incomplete: " + name)
    paths = [root / name for name in (
        "status.json", "followup_status.json", "campaign_manifest.json",
        "followup_plan.json", "capacity.jsonl",
        "numerical_controls/driver_controls.json")]
    for name in ("scheduling_amendment.json", "followup_capacity.jsonl"):
        if (root / name).exists():
            paths.append(root / name)
    for folder in ("receipts", "followup_receipts"):
        paths.extend(sorted((root / folder).glob("*.json")))
    for folder in ("code", "diagnostics", "tools"):
        paths.extend(sorted((root / folder).rglob("*.py")))
    for seed in range(10):
        sr = root / "seeds" / f"sm_s{seed}"
        paths.extend(sr / name for name in ("summary.json", "paired_errors.npz", "runs/fold_centering.json"))
        for arm in ("runs", "pooled"):
            for name in ("hpix.json", "hpix_corr.json", "hstk.json", "hpix_uq.json", "hpix_uq.npz"):
                candidate = sr / arm / name
                if candidate.exists():
                    paths.append(candidate)
        mr = root / "mismatch" / f"s{seed}"
        paths.extend(mr / name for name in ("summary.json", "paired_errors.npz"))
    for seed in range(3):
        for band in ("o2", "wco2", "sco2"):
            br = root / "oco_grid" / "seeds" / f"oco_{band}_s{seed}"
            paths.extend(br / name for name in (
                "experiment_identity.json", "grid_sensitivity.json",
                "recorded_grid.json", "expanded_grid.json",
                "recorded_grid_errors.npz", "expanded_grid_errors.npz"))
            paths.extend(br / f"network_{mode}_arrays.json" for mode in ("flat", "wnum", "radx"))
    paths = sorted(set(paths))
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError([str(p) for p in paths if not p.is_file()])
    hashes = {path.relative_to(root).as_posix(): sha(path) for path in paths}
    manifest = dict(schema=1, complete=True, created_at=time.time(),
                    centering_seeds=list(range(10)), mismatch_seeds=list(range(10)),
                    grid_seeds=[0, 1, 2], bands=["o2", "wco2", "sco2"],
                    files=hashes,
                    note="Executed source bytes are retained, including their original line endings.")
    with zipfile.ZipFile(args.out, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in paths:
            z.write(path, path.relative_to(root).as_posix())
        z.writestr("evidence_manifest.json", json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(args.out) as z:
        for name, expected in hashes.items():
            if hashlib.sha256(z.read(name)).hexdigest() != expected:
                raise RuntimeError("Archive verification failed: " + name)
    print(json.dumps(dict(archive=str(args.out), sha256=sha(args.out), files=len(hashes))))


if __name__ == "__main__":
    main()
