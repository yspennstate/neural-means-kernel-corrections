#!/usr/bin/env bash
# Download the LEAP subsampled low-res ClimSim arrays onto this box.
# Verifies the dataset listing through the HF API first and fails loudly with
# the actual file listing if the expected names are absent.
set -uo pipefail
ROOT="${NMKC_ROOT:-/srv/aiwork/nmkc10seed}"
TASK_ID="${TASK_ID:-climsim_dl}"
DST="$ROOT/climsim"
mkdir -p "$DST"
DSID="LEAP/subsampled_low_res"
API="https://huggingface.co/api/datasets/$DSID"

listing=$(curl -fsSL "$API" 2>&1)
if [ $? -ne 0 ]; then
  echo "API_FAIL $API"
  echo "$listing"
  exit 4
fi
echo "$listing" > /tmp/climsim_api.json

need="train_input.npy train_target.npy val_input.npy val_target.npy"
missing=""
for f in $need; do
  if ! grep -q "\"$f\"" /tmp/climsim_api.json; then
    missing="$missing $f"
  fi
done
if [ -n "$missing" ]; then
  echo "LISTING_MISMATCH missing:$missing"
  echo "available files (siblings):"
  python3 - <<'EOF'
import json
d = json.load(open("/tmp/climsim_api.json"))
for s in d.get("siblings", []):
    print(" ", s.get("rfilename"))
EOF
  exit 4
fi

for f in $need; do
  if [ -s "$DST/$f" ]; then
    echo "have $f ($(du -h "$DST/$f" | cut -f1))"
    continue
  fi
  url="https://huggingface.co/datasets/$DSID/resolve/main/$f"
  echo "fetch $url"
  wget -c -q --show-progress --progress=dot:giga -O "$DST/$f.part" "$url"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "WGET_FAIL rc=$rc $f"
    exit 5
  fi
  mv "$DST/$f.part" "$DST/$f"
done

python3 - "$DST" <<'EOF'
import sys, numpy as np, pathlib
d = pathlib.Path(sys.argv[1])
shapes = {}
for f in ("train_input", "train_target", "val_input", "val_target"):
    a = np.load(d / f"{f}.npy", mmap_mode="r")
    shapes[f] = a.shape
    print(f, a.shape, a.dtype)
assert shapes["train_input"][1] == 124, "input dim is not 124"
assert shapes["train_target"][1] == 128, "target dim is not 128"
assert shapes["train_input"][0] == shapes["train_target"][0]
assert shapes["val_input"][0] == shapes["val_target"][0]
print("SHAPES_OK")
EOF
rc=$?
if [ $rc -ne 0 ]; then
  echo "SHAPE_CHECK_FAIL rc=$rc"
  exit 6
fi

mkdir -p "$ROOT/results"
python3 - "$ROOT/results/$TASK_ID.json" "$TASK_ID" "$DST" <<'EOF'
import json, sys, pathlib, os
out, task_id, dst = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
files = {f.name: f.stat().st_size for f in dst.glob("*.npy")}
tmp = out + ".tmp"
json.dump({"task_id": task_id, "kind": "climsim_download", "files": files}, open(tmp, "w"), indent=1)
os.replace(tmp, out)
print("wrote", out)
EOF
echo "CLIMSIM_DL_DONE"
