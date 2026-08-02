"""Rewrite the paper's macro block from the seed campaign's aggregates.

Reads campaign/summary.json (written by campaign/collect.py) and updates the
\\newcommand block in paper/main.tex: each mean macro keeps its name, and a
companion Sd macro (for example \\errBestSd) is created next to it when the
paper does not define one yet. Values are percentages at two decimals, like
freeze.py, whose single-seed role this script replaces at the results freeze.

    python campaign/freeze_seeded.py [--dry-run]
"""
import argparse, json, pathlib, re

HERE = pathlib.Path(__file__).resolve().parent
MAIN = HERE.parent / "paper" / "main.tex"
SUMMARY = HERE / "summary.json"

ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()

if not SUMMARY.exists():
    raise SystemExit("campaign/summary.json not found; run campaign/collect.py first")
S = json.load(open(SUMMARY))


def stat(path):
    d = S
    for k in path:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


# macro name -> (path to the stats dict, scale)
MEANS = {
    "errBest": (("structmech", "corr"), 100),
    "errStack": (("structmech", "stack"), 100),
    "errMLP": (("structmech", "member_mlp"), 100),
    "errMSE": (("structmech", "member_mlpMSE"), 100),
    "errRef": (("structmech", "member_mlpR"), 100),
    "errFNOG": (("structmech", "member_fno"), 100),
    "errUNetG": (("structmech", "member_unet"), 100),
    "errKRRload": (("structmech", "krr"), 100),
    "uqCover": (("structmech", "uq_cover90_scaled"), 100),
}

txt = MAIN.read_text(encoding="utf-8")
report = []
for name, (path, scale) in MEANS.items():
    st = stat(path)
    if not st or st.get("mean") is None:
        report.append(f"  !! {name}: no campaign statistic")
        continue
    mean = f"{scale * st['mean']:.2f}"
    sd = f"{scale * st['sd']:.2f}"
    txt, n_mean = re.subn(
        rf"(\\newcommand{{\\{name}}}{{)[^}}]*(}})", rf"\g<1>{mean}\g<2>", txt)
    sd_name = name + "Sd"
    if re.search(rf"\\newcommand{{\\{sd_name}}}", txt):
        txt, n_sd = re.subn(
            rf"(\\newcommand{{\\{sd_name}}}{{)[^}}]*(}})", rf"\g<1>{sd}\g<2>", txt)
        made = "updated"
    else:
        txt, n_sd = re.subn(
            rf"(\\newcommand{{\\{name}}}{{[^}}]*}}[^\n]*\n)",
            rf"\g<1>\\newcommand{{\\{sd_name}}}{{{sd}}}   % seed sd\n", txt)
        made = "created"
    report.append(f"  {name} = {mean} +- {sd} (n={st['n']}) "
                  f"[{'ok' if n_mean else 'MACRO NOT FOUND'}; Sd {made if n_sd else 'FAILED'}]")

for line in report:
    print(line)
if args.dry_run:
    print("dry run; main.tex unchanged")
else:
    MAIN.write_text(txt, encoding="utf-8")
    print("main.tex macros updated from the seed campaign")
