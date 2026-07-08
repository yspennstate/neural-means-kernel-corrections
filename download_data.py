"""Fetch the datasets the paper reports on.

  python download_data.py oco2          # OCO-2 emulation files from OSF (~180 MB)
  python download_data.py structmech    # prints the Caltech record instructions
  python download_data.py all

Everything lands under data/ (git-ignored). See docs/data.md for the full list
of datasets, sizes, and the other Caltech benchmarks.
"""
import sys, pathlib, urllib.request, json

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"

# OCO-2 emulation files on OSF project u2t8a (osfstorage/data/), by file id
OSF = {
    "dimred_variables_4_mono.jld": "660e2282219e712003f6a7fa",
    "dimred_data_4_mono.jld":      "660e2275bba39a17bc729e70",
    "kf_results_o2_4_mono.jld":    "660e2381c053941800b4d3af",
    "kf_results_wco2_4_mono.jld":  "660e2397bba39a17c372a03a",
    "kf_results_sco2_4_mono.jld":  "660e23b0e65c601ccc7d9cd7",
}
OSF_URL = "https://files.us.osf.io/v1/resources/u2t8a/providers/osfstorage/{}"


def fetch(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have {dest.name}")
        return
    print(f"  downloading {dest.name} ...", flush=True)
    urllib.request.urlretrieve(url, dest)
    print(f"  wrote {dest.name} ({dest.stat().st_size/1e6:.0f} MB)")


def oco2():
    print("OCO-2 emulation data (OSF u2t8a) -> data/jpl_oco2/")
    # resolve current osfstorage ids from the API in case the pinned ids move
    ids = dict(OSF)
    try:
        api = "https://api.osf.io/v2/nodes/u2t8a/files/osfstorage/?page[size]=100"
        while api:
            page = json.load(urllib.request.urlopen(api))
            for f in page["data"]:
                name = f["attributes"]["name"]
                if name in ids:
                    ids[name] = f["id"]
            api = page["links"].get("next")
    except Exception as e:
        print(f"  (OSF API listing failed, using pinned ids: {e})")
    for name, fid in ids.items():
        fetch(OSF_URL.format(fid), DATA / "jpl_oco2" / name)


def structmech():
    print("Structural mechanics and the other Caltech benchmarks:")
    print("  Download from the Caltech record  https://data.caltech.edu/records/20091")
    print("  Place StructuralMechanics_inputs.npy / _outputs.npy in data/, then run")
    print("  prep_data.py. Advection_inputs.npy / _outputs.npy go in data/ as well.")
    print("  See docs/data.md for sizes and the Helmholtz / Navier-Stokes files.")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("oco2", "all"):
        oco2()
    if what in ("structmech", "all"):
        structmech()
