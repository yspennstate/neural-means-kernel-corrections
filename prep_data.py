"""One-time data preparation for the structural mechanics benchmark.

Verifies the input fields are constant along axis 1 (i.e. the true input is the
1-D boundary load), extracts float32 arrays in (N, ...) layout, and fixes the
canonical split: first 20000 samples = train pool, last 20000 = test,
matching the high-data protocol of de Hoop et al. / Batlle et al.
"""
import numpy as np, pathlib, json, os

# Point SRC at the directory holding the two distributed .npy files
# (StructuralMechanics_inputs.npy / _outputs.npy from the Caltech record).
# Override with the STRUCTMECH_SRC environment variable; default is ./data.
HERE = pathlib.Path(__file__).resolve().parent
SRC = pathlib.Path(os.environ.get("STRUCTMECH_SRC", HERE / "data"))
DST = HERE / "data"
DST.mkdir(exist_ok=True)

X = np.load(SRC / "StructuralMechanics_inputs.npy", mmap_mode="r")   # (41,41,40000)
Y = np.load(SRC / "StructuralMechanics_outputs.npy", mmap_mode="r")

N = X.shape[2]
print("samples:", N)

# verify column-constancy on every sample (input = 1D load tiled along axis 1)
CHUNK = 2000
max_dev = 0.0
for s in range(0, N, CHUNK):
    xb = np.asarray(X[:, :, s:s+CHUNK])            # (41,41,c)
    dev = np.abs(xb - xb[:, :1, :]).max()
    max_dev = max(max_dev, float(dev))
print("max deviation from column-constant inputs:", max_dev)
assert max_dev == 0.0, "inputs are NOT exactly column-constant"

loads = np.ascontiguousarray(np.asarray(X[:, 0, :]).T, dtype=np.float32)      # (N,41)
stress = np.ascontiguousarray(np.asarray(Y).transpose(2, 0, 1), dtype=np.float32)  # (N,41,41)
np.save(DST / "loads.npy", loads)
np.save(DST / "stress.npy", stress)
print("loads", loads.shape, "stress", stress.shape)

idx = np.arange(N)
split = dict(train=idx[:20000].tolist(), test=idx[20000:].tolist())
np.save(DST / "idx_train.npy", idx[:20000])
np.save(DST / "idx_test.npy", idx[20000:])

# float64 casting check for the metric: rel L2 stats of outputs
nrm = np.linalg.norm(stress.reshape(N, -1).astype(np.float64), axis=1)
print("output norms: min %.3f  med %.3f  max %.3f" % (nrm.min(), np.median(nrm), nrm.max()))
json.dump(dict(n=N, max_dev=max_dev, norm_min=float(nrm.min()), norm_med=float(np.median(nrm)),
               norm_max=float(nrm.max())), open(DST / "prep_summary.json", "w"), indent=1)
print("done")
