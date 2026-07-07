"""Loading for the OCO-2 radiative-transfer emulation data.

The data come from the OSF project of Lamminpaa, Susiluoto, Hobbs, McDuffie,
Braverman and Owhadi (osf.io/u2t8a, files `dimred_variables_4_mono.jld` and
`dimred_data_4_mono.jld`; JLD2 is HDF5, so h5py reads it). Per spectral band
the emulator maps a reduced atmospheric state (20-24 dimensions) to a reduced
radiance (40 PCA coefficients), 20000 training and 2000 test pairs. The same
project stores the kernel-flow emulator's own test predictions
(`kf_results_<band>_4_mono.jld`), which is what our models are compared with.

Two error metrics matter and they disagree. The reduced metric is the plain
relative L2 on the 40 standardized coefficients. The radiance metric first
maps back to the monochromatic spectrum through the stored PCA projection and
norms; since the projection is orthogonal, on the reduced side it is the
diagonal weighting s_z, and almost all of its energy sits on the first few
coefficients.
"""
import pathlib
import h5py
import numpy as np

DATA = pathlib.Path(__file__).resolve().parent / "data" / "jpl_oco2"
BANDS = ("o2", "wco2", "sco2")


def load_band(band, n_val=2000, seed=0):
    """Split dict (Xtr, Ytr, Xval, Yval, Xte, Yte) for one spectral band."""
    with h5py.File(DATA / "dimred_variables_4_mono.jld", "r") as h:
        X = h[f"xr_{band}"][:].astype(np.float64)
        Y = h[f"z_{band}"][:].astype(np.float64)
        Xte = h[f"xr_{band}_test"][:].astype(np.float64)
        Yte = h[f"z_{band}_test"][:].astype(np.float64)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    va, tr = perm[:n_val], perm[n_val:]
    return dict(Xtr=X[tr], Ytr=Y[tr], Xval=X[va], Yval=Y[va], Xte=Xte, Yte=Yte)


def reconstruction(band):
    """PCA projection and norms mapping reduced coefficients to radiance."""
    with h5py.File(DATA / "dimred_data_4_mono.jld", "r") as h:
        return dict(P=h[f"P_{band}"][:].astype(np.float64),
                    m=h[f"m_{band}"][:].astype(np.float64).ravel(),
                    m_z=h[f"m_z_{band}"][:].astype(np.float64).ravel(),
                    s_z=h[f"s_z_{band}"][:].astype(np.float64).ravel())


def to_radiance(z, recon):
    return (z * recon["s_z"] + recon["m_z"]) @ recon["P"] + recon["m"]


def radiance_error(pred_z, true_z, recon):
    """Relative L2 on the reconstructed monochromatic radiance."""
    P, T = to_radiance(pred_z, recon), to_radiance(true_z, recon)
    return float(np.mean(np.linalg.norm(P - T, axis=1) / np.linalg.norm(T, axis=1)))


def kernel_flow_predictions(band):
    """The kernel-flow emulator's own test predictions, for the head-to-head."""
    with h5py.File(DATA / f"kf_results_{band}_4_mono.jld", "r") as h:
        return h["pred_zs"][:].astype(np.float64)
