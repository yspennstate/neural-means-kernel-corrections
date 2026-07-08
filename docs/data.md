# Data

None of the datasets is redistributed in this repository: they are large and
already have canonical public homes. This file lists every dataset the paper
and the code use, where it lives, its size, and where to put it. The helper
`download_data.py` fetches the two sets the paper reports on (structural
mechanics and OCO-2); the rest are one-line `curl`/`wget` from the sources
below. Everything lands under `data/` (git-ignored).

## Structural mechanics and the other Caltech benchmarks

The seven-problem operator-learning suite of Batlle, Darcy, Hosseini and Owhadi
(*Kernel methods are competitive for operator learning*, JCP 496:112549, 2024)
draws its high-data problems from the record

    Caltech Data:  https://data.caltech.edu/records/20091

which holds `StructuralMechanics`, `Advection`, `Helmholtz` and
`NavierStokes` as `<Name>_inputs.npy` / `<Name>_outputs.npy`. Sizes:

| problem | array shape | file size | regime |
|---------|-------------|----------:|--------|
| Structural mechanics | inputs/outputs (41, 41, 40000) | ~270 MB each | high-data |
| Advection (II) | (200, 40000) | ~64 MB each | high-data |
| Helmholtz | (101, 101, ...) | ~6.6 GB | high-data |
| Navier–Stokes | (64, 64, ...) | ~2.6 GB | high-data |

The low-data problems (Burgers, Darcy, Advection I; 1000 training pairs) come
from the DeepONet/FNO benchmark repositories referenced by Batlle et al.
(`lu-group/deeponet-fno` and the `Zhengyu-Huang/Operator-Learning` data); the
kernel baselines are reproducible from `MatthieuDarcy/KernelsOperatorLearning`.

For structural mechanics, put `StructuralMechanics_inputs.npy` and
`StructuralMechanics_outputs.npy` in `data/`, then run `prep_data.py` (it
verifies the column-constant input structure and writes the 41-dimensional
loads, the stress fields, and the canonical split). For advection, put
`Advection_inputs.npy` / `Advection_outputs.npy` in `data/`.

## OCO-2 radiative-transfer emulation

The emulation data and the kernel-flow emulator's own test predictions of
Lamminpää, Susiluoto, Hobbs, McDuffie, Braverman and Owhadi (*AMT* 18:673–694,
2025) are on the OSF project

    OSF:  https://osf.io/u2t8a

as Julia `.jld` files (JLD2, read directly with `h5py`). Place under
`data/jpl_oco2/`:

| file | contents | size |
|------|----------|-----:|
| `dimred_variables_4_mono.jld` | reduced states and radiances, all bands | ~33 MB |
| `dimred_data_4_mono.jld` | PCA projections and norms for reconstruction | ~9 MB |
| `kf_results_o2_4_mono.jld` | kernel-flow emulator predictions, O2 | ~42 MB |
| `kf_results_wco2_4_mono.jld` | kernel-flow emulator predictions, WCO2 | ~48 MB |
| `kf_results_sco2_4_mono.jld` | kernel-flow emulator predictions, SCO2 | ~48 MB |

## More OCO-2 data for larger experiments

The emulation extract above is dimension-reduced. The underlying OCO-2 Level 2
diagnostic products — retrieved state vectors, measured radiances, and forward
model evaluations — are public through NASA GES DISC:

    GES DISC:  https://disc.gsfc.nasa.gov/datasets?keywords=OCO-2

(Earthdata login required). This is the path to larger or mission-real
emulation experiments beyond the OSF extract.

## Reproducing the numbers

Once the data is in place, `docs/reproduce.md` gives the exact command and
hyperparameters for every number in the paper.
