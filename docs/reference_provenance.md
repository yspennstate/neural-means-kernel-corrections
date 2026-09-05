# Source locations for the benchmark comparisons

Checked September 5, 2026 against the primary papers. These notes identify
source locations; they do not replace the experiment evidence.

- **Batlle et al. (2024), Table 3** records mechanics errors 5.20% (DeepONet),
  4.76% (FNO), 4.67% (PCA-Net), 4.55% (PARA-Net), 27.11% (linear kernel), and
  5.18% (best Matern/rational quadratic). The neural values come from de Hoop
  et al. Section 4.2.1 describes trapezoidal quadrature for discretized L2.
  [Author manuscript](https://arxiv.org/html/2304.13202v2),
  [journal DOI](https://doi.org/10.1016/j.jcp.2023.112549).
- **de Hoop et al. (2022), Section 4.2.3** describes the composite material,
  the 189-element mesh, 21-point quadratic-element load, and the interpolation
  to the released 41-by-41 grid. The Gaussian fluctuations are restricted to
  spatial-mean-zero functions. [Author manuscript](https://arxiv.org/html/2203.13181v3).
- **Mora et al. (2025), Table 1 and Section 3.1** specify 1250 mechanics
  training cases and 20000 test cases, with 6.49% for the FNO-mean GP.
  Sections 2.3.2 and 3.1 explicitly say that kernel parameters are fixed in
  the neural-mean experiments. Describing that implementation as joint kernel
  and network optimization is inaccurate. [Author manuscript](https://arxiv.org/html/2409.04538v1),
  [journal DOI](https://doi.org/10.1016/j.cma.2024.117581).
- **Lamminpaa et al. (2025)** evaluate an OCO-2 emulator in a fuller physical
  setting. Our two metrics concern the released forty-component
  reconstruction and do not include all of their acceptance criteria.
  [Journal article](https://amt.copernicus.org/articles/18/673/2025/),
  [data record](https://doi.org/10.17605/OSF.IO/U2T8A).
- **Ma et al. (2024)** use split conformal calibration for functional output
  with a spatial-coverage formulation. The whole-output L2 ball in this
  manuscript uses the rank principle but a different coverage event.
  [TMLR paper](https://openreview.net/pdf?id=cGpegxy12T).

Published scalar comparisons retain their source metrics and training
protocols. They cannot establish paired superiority or equivalence without
the corresponding predictions and information about retraining variability.
