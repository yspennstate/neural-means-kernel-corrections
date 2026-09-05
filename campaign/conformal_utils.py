"""Finite-sample split-conformal rank, including the infinite endpoint."""
import math

import numpy as np


def conformal_quantile(scores, alpha):
    """Return order statistic ceil((m+1)(1-alpha)), with score m+1 = inf.

    This arithmetic does not establish exchangeability or independence from
    model selection; callers must justify those assumptions separately.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Calibration requires a nonempty vector of finite scores")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("Miscoverage alpha must lie strictly between zero and one")
    k = math.ceil((len(values) + 1) * (1 - alpha))
    return float(np.partition(values, k - 1)[k - 1]) if k <= len(values) else math.inf
