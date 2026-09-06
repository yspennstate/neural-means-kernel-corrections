"""Mean per-case relative error from complete, nonoverlapping prediction tiles."""
from __future__ import annotations

import math
import numpy as np

from blocked_prediction_pool import PoolError, positive_integer


class RelativeErrorStream:
    def __init__(self, case_ids, n_pixels, *, case_block=512, pixel_block=32):
        case_ids = np.asarray(case_ids)
        if case_ids.ndim != 1 or len(case_ids) == 0 or case_ids.dtype.kind not in "iu":
            raise PoolError("Expected nonempty integer case identities")
        if len(np.unique(case_ids)) != len(case_ids) or np.any(case_ids < 0):
            raise PoolError("Expected unique nonnegative case identities")
        self.case_ids = case_ids.copy()
        self.case_ids.flags.writeable = False
        self.n_pixels = positive_integer(n_pixels, "Pixel count")
        self.case_block = positive_integer(case_block, "Case block")
        self.pixel_block = positive_integer(pixel_block, "Pixel block")
        self.squared_error = np.zeros(len(case_ids), dtype=np.float64)
        self.squared_target = np.zeros(len(case_ids), dtype=np.float64)
        self.covered = np.zeros(((len(case_ids) + self.case_block - 1) // self.case_block,
                                 (self.n_pixels + self.pixel_block - 1) // self.pixel_block), dtype=bool)

    def add(self, case_start, pixel_start, case_ids, prediction, target):
        for value, label in ((case_start, "Case start"), (pixel_start, "Pixel start")):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
                raise PoolError(f"{label} must be a nonnegative integer")
        if case_start >= len(self.case_ids) or case_start % self.case_block or pixel_start >= self.n_pixels or pixel_start % self.pixel_block:
            raise PoolError("Tile start does not match the declared tiling")
        case_stop = min(case_start + self.case_block, len(self.case_ids))
        pixel_stop = min(pixel_start + self.pixel_block, self.n_pixels)
        case_ids = np.asarray(case_ids)
        if case_ids.dtype.kind not in "iu" or not np.array_equal(case_ids, self.case_ids[case_start:case_stop]):
            raise PoolError("Tile case identities or row order differ")
        tile = case_start // self.case_block, pixel_start // self.pixel_block
        if self.covered[tile]:
            raise PoolError("Duplicate metric tile")
        prediction, target = np.asarray(prediction), np.asarray(target)
        shape = case_stop - case_start, pixel_stop - pixel_start
        if prediction.shape != shape or target.shape != shape:
            raise PoolError("Tile shape differs from the declared tiling")
        if prediction.dtype.kind not in "fiu" or target.dtype.kind not in "fiu":
            raise PoolError("Metric tiles must be real numeric arrays")
        prediction, target = prediction.astype(np.float64), target.astype(np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            residual = prediction - target
            error = np.einsum("ij,ij->i", residual, residual)
            norm = np.einsum("ij,ij->i", target, target)
            new_error = self.squared_error[case_start:case_stop] + error
            new_norm = self.squared_target[case_start:case_stop] + norm
        if not np.isfinite(new_error).all() or not np.isfinite(new_norm).all():
            raise PoolError("Nonfinite or overflowing metric tile")
        self.squared_error[case_start:case_stop] = new_error
        self.squared_target[case_start:case_stop] = new_norm
        self.covered[tile] = True

    def finish(self):
        if not self.covered.all():
            raise PoolError(f"Metric coverage incomplete: {np.count_nonzero(self.covered)} of {self.covered.size} tiles")
        if np.any(self.squared_target <= 0):
            raise PoolError("Relative error is undefined for a zero target norm")
        with np.errstate(over="ignore", invalid="ignore"):
            per_case = np.sqrt(self.squared_error) / np.sqrt(self.squared_target)
        if not np.isfinite(per_case).all():
            raise PoolError("Relative error overflowed despite finite squared sums")
        average = math.fsum(float(value) / len(per_case) for value in per_case)
        return dict(mean_relative_error=average, n_cases=len(self.case_ids),
                    n_pixels=self.n_pixels, covered_tiles=int(self.covered.size))
