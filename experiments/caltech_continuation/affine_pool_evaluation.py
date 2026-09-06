"""Compose pinned block reads with complete affine-model error evaluation.

This is a computation primitive, with no CLI or launcher. It is not host
admission. The caller must supply already-fitted weights, protect input files,
and establish the predictor/target row mapping from native data provenance.
"""
from __future__ import annotations

import hashlib

import numpy as np

from blocked_prediction_pool import PoolError, positive_integer
from streamed_error import RelativeErrorStream


def positions(values, size, label):
    values = np.asarray(values)
    if values.ndim != 1 or not len(values) or values.dtype.kind not in "iu":
        raise PoolError(f"{label} must be nonempty integer positions")
    if np.any(values < 0) or np.any(values >= size) or len(np.unique(values)) != len(values):
        raise PoolError(f"{label} must be unique in-range positions")
    return values.astype(np.int64)


def array_hash(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(array.tobytes()).hexdigest()


def evaluate_affine_models(pool, targets, *, calibration_rows, evaluation_rows, target_rows,
                           weights, case_block=512, pixel_block=32,
                           max_weight_bytes=64 * 1024 * 1024):
    calibration_rows = positions(calibration_rows, pool.n_cases, "Calibration rows")
    evaluation_rows = positions(evaluation_rows, pool.n_cases, "Evaluation rows")
    target_rows = positions(target_rows, targets.n_cases, "Target rows")
    if len(target_rows) != len(evaluation_rows):
        raise PoolError("Each evaluation row needs exactly one aligned target row")
    if np.intersect1d(calibration_rows, evaluation_rows).size:
        raise PoolError("Calibration and evaluation positions overlap")
    if len(targets.members) != 1 or targets.n_pixels != pool.n_pixels:
        raise PoolError("Targets require one array with the same pixel count")
    case_block = positive_integer(case_block, "Case block")
    pixel_block = positive_integer(pixel_block, "Pixel block")
    max_weight_bytes = positive_integer(max_weight_bytes, "Weight allocation budget")
    if not isinstance(weights, dict) or not weights or any(not isinstance(name, str) or not name for name in weights):
        raise PoolError("Expected named affine weight matrices")
    shape = pool.n_pixels, len(pool.members) + 1
    expected_bytes = len(weights) * shape[0] * shape[1] * 8
    if expected_bytes > max_weight_bytes:
        raise PoolError("Affine weights exceed the explicit weight budget")
    frozen_weights = {}
    for name, matrix in weights.items():
        matrix = np.asarray(matrix)
        if matrix.shape != shape or matrix.dtype.kind not in "fiu" or not np.isfinite(matrix).all():
            raise PoolError(f"Invalid affine weight matrix: {name}")
        frozen_weights[name] = matrix.astype(np.float64, copy=True)
    streams = {name: RelativeErrorStream(target_rows, pool.n_pixels, case_block=case_block, pixel_block=pixel_block)
               for name in frozen_weights}
    for p in range(0, pool.n_pixels, pixel_block):
        stop = min(p + pixel_block, pool.n_pixels)
        for c in range(0, len(evaluation_rows), case_block):
            rows = evaluation_rows[c:c+case_block]
            ids = target_rows[c:c+case_block]
            # Preserve stored predictor precision in the read, matching the
            # historical pool's explicit float32 conversion before regression.
            predictors = pool.block(rows, p, stop, dtype=np.float32)
            truth = targets.block(ids, p, stop, dtype=np.float64)[0]
            for name, matrix in frozen_weights.items():
                w = matrix[p:stop]
                prediction = np.einsum("mnp,pm->np", predictors, w[:, :-1]) + w[:, -1][None]
                streams[name].add(c, p, ids, prediction, truth)
    results = {name: stream.finish() for name, stream in streams.items()}
    # A result is returned only after complete metrics and full content rechecks.
    predictor_hashes = pool.verify_integrity()
    target_hashes = targets.verify_integrity()
    return dict(protocol="streamed-affine-evaluation-v1", results=results,
                calibration_rows_sha256=array_hash(calibration_rows),
                evaluation_rows_sha256=array_hash(evaluation_rows), target_rows_sha256=array_hash(target_rows),
                index_hash_encoding="contiguous native int64 bytes; positions within pinned input arrays",
                predictor_hashes=predictor_hashes, target_hashes=target_hashes,
                weights_sha256={name: array_hash(matrix) for name, matrix in frozen_weights.items()},
                n_members=len(pool.members), n_calibration=len(calibration_rows), n_evaluation=len(evaluation_rows),
                n_pixels=pool.n_pixels, case_block=case_block, pixel_block=pixel_block,
                limitation="Caller must separately attest fitted-weight provenance, input row alignment, host admission and launch/completion provenance")
