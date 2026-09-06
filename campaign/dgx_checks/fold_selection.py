"""Label-separated selection of a convex/global and pixel-ridge mixture.

The independent evaluation set is not an argument to this module. Each outer
fold fits global weights and chooses the pixel ridge using its training rows.
Arrays have member, case, pixel order. Pixel blocking bounds temporary arrays.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def block_size(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError("Pixel block size must be a positive integer")
    return int(value)


def validate(P, Y):
    P, Y = np.asarray(P), np.asarray(Y)
    if P.ndim != 3 or Y.shape != P.shape[1:] or min(P.shape) < 1:
        raise ValueError("Expected predictors (member, case, pixel) and targets (case, pixel)")
    if not np.isfinite(P).all() or not np.isfinite(Y).all():
        raise ValueError("Predictors and targets must be finite")
    if np.any(np.linalg.norm(Y, axis=1) == 0):
        raise ValueError("Relative error is undefined for a zero target")
    return P, Y


def grid(values, name, low=0., high=np.inf, strict_low=False):
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1 or len(v) == 0 or not np.isfinite(v).all() or len(np.unique(v)) != len(v):
        raise ValueError(f"Invalid {name} grid")
    if np.any(v <= low if strict_low else v < low) or np.any(v > high):
        raise ValueError(f"Out-of-range {name} grid")
    return v


def convex_weights(P, Y, pixel_block=32):
    pixel_block = block_size(pixel_block)
    P, Y = validate(P, Y)
    m, n, d = P.shape
    norms = np.linalg.norm(Y, axis=1)
    S = np.zeros((m, m))
    for begin in range(0, d, pixel_block):
        end = min(begin + pixel_block, d)
        residual = (P[:, :, begin:end].astype(np.float64) - Y[None, :, begin:end]) / norms[None, :, None]
        S += np.einsum("mnd,knd->mk", residual, residual) / n
    result = minimize(lambda w: w @ S @ w, np.ones(m) / m, jac=lambda w: 2 * S @ w,
                      bounds=[(0., 1.)] * m, constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                      method="SLSQP", options=dict(maxiter=3000, ftol=1e-15))
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Convex fit did not converge: {result.message}")
    weights = np.maximum(result.x, 0.)
    weights /= weights.sum()
    value = weights @ S @ weights
    # Both a feasible vertex and the equal mixture are independently available controls.
    if value > min(np.diag(S).min(), np.mean(S)) + 1e-10:
        raise RuntimeError("Convex fit is worse than an available feasible mixture")
    gradient = 2 * S @ weights
    # Frank-Wolfe gap is a primal suboptimality bound for this convex simplex objective.
    gap = float(weights @ gradient - gradient.min())
    if gap > 1e-7 * max(1., float(np.max(np.abs(S)))):
        raise RuntimeError(f"Convex fit has unresolved optimality gap {gap}")
    return weights


def augmented(P):
    m, n, d = P.shape
    return np.concatenate([P.astype(np.float64), np.ones((1, n, d))], axis=0).transpose(2, 1, 0)


def ridge_weights(P, Y, lam):
    X = augmented(P)
    n = P.shape[1]
    gram = np.einsum("dnm,dnk->dmk", X, X) / n
    rhs = np.einsum("dnm,nd->dm", X, Y) / n
    return np.linalg.solve(gram + float(lam) * np.eye(X.shape[2])[None], rhs[..., None])[..., 0]


def pixel_prediction(P, weights):
    return np.einsum("dnm,dm->nd", augmented(P), weights)


def loo_ridge_scores(P, Y, lambdas, pixel_block=32):
    """Exact linear-smoother LOO on training rows, with the original penalized intercept."""
    pixel_block = block_size(pixel_block)
    P, Y = validate(P, Y)
    lambdas = grid(lambdas, "ridge", strict_low=True)
    _, n, d = P.shape
    scores = np.zeros(len(lambdas))
    for begin in range(0, d, pixel_block):
        end = min(begin + pixel_block, d)
        X = augmented(P[:, :, begin:end])
        # SVD avoids squaring the design condition number before resolving
        # the small-eigenvalue directions; the intercept remains penalized.
        U, singular, _ = np.linalg.svd(X, full_matrices=False)
        projected_y = np.einsum("dnm,nd->dm", U, Y[:, begin:end])
        for idx, lam in enumerate(lambdas):
            gain = singular ** 2 / (singular ** 2 + n * lam)
            prediction = np.einsum("dnm,dm->nd", U, projected_y * gain)
            hat = np.einsum("dnm,dm->nd", U ** 2, gain)
            if np.any(1 - hat <= 1e-12):
                raise RuntimeError("LOO leverage is numerically singular")
            residual = (Y[:, begin:end] - prediction) / (1 - hat)
            scores[idx] += np.einsum("nd,nd->", residual, residual) / n
    if not np.isfinite(scores).all():
        raise RuntimeError("Nonfinite LOO scores")
    return scores


def fold_candidates(P_train, Y_train, P_hold, lambdas, fractions, pixel_block=32):
    """Fit all candidates without accepting held-out labels. This is the causal boundary."""
    pixel_block = block_size(pixel_block)
    P_train, Y_train = validate(P_train, Y_train)
    lambdas = grid(lambdas, "ridge", strict_low=True)
    fractions = grid(fractions, "shrinkage", high=1.)
    if P_hold.ndim != 3 or (P_hold.shape[0], P_hold.shape[2]) != (P_train.shape[0], P_train.shape[2]):
        raise ValueError("Held-out prediction shape differs from training")
    if not np.isfinite(P_hold).all():
        raise ValueError("Held-out predictions must be finite")
    weights = convex_weights(P_train, Y_train, pixel_block)
    scores = loo_ridge_scores(P_train, Y_train, lambdas, pixel_block)
    selected = float(lambdas[int(np.argmin(scores))])
    d = P_train.shape[2]
    predictions = np.empty((len(fractions), P_hold.shape[1], d), dtype=np.float64)
    for begin in range(0, d, pixel_block):
        end = min(begin + pixel_block, d)
        W = ridge_weights(P_train[:, :, begin:end], Y_train[:, begin:end], selected)
        pixel = pixel_prediction(P_hold[:, :, begin:end], W)
        global_prediction = np.einsum("m,mnd->nd", weights, P_hold[:, :, begin:end])
        predictions[:, :, begin:end] = ((1 - fractions[:, None, None]) * pixel[None]
                                        + fractions[:, None, None] * global_prediction[None])
    return predictions, dict(ridge_lambda=selected, convex_weights=weights.tolist(),
                              ridge_loo_scores=scores.tolist(), n_training=P_train.shape[1])


def select_shrinkage(P, Y, lambdas, fractions=None, n_splits=5, seed=1, pixel_block=32):
    pixel_block = block_size(pixel_block)
    P, Y = validate(P, Y)
    if fractions is None:
        fractions = np.linspace(0, 1, 21)
    fractions = grid(fractions, "shrinkage", high=1.)
    n = P.shape[1]
    if not isinstance(n_splits, int) or not 2 <= n_splits <= n:
        raise ValueError("Invalid fold count")
    folds = np.array_split(np.random.default_rng(seed).permutation(n), n_splits)
    sum_errors = np.zeros(len(fractions))
    records = []
    for held in folds:
        train = np.setdiff1d(np.arange(n), held)
        predictions, record = fold_candidates(P[:, train], Y[train], P[:, held], lambdas, fractions, pixel_block)
        errors = np.linalg.norm(predictions - Y[None, held], axis=2) / np.linalg.norm(Y[held], axis=1)[None]
        sum_errors += errors.sum(axis=1)
        record.update(held_rows=held.tolist(), training_rows=train.tolist(),
                      mean_relative_error=errors.mean(axis=1).tolist())
        records.append(record)
    means = sum_errors / n
    selected = float(fractions[int(np.argmin(means))])
    return selected, dict(protocol="nested-training-fold-only-ridge-and-convex-v1", n_calibration=n,
                          n_splits=n_splits, seed=seed, fractions=fractions.tolist(),
                          mean_relative_error=means.tolist(), selected_fraction=selected, folds=records)
