"""Exact finite-feature checks behind the lecture's Hilbert-space pictures.

These are illustrative algebra checks, not benchmark performance measurements.
Compare the Gram-system calculation against direct feature-space geometry and
an SVD projection, including redundant design vectors and zero error cases.
"""
import json
from pathlib import Path

import numpy as np


def geometry(design, query, ridge):
    design = np.asarray(design, dtype=float)
    query = np.asarray(query, dtype=float)
    n = len(design)
    gram = design @ design.T
    ku = design @ query
    coefficients = (np.linalg.solve(gram+n*ridge*np.eye(n), ku) if ridge > 0
                    else np.linalg.pinv(gram, rcond=1e-13) @ ku)
    approximation = design.T @ coefficients
    error = query-approximation
    posterior2 = float(query @ query-ku @ coefficients)
    exact2_matrix = posterior2-n*ridge*float(coefficients @ coefficients)
    # Independent projection path: right singular vectors of the design.
    _, singular_values, right = np.linalg.svd(design, full_matrices=False)
    basis = right[singular_values > max(design.shape)*np.finfo(float).eps*singular_values[0]]
    perpendicular = query-basis.T @ (basis @ query)
    interpolation2 = float(perpendicular @ perpendicular)
    exact2_direct = float(error @ error)
    spectral_extra = sum((n*ridge/(s*s+n*ridge))**2*float(v @ query)**2
                         for s, v in zip(singular_values, right) if s > 1e-12)
    np.testing.assert_allclose(exact2_matrix, exact2_direct, atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(exact2_direct, interpolation2+spectral_extra,
                               atol=2e-11, rtol=2e-11)
    assert exact2_direct >= interpolation2-2e-11
    assert posterior2 >= exact2_direct-2e-11
    # The constructed worst-case residual uses its own observed labels.
    radius = 1.7
    residual = radius*error/np.sqrt(exact2_direct) if exact2_direct > 1e-20 else np.zeros_like(query)
    observed = design @ residual
    attained = float(residual @ query-coefficients @ observed)
    np.testing.assert_allclose(attained, radius*np.sqrt(exact2_direct), atol=2e-10)
    # Two functions invisible on the design give the minimax ambiguity.
    invisible = perpendicular/max(np.sqrt(interpolation2), 1e-30)
    np.testing.assert_allclose(design @ invisible, 0, atol=2e-10)
    return dict(coefficients=coefficients.tolist(), query=query.tolist(),
                approximation=approximation.tolist(), error=error.tolist(),
                posterior_squared=posterior2, exact_squared=exact2_direct,
                interpolation_squared=interpolation2, spectral_extra=spectral_extra,
                equality_error=attained, radius=radius)


def verify():
    toy = geometry([[1, 0]], [3, 2], .5)
    assert toy['coefficients'] == [2.] and toy['error'] == [1., 2.]
    assert toy['posterior_squared'] == 7 and toy['exact_squared'] == 5
    assert toy['interpolation_squared'] == 4
    rng = np.random.default_rng(20260905)
    rows = []
    for n, dimension, redundant in ((1, 2, False), (4, 7, False), (5, 7, True)):
        design = rng.normal(size=(n, dimension))
        if redundant:
            design[-1] = design[0]+design[1]
        for ridge in (0., 1e-3, .5, 2.):
            result = geometry(design, rng.normal(size=dimension), ridge)
            rows.append(dict(n=n, dimension=dimension, redundant=redundant,
                             ridge=ridge, **result))
    zero = geometry([[1, 0]], [3, 0], 0.)
    assert zero['exact_squared'] == 0
    # A sign error in the regularization term must be detected.
    wrong_plus = toy['posterior_squared']+.5*4
    assert abs(wrong_plus-toy['exact_squared']) == 4
    return dict(kind='author_algebra_verification', scope='finite-feature illustrative examples',
                exact_toy=toy, cases=rows, zero_error_case=zero,
                negative_control='Wrong plus sign gives nine instead of five; rejected')


if __name__ == '__main__':
    out = Path(__file__).parent/'assets'/'kernel_geometry_check.json'
    out.write_text(json.dumps(verify(), indent=2), encoding='utf-8')
    print(out)
