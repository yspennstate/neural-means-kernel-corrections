"""Illustrative ensemble examples checked with exact rational arithmetic.

This checks the numbers used in lecture diagrams. It is not a measurement of
the trained predictors or a confidence statement about their population risk.
"""
from fractions import Fraction as F
import json
from pathlib import Path

import numpy as np


def two_member(first, second):
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    difference = second-first
    denominator = float(difference @ difference)
    unconstrained = -float(first @ difference)/denominator if denominator else 0.
    weight = float(np.clip(unconstrained, 0, 1))
    point = first+weight*difference
    e1, e2 = float(np.linalg.norm(first)), float(np.linalg.norm(second))
    return dict(weight=weight, point=point.tolist(), squared_error=float(point @ point),
                first_error=e1, second_error=e2,
                correlation=float(first @ second)/(e1*e2) if e1 > 0 and e2 > 0 else None,
                threshold=e1/e2 if e1 > 0 and e2 > 0 else None)


def verify():
    orthogonal = two_member([1, 0], [0, 2])
    exact_weight = F(1, 5)
    exact_error2 = (1-exact_weight)**2+(2*exact_weight)**2
    assert exact_error2 == F(4, 5)
    np.testing.assert_allclose(orthogonal['weight'], float(exact_weight), atol=1e-15)
    np.testing.assert_allclose(orthogonal['squared_error'], float(exact_error2), atol=1e-15)
    aligned = two_member([1, 0], [1.2, .4])
    assert aligned['weight'] == 0 and aligned['squared_error'] == 1
    antiparallel = two_member([1, 0], [-2, 0])
    np.testing.assert_allclose(antiparallel['weight'], 1/3, atol=1e-15)
    assert antiparallel['squared_error'] == 0
    tangent = two_member([1, 0], [1, 1.2])
    assert tangent['weight'] == 0 and tangent['squared_error'] == 1
    assert tangent['correlation'] == tangent['threshold']
    degenerate = [two_member(a, b) for a, b in (
        ([0, 0], [1, 1.2]), ([1, 1.2], [0, 0]), ([0, 0], [0, 0]))]
    for result in degenerate:
        assert result['squared_error'] == 0 and result['point'] == [0., 0.]
        assert result['correlation'] is None and result['threshold'] is None
    identical = two_member([1, 2], [1, 2])
    assert identical['point'] == [1., 2.] and identical['squared_error'] == 5
    # Direct residual averaging versus the second-moment quadratic form.
    residuals = np.array([[[1, 0], [0, 2]], [[.5, .2], [-.4, 1.3]],
                          [[.8, -.2], [.1, 1.1]]], dtype=float)
    weights = np.array([.7, .3])
    gram = np.einsum('imq,ikq->mk', residuals, residuals)/len(residuals)
    direct = float(np.mean(np.sum(np.einsum('m,imq->iq', weights, residuals)**2, axis=1)))
    quadratic = float(weights @ gram @ weights)
    np.testing.assert_allclose(direct, quadratic, atol=1e-15)
    # Centering discards a nonzero common error component in this example.
    centered = residuals-residuals.mean(axis=0, keepdims=True)
    covariance = np.einsum('imq,ikq->mk', centered, centered)/len(centered)
    wrong_centered = float(weights @ covariance @ weights)
    assert abs(direct-wrong_centered) > .1
    return dict(kind='author_algebra_verification', scope='illustrative residual vectors',
                orthogonal=orthogonal, aligned=aligned, antiparallel=antiparallel,
                tangent=tangent, zero_member_cases=degenerate, identical=identical,
                gram_identity=dict(direct=direct, quadratic=quadratic),
                negative_control=dict(wrong_centered_value=wrong_centered,
                                      result='Rejected: covariance is not the required second moment'))


if __name__ == '__main__':
    out = Path(__file__).parent/'assets'/'ensemble_geometry_check.json'
    out.write_text(json.dumps(verify(), indent=2), encoding='utf-8')
    print(out)
