import math
import numpy as np
import pytest
from rebot_companion.transforms import Transform


def test_transform_identity():
    t = Transform()
    np.testing.assert_allclose(t.matrix, np.eye(4))
    np.testing.assert_allclose(t.translation, [0, 0, 0])
    np.testing.assert_allclose(t.rotation, np.eye(3))
    assert t.to_euler_deg() == (0.0, 0.0, 0.0)


def test_transform_translation_euler():
    # Translation (1.0, 2.0, 3.0), Roll=90 deg
    t = Transform.from_translation_euler([1.0, 2.0, 3.0], 90.0, 0.0, 0.0)
    np.testing.assert_allclose(t.translation, [1.0, 2.0, 3.0])
    r, p, y = t.to_euler_deg()
    assert math.isclose(r, 90.0, abs_tol=1e-5)
    assert math.isclose(p, 0.0, abs_tol=1e-5)
    assert math.isclose(y, 0.0, abs_tol=1e-5)


def test_transform_inverse():
    t = Transform.from_translation_euler([0.5, -0.2, 0.8], 30.0, -45.0, 60.0)
    t_inv = t.inverse()
    composed = t @ t_inv
    np.testing.assert_allclose(composed.matrix, np.eye(4), atol=1e-7)


def test_transform_apply_point():
    t = Transform.from_translation_euler([10.0, 0.0, 0.0], 0.0, 0.0, 90.0)
    # Point at (1, 0, 0) in local frame
    # Rotated by 90 deg yaw -> (0, 1, 0)
    # Translated by (10, 0, 0) -> (10, 1, 0)
    p_out = t.apply_point([1.0, 0.0, 0.0])
    np.testing.assert_allclose(p_out, [10.0, 1.0, 0.0], atol=1e-7)


def test_dict_serialization():
    t = Transform.from_translation_euler([0.1, 0.2, 0.3], 10.0, 20.0, 30.0)
    d = t.to_dict()
    t_restored = Transform.from_dict(d)
    np.testing.assert_allclose(t.matrix, t_restored.matrix, atol=1e-7)
