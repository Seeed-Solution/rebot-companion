"""FK/IK must match the rebot-armd Web cockpit (values from tools/gen_web_fixtures.mjs)."""

import math

import numpy as np
import pytest

from rebot_companion import kinematics as K

# (q_model, [x, y, z, pitch]) from IK.fk in rebot-armd src/api/index.html
FK_JS = [
    ([0, 0, 0, 0], [0.2603, 0, 0.19169999999999998, 0.009552632615765397]),
    ([0.3, -1, -1.5, 0.5], [0.3115091882368852, 0.09636108400813623, 0.5235464339967445, 0.009552632615765397]),
    ([-0.8, -0.4, -2.2, 1.1], [-0.08994299549590407, 0.09260877609919006, 0.6336726277889997, -0.6904473673842347]),
    ([1.7, -2, -0.6, -0.9], [-0.05833767822299141, 0.44900189902215576, 0.02270302204594668, 0.5095526326157653]),
    ([2.9, -0.1, -2.9, 1.5], [0.404831550480472, -0.09975267768842268, 0.4484385708785956, -1.2904473673842345]),
    ([-2.5, -1.4, -1.9, 0], [-0.3148519605046409, -0.23520143482627281, 0.6873285297724072, -0.4904473673842346]),
]

# (target [x, y, z, pitch], prev, ok, q, branch, dist) from IK.solve
IK_JS = [
    (
        [0.15382466384487645, 0.031181803070104513, 0.7076364686838408, -0.39044736738423474],
        [0.25, -0.9, -2.1, 0.7],
        True,
        [0.19999999999999998, -0.9999999999999991, -2.1999999999999993, 0.8],
        1,
        0.18027756377319856,
    ),
    (
        [0.051025096002727977, -0.03490814632463476, 0.5386113894427281, -0.09044736738423469],
        [-0.5, -0.6, -1.7, 1],
        True,
        [-0.5999999999999999, -0.49999999999999956, -1.7999999999999994, 1.1999999999999997],
        1,
        0.26457513110645875,
    ),
    (
        [0.18396850519394803, 0.4731948890547675, 0.2319473110563578, 0.3095526326157655],
        [1, -1.5, -1.1, -0.2],
        True,
        [1.2, -1.5999999999999999, -0.9999999999999998, -0.3],
        1,
        0.2645751311064591,
    ),
]


@pytest.mark.parametrize("q,expected", FK_JS)
def test_fk_matches_js(q, expected):
    p = K.fk(q)
    np.testing.assert_allclose([p.x, p.y, p.z, p.pitch], expected, atol=1e-12)


@pytest.mark.parametrize("q,expected", FK_JS)
def test_tcp_pose_consistent_with_fk(q, expected):
    T = K.tcp_pose(q)
    np.testing.assert_allclose(T.translation, expected[:3], atol=1e-12)
    R = T.rotation
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
    # tool axis (x of the TCP frame) points along the yaw direction, tipped down by pitch + DW
    p4 = expected[3] + K.DW
    np.testing.assert_allclose(
        R[:, 0], [math.cos(q[0]) * math.cos(p4), math.sin(q[0]) * math.cos(p4), -math.sin(p4)], atol=1e-12
    )
    np.testing.assert_allclose(R[:, 1], [-math.sin(q[0]), math.cos(q[0]), 0.0], atol=1e-12)


@pytest.mark.parametrize("target,prev,ok,q,branch,dist", IK_JS)
def test_ik_matches_js(target, prev, ok, q, branch, dist):
    sol = K.solve_ik(K.TcpPoint(*target), prev)
    assert sol.ok is ok
    np.testing.assert_allclose(sol.q, q, atol=1e-9)
    assert sol.branch == branch
    assert sol.dist == pytest.approx(dist, abs=1e-9)


def test_ik_roundtrip_and_errors():
    q = [0.4, -1.1, -1.9, 0.6]
    sol = K.solve_ik(K.fk(q), prev=[0.35, -1.0, -1.8, 0.5])
    np.testing.assert_allclose(sol.q, q, atol=1e-9)
    far = K.solve_ik(K.TcpPoint(2.0, 0.0, 0.2, 0.0), prev=q)
    assert not far.ok and far.err == "reach"


def test_to_model_sign_offset():
    pos = [0.1, -0.5, -1.0, 0.3, 0.0, 0.0, 0.02]
    assert K.to_model(pos) == pytest.approx(pos[:4])
    sign, off = (1, -1, -1, 1), (0.0, 0.1, 0.0, -0.2)
    q = K.to_model(pos, sign, off)
    assert K.to_motor(q, sign, off) == pytest.approx(pos[:4])
    np.testing.assert_allclose(K.tcp_pose_from_joints(pos, sign, off).matrix, K.tcp_pose(q).matrix)
    with pytest.raises(ValueError):
        K.to_model([0.0, 0.0])
