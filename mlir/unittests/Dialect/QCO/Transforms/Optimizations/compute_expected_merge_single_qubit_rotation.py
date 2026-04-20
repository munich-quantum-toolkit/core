# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Compute expected U gate parameters and global phase for the merge-single-qubit-rotation pass.

Reference script for test_qco_merge_single_qubit_rotation.cpp.
Uses SymPy quaternion algebra to produce ground-truth values.
"""

import math

from sympy import N, Quaternion, cos, pi, sin


def r_gate(theta: float, phi: float) -> Quaternion:
    """Return the SU(2) quaternion for an R(theta, phi) gate."""
    return Quaternion(cos(theta / 2), sin(theta / 2) * cos(phi), sin(theta / 2) * sin(phi), 0)


def u2_gate(phi: float, lam: float) -> Quaternion:
    """Return the SU(2) quaternion for a U2(phi, lambda) gate."""
    return Quaternion.from_euler([phi, pi / 2, lam], "ZYZ")


def normalize_angle(a: float) -> float:
    """Normalize angle to [-pi, pi], matching the pass's normalizeAngle.

    Returns:
        Angle in the range [-pi, pi].
    """
    two_pi = 2 * math.pi
    return a - math.floor((a + math.pi) / two_pi) * two_pi


def angles_from_quaternion(w: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    """ZYZ Euler angles from quaternion, matching anglesFromQuaternion in the pass.

    Returns:
        Tuple (theta, phi, lambda) in ZYZ convention.
    """
    eps = 1e-12

    # Clamp before acos to guard against floating-point drift outside [-1, 1]
    arg = 2 * (w * w + z * z) - 1
    arg = max(-1.0, min(1.0, arg))
    beta = math.acos(arg)

    abs_beta = abs(beta)
    abs_beta_minus_pi = abs(beta - math.pi)

    safe1 = abs_beta >= eps  # not near 0
    safe2 = abs_beta_minus_pi >= eps  # not near pi
    safe = safe1 and safe2

    theta_plus = math.atan2(z, w)
    theta_minus = math.atan2(-x, y)

    if safe:
        alpha = theta_plus + theta_minus
        gamma = theta_plus - theta_minus
    elif not safe1:
        # beta near 0
        alpha = 2 * theta_plus
        gamma = 0.0
    else:
        # beta near pi
        alpha = 2 * theta_minus
        gamma = 0.0

    alpha = normalize_angle(alpha)
    gamma = normalize_angle(gamma)

    # U gate convention: theta=beta, phi=alpha, lambda=gamma
    return beta, alpha, gamma


def global_phase(gate_type: str, *angles: float) -> float:
    """Return the global phase contribution of a gate.

    U = e^{i*phase} * SU(2), this returns 'phase'.

    Returns:
        Global phase in radians.

    Raises:
        ValueError: If gate_type is not a recognized gate.
    """
    if gate_type in {"RX", "RY", "RZ", "R"}:
        return 0.0
    if gate_type == "P":
        return angles[0] / 2
    if gate_type == "U":
        # U(theta, phi, lambda): phase = (phi + lambda) / 2
        _theta, phi, lam = angles
        return (phi + lam) / 2
    if gate_type == "U2":
        # U2(phi, lambda): phase = (phi + lambda) / 2
        phi, lam = angles
        return (phi + lam) / 2
    msg = f"Unknown gate type: {gate_type!r}"
    raise ValueError(msg)


def output_phase(phi: float, lam: float) -> float:
    """Return the intrinsic phase of the synthesized U(theta, phi, lambda).

    Returns:
        Phase in radians.
    """
    return (phi + lam) / 2


def gphase_correction(input_phase: float, phi: float, lam: float) -> float:
    """Return the GPhaseOp correction = total_input_phase - output_UOp_phase.

    Returns:
        Phase correction in radians.
    """
    return input_phase - output_phase(phi, lam)


# ---- Helper to compute merge + gphase for a chain ----
def compute_merge(chain: list[tuple]) -> tuple[float, float, float, float]:
    """Merge a chain of gates into a single U gate with global phase.

    chain: list of (gate_type, quaternion, *angles).

    Uses our own Euler extraction that matches the C++ pass exactly:
    no quaternion sign normalization, same atan2/acos/clamp logic,
    same gimbal-lock handling, same angle normalization.

    Returns:
        Tuple (theta, phi, lambda, gphase) all as floats.
    """
    _, q0, *a0 = chain[0]
    q = q0
    total_input_phase = global_phase(chain[0][0], *a0)

    for entry in chain[1:]:
        gt, qi, *ai = entry
        q = qi.mul(q)  # Hamilton product in circuit order
        total_input_phase += global_phase(gt, *ai)

    # Extract Euler angles matching the pass (no sign normalization)
    w, x, y, z = float(N(q.a)), float(N(q.b)), float(N(q.c)), float(N(q.d))
    theta, phi, lam = angles_from_quaternion(w, x, y, z)

    corr = gphase_correction(total_input_phase, phi, lam)

    return theta, phi, lam, float(N(corr))


# ---- Build gates ----
rx = Quaternion.from_euler([1, 0, 0], "xyz")
ry = Quaternion.from_euler([0, 1, 0], "xyz")
rz = Quaternion.from_euler([0, 0, 1], "xyz")
mx = Quaternion.from_euler([-1, 0, 0], "xyz")
my = Quaternion.from_euler([0, -1, 0], "xyz")
mz = Quaternion.from_euler([0, 0, -1], "xyz")
px = Quaternion.from_euler([pi, 0, 0], "xyz")
py = Quaternion.from_euler([0, pi, 0], "xyz")
pz = Quaternion.from_euler([0, 0, pi], "xyz")
smallx = Quaternion.from_euler([0.001, 0, 0], "xyz")
smally = Quaternion.from_euler([0, 0.001, 0], "xyz")

# P gate has same SU(2) quaternion as RZ
p1 = Quaternion.from_euler([0, 0, 1], "xyz")  # P(1) same rotation as RZ(1)

u1 = Quaternion.from_euler([2, 1, 3], "ZYZ")  # U(1,2,3)
u2 = Quaternion.from_euler([5, 4, 6], "ZYZ")  # U(4,5,6)

u2_12 = u2_gate(1, 2)
u2_34 = u2_gate(3, 4)

r12 = r_gate(1, 2)
r34 = r_gate(3, 4)
r11 = r_gate(1, 1)

cases = [
    ("RX+RX", [("RX", rx), ("RX", rx)]),
    ("RX+RY", [("RX", rx), ("RY", ry)]),
    ("RX+RZ", [("RX", rx), ("RZ", rz)]),
    ("RY+RX", [("RY", ry), ("RX", rx)]),
    ("RY+RY", [("RY", ry), ("RY", ry)]),
    ("RY+RZ", [("RY", ry), ("RZ", rz)]),
    ("RZ+RX", [("RZ", rz), ("RX", rx)]),
    ("RZ+RY", [("RZ", rz), ("RY", ry)]),
    ("RZ+RZ", [("RZ", rz), ("RZ", rz)]),
    ("U+U", [("U", u1, 1.0, 2.0, 3.0), ("U", u2, 4.0, 5.0, 6.0)]),
    ("P+RX", [("P", p1, 1.0), ("RX", rx)]),
    ("R+R", [("R", r12, 1.0, 2.0), ("R", r34, 3.0, 4.0)]),
    ("U2+U2", [("U2", u2_12, 1.0, 2.0), ("U2", u2_34, 3.0, 4.0)]),
    ("RZ+RY+RX pi", [("RZ", pz), ("RY", py), ("RX", px)]),
    ("RY+RZ+RZ-+RY-", [("RY", ry), ("RZ", rz), ("RZ", mz), ("RY", my)]),
    ("small RX+RY", [("RX", smallx), ("RY", smally)]),
    ("RX(pi)+RY(pi)", [("RX", px), ("RY", py)]),
    ("R+R same", [("R", r11, 1.0, 1.0), ("R", r11, 1.0, 1.0)]),
]

if __name__ == "__main__":
    for name, chain in cases:
        theta, phi, lam, gphase = compute_merge(chain)
        print(f"{name}:  U({theta}, {phi}, {lam})  gphase={gphase}")
