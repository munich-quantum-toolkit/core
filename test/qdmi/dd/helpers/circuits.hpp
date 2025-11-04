/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

namespace qdmi_test {

inline constexpr const char* QASM3_Bell_Sampling = R"(
OPENQASM 3;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;
)";

inline constexpr const char* QASM3_Bell_State = R"(
OPENQASM 3;
include "stdgates.inc";
qubit[2] q;
h q[0];
cx q[0], q[1];
)";

inline constexpr const char* MALFORMED_PROGRAM = "Definitely not OpenQASM";

// A slightly heavier 5-qubit sampling circuit to prolong runtime slightly while
// remaining fast
inline constexpr const char* QASM3_Heavy_Sampling5 = R"(
OPENQASM 3;
include "stdgates.inc";
qubit[5] q;
bit[5] c;
// GHZ-like entanglement chain
h q[0];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
cx q[3], q[4];
// Some single-qubit rotations for additional depth
rx(0.7) q[0];
ry(0.5) q[1];
rz(1.1) q[2];
ry(0.3) q[3];
rx(0.9) q[4];
// Reverse entanglement to add more two-qubit layers
cx q[3], q[4];
cx q[2], q[3];
cx q[1], q[2];
cx q[0], q[1];
// Measure all qubits
c = measure q;
)";

} // namespace qdmi_test
