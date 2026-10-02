/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

namespace qdmi_test {

inline constexpr auto QASM2_BELL_SAMPLING = R"(
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
)";

inline constexpr auto QASM2_BELL_STATE = R"(
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0], q[1];
)";

inline constexpr auto QASM3_BELL_SAMPLING = R"(
OPENQASM 3;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;
)";

inline constexpr auto QASM3_BELL_STATE = R"(
OPENQASM 3;
include "stdgates.inc";
qubit[2] q;
h q[0];
cx q[0], q[1];
)";

inline constexpr const char* QASM3_MALFORMED = "Definitely not OpenQASM";

} // namespace qdmi_test
