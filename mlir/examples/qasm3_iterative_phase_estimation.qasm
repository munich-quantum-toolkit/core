// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

OPENQASM 3.0;
include "stdgates.inc";

const float theta = 2.356194490192345;
qubit control;
qubit target;
bit[3] result;

x target;

// MQT Bench generates the iterations as a straight-line Qiskit circuit. The
// corresponding OpenQASM keeps each measurement available to later feedback.
h control;
ctrl @ p(4 * theta) control, target;
h control;
result[2] = measure control;
if (result[2]) {
  x control;
}

h control;
ctrl @ p(2 * theta) control, target;
if (result[2]) {
  rz(-pi / 2) control;
}
h control;
result[1] = measure control;
if (result[1]) {
  x control;
}

h control;
ctrl @ p(theta) control, target;
if (result[1]) {
  rz(-pi / 2) control;
}
if (result[2]) {
  rz(-pi / 4) control;
}
h control;
result[0] = measure control;
