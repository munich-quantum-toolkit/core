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
qubit[3] counting;
qubit target;
bit[3] result;

h counting;
x target;

ctrl @ p(theta) counting[0], target;
ctrl @ p(2 * theta) counting[1], target;
ctrl @ p(4 * theta) counting[2], target;

swap counting[0], counting[2];
h counting[0];
cp(-pi / 2) counting[0], counting[1];
h counting[1];
cp(-pi / 4) counting[0], counting[2];
cp(-pi / 2) counting[1], counting[2];
h counting[2];

result[0] = measure counting[2];
result[1] = measure counting[1];
result[2] = measure counting[0];
