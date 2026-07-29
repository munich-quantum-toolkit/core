// Copyright (c) 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

OPENQASM 3.0;
include "stdgates.inc";

qubit[2] q;
h q[0];
swap q[0], q[1];
