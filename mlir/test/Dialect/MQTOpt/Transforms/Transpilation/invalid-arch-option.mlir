// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// Instead of applying checks, the routing verifier pass ensures the validity of this program.

// RUN: quantum-opt %s --placement-sc="arch=invalid-127" -verify-diagnostics
// RUN: quantum-opt %s --route-sc="arch=invalid-127"  -verify-diagnostics
// RUN: quantum-opt %s --verify-routing-sc="arch=invalid-127" -verify-diagnostics

// expected-error@unknown {{unsupported architecture}}
module {}
