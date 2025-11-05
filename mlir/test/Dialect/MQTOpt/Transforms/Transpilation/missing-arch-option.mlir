// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// Instead of applying checks, the routing verifier pass ensures the validity of this program.

// RUN: quantum-opt %s --placement-sc -verify-diagnostics
// RUN: quantum-opt %s --route-sc -verify-diagnostics
// RUN: quantum-opt %s --verify-routing-sc -verify-diagnostics

// expected-error@unknown {{required option 'arch' not provided}}
module {}
