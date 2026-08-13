// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// Three-bit quantum phase estimation for the phase 3/8 = 0.011 (binary).
module {
  func.func @main() {
    %q0 = qc.static 0 : !qc.qubit
    %q1 = qc.static 1 : !qc.qubit
    %q2 = qc.static 2 : !qc.qubit
    %target = qc.static 3 : !qc.qubit

    %theta = arith.constant 2.356194490192345 : f64
    %two_theta = arith.constant 4.71238898038469 : f64
    %four_theta = arith.constant 9.42477796076938 : f64
    %minus_pi_over_two = arith.constant -1.5707963267948966 : f64
    %minus_pi_over_four = arith.constant -0.7853981633974483 : f64

    // Prepare the counting register and the |1> eigenstate of P(3*pi/4).
    qc.h %q0 : !qc.qubit
    qc.h %q1 : !qc.qubit
    qc.h %q2 : !qc.qubit
    qc.x %target : !qc.qubit

    // Applying U, U^2, and U^4 encodes the eigenphase.
    qc.ctrl(%q0) targets(%arg = %target) {
      qc.p(%theta) %arg : !qc.qubit
      qc.yield
    } : {!qc.qubit}, {!qc.qubit}
    qc.ctrl(%q1) targets(%arg = %target) {
      qc.p(%two_theta) %arg : !qc.qubit
      qc.yield
    } : {!qc.qubit}, {!qc.qubit}
    qc.ctrl(%q2) targets(%arg = %target) {
      qc.p(%four_theta) %arg : !qc.qubit
      qc.yield
    } : {!qc.qubit}, {!qc.qubit}

    // Inverse QFT on the counting register.
    qc.swap %q0, %q2 : !qc.qubit, !qc.qubit
    qc.h %q0 : !qc.qubit
    qc.ctrl(%q0) targets(%arg = %q1) {
      qc.p(%minus_pi_over_two) %arg : !qc.qubit
      qc.yield
    } : {!qc.qubit}, {!qc.qubit}
    qc.h %q1 : !qc.qubit
    qc.ctrl(%q0) targets(%arg = %q2) {
      qc.p(%minus_pi_over_four) %arg : !qc.qubit
      qc.yield
    } : {!qc.qubit}, {!qc.qubit}
    qc.ctrl(%q1) targets(%arg = %q2) {
      qc.p(%minus_pi_over_two) %arg : !qc.qubit
      qc.yield
    } : {!qc.qubit}, {!qc.qubit}
    qc.h %q2 : !qc.qubit

    // Measure in most-significant-bit-first order.
    %bit0 = qc.measure %q2 : !qc.qubit -> i1
    %bit1 = qc.measure %q1 : !qc.qubit -> i1
    %bit2 = qc.measure %q0 : !qc.qubit -> i1
    return
  }
}
