// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// Iterative quantum phase estimation for the phase 3/8 = 0.011 (binary).
//
// QC's reference semantics make it natural to express the hybrid loop without
// explicitly carrying qubits. The QC-to-QCO conversion introduces the linear
// qubit values needed by QCO. The loop carries only the classical feedback
// angle, while a memref stores the estimated bits in MSB-first order.
module {
  func.func @main() {
    %control = qc.static 0 : !qc.qubit
    %target = qc.static 1 : !qc.qubit
    %bits = memref.alloc() : memref<3xi1>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %four = arith.constant 4 : i64
    %theta = arith.constant 2.356194490192345 : f64
    %zero = arith.constant 0.0 : f64
    %two = arith.constant 2.0 : f64
    %minus_pi_over_two = arith.constant -1.5707963267948966 : f64

    // |1> is an eigenstate of P(3*pi/4) with eigenvalue exp(2*pi*i*3/8).
    qc.x %target : !qc.qubit

    %final_feedback = scf.for %round = %c0 to %c3 step %c1
        iter_args(%feedback = %zero) -> f64 {
      qc.reset %control : !qc.qubit
      qc.h %control : !qc.qubit

      // The rounds apply U^4, U^2, and U. Computing the power with a shift
      // keeps the relationship to the algorithm visible to classical passes.
      %round_i64 = arith.index_cast %round : index to i64
      %power_i64 = arith.shrui %four, %round_i64 : i64
      %power = arith.sitofp %power_i64 : i64 to f64
      %angle = arith.mulf %theta, %power : f64
      qc.ctrl(%control) targets(%arg = %target) {
        qc.p(%angle) %arg : !qc.qubit
      qc.yield
      } : {!qc.qubit}, {!qc.qubit}

      // Semiclassical inverse-QFT feedback accumulated from earlier results.
      qc.p(%feedback) %control : !qc.qubit
      qc.h %control : !qc.qubit
      %bit = qc.measure %control : !qc.qubit -> i1
      %bit_index = arith.subi %c3, %c1 : index
      %store_index = arith.subi %bit_index, %round : index
      memref.store %bit, %bits[%store_index] : memref<3xi1>

      // If f is the current correction and m the new bit, the next correction
      // is f/2 - m*pi/2. This yields the usual binary-fraction phase feedback.
      %half_feedback = arith.divf %feedback, %two : f64
      %bit_f64 = arith.uitofp %bit : i1 to f64
      %bit_feedback = arith.mulf %bit_f64, %minus_pi_over_two : f64
      %next_feedback = arith.addf %half_feedback, %bit_feedback : f64
      scf.yield %next_feedback : f64
    }

    memref.dealloc %bits : memref<3xi1>
    return
  }
}
