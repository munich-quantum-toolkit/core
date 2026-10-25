// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

module attributes {mqt.compilation_seed = 42 : i64, mqt.source_qubit_count = 3 : i64, mqt.target_env = #mqt.target_env<compilation_target = <sites = [<id = 0>, <id = 1>, <id = 2>], connectivity = explicit, couplings = [<source = 0, target = 1>, <source = 1, target = 2>], native_operations = explicit, operations = [<name = "u", arity = <kind = fixed, value = 1>, num_parameters = 3, site_tuples = []>, <name = "cx", arity = <kind = fixed, value = 2>, num_parameters = 0, site_tuples = []>, <name = "gphase", arity = <kind = fixed, value = 0>, num_parameters = 1, site_tuples = []>, <name = "measure", arity = <kind = fixed, value = 1>, num_parameters = 0, site_tuples = []>]>, payload_specification = <format = <id = "openqasm", version = "3.1.0", profile = "", encoding = text>, capabilities = [], optional_capabilities_known = false>>} {
  func.func @main() attributes {mqt.entry_point} {
    %cst = arith.constant 4.100000e-01 : f64
    %cst_0 = arith.constant 2.300000e-01 : f64
    %c2 = arith.constant 2 : index
    %cst_1 = arith.constant 1.900000e-01 : f64
    %c1 = arith.constant 1 : index
    %cst_2 = arith.constant 3.700000e-01 : f64
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %q = qtensor.alloc(%c3) {mqt.register_name = "q", mqt.source_qubit_indices = array<i64: 0, 1, 2>} : tensor<3x!qco.qubit>
    %qs, %q_3 = qtensor.extract %q[%c0] : tensor<3x!qco.qubit>
    %q_4 = qco.h %q_3 : !qco.qubit
    %qs_5, %q_6 = qtensor.extract %qs[%c1] : tensor<3x!qco.qubit>
    %q_7 = qco.rx(%cst_2) %q_6 : !qco.qubit
    %qs_8, %q_9 = qtensor.extract %qs_5[%c2] : tensor<3x!qco.qubit>
    %q_10 = qco.rz(%cst_1) %q_9 : !qco.qubit
    %controls_out, %targets_out = qco.ctrl(%q_4) targets (%q_22 = %q_7) {
      %q_23 = qco.x %q_22 : !qco.qubit
      qco.yield %q_23 : !qco.qubit
    } : ({!qco.qubit}, {!qco.qubit})
    %controls_out_11, %targets_out_12 = qco.ctrl(%targets_out) targets (%q_22 = %q_10) {
      %q_23 = qco.x %q_22 : !qco.qubit
      qco.yield %q_23 : !qco.qubit
    } : ({!qco.qubit}, {!qco.qubit})
    %controls_out_13, %targets_out_14 = qco.ctrl(%controls_out) targets (%q_22 = %targets_out_12) {
      %q_23 = qco.x %q_22 : !qco.qubit
      qco.yield %q_23 : !qco.qubit
    } : ({!qco.qubit}, {!qco.qubit})
    %qubit0_out, %qubit1_out = qco.swap %controls_out_13, %controls_out_11 : !qco.qubit, !qco.qubit
    %q_15 = qco.ry(%cst_0) %qubit0_out : !qco.qubit
    %qs_16 = qtensor.insert %q_15 into %qs_8[%c0] : tensor<3x!qco.qubit>
    %qubit0_out_17, %qubit1_out_18 = qco.swap %qubit1_out, %targets_out_14 : !qco.qubit, !qco.qubit
    %qs_19 = qtensor.insert %qubit1_out_18 into %qs_16[%c2] : tensor<3x!qco.qubit>
    %q_20 = qco.rz(%cst) %qubit0_out_17 : !qco.qubit
    %qs_21 = qtensor.insert %q_20 into %qs_19[%c1] : tensor<3x!qco.qubit>
    qtensor.dealloc %qs_21 : tensor<3x!qco.qubit>
    return
  }
}
