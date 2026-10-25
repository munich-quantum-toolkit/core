// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

module attributes {mqt.layout = {initial = array<i64: 1, 0, 2>, input_count = 3 : i64, routing = array<i64: 1, 2, 0>, sites = array<i64: 10, 30, 20>}, mqt.target_env = #mqt.target_env<compilation_target = <sites = [<id = 10>, <id = 30>, <id = 20>], connectivity = explicit, couplings = [<source = 10, target = 30>, <source = 20, target = 30>], native_operations = explicit, operations = [<name = "u", arity = <kind = fixed, value = 1>, num_parameters = 3, site_tuples = []>, <name = "cx", arity = <kind = fixed, value = 2>, num_parameters = 0, site_tuples = []>, <name = "gphase", arity = <kind = fixed, value = 0>, num_parameters = 1, site_tuples = []>, <name = "measure", arity = <kind = fixed, value = 1>, num_parameters = 0, site_tuples = []>]>, payload_specification = <format = <id = "openqasm", version = "3.1.0", profile = "", encoding = text>, capabilities = [], optional_capabilities_known = false>>} {
  func.func @main() attributes {mqt.entry_point} {
    %cst = arith.constant 2.050000e-01 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %cst_1 = arith.constant -1.570796326794897 : f64
    %cst_2 = arith.constant 3.1415926535897931 : f64
    %cst_3 = arith.constant 2.3561944901923448 : f64
    %cst_4 = arith.constant -1.3807963267948966 : f64
    %cst_5 = arith.constant -4.4408920985006262E-16 : f64
    %cst_6 = arith.constant 1.9407963267948964 : f64
    %cst_7 = arith.constant -0.78539816339744739 : f64
    %cst_8 = arith.constant 1.5707963267948968 : f64
    %cst_9 = arith.constant -0.44051066300469843 : f64
    %cst_10 = arith.constant 1.9156838271876464 : f64
    %cst_11 = arith.constant 3.1415926535897927 : f64
    %cst_12 = arith.constant -0.78539816339744783 : f64
    %cst_13 = arith.constant -3.1415926535897931 : f64
    %cst_14 = arith.constant 2.9115926535897927 : f64
    %cst_15 = arith.constant -4.930380657631326E-32 : f64
    %cst_16 = arith.constant 1.570796326794897 : f64
    %cst_17 = arith.constant 4.4408920985006262E-16 : f64
    %cst_18 = arith.constant -4.7123889803846897 : f64
    %cst_19 = arith.constant 1.5707963267948963 : f64
    %cst_20 = arith.constant -2.2204460492503131E-16 : f64
    %cst_21 = arith.constant 1.5707963267948966 : f64
    %q = qco.static 10 : !qco.qubit
    %q_22 = qco.static 30 : !qco.qubit
    %q_23 = qco.static 20 : !qco.qubit
    %q_24 = qco.u(%cst_3, %cst_2, %cst_1) %q_22 : !qco.qubit
    %q_25 = qco.u(%cst_21, %cst_20, %cst_4) %q_23 : !qco.qubit
    %q_26 = qco.u(%cst_6, %cst_5, %cst_21) %q : !qco.qubit
    %controls_out, %targets_out = qco.ctrl(%q_24) targets (%q_44 = %q_26) {
      %q_45 = qco.x %q_44 : !qco.qubit
      qco.yield %q_45 : !qco.qubit
    } : ({!qco.qubit}, {!qco.qubit})
    %q_27 = qco.u(%cst_19, %cst_18, %cst_17) %targets_out : !qco.qubit
    %q_28 = qco.u(%cst_16, %cst_15, %cst_18) %controls_out : !qco.qubit
    %controls_out_29, %targets_out_30 = qco.ctrl(%q_28) targets (%q_44 = %q_27) {
      %q_45 = qco.x %q_44 : !qco.qubit
      qco.yield %q_45 : !qco.qubit
    } : ({!qco.qubit}, {!qco.qubit})
    %q_31 = qco.u(%cst_11, %cst_10, %cst_9) %targets_out_30 : !qco.qubit
    %q_32 = qco.u(%cst_21, %cst_8, %cst_7) %controls_out_29 : !qco.qubit
    %controls_out_33, %targets_out_34 = qco.ctrl(%q_32) targets (%q_44 = %q_25) {
      %q_45 = qco.x %q_44 : !qco.qubit
      qco.yield %q_45 : !qco.qubit
    } : ({!qco.qubit}, {!qco.qubit})
    %q_35 = qco.u(%cst_19, %cst_18, %cst_17) %targets_out_34 : !qco.qubit
    %q_36 = qco.u(%cst_16, %cst_15, %cst_18) %controls_out_33 : !qco.qubit
    %controls_out_37, %targets_out_38 = qco.ctrl(%q_36) targets (%q_44 = %q_35) {
      %q_45 = qco.x %q_44 : !qco.qubit
      qco.yield %q_45 : !qco.qubit
    } : ({!qco.qubit}, {!qco.qubit})
    %q_39 = qco.u(%cst_14, %cst_13, %cst_12) %targets_out_38 : !qco.qubit
    %controls_out_40, %targets_out_41 = qco.ctrl(%q_31) targets (%q_44 = %controls_out_37) {
      %q_45 = qco.x %q_44 : !qco.qubit
      qco.yield %q_45 : !qco.qubit
    } : ({!qco.qubit}, {!qco.qubit})
    qco.sink %q_39 : !qco.qubit
    %q_42 = qco.u(%cst_0, %cst, %cst) %targets_out_41 : !qco.qubit
    qco.sink %controls_out_40 : !qco.qubit
    qco.sink %q_42 : !qco.qubit
    %cst_43 = arith.constant 2.8894042348957698 : f64
    qco.gphase(%cst_43)
    return
  }
}
