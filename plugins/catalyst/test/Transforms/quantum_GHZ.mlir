// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s --catalystquantum-to-mqtopt | FileCheck %s

module @module {
  func.func public @jit_module() attributes {llvm.emit_c_interface} {
    catalyst.launch_kernel @module_circuit::@circuit() : () -> ()
    return
  }
  module @module_circuit {
    module attributes {transform.with_named_sequence} {
      transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
        %0 = transform.apply_registered_pass "catalystquantum-to-mqtopt" to %arg0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
        %1 = transform.apply_registered_pass "mqtqmap" to %0 {options = "coupling-map=[[0,3],[1,2],[1,3],[2,1],[2,3],[3,1],[3,2]]"} : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
        %2 = transform.apply_registered_pass "mqtopt-to-catalystquantum" to %1 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
        transform.yield 
      }
    }
    func.func public @circuit() attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
      %c0_i64 = arith.constant 0 : i64
      quantum.device shots(%c0_i64) ["/Users/patrickhopf/Code/mqt/mqt-core/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib", "LightningSimulator", "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
      %0 = quantum.alloc( 2) : !quantum.reg
      %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
      %out_qubits = quantum.custom "Hadamard"() %1 : !quantum.bit
      %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
      %out_qubits_0:2 = quantum.custom "CNOT"() %out_qubits, %2 : !quantum.bit, !quantum.bit
      %mres, %out_qubit = quantum.measure %out_qubits_0#0 : i1, !quantum.bit
      %mres_1, %out_qubit_2 = quantum.measure %out_qubits_0#1 : i1, !quantum.bit
      %3 = quantum.insert %0[ 0], %out_qubit : !quantum.reg, !quantum.bit
      %4 = quantum.insert %3[ 1], %out_qubit_2 : !quantum.reg, !quantum.bit
      quantum.dealloc %4 : !quantum.reg
      quantum.device_release
      return
    }
  }
  func.func @setup() {
    quantum.init
    return
  }
  func.func @teardown() {
    quantum.finalize
    return
  }
}

