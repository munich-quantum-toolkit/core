// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file --qir-to-mqtref | FileCheck %s

// This test checks if the alloc register call is correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertAllocRegister()
    llvm.func @testConvertAllocRegister() attributes {passthrough = ["entry_point", ["output_labeling_schema", "schema_id"], ["qir_profiles", "base_profile"], ["required_num_qubits", "2"], ["required_num_results", "0"], ["qir_major_version", "1"], ["qir_minor_version", "0"], ["dynamic_qubit_management", "true"], ["dynamic_result_management", "false"]]}  {
        // CHECK: %[[Int:.*]] = llvm.mlir.constant(14 : i64) : i64
        // CHECK: %[[Idx:.*]] = arith.index_cast %[[Int]] : i64 to index
        // CHECK: %[[r_0:.*]] = memref.alloc(%[[Idx]]) : memref<?x!mqtref.Qubit>

        %0 = llvm.mlir.zero : !llvm.ptr
        %c0 = llvm.mlir.constant(14 : i64) : i64
        llvm.call @__quantum__rt__initialize(%0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        %r = llvm.call @__quantum__rt__qubit_allocate_array(%c0) : (i64) -> !llvm.ptr
        llvm.br ^bb2
      ^bb2:
        llvm.br ^bb3
      ^bb3:
        llvm.return
    }

    llvm.func @__quantum__rt__initialize(!llvm.ptr)
    llvm.func @__quantum__rt__qubit_allocate_array(i64) -> !llvm.ptr
}

// -----
// This test checks if the dealloc register call is correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertDeallocRegister()
    llvm.func @testConvertDeallocRegister() attributes {passthrough = ["entry_point", ["output_labeling_schema", "schema_id"], ["qir_profiles", "base_profile"], ["required_num_qubits", "2"], ["required_num_results", "0"], ["qir_major_version", "1"], ["qir_minor_version", "0"], ["dynamic_qubit_management", "true"], ["dynamic_result_management", "false"]]}  {
        // CHECK: %[[Int:.*]] = llvm.mlir.constant(14 : i64) : i64
        // CHECK: %[[Idx:.*]] = arith.index_cast %[[Int]] : i64 to index
        // CHECK: %[[r_0:.*]] = memref.alloc(%[[Idx]]) : memref<?x!mqtref.Qubit>
        // CHECK: memref.dealloc %[[r_0:.*]] : memref<?x!mqtref.Qubit>

        %0 = llvm.mlir.zero : !llvm.ptr
        %c0 = llvm.mlir.constant(14 : i64) : i64
        llvm.call @__quantum__rt__initialize(%0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        %r0 = llvm.call @__quantum__rt__qubit_allocate_array(%c0) : (i64) -> !llvm.ptr
        llvm.call @__quantum__rt__qubit_release_array(%r0) : (!llvm.ptr) -> ()
        llvm.br ^bb2
      ^bb2:
        llvm.br ^bb3
      ^bb3:
        llvm.return
    }
    llvm.func @__quantum__rt__qubit_release_array(!llvm.ptr)
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
    llvm.func @__quantum__rt__qubit_allocate_array(i64) -> !llvm.ptr
}

// -----
// This test checks if the extract from register call is correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertExtractFromRegister()
    llvm.func @testConvertExtractFromRegister() attributes {passthrough = ["entry_point", ["output_labeling_schema", "schema_id"], ["qir_profiles", "base_profile"], ["required_num_qubits", "2"], ["required_num_results", "0"], ["qir_major_version", "1"], ["qir_minor_version", "0"], ["dynamic_qubit_management", "true"], ["dynamic_result_management", "false"]]}  {
        // CHECK: %[[Int1:.*]] = llvm.mlir.constant(1 : i64) : i64
        // CHECK: %[[Int0:.*]] = llvm.mlir.constant(0 : i64) : i64
        // CHECK: %[[Idx1:.*]] = arith.index_cast %[[Int1]] : i64 to index
        // CHECK: %[[r_0:.*]] = memref.alloc(%[[Idx1]]) : memref<?x!mqtref.Qubit>
        // CHECK: %[[Idx0:.*]] = arith.index_cast %[[Int0]] : i64 to index
        // CHECK: %[[q_0:.*]] = memref.load %[[r_0]][%[[Idx0]]] : memref<?x!mqtref.Qubit>

        %0 = llvm.mlir.zero : !llvm.ptr
        %c0 = llvm.mlir.constant(1 : i64) : i64
        %c1 = llvm.mlir.constant(0 : i64) : i64
        llvm.call @__quantum__rt__initialize(%0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        %r0 = llvm.call @__quantum__rt__qubit_allocate_array(%c0) : (i64) -> !llvm.ptr
        %ptr1 = llvm.call @__quantum__rt__array_get_element_ptr_1d(%r0, %c1) : (!llvm.ptr, i64) -> !llvm.ptr
        %q0 = llvm.load %ptr1 : !llvm.ptr -> !llvm.ptr
        llvm.br ^bb2
      ^bb2:
        llvm.br ^bb3
      ^bb3:
        llvm.return
    }
    llvm.func @__quantum__rt__array_get_element_ptr_1d(!llvm.ptr, i64) -> !llvm.ptr
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
    llvm.func @__quantum__rt__qubit_allocate_array(i64) -> !llvm.ptr
}

// -----
// This test checks if the alloc qubit call is correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertAllocateQubit()
    llvm.func @testConvertAllocateQubit() attributes {passthrough = ["entry_point", ["output_labeling_schema", "schema_id"], ["qir_profiles", "base_profile"], ["required_num_qubits", "2"], ["required_num_results", "0"], ["qir_major_version", "1"], ["qir_minor_version", "0"], ["dynamic_qubit_management", "true"], ["dynamic_result_management", "false"]]}  {
      // CHECK: %[[q_0:.*]] = mqtref.allocQubit
      // CHECK: %[[q_1:.*]] = mqtref.allocQubit

        %0 = llvm.mlir.zero : !llvm.ptr
        llvm.call @__quantum__rt__initialize(%0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        %q0 = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
        %q1 = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
        llvm.br ^bb2
      ^bb2:
        llvm.br ^bb3
      ^bb3:
        llvm.return
    }
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
    llvm.func @__quantum__rt__qubit_allocate() -> !llvm.ptr
}

// -----
// This test checks if the dealloc qubit call is correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertDeallocateQubit()
    llvm.func @testConvertDeallocateQubit() attributes {passthrough = ["entry_point", ["output_labeling_schema", "schema_id"], ["qir_profiles", "base_profile"], ["required_num_qubits", "2"], ["required_num_results", "0"], ["qir_major_version", "1"], ["qir_minor_version", "0"], ["dynamic_qubit_management", "true"], ["dynamic_result_management", "false"]]}  {
      // CHECK: %[[q_0:.*]] = mqtref.allocQubit
      // CHECK: %[[q_1:.*]] = mqtref.allocQubit
      // CHECK: mqtref.deallocQubit %[[q_0]]
      // CHECK: mqtref.deallocQubit %[[q_1]]

        %0 = llvm.mlir.zero : !llvm.ptr
        llvm.call @__quantum__rt__initialize(%0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        %q0 = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
        %q1 = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
        llvm.br ^bb2
      ^bb2:
        llvm.call @__quantum__rt__qubit_release(%q0) : (!llvm.ptr) -> ()
        llvm.call @__quantum__rt__qubit_release(%q1) : (!llvm.ptr) -> ()
        llvm.br ^bb3
      ^bb3:
        llvm.return
    }
    llvm.func @__quantum__rt__qubit_release(!llvm.ptr)
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
    llvm.func @__quantum__rt__qubit_allocate() -> !llvm.ptr
}

// -----
// This test checks if the reset operation is correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertResetOp()
    llvm.func @testConvertResetOp() attributes {passthrough = ["entry_point", ["output_labeling_schema", "schema_id"], ["qir_profiles", "base_profile"], ["required_num_qubits", "2"], ["required_num_results", "0"], ["qir_major_version", "1"], ["qir_minor_version", "0"], ["dynamic_qubit_management", "true"], ["dynamic_result_management", "false"]]}  {
        // CHECK: %[[q_0:.*]] = mqtref.allocQubit
        // CHECK: mqtref.reset %[[q_0]]

        %0 = llvm.mlir.zero : !llvm.ptr
        llvm.call @__quantum__rt__initialize(%0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        %q0 = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
        llvm.call @__quantum__qis__reset__body(%q0) : (!llvm.ptr) -> ()
        llvm.br ^bb2
      ^bb2:
        llvm.call @__quantum__rt__qubit_release(%q0) : (!llvm.ptr) -> ()
        llvm.br ^bb3
      ^bb3:
        llvm.return
    }
    llvm.func @__quantum__qis__reset__body(!llvm.ptr)
    llvm.func @__quantum__rt__qubit_release(!llvm.ptr)
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
    llvm.func @__quantum__rt__qubit_allocate() -> !llvm.ptr
}

// -----
// This test checks if the reset operation using static qubits is correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertResetOpStatic()
    llvm.func @testConvertResetOpStatic() attributes {passthrough = ["entry_point", ["output_labeling_schema", "schema_id"], ["qir_profiles", "base_profile"], ["required_num_qubits", "2"], ["required_num_results", "0"], ["qir_major_version", "1"], ["qir_minor_version", "0"], ["dynamic_qubit_management", "true"], ["dynamic_result_management", "false"]]}  {
        // CHECK: %[[q_0:.*]] = mqtref.qubit 0
        // CHECK: mqtref.reset %[[q_0]]

        %0 = llvm.mlir.zero : !llvm.ptr
        llvm.call @__quantum__rt__initialize(%0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        llvm.call @__quantum__qis__reset__body(%0) : (!llvm.ptr) -> ()
        llvm.br ^bb2
      ^bb2:
        llvm.br ^bb3
      ^bb3:
        llvm.return
    }
    llvm.func @__quantum__qis__reset__body(!llvm.ptr)
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
}

// -----
// This test checks if the measure operation is correctly converted
module {
    llvm.mlir.global internal constant @mlir.llvm.nameless_global_0("r0\00") {addr_space = 0 : i32, dso_local}
    // CHECK-LABEL: llvm.func @testConvertMeasure()
    llvm.func @testConvertMeasure() attributes {passthrough = ["entry_point", ["output_labeling_schema", "schema_id"], ["qir_profiles", "base_profile"], ["required_num_qubits", "2"], ["required_num_results", "1"], ["qir_major_version", "1"], ["qir_minor_version", "0"], ["dynamic_qubit_management", "true"], ["dynamic_result_management", "false"]]}  {
        // CHECK: %[[mem:.*]] = memref.alloca() : memref<1xi1>
        // CHECK: %[[q_0:.*]] = mqtref.allocQubit
        // CHECK: %[[m_0:.*]] = mqtref.measure %[[q_0]]
        // CHECK: %[[c_0:.*]] = arith.constant 0 : index
        // CHECK: memref.store %[[m_0]], %[[mem]][%[[c_0]]] : memref<1xi1>

        %0 = llvm.mlir.zero : !llvm.ptr
        %a0 = llvm.mlir.addressof @mlir.llvm.nameless_global_0 : !llvm.ptr
        llvm.call @__quantum__rt__initialize(%0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        %q0 = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
        llvm.br ^bb2
      ^bb2:
        llvm.call @__quantum__qis__mz__body(%q0, %0) : (!llvm.ptr, !llvm.ptr)  -> ()
        llvm.call @__quantum__rt__qubit_release(%q0) : (!llvm.ptr) -> ()
        llvm.br ^bb3
      ^bb3:
        llvm.call @__quantum__rt__result_record_output(%0, %a0) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.return
    }
    llvm.func @__quantum__qis__mz__body(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__rt__result_record_output(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
    llvm.func @__quantum__rt__qubit_release(!llvm.ptr)
    llvm.func @__quantum__rt__qubit_allocate() -> !llvm.ptr
}

// -----
// This test checks if the measure operation using static qubits is correctly converted
module {
    llvm.mlir.global internal constant @mlir.llvm.nameless_global_0("r0\00") {addr_space = 0 : i32, dso_local}
    // CHECK-LABEL: llvm.func @testConvertMeasureStatic()
    llvm.func @testConvertMeasureStatic() attributes {passthrough = ["entry_point", ["output_labeling_schema", "schema_id"], ["qir_profiles", "base_profile"], ["required_num_qubits", "2"], ["required_num_results", "1"], ["qir_major_version", "1"], ["qir_minor_version", "0"], ["dynamic_qubit_management", "true"], ["dynamic_result_management", "false"]]}  {
        // CHECK: %[[mem:.*]] = memref.alloca() : memref<1xi1>
        // CHECK: %[[q_0:.*]] = mqtref.qubit 0
        // CHECK: %[[m_0:.*]] = mqtref.measure %[[q_0]]
        // CHECK: %[[c_0:.*]] = arith.constant 0 : index
        // CHECK: memref.store %[[m_0]], %[[mem]][%[[c_0]]] : memref<1xi1>

        %0 = llvm.mlir.zero : !llvm.ptr
        %a0 = llvm.mlir.addressof @mlir.llvm.nameless_global_0 : !llvm.ptr
        llvm.call @__quantum__rt__initialize(%0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        llvm.br ^bb2
      ^bb2:
        llvm.call @__quantum__qis__mz__body(%0, %0) : (!llvm.ptr, !llvm.ptr)  -> ()
        llvm.br ^bb3
      ^bb3:
        llvm.call @__quantum__rt__result_record_output(%0, %a0) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.return
    }
    llvm.func @__quantum__qis__mz__body(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__rt__result_record_output(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
}

// -----
// This test checks if the measure operation is correctly converted that store the classical result on the same index
module {
    llvm.mlir.global internal constant @mlir.llvm.nameless_global_0("r0\00") {addr_space = 0 : i32, dso_local}
    // CHECK-LABEL: llvm.func @testConvertMeasureOnSameIndex()
    llvm.func @testConvertMeasureOnSameIndex() attributes {passthrough = ["entry_point", ["output_labeling_schema", "schema_id"], ["qir_profiles", "base_profile"], ["required_num_qubits", "2"], ["required_num_results", "1"], ["qir_major_version", "1"], ["qir_minor_version", "0"], ["dynamic_qubit_management", "true"], ["dynamic_result_management", "false"]]}  {
        // CHECK: %[[mem:.*]] = memref.alloca() : memref<1xi1>
        // CHECK: %[[q_0:.*]] = mqtref.allocQubit
        // CHECK: %[[q_1:.*]] = mqtref.allocQubit
        // CHECK: %[[m_0:.*]] = mqtref.measure %[[q_0]]
        // CHECK: %[[m_1:.*]] = mqtref.measure %[[q_1]]
        // CHECK: %[[c_0:.*]] = arith.constant 0 : index
        // CHECK: memref.store %[[m_0]], %[[mem]][%[[c_0]]] : memref<1xi1>
        // CHECK: %[[c_1:.*]] = arith.constant 0 : index
        // CHECK: memref.store %[[m_1]], %[[mem]][%[[c_1]]] : memref<1xi1>

        %ptr0 = llvm.mlir.zero : !llvm.ptr
        %c0 = llvm.mlir.constant(1 : i64) : i64
        %ptr1 = llvm.inttoptr %c0 : i64 to !llvm.ptr
        %a0 = llvm.mlir.addressof @mlir.llvm.nameless_global_0 : !llvm.ptr
        llvm.call @__quantum__rt__initialize(%ptr0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        %q0 = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
        %q1 = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
        llvm.br ^bb2
      ^bb2:
        llvm.call @__quantum__qis__mz__body(%q0, %ptr0) : (!llvm.ptr, !llvm.ptr)  -> ()
        llvm.call @__quantum__qis__mz__body(%q1, %ptr1) : (!llvm.ptr, !llvm.ptr)  -> ()
        llvm.call @__quantum__rt__qubit_release(%q0) : (!llvm.ptr) -> ()
        llvm.call @__quantum__rt__qubit_release(%q1) : (!llvm.ptr) -> ()
        llvm.br ^bb3
      ^bb3:
        llvm.call @__quantum__rt__result_record_output(%ptr0, %a0) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.call @__quantum__rt__result_record_output(%ptr1, %a0) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.return
    }
    llvm.func @__quantum__qis__mz__body(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__rt__result_record_output(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
    llvm.func @__quantum__rt__qubit_release(!llvm.ptr)
    llvm.func @__quantum__rt__qubit_allocate() -> !llvm.ptr
}

// -----
// This test checks if the measure operation using static qubits is correctly converted that store the classical result on the same index
module {
    llvm.mlir.global internal constant @mlir.llvm.nameless_global_0("r0\00") {addr_space = 0 : i32, dso_local}
    // CHECK-LABEL: llvm.func @testConvertMeasureOnSameIndexStatic()
    llvm.func @testConvertMeasureOnSameIndexStatic() attributes {passthrough = ["entry_point", ["output_labeling_schema", "schema_id"], ["qir_profiles", "base_profile"], ["required_num_qubits", "2"], ["required_num_results", "1"], ["qir_major_version", "1"], ["qir_minor_version", "0"], ["dynamic_qubit_management", "true"], ["dynamic_result_management", "false"]]}  {
        // CHECK: %[[mem:.*]] = memref.alloca() : memref<1xi1>
        // CHECK: %[[q_0:.*]] = mqtref.qubit 0
        // CHECK: %[[q_1:.*]] = mqtref.qubit 1
        // CHECK: %[[m_0:.*]] = mqtref.measure %[[q_0]]
        // CHECK: %[[m_1:.*]] = mqtref.measure %[[q_1]]
        // CHECK: %[[c_0:.*]] = arith.constant 0 : index
        // CHECK: memref.store %[[m_0]], %[[mem]][%[[c_0]]] : memref<1xi1>
        // CHECK: %[[c_1:.*]] = arith.constant 0 : index
        // CHECK: memref.store %[[m_1]], %[[mem]][%[[c_1]]] : memref<1xi1>

        %ptr0 = llvm.mlir.zero : !llvm.ptr
        %c0 = llvm.mlir.constant(1 : i64) : i64
        %ptr1 = llvm.inttoptr %c0 : i64 to !llvm.ptr
        %a0 = llvm.mlir.addressof @mlir.llvm.nameless_global_0 : !llvm.ptr
        llvm.call @__quantum__rt__initialize(%ptr0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        llvm.br ^bb2
      ^bb2:
        llvm.call @__quantum__qis__mz__body(%ptr0, %ptr0) : (!llvm.ptr, !llvm.ptr)  -> ()
        llvm.call @__quantum__qis__mz__body(%ptr1, %ptr1) : (!llvm.ptr, !llvm.ptr)  -> ()
        llvm.br ^bb3
      ^bb3:
        llvm.call @__quantum__rt__result_record_output(%ptr0, %a0) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.call @__quantum__rt__result_record_output(%ptr1, %a0) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.return
    }
    llvm.func @__quantum__qis__mz__body(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__rt__result_record_output(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
}

// -----
// This test checks if the single qubit gates are correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertSingleQubitOp()
    llvm.func @testConvertSingleQubitOp() attributes {passthrough = ["entry_point", ["output_labeling_schema", "schema_id"], ["qir_profiles", "base_profile"], ["required_num_qubits", "2"], ["required_num_results", "0"], ["qir_major_version", "1"], ["qir_minor_version", "0"], ["dynamic_qubit_management", "true"], ["dynamic_result_management", "false"]]}  {
        // CHECK: %[[q_0:.*]] = mqtref.allocQubit
        // CHECK: mqtref.h() %[[q_0]]
        // CHECK: mqtref.i() %[[q_0]]
        // CHECK: mqtref.x() %[[q_0]]
        // CHECK: mqtref.y() %[[q_0]]
        // CHECK: mqtref.z() %[[q_0]]
        // CHECK: mqtref.s() %[[q_0]]
        // CHECK: mqtref.sdg() %[[q_0]]
        // CHECK: mqtref.t() %[[q_0]]
        // CHECK: mqtref.tdg() %[[q_0]]
        // CHECK: mqtref.v() %[[q_0]]
        // CHECK: mqtref.vdg() %[[q_0]]
        // CHECK: mqtref.sx() %[[q_0]]
        // CHECK: mqtref.sxdg() %[[q_0]]

        %ptr0 = llvm.mlir.zero : !llvm.ptr
        llvm.call @__quantum__rt__initialize(%ptr0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        %q0 = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
        llvm.call @__quantum__qis__h__body(%q0) : (!llvm.ptr) -> ()
        llvm.call @__quantum__qis__i__body(%q0) : (!llvm.ptr) -> ()
        llvm.call @__quantum__qis__x__body(%q0) : (!llvm.ptr) -> ()
        llvm.call @__quantum__qis__y__body(%q0) : (!llvm.ptr) -> ()
        llvm.call @__quantum__qis__z__body(%q0) : (!llvm.ptr) -> ()
        llvm.call @__quantum__qis__s__body(%q0) : (!llvm.ptr) -> ()
        llvm.call @__quantum__qis__sdg__body(%q0) : (!llvm.ptr) -> ()
        llvm.call @__quantum__qis__t__body(%q0) : (!llvm.ptr) -> ()
        llvm.call @__quantum__qis__tdg__body(%q0) : (!llvm.ptr) -> ()
        llvm.call @__quantum__qis__v__body(%q0) : (!llvm.ptr) -> ()
        llvm.call @__quantum__qis__vdg__body(%q0) : (!llvm.ptr) -> ()
        llvm.call @__quantum__qis__sx__body(%q0) : (!llvm.ptr) -> ()
        llvm.call @__quantum__qis__sxdg__body(%q0) : (!llvm.ptr) -> ()
        llvm.br ^bb2
      ^bb2:
        llvm.call @__quantum__rt__qubit_release(%q0) : (!llvm.ptr) -> ()
        llvm.br ^bb3
      ^bb3:
        llvm.return
    }
    llvm.func @__quantum__qis__h__body(!llvm.ptr)
    llvm.func @__quantum__qis__i__body(!llvm.ptr)
    llvm.func @__quantum__qis__x__body(!llvm.ptr)
    llvm.func @__quantum__qis__y__body(!llvm.ptr)
    llvm.func @__quantum__qis__z__body(!llvm.ptr)
    llvm.func @__quantum__qis__s__body(!llvm.ptr)
    llvm.func @__quantum__qis__sdg__body(!llvm.ptr)
    llvm.func @__quantum__qis__t__body(!llvm.ptr)
    llvm.func @__quantum__qis__tdg__body(!llvm.ptr)
    llvm.func @__quantum__qis__v__body(!llvm.ptr)
    llvm.func @__quantum__qis__vdg__body(!llvm.ptr)
    llvm.func @__quantum__qis__sx__body(!llvm.ptr)
    llvm.func @__quantum__qis__sxdg__body(!llvm.ptr)
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
    llvm.func @__quantum__rt__qubit_allocate() -> !llvm.ptr
    llvm.func @__quantum__rt__qubit_release(!llvm.ptr) -> ()
}

// -----
// This test checks if the single qubit gates using static qubits are correctly
module {
    // CHECK-LABEL: llvm.func @testConvertSingleQubitOpStatic()
    llvm.func @testConvertSingleQubitOpStatic() attributes {passthrough = ["entry_point", ["output_labeling_schema", "schema_id"], ["qir_profiles", "base_profile"], ["required_num_qubits", "2"], ["required_num_results", "0"], ["qir_major_version", "1"], ["qir_minor_version", "0"], ["dynamic_qubit_management", "true"], ["dynamic_result_management", "false"]]}  {
        // CHECK: %[[q_0:.*]] = mqtref.qubit 0
        // CHECK: mqtref.h() %[[q_0]]
        // CHECK: mqtref.i() %[[q_0]]
        // CHECK: mqtref.x() %[[q_0]]
        // CHECK: mqtref.y() %[[q_0]]
        // CHECK: mqtref.z() %[[q_0]]
        // CHECK: mqtref.s() %[[q_0]]
        // CHECK: mqtref.sdg() %[[q_0]]
        // CHECK: mqtref.t() %[[q_0]]
        // CHECK: mqtref.tdg() %[[q_0]]
        // CHECK: mqtref.v() %[[q_0]]
        // CHECK: mqtref.vdg() %[[q_0]]
        // CHECK: mqtref.sx() %[[q_0]]
        // CHECK: mqtref.sxdg() %[[q_0]]

        %0 = llvm.mlir.zero : !llvm.ptr
        llvm.call @__quantum__rt__initialize(%0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        llvm.call @__quantum__qis__h__body(%0) : (!llvm.ptr) -> ()
        llvm.call @__quantum__qis__i__body(%0) : (!llvm.ptr) -> ()
        llvm.call @__quantum__qis__x__body(%0) : (!llvm.ptr) -> ()
        llvm.call @__quantum__qis__y__body(%0) : (!llvm.ptr) -> ()
        llvm.call @__quantum__qis__z__body(%0) : (!llvm.ptr) -> ()
        llvm.call @__quantum__qis__s__body(%0) : (!llvm.ptr) -> ()
        llvm.call @__quantum__qis__sdg__body(%0) : (!llvm.ptr) -> ()
        llvm.call @__quantum__qis__t__body(%0) : (!llvm.ptr) -> ()
        llvm.call @__quantum__qis__tdg__body(%0) : (!llvm.ptr) -> ()
        llvm.call @__quantum__qis__v__body(%0) : (!llvm.ptr) -> ()
        llvm.call @__quantum__qis__vdg__body(%0) : (!llvm.ptr) -> ()
        llvm.call @__quantum__qis__sx__body(%0) : (!llvm.ptr) -> ()
        llvm.call @__quantum__qis__sxdg__body(%0) : (!llvm.ptr) -> ()
        llvm.br ^bb2
      ^bb2:
        llvm.br ^bb3
      ^bb3:
        llvm.return
    }
    llvm.func @__quantum__qis__h__body(!llvm.ptr)
    llvm.func @__quantum__qis__i__body(!llvm.ptr)
    llvm.func @__quantum__qis__x__body(!llvm.ptr)
    llvm.func @__quantum__qis__y__body(!llvm.ptr)
    llvm.func @__quantum__qis__z__body(!llvm.ptr)
    llvm.func @__quantum__qis__s__body(!llvm.ptr)
    llvm.func @__quantum__qis__sdg__body(!llvm.ptr)
    llvm.func @__quantum__qis__t__body(!llvm.ptr)
    llvm.func @__quantum__qis__tdg__body(!llvm.ptr)
    llvm.func @__quantum__qis__v__body(!llvm.ptr)
    llvm.func @__quantum__qis__vdg__body(!llvm.ptr)
    llvm.func @__quantum__qis__sx__body(!llvm.ptr)
    llvm.func @__quantum__qis__sxdg__body(!llvm.ptr)
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
}

// -----
// This test checks if the two target gates are correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertTwoTargetOp()
    llvm.func @testConvertTwoTargetOp() attributes {passthrough = ["entry_point", ["output_labeling_schema", "schema_id"], ["qir_profiles", "base_profile"], ["required_num_qubits", "2"], ["required_num_results", "0"], ["qir_major_version", "1"], ["qir_minor_version", "0"], ["dynamic_qubit_management", "true"], ["dynamic_result_management", "false"]]}  {
        // CHECK: %[[q_0:.*]] = mqtref.allocQubit
        // CHECK: %[[q_1:.*]] = mqtref.allocQubit
        // CHECK: mqtref.swap() %[[q_0]], %[[q_1]]
        // CHECK: mqtref.iswap() %[[q_0]], %[[q_1]]
        // CHECK: mqtref.iswapdg() %[[q_0]], %[[q_1]]
        // CHECK: mqtref.peres() %[[q_0]], %[[q_1]]
        // CHECK: mqtref.peresdg() %[[q_0]], %[[q_1]]
        // CHECK: mqtref.dcx() %[[q_0]], %[[q_1]]
        // CHECK: mqtref.ecr() %[[q_0]], %[[q_1]]

        %ptr0 = llvm.mlir.zero : !llvm.ptr
        llvm.call @__quantum__rt__initialize(%ptr0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        %q0 = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
        %q1 = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
        llvm.call @__quantum__qis__swap__body(%q0, %q1) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.call @__quantum__qis__iswap__body(%q0, %q1) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.call @__quantum__qis__iswapdg__body(%q0, %q1) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.call @__quantum__qis__peres__body(%q0, %q1) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.call @__quantum__qis__peresdg__body(%q0, %q1) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.call @__quantum__qis__dcx__body(%q0, %q1) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.call @__quantum__qis__ecr__body(%q0, %q1) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.br ^bb2
      ^bb2:
        llvm.call @__quantum__rt__qubit_release(%q0) : (!llvm.ptr) -> ()
        llvm.call @__quantum__rt__qubit_release(%q1) : (!llvm.ptr) -> ()
        llvm.br ^bb3
      ^bb3:
        llvm.return
    }
    llvm.func @__quantum__qis__swap__body(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__qis__iswap__body(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__qis__iswapdg__body(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__qis__peres__body(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__qis__peresdg__body(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__qis__dcx__body(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__qis__ecr__body(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
    llvm.func @__quantum__rt__qubit_allocate() -> !llvm.ptr
    llvm.func @__quantum__rt__qubit_release(!llvm.ptr) -> ()
}

// -----
// This test checks if the two target gates using static qubits are correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertTwoTargetOpStatic()
    llvm.func @testConvertTwoTargetOpStatic() attributes {passthrough = ["entry_point", ["output_labeling_schema", "schema_id"], ["qir_profiles", "base_profile"], ["required_num_qubits", "2"], ["required_num_results", "0"], ["qir_major_version", "1"], ["qir_minor_version", "0"], ["dynamic_qubit_management", "true"], ["dynamic_result_management", "false"]]}  {
        // CHECK: %[[q_0:.*]] = mqtref.qubit 0
        // CHECK: %[[q_1:.*]] = mqtref.qubit 1
        // CHECK: mqtref.swap() %[[q_0]], %[[q_1]]
        // CHECK: mqtref.iswap() %[[q_0]], %[[q_1]]
        // CHECK: mqtref.iswapdg() %[[q_0]], %[[q_1]]
        // CHECK: mqtref.peres() %[[q_0]], %[[q_1]]
        // CHECK: mqtref.peresdg() %[[q_0]], %[[q_1]]
        // CHECK: mqtref.dcx() %[[q_0]], %[[q_1]]
        // CHECK: mqtref.ecr() %[[q_0]], %[[q_1]]

        %0 = llvm.mlir.zero : !llvm.ptr
        %c0 = llvm.mlir.constant(1 : i64) : i64
        %1 = llvm.inttoptr %c0 : i64 to !llvm.ptr
        llvm.call @__quantum__rt__initialize(%0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        llvm.call @__quantum__qis__swap__body(%0, %1) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.call @__quantum__qis__iswap__body(%0, %1) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.call @__quantum__qis__iswapdg__body(%0, %1) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.call @__quantum__qis__peres__body(%0, %1) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.call @__quantum__qis__peresdg__body(%0, %1) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.call @__quantum__qis__dcx__body(%0, %1) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.call @__quantum__qis__ecr__body(%0, %1) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.br ^bb2
      ^bb2:
        llvm.br ^bb3
      ^bb3:
        llvm.return
    }
    llvm.func @__quantum__qis__swap__body(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__qis__iswap__body(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__qis__iswapdg__body(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__qis__peres__body(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__qis__peresdg__body(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__qis__dcx__body(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__qis__ecr__body(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
}

// -----
// This test checks if the single qubit rotation gates are correctly converted
module {
    // CHECK-LABEL: llvm.func @testSingleQubitRotationOp()
    llvm.func @testSingleQubitRotationOp() attributes {passthrough = ["entry_point", ["output_labeling_schema", "schema_id"], ["qir_profiles", "base_profile"], ["required_num_qubits", "2"], ["required_num_results", "0"], ["qir_major_version", "1"], ["qir_minor_version", "0"], ["dynamic_qubit_management", "true"], ["dynamic_result_management", "false"]]}  {
        // CHECK: %[[c_0:.*]] = llvm.mlir.constant(1.000000e-01 : f64) : f64
        // CHECK: %[[c_1:.*]] = llvm.mlir.constant(2.000000e-01 : f64) : f64
        // CHECK: %[[c_2:.*]] = llvm.mlir.constant(3.000000e-01 : f64) : f64
        // CHECK: %[[q_0:.*]] = mqtref.allocQubit
        // CHECK: mqtref.u2(%[[c_0]], %[[c_1]]) %[[q_0]]
        // CHECK: mqtref.p(%[[c_0]]) %[[q_0]]
        // CHECK: mqtref.u(%[[c_0]], %[[c_1]], %[[c_2]]) %[[q_0]]
        // CHECK: mqtref.p(%[[c_0]]) %[[q_0]]
        // CHECK: mqtref.rx(%[[c_0]]) %[[q_0]]
        // CHECK: mqtref.ry(%[[c_0]]) %[[q_0]]
        // CHECK: mqtref.rz(%[[c_0]]) %[[q_0]]

        %0 = llvm.mlir.zero : !llvm.ptr
        %c2 = llvm.mlir.constant(1.000000e-01 : f64) : f64
        %c3 = llvm.mlir.constant(2.000000e-01 : f64) : f64
        %c4 = llvm.mlir.constant(3.000000e-01 : f64) : f64
        llvm.call @__quantum__rt__initialize(%0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        %q0 = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
        llvm.call @__quantum__qis__u2__body(%q0, %c2, %c3) : (!llvm.ptr, f64, f64) -> ()
        llvm.call @__quantum__qis__u1__body(%q0, %c2) : (!llvm.ptr, f64) -> ()
        llvm.call @__quantum__qis__u3__body(%q0, %c2, %c3, %c4) : (!llvm.ptr, f64, f64, f64) -> ()
        llvm.call @__quantum__qis__p__body(%q0, %c2) : (!llvm.ptr, f64) -> ()
        llvm.call @__quantum__qis__rx__body(%q0, %c2) : (!llvm.ptr, f64) -> ()
        llvm.call @__quantum__qis__ry__body(%q0, %c2) : (!llvm.ptr, f64) -> ()
        llvm.call @__quantum__qis__rz__body(%q0, %c2) : (!llvm.ptr, f64) -> ()
        llvm.br ^bb2
      ^bb2:
        llvm.call @__quantum__rt__qubit_release(%q0) : (!llvm.ptr) -> ()
        llvm.br ^bb3
      ^bb3:
        llvm.return
    }
    llvm.func @__quantum__qis__u1__body(!llvm.ptr, f64)
    llvm.func @__quantum__qis__u2__body(!llvm.ptr, f64, f64)
    llvm.func @__quantum__qis__u3__body(!llvm.ptr, f64, f64, f64)
    llvm.func @__quantum__qis__p__body(!llvm.ptr, f64)
    llvm.func @__quantum__qis__rx__body(!llvm.ptr, f64)
    llvm.func @__quantum__qis__ry__body(!llvm.ptr, f64)
    llvm.func @__quantum__qis__rz__body(!llvm.ptr, f64)
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
    llvm.func @__quantum__rt__qubit_allocate() -> !llvm.ptr
    llvm.func @__quantum__rt__qubit_release(!llvm.ptr) -> ()
}

// -----
// This test checks if the single qubit rotation gates using static qubitss are correctly converted
module {
    // CHECK-LABEL: llvm.func @testSingleQubitRotationOpStatic()
    llvm.func @testSingleQubitRotationOpStatic() attributes {passthrough = ["entry_point", ["output_labeling_schema", "schema_id"], ["qir_profiles", "base_profile"], ["required_num_qubits", "2"], ["required_num_results", "0"], ["qir_major_version", "1"], ["qir_minor_version", "0"], ["dynamic_qubit_management", "true"], ["dynamic_result_management", "false"]]}  {
        // CHECK: %[[q_0:.*]] = mqtref.qubit 0
        // CHECK: %[[c_0:.*]] = llvm.mlir.constant(1.000000e-01 : f64) : f64
        // CHECK: %[[c_1:.*]] = llvm.mlir.constant(2.000000e-01 : f64) : f64
        // CHECK: %[[c_2:.*]] = llvm.mlir.constant(3.000000e-01 : f64) : f64
        // CHECK: mqtref.u2(%[[c_0]], %[[c_1]]) %[[q_0]]
        // CHECK: mqtref.p(%[[c_0]]) %[[q_0]]
        // CHECK: mqtref.u(%[[c_0]], %[[c_1]], %[[c_2]]) %[[q_0]]
        // CHECK: mqtref.p(%[[c_0]]) %[[q_0]]
        // CHECK: mqtref.rx(%[[c_0]]) %[[q_0]]
        // CHECK: mqtref.ry(%[[c_0]]) %[[q_0]]
        // CHECK: mqtref.rz(%[[c_0]]) %[[q_0]]

        %0 = llvm.mlir.zero : !llvm.ptr
        %c2 = llvm.mlir.constant(1.000000e-01 : f64) : f64
        %c3 = llvm.mlir.constant(2.000000e-01 : f64) : f64
        %c4 = llvm.mlir.constant(3.000000e-01 : f64) : f64
        llvm.call @__quantum__rt__initialize(%0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        llvm.call @__quantum__qis__u2__body(%0, %c2, %c3) : (!llvm.ptr, f64, f64) -> ()
        llvm.call @__quantum__qis__u1__body(%0, %c2) : (!llvm.ptr, f64) -> ()
        llvm.call @__quantum__qis__u3__body(%0, %c2, %c3, %c4) : (!llvm.ptr, f64, f64, f64) -> ()
        llvm.call @__quantum__qis__p__body(%0, %c2) : (!llvm.ptr, f64) -> ()
        llvm.call @__quantum__qis__rx__body(%0, %c2) : (!llvm.ptr, f64) -> ()
        llvm.call @__quantum__qis__ry__body(%0, %c2) : (!llvm.ptr, f64) -> ()
        llvm.call @__quantum__qis__rz__body(%0, %c2) : (!llvm.ptr, f64) -> ()

        llvm.br ^bb2
      ^bb2:
        llvm.br ^bb3
      ^bb3:
        llvm.return
    }
    llvm.func @__quantum__qis__u1__body(!llvm.ptr, f64)
    llvm.func @__quantum__qis__u2__body(!llvm.ptr, f64, f64)
    llvm.func @__quantum__qis__u3__body(!llvm.ptr, f64, f64, f64)
    llvm.func @__quantum__qis__p__body(!llvm.ptr, f64)
    llvm.func @__quantum__qis__rx__body(!llvm.ptr, f64)
    llvm.func @__quantum__qis__ry__body(!llvm.ptr, f64)
    llvm.func @__quantum__qis__rz__body(!llvm.ptr, f64)
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
}

// -----
// This test checks if the multiple qubit rotation gates are correctly converted
module {
    // CHECK-LABEL: llvm.func @testMultipleQubitRotationOp()
    llvm.func @testMultipleQubitRotationOp() attributes {passthrough = ["entry_point", ["output_labeling_schema", "schema_id"], ["qir_profiles", "base_profile"], ["required_num_qubits", "2"], ["required_num_results", "0"], ["qir_major_version", "1"], ["qir_minor_version", "0"], ["dynamic_qubit_management", "true"], ["dynamic_result_management", "false"]]}  {
        // CHECK: %[[c_0:.*]] = llvm.mlir.constant(2.000000e-01 : f64) : f64
        // CHECK: %[[c_1:.*]] = llvm.mlir.constant(3.000000e-01 : f64) : f64
        // CHECK: %[[q_0:.*]] = mqtref.allocQubit
        // CHECK: %[[q_1:.*]] = mqtref.allocQubit
        // CHECK: mqtref.rxx(%[[c_0]]) %[[q_0]], %[[q_1]]
        // CHECK: mqtref.ryy(%[[c_0]]) %[[q_0]], %[[q_1]]
        // CHECK: mqtref.rzz(%[[c_0]]) %[[q_0]], %[[q_1]]
        // CHECK: mqtref.rzx(%[[c_0]]) %[[q_0]], %[[q_1]]
        // CHECK: mqtref.xx_minus_yy(%[[c_0]], %[[c_1]]) %[[q_0]], %[[q_1]]
        // CHECK: mqtref.xx_plus_yy(%[[c_0]], %[[c_1]]) %[[q_0]], %[[q_1]]

        %ptr0 = llvm.mlir.zero : !llvm.ptr
        %c0 = llvm.mlir.constant(2.000000e-01 : f64) : f64
        %c1 = llvm.mlir.constant(3.000000e-01 : f64) : f64
        llvm.call @__quantum__rt__initialize(%ptr0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        %q0 = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
        %q1 = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
        llvm.call @__quantum__qis__rxx__body(%q0, %q1, %c0) : (!llvm.ptr, !llvm.ptr, f64) -> ()
        llvm.call @__quantum__qis__ryy__body(%q0, %q1, %c0) : (!llvm.ptr, !llvm.ptr, f64) -> ()
        llvm.call @__quantum__qis__rzz__body(%q0, %q1, %c0) : (!llvm.ptr, !llvm.ptr, f64) -> ()
        llvm.call @__quantum__qis__rzx__body(%q0, %q1, %c0) : (!llvm.ptr, !llvm.ptr, f64) -> ()
        llvm.call @__quantum__qis__xx_minus_yy__body(%q0, %q1, %c0, %c1) : (!llvm.ptr, !llvm.ptr, f64, f64) -> ()
        llvm.call @__quantum__qis__xx_plus_yy__body(%q0, %q1, %c0, %c1) : (!llvm.ptr, !llvm.ptr, f64, f64) -> ()
        llvm.br ^bb2
      ^bb2:
        llvm.call @__quantum__rt__qubit_release(%q0) : (!llvm.ptr) -> ()
        llvm.call @__quantum__rt__qubit_release(%q1) : (!llvm.ptr) -> ()
        llvm.br ^bb3
      ^bb3:
        llvm.return
    }
    llvm.func @__quantum__qis__rxx__body(!llvm.ptr, !llvm.ptr, f64)
    llvm.func @__quantum__qis__ryy__body(!llvm.ptr, !llvm.ptr, f64)
    llvm.func @__quantum__qis__rzz__body(!llvm.ptr, !llvm.ptr, f64)
    llvm.func @__quantum__qis__rzx__body(!llvm.ptr, !llvm.ptr, f64)
    llvm.func @__quantum__qis__xx_minus_yy__body(!llvm.ptr, !llvm.ptr, f64, f64)
    llvm.func @__quantum__qis__xx_plus_yy__body(!llvm.ptr, !llvm.ptr, f64, f64)
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
    llvm.func @__quantum__rt__qubit_allocate() -> !llvm.ptr
    llvm.func @__quantum__rt__qubit_release(!llvm.ptr) -> ()
}
// -----
// This test checks if the multiple qubit rotation gates using static qubits are correctly converted
module {
    // CHECK-LABEL: llvm.func @testMultipleQubitRotationOpStatic()
    llvm.func @testMultipleQubitRotationOpStatic() attributes {passthrough = ["entry_point", ["output_labeling_schema", "schema_id"], ["qir_profiles", "base_profile"], ["required_num_qubits", "2"], ["required_num_results", "0"], ["qir_major_version", "1"], ["qir_minor_version", "0"], ["dynamic_qubit_management", "true"], ["dynamic_result_management", "false"]]}  {
        // CHECK: %[[q_0:.*]] = mqtref.qubit 0
        // CHECK: %[[q_1:.*]] = mqtref.qubit 1
        // CHECK: %[[c_0:.*]] = llvm.mlir.constant(2.000000e-01 : f64) : f64
        // CHECK: %[[c_1:.*]] = llvm.mlir.constant(3.000000e-01 : f64) : f64
        // CHECK: mqtref.rxx(%[[c_0]]) %[[q_0]], %[[q_1]]
        // CHECK: mqtref.ryy(%[[c_0]]) %[[q_0]], %[[q_1]]
        // CHECK: mqtref.rzz(%[[c_0]]) %[[q_0]], %[[q_1]]
        // CHECK: mqtref.rzx(%[[c_0]]) %[[q_0]], %[[q_1]]
        // CHECK: mqtref.xx_minus_yy(%[[c_0]], %[[c_1]]) %[[q_0]], %[[q_1]]
        // CHECK: mqtref.xx_plus_yy(%[[c_0]], %[[c_1]]) %[[q_0]], %[[q_1]]

        %ptr0 = llvm.mlir.zero : !llvm.ptr
        %c0 = llvm.mlir.constant(1 : i64) : i64
        %ptr1 = llvm.inttoptr %c0 : i64 to !llvm.ptr
        %c1 = llvm.mlir.constant(2.000000e-01 : f64) : f64
        %c2 = llvm.mlir.constant(3.000000e-01 : f64) : f64
        llvm.call @__quantum__rt__initialize(%ptr0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        llvm.call @__quantum__qis__rxx__body(%ptr0, %ptr1, %c1) : (!llvm.ptr, !llvm.ptr, f64) -> ()
        llvm.call @__quantum__qis__ryy__body(%ptr0, %ptr1, %c1) : (!llvm.ptr, !llvm.ptr, f64) -> ()
        llvm.call @__quantum__qis__rzz__body(%ptr0, %ptr1, %c1) : (!llvm.ptr, !llvm.ptr, f64) -> ()
        llvm.call @__quantum__qis__rzx__body(%ptr0, %ptr1, %c1) : (!llvm.ptr, !llvm.ptr, f64) -> ()
        llvm.call @__quantum__qis__xx_minus_yy__body(%ptr0, %ptr1, %c1, %c2) : (!llvm.ptr, !llvm.ptr, f64, f64) -> ()
        llvm.call @__quantum__qis__xx_plus_yy__body(%ptr0, %ptr1, %c1, %c2) : (!llvm.ptr, !llvm.ptr, f64, f64) -> ()
        llvm.br ^bb2
      ^bb2:
        llvm.br ^bb3
      ^bb3:
        llvm.return
    }
    llvm.func @__quantum__qis__rxx__body(!llvm.ptr, !llvm.ptr, f64)
    llvm.func @__quantum__qis__ryy__body(!llvm.ptr, !llvm.ptr, f64)
    llvm.func @__quantum__qis__rzz__body(!llvm.ptr, !llvm.ptr, f64)
    llvm.func @__quantum__qis__rzx__body(!llvm.ptr, !llvm.ptr, f64)
    llvm.func @__quantum__qis__xx_minus_yy__body(!llvm.ptr, !llvm.ptr, f64, f64)
    llvm.func @__quantum__qis__xx_plus_yy__body(!llvm.ptr, !llvm.ptr, f64, f64)
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
}

// -----
// This test checks if controlled gates are correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertControlledOp()
    llvm.func @testConvertControlledOp() attributes {passthrough = ["entry_point", ["output_labeling_schema", "schema_id"], ["qir_profiles", "base_profile"], ["required_num_qubits", "2"], ["required_num_results", "0"], ["qir_major_version", "1"], ["qir_minor_version", "0"], ["dynamic_qubit_management", "true"], ["dynamic_result_management", "false"]]}  {
        // CHECK: %[[c_0:.*]] = llvm.mlir.constant(3.000000e-01 : f64) : f64
        // CHECK: %[[q_0:.*]] = mqtref.allocQubit
        // CHECK: %[[q_1:.*]] = mqtref.allocQubit
        // CHECK: %[[q_2:.*]] = mqtref.allocQubit
        // CHECK: mqtref.x() %[[q_0:.*]] ctrl %[[q_1:.*]]
        // CHECK: mqtref.x() %[[q_0]] ctrl %[[q_1]], %[[q_2]]
        // CHECK: mqtref.z() %[[q_0]] ctrl %[[q_1]]
        // CHECK: mqtref.rx(%[[c_0]]) %[[q_0]] ctrl %[[q_1]]

        %ptr0 = llvm.mlir.zero : !llvm.ptr
        %c3 = llvm.mlir.constant(3.000000e-01 : f64) : f64
        llvm.call @__quantum__rt__initialize(%ptr0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        %q0 = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
        %q1 = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
        %q2 = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
        llvm.call @__quantum__qis__cnot__body(%q1, %q0) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.call @__quantum__qis__ccx__body(%q1, %q2, %q0) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
        llvm.call @__quantum__qis__cz__body(%q1, %q0) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.call @__quantum__qis__crx__body(%q1, %q0, %c3) : (!llvm.ptr, !llvm.ptr, f64) -> ()
        llvm.br ^bb2
      ^bb2:
        llvm.call @__quantum__rt__qubit_release(%q0) : (!llvm.ptr) -> ()
        llvm.call @__quantum__rt__qubit_release(%q1) : (!llvm.ptr) -> ()
        llvm.call @__quantum__rt__qubit_release(%q2) : (!llvm.ptr) -> ()
        llvm.br ^bb3
      ^bb3:
        llvm.return
    }
    llvm.func @__quantum__qis__cnot__body(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__qis__ccx__body(!llvm.ptr, !llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__qis__cz__body(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__qis__crx__body(!llvm.ptr, !llvm.ptr, f64)
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
    llvm.func @__quantum__rt__qubit_allocate() -> !llvm.ptr
    llvm.func @__quantum__rt__qubit_release(!llvm.ptr) -> ()
}

// -----
// This test checks if controlled gates using static qubits are correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertControlledOpStatic()
    llvm.func @testConvertControlledOpStatic() attributes {passthrough = ["entry_point", ["output_labeling_schema", "schema_id"], ["qir_profiles", "base_profile"], ["required_num_qubits", "2"], ["required_num_results", "0"], ["qir_major_version", "1"], ["qir_minor_version", "0"], ["dynamic_qubit_management", "true"], ["dynamic_result_management", "false"]]}  {
        // CHECK: %[[q_0:.*]] = mqtref.qubit 0
        // CHECK: %[[q_1:.*]] = mqtref.qubit 1
        // CHECK: %[[q_2:.*]] = mqtref.qubit 2
        // CHECK: mqtref.x() %[[q_0:.*]] ctrl %[[q_1:.*]]
        // CHECK: mqtref.x() %[[q_0]] ctrl %[[q_1]], %[[q_2]]
        // CHECK: mqtref.z() %[[q_0]] ctrl %[[q_1]]
        // CHECK: mqtref.rx(%[[ANY:.*]]) %[[q_0]] ctrl %[[q_1]]

        %ptr0 = llvm.mlir.zero : !llvm.ptr
        %c0 = llvm.mlir.constant(1 : i64) : i64
        %ptr1 = llvm.inttoptr %c0 : i64 to !llvm.ptr
        %c1 = llvm.mlir.constant(2 : i64) : i64
        %ptr2 = llvm.inttoptr %c1 : i64 to !llvm.ptr
        %c3 = llvm.mlir.constant(3.000000e-01 : f64) : f64
        llvm.call @__quantum__rt__initialize(%ptr0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        llvm.call @__quantum__qis__cnot__body(%ptr1, %ptr0) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.call @__quantum__qis__ccx__body(%ptr1, %ptr2, %ptr0) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
        llvm.call @__quantum__qis__cz__body(%ptr1, %ptr0) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.call @__quantum__qis__crx__body(%ptr1, %ptr0, %c3) : (!llvm.ptr, !llvm.ptr, f64) -> ()
        llvm.br ^bb2
      ^bb2:
        llvm.br ^bb3
      ^bb3:
        llvm.return
    }
    llvm.func @__quantum__qis__cnot__body(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__qis__ccx__body(!llvm.ptr, !llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__qis__cz__body(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__qis__crx__body(!llvm.ptr, !llvm.ptr, f64)
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
}

// -----
// This test checks if the gphase operation is correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertGPhaseOp()
    llvm.func @testConvertGPhaseOp() attributes {passthrough = ["entry_point", ["output_labeling_schema", "schema_id"], ["qir_profiles", "base_profile"], ["required_num_qubits", "2"], ["required_num_results", "0"], ["qir_major_version", "1"], ["qir_minor_version", "0"], ["dynamic_qubit_management", "true"], ["dynamic_result_management", "false"]]}  {
        // CHECK: %[[c_0:.*]] = llvm.mlir.constant(3.000000e-01 : f64) : f64
        // CHECK: mqtref.gphase(%[[c_0]])

        %0 = llvm.mlir.zero : !llvm.ptr
        %c0 = llvm.mlir.constant(3.000000e-01 : f64) : f64
        llvm.call @__quantum__rt__initialize(%0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        llvm.call @__quantum__qis__gphase__body(%c0) : (f64) -> ()
        llvm.br ^bb2
      ^bb2:
        llvm.br ^bb3
      ^bb3:
        llvm.return
    }
    llvm.func @__quantum__qis__gphase__body(f64)
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
}

// -----
// This test checks if the controlled gphase operation is correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertGPhaseOpControlled()
    llvm.func @testConvertGPhaseOpControlled() attributes {passthrough = ["entry_point", ["output_labeling_schema", "schema_id"], ["qir_profiles", "base_profile"], ["required_num_qubits", "2"], ["required_num_results", "0"], ["qir_major_version", "1"], ["qir_minor_version", "0"], ["dynamic_qubit_management", "true"], ["dynamic_result_management", "false"]]}  {
        // CHECK: %[[c_0:.*]] = llvm.mlir.constant(3.000000e-01 : f64) : f64
        // CHECK: %[[q_0:.*]] = mqtref.allocQubit
        // CHECK: mqtref.gphase(%[[c_0]]) ctrl %[[q_0]]

        %ptr0 = llvm.mlir.zero : !llvm.ptr
        %c0 = llvm.mlir.constant(3.000000e-01 : f64) : f64
        llvm.call @__quantum__rt__initialize(%ptr0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        %q0 = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
        llvm.call @__quantum__qis__cgphase__body(%q0, %c0) : (!llvm.ptr, f64) -> ()
        llvm.br ^bb2
      ^bb2:
         llvm.call @__quantum__rt__qubit_release(%q0) : (!llvm.ptr) -> ()
        llvm.br ^bb3
      ^bb3:
        llvm.return
    }
    llvm.func @__quantum__qis__cgphase__body(!llvm.ptr, f64)
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
    llvm.func @__quantum__rt__qubit_allocate() -> !llvm.ptr
    llvm.func @__quantum__rt__qubit_release(!llvm.ptr) -> ()
}

// -----
// This test checks if the controlled gphase operation using static qubits is correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertGPhaseOpControlledStatic()
    llvm.func @testConvertGPhaseOpControlledStatic() attributes {passthrough = ["entry_point", ["output_labeling_schema", "schema_id"], ["qir_profiles", "base_profile"], ["required_num_qubits", "2"], ["required_num_results", "0"], ["qir_major_version", "1"], ["qir_minor_version", "0"], ["dynamic_qubit_management", "true"], ["dynamic_result_management", "false"]]}  {
        // CHECK: %[[q_0:.*]] = mqtref.qubit 0
        // CHECK: %[[c_0:.*]] = llvm.mlir.constant(3.000000e-01 : f64) : f64
        // CHECK: mqtref.gphase(%[[c_0]]) ctrl %[[q_0]]

        %ptr0 = llvm.mlir.zero : !llvm.ptr
        %c0 = llvm.mlir.constant(3.000000e-01 : f64) : f64
        llvm.call @__quantum__rt__initialize(%ptr0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        llvm.call @__quantum__qis__cgphase__body(%ptr0, %c0) : (!llvm.ptr, f64) -> ()
        llvm.br ^bb2
      ^bb2:
        llvm.br ^bb3
      ^bb3:
        llvm.return
    }
    llvm.func @__quantum__qis__cgphase__body(!llvm.ptr, f64)
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
}

// -----
// This test checks if the barrierOp is converted correctly
module {
    // CHECK-LABEL: llvm.func @testConvertBarrierOp()
    llvm.func @testConvertBarrierOp() attributes {passthrough = ["entry_point", ["output_labeling_schema", "schema_id"], ["qir_profiles", "base_profile"], ["required_num_qubits", "2"], ["required_num_results", "0"], ["qir_major_version", "1"], ["qir_minor_version", "0"], ["dynamic_qubit_management", "true"], ["dynamic_result_management", "false"]]}  {
        // CHECK: %[[q_0:.*]] = mqtref.allocQubit
        // CHECK: mqtref.barrier() %[[q_0]]

        %ptr0 = llvm.mlir.zero : !llvm.ptr
        llvm.call @__quantum__rt__initialize(%ptr0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        %q0 = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
        llvm.call @__quantum__qis__barrier__body(%q0) : (!llvm.ptr) -> ()
        llvm.br ^bb2
      ^bb2:
        llvm.call @__quantum__rt__qubit_release(%q0) : (!llvm.ptr) -> ()
        llvm.br ^bb3
      ^bb3:
        llvm.return
    }
    llvm.func @__quantum__qis__barrier__body(!llvm.ptr)
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
    llvm.func @__quantum__rt__qubit_allocate() -> !llvm.ptr
    llvm.func @__quantum__rt__qubit_release(!llvm.ptr) -> ()
}

// -----
// This test checks if the barrierOp using static qubits is converted correctly
module {
    // CHECK-LABEL: llvm.func @testConvertBarrierOpStatic()
    llvm.func @testConvertBarrierOpStatic() attributes {passthrough = ["entry_point", ["output_labeling_schema", "schema_id"], ["qir_profiles", "base_profile"], ["required_num_qubits", "2"], ["required_num_results", "0"], ["qir_major_version", "1"], ["qir_minor_version", "0"], ["dynamic_qubit_management", "true"], ["dynamic_result_management", "false"]]}  {
        // CHECK: %[[q_0:.*]] = mqtref.qubit 0
        // CHECK: mqtref.barrier() %[[q_0]]

        %ptr0 = llvm.mlir.zero : !llvm.ptr
        llvm.call @__quantum__rt__initialize(%ptr0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        llvm.call @__quantum__qis__barrier__body(%ptr0) : (!llvm.ptr) -> ()
        llvm.br ^bb2
      ^bb2:
        llvm.br ^bb3
      ^bb3:
        llvm.return
    }
    llvm.func @__quantum__qis__barrier__body(!llvm.ptr)
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
}

// -----
// This test checks if a Bell state is converted correctly.
module {
    llvm.mlir.global internal constant @mlir.llvm.nameless_global_0("r0\00") {addr_space = 0 : i32, dso_local}
    llvm.mlir.global internal constant @mlir.llvm.nameless_global_1("r1\00") {addr_space = 0 : i32, dso_local}

    // CHECK-LABEL: llvm.func @bellState()
    llvm.func @bellState() attributes {passthrough = ["entry_point", ["output_labeling_schema", "schema_id"], ["qir_profiles", "base_profile"], ["required_num_qubits", "2"], ["required_num_results", "2"], ["qir_major_version", "1"], ["qir_minor_version", "0"], ["dynamic_qubit_management", "true"], ["dynamic_result_management", "false"]]}  {
        // CHECK: %[[mem]] = memref.alloca() : memref<2xi1>
        // CHECK: %[[q_0:.*]] = mqtref.allocQubit
        // CHECK: %[[q_1:.*]] = mqtref.allocQubit
        // CHECK: mqtref.h() %[[q_0]]
        // CHECK: mqtref.x() %[[q_1]] ctrl %[[q_0]]
        // CHECK: %[[m_0:.*]] = mqtref.measure %[[q_0]]
        // CHECK: %[[m_1:.*]] = mqtref.measure %[[q_1]]
        // CHECK: mqtref.deallocQubit %[[q_0]]
        // CHECK: mqtref.deallocQubit %[[q_1]]
        // CHECK: %[[c_0:.*]] = arith.constant 0 : index
        // CHECK: memref.store %[[m_0]], %[[mem]][%[[c_0]]] : memref<2xi1>
        // CHECK: %[[c_1:.*]] = arith.constant 1 : index
        // CHECK: memref.store %[[m_1]], %[[mem]][%[[c_1]]] : memref<2xi1>


        %ptr0 = llvm.mlir.zero : !llvm.ptr
        %a0 = llvm.mlir.addressof @mlir.llvm.nameless_global_0 : !llvm.ptr
        %a1 = llvm.mlir.addressof @mlir.llvm.nameless_global_1 : !llvm.ptr
        %c1 = llvm.mlir.constant(1 : i64) : i64
        %ptr1 = llvm.inttoptr %c1 : i64 to !llvm.ptr
        llvm.call @__quantum__rt__initialize(%ptr0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        %q0 = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
        %q1 = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
        llvm.call @__quantum__qis__h__body(%q0) : (!llvm.ptr) -> ()
        llvm.call @__quantum__qis__cnot__body(%q0, %q1) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.br ^bb2
      ^bb2:
        llvm.call @__quantum__qis__mz__body(%q0, %ptr0) : (!llvm.ptr,  !llvm.ptr) -> ()
        llvm.call @__quantum__qis__mz__body(%q1, %ptr1) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.call @__quantum__rt__qubit_release(%q0) : (!llvm.ptr) -> ()
        llvm.call @__quantum__rt__qubit_release(%q1) : (!llvm.ptr) -> ()
        llvm.br ^bb3
      ^bb3:
        llvm.call @__quantum__rt__result_record_output(%ptr0, %a0) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.call @__quantum__rt__result_record_output(%ptr1, %a1) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.return
    }
    llvm.func @__quantum__qis__mz__body(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__rt__result_record_output(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__qis__h__body(!llvm.ptr)
    llvm.func @__quantum__qis__cnot__body(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
    llvm.func @__quantum__rt__qubit_release(!llvm.ptr)
    llvm.func @__quantum__rt__qubit_allocate() -> !llvm.ptr

}

// -----
// This test checks if a Bell state using static qubits is converted correctly.
module {
    llvm.mlir.global internal constant @mlir.llvm.nameless_global_0("r0\00") {addr_space = 0 : i32, dso_local}
    llvm.mlir.global internal constant @mlir.llvm.nameless_global_1("r1\00") {addr_space = 0 : i32, dso_local}

    // CHECK-LABEL: llvm.func @bellStateStaticAddressing()
    llvm.func @bellStateStaticAddressing() attributes {passthrough = ["entry_point", ["output_labeling_schema", "schema_id"], ["qir_profiles", "base_profile"], ["required_num_qubits", "2"], ["required_num_results", "2"], ["qir_major_version", "1"], ["qir_minor_version", "0"], ["dynamic_qubit_management", "true"], ["dynamic_result_management", "false"]]}  {
        // CHECK: %[[mem]] = memref.alloca() : memref<2xi1>
        // CHECK: %[[q_0:.*]] = mqtref.qubit 0
        // CHECK: %[[q_1:.*]] = mqtref.qubit 1
        // CHECK: mqtref.h() %[[q_0]]
        // CHECK: mqtref.x() %[[q_1]] ctrl %[[q_0]]
        // CHECK: %[[m_0:.*]] = mqtref.measure %[[q_0]]
        // CHECK: %[[m_1:.*]] = mqtref.measure %[[q_1]]
        // CHECK: %[[c_0:.*]] = arith.constant 0 : index
        // CHECK: memref.store %[[m_0]], %[[mem]][%[[c_0]]] : memref<2xi1>
        // CHECK: %[[c_1:.*]] = arith.constant 1 : index
        // CHECK: memref.store %[[m_1]], %[[mem]][%[[c_1]]] : memref<2xi1>

        %ptr0 = llvm.mlir.zero : !llvm.ptr
        %c0 = llvm.mlir.constant(1 : i64) : i64
        %ptr1 = llvm.inttoptr %c0 : i64 to !llvm.ptr
        %a0 = llvm.mlir.addressof @mlir.llvm.nameless_global_0 : !llvm.ptr
        %a1 = llvm.mlir.addressof @mlir.llvm.nameless_global_1 : !llvm.ptr
        llvm.call @__quantum__rt__initialize(%ptr0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        llvm.call @__quantum__qis__h__body(%ptr0) : (!llvm.ptr) -> ()
        llvm.call @__quantum__qis__cnot__body(%ptr0, %ptr1) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.br ^bb2
      ^bb2:
        llvm.call @__quantum__qis__mz__body(%ptr0, %ptr0) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.call @__quantum__qis__mz__body(%ptr1, %ptr1) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.br ^bb3
      ^bb3:
        llvm.call @__quantum__rt__result_record_output(%ptr0, %a0) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.call @__quantum__rt__result_record_output(%ptr1, %a1) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.return
    }
    llvm.func @__quantum__qis__mz__body(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__rt__result_record_output(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__qis__h__body(!llvm.ptr)
    llvm.func @__quantum__qis__cnot__body(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
}
