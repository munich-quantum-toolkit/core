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
    llvm.func @testConvertAllocRegister() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[size:.*]] = llvm.mlir.constant(14 : i64) : i64
        // CHECK: %[[r_0:.*]] = "mqtref.allocQubitRegister"(%[[size]]) : (i64) -> !mqtref.QubitRegister

        %0 = llvm.mlir.zero : !llvm.ptr
        %c = llvm.mlir.constant(14 : i64) : i64
        llvm.call @__quantum__rt__initialize(%0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        %r = llvm.call @__quantum__rt__qubit_allocate_array(%c) : (i64) -> !llvm.ptr
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
// This test checks if the extract from register call is correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertExtractFromRegister()
    llvm.func @testConvertExtractFromRegister() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[size:.*]] = llvm.mlir.constant(14 : i64) : i64
        // CHECK: %[[index:.*]] = llvm.mlir.constant(0 : i64) : i64
        // CHECK: %[[r_0:.*]] = "mqtref.allocQubitRegister"(%[[size]]) : (i64) -> !mqtref.QubitRegister
        // CHECK: %[[q_0:.*]] = "mqtref.extractQubit"(%[[r_0]], %[[index]]) : (!mqtref.QubitRegister, i64) -> !mqtref.Qubit

        %0 = llvm.mlir.zero : !llvm.ptr
        %c0 = llvm.mlir.constant(14 : i64) : i64
        %c1 = llvm.mlir.constant(0 : i64) : i64
        llvm.call @__quantum__rt__initialize(%0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        %r0 = llvm.call @__quantum__rt__qubit_allocate_array(%c0) : (i64) -> !llvm.ptr
        %1 = llvm.call @__quantum__rt__array_get_element_ptr_1d(%r0, %c1) : (!llvm.ptr, i64) -> !llvm.ptr
        %q0 = llvm.load %1 : !llvm.ptr -> !llvm.ptr
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
    llvm.func @testConvertAllocateQubit() attributes {passthrough = ["entry_point"]}  {
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
    // CHECK-LABEL: llvm.func @testConvertDeAllocateQubit()
    llvm.func @testConvertDeAllocateQubit() attributes {passthrough = ["entry_point"]}  {
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
// This test checks if the dealloc register call is correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertDeallocRegister()
    llvm.func @testConvertDeallocRegister() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[size:.*]] = llvm.mlir.constant(14 : i64) : i64
        // CHECK: %[[r_0:.*]] = "mqtref.allocQubitRegister"(%[[size]]) : (i64) -> !mqtref.QubitRegister
        // CHECK: "mqtref.deallocQubitRegister"(%[[r_0]])

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
// This test checks if the reset operation is correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertResetOp()
    llvm.func @testConvertResetOp() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[size:.*]] = llvm.mlir.constant(14 : i64) : i64
        // CHECK: %[[index:.*]] = llvm.mlir.constant(0 : i64) : i64
        // CHECK: %[[r_0:.*]] = "mqtref.allocQubitRegister"(%[[size]]) : (i64) -> !mqtref.QubitRegister
        // CHECK: %[[q_0:.*]] = "mqtref.extractQubit"(%[[r_0]], %[[index]]) : (!mqtref.QubitRegister, i64) -> !mqtref.Qubit
        // CHECK: "mqtref.reset"(%[[q_0]])

        %0 = llvm.mlir.zero : !llvm.ptr
        %c0 = llvm.mlir.constant(14 : i64) : i64
        %c1 = llvm.mlir.constant(0 : i64) : i64
        llvm.call @__quantum__rt__initialize(%0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        %r0 = llvm.call @__quantum__rt__qubit_allocate_array(%c0) : (i64) -> !llvm.ptr
        %1 = llvm.call @__quantum__rt__array_get_element_ptr_1d(%r0, %c1) : (!llvm.ptr, i64) -> !llvm.ptr
        %q0 = llvm.load %1 : !llvm.ptr -> !llvm.ptr
        llvm.call @__quantum__qis__reset__body(%q0) : (!llvm.ptr) -> ()
        llvm.br ^bb2
      ^bb2:
        llvm.br ^bb3
      ^bb3:
        llvm.return
    }
    llvm.func @__quantum__qis__reset__body(!llvm.ptr)
    llvm.func @__quantum__rt__array_get_element_ptr_1d(!llvm.ptr, i64) -> !llvm.ptr
    llvm.func @__quantum__rt__qubit_release_array(!llvm.ptr)
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
    llvm.func @__quantum__rt__qubit_allocate_array(i64) -> !llvm.ptr
}

// -----
// This test checks if the measure operation is correctly converted
module {
    llvm.mlir.global internal constant @mlir.llvm.nameless_global_0("r0\00") {addr_space = 0 : i32, dso_local}
    // CHECK-LABEL: llvm.func @testConvertMeasure()
    llvm.func @testConvertMeasure() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[size:.*]] = llvm.mlir.constant(1 : i64) : i64
        // CHECK: %[[index:.*]] = llvm.mlir.constant(0 : i64) : i64
        // CHECK: %[[r_0:.*]] = "mqtref.allocQubitRegister"(%[[size]]) : (i64) -> !mqtref.QubitRegister
        // CHECK: %[[q_0:.*]] = "mqtref.extractQubit"(%[[r_0]], %[[index]]) : (!mqtref.QubitRegister, i64) -> !mqtref.Qubit
        // CHECK:  [[m_0:.*]] = mqtref.measure %[[q_0]]

        %0 = llvm.mlir.zero : !llvm.ptr
        %a0 = llvm.mlir.addressof @mlir.llvm.nameless_global_0 : !llvm.ptr
        %c0 = llvm.mlir.constant(1 : i64) : i64
        %c1 = llvm.mlir.constant(0 : i64) : i64
        %c2 = llvm.mlir.constant(-1 : i32) : i32
        llvm.call @__quantum__rt__initialize(%0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        %r0 = llvm.call @__quantum__rt__qubit_allocate_array(%c0) : (i64) -> !llvm.ptr
        %1 = llvm.call @__quantum__rt__array_get_element_ptr_1d(%r0, %c1) : (!llvm.ptr, i64) -> !llvm.ptr
        %q0 = llvm.load %1 : !llvm.ptr -> !llvm.ptr
        llvm.br ^bb2
      ^bb2:
        llvm.call @__quantum__qis__mz__body(%q0, %0) : (!llvm.ptr, !llvm.ptr)  -> ()
        llvm.br ^bb3
      ^bb3:
        llvm.call @__quantum__rt__result_record_output(%0, %a0) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.return
    }
    llvm.func @__quantum__qis__mz__body(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__rt__result_record_output(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__rt__array_get_element_ptr_1d(!llvm.ptr, i64) -> !llvm.ptr
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
    llvm.func @__quantum__rt__qubit_allocate_array(i64) -> !llvm.ptr
}

// -----
// This test checks if the measure operation is correctly converted
module {
    llvm.mlir.global internal constant @mlir.llvm.nameless_global_0("r0\00") {addr_space = 0 : i32, dso_local}
    // CHECK-LABEL: llvm.func @testConvertStaticMeasure()
    llvm.func @testConvertStaticMeasure() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[q_0:.*]] = mqtref.qubit 0
        // CHECK:  [[m_0:.*]] = mqtref.measure %[[q_0]]

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
// This test checks if the single qubit gates are correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertSingleQubitOp()
    llvm.func @testConvertSingleQubitOp() attributes {passthrough = ["entry_point"]}  {
        // CHECK: mqtref.h() %[[q_0:.*]]
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
    llvm.func @__quantum__rt__array_get_element_ptr_1d(!llvm.ptr, i64) -> !llvm.ptr
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
    llvm.func @__quantum__rt__qubit_allocate_array(i64) -> !llvm.ptr
}

// -----
// This test checks if the two target gates are correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertTwoTargetOp()
    llvm.func @testConvertTwoTargetOp() attributes {passthrough = ["entry_point"]}  {
        // CHECK: mqtref.swap() %[[q_0:.*]], %[[q_1:.*]]
        // CHECK: mqtref.iswap() %[[q_0]], %[[q_1]]
        // CHECK: mqtref.iswapdg() %[[q_0]], %[[q_1]]
        // CHECK: mqtref.peres() %[[q_0]], %[[q_1]]
        // CHECK: mqtref.peresdg() %[[q_0]], %[[q_1]]
        // CHECK: mqtref.dcx() %[[q_0]], %[[q_1]]
        // CHECK: mqtref.ecr() %[[q_0]], %[[q_1]]

        %0 = llvm.mlir.zero : !llvm.ptr
        %c0 = llvm.mlir.constant(2 : i64) : i64
        %c1 = llvm.mlir.constant(0 : i64) : i64
        %c2 = llvm.mlir.constant(1 : i64) : i64
        llvm.call @__quantum__rt__initialize(%0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        %r0 = llvm.call @__quantum__rt__qubit_allocate_array(%c0) : (i64) -> !llvm.ptr
        %1 = llvm.call @__quantum__rt__array_get_element_ptr_1d(%r0, %c1) : (!llvm.ptr, i64) -> !llvm.ptr
        %q0 = llvm.load %1 : !llvm.ptr -> !llvm.ptr
        %2 = llvm.call @__quantum__rt__array_get_element_ptr_1d(%r0, %c2) : (!llvm.ptr, i64) -> !llvm.ptr
        %q1 = llvm.load %2 : !llvm.ptr -> !llvm.ptr
        llvm.call @__quantum__qis__swap__body(%q0, %q1) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.call @__quantum__qis__iswap__body(%q0, %q1) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.call @__quantum__qis__iswapdg__body(%q0, %q1) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.call @__quantum__qis__peres__body(%q0, %q1) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.call @__quantum__qis__peresdg__body(%q0, %q1) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.call @__quantum__qis__dcx__body(%q0, %q1) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.call @__quantum__qis__ecr__body(%q0, %q1) : (!llvm.ptr, !llvm.ptr) -> ()
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
    llvm.func @__quantum__rt__array_get_element_ptr_1d(!llvm.ptr, i64) -> !llvm.ptr
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
    llvm.func @__quantum__rt__qubit_allocate_array(i64) -> !llvm.ptr
}

// -----
// This test checks if the single qubit rotation gates are correctly converted
module {
    // CHECK-LABEL: llvm.func @testSingleQubitRotationOp()
    llvm.func @testSingleQubitRotationOp() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[c_0:.*]] = llvm.mlir.constant(3.000000e-01 : f64) : f64
        // CHECK: %[[c_1:.*]] = llvm.mlir.constant(1.000000e-01 : f64) : f64
        // CHECK: %[[c_2:.*]] = llvm.mlir.constant(2.000000e-01 : f64) : f64
        // CHECK: mqtref.u2(%[[c_0]], %[[c_1]]) %[[q_0:.*]]
        // CHECK: mqtref.p(%[[c_0]]) %[[q_0]]
        // CHECK: mqtref.u(%[[c_0]], %[[c_1]], %[[c_2]]) %[[q_0]]
        // CHECK: mqtref.p(%[[c_0]]) %[[q_0]]
        // CHECK: mqtref.rx(%[[c_0]]) %[[q_0]]
        // CHECK: mqtref.ry(%[[c_0]]) %[[q_0]]
        // CHECK: mqtref.rz(%[[c_0]]) %[[q_0]]

        %0 = llvm.mlir.zero : !llvm.ptr
        %c0 = llvm.mlir.constant(2 : i64) : i64
        %c1 = llvm.mlir.constant(0 : i64) : i64
        %c2 = llvm.mlir.constant(3.000000e-01 : f64) : f64
        %c3 = llvm.mlir.constant(1.000000e-01 : f64) : f64
        %c4 = llvm.mlir.constant(2.000000e-01 : f64) : f64
        llvm.call @__quantum__rt__initialize(%0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        %r0 = llvm.call @__quantum__rt__qubit_allocate_array(%c0) : (i64) -> !llvm.ptr
        %1 = llvm.call @__quantum__rt__array_get_element_ptr_1d(%r0, %c1) : (!llvm.ptr, i64) -> !llvm.ptr
        %q0 = llvm.load %1 : !llvm.ptr -> !llvm.ptr

        llvm.call @__quantum__qis__u2__body(%q0, %c2, %c3) : (!llvm.ptr, f64, f64) -> ()
        llvm.call @__quantum__qis__u1__body(%q0, %c2) : (!llvm.ptr, f64) -> ()
        llvm.call @__quantum__qis__u3__body(%q0, %c2, %c3, %c4) : (!llvm.ptr, f64, f64, f64) -> ()
        llvm.call @__quantum__qis__p__body(%q0, %c2) : (!llvm.ptr, f64) -> ()
        llvm.call @__quantum__qis__rx__body(%q0, %c2) : (!llvm.ptr, f64) -> ()
        llvm.call @__quantum__qis__ry__body(%q0, %c2) : (!llvm.ptr, f64) -> ()
        llvm.call @__quantum__qis__rz__body(%q0, %c2) : (!llvm.ptr, f64) -> ()

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
    llvm.func @__quantum__rt__array_get_element_ptr_1d(!llvm.ptr, i64) -> !llvm.ptr
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
    llvm.func @__quantum__rt__qubit_allocate_array(i64) -> !llvm.ptr
}

// -----
// This test checks if the multiple qubit rotation gates are correctly converted
module {
    // CHECK-LABEL: llvm.func @testMultipleQubitRotationOp()
    llvm.func @testMultipleQubitRotationOp() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[c_0:.*]] = llvm.mlir.constant(3.000000e-01 : f64) : f64
        // CHECK: %[[c_1:.*]] = llvm.mlir.constant(2.000000e-01 : f64) : f64
        // CHECK: mqtref.rxx(%[[c_0]]) %[[q_0:.*]], %[[q_1:.*]]
        // CHECK: mqtref.ryy(%[[c_0]]) %[[q_0]], %[[q_1]]
        // CHECK: mqtref.rzz(%[[c_0]]) %[[q_0]], %[[q_1]]
        // CHECK: mqtref.rzx(%[[c_0]]) %[[q_0]], %[[q_1]]
        // CHECK: mqtref.xxminusyy(%[[c_0]], %[[c_1]]) %[[q_0]], %[[q_1]]
        // CHECK: mqtref.xxplusyy(%[[c_0]], %[[c_1]]) %[[q_0]], %[[q_1]]

        %0 = llvm.mlir.zero : !llvm.ptr
        %c0 = llvm.mlir.constant(2 : i64) : i64
        %c1 = llvm.mlir.constant(0 : i64) : i64
        %c2 = llvm.mlir.constant(1 : i64) : i64
        %c3 = llvm.mlir.constant(3.000000e-01 : f64) : f64
        %c4 = llvm.mlir.constant(2.000000e-01 : f64) : f64
        llvm.call @__quantum__rt__initialize(%0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        %r0 = llvm.call @__quantum__rt__qubit_allocate_array(%c0) : (i64) -> !llvm.ptr
        %1 = llvm.call @__quantum__rt__array_get_element_ptr_1d(%r0, %c1) : (!llvm.ptr, i64) -> !llvm.ptr
        %q0 = llvm.load %1 : !llvm.ptr -> !llvm.ptr
        %2 = llvm.call @__quantum__rt__array_get_element_ptr_1d(%r0, %c2) : (!llvm.ptr, i64) -> !llvm.ptr
        %q1 = llvm.load %2 : !llvm.ptr -> !llvm.ptr

        llvm.call @__quantum__qis__rxx__body(%q0, %q1, %c3) : (!llvm.ptr, !llvm.ptr, f64) -> ()
        llvm.call @__quantum__qis__ryy__body(%q0, %q1, %c3) : (!llvm.ptr, !llvm.ptr, f64) -> ()
        llvm.call @__quantum__qis__rzz__body(%q0, %q1, %c3) : (!llvm.ptr, !llvm.ptr, f64) -> ()
        llvm.call @__quantum__qis__rzx__body(%q0, %q1, %c3) : (!llvm.ptr, !llvm.ptr, f64) -> ()
        llvm.call @__quantum__qis__xxminusyy__body(%q0, %q1, %c3, %c4) : (!llvm.ptr, !llvm.ptr, f64, f64) -> ()
        llvm.call @__quantum__qis__xxplusyy__body(%q0, %q1, %c3, %c4) : (!llvm.ptr, !llvm.ptr, f64, f64) -> ()

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
    llvm.func @__quantum__qis__xxminusyy__body(!llvm.ptr, !llvm.ptr, f64, f64)
    llvm.func @__quantum__qis__xxplusyy__body(!llvm.ptr, !llvm.ptr, f64, f64)
    llvm.func @__quantum__rt__array_get_element_ptr_1d(!llvm.ptr, i64) -> !llvm.ptr
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
    llvm.func @__quantum__rt__qubit_allocate_array(i64) -> !llvm.ptr
}

// -----
// This test checks if controlled gates are correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertControlledOp()
    llvm.func @testConvertControlledOp() attributes {passthrough = ["entry_point"]}  {
        // CHECK: mqtref.x() %[[q_0:.*]] ctrl %[[q_1:.*]]
        // CHECK: mqtref.x() %[[q_0]] ctrl %[[q_1]], %[[q_2:.*]]
        // CHECK: mqtref.z() %[[q_0]] ctrl %[[q_1]]
        // CHECK: mqtref.rx(%[[ANY:.*]]) %[[q_0]] ctrl %[[q_1]]

        %0 = llvm.mlir.zero : !llvm.ptr
        %c0 = llvm.mlir.constant(3 : i64) : i64
        %c1 = llvm.mlir.constant(0 : i64) : i64
        %c2 = llvm.mlir.constant(1 : i64) : i64
        %c3 = llvm.mlir.constant(3.000000e-01 : f64) : f64
        llvm.call @__quantum__rt__initialize(%0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        %r0 = llvm.call @__quantum__rt__qubit_allocate_array(%c0) : (i64) -> !llvm.ptr
        %1 = llvm.call @__quantum__rt__array_get_element_ptr_1d(%r0, %c1) : (!llvm.ptr, i64) -> !llvm.ptr
        %q0 = llvm.load %1 : !llvm.ptr -> !llvm.ptr
        %2 = llvm.call @__quantum__rt__array_get_element_ptr_1d(%r0, %c2) : (!llvm.ptr, i64) -> !llvm.ptr
        %q1 = llvm.load %2 : !llvm.ptr -> !llvm.ptr
        %3 = llvm.call @__quantum__rt__array_get_element_ptr_1d(%r0, %c2) : (!llvm.ptr, i64) -> !llvm.ptr
        %q2 = llvm.load %3 : !llvm.ptr -> !llvm.ptr

        llvm.call @__quantum__qis__cnot__body(%q1, %q0) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.call @__quantum__qis__ccx__body(%q1, %q2, %q0) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
        llvm.call @__quantum__qis__cz__body(%q1, %q0) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.call @__quantum__qis__crx__body(%q1, %q0, %c3) : (!llvm.ptr, !llvm.ptr, f64) -> ()
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
    llvm.func @__quantum__rt__array_get_element_ptr_1d(!llvm.ptr, i64) -> !llvm.ptr
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
    llvm.func @__quantum__rt__qubit_allocate_array(i64) -> !llvm.ptr
}

// -----
// This test checks if the gphase operation is correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertGPhaseOp()
    llvm.func @testConvertGPhaseOp() attributes {passthrough = ["entry_point"]}  {
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
    llvm.func @testConvertGPhaseOpControlled() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[c_0:.*]] = llvm.mlir.constant(3.000000e-01 : f64) : f64
        // CHECK: mqtref.gphase(%[[c_0]]) ctrl %[[ANY:.*]]

        %0 = llvm.mlir.zero : !llvm.ptr
        %c0 = llvm.mlir.constant(1 : i64) : i64
        %c1 = llvm.mlir.constant(0 : i64) : i64
        %c2 = llvm.mlir.constant(3.000000e-01 : f64) : f64
        llvm.call @__quantum__rt__initialize(%0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        %r0 = llvm.call @__quantum__rt__qubit_allocate_array(%c0) : (i64) -> !llvm.ptr
        %1 = llvm.call @__quantum__rt__array_get_element_ptr_1d(%r0, %c1) : (!llvm.ptr, i64) -> !llvm.ptr
        %q0 = llvm.load %1 : !llvm.ptr -> !llvm.ptr
        llvm.call @__quantum__qis__cgphase__body(%q0, %c2) : (!llvm.ptr, f64) -> ()
        llvm.br ^bb2
      ^bb2:
        llvm.br ^bb3
      ^bb3:
        llvm.return
    }
    llvm.func @__quantum__qis__cgphase__body(!llvm.ptr, f64)
    llvm.func @__quantum__rt__array_get_element_ptr_1d(!llvm.ptr, i64) -> !llvm.ptr
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
    llvm.func @__quantum__rt__qubit_allocate_array(i64) -> !llvm.ptr
}

// -----
// This test checks if the barrierOp is converted correctly
module {
    // CHECK-LABEL: llvm.func @testConvertBarrierOp()
    llvm.func @testConvertBarrierOp() attributes {passthrough = ["entry_point"]}  {
        // CHECK: mqtref.barrier() %[[ANY:.*]]

        %0 = llvm.mlir.zero : !llvm.ptr
        %c0 = llvm.mlir.constant(1 : i64) : i64
        %c1 = llvm.mlir.constant(0 : i64) : i64
        llvm.call @__quantum__rt__initialize(%0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        %r0 = llvm.call @__quantum__rt__qubit_allocate_array(%c0) : (i64) -> !llvm.ptr
        %1 = llvm.call @__quantum__rt__array_get_element_ptr_1d(%r0, %c1) : (!llvm.ptr, i64) -> !llvm.ptr
        %q0 = llvm.load %1 : !llvm.ptr -> !llvm.ptr
        llvm.call @__quantum__qis__barrier__body(%q0) : (!llvm.ptr) -> ()
        llvm.br ^bb2
      ^bb2:
        llvm.br ^bb3
      ^bb3:
        llvm.return
    }
    llvm.func @__quantum__qis__barrier__body(!llvm.ptr)
    llvm.func @__quantum__rt__array_get_element_ptr_1d(!llvm.ptr, i64) -> !llvm.ptr
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
    llvm.func @__quantum__rt__qubit_allocate_array(i64) -> !llvm.ptr
}

// -----
// This test checks if a Bell state is converted correctly.
module {
    llvm.mlir.global internal constant @mlir.llvm.nameless_global_0("r0\00") {addr_space = 0 : i32, dso_local}
    llvm.mlir.global internal constant @mlir.llvm.nameless_global_1("r1\00") {addr_space = 0 : i32, dso_local}

    // CHECK-LABEL: llvm.func @bellState()
    llvm.func @bellState() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[r_0:.*]] = "mqtref.allocQubitRegister"(%[[ANY:.*]]) : (i64) -> !mqtref.QubitRegister
        // CHECK: %[[q_0:.*]] = "mqtref.extractQubit"(%[[r_0]], %[[ANY:.*]]) : (!mqtref.QubitRegister, i64) -> !mqtref.Qubit
        // CHECK: %[[q_1:.*]] = "mqtref.extractQubit"(%[[r_0]], %[[ANY:.*]]) : (!mqtref.QubitRegister, i64) -> !mqtref.Qubit
        // CHECK: mqtref.h() %[[q_0]]
        // CHECK: mqtref.x() %[[q_1]] ctrl %[[q_0]]
        // CHECK: %[[m_0:.*]] = mqtref.measure %[[q_0]]
        // CHECK: %[[m_1:.*]] = mqtref.measure %[[q_1]]
        // CHECK: "mqtref.deallocQubitRegister"(%[[r_0]]) : (!mqtref.QubitRegister) -> ()


        %0 = llvm.mlir.zero : !llvm.ptr
        %a0 = llvm.mlir.addressof @mlir.llvm.nameless_global_0 : !llvm.ptr
        %a1 = llvm.mlir.addressof @mlir.llvm.nameless_global_1 : !llvm.ptr
        %c0 = llvm.mlir.constant(2 : i64) : i64
        %c1 = llvm.mlir.constant(0 : i64) : i64
        %c2 = llvm.mlir.constant(1 : i64) : i64
        %res1 = llvm.inttoptr %c1 : i64 to !llvm.ptr
        llvm.call @__quantum__rt__initialize(%0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        %r0 = llvm.call @__quantum__rt__qubit_allocate_array(%c0) : (i64) -> !llvm.ptr
        %1 = llvm.call @__quantum__rt__array_get_element_ptr_1d(%r0, %c1) : (!llvm.ptr, i64) -> !llvm.ptr
        %q0 = llvm.load %1 : !llvm.ptr -> !llvm.ptr
        %2 = llvm.call @__quantum__rt__array_get_element_ptr_1d(%r0, %c2) : (!llvm.ptr, i64) -> !llvm.ptr
        %q1 = llvm.load %2 : !llvm.ptr -> !llvm.ptr
        llvm.call @__quantum__qis__h__body(%q0) : (!llvm.ptr) -> ()
        llvm.call @__quantum__qis__cnot__body(%q0, %q1) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.br ^bb2
      ^bb2:
        llvm.call @__quantum__qis__mz__body(%q0, %0) : (!llvm.ptr,  !llvm.ptr) -> ()
        llvm.call @__quantum__qis__mz__body(%q1, %res1) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.call @__quantum__rt__qubit_release_array(%r0) : (!llvm.ptr) -> ()
        llvm.br ^bb3
      ^bb3:
        llvm.call @__quantum__rt__result_record_output(%0, %a0) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.call @__quantum__rt__result_record_output(%res1, %a1) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.return
    }
    llvm.func @__quantum__rt__qubit_release_array(!llvm.ptr)
    llvm.func @__quantum__rt__result_update_reference_count(!llvm.ptr, i32)
    llvm.func @__quantum__qis__mz__body(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__rt__result_record_output(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__qis__h__body(!llvm.ptr)
    llvm.func @__quantum__qis__cnot__body(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__rt__array_get_element_ptr_1d(!llvm.ptr, i64) -> !llvm.ptr
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
    llvm.func @__quantum__rt__qubit_allocate_array(i64) -> !llvm.ptr
}

// -----
// This test checks if a Bell state with static qubit addressing is converted correctly.
module {
    llvm.mlir.global internal constant @mlir.llvm.nameless_global_0("r0\00") {addr_space = 0 : i32, dso_local}
    llvm.mlir.global internal constant @mlir.llvm.nameless_global_1("r1\00") {addr_space = 0 : i32, dso_local}

    // CHECK-LABEL: llvm.func @bellStateStaticAddressing()
    llvm.func @bellStateStaticAddressing() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[q_0:.*]] = mqtref.qubit 0
        // CHECK: %[[q_1:.*]] = mqtref.qubit 1
        // CHECK: mqtref.h() %[[q_0]]
        // CHECK: mqtref.x() %[[q_1]] ctrl %[[q_0]]
        // CHECK: %[[m_0:.*]] = mqtref.measure %[[q_0]]
        // CHECK: %[[m_1:.*]] = mqtref.measure %[[q_1]]

        %q0 = llvm.mlir.zero : !llvm.ptr
        %c0 = llvm.mlir.constant(1 : i64) : i64
        %q1 = llvm.inttoptr %c0 : i64 to !llvm.ptr
        %a0 = llvm.mlir.addressof @mlir.llvm.nameless_global_0 : !llvm.ptr
        %a1 = llvm.mlir.addressof @mlir.llvm.nameless_global_1 : !llvm.ptr
        %res0 = llvm.mlir.zero : !llvm.ptr
        %c1 = llvm.mlir.constant(1 : i64) : i64
        %res1 = llvm.inttoptr %c1 : i64 to !llvm.ptr
        llvm.call @__quantum__rt__initialize(%q0) : (!llvm.ptr) -> ()
        llvm.br ^bb1
      ^bb1:
        llvm.call @__quantum__qis__h__body(%q0) : (!llvm.ptr) -> ()
        llvm.call @__quantum__qis__cnot__body(%q0, %q1) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.br ^bb2
      ^bb2:
        llvm.call @__quantum__qis__mz__body(%q0, %res0) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.call @__quantum__qis__mz__body(%q1, %res1) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.br ^bb3
      ^bb3:
        llvm.call @__quantum__rt__result_record_output(%res0, %a0) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.call @__quantum__rt__result_record_output(%res1, %a1) : (!llvm.ptr, !llvm.ptr) -> ()
        llvm.return
    }
    llvm.func @__quantum__qis__mz__body(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__rt__result_record_output(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__qis__h__body(!llvm.ptr)
    llvm.func @__quantum__qis__cnot__body(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
}
