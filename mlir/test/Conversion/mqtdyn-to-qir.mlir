// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file --mqtdyn-to-qir | FileCheck %s


// TODO
module {
    // CHECK-LABEL: llvm.func @testConvertAllocRegisterAttribute()
    llvm.func @testConvertAllocRegisterAttribute() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[size:.*]] = llvm.mlir.constant(2 : i64) : i64
        // CHECK: %[[r_0:.*]] = llvm.call @__quantum__rt__qubit_allocate_array(%[[size]]) : (i64) -> !llvm.ptr

        %0 = llvm.mlir.zero : !llvm.ptr
        llvm.br ^bb1
      ^bb1:
        %r0 = "mqtdyn.allocQubitRegister" () {"size_attr" = 2 : i64} : () -> !mqtdyn.QubitRegister
        llvm.br ^bb2
      ^bb2:
        llvm.return
    }

    llvm.func @__quantum__rt__initialize(!llvm.ptr)
}

// -----
// This test checks if the AllocOp is converted correctly using a dynamic operand.
module {
    // CHECK-LABEL: llvm.func @testConvertAllocRegisterOperand()
    llvm.func @testConvertAllocRegisterOperand() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[size:.*]] = llvm.mlir.constant(2 : i64) : i64
        // CHECK: %[[r_0:.*]] = llvm.call @__quantum__rt__qubit_allocate_array(%[[size]]) : (i64) -> !llvm.ptr

        %0 = llvm.mlir.zero : !llvm.ptr
        %c0 = llvm.mlir.constant(2 : i64) : i64
        llvm.br ^bb1
      ^bb1:
        %r0 = "mqtdyn.allocQubitRegister" (%c0) : (i64) -> !mqtdyn.QubitRegister
        llvm.br ^bb2
      ^bb2:
        llvm.return
    }

    llvm.func @__quantum__rt__initialize(!llvm.ptr)
}

// -----
// This test checks if the extract from register call is correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertExtractOpAttribute()
    llvm.func @testConvertExtractOpAttribute() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[size:.*]] = llvm.mlir.constant(2 : i64) : i64
        // CHECK: %[[r_0:.*]] = llvm.call @__quantum__rt__qubit_allocate_array(%[[size]]) : (i64) -> !llvm.ptr
        // CHECK-DAG: %[[index:.*]] = llvm.mlir.constant(0 : i64) : i64
        // CHECK: %[[ptr_0:.*]] = llvm.call @__quantum__rt__array_get_element_ptr_1d(%[[r_0]], %[[index]]) : (!llvm.ptr, i64) -> !llvm.ptr
        // CHECK: %[[q_0:.*]] = llvm.load %[[ptr_0]] : !llvm.ptr -> !llvm.ptr

        %0 = llvm.mlir.zero : !llvm.ptr
        %c0 = llvm.mlir.constant(2 : i64) : i64
        llvm.br ^bb1
      ^bb1:
        %r0 = "mqtdyn.allocQubitRegister" (%c0) : (i64) -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        llvm.br ^bb2
      ^bb2:
        llvm.return
    }
    llvm.func @__quantum__rt__array_get_element_ptr_1d(!llvm.ptr, i64) -> !llvm.ptr
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
    llvm.func @__quantum__rt__qubit_allocate_array(i64) -> !llvm.ptr
}

// -----
// This test checks if the extract from register call is correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertExtractOpAttribute()
    llvm.func @testConvertExtractOpAttribute() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[size:.*]] = llvm.mlir.constant(2 : i64) : i64
        // CHECK: %[[index:.*]] = llvm.mlir.constant(0 : i64) : i64
        // CHECK: %[[r_0:.*]] = llvm.call @__quantum__rt__qubit_allocate_array(%[[size]]) : (i64) -> !llvm.ptr
        // CHECK: %[[ptr_0:.*]] = llvm.call @__quantum__rt__array_get_element_ptr_1d(%[[r_0]], %[[index]]) : (!llvm.ptr, i64) -> !llvm.ptr
        // CHECK: %[[q_0:.*]] = llvm.load %[[ptr_0]] : !llvm.ptr -> !llvm.ptr

        %0 = llvm.mlir.zero : !llvm.ptr
        %c0 = llvm.mlir.constant(2 : i64) : i64
        %c1 = llvm.mlir.constant(0 : i64) : i64
        llvm.br ^bb1
      ^bb1:
        %r0 = "mqtdyn.allocQubitRegister" (%c0) : (i64) -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%r0, %c1) : (!mqtdyn.QubitRegister, i64) -> !mqtdyn.Qubit
        llvm.br ^bb2
      ^bb2:
        llvm.return
    }
    llvm.func @__quantum__rt__array_get_element_ptr_1d(!llvm.ptr, i64) -> !llvm.ptr
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
    llvm.func @__quantum__rt__qubit_allocate_array(i64) -> !llvm.ptr
}

// -----
// This test checks if the dealloc register call is correctly converted
module {
    // CHECK-LABEL: llvm.func @testDeallocRegister()
    llvm.func @testDeallocRegister() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[r_0:.*]] = llvm.call @__quantum__rt__qubit_allocate_array(%[[ANY:.*]]) : (i64) -> !llvm.ptr
        // CHECK: llvm.call @__quantum__rt__qubit_release_array(%[[r_0]]) : (!llvm.ptr) -> ()

        %0 = llvm.mlir.zero : !llvm.ptr
        llvm.br ^bb1
      ^bb1:
         %r0 = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        "mqtdyn.deallocQubitRegister"(%r0) : (!mqtdyn.QubitRegister) -> ()
        llvm.br ^bb2
      ^bb2:
        llvm.return
    }
    llvm.func @__quantum__rt__qubit_release_array(!llvm.ptr)
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
    llvm.func @__quantum__rt__qubit_allocate_array(i64) -> !llvm.ptr
}

// -----
// This test checks if the dealloc register call is correctly converted
module {
    llvm.mlir.global internal constant @mlir.llvm.nameless_global_0("r0\00") {addr_space = 0 : i32, dso_local}
    // CHECK-LABEL: llvm.func @testMeasureOp()
    llvm.func @testMeasureOp() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[a_0:.*]] = llvm.mlir.addressof @mlir.llvm.nameless_global_0 : !llvm.ptr
        // CHECK: %[[c_0:.*]] = llvm.mlir.constant(-1 : i32) : i32

        // CHECK: %[[r_0:.*]] = llvm.call @__quantum__rt__qubit_allocate_array(%[[ANY:.*]]) : (i64) -> !llvm.ptr
        // CHECK: %[[ptr_0:.*]] = llvm.call @__quantum__rt__array_get_element_ptr_1d(%[[r_0]], %[[ANY:.*]]) : (!llvm.ptr, i64) -> !llvm.ptr
        // CHECK: %[[q_0:.*]] = llvm.load %[[ptr_0]] : !llvm.ptr -> !llvm.ptr
        // CHECK: %[[m_0:.*]] = llvm.call @__quantum__qis__m__body(%[[q_0]]) : (!llvm.ptr) -> !llvm.ptr
        // CHECK  llvm.call @__quantum__rt__result_record_output(%[[m_0]], %[[a_0]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__rt__result_update_reference_count(%[[m_0]], %[[c_0]]) : (!llvm.ptr, i32) -> ()

        %0 = llvm.mlir.zero : !llvm.ptr
        %a0 = llvm.mlir.addressof @mlir.llvm.nameless_global_0 : !llvm.ptr
        %c0 = llvm.mlir.constant(-1 : i32) : i32
        llvm.br ^bb1
      ^bb1:
        %r0 = "mqtdyn.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %m0 = "mqtdyn.measure"(%q0) : (!mqtdyn.Qubit) -> i1
        "mqtdyn.deallocQubitRegister"(%r0) : (!mqtdyn.QubitRegister) -> ()
        llvm.br ^bb2
      ^bb2:
        llvm.return
    }
    llvm.func @__quantum__rt__result_update_reference_count(!llvm.ptr, i32)
    llvm.func @__quantum__qis__m__body(!llvm.ptr) -> !llvm.ptr
    llvm.func @__quantum__rt__result_record_output(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__rt__qubit_release_array(!llvm.ptr)
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
    llvm.func @__quantum__rt__qubit_allocate_array(i64) -> !llvm.ptr
}

// -----
// This test checks if the dealloc register call is correctly converted
module {
    llvm.mlir.global internal constant @mlir.llvm.nameless_global_0("r0\00") {addr_space = 0 : i32, dso_local}
    llvm.mlir.global internal constant @mlir.llvm.nameless_global_1("r1\00") {addr_space = 0 : i32, dso_local}
    // CHECK-LABEL: llvm.func @testMultipleMeasureOp()
    llvm.func @testMultipleMeasureOp() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[a_0:.*]] = llvm.mlir.addressof @mlir.llvm.nameless_global_0 : !llvm.ptr
        // CHECK: %[[a_1:.*]] = llvm.mlir.addressof @mlir.llvm.nameless_global_1 : !llvm.ptr
        // CHECK: %[[c_0:.*]] = llvm.mlir.constant(-1 : i32) : i32

        // CHECK: %[[r_0:.*]] = llvm.call @__quantum__rt__qubit_allocate_array(%[[ANY:.*]]) : (i64) -> !llvm.ptr
        // CHECK: %[[ptr_0:.*]] = llvm.call @__quantum__rt__array_get_element_ptr_1d(%[[r_0]], %[[ANY:.*]]) : (!llvm.ptr, i64) -> !llvm.ptr
        // CHECK: %[[q_0:.*]] = llvm.load %[[ptr_0]] : !llvm.ptr -> !llvm.ptr
        // CHECK: %[[ptr_1:.*]] = llvm.call @__quantum__rt__array_get_element_ptr_1d(%[[r_0]], %[[ANY:.*]]) : (!llvm.ptr, i64) -> !llvm.ptr
        // CHECK: %[[q_1:.*]] = llvm.load %[[ptr_1]] : !llvm.ptr -> !llvm.ptr
        // CHECK: %[[m_0:.*]] = llvm.call @__quantum__qis__m__body(%[[q_0]]) : (!llvm.ptr) -> !llvm.ptr
        // CHECK  llvm.call @__quantum__rt__result_record_output(%[[m_0]], %[[a_0]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__rt__result_update_reference_count(%[[m_0]], %[[c_0]]) : (!llvm.ptr, i32) -> ()
        // CHECK: %[[m_1:.*]] = llvm.call @__quantum__qis__m__body(%[[q_1]]) : (!llvm.ptr) -> !llvm.ptr
        // CHECK  llvm.call @__quantum__rt__result_record_output(%[[m_1]], %[[a_1]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__rt__result_update_reference_count(%[[m_1]], %[[c_0]]) : (!llvm.ptr, i32) -> ()

        %0 = llvm.mlir.zero : !llvm.ptr
        %a0 = llvm.mlir.addressof @mlir.llvm.nameless_global_0 : !llvm.ptr
        %a1 = llvm.mlir.addressof @mlir.llvm.nameless_global_1 : !llvm.ptr
        %c0 = llvm.mlir.constant(-1 : i32) : i32
        llvm.br ^bb1
      ^bb1:
        %r0 = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q1 = "mqtdyn.extractQubit"(%r0) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %m0 = "mqtdyn.measure"(%q0) : (!mqtdyn.Qubit) -> i1
        %m1 = "mqtdyn.measure"(%q1) : (!mqtdyn.Qubit) -> i1
        "mqtdyn.deallocQubitRegister"(%r0) : (!mqtdyn.QubitRegister) -> ()
        llvm.br ^bb2
      ^bb2:
        llvm.return
    }
    llvm.func @__quantum__rt__result_update_reference_count(!llvm.ptr, i32)
    llvm.func @__quantum__qis__m__body(!llvm.ptr) -> !llvm.ptr
    llvm.func @__quantum__rt__result_record_output(!llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__rt__qubit_release_array(!llvm.ptr)
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
    llvm.func @__quantum__rt__qubit_allocate_array(i64) -> !llvm.ptr
}

// -----
// This test checks if the dealloc register call is correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertSingleQubitOp()
    llvm.func @testConvertSingleQubitOp() attributes {passthrough = ["entry_point"]}  {
        // CHECK: llvm.call @__quantum__qis__i__body(%[[q_0:.*]]) : (!llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__h__body(%[[q_0]]) : (!llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__x__body(%[[q_0]]) : (!llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__y__body(%[[q_0]]) : (!llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__z__body(%[[q_0]]) : (!llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__s__body(%[[q_0]]) : (!llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__sdg__body(%[[q_0]]) : (!llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__t__body(%[[q_0]]) : (!llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__tdg__body(%[[q_0]]) : (!llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__v__body(%[[q_0]]) : (!llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__vdg__body(%[[q_0]]) : (!llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__sx__body(%[[q_0]]) : (!llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__sxdg__body(%[[q_0]]) : (!llvm.ptr) -> ()

        %0 = llvm.mlir.zero : !llvm.ptr
        llvm.br ^bb1
      ^bb1:
        %r0 = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        mqtdyn.i() %q0
        mqtdyn.h() %q0
        mqtdyn.x() %q0
        mqtdyn.y() %q0
        mqtdyn.z() %q0
        mqtdyn.s() %q0
        mqtdyn.sdg() %q0
        mqtdyn.t() %q0
        mqtdyn.tdg() %q0
        mqtdyn.v() %q0
        mqtdyn.vdg() %q0
        mqtdyn.sx() %q0
        mqtdyn.sxdg() %q0

        "mqtdyn.deallocQubitRegister"(%r0) : (!mqtdyn.QubitRegister) -> ()
        llvm.br ^bb2
      ^bb2:
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
// This test checks if the dealloc register call is correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertTwoTargetOp()
    llvm.func @testConvertTwoTargetOp() attributes {passthrough = ["entry_point"]}  {
        // CHECK: llvm.call @__quantum__qis__swap__body(%[[q_0:.*]], %[[q_1:.*]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__iswap__body(%[[q_0]], %[[q_1]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__iswapdg__body(%[[q_0]], %[[q_1]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__peres__body(%[[q_0]], %[[q_1]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__peresdg__body(%[[q_0]], %[[q_1]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__dcx__body(%[[q_0]], %[[q_1]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__ecr__body(%[[q_0]], %[[q_1]]) : (!llvm.ptr, !llvm.ptr) -> ()

        %0 = llvm.mlir.zero : !llvm.ptr
        llvm.br ^bb1
      ^bb1:
        %r0 = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q1 = "mqtdyn.extractQubit"(%r0) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        mqtdyn.swap() %q0, %q1
        mqtdyn.iswap() %q0, %q1
        mqtdyn.iswapdg() %q0, %q1
        mqtdyn.peres() %q0, %q1
        mqtdyn.peresdg() %q0, %q1
        mqtdyn.dcx() %q0, %q1
        mqtdyn.ecr() %q0, %q1

        "mqtdyn.deallocQubitRegister"(%r0) : (!mqtdyn.QubitRegister) -> ()
        llvm.br ^bb2
      ^bb2:
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
// This test checks if the dealloc register call is correctly converted
module {
    // CHECK-LABEL: llvm.func @testSingleQubitRotationOp()
    llvm.func @testSingleQubitRotationOp() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[c_0:.*]] = llvm.mlir.constant(3.000000e-01 : f64) : f64
        // CHECK: llvm.call @__quantum__qis__u2__body(%[[c_0]], %[[c_0]], %[[q_0:.*]]) : (f64, f64, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__p__body(%[[c_0]], %[[q_0]]) : (f64, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__rx__body(%[[c_0]], %[[q_0]]) : (f64, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__ry__body(%[[c_0]], %[[q_0]]) : (f64, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__rz__body(%[[c_0]], %[[q_0]]) : (f64, !llvm.ptr) -> ()

        %0 = llvm.mlir.zero : !llvm.ptr
        %cst = llvm.mlir.constant (3.000000e-01 : f64) : f64
        llvm.br ^bb1
      ^bb1:

        %r0 = "mqtdyn.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        mqtdyn.u(%cst, %cst, %cst) %q0
        mqtdyn.u2(%cst, %cst) %q0
        mqtdyn.p(%cst) %q0
        mqtdyn.rx(%cst) %q0
        mqtdyn.ry(%cst) %q0
        mqtdyn.rz(%cst) %q0

        "mqtdyn.deallocQubitRegister"(%r0) : (!mqtdyn.QubitRegister) -> ()

        llvm.br ^bb2
      ^bb2:
        llvm.return
    }
    llvm.func @__quantum__qis__u2__body(f64, f64, !llvm.ptr)
    llvm.func @__quantum__qis__p__body(f64, !llvm.ptr)
    llvm.func @__quantum__qis__rx__body(f64, !llvm.ptr)
    llvm.func @__quantum__qis__ry__body(f64, !llvm.ptr)
    llvm.func @__quantum__qis__rz__body(f64, !llvm.ptr)
    llvm.func @__quantum__rt__array_get_element_ptr_1d(!llvm.ptr, i64) -> !llvm.ptr
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
    llvm.func @__quantum__rt__qubit_allocate_array(i64) -> !llvm.ptr
}

// -----
// This test checks if the dealloc register call is correctly converted
module {
    // CHECK-LABEL: llvm.func @testMultipleQubitRotationOp()
    llvm.func @testMultipleQubitRotationOp() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[c_0:.*]] = llvm.mlir.constant(3.000000e-01 : f64) : f64
        // CHECK: llvm.call @__quantum__qis__rxx__body(%[[c_0]], %[[q_0:.*]], %[[q_1:.*]]) : (f64, !llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__ryy__body(%[[c_0]], %[[q_0]], %[[q_1]]) : (f64, !llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__rzz__body(%[[c_0]], %[[q_0]], %[[q_1]]) : (f64, !llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__rzx__body(%[[c_0]], %[[q_0]], %[[q_1]]) : (f64, !llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__xxminusyy__body(%[[c_0]], %[[c_0]], %[[q_0]], %[[q_1]]) : (f64, f64, !llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__xxplusyy__body(%[[c_0]], %[[c_0]], %[[q_0]], %[[q_1]]) : (f64, f64, !llvm.ptr, !llvm.ptr) -> ()


        %0 = llvm.mlir.zero : !llvm.ptr
        %cst = llvm.mlir.constant(3.000000e-01 : f64) : f64
        llvm.br ^bb1
      ^bb1:
        %r0 = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q1 = "mqtdyn.extractQubit"(%r0) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        mqtdyn.rxx(%cst) %q0, %q1
        mqtdyn.ryy(%cst) %q0, %q1
        mqtdyn.rzz(%cst) %q0, %q1
        mqtdyn.rzx(%cst) %q0, %q1
        mqtdyn.xxminusyy(%cst, %cst) %q0, %q1
        mqtdyn.xxplusyy(%cst, %cst) %q0, %q1

        "mqtdyn.deallocQubitRegister"(%r0) : (!mqtdyn.QubitRegister) -> ()
        llvm.br ^bb2
      ^bb2:
        llvm.return
    }
    llvm.func @__quantum__qis__rxx__body(f64, !llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__qis__ryy__body(f64, !llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__qis__rzz__body(f64, !llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__qis__rzx__body(f64, !llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__qis__xxminusyy__body(f64, f64, !llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__qis__xxplusyy__body(f64, f64, !llvm.ptr, !llvm.ptr)
    llvm.func @__quantum__rt__array_get_element_ptr_1d(!llvm.ptr, i64) -> !llvm.ptr
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
    llvm.func @__quantum__rt__qubit_allocate_array(i64) -> !llvm.ptr
}

// -----
// This test checks if the dealloc register call is correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertControlledOp()
    llvm.func @testConvertControlledOp() attributes {passthrough = ["entry_point"]}  {
        // CHECK: llvm.call @__quantum__qis__cnot__body(%[[ANY:.*]], %[[ANY:.*]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__ccx__body(%[[ANY:.*]], %[[ANY:.*]], %[[ANY:.*]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__cz__body(%[[ANY:.*]], %[[ANY:.*]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__crx__body(%[[ANY:.*]], %[[ANY:.*]], %[[ANY:.*]]) : (f64, !llvm.ptr, !llvm.ptr) -> ()

        %0 = llvm.mlir.zero : !llvm.ptr
        %cst = llvm.mlir.constant(3.000000e-01 : f64) : f64
        llvm.br ^bb1
      ^bb1:
        %r0 = "mqtdyn.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q1 = "mqtdyn.extractQubit"(%r0) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q2 = "mqtdyn.extractQubit"(%r0) <{index_attr = 2 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        mqtdyn.x() %q0 ctrl %q1
        mqtdyn.x() %q0 ctrl %q1, %q2
        mqtdyn.z() %q0 nctrl %q1
        mqtdyn.rx(%cst) %q0 ctrl %q1

        "mqtdyn.deallocQubitRegister"(%r0) : (!mqtdyn.QubitRegister) -> ()
        llvm.br ^bb2
      ^bb2:
        llvm.return
    }
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
}

// -----
// This test checks if the dealloc register call is correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertGPhaseOp()
    llvm.func @testConvertGPhaseOp() attributes {passthrough = ["entry_point"]}  {
        // CHECK: llvm.call @__quantum__qis__gphase__body(%[[ANY:.*]]) : (f64) -> ()

        %0 = llvm.mlir.zero : !llvm.ptr
        %cst = llvm.mlir.constant(3.000000e-01 : f64) : f64
        llvm.br ^bb1
      ^bb1:
        mqtdyn.gphase(%cst)
        llvm.br ^bb2
      ^bb2:
        llvm.return
    }
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
}

// -----
// This test checks if the dealloc register call is correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertGPhaseOpControlled()
    llvm.func @testConvertGPhaseOpControlled() attributes {passthrough = ["entry_point"]}  {
        // CHECK: llvm.call @__quantum__qis__cgphase__body(%[[ANY:.*]], %[[ANY:.*]]) : (f64, !llvm.ptr) -> ()

        %0 = llvm.mlir.zero : !llvm.ptr
        %cst = llvm.mlir.constant(3.000000e-01 : f64) : f64
        llvm.br ^bb1
      ^bb1:
        %r0 = "mqtdyn.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        mqtdyn.gphase(%cst) ctrl %q0
        "mqtdyn.deallocQubitRegister"(%r0) : (!mqtdyn.QubitRegister) -> ()
        llvm.br ^bb2
      ^bb2:
        llvm.return
    }
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
}

// -----
// This test checks if a barrierOp is converted correctly
module {
    // CHECK-LABEL: llvm.func @testConvertBarrierOp()
    llvm.func @testConvertBarrierOp() attributes {passthrough = ["entry_point"]}  {
        // CHECK: llvm.call @__quantum__qis__barrier__body(%[[ANY:.*]]) : (!llvm.ptr) -> ()

        %0 = llvm.mlir.zero : !llvm.ptr
        llvm.br ^bb1
      ^bb1:
        %r0 = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        mqtdyn.barrier() %q0
        "mqtdyn.deallocQubitRegister"(%r0) : (!mqtdyn.QubitRegister) -> ()
        llvm.br ^bb2
      ^bb2:
        llvm.return
    }
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
}

// -----
// This test checks if a Bell state is converted correctly.
module {
    llvm.mlir.global internal constant @mlir.llvm.nameless_global_0("r0\00") {addr_space = 0 : i32, dso_local}
    llvm.mlir.global internal constant @mlir.llvm.nameless_global_1("r1\00") {addr_space = 0 : i32, dso_local}

    // CHECK-LABEL: llvm.func @bellState()
    llvm.func @bellState() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[r_0:.*]] = llvm.call @__quantum__rt__qubit_allocate_array(%[[ANY:.*]]) : (i64) -> !llvm.ptr
        // CHECK-DAG: %[[index_0:.*]] = llvm.mlir.constant(0 : i64) : i64
        // CHECK: %[[ptr_0:.*]] = llvm.call @__quantum__rt__array_get_element_ptr_1d(%[[r_0]], %[[index_0]]) : (!llvm.ptr, i64) -> !llvm.ptr
        // CHECK: %[[q_0:.*]] = llvm.load %[[ptr_0]] : !llvm.ptr -> !llvm.ptr
        // CHECK-DAG: %[[index_1:.*]] = llvm.mlir.constant(1 : i64) : i64
        // CHECK: %[[ptr_1:.*]] = llvm.call @__quantum__rt__array_get_element_ptr_1d(%[[r_0]], %[[index_1]]) : (!llvm.ptr, i64) -> !llvm.ptr
        // CHECK: %[[q_1:.*]] = llvm.load %[[ptr_1]] : !llvm.ptr -> !llvm.ptr
        // CHECK: llvm.call @__quantum__qis__h__body(%[[q_0]]) : (!llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__cnot__body(%[[q_1]], %[[q_0]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: %[[m_0:.*]] = llvm.call @__quantum__qis__m__body(%[[q_0]]) : (!llvm.ptr) -> !llvm.ptr
        // CHECK: llvm.call @__quantum__rt__result_record_output(%[[m_0]], %[[ANY:.*]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__rt__result_update_reference_count(%[[m_0]], %[[ANY:.*]]) : (!llvm.ptr, i32) -> ()
        // CHECK: %[[m_1:.*]] = llvm.call @__quantum__qis__m__body(%[[q_1]]) : (!llvm.ptr) -> !llvm.ptr
        // CHECK: llvm.call @__quantum__rt__result_record_output(%[[m_1]], %[[ANY:.*]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__rt__result_update_reference_count(%[[m_1]], %[[ANY:.*]]) : (!llvm.ptr, i32) -> ()
        // CHECK: llvm.call @__quantum__rt__qubit_release_array(%[[r_0]]) : (!llvm.ptr) -> ()

        %0 = llvm.mlir.zero : !llvm.ptr
        %a0 = llvm.mlir.addressof @mlir.llvm.nameless_global_0 : !llvm.ptr
        %a1 = llvm.mlir.addressof @mlir.llvm.nameless_global_1 : !llvm.ptr

        %c3 = llvm.mlir.constant(-1 : i32) : i32
        llvm.br ^bb1
      ^bb1:
        %r0 = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q1 = "mqtdyn.extractQubit"(%r0) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        mqtdyn.h() %q0
        mqtdyn.x() %q1 ctrl %q0
        %m0 = "mqtdyn.measure"(%q0) : (!mqtdyn.Qubit) -> i1
        %m1 = "mqtdyn.measure"(%q1) : (!mqtdyn.Qubit) -> i1

        "mqtdyn.deallocQubitRegister"(%r0) : (!mqtdyn.QubitRegister) -> ()
        llvm.br ^bb2
      ^bb2:
        llvm.return
    }
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
}
