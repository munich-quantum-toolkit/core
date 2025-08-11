// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file --mqtdyn-to-qir | FileCheck %s

// This test checks if the AllocOp is converted correctly using a static attribute
module {
    // CHECK-LABEL: llvm.func @testConvertAllocRegisterAttribute()
    func.func @testConvertAllocRegisterAttribute() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[size:.*]] = llvm.mlir.constant(2 : i64) : i64
        // CHECK: %[[r_0:.*]] = llvm.call @__quantum__rt__qubit_allocate_array(%[[size]]) : (i64) -> !llvm.ptr
        cf.br ^bb1
      ^bb1:
        %r0 = "mqtdyn.allocQubitRegister" () {"size_attr" = 2 : i64} : () -> !mqtdyn.QubitRegister
        cf.br ^bb2
      ^bb2:
        return
    }

}

// -----
// This test checks if the blocks for the base profile of QIR are correctly created
module {
    // CHECK-LABEL: llvm.func @testBlockCreation()
    func.func @testBlockCreation() attributes {passthrough = ["entry_point"]}  {
      // CHECK: llvm.br ^[[main:.*]]
      // CHECK: ^[[main]]:
      // CHECK: %[[size:.*]] = llvm.mlir.constant(2 : i64) : i64
      // CHECK: %[[r_0:.*]] = llvm.call @__quantum__rt__qubit_allocate_array(%[[size]]) : (i64) -> !llvm.ptr
      // CHECK: llvm.br ^[[end:.*]]
      // CHECK: ^[[end]]:
      // CHECK: llvm.return

      %r0 = "mqtdyn.allocQubitRegister" () {"size_attr" = 2 : i64} : () -> !mqtdyn.QubitRegister
      return
    }

}

// -----
// This test checks if the initialize operation and zero operation is inserted
module {
    // CHECK-LABEL: llvm.func @testInitialize()
    func.func @testInitialize() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[z_0:.*]] = llvm.mlir.zero : !llvm.ptr
        // CHECK: llvm.call @__quantum__rt__initialize(%[[z_0]]) : (!llvm.ptr) -> ()

        cf.br ^bb1
      ^bb1:
        cf.br ^bb2
      ^bb2:
        return
    }

}

// -----
// This test checks if the AllocOp is converted correctly using a dynamic operand
module {
    // CHECK-LABEL: llvm.func @testConvertAllocRegisterOperand()
    func.func @testConvertAllocRegisterOperand() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[size:.*]] = llvm.mlir.constant(2 : i64) : i64
        // CHECK: %[[r_0:.*]] = llvm.call @__quantum__rt__qubit_allocate_array(%[[size]]) : (i64) -> !llvm.ptr

        %c0 = arith.constant 2 : i64
        cf.br ^bb1
      ^bb1:
        %r0 = "mqtdyn.allocQubitRegister" (%c0) : (i64) -> !mqtdyn.QubitRegister
        cf.br ^bb2
      ^bb2:
        return
    }

}

// -----
// This test checks if the ExtractOp is converted correctly using a static attribute
module {
    // CHECK-LABEL: llvm.func @testConvertExtractOpAttribute()
    func.func @testConvertExtractOpAttribute() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[size:.*]] = llvm.mlir.constant(2 : i64) : i64
        // CHECK: %[[r_0:.*]] = llvm.call @__quantum__rt__qubit_allocate_array(%[[size]]) : (i64) -> !llvm.ptr
        // CHECK-DAG: %[[index:.*]] = llvm.mlir.constant(0 : i64) : i64
        // CHECK: %[[ptr_0:.*]] = llvm.call @__quantum__rt__array_get_element_ptr_1d(%[[r_0]], %[[index]]) : (!llvm.ptr, i64) -> !llvm.ptr
        // CHECK: %[[q_0:.*]] = llvm.load %[[ptr_0]] : !llvm.ptr -> !llvm.ptr

        %c0 = arith.constant 2 : i64
        cf.br ^bb1
      ^bb1:
        %r0 = "mqtdyn.allocQubitRegister" (%c0) : (i64) -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        cf.br ^bb2
      ^bb2:
        return
    }
}

// -----
// This test checks if the ExtractOp is converted correctly using a dynamic operand
module {
    // CHECK-LABEL: llvm.func @testConvertExtractOpAttribute()
    func.func @testConvertExtractOpAttribute() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[size:.*]] = llvm.mlir.constant(2 : i64) : i64
        // CHECK: %[[index:.*]] = llvm.mlir.constant(0 : i64) : i64
        // CHECK: %[[r_0:.*]] = llvm.call @__quantum__rt__qubit_allocate_array(%[[size]]) : (i64) -> !llvm.ptr
        // CHECK: %[[ptr_0:.*]] = llvm.call @__quantum__rt__array_get_element_ptr_1d(%[[r_0]], %[[index]]) : (!llvm.ptr, i64) -> !llvm.ptr
        // CHECK: %[[q_0:.*]] = llvm.load %[[ptr_0]] : !llvm.ptr -> !llvm.ptr

        %c0 = arith.constant 2 : i64
        %c1 = arith.constant 0 : i64
        cf.br ^bb1
      ^bb1:
        %r0 = "mqtdyn.allocQubitRegister" (%c0) : (i64) -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%r0, %c1) : (!mqtdyn.QubitRegister, i64) -> !mqtdyn.Qubit
        cf.br ^bb2
      ^bb2:
        return
    }
}

// -----
// This test checks if the dealloc register call is correctly converted
module {
    // CHECK-LABEL: llvm.func @testDeallocRegister()
    func.func @testDeallocRegister() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[r_0:.*]] = llvm.call @__quantum__rt__qubit_allocate_array(%[[ANY:.*]]) : (i64) -> !llvm.ptr
        // CHECK: llvm.call @__quantum__rt__qubit_release_array(%[[r_0]]) : (!llvm.ptr) -> ()

        cf.br ^bb1
      ^bb1:
         %r0 = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        "mqtdyn.deallocQubitRegister"(%r0) : (!mqtdyn.QubitRegister) -> ()
        cf.br ^bb2
      ^bb2:
        return
    }
}

// -----
// This test checks if the measure operation is correctly converted
module {
    llvm.mlir.global internal constant @mlir.llvm.nameless_global_0("r0\00") {addr_space = 0 : i32, dso_local}
    // CHECK-LABEL: llvm.func @testMeasureOp()
    func.func @testMeasureOp() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[a_0:.*]] = llvm.mlir.addressof @mlir.llvm.nameless_global_0 : !llvm.ptr
        // CHECK: %[[c_0:.*]] = llvm.mlir.constant(-1 : i32) : i32

        // CHECK: %[[r_0:.*]] = llvm.call @__quantum__rt__qubit_allocate_array(%[[ANY:.*]]) : (i64) -> !llvm.ptr
        // CHECK: %[[ptr_0:.*]] = llvm.call @__quantum__rt__array_get_element_ptr_1d(%[[r_0]], %[[ANY:.*]]) : (!llvm.ptr, i64) -> !llvm.ptr
        // CHECK: %[[q_0:.*]] = llvm.load %[[ptr_0]] : !llvm.ptr -> !llvm.ptr
        // CHECK: %[[m_0:.*]] = llvm.call @__quantum__qis__m__body(%[[q_0]]) : (!llvm.ptr) -> !llvm.ptr
        // CHECK: llvm.call @__quantum__rt__result_record_output(%[[m_0]], %[[a_0]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__rt__result_update_reference_count(%[[m_0]], %[[c_0]]) : (!llvm.ptr, i32) -> ()
        // CHECK: %[[i_0:.*]] = llvm.call @__quantum__rt__read_result(%[[m_0]]) : (!llvm.ptr) -> i1

        %a0 = llvm.mlir.addressof @mlir.llvm.nameless_global_0 : !llvm.ptr
        %c0 = arith.constant -1 : i32
        cf.br ^bb1
      ^bb1:
        %r0 = "mqtdyn.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %m0 = "mqtdyn.measure"(%q0) : (!mqtdyn.Qubit) -> i1
        "mqtdyn.deallocQubitRegister"(%r0) : (!mqtdyn.QubitRegister) -> ()
        cf.br ^bb2
      ^bb2:
        return
    }
}

// -----
// This test checks if multiple measure operations are correctly converted
module {
    llvm.mlir.global internal constant @mlir.llvm.nameless_global_0("r0\00") {addr_space = 0 : i32, dso_local}
    llvm.mlir.global internal constant @mlir.llvm.nameless_global_1("r1\00") {addr_space = 0 : i32, dso_local}
    // CHECK-LABEL: llvm.func @testMultipleMeasureOp()
    func.func @testMultipleMeasureOp() attributes {passthrough = ["entry_point"]}  {
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
        // CHECK: %[[i_0:.*]] = llvm.call @__quantum__rt__read_result(%[[m_0]]) : (!llvm.ptr) -> i1
        // CHECK: %[[m_1:.*]] = llvm.call @__quantum__qis__m__body(%[[q_1]]) : (!llvm.ptr) -> !llvm.ptr
        // CHECK  llvm.call @__quantum__rt__result_record_output(%[[m_1]], %[[a_1]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__rt__result_update_reference_count(%[[m_1]], %[[c_0]]) : (!llvm.ptr, i32) -> ()
        // CHECK: %[[i_1:.*]] = llvm.call @__quantum__rt__read_result(%[[m_1]]) : (!llvm.ptr) -> i1

        %a0 = llvm.mlir.addressof @mlir.llvm.nameless_global_0 : !llvm.ptr
        %a1 = llvm.mlir.addressof @mlir.llvm.nameless_global_1 : !llvm.ptr
        %c0 = arith.constant -1  : i32
        cf.br ^bb1
      ^bb1:
        %r0 = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q1 = "mqtdyn.extractQubit"(%r0) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %m0, %m1 = "mqtdyn.measure"(%q0, %q1) : (!mqtdyn.Qubit, !mqtdyn.Qubit) -> (i1, i1)
        "mqtdyn.deallocQubitRegister"(%r0) : (!mqtdyn.QubitRegister) -> ()
        cf.br ^bb2
      ^bb2:
        return
    }
}

// -----
// This test checks if the single qubit gates are correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertSingleQubitOp()
    func.func @testConvertSingleQubitOp() attributes {passthrough = ["entry_point"]}  {
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

        cf.br ^bb1
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
        cf.br ^bb2
      ^bb2:
        return
    }
}

// -----
// This test checks if the two target gates are correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertTwoTargetOp()
    func.func @testConvertTwoTargetOp() attributes {passthrough = ["entry_point"]}  {
        // CHECK: llvm.call @__quantum__qis__swap__body(%[[q_0:.*]], %[[q_1:.*]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__iswap__body(%[[q_0]], %[[q_1]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__iswapdg__body(%[[q_0]], %[[q_1]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__peres__body(%[[q_0]], %[[q_1]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__peresdg__body(%[[q_0]], %[[q_1]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__dcx__body(%[[q_0]], %[[q_1]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__ecr__body(%[[q_0]], %[[q_1]]) : (!llvm.ptr, !llvm.ptr) -> ()

        cf.br ^bb1
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
        cf.br ^bb2
      ^bb2:
        return
    }
}

// -----
// This test checks if the single qubit rotation gates are correctly converted
module {
    // CHECK-LABEL: llvm.func @testSingleQubitRotationOp()
    func.func @testSingleQubitRotationOp() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[c_0:.*]] = llvm.mlir.constant(3.000000e-01 : f64) : f64
        // CHECK: llvm.call @__quantum__qis__u2__body(%[[c_0]], %[[c_0]], %[[q_0:.*]]) : (f64, f64, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__p__body(%[[c_0]], %[[q_0]]) : (f64, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__rx__body(%[[c_0]], %[[q_0]]) : (f64, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__ry__body(%[[c_0]], %[[q_0]]) : (f64, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__rz__body(%[[c_0]], %[[q_0]]) : (f64, !llvm.ptr) -> ()

        %c0 = arith.constant 3.000000e-01 : f64
        cf.br ^bb1
      ^bb1:

        %r0 = "mqtdyn.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        mqtdyn.u(%c0, %c0, %c0) %q0
        mqtdyn.u2(%c0, %c0) %q0
        mqtdyn.p(%c0) %q0
        mqtdyn.rx(%c0) %q0
        mqtdyn.ry(%c0) %q0
        mqtdyn.rz(%c0) %q0

        "mqtdyn.deallocQubitRegister"(%r0) : (!mqtdyn.QubitRegister) -> ()

        cf.br ^bb2
      ^bb2:
        return
    }
}

// -----
// This test checks if the multiple qubit rotation gates are correctly converted
module {
    // CHECK-LABEL: llvm.func @testMultipleQubitRotationOp()
    func.func @testMultipleQubitRotationOp() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[c_0:.*]] = llvm.mlir.constant(3.000000e-01 : f64) : f64
        // CHECK: llvm.call @__quantum__qis__rxx__body(%[[c_0]], %[[q_0:.*]], %[[q_1:.*]]) : (f64, !llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__ryy__body(%[[c_0]], %[[q_0]], %[[q_1]]) : (f64, !llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__rzz__body(%[[c_0]], %[[q_0]], %[[q_1]]) : (f64, !llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__rzx__body(%[[c_0]], %[[q_0]], %[[q_1]]) : (f64, !llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__xxminusyy__body(%[[c_0]], %[[c_0]], %[[q_0]], %[[q_1]]) : (f64, f64, !llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__xxplusyy__body(%[[c_0]], %[[c_0]], %[[q_0]], %[[q_1]]) : (f64, f64, !llvm.ptr, !llvm.ptr) -> ()

        %c0 = arith.constant 3.000000e-01 : f64
        cf.br ^bb1
      ^bb1:
        %r0 = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q1 = "mqtdyn.extractQubit"(%r0) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        mqtdyn.rxx(%c0) %q0, %q1
        mqtdyn.ryy(%c0) %q0, %q1
        mqtdyn.rzz(%c0) %q0, %q1
        mqtdyn.rzx(%c0) %q0, %q1
        mqtdyn.xxminusyy(%c0, %c0) %q0, %q1
        mqtdyn.xxplusyy(%c0, %c0) %q0, %q1

        "mqtdyn.deallocQubitRegister"(%r0) : (!mqtdyn.QubitRegister) -> ()
        cf.br ^bb2
      ^bb2:
        return
    }
}

// -----
// This test checks if controlled gates are correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertControlledOp()
    func.func @testConvertControlledOp() attributes {passthrough = ["entry_point"]}  {
        // CHECK: llvm.call @__quantum__qis__cnot__body(%[[ANY:.*]], %[[ANY:.*]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__ccx__body(%[[ANY:.*]], %[[ANY:.*]], %[[ANY:.*]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__cz__body(%[[ANY:.*]], %[[ANY:.*]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__crx__body(%[[ANY:.*]], %[[ANY:.*]], %[[ANY:.*]]) : (f64, !llvm.ptr, !llvm.ptr) -> ()

        %c0 = arith.constant 3.000000e-01 : f64
        cf.br ^bb1
      ^bb1:
        %r0 = "mqtdyn.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q1 = "mqtdyn.extractQubit"(%r0) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q2 = "mqtdyn.extractQubit"(%r0) <{index_attr = 2 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        mqtdyn.x() %q0 ctrl %q1
        mqtdyn.x() %q0 ctrl %q1, %q2
        mqtdyn.z() %q0 nctrl %q1
        mqtdyn.rx(%c0) %q0 ctrl %q1

        "mqtdyn.deallocQubitRegister"(%r0) : (!mqtdyn.QubitRegister) -> ()
        cf.br ^bb2
      ^bb2:
        return
    }
}

// -----
// This test checks if the gphase operation is correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertGPhaseOp()
    func.func @testConvertGPhaseOp() attributes {passthrough = ["entry_point"]}  {
        // CHECK: llvm.call @__quantum__qis__gphase__body(%[[ANY:.*]]) : (f64) -> ()

        %0 = llvm.mlir.zero : !llvm.ptr
        %cst = llvm.mlir.constant(3.000000e-01 : f64) : f64
        cf.br ^bb1
      ^bb1:
        mqtdyn.gphase(%cst)
        cf.br ^bb2
      ^bb2:
        return
    }
}

// -----
// This test checks if the controlled gphase operation is correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertGPhaseOpControlled()
    func.func @testConvertGPhaseOpControlled() attributes {passthrough = ["entry_point"]}  {
        // CHECK: llvm.call @__quantum__qis__cgphase__body(%[[ANY:.*]], %[[ANY:.*]]) : (f64, !llvm.ptr) -> ()

        %cst = arith.constant 3.000000e-01 : f64
        cf.br ^bb1
      ^bb1:
        %r0 = "mqtdyn.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        mqtdyn.gphase(%cst) ctrl %q0
        "mqtdyn.deallocQubitRegister"(%r0) : (!mqtdyn.QubitRegister) -> ()
        cf.br ^bb2
      ^bb2:
        return
    }
}

// -----
// This test checks if the barrierOp is converted correctly
module {
    // CHECK-LABEL: llvm.func @testConvertBarrierOp()
    func.func @testConvertBarrierOp() attributes {passthrough = ["entry_point"]}  {
        // CHECK: llvm.call @__quantum__qis__barrier__body(%[[ANY:.*]]) : (!llvm.ptr) -> ()

        cf.br ^bb1
      ^bb1:
        %r0 = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        mqtdyn.barrier() %q0
        "mqtdyn.deallocQubitRegister"(%r0) : (!mqtdyn.QubitRegister) -> ()
        cf.br ^bb2
      ^bb2:
        return
    }
}

// -----
// This test checks if a Bell state is converted correctly.
module {
    llvm.mlir.global internal constant @mlir.llvm.nameless_global_0("r0\00") {addr_space = 0 : i32, dso_local}
    llvm.mlir.global internal constant @mlir.llvm.nameless_global_1("r1\00") {addr_space = 0 : i32, dso_local}

    // CHECK-LABEL: llvm.func @bellState()
    func.func @bellState() attributes {passthrough = ["entry_point"]}  {
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
        // CHECK: %[[i_0:.*]] = llvm.call @__quantum__rt__read_result(%[[m_0]]) : (!llvm.ptr) -> i1
        // CHECK: %[[m_1:.*]] = llvm.call @__quantum__qis__m__body(%[[q_1]]) : (!llvm.ptr) -> !llvm.ptr
        // CHECK: llvm.call @__quantum__rt__result_record_output(%[[m_1]], %[[ANY:.*]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__rt__result_update_reference_count(%[[m_1]], %[[ANY:.*]]) : (!llvm.ptr, i32) -> ()
        // CHECK: %[[i_1:.*]] = llvm.call @__quantum__rt__read_result(%[[m_1]]) : (!llvm.ptr) -> i1
        // CHECK: llvm.call @__quantum__rt__qubit_release_array(%[[r_0]]) : (!llvm.ptr) -> ()

        %a0 = llvm.mlir.addressof @mlir.llvm.nameless_global_0 : !llvm.ptr
        %a1 = llvm.mlir.addressof @mlir.llvm.nameless_global_1 : !llvm.ptr

        %c3 = arith.constant -1 : i32
        cf.br ^bb1
      ^bb1:
        %r0 = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q1 = "mqtdyn.extractQubit"(%r0) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        mqtdyn.h() %q0
        mqtdyn.x() %q1 ctrl %q0
        %m0 = "mqtdyn.measure"(%q0) : (!mqtdyn.Qubit) -> i1
        %m1 = "mqtdyn.measure"(%q1) : (!mqtdyn.Qubit) -> i1

        "mqtdyn.deallocQubitRegister"(%r0) : (!mqtdyn.QubitRegister) -> ()
        cf.br ^bb2
      ^bb2:
        return
    }
}
