// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file --mqtref-to-qir | FileCheck %s


// This test checks if the initialize operation and zero operation is inserted
module {
    // CHECK-LABEL: llvm.func @testInitialize()
    func.func @testInitialize() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[ptr_0:.*]] = llvm.mlir.zero : !llvm.ptr
        // CHECK: llvm.call @__quantum__rt__initialize(%[[ptr_0]]) : (!llvm.ptr) -> ()

        return
    }
}

// -----
// This test checks if the AllocaOp is converted correctly using a static attribute
module {
    // CHECK-LABEL: llvm.func @testConvertAllocaStatic()
    func.func @testConvertAllocaStatic() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[size:.*]] = llvm.mlir.constant(2 : i64) : i64
        // CHECK: %[[r_0:.*]] = llvm.call @__quantum__rt__qubit_allocate_array(%[[size]]) : (i64) -> !llvm.ptr

        %qreg = memref.alloc() : memref<2x!mqtref.Qubit>

        return
    }
}


// -----
// This test checks if the AllocaOp is converted correctly using a dynamic operand
module {
    // CHECK-LABEL: llvm.func @testConvertAllocaDynamic()
    func.func @testConvertAllocaDynamic() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[size:.*]] = llvm.mlir.constant(2 : index) : i64
        // CHECK: %[[r_0:.*]] = llvm.call @__quantum__rt__qubit_allocate_array(%[[size]]) : (i64) -> !llvm.ptr

        %i2 = arith.constant 2 : index
        %qreg = memref.alloc(%i2) : memref<?x!mqtref.Qubit>

        return
    }
}

// -----
// This test checks if the allocQubit is converted correctly
module {
    // CHECK-LABEL: llvm.func @testConvertAllocQubit()
    func.func @testConvertAllocQubit() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[q_0:.*]] = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
        // CHECK: %[[q_1:.*]] = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr

        %q0 = mqtref.allocQubit
        %q1 = mqtref.allocQubit

        return
    }
}

// -----
// This test checks if the deallocQubit call is converted correctly
module {
    // CHECK-LABEL: llvm.func @testConvertDeAllocQubit()
    func.func @testConvertDeAllocQubit() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[q_0:.*]] = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
        // CHECK: %[[q_1:.*]] = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
        // CHECK: llvm.call @__quantum__rt__qubit_release(%[[q_0]]) : (!llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__rt__qubit_release(%[[q_1]]) : (!llvm.ptr) -> ()

        %q0 = mqtref.allocQubit
        %q1 = mqtref.allocQubit

        mqtref.deallocQubit %q0
        mqtref.deallocQubit %q1
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

      %qreg = memref.alloc() : memref<2x!mqtref.Qubit>

      return
    }
}

// -----
// This test checks if the LoadOp is converted correctly
module {
    // CHECK-LABEL: llvm.func @testConvertLoadOp()
    func.func @testConvertLoadOp() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[size:.*]] = llvm.mlir.constant(1 : index) : i64
        // CHECK: %[[index:.*]] = llvm.mlir.constant(0 : index) : i64
        // CHECK: %[[r_0:.*]] = llvm.call @__quantum__rt__qubit_allocate_array(%[[size]]) : (i64) -> !llvm.ptr
        // CHECK: %[[ptr_0:.*]] = llvm.call @__quantum__rt__array_get_element_ptr_1d(%[[r_0]], %[[index]]) : (!llvm.ptr, i64) -> !llvm.ptr
        // CHECK: %[[q_0:.*]] = llvm.load %[[ptr_0]] : !llvm.ptr -> !llvm.ptr

        %i1 = arith.constant 1 : index
        %i0 = arith.constant 0 : index
        %qreg = memref.alloc(%i1) : memref<?x!mqtref.Qubit>
        %q0 = memref.load %qreg[%i0] : memref<?x!mqtref.Qubit>

        return
    }
}


// -----
// This test checks if the reset operation is correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertResetOp()
    func.func @testConvertResetOp() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[q_0:.*]] = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
        // CHECK: llvm.call @__quantum__qis__reset__body(%[[q_0]]) : (!llvm.ptr) -> ()

        %q0 = mqtref.allocQubit
        mqtref.reset %q0
        mqtref.deallocQubit %q0
        return
    }
}

// -----
// This test checks if the reset operation using static qubits is correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertResetOpStatic()
    func.func @testConvertResetOpStatic() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[q_0:.*]] = llvm.mlir.zero : !llvm.ptr
        // CHECK: llvm.call @__quantum__qis__reset__body(%[[q_0]]) : (!llvm.ptr) -> ()

        %q0 = mqtref.qubit 0
        mqtref.reset %q0
        mqtref.deallocQubit %q0
        return
    }
}

// -----
// This test checks if measure operations are converted correctly
module {
    // CHECK: llvm.mlir.global internal constant @mlir.llvm.nameless_global_1("r1\00") {addr_space = 0 : i32, dso_local}
    // CHECK: llvm.mlir.global internal constant @mlir.llvm.nameless_global_0("r0\00") {addr_space = 0 : i32, dso_local}

    // CHECK-LABEL: llvm.func @testMeasureOp()
    func.func @testMeasureOp() attributes {passthrough = ["entry_point"]}  {
        // CHECK-DAG: %[[a_0:.*]] = llvm.mlir.addressof @mlir.llvm.nameless_global_0 : !llvm.ptr
        // CHECK-DAG: %[[a_1:.*]] = llvm.mlir.addressof @mlir.llvm.nameless_global_1 : !llvm.ptr
        // CHECK-DAG: %[[ptr_0:.*]] = llvm.mlir.zero : !llvm.ptr
        // CHECK: %[[q_0:.*]] = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
        // CHECK: %[[q_1:.*]] = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
        // CHECK: llvm.call @__quantum__qis__mz__body(%[[q_0]], %[[ptr_0]]) : (!llvm.ptr, !llvm.ptr)
        // CHECK-DAG: %[[c_1:.*]] = llvm.mlir.constant(1 : i64) : i64
        // CHECK-DAG: %[[ptr_1:.*]] = llvm.inttoptr %[[c_1]] : i64 to !llvm.ptr
        // CHECK: llvm.call @__quantum__qis__mz__body(%[[q_1]], %[[ptr_1]]) : (!llvm.ptr, !llvm.ptr)
        // CHECK  llvm.call @__quantum__rt__result_record_output(%[[ptr_1]], %[[a_1]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK  llvm.call @__quantum__rt__result_record_output(%[[ptr_0]], %[[a_0]]) : (!llvm.ptr, !llvm.ptr) -> ()

        %q0 = mqtref.allocQubit
        %q1 = mqtref.allocQubit
        %mem = memref.alloca() : memref<2xi1>
        %m0 = mqtref.measure %q0
        %c0 = arith.constant 0 : index
        memref.store %m0, %mem[%c0] : memref<2xi1>
        %m1 = mqtref.measure %q1
        %c1 = arith.constant 1 : index
        memref.store %m1, %mem[%c1] : memref<2xi1>
        mqtref.deallocQubit %q0
        mqtref.deallocQubit %q1
        return
    }
}

// -----
// This test checks if measure operations using static qubits are converted correctly
module {
    // CHECK: llvm.mlir.global internal constant @mlir.llvm.nameless_global_1("r1\00") {addr_space = 0 : i32, dso_local}
    // CHECK: llvm.mlir.global internal constant @mlir.llvm.nameless_global_0("r0\00") {addr_space = 0 : i32, dso_local}

    // CHECK-LABEL: llvm.func @testMeasureOpStatic()
    func.func @testMeasureOpStatic() attributes {passthrough = ["entry_point"]}  {
        // CHECK-DAG: %[[a_0:.*]] = llvm.mlir.addressof @mlir.llvm.nameless_global_0 : !llvm.ptr
        // CHECK-DAG: %[[a_1:.*]] = llvm.mlir.addressof @mlir.llvm.nameless_global_1 : !llvm.ptr
        // CHECK: %[[ptr_0:.*]] = llvm.mlir.zero : !llvm.ptr
        // CHECK: %[[c_1:.*]] = llvm.mlir.constant(1 : i64) : i64
        // CHECK: %[[ptr_1:.*]] = llvm.inttoptr %[[c_1]] : i64 to !llvm.ptr
        // CHECK: llvm.call @__quantum__qis__mz__body(%[[ptr_0]], %[[ptr_0]]) : (!llvm.ptr, !llvm.ptr)
        // CHECK: llvm.call @__quantum__qis__mz__body(%[[ptr_1]], %[[ptr_1]]) : (!llvm.ptr, !llvm.ptr)
        // CHECK  llvm.call @__quantum__rt__result_record_output(%[[ptr_1]], %[[a_1]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK  llvm.call @__quantum__rt__result_record_output(%[[ptr_0]], %[[a_0]]) : (!llvm.ptr, !llvm.ptr) -> ()

        %q0 = mqtref.qubit 0
        %q1 = mqtref.qubit 1
        %mem = memref.alloca() : memref<2xi1>
        %m0 = mqtref.measure %q0
        %c0 = arith.constant 0 : index
        memref.store %m0, %mem[%c0] : memref<2xi1>
        %m1 = mqtref.measure %q1
        %c1 = arith.constant 1 : index
        memref.store %m1, %mem[%c1] : memref<2xi1>
        return
    }
}

// -----
// This test checks if measure operations are converted correctly that store the classical result on the same index
module {
    // CHECK: llvm.mlir.global internal constant @mlir.llvm.nameless_global_0("r0\00") {addr_space = 0 : i32, dso_local}

    // CHECK-LABEL: llvm.func @testMeasureOpOnSameIndex()
    func.func @testMeasureOpOnSameIndex() attributes {passthrough = ["entry_point"]}  {
        // CHECK-DAG: %[[a_0:.*]] = llvm.mlir.addressof @mlir.llvm.nameless_global_0 : !llvm.ptr
        // CHECK-DAG: %[[ptr_0:.*]] = llvm.mlir.zero : !llvm.ptr
        // CHECK: %[[q_0:.*]] = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
        // CHECK: %[[q_1:.*]] = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
        // CHECK: llvm.call @__quantum__qis__mz__body(%[[q_0]], %[[ptr_0]]) : (!llvm.ptr, !llvm.ptr)
        // CHECK-DAG: %[[c_1:.*]] = llvm.mlir.constant(1 : i64) : i64
        // CHECK-DAG: %[[ptr_1:.*]] = llvm.inttoptr %[[c_1]] : i64 to !llvm.ptr
        // CHECK: llvm.call @__quantum__qis__mz__body(%[[q_1]], %[[ptr_1]]) : (!llvm.ptr, !llvm.ptr)
        // CHECK  llvm.call @__quantum__rt__result_record_output(%[[ptr_1]], %[[a_0]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK  llvm.call @__quantum__rt__result_record_output(%[[ptr_0]], %[[a_0]]) : (!llvm.ptr, !llvm.ptr) -> ()

        %q0 = mqtref.allocQubit
        %q1 = mqtref.allocQubit
        %mem = memref.alloca() : memref<2xi1>
        %m0 = mqtref.measure %q0
        %c0 = arith.constant 0 : index
        memref.store %m0, %mem[%c0] : memref<2xi1>
        %m1 = mqtref.measure %q1
        memref.store %m1, %mem[%c0] : memref<2xi1>
        mqtref.deallocQubit %q0
        mqtref.deallocQubit %q1
        return
    }
}

// -----
// This test checks if measure operations using static qubits are converted correctly that store the classical result on the same index
module {
    // CHECK: llvm.mlir.global internal constant @mlir.llvm.nameless_global_0("r0\00") {addr_space = 0 : i32, dso_local}

    // CHECK-LABEL: llvm.func @testMeasureOpOnSameIndexStatic()
    func.func @testMeasureOpOnSameIndexStatic() attributes {passthrough = ["entry_point"]}  {
        // CHECK-DAG: %[[a_0:.*]] = llvm.mlir.addressof @mlir.llvm.nameless_global_0 : !llvm.ptr
        // CHECK: %[[ptr_0:.*]] = llvm.mlir.zero : !llvm.ptr
        // CHECK: %[[c_1:.*]] = llvm.mlir.constant(1 : i64) : i64
        // CHECK: %[[ptr_1:.*]] = llvm.inttoptr %[[c_1]] : i64 to !llvm.ptr
        // CHECK: llvm.call @__quantum__qis__mz__body(%[[ptr_0]], %[[ptr_0]]) : (!llvm.ptr, !llvm.ptr)
        // CHECK: llvm.call @__quantum__qis__mz__body(%[[ptr_1]], %[[ptr_1]]) : (!llvm.ptr, !llvm.ptr)
        // CHECK  llvm.call @__quantum__rt__result_record_output(%[[ptr_1]], %[[a_0]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK  llvm.call @__quantum__rt__result_record_output(%[[ptr_0]], %[[a_0]]) : (!llvm.ptr, !llvm.ptr) -> ()

        %q0 = mqtref.qubit 0
        %q1 = mqtref.qubit 1
        %mem = memref.alloca() : memref<2xi1>
        %m0 = mqtref.measure %q0
        %c0 = arith.constant 0 : index
        memref.store %m0, %mem[%c0] : memref<2xi1>
        %m1 = mqtref.measure %q1
        memref.store %m1, %mem[%c0] : memref<2xi1>
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

        %q0 = mqtref.allocQubit
        mqtref.i() %q0
        mqtref.h() %q0
        mqtref.x() %q0
        mqtref.y() %q0
        mqtref.z() %q0
        mqtref.s() %q0
        mqtref.sdg() %q0
        mqtref.t() %q0
        mqtref.tdg() %q0
        mqtref.v() %q0
        mqtref.vdg() %q0
        mqtref.sx() %q0
        mqtref.sxdg() %q0
        mqtref.deallocQubit %q0
        return
    }
}

// -----
// This test checks if the single qubit gates using static qubits are correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertSingleQubitOpStatic()
    func.func @testConvertSingleQubitOpStatic() attributes {passthrough = ["entry_point"]}  {
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

        %q0 = mqtref.qubit 0
        mqtref.i() %q0
        mqtref.h() %q0
        mqtref.x() %q0
        mqtref.y() %q0
        mqtref.z() %q0
        mqtref.s() %q0
        mqtref.sdg() %q0
        mqtref.t() %q0
        mqtref.tdg() %q0
        mqtref.v() %q0
        mqtref.vdg() %q0
        mqtref.sx() %q0
        mqtref.sxdg() %q0
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

        %q0 = mqtref.allocQubit
        %q1 = mqtref.allocQubit
        mqtref.swap() %q0, %q1
        mqtref.iswap() %q0, %q1
        mqtref.iswapdg() %q0, %q1
        mqtref.peres() %q0, %q1
        mqtref.peresdg() %q0, %q1
        mqtref.dcx() %q0, %q1
        mqtref.ecr() %q0, %q1
        mqtref.deallocQubit %q0
        mqtref.deallocQubit %q1
        return
    }
}

// -----
// This test checks if the two target gates using static qubits are correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertTwoTargetOpStatic()
    func.func @testConvertTwoTargetOpStatic() attributes {passthrough = ["entry_point"]}  {
        // CHECK: llvm.call @__quantum__qis__swap__body(%[[q_0:.*]], %[[q_1:.*]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__iswap__body(%[[q_0]], %[[q_1]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__iswapdg__body(%[[q_0]], %[[q_1]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__peres__body(%[[q_0]], %[[q_1]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__peresdg__body(%[[q_0]], %[[q_1]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__dcx__body(%[[q_0]], %[[q_1]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__ecr__body(%[[q_0]], %[[q_1]]) : (!llvm.ptr, !llvm.ptr) -> ()

        %q0 = mqtref.qubit 0
        %q1 = mqtref.qubit 1
        mqtref.swap() %q0, %q1
        mqtref.iswap() %q0, %q1
        mqtref.iswapdg() %q0, %q1
        mqtref.peres() %q0, %q1
        mqtref.peresdg() %q0, %q1
        mqtref.dcx() %q0, %q1
        mqtref.ecr() %q0, %q1
        return
    }
}

// -----
// This test checks if the single qubit rotation gates are correctly converted
module {
    // CHECK-LABEL: llvm.func @testSingleQubitRotationOp()
    func.func @testSingleQubitRotationOp() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[c_0:.*]] = llvm.mlir.constant(3.000000e-01 : f64) : f64
        // CHECK: %[[c_1:.*]] = llvm.mlir.constant(1.000000e-01 : f64) : f64
        // CHECK: %[[c_2:.*]] = llvm.mlir.constant(2.000000e-01 : f64) : f64
        // CHECK: llvm.call @__quantum__qis__u3__body(%[[q_0:.*]], %[[c_0]], %[[c_1]], %[[c_2]]) : (!llvm.ptr, f64, f64, f64) -> ()
        // CHECK: llvm.call @__quantum__qis__u2__body(%[[q_0:.*]], %[[c_0]], %[[c_1]]) : (!llvm.ptr, f64, f64) -> ()
        // CHECK: llvm.call @__quantum__qis__p__body(%[[q_0]], %[[c_0]]) : (!llvm.ptr, f64) -> ()
        // CHECK: llvm.call @__quantum__qis__rx__body(%[[q_0]], %[[c_0]]) : (!llvm.ptr, f64) -> ()
        // CHECK: llvm.call @__quantum__qis__ry__body(%[[q_0]], %[[c_0]]) : (!llvm.ptr, f64) -> ()
        // CHECK: llvm.call @__quantum__qis__rz__body(%[[q_0]], %[[c_0]]) : (!llvm.ptr, f64) -> ()

        %c0 = arith.constant 3.000000e-01 : f64
        %c1 = arith.constant 1.000000e-01 : f64
        %c2 = arith.constant 2.000000e-01 : f64
        %q0 = mqtref.allocQubit
        mqtref.u(%c0, %c1, %c2) %q0
        mqtref.u2(%c0, %c1) %q0
        mqtref.p(%c0) %q0
        mqtref.rx(%c0) %q0
        mqtref.ry(%c0) %q0
        mqtref.rz(%c0) %q0
        return
    }
}

// -----
// This test checks if the single qubit rotation gates using static qubits are correctly converted
module {
    // CHECK-LABEL: llvm.func @testSingleQubitRotationOpStatic()
    func.func @testSingleQubitRotationOpStatic() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[c_0:.*]] = llvm.mlir.constant(3.000000e-01 : f64) : f64
        // CHECK: %[[c_1:.*]] = llvm.mlir.constant(1.000000e-01 : f64) : f64
        // CHECK: %[[c_2:.*]] = llvm.mlir.constant(2.000000e-01 : f64) : f64
        // CHECK: llvm.call @__quantum__qis__u3__body(%[[q_0:.*]], %[[c_0]], %[[c_1]], %[[c_2]]) : (!llvm.ptr, f64, f64, f64) -> ()
        // CHECK: llvm.call @__quantum__qis__u2__body(%[[q_0:.*]], %[[c_0]], %[[c_1]]) : (!llvm.ptr, f64, f64) -> ()
        // CHECK: llvm.call @__quantum__qis__p__body(%[[q_0]], %[[c_0]]) : (!llvm.ptr, f64) -> ()
        // CHECK: llvm.call @__quantum__qis__rx__body(%[[q_0]], %[[c_0]]) : (!llvm.ptr, f64) -> ()
        // CHECK: llvm.call @__quantum__qis__ry__body(%[[q_0]], %[[c_0]]) : (!llvm.ptr, f64) -> ()
        // CHECK: llvm.call @__quantum__qis__rz__body(%[[q_0]], %[[c_0]]) : (!llvm.ptr, f64) -> ()

        %c0 = arith.constant 3.000000e-01 : f64
        %c1 = arith.constant 1.000000e-01 : f64
        %c2 = arith.constant 2.000000e-01 : f64
        %q0 = mqtref.qubit 0
        mqtref.u(%c0, %c1, %c2) %q0
        mqtref.u2(%c0, %c1) %q0
        mqtref.p(%c0) %q0
        mqtref.rx(%c0) %q0
        mqtref.ry(%c0) %q0
        mqtref.rz(%c0) %q0
        return
    }
}
// -----
// This test checks if the multiple qubit rotation gates are correctly converted
module {
    // CHECK-LABEL: llvm.func @testMultipleQubitRotationOp()
    func.func @testMultipleQubitRotationOp() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[c_0:.*]] = llvm.mlir.constant(3.000000e-01 : f64) : f64
        // CHECK: %[[c_1:.*]] = llvm.mlir.constant(1.000000e-01 : f64) : f64
        // CHECK: llvm.call @__quantum__qis__rxx__body(%[[q_0:.*]], %[[q_1:.*]], %[[c_0]]) : (!llvm.ptr, !llvm.ptr, f64) -> ()
        // CHECK: llvm.call @__quantum__qis__ryy__body(%[[q_0]], %[[q_1]], %[[c_0]]) : (!llvm.ptr, !llvm.ptr, f64) -> ()
        // CHECK: llvm.call @__quantum__qis__rzz__body(%[[q_0]], %[[q_1]], %[[c_0]]) : (!llvm.ptr, !llvm.ptr, f64) -> ()
        // CHECK: llvm.call @__quantum__qis__rzx__body(%[[q_0]], %[[q_1]], %[[c_0]]) : (!llvm.ptr, !llvm.ptr, f64) -> ()
        // CHECK: llvm.call @__quantum__qis__xx_minus_yy__body(%[[q_0]], %[[q_1]], %[[c_0]], %[[c_1]]) : (!llvm.ptr, !llvm.ptr, f64, f64) -> ()
        // CHECK: llvm.call @__quantum__qis__xx_plus_yy__body(%[[q_0]], %[[q_1]], %[[c_0]], %[[c_1]]) : (!llvm.ptr, !llvm.ptr, f64, f64) -> ()

        %c0 = arith.constant 3.000000e-01 : f64
        %c1 = arith.constant 1.000000e-01 : f64
        %q0 = mqtref.allocQubit
        %q1 = mqtref.allocQubit
        mqtref.rxx(%c0) %q0, %q1
        mqtref.ryy(%c0) %q0, %q1
        mqtref.rzz(%c0) %q0, %q1
        mqtref.rzx(%c0) %q0, %q1
        mqtref.xx_minus_yy(%c0, %c1) %q0, %q1
        mqtref.xx_plus_yy(%c0, %c1) %q0, %q1
        mqtref.deallocQubit %q0
        mqtref.deallocQubit %q1
        return
    }
}

// -----
// This test checks if the multiple qubit rotation gates using static qubits are correctly converted
module {
    // CHECK-LABEL: llvm.func @testMultipleQubitRotationOpStatic()
    func.func @testMultipleQubitRotationOpStatic() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[c_0:.*]] = llvm.mlir.constant(3.000000e-01 : f64) : f64
        // CHECK: %[[c_1:.*]] = llvm.mlir.constant(1.000000e-01 : f64) : f64
        // CHECK: llvm.call @__quantum__qis__rxx__body(%[[q_0:.*]], %[[q_1:.*]], %[[c_0]]) : (!llvm.ptr, !llvm.ptr, f64) -> ()
        // CHECK: llvm.call @__quantum__qis__ryy__body(%[[q_0]], %[[q_1]], %[[c_0]]) : (!llvm.ptr, !llvm.ptr, f64) -> ()
        // CHECK: llvm.call @__quantum__qis__rzz__body(%[[q_0]], %[[q_1]], %[[c_0]]) : (!llvm.ptr, !llvm.ptr, f64) -> ()
        // CHECK: llvm.call @__quantum__qis__rzx__body(%[[q_0]], %[[q_1]], %[[c_0]]) : (!llvm.ptr, !llvm.ptr, f64) -> ()
        // CHECK: llvm.call @__quantum__qis__xx_minus_yy__body(%[[q_0]], %[[q_1]], %[[c_0]], %[[c_1]]) : (!llvm.ptr, !llvm.ptr, f64, f64) -> ()
        // CHECK: llvm.call @__quantum__qis__xx_plus_yy__body(%[[q_0]], %[[q_1]], %[[c_0]], %[[c_1]]) : (!llvm.ptr, !llvm.ptr, f64, f64) -> ()

        %c0 = arith.constant 3.000000e-01 : f64
        %c1 = arith.constant 1.000000e-01 : f64
        %q0 = mqtref.qubit 0
        %q1 = mqtref.qubit 1
        mqtref.rxx(%c0) %q0, %q1
        mqtref.ryy(%c0) %q0, %q1
        mqtref.rzz(%c0) %q0, %q1
        mqtref.rzx(%c0) %q0, %q1
        mqtref.xx_minus_yy(%c0, %c1) %q0, %q1
        mqtref.xx_plus_yy(%c0, %c1) %q0, %q1
        return
    }
}
// -----
// This test checks if controlled gates are correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertControlledOp()
    func.func @testConvertControlledOp() attributes {passthrough = ["entry_point"]}  {
        // CHECK: llvm.call @__quantum__qis__cx__body(%[[q_1:.*]], %[[q_0:.*]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__ccx__body(%[[q_1]], %[[q_2:.*]], %[[q_0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__x__body(%[[q_1]]) : (!llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__cz__body(%[[q_1]], %[[q_0]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__x__body(%[[q_1]]) : (!llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__x__body(%[[q_1]]) : (!llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__ccx__body(%[[q_2]], %[[q_1]], %[[q_0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__x__body(%[[q_1]]) : (!llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__crx__body(%[[q_1]], %[[q_0]], %[[c_0:.*]]) : (!llvm.ptr, !llvm.ptr, f64) -> ()

        %c0 = arith.constant 3.000000e-01 : f64
        %q0 = mqtref.allocQubit
        %q1 = mqtref.allocQubit
        %q2 = mqtref.allocQubit
        mqtref.x() %q0 ctrl %q1
        mqtref.x() %q0 ctrl %q1, %q2
        mqtref.z() %q0 nctrl %q1
        mqtref.x() %q0 ctrl %q2 nctrl %q1
        mqtref.rx(%c0) %q0 ctrl %q1
        mqtref.deallocQubit %q0
        mqtref.deallocQubit %q1
        mqtref.deallocQubit %q2
        return
    }
}

// -----
// This test checks if controlled gates using static qubits are correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertControlledOpStatic()
    func.func @testConvertControlledOpStatic() attributes {passthrough = ["entry_point"]}  {
        // CHECK: llvm.call @__quantum__qis__cx__body(%[[q_1:.*]], %[[q_0:.*]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__ccx__body(%[[q_1]], %[[q_2:.*]], %[[q_0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__x__body(%[[q_1]]) : (!llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__cz__body(%[[q_1]], %[[q_0]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__x__body(%[[q_1]]) : (!llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__x__body(%[[q_1]]) : (!llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__ccx__body(%[[q_2]], %[[q_1]], %[[q_0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__x__body(%[[q_1]]) : (!llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__crx__body(%[[q_1]], %[[q_0]], %[[c_0:.*]]) : (!llvm.ptr, !llvm.ptr, f64) -> ()

        %c0 = arith.constant 3.000000e-01 : f64
        %q0 = mqtref.qubit 0
        %q1 = mqtref.qubit 1
        %q2 = mqtref.qubit 2
        mqtref.x() %q0 ctrl %q1
        mqtref.x() %q0 ctrl %q1, %q2
        mqtref.z() %q0 nctrl %q1
        mqtref.x() %q0 ctrl %q2 nctrl %q1
        mqtref.rx(%c0) %q0 ctrl %q1
        return
    }
}
// -----
// This test checks if static params are converted correctly
module {
    // CHECK-LABEL: llvm.func @testConvertStaticParams()
    func.func @testConvertStaticParams() attributes {passthrough = ["entry_point"]} {
        // CHECK-DAG: %[[c_0:.*]] = llvm.mlir.constant(1.000000e-01 : f64) : f64
        // CHECK-DAG: %[[c_1:.*]] = llvm.mlir.constant(2.000000e-01 : f64) : f64
        // CHECK-DAG: %[[c_2:.*]] = llvm.mlir.constant(3.000000e-01 : f64) : f64
        // CHECK-DAG: %[[c_3:.*]] = llvm.mlir.constant(4.000000e-01 : f64) : f64
        // CHECK-DAG: %[[c_4:.*]] = llvm.mlir.constant(5.000000e-01 : f64) : f64
        // CHECK-DAG: %[[c_5:.*]] = llvm.mlir.constant(6.000000e-01 : f64) : f64
        // CHECK: llvm.call @__quantum__qis__u3__body(%[[q_0:.*]], %[[c_0]], %[[c_1]], %[[c_2]]) : (!llvm.ptr, f64, f64, f64) -> ()
        // CHECK: llvm.call @__quantum__qis__u3__body(%[[q_0:.*]], %[[c_3]], %[[c_4]], %[[c_5]]) : (!llvm.ptr, f64, f64, f64) -> ()

        %q0 = mqtref.allocQubit
        mqtref.u(static [1.00000e-01, 2.00000e-01, 3.00000e-01] mask [true, true, true]) %q0
        mqtref.u(static [4.00000e-01, 5.00000e-01, 6.00000e-01]) %q0
        mqtref.deallocQubit %q0
        return
    }
}

// -----
// This test checks if static params using static qubits are converted correctly
module {
    // CHECK-LABEL: llvm.func @testConvertStaticParamsStatic()
    func.func @testConvertStaticParamsStatic() attributes {passthrough = ["entry_point"]} {
        // CHECK-DAG: %[[c_0:.*]] = llvm.mlir.constant(1.000000e-01 : f64) : f64
        // CHECK-DAG: %[[c_1:.*]] = llvm.mlir.constant(2.000000e-01 : f64) : f64
        // CHECK-DAG: %[[c_2:.*]] = llvm.mlir.constant(3.000000e-01 : f64) : f64
        // CHECK-DAG: %[[c_3:.*]] = llvm.mlir.constant(4.000000e-01 : f64) : f64
        // CHECK-DAG: %[[c_4:.*]] = llvm.mlir.constant(5.000000e-01 : f64) : f64
        // CHECK-DAG: %[[c_5:.*]] = llvm.mlir.constant(6.000000e-01 : f64) : f64
        // CHECK: llvm.call @__quantum__qis__u3__body(%[[q_0:.*]], %[[c_0]], %[[c_1]], %[[c_2]]) : (!llvm.ptr, f64, f64, f64) -> ()
        // CHECK: llvm.call @__quantum__qis__u3__body(%[[q_0:.*]], %[[c_3]], %[[c_4]], %[[c_5]]) : (!llvm.ptr, f64, f64, f64) -> ()

        %q0 = mqtref.qubit 0
        mqtref.u(static [1.00000e-01, 2.00000e-01, 3.00000e-01] mask [true, true, true]) %q0
        mqtref.u(static [4.00000e-01, 5.00000e-01, 6.00000e-01]) %q0
        return
    }
}

// -----
// This test checks if mixed static params are converted correctly
module {
    // CHECK-LABEL: llvm.func @testConvertMixedParams()
    func.func @testConvertMixedParams() attributes {passthrough = ["entry_point"]} {
        // CHECK-DAG: %[[c_0:.*]] = llvm.mlir.constant(1.000000e-01 : f64) : f64
        // CHECK-DAG: %[[c_1:.*]] = llvm.mlir.constant(2.000000e-01 : f64) : f64
        // CHECK-DAG: %[[c_2:.*]] = llvm.mlir.constant(3.000000e-01 : f64) : f64
        // CHECK-DAG: %[[c_3:.*]] = llvm.mlir.constant(4.000000e-01 : f64) : f64
        // CHECK-DAG: %[[c_4:.*]] = llvm.mlir.constant(5.000000e-01 : f64) : f64
        // CHECK-DAG: %[[c_5:.*]] = llvm.mlir.constant(6.000000e-01 : f64) : f64
        // CHECK: llvm.call @__quantum__qis__u3__body(%[[q_0:.*]], %[[c_0]], %[[c_1]], %[[c_2]]) : (!llvm.ptr, f64, f64, f64) -> ()
        // CHECK: llvm.call @__quantum__qis__u3__body(%[[q_0:.*]], %[[c_3]], %[[c_4]], %[[c_5]]) : (!llvm.ptr, f64, f64, f64) -> ()

        %q0 = mqtref.allocQubit
        %c0 = arith.constant 1.000000e-01 : f64
        %c1 = arith.constant 3.000000e-01 : f64
        %c2 = arith.constant 4.000000e-01 : f64
        mqtref.u(%c0, %c1 static [2.000000e-01] mask [false, true, false]) %q0
        mqtref.u(%c2 static [5.00000e-01, 6.00000e-01] mask [false, true, true]) %q0
        mqtref.deallocQubit %q0
        return
    }
}

// -----
// This test checks if mixed static params using static qubits are converted correctly
module {
    // CHECK-LABEL: llvm.func @testConvertMixedParams()
    func.func @testConvertMixedParams() attributes {passthrough = ["entry_point"]} {
        // CHECK-DAG: %[[c_0:.*]] = llvm.mlir.constant(1.000000e-01 : f64) : f64
        // CHECK-DAG: %[[c_1:.*]] = llvm.mlir.constant(2.000000e-01 : f64) : f64
        // CHECK-DAG: %[[c_2:.*]] = llvm.mlir.constant(3.000000e-01 : f64) : f64
        // CHECK-DAG: %[[c_3:.*]] = llvm.mlir.constant(4.000000e-01 : f64) : f64
        // CHECK-DAG: %[[c_4:.*]] = llvm.mlir.constant(5.000000e-01 : f64) : f64
        // CHECK-DAG: %[[c_5:.*]] = llvm.mlir.constant(6.000000e-01 : f64) : f64
        // CHECK: llvm.call @__quantum__qis__u3__body(%[[q_0:.*]], %[[c_0]], %[[c_1]], %[[c_2]]) : (!llvm.ptr, f64, f64, f64) -> ()
        // CHECK: llvm.call @__quantum__qis__u3__body(%[[q_0:.*]], %[[c_3]], %[[c_4]], %[[c_5]]) : (!llvm.ptr, f64, f64, f64) -> ()

        %q0 = mqtref.qubit 0
        %c0 = arith.constant 1.000000e-01 : f64
        %c1 = arith.constant 3.000000e-01 : f64
        %c2 = arith.constant 4.000000e-01 : f64
        mqtref.u(%c0, %c1 static [2.000000e-01] mask [false, true, false]) %q0
        mqtref.u(%c2 static [5.00000e-01, 6.00000e-01] mask [false, true, true]) %q0
        return
    }
}

// -----
// This test checks if the gphase operation is correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertGPhaseOp()
    func.func @testConvertGPhaseOp() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[c_0:.*]] = llvm.mlir.constant(3.000000e-01 : f64) : f64
        // CHECK: llvm.call @__quantum__qis__gphase__body(%[[c_0]]) : (f64) -> ()

        %cst = arith.constant 3.000000e-01 : f64
        mqtref.gphase(%cst)
        return
    }
}
// -----
// This test checks if the controlled gphase operation is correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertGPhaseOpControlled()
    func.func @testConvertGPhaseOpControlled() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[c_0:.*]] = llvm.mlir.constant(3.000000e-01 : f64) : f64
        // CHECK: llvm.call @__quantum__qis__cgphase__body(%[[ANY:.*]], %[[c_0]]) : (!llvm.ptr, f64) -> ()

        %cst = arith.constant 3.000000e-01 : f64
        %q0 = mqtref.allocQubit
        mqtref.gphase(%cst) ctrl %q0
        mqtref.deallocQubit %q0
        return
    }
}

// -----
// This test checks if the controlled gphase operation using static qubits is correctly converted
module {
    // CHECK-LABEL: llvm.func @testConvertGPhaseOpControlled()
    func.func @testConvertGPhaseOpControlled() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[c_0:.*]] = llvm.mlir.constant(3.000000e-01 : f64) : f64
        // CHECK: llvm.call @__quantum__qis__cgphase__body(%[[ANY:.*]], %[[c_0]]) : (!llvm.ptr, f64) -> ()

        %cst = arith.constant 3.000000e-01 : f64
        %q0 = mqtref.qubit 0
        mqtref.gphase(%cst) ctrl %q0
        return
    }
}
// -----
// This test checks if the barrierOp is converted correctly
module {
    // CHECK-LABEL: llvm.func @testConvertBarrierOp()
    func.func @testConvertBarrierOp() attributes {passthrough = ["entry_point"]}  {
        // CHECK: llvm.call @__quantum__qis__barrier__body(%[[ANY:.*]]) : (!llvm.ptr) -> ()

        %q0 = mqtref.allocQubit
        mqtref.barrier() %q0
        mqtref.deallocQubit %q0
        return
    }
}

// -----
// This test checks if the barrierOp using static qubits is converted correctly
module {
    // CHECK-LABEL: llvm.func @testConvertBarrierOpStatic()
    func.func @testConvertBarrierOpStatic() attributes {passthrough = ["entry_point"]}  {
        // CHECK: llvm.call @__quantum__qis__barrier__body(%[[ANY:.*]]) : (!llvm.ptr) -> ()

        %q0 = mqtref.qubit 0
        mqtref.barrier() %q0
        return
    }
}

// -----
// This test checks if the operations are moved correctly to the correct blocks during the conversion
module {
    // CHECK: llvm.mlir.global internal constant @mlir.llvm.nameless_global_0("r0\00") {addr_space = 0 : i32, dso_local}

    // CHECK-LABEL: llvm.func @testOperationMovement()
    func.func @testOperationMovement() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[a_0:.*]] = llvm.mlir.addressof @mlir.llvm.nameless_global_0 : !llvm.ptr
        // CHECK: %[[ptr_0:.*]] = llvm.mlir.zero : !llvm.ptr
        // CHECK: llvm.call @__quantum__rt__initialize(%[[ptr_0]]) : (!llvm.ptr) -> ()
        // CHECK: llvm.br ^[[main:.*]]
        // CHECK: ^[[main]]:
        // CHECK: %[[q_0:.*]] = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
        // CHECK: %[[q_1:.*]] = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
        // CHECK: llvm.call @__quantum__qis__h__body(%[[q_0]]) : (!llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__h__body(%[[q_1]]) : (!llvm.ptr) -> ()
        // CHECK: llvm.br ^[[main2:.*]]
        // CHECK: ^[[main2]]:
        // CHECK: llvm.call @__quantum__qis__mz__body(%[[q_0]], %[[ptr_0]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__rt__qubit_release(%[[q_0]]) : (!llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__reset__body(%[[q_1]]) : (!llvm.ptr) -> ()
        // CHECK: llvm.br ^[[end:.*]]
        // CHECK: ^[[end]]:
        // CHECK: llvm.call @__quantum__rt__result_record_output(%[[ptr_0]], %[[a_0]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.return

        %q0 = mqtref.allocQubit
        %q1 = mqtref.allocQubit
        %mem = memref.alloca() : memref<1xi1>
        mqtref.h() %q0
        %m0 = mqtref.measure %q0
        %c0 = arith.constant 0 : index
        memref.store %m0, %mem[%c0] : memref<1xi1>
        mqtref.deallocQubit %q0
        mqtref.h() %q1
        mqtref.reset %q1
        mqtref.deallocQubit %q1
        return
    }
}

// -----
// This test checks if a Bell state is converted correctly.
module {
    // CHECK: llvm.mlir.global internal constant @mlir.llvm.nameless_global_1("r1\00") {addr_space = 0 : i32, dso_local}
    // CHECK: llvm.mlir.global internal constant @mlir.llvm.nameless_global_0("r0\00") {addr_space = 0 : i32, dso_local}

    // CHECK-LABEL: llvm.func @bellState() attributes {passthrough = ["entry_point", ["output_labeling_schema", "schema_id"], ["qir_profiles", "base_profile"], ["required_num_qubits", "2"], ["required_num_results", "2"], ["qir_major_version", "1"], ["qir_minor_version", "0"], ["dynamic_qubit_management", "true"], ["dynamic_result_management", "false"]]}
    func.func @bellState() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[a_1:.*]] = llvm.mlir.addressof @mlir.llvm.nameless_global_1 : !llvm.ptr
        // CHECK: %[[a_0:.*]] = llvm.mlir.addressof @mlir.llvm.nameless_global_0 : !llvm.ptr
        // CHECK: %[[ptr_0:.*]] = llvm.mlir.zero : !llvm.ptr
        // CHECK: llvm.call @__quantum__rt__initialize(%[[ptr_0]]) : (!llvm.ptr) -> ()
        // CHECK: llvm.br ^[[main:.*]]
        // CHECK: ^[[main]]:
        // CHECK: %[[q_0:.*]] = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
        // CHECK: %[[q_1:.*]] = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
        // CHECK: llvm.call @__quantum__qis__h__body(%[[q_0]]) : (!llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__cx__body(%[[q_0]], %[[q_1]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.br ^[[main2:.*]]
        // CHECK: ^[[main2]]:
        // CHECK: llvm.call @__quantum__qis__mz__body(%[[q_0]], %[[ptr_0]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: %[[c_0:.*]] = llvm.mlir.constant(1 : i64) : i64
        // CHECK: %[[ptr_1:.*]] = llvm.inttoptr %[[c_0]] : i64 to !llvm.ptr
        // CHECK: llvm.call @__quantum__qis__mz__body(%[[q_1]], %[[ptr_1]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__rt__qubit_release(%[[q_0]]) : (!llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__rt__qubit_release(%[[q_1]]) : (!llvm.ptr) -> ()
        // CHECK: llvm.br ^[[end:.*]]
        // CHECK: ^[[end]]:
        // CHECK: llvm.call @__quantum__rt__result_record_output(%[[ptr_1]], %[[a_1]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__rt__result_record_output(%[[ptr_0]], %[[a_0]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.return

        %q0 = mqtref.allocQubit
        %q1 = mqtref.allocQubit
        %mem = memref.alloca() : memref<2xi1>
        mqtref.h() %q0
        mqtref.x() %q1 ctrl %q0
        %m0 = mqtref.measure %q0
        %c0 = arith.constant 0 : index
        memref.store %m0, %mem[%c0] : memref<2xi1>
        %m1 = mqtref.measure %q1
        %c1 = arith.constant 1 : index
        memref.store %m1, %mem[%c1] : memref<2xi1>
        mqtref.deallocQubit %q0
        mqtref.deallocQubit %q1
        return
    }
}

// -----
// This test checks if a Bell state using static qubits is converted correctly.
module {
    // CHECK: llvm.mlir.global internal constant @mlir.llvm.nameless_global_1("r1\00") {addr_space = 0 : i32, dso_local}
    // CHECK: llvm.mlir.global internal constant @mlir.llvm.nameless_global_0("r0\00") {addr_space = 0 : i32, dso_local}

    // CHECK-LABEL: llvm.func @bellStateStaticQubit() attributes {passthrough = ["entry_point", ["output_labeling_schema", "schema_id"], ["qir_profiles", "base_profile"], ["required_num_qubits", "2"], ["required_num_results", "2"], ["qir_major_version", "1"], ["qir_minor_version", "0"], ["dynamic_qubit_management", "false"], ["dynamic_result_management", "false"]]}
    func.func @bellStateStaticQubit() attributes {passthrough = ["entry_point"]}  {
        // CHECK: %[[a_1:.*]] = llvm.mlir.addressof @mlir.llvm.nameless_global_1 : !llvm.ptr
        // CHECK: %[[a_0:.*]] = llvm.mlir.addressof @mlir.llvm.nameless_global_0 : !llvm.ptr
        // CHECK: %[[ptr_0:.*]] = llvm.mlir.zero : !llvm.ptr
        // CHECK: llvm.call @__quantum__rt__initialize(%[[ptr_0]]) : (!llvm.ptr) -> ()
        // CHECK: llvm.br ^[[main:.*]]
        // CHECK: ^[[main]]:
        // CHECK: %[[index_0:.*]] = llvm.mlir.constant(1 : i64) : i64
        // CHECK: %[[ptr_1:.*]] = llvm.inttoptr %[[index_0]] : i64 to !llvm.ptr
        // CHECK: llvm.call @__quantum__qis__h__body(%[[ptr_0]]) : (!llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__cx__body(%[[ptr_0]], %[[ptr_1]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.br ^[[main2:.*]]
        // CHECK: ^[[main2]]:
        // CHECK: llvm.call @__quantum__qis__mz__body(%[[ptr_0]], %[[ptr_0]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__qis__mz__body(%[[ptr_1]], %[[ptr_1]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.br ^[[end:.*]]
        // CHECK: ^[[end]]:
        // CHECK: llvm.call @__quantum__rt__result_record_output(%[[ptr_1]], %[[a_1]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.call @__quantum__rt__result_record_output(%[[ptr_0]], %[[a_0]]) : (!llvm.ptr, !llvm.ptr) -> ()
        // CHECK: llvm.return

        %q0 = mqtref.qubit 0
        %q1 = mqtref.qubit 1
        %mem = memref.alloca() : memref<2xi1>
        mqtref.h() %q0
        mqtref.x() %q1 ctrl %q0
        %m0 = mqtref.measure %q0
        %c0 = arith.constant 0 : index
        memref.store %m0, %mem[%c0] : memref<2xi1>
        %m1 = mqtref.measure %q1
        %c1 = arith.constant 1 : index
        memref.store %m1, %mem[%c1] : memref<2xi1>
        return
    }
}
