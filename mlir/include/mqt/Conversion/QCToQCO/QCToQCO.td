// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def QCToQCO : Pass<"qc-to-qco", "mlir::ModuleOp"> {
  let summary = "Convert QC dialect to QCO dialect.";

  let description = [{
    Convert QC references to linear QCO values and thread their current state
    through gates, functions and structured control flow. A program allocates
    qubits in the entry block of its `mqt.entry_point` function. Helpers can group existing
    qubits in local reference buffers. Nested regions can use qubits through
    supported SCF operations. Control flow with block successors is unsupported.

    Private helpers may borrow fixed-size rank-one qubit memrefs with identity
    layout and default memory space. These become qubit tensors. Each updated
    qubit or register argument becomes an additional result in argument order.
    Borrowed registers must stay complete across calls and region boundaries.
    A helper cannot consume a borrowed register or return it explicitly.

    Each loaded reference denotes a stable slot, including across helper calls.
    Local registers with constant-index accesses confined to one block can keep
    their qubits in scalar SSA form until deallocation. Other accesses extract
    and restore the qubit for each operation. A store may initialize a slot in
    a reference buffer or write the same reference back to its slot. Entry-block
    reference buffers become tensors of distinct physical qubits or, in
    [gate-set synthesis](target_compilation.md#gate-set-synthesis), program
    qubits. Static qubit references are hoisted and coalesced as needed.

    Indices must be in bounds and simultaneous operands must be disjoint.
    The pass diagnoses known violations without inserting runtime checks.
    See the [quantum resource rules](index.md#quantum-resource-rules).
  }];

  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::qco::QCODialect",
                           "mlir::qtensor::QTensorDialect"];
}
