// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def QCOToQC : Pass<"qco-to-qc", "mlir::ModuleOp"> {
  let summary = "Convert QCO dialect to QC dialect.";

  let description = [{
    Convert linear QCO values to QC references and QTensor registers to memrefs.
    A function returns the current values of its quantum arguments in order,
    after its ordinary results. QC updates the arguments in place and retains
    the ordinary results. Private helpers may borrow fixed-size rank-one qubit
    tensors. They may return fresh scalar qubits. Register ownership stays
    with the caller.

    Quantum values yielded by modifiers, branches and loops must correspond to
    their region arguments in order. The pass retains classical region results
    and loop-carried state. Extracting a qubit becomes a load. Reinsertion needs
    no store when the pass can prove that it restores the original slot.
    Borrowed registers must be complete when passed through calls or regions
    and when returned. A scalar helper can operate on an extracted qubit while
    the remaining tensor stays in the caller.

    A module may use dynamic allocations or static physical qubits.
    A register created by `qtensor.alloc` must be complete at deallocation.
    A tensor assembled from existing qubits requires placement or a gate-set
    synthesis environment and becomes a local reference buffer. See the
    [quantum resource rules](index.md#quantum-resource-rules).
  }];

  let dependentDialects = ["mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect", "mlir::qc::QCDialect",
                           "mlir::arith::ArithDialect"];
}
