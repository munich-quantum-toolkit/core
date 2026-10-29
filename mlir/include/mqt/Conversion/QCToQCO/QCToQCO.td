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
    This pass converts QC qubit references to QCO qubit values and threads
    their state through quantum operations, functions and structured control
    flow.
    Control flow must use structured SCF operations. Operations with block
    successors, including `cf.br`, `cf.cond_br`, and `cf.switch`, are diagnosed
    before conversion.
    Quantum allocations must be in a function's entry block. Nested regions
    may use outer allocations through the supported SCF operations.
    Private helpers may borrow fixed-size rank-one qubit memrefs with identity
    layout and default memory space. They become qubit tensors. Updated scalar
    qubits and complete tensors are appended to ordinary results in argument
    order. Duplicate register operands, register/element aliases, and consumption
    or explicit returns of borrowed registers are diagnosed before conversion.
    Register slots retain their qubit identities. Loads refer to stable slots;
    identity stores are erased. Local fixed-size registers with constant-index
    accesses confined to one block can keep accessed slots in scalar SSA form
    until deallocation. Registers that escape or have other accesses use
    per-operation extraction and reinsertion.
    Entry-block physical reference buffers may be initialized once with
    distinct static qubits and become tensors of those qubits. Other reference
    stores are unsupported. Loaded references remain
    valid across helper calls. Bounds, disjoint runtime operands, and matching
    dynamic store indices are program preconditions; this pass diagnoses known
    violations without inserting runtime checks. See the shared
    [quantum resource contract](index.md#quantum-resource-contract).
    Static qubit references are hoisted and coalesced when needed. General
    classical folding and common subexpression elimination are not guaranteed
    by this conversion; use the corresponding simplification passes when needed.
  }];

  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::qco::QCODialect",
                           "mlir::qtensor::QTensorDialect"];
}
