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
    Convert QCO qubit values to QC references and QTensor registers to memrefs.
    Scalar qubits and fixed-size rank-one qubit-tensor arguments become in-place
    references; each function must return its quantum arguments as a positional
    suffix. Register helpers must be private. Registers must be complete when
    passed through calls or structured regions, returned, or dynamically freed.
    Extracted qubits may be passed to scalar helpers while their remaining
    tensor stays in the caller. Transferring the register itself is unsupported.
    Other results remain explicit, including qubits created by a function.

    Before rewriting, prove that quantum values yielded by modifiers, branches,
    and loops correspond positionally to their region arguments. Reject wire
    permutations and changes to quantum loop-state counts or types. Classical
    loop state can change type in `scf.while`. Extraction becomes `memref.load`;
    reinsertion preserves the same slot and needs no store. Known slot changes
    are rejected; equality of dynamic extraction and insertion indices is a
    program precondition. See the shared
    [quantum resource contract](index.md#quantum-resource-contract).

    Determine the module's allocation mode before converting any function.
    Dynamic qubit sinks become deallocations; static qubit sinks are removed.
    Mixing static and dynamic allocation roots is unsupported.
    Tensors of existing qubits must contain static qubits, assigned by placement,
    and use local reference buffers. Static tensor release does not release
    physical qubits. Dynamic tensors must restore every extracted slot before
    region exit or deallocation. Unsupported lifetimes fail before rewriting.
  }];

  let dependentDialects = ["mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect", "mlir::qc::QCDialect",
                           "mlir::arith::ArithDialect"];
}
