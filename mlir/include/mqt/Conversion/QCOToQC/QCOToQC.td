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
    Scalar function arguments become in-place references; each function must
    return its qubit arguments as a positional suffix. Other results remain
    explicit, including qubits created by a function.

    Before rewriting, prove that quantum values yielded by modifiers, branches,
    and loops correspond positionally to their region arguments. Reject wire
    permutations and changes to quantum loop-state counts or types. Classical
    loop state can change type in `scf.while`. QTensor insertions can update
    register slots and become stores when the slot value changes.

    Determine the module's allocation mode before converting any function.
    Dynamic qubit sinks become deallocations; static qubit sinks are removed.
    Mixing static and dynamic allocation roots is unsupported.
  }];

  let dependentDialects = ["mlir::memref::MemRefDialect",
                           "mlir::qc::QCDialect"];
}
