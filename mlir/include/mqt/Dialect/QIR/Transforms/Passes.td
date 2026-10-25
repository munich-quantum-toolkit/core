// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MLIR_DIALECT_QIR_TRANSFORMS_PASSES_TD
#define MLIR_DIALECT_QIR_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def QIRSetAttributesAndMetadata
    : Pass<"set-qir-attributes-and-metadata", "mlir::ModuleOp"> {
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
  let summary = "Sets the required attributes to the entry point function and "
                "adds the required module flags, compliant with QIR 2.1";
  let description = [{
    Static resource counts specify the capacity through the highest qubit ID
    used by a quantum instruction and the highest used result ID. Rejects
    IDs whose required capacity cannot fit in an unsigned 64-bit integer.
    Qubit capacity includes static IDs in compiler-marked qubit stores,
    including packed arguments and registers accessed with runtime indices.
    Arbitrary external pointer aliases are not inferred.
    Direct helper arguments and CFG forwarding contribute to resource counts.
    Capabilities cover all emitted function definitions. Loop exits with unknown
    measurement dependence require both backward-branch capabilities. Native
    pointer buffers conservatively require arrays, including static resources.
  }];
  let options = [Option<"useAdaptive", "use-adaptive", "bool",
                        /*default= */ "true", "Specifies the profile.">];
}

def QIRCleanupPass : Pass<"qir-cleanup", "mlir::ModuleOp"> {
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
  let summary = "Remove redundant QIR runtime bookkeeping.";
  let description = [{
    Removes redundant QIR runtime qubit-array allocation/release pairs that do
    not contribute to observable behavior, and keeps QIR modules compact.
  }];
}

def QIRDropGlobalPhase : Pass<"qir-drop-global-phase", "mlir::ModuleOp"> {
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
  let summary = "Drop finite constant global phases for sample-only output";
  let description = [{
    Removes direct `__quantum__qis__gphase__body` calls with a finite constant
    angle. Rejects other global-phase signatures and dynamic or non-finite
    angles. Controlled phases are relative phases and remain unchanged.
    Use only when the output contract disregards global phase, such as IQM JSON.
  }];
}

#endif // MLIR_DIALECT_QIR_TRANSFORMS_PASSES_TD
