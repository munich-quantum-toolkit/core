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

#endif // MLIR_DIALECT_QIR_TRANSFORMS_PASSES_TD
