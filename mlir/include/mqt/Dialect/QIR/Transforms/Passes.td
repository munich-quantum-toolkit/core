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
                "adds the required module flags, selecting the minimum "
                "required QIR version";
  let description = [{
    Base Profile input must use only the Base Profile instruction subset, and
    every recorded output bit must be measured; other input is rejected before
    any metadata is attached. Adaptive Profile input must already conform to
    the profile.

    The minimum version is QIR 2.0, or 2.1 for Adaptive programs with dynamic
    resources or runtime arrays. An attached QIR target may require a higher minimum version.
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
  let options = [Option<"profile", "profile", "::mlir::QIRProfile",
                        /*default=*/"::mlir::QIRProfile::Adaptive",
                        "QIR profile of the program", [{
    ::llvm::cl::values(
        clEnumValN(::mlir::QIRProfile::Base, "base", "QIR Base Profile"),
        clEnumValN(::mlir::QIRProfile::Adaptive, "adaptive",
                   "QIR Adaptive Profile"))
  }]>];
}

def QIRCleanupPass : Pass<"qir-cleanup", "mlir::ModuleOp"> {
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
  let summary = "Remove unused external LLVM function declarations.";
}

#endif // MLIR_DIALECT_QIR_TRANSFORMS_PASSES_TD
