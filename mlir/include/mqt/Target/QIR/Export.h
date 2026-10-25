/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"

#include <memory>

namespace llvm {
class LLVMContext;
class Module;
class raw_ostream;
} // namespace llvm

namespace mlir {
class ModuleOp;

namespace qir {

/// Translate lowered QIR to verified LLVM IR with normalized module flags.
///
/// Preserves QIS callee names for converters that consume canonical QIR.
/// Text and bitcode export instead materialize the target's native QIS names.
[[nodiscard]] std::unique_ptr<llvm::Module>
translateQIRToLLVM(ModuleOp moduleOp, llvm::LLVMContext& context);

/// Consume lowered QIR and export verified LLVM IR with normalized module
/// flags.
///
/// The module is consumed even on failure. The stream is unchanged when
/// translation or verification fails.
LogicalResult translateQIRToLLVMIR(OwningOpRef<ModuleOp> moduleOp,
                                   llvm::raw_ostream& output);

/// Consume lowered QIR and export LLVM bitcode with normalized module flags.
///
/// The module is consumed even on failure. The stream is unchanged when
/// translation or verification fails.
LogicalResult translateQIRToBitcode(OwningOpRef<ModuleOp> moduleOp,
                                    llvm::raw_ostream& output);

/// Consume lowered QIR and write verified bitcode, reporting file errors.
///
/// The module is consumed even on failure.
LogicalResult writeQIRBitcode(OwningOpRef<ModuleOp> moduleOp,
                              StringRef filename);

// OwningOpRef implicitly accepts borrowed handles; reject accidental ownership.
LogicalResult translateQIRToLLVMIR(ModuleOp, llvm::raw_ostream&) = delete;
LogicalResult translateQIRToBitcode(ModuleOp, llvm::raw_ostream&) = delete;
LogicalResult writeQIRBitcode(ModuleOp, StringRef) = delete;

} // namespace qir
} // namespace mlir
