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

#include <string>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir {
class ModuleOp;

namespace qc {

/// Consume a QC module and translate it to portable OpenQASM.
///
/// Non-gate helpers are inlined and array views normalized in the supplied
/// module. Gate definitions and supported structured control flow are
/// preserved. The module is consumed even on failure. Translation is buffered:
/// the output stream is unchanged if the module cannot be translated.
LogicalResult translateQCToOpenQASM3(OwningOpRef<ModuleOp> moduleOp,
                                     llvm::raw_ostream& output);

/// Consume a QC module and translate it to an owned OpenQASM source string.
FailureOr<std::string> translateQCToOpenQASM3(OwningOpRef<ModuleOp> moduleOp);

// OwningOpRef implicitly accepts borrowed handles; reject accidental ownership.
LogicalResult translateQCToOpenQASM3(ModuleOp, llvm::raw_ostream&) = delete;
FailureOr<std::string> translateQCToOpenQASM3(ModuleOp) = delete;

} // namespace qc
} // namespace mlir
