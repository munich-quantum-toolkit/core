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

#include <llvm/ADT/StringRef.h>
#include <mlir/Support/LLVM.h>

#include <string>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir {
class ModuleOp;

namespace qc {

/// Function-result attribute retaining an OpenQASM output identifier.
inline constexpr llvm::StringLiteral OPENQASM_OUTPUT_NAME_ATTR =
    "qc.openqasm.output_name";

/// Function-result attribute retaining an OpenQASM output type category.
inline constexpr llvm::StringLiteral OPENQASM_OUTPUT_KIND_ATTR =
    "qc.openqasm.output_kind";

/**
 * @brief Translate a QC module to portable OpenQASM.
 *
 * @details Translation is buffered. The output stream is unchanged when the
 * module contains an unsupported construct or cannot be translated.
 */
LogicalResult translateQCToOpenQASM3(ModuleOp moduleOp,
                                     llvm::raw_ostream& output);

/**
 * @brief Translate a QC module to an owned OpenQASM source string.
 */
FailureOr<std::string> translateQCToOpenQASM3(ModuleOp moduleOp);

} // namespace qc
} // namespace mlir
