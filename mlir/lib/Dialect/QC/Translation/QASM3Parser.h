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

#include <mlir/IR/OwningOpRef.h>

#include <string_view>

namespace mlir {

// Forward declarations
class MLIRContext;
class ModuleOp;

namespace qc::detail {

/**
 * @brief Parse an OpenQASM 3 program and emit a QC program.
 *
 * @details
 * Implementation detail of `translateQASM3ToQC`; not part of the public API.
 * Prefer the `translateQASM3ToQC` entry points.
 *
 * @param source String containing the OpenQASM3 program.
 * @param context The MLIRContext to create the module in.
 * @return A module containing the QC program.
 */
[[nodiscard]] OwningOpRef<ModuleOp> parseQASM3(std::string_view source,
                                               MLIRContext* context);

} // namespace qc::detail

} // namespace mlir
