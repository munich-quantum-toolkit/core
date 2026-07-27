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

#include "mlir/Target/OpenQASM/Frontend.h"

#include <mlir/IR/OwningOpRef.h>

namespace mlir {
class MLIRContext;
class ModuleOp;
class Location;

namespace qc::detail {

[[nodiscard]] Location
getOpenQASMLocation(const oq3::frontend::SourceLocation& source,
                    MLIRContext& context);

[[nodiscard]] OwningOpRef<ModuleOp>
emitOpenQASMToQC(const oq3::frontend::TypedProgram& program,
                 MLIRContext& context);

} // namespace qc::detail
} // namespace mlir
