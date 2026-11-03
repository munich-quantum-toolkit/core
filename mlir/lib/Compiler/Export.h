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

#include "mqt/Compiler/Programs.h"

namespace mlir {
/// Prepare and export native QC using the selected output format.
[[nodiscard]] FailureOr<CompilerProgram>
exportQCProgram(QCProgram program, ProgramFormat output,
                const TargetEnvironment* environment,
                const CompilationOptions& options = {});

} // namespace mlir
