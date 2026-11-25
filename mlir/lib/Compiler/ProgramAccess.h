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

#include <utility>

namespace mlir {

/// Move module ownership between program types inside the compiler library.
///
/// Pipelines that verify their result use this instead of the program
/// factories, which verify the module again.
struct ProgramAccess {
  using Storage = Program::Storage;

  /// Take ownership of a program's context and module.
  [[nodiscard]] static Storage release(Program&& program) {
    return std::move(program).releaseStorage();
  }

  /// Adopt a verified, linear QCO module.
  [[nodiscard]] static QCOProgram makeQCO(Storage storage) {
    return QCOProgram(std::move(storage));
  }
};

} // namespace mlir
