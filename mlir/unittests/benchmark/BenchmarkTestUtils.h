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

#include <string>

namespace mqt::benchmark {

/// Replaces the characters that a test name cannot contain.
inline std::string testName(llvm::StringRef name) {
  auto sanitized = name.str();
  for (auto& character : sanitized) {
    if (character == '-') {
      character = '_';
    }
  }
  return sanitized;
}

} // namespace mqt::benchmark
