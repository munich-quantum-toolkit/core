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

#include <stdexcept>
#include <string>
#include <utility>

namespace mlir::detail {

/// Recoverable failure reported by the third-party JEFF deserializer.
class JeffDeserializerError final : public std::runtime_error {
public:
  explicit JeffDeserializerError(std::string message)
      : std::runtime_error(std::move(message)) {}
};

} // namespace mlir::detail
