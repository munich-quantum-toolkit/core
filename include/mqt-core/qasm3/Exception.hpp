/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "Statement.hpp"

#include <exception>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

namespace qasm3 {
class CompilerError final : public std::exception {
public:
  std::string message;
  std::shared_ptr<DebugInfo> debugInfo;
  mutable std::string cachedMessage;

  CompilerError(std::string msg, std::shared_ptr<DebugInfo> debug)
      : message(std::move(msg)), debugInfo(std::move(debug)) {}

  [[nodiscard]] std::string toString() const {
    std::stringstream ss{};
    ss << debugInfo->toString();

    auto parentDebugInfo = debugInfo->parent;
    while (parentDebugInfo != nullptr) {
      ss << "\n  (included from " << parentDebugInfo->toString() << ")";
      parentDebugInfo = parentDebugInfo->parent;
    }

    ss << ":\n" << message;

    return ss.str();
  }

  [[nodiscard]] const char* what() const noexcept override {
    cachedMessage = toString();
    return cachedMessage.c_str();
  }
};

class ConstEvalError final : public std::exception {
public:
  std::string message;
  mutable std::string cachedMessage;

  explicit ConstEvalError(std::string msg) : message(std::move(msg)) {}

  [[nodiscard]] std::string toString() const {
    return "Constant Evaluation: " + message;
  }

  [[nodiscard]] const char* what() const noexcept override {
    cachedMessage = toString();
    return cachedMessage.c_str();
  }
};

class TypeCheckError final : public std::exception {
public:
  std::string message;
  mutable std::string cachedMessage;

  explicit TypeCheckError(std::string msg) : message(std::move(msg)) {}

  [[nodiscard]] std::string toString() const {
    return "Type Check Error: " + message;
  }

  [[nodiscard]] const char* what() const noexcept override {
    cachedMessage = toString();
    return cachedMessage.c_str();
  }
};
} // namespace qasm3
