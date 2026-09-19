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

#include "dd/Error.hpp"
#include "mqt/Dialect/QCO/Utils/DDAdapter.h"

#include "llvm/Support/Error.h"

#include <optional>
#include <utility>

namespace dd::test {
template <typename T> T value(llvm::Expected<T> result) {
  return llvm::cantFail(std::move(result));
}
inline void value(llvm::Error error) { llvm::cantFail(std::move(error)); }
template <typename T> T value(dd::Result<T> result) {
  return value(mlir::qco::ddResult(std::move(result)));
}
inline void value(const std::optional<dd::Error>& error) {
  value(mlir::qco::ddResult(error));
}
} // namespace dd::test
