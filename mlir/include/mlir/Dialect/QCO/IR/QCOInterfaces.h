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

#include <Eigen/Core>
#include <complex>
#include <cstddef>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <optional>
#include <type_traits>
#include <utility>

// clang-format:off
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h.inc" // IWYU pragma: export
// clang-format:on
