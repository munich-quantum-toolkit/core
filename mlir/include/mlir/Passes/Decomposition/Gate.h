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

#include "Helpers.h"
#include "ir/operations/OpType.hpp"

#include <llvm/ADT/SmallVector.h>

namespace mlir::qco::decomposition {

using QubitId = std::size_t;

/**
 * Gate description which should be able to represent every possible
 * one-qubit or two-qubit operation.
 */
struct Gate {
  qc::OpType type{qc::I};
  llvm::SmallVector<fp, 3> parameter;
  llvm::SmallVector<QubitId, 2> qubitId = {0};
};

} // namespace mlir::qco::decomposition
