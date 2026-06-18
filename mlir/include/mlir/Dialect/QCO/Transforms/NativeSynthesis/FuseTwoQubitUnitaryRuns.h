/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// \file
/// Fuse maximal two-qubit unitary windows (with absorbed single-qubit padding).

#pragma once

#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Types.h"

#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

namespace mlir::qco::native_synth {

/// Scan `root` for maximal two-qubit windows (including absorbed single-qubit
/// ops on the same wire pair) and replace each window when Weyl/KAK
/// resynthesis to the native profile is profitable.
LogicalResult fuseTwoQubitUnitaryRuns(IRRewriter& rewriter, Operation* root,
                                      const NativeProfileSpec& spec);

} // namespace mlir::qco::native_synth
