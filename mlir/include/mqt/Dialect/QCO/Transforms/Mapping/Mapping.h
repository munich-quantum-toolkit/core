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

#include "mqt/Dialect/QCO/Transforms/Passes.h"

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {

namespace qco {

/// Elide terminal SWAPs in prepared straight-line programs before placement.
///
/// Accepts owned scalar qubits and flat, static tensor unpack/pack chains.
/// Run after allocation cleanup; preserve roots until placement consumes the
/// circuit-wire permutation in `mqt.layout`.
std::unique_ptr<Pass> createElideTerminalSwapsPass();

} // namespace qco
} // namespace mlir
