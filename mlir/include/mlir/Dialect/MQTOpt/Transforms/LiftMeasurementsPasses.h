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

#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"

#include <mlir/IR/PatternMatch.h>

/**
 * Move a measurement operation to precede a specified unitary gate by swapping
 * their positions.
 *
 * @param gate The unitary gate to swap with the measurement.
 * @param measurement The measurement operation to move before the gate.
 */
namespace mqt::ir::opt {
/**
 * @brief Moves a measurement before the given gate.
 * @param gate The UnitaryInterface gate to swap with the measurement.
 * @param measurement The MeasureOp measurement to swap with the gate.
 * @param rewriter The pattern rewriter to use for the swap operation.
 */
void swapGateWithMeasurement(UnitaryInterface gate, MeasureOp measurement,
                             mlir::PatternRewriter& rewriter);
} // namespace mqt::ir::opt
