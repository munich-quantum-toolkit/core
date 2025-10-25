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

namespace mlir {

class PatternRewriter;

} // namespace mlir

namespace mqt::ir::opt {

class UnitaryInterface;
class MeasureOp;

/**
 * @brief Moves a measurement before the given gate.
 * @param gate The UnitaryInterface gate to swap with the measurement.
 * @param measurement The MeasureOp measurement to swap with the gate.
 * @param rewriter The pattern rewriter to use for the swap operation.
 */
void swapGateWithMeasurement(UnitaryInterface gate, MeasureOp measurement,
                             mlir::PatternRewriter& rewriter);
} // namespace mqt::ir::opt
