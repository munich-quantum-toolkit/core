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

#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace mlir::qc {
class QCProgramBuilder;
} // namespace mlir::qc

namespace mqt::bench {
class BV;
class GHZ;
class Grover;
class ModularMultiplier;
class Multiplexer;
class QFT;
class QFTAdder;
class QPE;
class RepeatUntilSuccess;
class Shor;
class Teleportation;
class WState;
} // namespace mqt::bench

namespace mqt::bench {

using namespace mlir;

/// Emit one configured Bernstein--Vazirani benchmark.
SmallVector<Value> bv(qc::QCProgramBuilder& builder, const BV& benchmark);

/// Emit one configured GHZ benchmark.
SmallVector<Value> ghz(qc::QCProgramBuilder& builder, const GHZ& benchmark);

/// Emit one configured Grover benchmark.
SmallVector<Value> grover(qc::QCProgramBuilder& builder,
                          const Grover& benchmark);

/// Emit one configured modular multiplier benchmark.
SmallVector<Value> modularMultiplier(qc::QCProgramBuilder& builder,
                                     const ModularMultiplier& benchmark);

/// Emit one configured quantum multiplexer benchmark.
SmallVector<Value> multiplexer(qc::QCProgramBuilder& builder,
                               const Multiplexer& benchmark);

/// Emit one configured QFT benchmark.
SmallVector<Value> qft(qc::QCProgramBuilder& builder, const QFT& benchmark);

/// Emit one configured QFT adder benchmark.
SmallVector<Value> qftAdder(qc::QCProgramBuilder& builder,
                            const QFTAdder& benchmark);

/// Emit one configured QPE benchmark.
SmallVector<Value> qpe(qc::QCProgramBuilder& builder, const QPE& benchmark);

/// Emit the repeat-until-success benchmark.
SmallVector<Value> repeatUntilSuccess(qc::QCProgramBuilder& builder,
                                      const RepeatUntilSuccess& benchmark);

/// Emit structured semiclassical Shor order finding.
SmallVector<Value> shor(qc::QCProgramBuilder& builder, const Shor& benchmark);

/// Emit the quantum teleportation benchmark.
SmallVector<Value> teleportation(qc::QCProgramBuilder& builder,
                                 const Teleportation& benchmark);

/// Emit one configured W-state benchmark.
SmallVector<Value> wState(qc::QCProgramBuilder& builder,
                          const WState& benchmark);

} // namespace mqt::bench
