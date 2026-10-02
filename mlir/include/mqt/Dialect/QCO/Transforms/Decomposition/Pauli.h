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

#include "mqt/Compiler/Target.h"
#include "mqt/Dialect/QCO/Utils/Matrix.h"

#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <variant>

namespace mlir {
class Operation;
class OpBuilder;
class Location;
class RewriterBase;
} // namespace mlir

namespace mlir::qco::decomposition {

enum class PauliAxis : uint8_t { I, X, Y, Z };

/// A constant Clifford C such that C Z C^dagger is the requested axis.
/// The identity axis returns the identity matrix.
[[nodiscard]] Matrix2x2 pauliFrame(PauliAxis axis);

/// Generator axes of a recognized one- or two-qubit Pauli rotation.
/// A one-qubit rotation has I as its second axis.
[[nodiscard]] std::array<PauliAxis, 2> pauliAxes(CompilerTarget::GateKind gate);

/// Emits RXX, RYY, RZX, or RZZ with a scalar or SSA angle.
[[nodiscard]] Operation* emitPauliRotation2Q(OpBuilder& builder, Location loc,
                                             Value qubit0, Value qubit1,
                                             CompilerTarget::GateKind gate,
                                             std::variant<double, Value> angle);

struct PauliRotation {
  std::array<PauliAxis, 2> axes;
  double angleScale = 1.;
};

/// Commuting rotations sharing one angle, with an explicit global phase.
/// Represents the existing Pauli rotations, P, and their one-control forms.
struct PauliRotationSequence {
  Value angle;
  SmallVector<PauliRotation, 3> rotations;
  double globalPhaseScale = 0.;

  [[nodiscard]] size_t numEntanglingRotations() const;
  [[nodiscard]] bool requiresCX(CompilerTarget::Entangler entangler) const;
  [[nodiscard]] std::optional<size_t>
  nativeEntanglerCount(CompilerTarget::Entangler entangler,
                       std::optional<size_t> cxCount = std::nullopt) const;
};

/// Recognizes exact Pauli-generator decompositions without changing the IR.
[[nodiscard]] std::optional<PauliRotationSequence>
getPauliRotations(Operation* operation);

struct TwoQubitNativeDecomposition;

/// Emits a recognized sequence directly in the target's synthesis basis.
/// For fixed entanglers, cx must implement CX in the selected operand order.
/// Hoists supporting scalar operations from a recognized control body; the
/// caller replaces the original operation with the returned qubits.
[[nodiscard]] SmallVector<Value, 2>
emitPauliRotations(RewriterBase& rewriter, Operation* operation,
                   const PauliRotationSequence& sequence,
                   const CompilerTarget::SynthesisBasis& basis, bool reverse,
                   const TwoQubitNativeDecomposition* cx = nullptr);

} // namespace mlir::qco::decomposition
