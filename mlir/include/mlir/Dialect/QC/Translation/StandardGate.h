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

#include <llvm/ADT/StringRef.h>
#include <mlir/Support/LogicalResult.h>

#include <cstddef>
#include <cstdint>

namespace mlir {
class Location;
class OpBuilder;
class ValueRange;
} // namespace mlir

namespace mlir::qc {

/** A QC standard-gate operation that frontends can emit directly. */
enum class StandardGate : uint8_t {
  GPhase,
  Id,
  X,
  Y,
  Z,
  H,
  S,
  Sdg,
  T,
  Tdg,
  SX,
  SXdg,
  P,
  RX,
  RY,
  RZ,
  R,
  U2,
  U3,
  BuiltinU,
  CU,
  SWAP,
  ISWAP,
  DCX,
  ECR,
  RCCX,
  RXX,
  RYY,
  RZX,
  RZZ,
  XXPlusYY,
  XXMinusYY,
};

struct StandardGateDescriptor {
  constexpr StandardGateDescriptor(const StandardGate gate,
                                   const llvm::StringRef operationSymbol,
                                   const size_t parameterCount,
                                   const size_t controlCount,
                                   const size_t targetCount)
      : gate(gate), operationSymbol(operationSymbol),
        parameterCount(parameterCount), controlCount(controlCount),
        targetCount(targetCount) {}

  StandardGate gate = StandardGate::GPhase;
  llvm::StringRef operationSymbol{};
  size_t parameterCount = 0;
  size_t controlCount = 0;
  size_t targetCount = 0;
};

/** Return the descriptor for a standard gate. */
[[nodiscard]] const StandardGateDescriptor&
getStandardGateDescriptor(StandardGate gate);

/** Return the descriptor for an operation symbol, or null if none matches. */
[[nodiscard]] const StandardGateDescriptor*
lookupStandardGateByOperationSymbol(llvm::StringRef symbol);

/** Emit one primitive QC standard gate without source-language phase rules. */
[[nodiscard]] LogicalResult emitStandardGate(OpBuilder& builder, Location loc,
                                             StandardGate gate,
                                             ValueRange parameters,
                                             ValueRange qubits);

} // namespace mlir::qc
