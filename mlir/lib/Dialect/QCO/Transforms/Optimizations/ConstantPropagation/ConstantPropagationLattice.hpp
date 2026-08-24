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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"

#include <complex>
#include <cstdint>
#include <optional>
#include <utility>
#include <variant>

namespace mlir::mqt::qco {

using Complex = std::complex<double>;

using Matrix2x2 = std::array<std::array<Complex, 2>, 2>;
using Matrix4x4 = std::array<std::array<Complex, 4>, 4>;
using UnitaryMatrix = std::variant<Matrix2x2, Matrix4x4>;

struct QuantumState {
  bool isTop = false;
  unsigned int maxTrackedAmplitudes;
  SmallVector<Value> qubits;
  llvm::DenseMap<uint64_t, Complex> amplitudes;

  explicit QuantumState(const unsigned int maxTrackedAmplitudes)
      : maxTrackedAmplitudes(maxTrackedAmplitudes) {}

  /**
   * Create a new QuantumState that is initialized to |0>.
   *
   * @param maxTrackedAmplitudes The maximum number of amplitudes before
   * QuantumStates becomes top.
   * @param qubit The qubit value that the new quantum component should own.
   * @return The newly created QuantumState.
   */
  static QuantumState singletonZero(unsigned int maxTrackedAmplitudes,
                                    Value qubit);

  bool operator==(const QuantumState& other) const;

  /**
   * Check if the QuantumState contains a certain value.
   *
   * @param v The value to be checked.
   * @return True if QuantumState contains v.
   */
  [[nodiscard("QuantumState::contains called but ignored.")]]
  bool contains(Value v) const;

  /**
   * Checks what the index of value v in the QuantumState is.
   *
   * @param v The value to look for.
   * @return The index where v is in QuantumState.
   */
  [[nodiscard("QuantumState::indexOf called but ignored.")]]
  std::optional<unsigned> indexOf(Value v) const;

  /**
   * Check if a value is always zero.
   *
   * @param q The value to check for.
   * @return True if the value is always zero.
   */
  [[nodiscard("QuantumState::isAlwaysZero called but ignored.")]]
  bool isAlwaysZero(Value q) const;

  /**
   * Check if a value is always one.
   *
   * @param q The value to check for.
   * @return True if the value is always one.
   */
  [[nodiscard("QuantumState::isAlwaysOne called but ignored.")]]
  bool isAlwaysOne(Value q) const;

  /**
   * Put QuantumState to top.
   */
  void markTop();

  void forwardQubit(Value from, Value to);

  [[nodiscard("QuantumState::mergeQuantumStates called but ignored.")]]
  QuantumState mergeQuantumStates(QuantumState that);

  LogicalResult applyUnitary(ArrayRef<Value> inputs,
                             const UnitaryMatrix& matrix,
                             ArrayRef<Value> outputs);

  /// Returns successor states paired with the measured classical result.
  std::unordered_map<unsigned int, std::pair<QuantumState, double>>
  measure(Value inQubit, Value outQubit, MLIRContext* ctx) const;
};

struct HybridState {
  llvm::DenseMap<Value, Attribute> classicalValues;
  QuantumState quantumState;
  double probability = 1.0;

  bool operator==(const HybridState& other) const;

  std::optional<Attribute> getClassical(Value v) const;
  void setClassical(Value v, Attribute attr);
};

struct HybridStateSet {
  bool isTop = false;
  llvm::SmallVector<HybridState> states;

  bool operator==(const HybridStateSet& other) const;

  static HybridStateSet top();
  static HybridStateSet singletonInitial();

  void addState(HybridState state);
  void canonicalize();
  void join(const HybridStateSet& other);

  void enforceMaxStates(unsigned maxTrackedStates);

  bool isAlwaysZero(Value v) const;
  bool isAlwaysOne(Value v) const;
  std::optional<Attribute> getUniqueConstant(Value v) const;
};

/// Utility used by the pass analysis.
bool isZeroAttribute(Attribute attr);
bool isOneAttribute(Attribute attr);

} // namespace mlir::mqt::qco