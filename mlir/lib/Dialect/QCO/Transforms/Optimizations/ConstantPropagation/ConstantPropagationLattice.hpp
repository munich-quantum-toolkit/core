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

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>

#include <complex>
#include <optional>
#include <utility>
#include <variant>

namespace mlir::qco {

using Complex = std::complex<double>;

using Matrix2x2 = std::array<std::array<Complex, 2>, 2>;
using Matrix4x4 = std::array<std::array<Complex, 4>, 4>;
using UnitaryMatrix = std::variant<Matrix2x2, Matrix4x4>;

/**
 * This struct represents a QuantumState. It contains of the amplitudes of
 * different qubit states. It is top if the number of non-zero amplitudes
 * exceeds a given maximum number of amplitudes.
 */
struct QuantumState {
private:
  bool isTop = false;
  unsigned int maxTrackedAmplitudes;
  SmallVector<Value> qubits;
  llvm::DenseMap<uint64_t, Complex> amplitudes;

  /**
   * Checks what the index of value v in the QuantumState is.
   *
   * @param v The value to look for.
   * @return The index where v is in QuantumState.
   */
  [[nodiscard("QuantumState::indexOf called but ignored.")]]
  std::optional<unsigned> indexOf(Value v) const;

  /**
   * Applies a 2x2 unitary matrix to a single qubit in the QuantumState.
   * This operation updates the quantum state's amplitude distribution
   * and the tracked qubit values.
   *
   * @param input The qubit identifier to which the matrix is applied.
   * @param output The updated qubit identifier after transformation.
   * @param matrix The 2x2 unitary matrix describing the transformation
   *               to be applied to the specified qubit.
   */
  void applyMatrix1Q(Value input, Value output, const Matrix2x2& matrix);

  /**
   * Applies a 4x4 unitary matrix to a single qubit in the QuantumState.
   * This operation updates the quantum state's amplitude distribution
   * and the tracked qubit values.
   *
   * @param input0 The first qubit identifier (corresponding to the lower index
   * of the matrix) to which the matrix is applied.
   * @param input1 The second qubit identifier (corresponding to the higher
   * index of the matrix) to which the matrix is applied.
   * @param output0 The updated first qubit identifier after transformation.
   * @param output1 The updated second qubit identifier after transformation.
   * @param matrix The 4x4 unitary matrix describing the transformation
   *               to be applied to the specified qubit.
   */
  void applyMatrix2Q(Value input0, Value input1, Value output0, Value output1,
                     const Matrix4x4& matrix);

public:
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
   * Computes the tensor product of this QuantumState with another QuantumState.
   * The tensor product combines the qubits and amplitudes of both states,
   * producing a new QuantumState that represents the combined quantum system.
   *
   * @param that The QuantumState to be combined with this QuantumState.
   * @return A new QuantumState representing the tensor product of the two
   * states.
   */
  [[nodiscard("QuantumState::tensorProduct called but ignored.")]]
  QuantumState tensorProduct(const QuantumState& that);

  [[nodiscard("QuantumState::isStateTop called but ignored.")]]
  bool isStateTop() const;

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

  /**
   * Changes qubit value one to another.
   *
   * @param from The original qubit value.
   * @param to The new qubit value.
   */
  void forwardQubit(Value from, Value to);

  /**
   * Applies a unitary matrix to the QuantumState.
   *
   * @param inputs The values that the matrix is applied to.
   * @param matrix The matrix that is applied to the QuantumState.
   * @param outputs The values that replace the input values after matrix
   * application.
   * @return Whether the application was successful or not.
   */
  LogicalResult applyUnitary(ArrayRef<Value> inputs,
                             const UnitaryMatrix& matrix,
                             ArrayRef<Value> outputs);

  /**
   * Simulates a quantum measurement on a given qubit and updates the quantum
   * state, producing possible successor states along with their classical
   * outcomes.
   *
   * @param inQubit The qubit to be measured.
   * @param outQubit The qubit value after the measurement.
   * @param ctx The MLIRContext used for type creation and attribute
   * propagation.
   * @return A map of possible successor states paired with their probability.
   * The keys are the measurement results.
   */
  std::unordered_map<unsigned int, std::pair<QuantumState, double>>
  measure(Value inQubit, Value outQubit, MLIRContext* ctx);
};

/**
 * This struct represents a HybridState. It contains a QuantumState and
 * classical values that are tracked alongside the QuantumState. It is top if
 * the QuantumState is top.
 */
struct HybridState {
private:
  llvm::DenseMap<Value, Attribute> classicalValues;
  std::unique_ptr<QuantumState> quantumState;
  unsigned int maxTrackedAmplitudes;
  double probability = 1.0;
  bool isTop = false;

public:
  explicit HybridState(const unsigned int maxTrackedAmplitudes,
                       const Value qubit)
      : quantumState(std::make_unique<QuantumState>(
            QuantumState::singletonZero(maxTrackedAmplitudes, qubit))),
        maxTrackedAmplitudes(maxTrackedAmplitudes) {}

  explicit HybridState(const unsigned int maxTrackedAmplitudes)
      : quantumState(nullptr), maxTrackedAmplitudes(maxTrackedAmplitudes) {}

  HybridState(const HybridState& that)
      : classicalValues(that.classicalValues),
        quantumState(that.quantumState
                         ? std::make_unique<QuantumState>(*that.quantumState)
                         : nullptr),
        maxTrackedAmplitudes(that.maxTrackedAmplitudes),
        probability(that.probability), isTop(that.isTop) {}

  HybridState& operator=(const HybridState& that) {
    if (this == &that) {
      return *this;
    }

    classicalValues = that.classicalValues;
    maxTrackedAmplitudes = that.maxTrackedAmplitudes;
    probability = that.probability;
    isTop = that.isTop;

    if (that.quantumState) {
      quantumState = std::make_unique<QuantumState>(*that.quantumState);
    } else {
      quantumState.reset();
    }

    return *this;
  }

  bool operator==(const HybridState& that) const;

  /**
   * Gets the attribute of a classical value if present.
   *
   * @param v The classical value to be checked.
   * @return The Attribute of the classical value.
   */
  [[nodiscard("HybridState::getClassical called but ignored.")]] std::optional<
      Attribute>
  getClassical(Value v) const;

  /**
   * Sets the attribute of a classical value. If the value already has an
   * attribute, it is overwritten.
   *
   * @param v The classical value to be set.
   * @param attr The attribute to be set.
   */
  void setClassical(Value v, Attribute attr);

  /**
   * Checks whether a HybridState contains a value. The value can be a quantum
   * or a classical one.
   *
   * @param v The value to check for.
   * @return Whether the value is in the HybridState.
   */
  bool contains(Value v) const;

  [[nodiscard("HybridState::isStateTop called but ignored.")]]
  bool isStateTop() const;

  /**
   * Checks if a value is always false, i.e., false/zero if it is a classical
   * value and |0> if it is a quantum value. If the value is not part of the
   * HybridState, the result is false.
   *
   * @param v The value to be checked.
   * @return Whether the value is always false.
   */
  [[nodiscard("HybridState::isAlwaysFalse called but ignored.")]]
  bool isAlwaysFalse(Value v) const;

  /**
   * Checks if a value is always false, i.e., true/nonzero if it is a classical
   * value and |1> if it is a quantum value. If the value is not part of the
   * HybridState, the result is false.
   *
   * @param v The value to be checked.
   * @return Whether the value is always true.
   */
  [[nodiscard("HybridState::isAlwaysTrue called but ignored.")]]
  bool isAlwaysTrue(Value v) const;

  // TODO: Application of various operations
  /**
   * Merges two HybridStates which have QuantumState with different qubits.
   *
   * @param that The HybridStateSet to be merged with this.
   * @return A new merged HybridState.
   */
  HybridState mergeStates(const HybridState& that) const;

  /**
   * Apply a classical state to the Hybrid state.
   *
   * @param op The classical operation to apply.
   */
  void applyClassicalOperation(Operation* op);
};

/**
 * A set of all HybridStates in the current pass. It becomes top if either all
 * HybridStates are top or if the number of HybridStates exceeds the maximum
 * number.
 */
struct HybridStateSet {
private:
  bool isTop = false;
  unsigned int maxTrackedAmplitudes;
  unsigned int maxTrackedHybridStates;
  SmallVector<HybridState> states;

public:
  explicit HybridStateSet(const unsigned int maxTrackedAmplitudes,
                          const unsigned int maxTrackedHybridStates)
      : maxTrackedAmplitudes(maxTrackedAmplitudes),
        maxTrackedHybridStates(maxTrackedHybridStates) {}

  bool operator==(const HybridStateSet& that) const;

  /**
   * Adds a hybridState to the set.
   *
   * @param state The HybridState to be added.
   */
  void addState(HybridState state);

  // TODO: Application of various operations

  /**
   * Joins HybridStateSets after branching. In that case, the new HybridStateSet
   * is the union of the states in both old sets. If either of the old sets is
   * top, the new state is top.
   *
   * @param other the HybridStateSet to join the current set with.
   */
  void join(const HybridStateSet& other);

  /**
   * Checks if there are too many HybridStates in the set. If the number of
   * states exceeds the specified maximum, the state set is marked as "top", and
   * all individual states tracked in the set are cleared.
   */
  void enforceMaxStates();

  /**
   * Merges two HybridStateSets which have QuantumState with different qubits.
   * Needs to be done before an operation entangles qubits from two
   * HybridStates.
   *
   * @param that The HybridStateSet to be merged with this.
   * @returns The new HybridStateSet.
   */
  HybridStateSet mergeStates(const HybridStateSet& that) const;

  [[nodiscard("HybridStateSet::areStatesTop called but ignored.")]]
  bool areStatesTop() const;

  /**
   * Checks if a value is always false, i.e., false/zero if it is a classical
   * value and |0> if it is a quantum value. If the value is not part of the
   * HybridState, the result is false.
   *
   * @param v The value to be checked.
   * @return Whether the value is always false.
   */
  [[nodiscard("HybridStateSet::isAlwaysFalse called but ignored.")]]
  bool isAlwaysFalse(Value v) const;

  /**
   * Checks if a value is always false, i.e., true/nonzero if it is a classical
   * value and |1> if it is a quantum value. If the value is not part of the
   * HybridState, the result is false.
   *
   * @param v The value to be checked.
   * @return Whether the value is always true.
   */
  [[nodiscard("HybridStateSet::isAlwaysTrue called but ignored.")]]
  bool isAlwaysTrue(Value v) const;

  /**
   * Applies a classical operation on all HybridStates of the set and returns a
   * new set.
   *
   * @param op The operation to apply.
   * @return The set with applied operation.
   */
  void applyClassicalOperation(Operation* op);
};

/// Utility used by the pass analysis.
bool isZeroAttribute(Attribute attr);
bool isTrueAttribute(Attribute attr);

} // namespace mlir::qco