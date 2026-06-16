/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#ifndef MQT_CORE_HYBRIDSTATE_H
#define MQT_CORE_HYBRIDSTATE_H

#include "QuantumState.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mlir::qco {
/**
 * @brief This class represents a hybrid state.
 *
 * This class holds a QuantumState and a mapping form values to integers or
 * doubles, as well as a probability.
 */
class HybridState {
  bool top = false;
  std::shared_ptr<QuantumState> qState;
  double probability;
  llvm::DenseMap<Value, int64_t> integerValues;
  llvm::DenseMap<Value, double> doubleValues;

public:
  explicit HybridState(std::size_t nQubits, std::size_t maxNonzeroAmplitudes,
                       double probability = 1.0);

  ~HybridState();

  void print(std::ostream& os) const;

  [[nodiscard("HybridState::toString called but ignored")]]
  std::string toString() const;

  bool isHybridStateTop() const { return top; }

  /**
   * @brief This method adds a classical integer value to the hybrid state.
   *
   * @param value The value object of the new value.
   * @param number The number of the new value.
   */
  void addIntegerValue(Value value, int64_t number);

  /**
   * @brief This method adds a classical double value to the hybrid state.
   *
   * @param value The value object of the new value.
   * @param number The number of the new value.
   */
  void addDoubleValue(Value value, double number);

  /**
   * @brief This method applies a gate to the state.
   *
   * This method changes the hybrid state according to a gate.
   *
   * @param gate The name of the gate to be applied.
   * @param targets An array of the indices of the target qubits.
   * @param ctrlsQuantum An array of the indices of the ctrl qubits.
   * @param ctrlsClassical An array of the indices of the ctrl bits.
   * @param params The parameter applied to the gate.
   * @throw std::domain_error If the number of nonzero amplitudes would exceed
   * maxNonzeroAmplitudes or if the hybridState does not hold a quantumState.
   */
  void propagateGate(Operation* gate, std::span<unsigned int> targets,
                     std::span<unsigned int> ctrlsQuantum = {},
                     std::span<unsigned int> ctrlsClassical = {},
                     std::span<double> params = {});

  /**
   * @brief This method applies a measurement.
   *
   * This method applies a measurement, changing the qubits and the classical
   * values corresponding to the measurement.
   *
   * @param quantumTarget The index of the qubit to be measured.
   * @param classicalTarget The value to save the measurement result to.
   * @param ctrlsClassical An array of ctrl values.
   * @throws domain_error If the quantum state of the hybrid state is TOP.
   * @return One or two hybrid states corresponding to the measurement
   * outcomes.
   */
  std::vector<HybridState>
  propagateMeasurement(unsigned int quantumTarget, Value classicalTarget,
                       std::span<Value> ctrlsClassical = {});

  /**
   * @brief This method applies a reset.
   *
   * This method applies a reset, changing the qubits and creates one or two new
   * states. The procedure is done as if the qubit was measured, put to zero if
   * the measurement was one, and the result discarded.
   *
   * @param target The index of the qubit to be measured.
   * @param ctrlsClassical An array of the ctrl values.
   * @throws domain_error If the quantum state of the hybrid state is TOP.
   * @return One or two hybrid states corresponding to the measurement outcomes
   * during the reset, but with the qubit always in the zero state.
   */
  std::vector<HybridState> propagateReset(unsigned int target,
                                          std::span<Value> ctrlsClassical = {});

  /**
   * @brief This method applies a classical operation.
   *
   * This method changes the hybrid state according to a classical operation.
   * The operation might be controlled by classical values.
   *
   * @param op The operation to be applied.
   * @param dest The value the result of the operation is written to.
   * @param operand1 The first value used by the operation.
   * @param operand2 The second value used by the operation, might be null.
   * @param operand3 The third value used by the operation, might be null.
   * @param ctrls An array of the ctrl values.
   */
  void propagateClassicalOperation(Operation* op, Value dest, Value operand1,
                                   Value operand2 = nullptr,
                                   Value operand3 = nullptr,
                                   std::span<Value> ctrls = {});

  /**
   * @brief This method unifies two HybridStates.
   *
   * This method unifies the current HybridState with the given one and returns
   * a new HybridState, if the new state has no more than maxNonzeroAmplitudes.
   * Otherwise, throws a domain_error.
   *
   * @param that The HybridState to unify this with.
   * @throw std::domain_error If the unified QuantumState would exceed
   * maxNonzeroAmplitudes of this.
   */
  HybridState unify(HybridState that);

  bool operator==(const HybridState& that) const;

  [[nodiscard("HybridState::isQubitAlwaysOne called but ignored")]] bool
  isQubitAlwaysOne(unsigned int q) const;

  [[nodiscard("HybridState::isQubitAlwaysZero called but ignored")]] bool
  isQubitAlwaysZero(unsigned int q) const;

  [[nodiscard("HybridState::isValueTrue called but ignored")]] bool
  isValueTrue(Value v) const;

  [[nodiscard("HybridState::isValueFalse called but ignored")]] bool
  isValueFalse(Value v) const;

  /**
   * @brief Checks if a given combination of values-qubit values has a nonzero
   * probability.
   *
   * This method receives a number of qubit and values and checks whether
   * they have for a given value always a zero amplitude. If the hybridState is
   * top, it is not guaranteed that the amplitude is always zero and false is
   * returned.
   * The values for the classical values are not the numeric ones, but whether
   * they are zero (false) or non-zero (true).
   *
   * @param qubits The qubits which are being checked.
   * @param qubitValue The value for which is tested whether there is a nonzero
   * amplitude.
   * @param values The values to check.
   * @param classicalValuesToCheck Whether to check if the values are zero
   * (false) or non-zero (true).
   * @returns True if the amplitude is always zero, false otherwise.
   */
  [[nodiscard("HybridState::hasAlwaysZeroAmplitude called but ignored")]] bool
  hasAlwaysZeroProbability(std::span<unsigned int> qubits,
                           unsigned int qubitValue,
                           std::span<unsigned int> values = {},
                           std::span<Value> classicalValuesToCheck = {}) const;

  /**
   * @brief Returns a classical value that is equivalent to qubit.
   *
   * Returns a classical value that is always true (=/= 0) when the given qubit
   * is 1, and the boolean value true. Alternatively, it can return a value that
   * is always false (== 0) if the qubit is 1. In that case, the returned bool
   * is false.
   *
   * @param qubit Index of qubit.
   * @returns Classical value that is equivalent or inverse to qubit if it
   * exists and true, if the qubit is equivalent to the value. False, if the
   * qubit is the inverse of the value.
   */
  std::pair<std::optional<Value>, bool>
  getValueThatIsEquivalentToQubit(unsigned int qubit);
};
} // namespace mlir::qco

#endif // MQT_CORE_HYBRIDSTATE_H
