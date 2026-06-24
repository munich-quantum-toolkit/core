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

#include <llvm/ADT/DenseMap.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Value.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
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

  HybridState() : probability(0.0) {}

  /**
   * @brief Checks if all positive classical controls hold and all negative
   * classical controls do not hold.
   *
   * @param posCtrlsClassical An array of the classical positive control values.
   * @param negCtrlsClassical An array of the classical negative control values.
   * @return True if the controls together evaluate to true.
   * @throws domain_error If a classical control value cannot be found.
   */
  bool isOperationExecutable(const std::span<Value> posCtrlsClassical,
                             const std::span<Value> negCtrlsClassical) {
    for (const Value posCtrl : posCtrlsClassical) {
      if (integerValues.contains(posCtrl) && integerValues.at(posCtrl) == 0) {
        return false;
      }
      if (doubleValues.contains(posCtrl) &&
          std::norm(doubleValues.at(posCtrl)) < 1e-4) {
        return false;
      }
      if (!doubleValues.contains(posCtrl) && !integerValues.contains(posCtrl)) {
        throw std::domain_error(
            "HybridState needs a classical value for operation control that is "
            "not existent in current HybridState.");
      }
    }
    for (const Value negCtrl : negCtrlsClassical) {
      if (integerValues.contains(negCtrl) && integerValues.at(negCtrl) != 0) {
        return false;
      }
      if (doubleValues.contains(negCtrl) &&
          std::norm(doubleValues.at(negCtrl)) > 1e-4) {
        return false;
      }
      if (!doubleValues.contains(negCtrl) && !integerValues.contains(negCtrl)) {
        throw std::domain_error(
            "HybridState needs a classical value for operation control that is "
            "not existent in current HybridState.");
      }
    }
    return true;
  }

  /**
   * @brief This method applies a measurement or reset.
   *
   * This method applies a measurement or reset, changing the qubits and the
   * classical values (if a measurement is applied) corresponding to the
   * measurement.
   *
   * @param quantumTarget The index of the qubit to be measured.
   * @param reset True if a reset is applied.
   * @param classicalTarget The value to save the measurement result to.
   * @param posCtrlsClassical An array of the classical positive control values.
   * @param negCtrlsClassical An array of the classical negative control values.
   * @throws domain_error If a classical control value cannot be found.
   * @return One or two hybrid states corresponding to the measurement or reset
   * outcomes.
   */
  std::vector<HybridState>
  propagateMeasurementOrReset(const unsigned int quantumTarget,
                              const bool reset,
                              const Value classicalTarget = nullptr,
                              const std::span<Value> posCtrlsClassical = {},
                              const std::span<Value> negCtrlsClassical = {}) {
    if (top || !isOperationExecutable(posCtrlsClassical, negCtrlsClassical)) {
      return {*this};
    }

    std::vector<HybridState> results;

    const auto [newQuantumStates, availableStates] =
        reset ? qState->resetQubit(quantumTarget)
              : qState->measureQubit(quantumTarget);

    for (const int64_t i : {0, 1}) {
      if (!availableStates.contains(i) || !availableStates.at(i)) {
        continue;
      }

      const auto& [measProbability, measQS] = newQuantumStates.at(i);
      auto newHybrid = HybridState();
      newHybrid.probability = measProbability * probability;
      newHybrid.integerValues = integerValues;
      newHybrid.doubleValues = doubleValues;
      if (!reset) {
        newHybrid.integerValues[classicalTarget] = i;
      }
      newHybrid.qState = measQS;

      results.push_back(newHybrid);
    }

    return results;
  }

public:
  explicit HybridState(std::span<unsigned int> globalQubitNumber,
                       std::size_t maxNonzeroAmplitudes,
                       double probability = 1.0);

  ~HybridState();

  void print(std::ostream& os) const;

  [[nodiscard("HybridState::toString called but ignored")]]
  std::string toString() const;

  [[nodiscard("HybridState::isHybridStateTop called but ignored")]] bool
  isHybridStateTop() const {
    return top;
  }

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
   * @brief This method changes the global index of a qubit in the quantum
   * state.
   *
   * @param target The old global index of a qubit.
   * @param newIndex The new global index for the qubit.
   */
  void changeGlobalIndex(unsigned int target, unsigned int newIndex) const;

  /**
   * @brief This method applies a gate to the state.
   *
   * This method changes the hybrid state according to a gate.
   *
   * @param gate The name of the gate to be applied.
   * @param targets An array of the indices of the target qubits.
   * @param ctrlsQuantum An array of the global indices of the ctrl qubits.
   * @param posCtrlsClassical An array of the classical positive control values.
   * @param negCtrlsClassical An array of the classical negative control values.
   * @param params The values of parameters applied to the gate.
   * @throws domain_error If a classical control value cannot be found.
   */
  void propagateGate(Operation* gate, std::span<unsigned int> targets,
                     std::span<unsigned int> ctrlsQuantum = {},
                     std::span<Value> posCtrlsClassical = {},
                     std::span<Value> negCtrlsClassical = {},
                     std::span<Value> params = {});

  /**
   * @brief This method applies a measurement.
   *
   * This method applies a measurement, changing the qubits and the classical
   * values corresponding to the measurement.
   *
   * @param quantumTarget The index of the qubit to be measured.
   * @param classicalTarget The value to save the measurement result to.
   * @param posCtrlsClassical An array of the classical positive control values.
   * @param negCtrlsClassical An array of the classical negative control values.
   * @throws domain_error If a classical control value cannot be found.
   * @return One or two hybrid states corresponding to the measurement
   * outcomes.
   */
  std::vector<HybridState>
  propagateMeasurement(unsigned int quantumTarget, Value classicalTarget,
                       std::span<Value> posCtrlsClassical = {},
                       std::span<Value> negCtrlsClassical = {});

  /**
   * @brief This method applies a reset.
   *
   * This method applies a reset, changing the qubits and creates one or two new
   * states. The procedure is done as if the qubit was measured, put to zero if
   * the measurement was one, and the result discarded.
   *
   * @param target The index of the qubit to be measured.
   * @param posCtrlsClassical An array of the classical positive control values.
   * @param negCtrlsClassical An array of the classical negative control values.
   * @throws domain_error If a classical control value cannot be found.
   * @return One or two hybrid states corresponding to the measurement outcomes
   * during the reset, but with the qubit always in the zero state.
   */
  std::vector<HybridState>
  propagateReset(unsigned int target, std::span<Value> posCtrlsClassical = {},
                 std::span<Value> negCtrlsClassical = {});

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
   * @param posCtrlsClassical An array of the classical positive control values.
   * @param negCtrlsClassical An array of the classical negative control values.
   * @throws domain_error If a classical value cannot be found.
   * @throws runtime_error If classical operation is not supported.
   */
  void propagateClassicalOperation(Operation* op, Value dest, Value operand1,
                                   Value operand2 = nullptr,
                                   Value operand3 = nullptr,
                                   std::span<Value> posCtrlsClassical = {},
                                   std::span<Value> negCtrlsClassical = {});

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
  HybridState unify(const HybridState& that);

  bool operator==(const HybridState& that) const;

  [[nodiscard("HybridState::isQubitAlwaysOne called but ignored")]] bool
  isQubitAlwaysOne(unsigned int q) const;

  [[nodiscard("HybridState::isQubitAlwaysZero called but ignored")]] bool
  isQubitAlwaysZero(unsigned int q) const;

  [[nodiscard("HybridState::isValueTrue called but ignored")]] bool
  isValueTrue(Value v) const;

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
   * @param qubitValues Pairs of the qubits that are being checked and the
   * values that they are being checked for.
   * @param classicalValues The classical values to check.
   * @throws domain_error If a classical value cannot be found.
   * @returns True if the amplitude is always zero, false otherwise.
   */
  [[nodiscard("HybridState::hasAlwaysZeroAmplitude called but ignored")]] bool
  hasAlwaysZeroProbability(
      const std::unordered_map<unsigned int, bool>& qubitValues,
      const llvm::DenseMap<Value, bool>& classicalValues) const;

  /**
   * @brief Returns a classical value that is equivalent to qubit.
   *
   * Returns a classical value that is always true (=/= 0) when the given qubit
   * is 1, and the boolean value true. Alternatively, it can return a value that
   * is always false (== 0) if the qubit is 1. In that case, the returned bool
   * is false.
   *
   * @param qubit Index of qubit.
   * @returns A map of classical values that are equivalent or inverse to qubit.
   * Th emaps value is true, if the qubit is equivalent to the value. False, if
   * the qubit is the inverse of the value.
   */
  [[nodiscard("HybridState::getValueThatIsEquivalentToQubit called but "
              "ignored")]] llvm::DenseMap<Value, bool>
  getValueThatIsEquivalentToQubit(unsigned int qubit) const;
};
} // namespace mlir::qco

#endif // MQT_CORE_HYBRIDSTATE_H
