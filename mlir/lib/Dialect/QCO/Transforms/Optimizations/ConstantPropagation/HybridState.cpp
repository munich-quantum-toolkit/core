/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#ifndef MQT_CORE_HYBRIDSTATE
#define MQT_CORE_HYBRIDSTATE
#include "mlir/Dialect/QCO/Transforms/Optimizations/ConstantPropagation/HybridState.hpp"

#include "mlir/Dialect/QCO/Transforms/Optimizations/ConstantPropagation/ClassicalArithOperation.h"
#include "mlir/Dialect/QCO/Transforms/Optimizations/ConstantPropagation/QuantumState.hpp"

#include <format>

namespace mlir::qco {

HybridState::HybridState(std::span<unsigned int> globalQubitNumber,
                         std::size_t maxNonzeroAmplitudes,
                         const double probability)
    : probability(probability) {
  qState = make_shared<QuantumState>(globalQubitNumber, maxNonzeroAmplitudes);
}

HybridState::~HybridState() {
  qState.reset();
  integerValues.clear();
  doubleValues.clear();
}

void HybridState::print(std::ostream& os) const { os << this->toString(); }
std::string HybridState::toString() const {
  std::string str = "{" + this->qState->toString() + "}: ";
  unsigned int i = 0;
  for (const auto& key : integerValues.keys()) {
    str += "integerValue" + std::to_string(i) + " = " +
           std::to_string(integerValues.at(key)) + ", ";
    ++i;
  }
  unsigned int j = 0;
  for (const auto& key : integerValues.keys()) {
    str += "doubleValue" + std::to_string(j) + " = " +
           std::format("{:.2f}", doubleValues.at(key)) + ", ";
    ++j;
  }
  if (i > 0 || j > 0) {
    str += "; ";
  }
  str += "p = " + std::format("{:.2f}", this->probability) + ";";
  return str;
}

bool HybridState::operator==(const HybridState& that) const {
  if (top) {
    return that.top;
  }
  if (probability != that.probability || *qState.get() != *that.qState.get()) {
    return false;
  }

  if (integerValues.size() != that.integerValues.size() ||
      doubleValues.size() != that.doubleValues.size()) {
    return false;
  }

  for (const auto& [i, v] : integerValues) {
    if (!that.integerValues.contains(i) || that.integerValues.at(i) != v) {
      return false;
    }
  }

  for (const auto& [d, v] : doubleValues) {
    if (!that.doubleValues.contains(d) ||
        std::norm(that.doubleValues.at(d)) > 1e-4) {
      return false;
    }
  }

  return true;
}

void HybridState::addIntegerValue(const Value value, const int64_t number) {
  integerValues[value] = number;
}

void HybridState::addDoubleValue(const Value value, const double number) {
  doubleValues[value] = number;
}

void HybridState::propagateGate(Operation* gate,
                                const std::span<unsigned int> targets,
                                const std::span<unsigned int> ctrlsQuantum,
                                const std::span<Value> posCtrlsClassical,
                                const std::span<Value> negCtrlsClassical,
                                const std::span<Value> params) {
  if (top) {
    return;
  }

  if (!isOperationExecutable(posCtrlsClassical, negCtrlsClassical)) {
    return;
  }

  if (!params.empty()) {
    std::vector<double> paramValues;
    paramValues.reserve(params.size());
    for (Value p : params) {
      if (integerValues.contains(p)) {
        paramValues.push_back(integerValues.at(p));
      } else if (doubleValues.contains(p)) {
        paramValues.push_back(doubleValues.at(p));
      } else {
        throw std::domain_error(
            "HybridState needs a classical value for gate parameters that is "
            "not existent in current HybridState.");
      }
    }
    try {
      qState->propagateGate(gate, targets, ctrlsQuantum, paramValues);
    } catch (std::domain_error const&) {
      top = true;
    }
  }

  try {
    qState->propagateGate(gate, targets, ctrlsQuantum);
  } catch (std::domain_error const&) {
    top = true;
  }
}

std::vector<HybridState>
HybridState::propagateMeasurement(const unsigned int quantumTarget,
                                  const Value classicalTarget,
                                  const std::span<Value> posCtrlsClassical,
                                  const std::span<Value> negCtrlsClassical) {

  return propagateMeasurementOrReset(quantumTarget, false, classicalTarget,
                                     posCtrlsClassical, negCtrlsClassical);
}

std::vector<HybridState>
HybridState::propagateReset(const unsigned int target,
                            const std::span<Value> posCtrlsClassical,
                            const std::span<Value> negCtrlsClassical) {
  return propagateMeasurementOrReset(target, true, nullptr, posCtrlsClassical,
                                     negCtrlsClassical);
}

void HybridState::propagateClassicalOperation(
    Operation* op, const Value dest, const Value operand1, const Value operand2,
    const Value operand3, const std::span<Value> posCtrlsClassical,
    const std::span<Value> negCtrlsClassical) {
  if (top || !isOperationExecutable(posCtrlsClassical, negCtrlsClassical)) {
    return;
  }

  if (isa<IntegerType>(op->getResult(0).getType())) {
    if (!integerValues.contains(operand1) ||
        (operand2 != nullptr && !integerValues.contains(operand2)) ||
        (operand3 != nullptr && !integerValues.contains(operand3)) ||
        !integerValues.contains(dest)) {
      throw std::domain_error(
          "HybridState needs a classical value for a classical operation that "
          "is not existent in current HybridState.");
    }
    const int64_t opRes =
        getArithOpResult(op, integerValues.at(operand1),
                         operand2 == nullptr ? 0 : integerValues.at(operand2),
                         operand3 == nullptr ? 0 : integerValues.at(operand3));
    integerValues[dest] = opRes;
  } else {
    if (!doubleValues.contains(operand1) ||
        (operand2 != nullptr && !doubleValues.contains(operand2)) ||
        (operand3 != nullptr && !doubleValues.contains(operand3)) ||
        !doubleValues.contains(dest)) {
      throw std::domain_error(
          "HybridState needs a classical value for a classical operation that "
          "is not existent in current HybridState.");
    }
    const double opRes =
        getArithOpResult(op, doubleValues.at(operand1),
                         operand2 == nullptr ? 0 : doubleValues.at(operand2),
                         operand3 == nullptr ? 0 : doubleValues.at(operand3));
    doubleValues[dest] = opRes;
  }
}

HybridState HybridState::unify(HybridState that) {
  auto newHybridState = HybridState();
  try {
    newHybridState.qState =
        std::make_shared<QuantumState>(qState->unify(*that.qState));
  } catch (std::domain_error const&) {
    newHybridState.top = true;
    return newHybridState;
  }
  newHybridState.probability *= this->probability;

  auto newIntegerValues = llvm::DenseMap<Value, int64_t>(
      integerValues.size() + that.integerValues.size());
  auto newDoubleValues = llvm::DenseMap<Value, double>(
      doubleValues.size() + that.doubleValues.size());

  for (const auto& [v, i] : integerValues) {
    newIntegerValues[v] = i;
  }
  for (const auto& [v, i] : that.integerValues) {
    newIntegerValues[v] = i;
  }

  for (const auto& [v, d] : doubleValues) {
    newDoubleValues[v] = d;
  }
  for (const auto& [v, d] : that.doubleValues) {
    newDoubleValues[v] = d;
  }

  newHybridState.integerValues = newIntegerValues;
  newHybridState.doubleValues = newDoubleValues;

  return newHybridState;
}

bool HybridState::isQubitAlwaysOne(unsigned int q) const {
  return qState->isQubitAlwaysOne(q);
}

bool HybridState::isQubitAlwaysZero(unsigned int q) const {
  return qState->isQubitAlwaysZero(q);
}

bool HybridState::isValueTrue(const Value v) const {
  if (integerValues.contains(v)) {
    return integerValues.at(v) != 0;
  }
  if (doubleValues.contains(v)) {
    return std::norm(doubleValues.at(v)) > 1e-4;
  }
  throw std::domain_error("Value of a classical value is asked which does not "
                          "exist in the HybridState.");
}

bool HybridState::hasAlwaysZeroProbability(
    const std::span<unsigned int> qubits, const unsigned int qubitValue,
    std::span<std::pair<Value, int64_t>> classicalIntegerValues,
    std::span<std::pair<Value, double>> classicalDoubleValues) const {
  for (const auto& [v, i] : classicalIntegerValues) {
    if (!integerValues.contains(v)) {
      throw std::domain_error("Value of a classical value is asked which does "
                              "not exist in the HybridState.");
    }
    if (integerValues.at(v) != i) {
      return true;
    }
  }

  for (const auto& [v, d] : classicalDoubleValues) {
    if (!doubleValues.contains(v)) {
      throw std::domain_error("Value of a classical value is asked which does "
                              "not exist in the HybridState.");
    }
    if (std::norm(doubleValues.at(v) - d) > 1e-4) {
      return true;
    }
  }

  return qState->hasAlwaysZeroAmplitude(qubits, qubitValue);
}

std::pair<std::optional<Value>, bool>
HybridState::getValueThatIsEquivalentToQubit(const unsigned int qubit) const {
  if (integerValues.empty() && doubleValues.empty()) {
    return {std::optional<Value>(), false};
  }
  const bool qubitZero = qState->isQubitAlwaysZero(qubit);
  const bool qubitOne = qubitZero ? false : qState->isQubitAlwaysOne(qubit);
  if (!qubitZero && !qubitOne) {
    return {std::optional<Value>(), false};
  }
  for (const auto& [v, i] : integerValues) {
    if ((qubitZero && i == 0) || (qubitOne && i != 0)) {
      return {std::optional(v), true};
    }
    return {std::optional(v), false};
  }
  for (const auto& [v, d] : doubleValues) {
    if ((qubitZero && std::norm(d) < 1e-4) ||
        (qubitOne && std::norm(d) >= 1e-4)) {
      return {std::optional(v), true};
    }
    return {std::optional(v), false};
  }
  return {std::optional<Value>(), false};
}

} // namespace mlir::qco

#endif // MQT_CORE_HYBRIDSTATE
