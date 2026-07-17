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

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <ios>
#include <memory>
#include <ostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

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
  if (top) {
    return "TOP";
  }
  std::ostringstream oss;

  oss << "{" << this->qState->toString() << "}: ";
  unsigned int i = 0;
  bool first = true;
  for (const auto& key : integerValues.keys()) {
    if (!first) {
      oss << ", ";
    }
    first = false;
    oss << "integerValue" << i << " = " << integerValues.at(key);
    ++i;
  }
  unsigned int j = 0;
  for (const auto& key : doubleValues.keys()) {
    if (!first) {
      oss << ", ";
    }
    first = false;

    oss << "doubleValue" << j << " = " << std::fixed << std::setprecision(2)
        << doubleValues.at(key);

    ++j;
  }
  if (i > 0 || j > 0) {
    oss << "; ";
  }
  oss << "p = " << std::fixed << std::setprecision(2) << this->probability
      << ";";
  return oss.str();
}

bool HybridState::operator==(const HybridState& that) const {
  if (top) {
    return that.top;
  }
  if (std::fabs(probability - that.probability) > 1e-4 ||
      *qState.get() != *that.qState.get()) {
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

  return std::ranges::all_of(doubleValues, [&](const auto& p) {
    auto it = that.doubleValues.find(p.first);
    return it != that.doubleValues.end() &&
           std::fabs(it->second - p.second) <= 1e-4;
  });
}

void HybridState::addIntegerValue(const Value value, const int64_t number) {
  integerValues[value] = number;
}

void HybridState::addDoubleValue(const Value value, const double number) {
  doubleValues[value] = number;
}

void HybridState::changeGlobalIndex(const unsigned int target,
                                    const unsigned int newIndex) const {
  qState->changeGlobalIndex(target, newIndex);
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
        paramValues.push_back(static_cast<double>(integerValues.at(p)));
      } else if (doubleValues.contains(p)) {
        paramValues.push_back(static_cast<double>(doubleValues.at(p)));
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
  } else {
    try {
      qState->propagateGate(gate, targets, ctrlsQuantum);
    } catch (std::domain_error const&) {
      top = true;
    }
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
    const int64_t opRes = getArithIntegerOpResult(
        op, integerValues.at(operand1),
        operand2 == nullptr ? 0 : integerValues.at(operand2),
        operand3 == nullptr ? 0 : integerValues.at(operand3));
    integerValues[dest] = opRes;
  } else {
    if (!doubleValues.contains(operand1) ||
        (operand2 != nullptr && !doubleValues.contains(operand2)) ||
        !doubleValues.contains(dest)) {
      throw std::domain_error(
          "HybridState needs a classical value for a classical operation that "
          "is not existent in current HybridState.");
    }
    const double opRes = getArithDoubleOpResult(
        op, doubleValues.at(operand1),
        operand2 == nullptr ? 0.0 : doubleValues.at(operand2));
    doubleValues[dest] = opRes;
  }
}

HybridState HybridState::unify(const HybridState& that) {
  auto newHybridState = HybridState();
  try {
    newHybridState.qState =
        std::make_shared<QuantumState>(qState->unify(*that.qState));
  } catch (std::domain_error const&) {
    newHybridState.top = true;
    return newHybridState;
  }
  newHybridState.probability = this->probability * that.probability;

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
    return std::fabs(doubleValues.at(v)) > 1e-4;
  }
  throw std::domain_error("Value of a classical value is asked which does not "
                          "exist in the HybridState.");
}

bool HybridState::hasAlwaysZeroProbability(
    const std::unordered_map<unsigned int, bool>& qubitValues,
    const llvm::DenseMap<Value, bool>& classicalValues) const {
  for (const auto& [v, i] : classicalValues) {
    if (integerValues.contains(v)) {
      if ((integerValues.at(v) == 0 && i) || (integerValues.at(v) != 0 && !i)) {
        return true;
      }
    } else if (doubleValues.contains(v)) {
      const bool zeroDouble = std::fabs(doubleValues.at(v)) < 1e-4;
      if ((zeroDouble && i) || (!zeroDouble && !i)) {
        return true;
      }
    } else {
      throw std::domain_error("Value of a classical value is asked which does "
                              "not exist in the HybridState.");
    }
  }

  return qState->hasAlwaysZeroAmplitude(qubitValues);
}

llvm::DenseMap<Value, bool>
HybridState::getValueThatIsEquivalentToQubit(const unsigned int qubit) const {
  llvm::DenseMap<Value, bool> result;
  const bool qubitZero = qState->isQubitAlwaysZero(qubit);
  const bool qubitOne = qubitZero ? false : qState->isQubitAlwaysOne(qubit);
  if (!qubitZero && !qubitOne) {
    return result;
  }
  for (const auto& [v, i] : integerValues) {
    if ((qubitZero && i == 0) || (qubitOne && i != 0)) {
      result[v] = true;
    } else {
      result[v] = false;
    }
  }
  for (const auto& [v, d] : doubleValues) {
    if ((qubitZero && std::fabs(d) < 1e-4) ||
        (qubitOne && std::fabs(d) >= 1e-4)) {
      result[v] = true;
    } else {
      result[v] = false;
    }
  }
  return result;
}

} // namespace mlir::qco

#endif // MQT_CORE_HYBRIDSTATE
