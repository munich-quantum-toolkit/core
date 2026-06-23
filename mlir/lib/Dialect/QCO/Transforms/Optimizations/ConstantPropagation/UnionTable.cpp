/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#ifndef MQT_CORE_UNIONTABLE
#define MQT_CORE_UNIONTABLE
#include "mlir/Dialect/QCO/Transforms/Optimizations/ConstantPropagation/UnionTable.hpp"

#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"

#include <mlir/IR/Operation.h>

#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace mlir::qco {

UnionTable::UnionTable(std::size_t maxNonzeroAmplitudes,
                       std::size_t maximumHybridEntries)
    : maxNonzeroAmplitudes(maxNonzeroAmplitudes),
      maximumHybridEntries(maximumHybridEntries) {}

UnionTable::~UnionTable() = default;

void UnionTable::print(std::ostream& os) const { os << this->toString(); }

std::string UnionTable::toString() const {
  std::string result;
  for (const auto& entry : entries) {
    std::vector<unsigned int> qubitIndices;
    for (const auto& q : entry->participatingQubits) {
      qubitIndices.push_back(qubitsToGlobalIndices.at(q));
    }
    std::ranges::sort(qubitIndices, std::greater<int>());
    result += "Qubits: ";
    for (const auto qit : qubitIndices) {
      result += std::to_string(qit);
    }
    result += ", HybridStates: {";
    if (entry->top) {
      result += "TOP";
    } else {
      bool first = true;
      for (HybridState const& hs : entry->states) {
        if (!first) {
          result += " ";
        }
        first = false;
        result += hs.toString();
      }
    }
    result += "}\n";
  }
  return result;
}

bool UnionTable::areStatesAllTop() {
  if (allTop) {
    return true;
  }
  for (const auto& ute : entries) {
    if (!ute->top) {
      return false;
    }
  }
  allTop = true;
  return true;
}

void UnionTable::propagateGate(Operation* gate, const std::span<Value> targets,
                               const std::span<Value> newQuantumTargets,
                               const std::span<Value> ctrlsQuantum,
                               const std::span<Value> posCtrlsClassical,
                               const std::span<Value> negCtrlsClassical,
                               const std::span<Value> params) {
  const std::set<UnionTableEntry> participatingEntries =
      collectParticipatingEntries(targets, ctrlsQuantum, posCtrlsClassical,
                                  negCtrlsClassical, params);

  if (isa<SWAPOp>(*gate) && ctrlsQuantum.empty() && posCtrlsClassical.empty() &&
      negCtrlsClassical.empty() && participatingEntries.size() == 2 &&
      !valuesToEntries.at(targets[0])->top &&
      !valuesToEntries.at(targets[1])->top) {
    applySwapGate(targets, newQuantumTargets);
    return;
  }

  try {
    unifyEntries(participatingEntries);
  } catch (std::domain_error&) {
    putEntriesToTop(participatingEntries);
    replaceValuesGlobally(targets, newQuantumTargets);
    return;
  }
  if (valuesToEntries.at(targets[0])->top) {
    replaceValuesGlobally(targets, newQuantumTargets);
    return;
  }

  std::vector<unsigned int> targetQubitIndices;
  std::vector<unsigned int> ctrlQubitIndices;
  for (auto const q : targets) {
    targetQubitIndices.push_back(qubitsToGlobalIndices.at(q));
  }
  for (auto const q : ctrlsQuantum) {
    ctrlQubitIndices.push_back(qubitsToGlobalIndices.at(q));
  }

  const auto ute = valuesToEntries.at(*targets.begin());
  for (auto hs : ute->states) {
    hs.propagateGate(gate, targetQubitIndices, ctrlQubitIndices,
                     posCtrlsClassical, negCtrlsClassical, params);
    if (hs.isHybridStateTop()) {
      putEntriesToTop({*ute});
      break;
    }
  }
  replaceValuesGlobally(targets, newQuantumTargets);
}

void UnionTable::propagateMeasurement(
    const Value quantumTarget, const Value newQuantumValue,
    const Value classicalTarget, const std::span<Value> posCtrlsClassical,
    const std::span<Value> negCtrlsClassical) {

  std::vector targetVec = {quantumTarget, classicalTarget};
  const std::set<UnionTableEntry> participatingEntries =
      collectParticipatingEntries(targetVec, {}, posCtrlsClassical,
                                  negCtrlsClassical);

  std::vector quantumTargetVec = {quantumTarget};
  std::vector newQuantumValueVec = {newQuantumValue};
  try {
    unifyEntries(participatingEntries);
  } catch (std::domain_error&) {
    putEntriesToTop(participatingEntries);
    replaceValuesGlobally(quantumTargetVec, newQuantumValueVec);
    return;
  }

  const auto ute = valuesToEntries.at(quantumTarget);
  if (ute->top) {
    replaceValuesGlobally(quantumTargetVec, newQuantumValueVec);
    return;
  }

  std::vector<HybridState> vecOfNewStates;

  for (auto hs : ute->states) {
    auto newStates = hs.propagateMeasurement(
        qubitsToGlobalIndices.at(quantumTarget), classicalTarget,
        posCtrlsClassical, negCtrlsClassical);
    if (hs.isHybridStateTop()) {
      putEntriesToTop({*ute});
      vecOfNewStates.clear();
      break;
    }
    vecOfNewStates.insert(vecOfNewStates.end(), newStates.begin(),
                          newStates.end());
  }
  if (vecOfNewStates.size() > maximumHybridEntries) {
    putEntriesToTop({*ute});
  } else {
    ute->states = vecOfNewStates;
  }
  replaceValuesGlobally(quantumTargetVec, newQuantumValueVec);
}

void UnionTable::propagateReset(const Value quantumTarget,
                                const Value newQuantumValue,
                                const std::span<Value> posCtrlsClassical,
                                const std::span<Value> negCtrlsClassical) {

  std::vector quantumTargetVec = {quantumTarget};
  const std::set<UnionTableEntry> participatingEntries =
      collectParticipatingEntries(quantumTargetVec, {}, posCtrlsClassical,
                                  negCtrlsClassical);

  std::vector newQuantumValueVec = {newQuantumValue};
  try {
    unifyEntries(participatingEntries);
  } catch (std::domain_error&) {
    putEntriesToTop(participatingEntries);
    replaceValuesGlobally(quantumTargetVec, newQuantumValueVec);
    return;
  }

  const auto ute = valuesToEntries.at(quantumTarget);
  if (ute->top) {
    replaceValuesGlobally(quantumTargetVec, newQuantumValueVec);
    return;
  }

  std::vector<HybridState> vecOfNewStates;

  for (auto hs : ute->states) {
    try {
      auto newStates =
          hs.propagateReset(qubitsToGlobalIndices.at(quantumTarget),
                            posCtrlsClassical, negCtrlsClassical);
      vecOfNewStates.insert(vecOfNewStates.end(), newStates.begin(),
                            newStates.end());
    } catch (std::domain_error&) {
      putEntriesToTop({*ute});
      vecOfNewStates.clear();
      break;
    }
  }
  ute->states = vecOfNewStates;
  replaceValuesGlobally(quantumTargetVec, newQuantumValueVec);
}

void UnionTable::propagateQubitAlloc(const Value qubit) {
  unsigned int maxIndex = 0;
  if (!qubitsToGlobalIndices.empty()) {
    maxIndex = std::ranges::max(qubitsToGlobalIndices.values()) + 1;
  }
  std::vector globalQubitIndex = {maxIndex};
  const auto hs = HybridState(globalQubitIndex, maxNonzeroAmplitudes);

  qubitsToGlobalIndices[qubit] = maxIndex;
  auto ute = UnionTableEntry();
  ute.states.push_back(hs);
  ute.participatingQubits.insert(qubit);
  const auto ptrToUTE = std::make_shared<UnionTableEntry>(ute);
  entries.insert(ptrToUTE);
  valuesToEntries[qubit] = ptrToUTE;
}

void UnionTable::propagateIntAlloc(const Value intValue, const int64_t number) {
  auto hs = HybridState({}, maxNonzeroAmplitudes);
  hs.addIntegerValue(intValue, number);

  auto ute = UnionTableEntry();
  ute.states.push_back(hs);
  ute.participatingClassicalValues.insert(intValue);
  entries.insert(std::make_shared<UnionTableEntry>(ute));
  valuesToEntries[intValue] = std::make_shared<UnionTableEntry>(ute);
}

void UnionTable::propagateDoubleAlloc(const Value doubleValue,
                                      const double number) {
  auto hs = HybridState({}, maxNonzeroAmplitudes);
  hs.addDoubleValue(doubleValue, number);

  auto ute = UnionTableEntry();
  ute.states.push_back(hs);
  ute.participatingClassicalValues.insert(doubleValue);
  entries.insert(std::make_shared<UnionTableEntry>(ute));
  valuesToEntries[doubleValue] = std::make_shared<UnionTableEntry>(ute);
}

bool UnionTable::isQubitAlwaysOne(const Value q) const {
  const unsigned int qubitIndex = qubitsToGlobalIndices.at(q);
  const auto ute = valuesToEntries.at(q);
  for (auto const& hs : ute->states) {
    if (!hs.isQubitAlwaysOne(qubitIndex)) {
      return false;
    }
  }
  return true;
}

bool UnionTable::isQubitAlwaysZero(const Value q) const {
  const unsigned int qubitIndex = qubitsToGlobalIndices.at(q);
  const auto ute = valuesToEntries.at(q);
  for (auto const& hs : ute->states) {
    if (!hs.isQubitAlwaysZero(qubitIndex)) {
      return false;
    }
  }
  return true;
}

bool UnionTable::isClassicalValueAlwaysTrue(const Value c) const {
  const auto ute = valuesToEntries.at(c);
  for (auto const& hs : ute->states) {
    if (!hs.isValueTrue(c)) {
      return false;
    }
  }
  return true;
}

bool UnionTable::isClassicalValueAlwaysFalse(const Value c) const {
  const auto ute = valuesToEntries.at(c);
  for (auto const& hs : ute->states) {
    if (hs.isValueTrue(c)) {
      return false;
    }
  }
  return true;
}

bool UnionTable::hasAlwaysZeroProbability(
    const llvm::DenseMap<Value, bool>& qubitValues,
    const llvm::DenseMap<Value, int64_t>& classicalIntegerValues,
    const llvm::DenseMap<Value, double>& classicalDoubleValues) const {
  std::set<UnionTableEntry> participatingEntries;
  for (auto& [qV, _] : qubitValues) {
    participatingEntries.insert(*valuesToEntries.at(qV));
  }
  for (auto& [iV, _] : classicalIntegerValues) {
    participatingEntries.insert(*valuesToEntries.at(iV));
  }
  for (auto& [dV, _] : classicalDoubleValues) {
    participatingEntries.insert(*valuesToEntries.at(dV));
  }
  for (const auto& ute : participatingEntries) {
    std::unordered_map<unsigned int, bool> qubitValuesThisEntry;
    llvm::DenseMap<Value, int64_t> intValuesThisEntry;
    llvm::DenseMap<Value, double> doubleValuesThisEntry;
    for (const auto& [qV, qBool] : qubitValues) {
      if (ute.participatingQubits.contains(qV)) {
        qubitValuesThisEntry[qubitsToGlobalIndices.at(qV)] = qBool;
      }
    }
    for (const auto& [iV, number] : classicalIntegerValues) {
      if (ute.participatingClassicalValues.contains(iV)) {
        intValuesThisEntry[iV] = number;
      }
    }
    for (const auto& [dV, number] : classicalDoubleValues) {
      if (ute.participatingClassicalValues.contains(dV)) {
        doubleValuesThisEntry[dV] = number;
      }
    }
    bool oneEntryIsNonzero = false;
    for (const auto& hs : ute.states) {
      if (!hs.hasAlwaysZeroProbability(qubitValuesThisEntry, intValuesThisEntry,
                                       doubleValuesThisEntry)) {
        oneEntryIsNonzero = true;
        break;
      }
    }
    if (!oneEntryIsNonzero) {
      return true;
    }
  }
  return false;
}

llvm::DenseMap<Value, bool>
UnionTable::getValueThatIsEquivalentToQubit(Value qubit) const {
  const auto uteOfQubit = valuesToEntries.at(qubit);
  const auto indexOfQubit = qubitsToGlobalIndices.at(qubit);
  llvm::DenseMap<Value, bool> result;
  bool found = false;
  bool alwaysOne = true;
  bool alwaysZero = true;
  for (const auto& hs : uteOfQubit->states) {
    alwaysOne &= hs.isQubitAlwaysOne(indexOfQubit);
    alwaysZero &= hs.isQubitAlwaysZero(indexOfQubit);
    auto currentResult = hs.getValueThatIsEquivalentToQubit(indexOfQubit);
    if (currentResult.empty() || (result.empty() && found)) {
      result.clear();
      continue;
    }
    if (!found) {
      result = currentResult;
      found = true;
    } else {
      for (const auto& [value, inverse] : currentResult) {
        if (!result.contains(value) || result.at(value) != inverse) {
          result.erase(value);
        }
      }
    }
  }
  if (alwaysOne) {
    auto valuesThatAreAlwaysTrue = getClassicalValuesThatAreAlwaysTrueOrFalse();
    result.reserve(valuesThatAreAlwaysTrue.size());
    for (const auto& [k, v] : valuesThatAreAlwaysTrue) {
      result[k] = v;
    }
  } else if (alwaysZero) {
    auto valuesThatAreAlwaysTrue = getClassicalValuesThatAreAlwaysTrueOrFalse();
    result.reserve(valuesThatAreAlwaysTrue.size());
    for (const auto& [k, v] : valuesThatAreAlwaysTrue) {
      result[k] = !v;
    }
  }
  return result;
}

std::optional<double> UnionTable::globalPhaseThatIsAdded(
    Operation* diagonalOp, std::span<Value> targets,
    std::span<Value> ctrlsQuantum, std::span<Value> posCtrlsClassical,
    std::span<Value> negCtrlsClassical) {}

SuperfluousResult UnionTable::getSuperfluousControls(
    std::span<Value> qubitTargets, std::span<Value> qubitCtrls,
    std::span<Value> posCtrlsClassical, std::span<Value> negCtrlsClassical) {}

bool UnionTable::areThereSatisfiableCombinations(
    std::span<Value> qubitCtrls, std::span<Value> posCtrlsClassical,
    std::span<Value> negCtrlsClassical) {}
std::pair<std::set<Value>, std::set<Value>>

UnionTable::getAntecedentsOfQubit(Value q, std::span<Value> qubits,
                                  std::span<Value> classicalPositive,
                                  std::span<Value> classicalNegative) {}

} // namespace mlir::qco

#endif // MQT_CORE_UNIONTABLE
