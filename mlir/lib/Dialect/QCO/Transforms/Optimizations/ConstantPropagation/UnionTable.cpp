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

#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Optimizations/ConstantPropagation/HybridState.hpp"

#include <llvm/ADT/DenseMap.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numbers>
#include <optional>
#include <ostream>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlir::qco {

UnionTable::UnionTable(const std::size_t maxNonzeroAmplitudes,
                       const std::size_t maximumHybridEntries)
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
    std::ranges::sort(qubitIndices, std::greater{});
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

void UnionTable::replaceValuesGlobally(const std::span<Value> replacedValues,
                                       const std::span<Value> newValues) {
  if (replacedValues.size() != newValues.size()) {
    throw std::domain_error(
        "replacedValues and newValues do not have the same size.");
  }

  for (unsigned int i = 0; i < replacedValues.size(); ++i) {
    const auto rV = replacedValues[i];
    const auto nV = newValues[i];
    qubitsToGlobalIndices[nV] = qubitsToGlobalIndices.at(rV);
    qubitsToGlobalIndices.erase(rV);
    const auto ute = valuesToEntries.at(rV);
    valuesToEntries.erase(rV);
    valuesToEntries[nV] = ute;
    if (ute->participatingQubits.contains(rV)) {
      ute->participatingQubits.insert(nV);
      ute->participatingQubits.erase(rV);
    } else {
      ute->participatingClassicalValues.insert(nV);
      ute->participatingClassicalValues.erase(rV);
    }
  }
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
                               const std::span<Value> newCtrlsQuantum,
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
  replaceValuesGlobally(ctrlsQuantum, newCtrlsQuantum);
}

void UnionTable::propagateClassicalOperation(
    Operation* op, std::span<Value> targets, std::span<Value> results,
    std::span<Value> posCtrlsClassical, std::span<Value> negCtrlsClassical) {
  const auto result = results[0];
  if (!valuesToEntries.contains(result)) {
    if (op->getResult(0).getType().isFloat()) {
      propagateDoubleAlloc(result, 0.0);
    } else {
      propagateIntAlloc(result, 0);
    }
  }
  const std::set<UnionTableEntry> participatingEntries =
      collectParticipatingEntries(targets, results, posCtrlsClassical,
                                  negCtrlsClassical);

  try {
    unifyEntries(participatingEntries);
  } catch (std::domain_error&) {
    putEntriesToTop(participatingEntries);
    return;
  }
  const Value operand1 = targets[0];
  const Value operand2 = targets.size() > 1 ? targets[1] : nullptr;
  const Value operand3 = targets.size() > 2 ? targets[2] : nullptr;

  const auto ute = valuesToEntries.at(*targets.begin());
  for (auto& hs : ute->states) {
    hs.propagateClassicalOperation(op, result, operand1, operand2, operand3,
                                   posCtrlsClassical, negCtrlsClassical);
    if (hs.isHybridStateTop()) {
      putEntriesToTop({*ute});
      break;
    }
  }
}

void UnionTable::propagateMeasurement(
    const Value quantumTarget, const Value newQuantumValue,
    const Value classicalTarget, const std::span<Value> posCtrlsClassical,
    const std::span<Value> negCtrlsClassical) {

  if (!valuesToEntries.contains(classicalTarget)) {
    propagateIntAlloc(classicalTarget, 0);
  }

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
  return std::ranges::all_of(ute->states, [&](const auto& hs) {
    return hs.isQubitAlwaysOne(qubitIndex);
  });
}

bool UnionTable::isQubitAlwaysZero(const Value q) const {
  const unsigned int qubitIndex = qubitsToGlobalIndices.at(q);
  const auto ute = valuesToEntries.at(q);
  return std::ranges::all_of(ute->states, [&](const auto& hs) {
    return hs.isQubitAlwaysZero(qubitIndex);
  });
}

bool UnionTable::isClassicalValueAlwaysTrue(const Value c) const {
  const auto ute = valuesToEntries.at(c);
  return std::ranges::all_of(ute->states,
                             [&](const auto& hs) { return hs.isValueTrue(c); });
}

bool UnionTable::isClassicalValueAlwaysFalse(const Value c) const {
  const auto ute = valuesToEntries.at(c);
  return std::ranges::all_of(
      ute->states, [&](const auto& hs) { return !hs.isValueTrue(c); });
}

bool UnionTable::hasAlwaysZeroProbability(
    const llvm::DenseMap<Value, bool>& qubitValues,
    const llvm::DenseMap<Value, bool>& classicalValues) const {
  std::set<UnionTableEntry> participatingEntries;
  for (const auto& [qV, _] : qubitValues) {
    participatingEntries.insert(*valuesToEntries.at(qV));
  }
  for (const auto& [cV, _] : classicalValues) {
    participatingEntries.insert(*valuesToEntries.at(cV));
  }
  for (const auto& ute : participatingEntries) {
    std::unordered_map<unsigned int, bool> qubitValuesThisEntry;
    llvm::DenseMap<Value, bool> classicalValuesThisEntry;
    for (const auto& [qV, qBool] : qubitValues) {
      if (ute.participatingQubits.contains(qV)) {
        qubitValuesThisEntry[qubitsToGlobalIndices.at(qV)] = qBool;
      }
    }
    for (const auto& [v, vBool] : classicalValues) {
      if (ute.participatingClassicalValues.contains(v)) {
        classicalValuesThisEntry[v] = vBool;
      }
    }
    bool oneEntryIsNonzero = false;
    for (const auto& hs : ute.states) {
      if (!hs.hasAlwaysZeroProbability(qubitValuesThisEntry,
                                       classicalValuesThisEntry)) {
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
UnionTable::getValueThatIsEquivalentToQubit(const Value qubit) const {
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

std::optional<double>
UnionTable::globalPhaseThatIsAdded(Operation* op, const Value target,
                                   const std::span<Value> ctrlsQuantum,
                                   const std::span<Value> posCtrlsClassical,
                                   const std::span<Value> negCtrlsClassical) {
  // Diagonal gates w/o parameters: IdOp, ZOp, SOp, SdgOp, TOp, TdgOp
  if (isa<IdOp>(op)) {
    return {0.0};
  }
  if (!(isa<ZOp>(op) || isa<SOp>(op) || isa<SdgOp>(op) || isa<TOp>(op) ||
        isa<TdgOp>(op))) {
    return {};
  }
  const auto targetIndex = qubitsToGlobalIndices.at(target);
  const auto targetUte = valuesToEntries.at(target);
  bool alwaysOne = true;
  bool alwaysZero = true;

  for (const auto& hs : targetUte->states) {
    alwaysOne &= hs.isQubitAlwaysOne(targetIndex);
    alwaysZero &= hs.isQubitAlwaysZero(targetIndex);
    if (!alwaysOne && !alwaysZero) {
      break;
    }
  }

  if (alwaysZero) {
    return {0.0};
  }
  if (!alwaysOne && ctrlsQuantum.empty() && posCtrlsClassical.empty() &&
      negCtrlsClassical.empty()) {
    return {};
  }

  const auto participatingEntries = collectParticipatingEntries(
      {}, ctrlsQuantum, posCtrlsClassical, negCtrlsClassical, {});

  // Check if state |11...11> can be reached
  bool highestStateReachable = alwaysOne;
  bool highestStateAlwaysReached = alwaysOne;
  for (const auto& ute : participatingEntries) {
    std::unordered_map<unsigned int, bool> qubitCtrlThisEntry;
    llvm::DenseMap<Value, bool> classicalCtrlThisEntry;
    for (const auto q : ctrlsQuantum) {
      if (ute.participatingQubits.contains(q)) {
        qubitCtrlThisEntry[qubitsToGlobalIndices.at(q)] = true;
      }
    }
    for (const auto c : posCtrlsClassical) {
      if (ute.participatingClassicalValues.contains(c)) {
        classicalCtrlThisEntry[c] = true;
      }
    }
    for (const auto c : negCtrlsClassical) {
      if (ute.participatingClassicalValues.contains(c)) {
        classicalCtrlThisEntry[c] = false;
      }
    }
    for (const auto& hs : ute.states) {
      const auto ctrlsZeroProbability = hs.hasAlwaysZeroProbability(
          qubitCtrlThisEntry, classicalCtrlThisEntry);
      highestStateReachable |= !ctrlsZeroProbability;
      highestStateAlwaysReached &= !ctrlsZeroProbability;
    }
    if (highestStateReachable && !highestStateAlwaysReached) {
      return {};
    }
  }
  if (!highestStateReachable) {
    return {0.0};
  }
  if (!highestStateAlwaysReached) {
    return {};
  }

  // Only highest state reachable, return respective phase
  if (isa<ZOp>(op)) {
    return {std::numbers::pi};
  }
  if (isa<SOp>(op)) {
    return {std::numbers::pi / 2};
  }
  if (isa<SdgOp>(op)) {
    return {3.0 * std::numbers::pi / 2};
  }
  if (isa<TOp>(op)) {
    return {std::numbers::pi / 4};
  }
  // Tdg Op
  return {-std::numbers::pi / 2};
}

SuperfluousResult
UnionTable::getSuperfluousControls(const std::span<Value> qubitCtrls,
                                   const std::span<Value> posCtrlsClassical,
                                   const std::span<Value> negCtrlsClassical) {
  SuperfluousResult res;
  for (const auto& qCtrl : qubitCtrls) {
    const auto qIndex = qubitsToGlobalIndices.at(qCtrl);
    bool alwaysOne = true;
    bool alwaysZero = true;
    for (const auto& hs : valuesToEntries.at(qCtrl)->states) {
      if (alwaysZero && !hs.isQubitAlwaysZero(qIndex)) {
        alwaysZero = false;
      }
      alwaysOne &= !alwaysZero && hs.isQubitAlwaysOne(qIndex);
    }
    if (alwaysZero) {
      res.completelySuperfluous = true;
      return res;
    }
    if (alwaysOne) {
      res.superfluousQubits.insert(qCtrl);
    }
  }
  for (const auto& posCtrl : posCtrlsClassical) {
    bool alwaysTrue = true;
    bool alwaysFalse = true;
    for (const auto& hs : valuesToEntries.at(posCtrl)->states) {
      if (!alwaysTrue && !alwaysFalse) {
        break;
      }
      const bool valueTrue = hs.isValueTrue(posCtrl);
      alwaysTrue &= valueTrue;
      alwaysFalse &= !valueTrue;
    }
    if (alwaysFalse) {
      res.completelySuperfluous = true;
      return res;
    }
    if (alwaysTrue) {
      res.superfluousClassicalValues.insert(posCtrl);
    }
  }
  for (const auto& negCtrl : negCtrlsClassical) {
    bool alwaysTrue = true;
    bool alwaysFalse = true;
    for (const auto& hs : valuesToEntries.at(negCtrl)->states) {
      if (!alwaysTrue && !alwaysFalse) {
        break;
      }
      const bool valueTrue = hs.isValueTrue(negCtrl);
      alwaysTrue &= valueTrue;
      alwaysFalse &= !valueTrue;
    }
    if (alwaysTrue) {
      res.completelySuperfluous = true;
      return res;
    }
    if (alwaysFalse) {
      res.superfluousClassicalValues.insert(negCtrl);
    }
  }
  return res;
}

bool UnionTable::areThereSatisfiableCombinations(
    const std::span<Value> qubitCtrls, const std::span<Value> posCtrlsClassical,
    const std::span<Value> negCtrlsClassical) const {
  llvm::DenseMap<Value, bool> qubitValues;
  llvm::DenseMap<Value, bool> classicalValues;
  for (const auto& q : qubitCtrls) {
    qubitValues[q] = true;
  }
  for (const auto& v : posCtrlsClassical) {
    classicalValues[v] = true;
  }
  for (const auto& v : negCtrlsClassical) {
    classicalValues[v] = false;
  }
  return !hasAlwaysZeroProbability(qubitValues, classicalValues);
}

bool UnionTable::isQubitImplied(
    const Value q, const std::span<Value> qubits,
    const std::span<Value> classicalPositive,
    const std::span<Value> classicalNegative) const {
  llvm::DenseMap<Value, bool> qMap;
  llvm::DenseMap<Value, bool> cPMap;
  qMap[q] = false;
  for (const auto& qV : qubits) {
    qMap[qV] = true;
    if (hasAlwaysZeroProbability(qMap, cPMap)) {
      return true;
    }
    qMap.erase(qV);
  }
  for (const auto& cP : classicalPositive) {
    cPMap[cP] = true;
    if (hasAlwaysZeroProbability(qMap, cPMap)) {
      return true;
    }
    cPMap.erase(cP);
  }
  for (const auto& cP : classicalNegative) {
    cPMap[cP] = false;
    if (hasAlwaysZeroProbability(qMap, cPMap)) {
      return true;
    }
    cPMap.erase(cP);
  }
  return false;
}

} // namespace mlir::qco

#endif // MQT_CORE_UNIONTABLE
