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
    for (const auto& [_, q] : entry.participatingQubits) {
      qubitIndices.push_back(q);
    }
    std::ranges::sort(qubitIndices, std::greater<int>());
    result += "Qubits: ";
    for (const auto qit : qubitIndices) {
      result += std::to_string(qit);
    }
    result += ", HybridStates: {";
    bool first = true;
    for (HybridState const& hs : entry.states) {
      if (!first) {
        result += " ";
      }
      first = false;
      result += hs.toString();
    }
    result += "}\n";
  }
  return result;
}

bool UnionTable::areStatesAllTop() const { return allTop; }

void UnionTable::propagateGate(Operation* gate, std::span<Value> targets,
                               std::span<Value> newQuantumTargets,
                               std::span<Value> ctrlsQuantum,
                               std::span<Value> posCtrlsClassical,
                               std::span<Value> negCtrlsClassical,
                               std::vector<Value> params) {
  if (isa<SWAPOp>(*gate) && ctrlsQuantum.empty() && posCtrlsClassical.empty() &&
      negCtrlsClassical.empty()) {
    applySwapGate(targets, newQuantumTargets);
    return;
  }

  const std::set<UnionTableEntry> participatingEntries =
      collectParticipatingEntries(targets, ctrlsQuantum, posCtrlsClassical,
                                  negCtrlsClassical, params);

  try {
    unifyEntries(participatingEntries);
  } catch (std::domain_error&) {
    putEntriesToTop(participatingEntries);
    replaceValuesGlobally(targets, newQuantumTargets);
    return;
  }

  auto ute = valuesToEntries.at(*targets.begin());
  std::vector<unsigned int> targetQubitIndices;
  std::vector<unsigned int> ctrlQubitIndices;
  for (auto const q : targets) {
    targetQubitIndices.push_back(ute->participatingQubits.at(q));
  }
  for (auto const q : ctrlsQuantum) {
    ctrlQubitIndices.push_back(ute->participatingQubits.at(q));
  }

  for (auto hs : ute->states) {
    try {
      hs.propagateGate(gate, targetQubitIndices, ctrlQubitIndices,
                       posCtrlsClassical, negCtrlsClassical, params);
    } catch (std::domain_error&) {
      putEntriesToTop(participatingEntries);
      break;
    }
  }
  replaceValuesGlobally(targets, newQuantumTargets);
}

void UnionTable::propagateMeasurement(Value quantumTarget,
                                      Value newQuantumValue,
                                      Value classicalTarget,
                                      std::span<Value> posCtrlsClassical,
                                      std::span<Value> negCtrlsClassical) {}

void UnionTable::propagateReset(Value quantumTarget, Value newQuantumValue,
                                std::span<Value> posCtrlsClassical,
                                std::span<Value> negCtrlsClassical) {}

void UnionTable::propagateQubitAlloc(Value qubit) {}

void UnionTable::propagateIntAlloc(Value intValue, int64_t number) {}

void UnionTable::propagateDoubleAlloc(Value doubleValue, double number) {}

bool UnionTable::isQubitAlwaysOne(Value q) const {}

bool UnionTable::isQubitAlwaysZero(Value q) const {}

bool UnionTable::isClassicalValueAlwaysTrue(Value c) const {}

bool UnionTable::isClassicalValueAlwaysFalse(Value c) const {}

bool UnionTable::hasAlwaysZeroProbability(
    std::span<Value> qubits, unsigned int qubitValue,
    std::span<std::pair<Value, int64_t>> classicalIntegerValues,
    std::span<std::pair<Value, double>> classicalDoubleValues) const {}

std::pair<std::optional<Value>, bool>
UnionTable::getValueThatIsEquivalentToQubit(unsigned int qubit) const {}

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

UnionTable::getAntecedentsOfQubit(unsigned int q, std::span<Value> qubits,
                                  std::span<Value> classicalPositive,
                                  std::span<Value> classicalNegative) {}

} // namespace mlir::qco

#endif // MQT_CORE_UNIONTABLE
