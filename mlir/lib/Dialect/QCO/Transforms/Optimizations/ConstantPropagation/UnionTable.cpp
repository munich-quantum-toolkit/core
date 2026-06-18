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

namespace mlir::qco {

UnionTable::UnionTable(std::size_t maxNonzeroAmplitudes,
                       std::size_t maximumHybridEntries) {}

UnionTable::~UnionTable() {}

void UnionTable::print(std::ostream& os) const {}

std::string UnionTable::toString() const {}

bool UnionTable::areStatesAllTop() {}

void UnionTable::propagateGate(Operation* gate, std::span<Value> targets,
                               std::span<Value> newQuantumTargets,
                               std::span<Value> ctrlsQuantum,
                               std::span<Value> posCtrlsClassical,
                               std::span<Value> negCtrlsClassical,
                               std::vector<Value> params) {}

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
