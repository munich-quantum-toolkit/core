/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#ifndef MQT_CORE_UNIONTABLE_H
#define MQT_CORE_UNIONTABLE_H
#include "HybridState.hpp"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <mlir/IR/Value.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mlir::qco {

/**
 * @brief Result of the check for superfluous values.
 */
struct SuperfluousResult {
  bool completelySuperfluous = false;
  llvm::DenseSet<Value> superfluousQubits;
  llvm::DenseSet<Value> superfluousClassicalValues;
};

/**
 * @brief Managing entries of the union table
 */
struct UnionTableEntry {
  const unsigned int index;
  bool top = false;
  std::vector<HybridState> states;
  // Values and global indices of the participating qubits
  llvm::DenseSet<Value> participatingQubits = {};
  llvm::DenseSet<Value> participatingClassicalValues = {};

  bool operator<(const UnionTableEntry& ute) const noexcept {
    return index < ute.index;
  }

  bool operator==(const UnionTableEntry& ute) const noexcept {
    return index == ute.index;
  }

  UnionTableEntry() : index(nextId()) {}

private:
  static std::uint64_t nextId() {
    static unsigned int counter = 0;
    return ++counter;
  }
};

/**
 * @brief This class represents a union table.
 *
 * This class holds multiple hybrid states and can propagate operations on the
 * values in the states.
 */
class UnionTable {
  bool allTop = false;
  std::size_t maxNonzeroAmplitudes;
  std::size_t maximumHybridEntries;
  llvm::DenseMap<Value, std::shared_ptr<UnionTableEntry>> valuesToEntries =
      llvm::DenseMap<Value, std::shared_ptr<UnionTableEntry>>();
  std::set<std::shared_ptr<UnionTableEntry>> entries;
  llvm::DenseMap<Value, unsigned int> qubitsToGlobalIndices;

  /** @brief: Collects a set of all participating entries.
   *
   * @param targets An array of the Values of the target qubits.
   * @param ctrlsQuantum An array of the values of the ctrl qubits.
   * @param posCtrlsClassical An array of the values of the ctrl bits.
   * @param negCtrlsClassical An array of the values of the negative ctrl bits.
   * @param params The parameter applied to the gate.
   */
  std::set<UnionTableEntry>
  collectParticipatingEntries(const std::span<Value> targets,
                              const std::span<Value> ctrlsQuantum,
                              const std::span<Value> posCtrlsClassical,
                              const std::span<Value> negCtrlsClassical,
                              const std::span<Value> params = {}) {
    std::set<UnionTableEntry> participatingEntries;
    for (auto const q : targets) {
      participatingEntries.insert(*valuesToEntries.at(q));
    }
    for (auto const q : ctrlsQuantum) {
      participatingEntries.insert(*valuesToEntries.at(q));
    }
    for (auto const i : posCtrlsClassical) {
      participatingEntries.insert(*valuesToEntries.at(i));
    }
    for (auto const i : negCtrlsClassical) {
      participatingEntries.insert(*valuesToEntries.at(i));
    }
    for (auto const i : params) {
      participatingEntries.insert(*valuesToEntries.at(i));
    }
    return participatingEntries;
  }

  /** @brief Puts the given UnionTableEntries to top
   *
   * @param entriesToTop The UnionTableEntries to become top.
   */
  void putEntriesToTop(const std::set<UnionTableEntry>& entriesToTop) {
    auto topUnionTableEntry = UnionTableEntry();
    topUnionTableEntry.top = true;
    for (const auto& e : entriesToTop) {
      topUnionTableEntry.participatingQubits.insert(
          e.participatingQubits.begin(), e.participatingQubits.end());
      topUnionTableEntry.participatingClassicalValues.insert(
          e.participatingClassicalValues.begin(),
          e.participatingClassicalValues.end());
      auto it = std::ranges::find_if(
          entries, [&](auto const& entry) { return *entry == e; });
      if (it != entries.end()) {
        entries.erase(it);
      }
    }
    const auto ptrUte = std::make_shared<UnionTableEntry>(topUnionTableEntry);
    entries.insert(ptrUte);
    for (auto q : ptrUte->participatingQubits) {
      valuesToEntries.erase(q);
      valuesToEntries[q] = ptrUte;
    }
    for (auto c : ptrUte->participatingClassicalValues) {
      valuesToEntries.erase(c);
      valuesToEntries[c] = ptrUte;
    }
  }

  /**
   * @brief This method unifies the given UnionTableEntries.
   *
   * This method unifies the given UnionTableEntries. If the new states have
   * more than maxNonzeroAmplitudes, it throws a domain_error. The same holds
   * if the resulting hybridStates are more than maximumHybridEntries.
   *
   * @param entriesToUnify The UnionTableEntries to be unified.
   * @throws domain_error If more than maNonzeroAmplitudes are created in a
   * quantumstate or more than maximumHybridEntries are created.
   */
  void unifyEntries(const std::set<UnionTableEntry>& entriesToUnify) {
    if (entriesToUnify.size() == 1) {
      return;
    }
    bool entriesBecomeTop = false;
    for (const auto& e : entriesToUnify) {
      entriesBecomeTop |= e.top;
    }

    // Check if the number of entries would be too large
    unsigned int numberOfNewEntries = 1;
    for (const auto& e : entriesToUnify) {
      numberOfNewEntries *= e.states.size();
    }
    if (numberOfNewEntries > maximumHybridEntries) {
      throw std::domain_error("Maximum of allowed hybrid entries exceeded.");
    }

    // Create new entry
    auto newEntry = UnionTableEntry();
    newEntry.top = entriesBecomeTop;
    for (const auto& e : entriesToUnify) {
      auto classicalValues = e.participatingClassicalValues;
      auto qubits = e.participatingQubits;
      newEntry.participatingClassicalValues.insert(classicalValues.begin(),
                                                   classicalValues.end());
      newEntry.participatingQubits.insert(qubits.begin(), qubits.end());
      if (!newEntry.top & newEntry.states.empty()) {
        newEntry.states = e.states;
        continue;
      }
      if (newEntry.top || e.states.empty()) {
        continue;
      }
      std::vector<HybridState> unifiedHS = {};
      for (auto hs1 : newEntry.states) {
        for (auto hs2 : e.states) {
          unifiedHS.push_back(hs1.unify(hs2));
        }
      }
      newEntry.states = unifiedHS;
    }

    // Adapt global data structures to new entry
    auto valuesToReplace = llvm::DenseSet<Value>();
    for (const auto& [v, _] : valuesToEntries) {
      if (newEntry.participatingQubits.contains(v) ||
          newEntry.participatingClassicalValues.contains(v)) {
        valuesToReplace.insert(v);
      }
    }
    const auto ptrUTE = std::make_shared<UnionTableEntry>(newEntry);
    for (const auto& v : valuesToReplace) {
      valuesToEntries[v] = ptrUTE;
    }
    for (const auto& e : entriesToUnify) {
      auto it = std::ranges::find_if(
          entries, [&](auto const& entry) { return *entry == e; });
      if (it != entries.end()) {
        entries.erase(it);
      }
    }
    entries.insert(ptrUTE);
  }

  /**
   * @brief This method applies a swap gate by switching two qubits if the
   * qubits are in different entries.
   *
   * @param targets The values that partake in the swap.
   * @param newQuantumTargets The values after the swap.
   */
  void applySwapGate(const std::span<Value> targets,
                     const std::span<Value> newQuantumTargets) {
    for (const auto& hs : valuesToEntries.at(targets[0])->states) {
      hs.changeGlobalIndex(qubitsToGlobalIndices.at(targets[0]),
                           qubitsToGlobalIndices.at(targets[1]));
    }
    for (const auto& hs : valuesToEntries.at(targets[1])->states) {
      hs.changeGlobalIndex(qubitsToGlobalIndices.at(targets[1]),
                           qubitsToGlobalIndices.at(targets[0]));
    }
    const auto targetOneIndex = qubitsToGlobalIndices.at(targets[0]);
    qubitsToGlobalIndices[targets[0]] = qubitsToGlobalIndices.at(targets[1]);
    qubitsToGlobalIndices[targets[1]] = targetOneIndex;
    std::ranges::reverse(newQuantumTargets);
    replaceValuesGlobally(targets, newQuantumTargets);
  }

  /**
   * @brief This method returns classical values which are always either true or
   * false.
   *
   * @return A map of values as keys. The values of th emap say wether the
   * classical values are always true or always false.
   */
  llvm::DenseMap<Value, bool>
  getClassicalValuesThatAreAlwaysTrueOrFalse() const {
    llvm::DenseMap<Value, bool> result;
    for (auto const& e : entries) {
      for (auto const& v : e->participatingClassicalValues) {
        for (auto const& hs : e->states) {
          const auto isTrue = hs.isValueTrue(v);
          if (result.contains(v) && result.at(v) != isTrue) {
            result.erase(v);
            break;
          }
          result[v] = isTrue;
        }
      }
    }
    return result;
  }

public:
  explicit UnionTable(std::size_t maxNonzeroAmplitudes,
                      std::size_t maximumHybridEntries);

  ~UnionTable();

  void print(std::ostream& os) const;

  [[nodiscard("UnionTable::toString called but ignored")]]
  std::string toString() const;

  /** @brief: Replaces values globally by new values
   *
   * @param replacedValues Values to be replaced
   * @param newValues Values the first values are replaced with.
   * @throws runtime_error if the size of the two parameters is not equal.
   */
  void replaceValuesGlobally(std::span<Value> replacedValues,
                             std::span<Value> newValues);

  [[nodiscard("UnionTable::allTop called but ignored")]]
  bool areStatesAllTop();

  /**
   * @brief This method applies a gate to the qubits.
   *
   * This method changes the amplitudes of a QuantumState according to the
   * applied gate.
   *
   * @param gate The gate to be applied.
   * @param targets An array of the Values of the target qubits.
   * @param newQuantumTargets The value of the qubits after the gate.
   * @param ctrlsQuantum An array of the values of the ctrl qubits.
   * @param newCtrlsQuantum An values of the ctrl qubits after the gate.
   * @param posCtrlsClassical An array of the values of the ctrl bits.
   * @param negCtrlsClassical An array of the values of the negative ctrl bits.
   * @param params The parameter applied to the gate.
   * @throws invalid_argument if a value is given, but is not found in the
   * existing ones.
   */
  void propagateGate(Operation* gate, std::span<Value> targets,
                     std::span<Value> newQuantumTargets,
                     std::span<Value> ctrlsQuantum = {},
                     std::span<Value> newCtrlsQuantum = {},
                     std::span<Value> posCtrlsClassical = {},
                     std::span<Value> negCtrlsClassical = {},
                     std::span<Value> params = {});

  /**
   * @brief This method propagates a classical operation.
   *
   *
   * @param op The operation to be applied.
   * @param targets An array of the Values of the target classical values.
   * @param results The value of the result.
   * @param posCtrlsClassical An array of the values of the ctrl bits.
   * @param negCtrlsClassical An array of the values of the negative ctrl bits.
   * @throws invalid_argument if a value is given, but is not found in the
   * existing ones.
   */
  void propagateClassicalOperation(Operation* op, std::span<Value> targets,
                                   std::span<Value> results,
                                   std::span<Value> posCtrlsClassical = {},
                                   std::span<Value> negCtrlsClassical = {});

  /**
   * @brief This method applies a measurement.
   *
   * This method applies a measurement, changing the qubits and the classical
   * bit corresponding to the measurement.
   *
   * @param quantumTarget The value of the qubit to be measured.
   * @param newQuantumValue The value of the qubit after the measurement.
   * @param classicalTarget The value of the bit to save the measurement result
   * in.
   * @param posCtrlsClassical An array of the values of the ctrl bits.
   * @param negCtrlsClassical An array of the values of the negative ctrl bits.
   * @throws invalid_argument if a value is given, but is not found in the
   * existing ones. This does not hold for the classical target, which can be
   * newly created.
   */
  void propagateMeasurement(Value quantumTarget, Value newQuantumValue,
                            Value classicalTarget,
                            std::span<Value> posCtrlsClassical = {},
                            std::span<Value> negCtrlsClassical = {});

  /**
   * @brief This method propagates a qubit reset.
   *
   * This method propagates a qubit reset. This means that the qubit is put into
   * zero state. It is also put in its own QubitState again if it does not
   * correspond to already assigned bit values.
   *
   * @param quantumTarget The value of the qubit to be reset.
   * @param newQuantumValue The value of the qubit after the reset.
   * @param posCtrlsClassical An array of the values of the ctrl bits.
   * @param negCtrlsClassical An array of the values of the negative ctrl bits.
   * @throws invalid_argument if a value is given, but is not found in the
   * existing ones.
   */
  void propagateReset(Value quantumTarget, Value newQuantumValue,
                      std::span<Value> posCtrlsClassical = {},
                      std::span<Value> negCtrlsClassical = {});

  /**
   * @brief This method propagates a qubit alloc.
   *
   * This method propagates a qubit alloc. This means that the qubit is added to
   * the UnionTable in zero state.
   *
   * @param qubit The value of the qubit to be allocated.
   */
  void propagateQubitAlloc(Value qubit);

  /**
   * @brief This method propagates an int alloc.
   *
   * This method propagates an int alloc. This means that the int is added to
   * the UnionTable as new HybridState.
   *
   * @param intValue The value of the int to be allocated.
   * @param number The number that the int is initialized with.
   */
  void propagateIntAlloc(Value intValue, int64_t number);

  /**
   * @brief This method propagates a double alloc.
   *
   * This method propagates a double alloc. This means that the double is added
   * to the UnionTable as new HybridState.
   *
   * @param doubleValue The value of the double to be allocated.
   * @param number The number that the double is initialized with.
   */
  void propagateDoubleAlloc(Value doubleValue, double number);

  [[nodiscard("UnionTable::isQubitAlwaysOne called but ignored")]] bool
  isQubitAlwaysOne(Value q) const;

  [[nodiscard("UnionTable::isQubitAlwaysZero called but ignored")]] bool
  isQubitAlwaysZero(Value q) const;

  [[nodiscard(
      "UnionTable::isClassicalValueAlwaysTrue called but ignored")]] bool
  isClassicalValueAlwaysTrue(Value c) const;

  [[nodiscard(
      "UnionTable::isClassicalValueAlwaysFalse called but ignored")]] bool
  isClassicalValueAlwaysFalse(Value c) const;

  /**
   * @brief Checks if a given combination of values-qubit values has a nonzero
   * probability.
   *
   * This method receives a number of qubit and values and checks whether
   * they have for a given value always a zero amplitude.
   * The values for the classical values are not the numeric ones, but whether
   * they are zero (false) or non-zero (true).
   *
   * @param qubitValues Pairs of the qubits that are being checked and the
   * values that they are being checked for.
   * @param classicalValues The classical values to check.
   * @throws invalid_argument if a value is given, but is not found in the
   * existing ones.
   * @returns True if the amplitude is always zero, false otherwise.
   */
  [[nodiscard("HybridState::hasAlwaysZeroAmplitude called but ignored")]] bool
  hasAlwaysZeroProbability(
      const llvm::DenseMap<Value, bool>& qubitValues,
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
   * The value of the map is true, if the qubit is equivalent to the value.
   * False, if the qubit is the inverse of the value.
   */
  [[nodiscard(
      "UnionTable::getValueThatIsEquivalentToQubit called but ignored")]] llvm::
      DenseMap<Value, bool>
      getValueThatIsEquivalentToQubit(Value qubit) const;

  /**
   * @brief This method checks whether a diagonal gate only adds a global phase
   * and returns it.
   *
   * This method receives a diagonal gate and checks, if only a global phase is
   * added to the circuit by it under the current configuration. If that is the
   * case, the returned optional contains the global phase. Only works with
   * 1-qubit gates without parameters.
   *
   * @param op The gate to be checked.
   * @param target The Values of the target qubits.
   * @param ctrlsQuantum An array of the values of the ctrl qubits.
   * @param posCtrlsClassical An array of the values of the ctrl bits.
   * @param negCtrlsClassical An array of the values of the negative ctrl bits.
   * @throws invalid_argument if a value is given, but is not found in the
   * existing ones.
   * @returns An optional containing the globally added value, if applicable.
   */
  [[nodiscard("UnionTable::globalPhaseThatIsAdded called but ignored")]]
  std::optional<double>
  globalPhaseThatIsAdded(Operation* op, Value target,
                         std::span<Value> ctrlsQuantum = {},
                         std::span<Value> posCtrlsClassical = {},
                         std::span<Value> negCtrlsClassical = {});

  /**
   * @brief This method checks which qubits and classical values are superfluous
   * given a controlled gate.
   *
   * This method checks which qubits and classical values are superfluous given
   * a controlled gate. If the gate can never be executed, the SuperfluousResult
   * will indicate that the gate is completely superfluous. Apart from that, all
   * posCtrl (negCtrl) qubits/values that are always true (false) are
   * superfluous.
   *
   * @param qubitCtrls The valuess of the positively controlling qubits.
   * @param posCtrlsClassical The values of the positively controlling classical
   * values.
   * @param negCtrlsClassical The values of the negatively controlling classical
   * values.
   * @returns The superfluous result, i.e. the qubits and classical values that
   * are superfluous and whether the whole operation is superfluous.
   */
  SuperfluousResult
  getSuperfluousControls(std::span<Value> qubitCtrls,
                         std::span<Value> posCtrlsClassical = {},
                         std::span<Value> negCtrlsClassical = {});

  /**
   * @brief This method checks whether there are satisfiable combinations of
   * controls.
   *
   * @param qubitCtrls The values of the controlling qubit values.
   * @param posCtrlsClassical The values of the positively controlling classical
   * values.
   * @param negCtrlsClassical The values of the negatively controlling classical
   * values.
   * @returns Whether there are satisfiable combinations or not.
   */
  bool areThereSatisfiableCombinations(
      std::span<Value> qubitCtrls, std::span<Value> posCtrlsClassical = {},
      std::span<Value> negCtrlsClassical = {}) const;

  /**
   * @brief Returns whether the given qubits and classical values that imply the
   * given qubit.
   *
   * This method checks whether in the given list are qubits or classical values
   * that imply (are antecedents of) the given qubit. I.e. if there are
   * qubits/values a for which holds: a -> q.
   *
   * @param q The qubit for which is checked whether it is implied.
   * @param qubits The qubits for which are checked if they imply q.
   * @param classicalPositive The values for which are checked if they imply q.
   * @param classicalNegative The values for which their negations are checked
   * if they imply q.
   * @returns A pair of 1. qubits and 2. classical values that are antecedents
   * of q.
   */
  bool isQubitImplied(Value q, std::span<Value> qubits,
                      std::span<Value> classicalPositive,
                      std::span<Value> classicalNegative) const;
};
} // namespace mlir::qco

#endif // MQT_CORE_UNIONTABLE_H
