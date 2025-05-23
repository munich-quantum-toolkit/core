/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "ir/Definitions.hpp"
#include "ir/Permutation.hpp"
#include "ir/Register.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/OpType.hpp"

#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace qc {
class Operation {
protected:
  Controls controls;
  Targets targets;
  std::vector<fp> parameter;

  OpType type = None;
  std::string name;

  static constexpr size_t OUTPUT_INDENT_SIZE = 2;

  static bool isWholeQubitRegister(const QubitIndexToRegisterMap& regMap,
                                   const Qubit start, const Qubit end) {
    const auto& startReg = regMap.at(start).first;
    const auto& endReg = regMap.at(end).first;
    return startReg == endReg && startReg.getStartIndex() == start &&
           endReg.getEndIndex() == end;
  }

public:
  Operation() = default;
  Operation(const Operation& op) = default;
  Operation(Operation&& op) noexcept = default;
  Operation& operator=(const Operation& op) = default;
  Operation& operator=(Operation&& op) noexcept = default;

  // Virtual Destructor
  virtual ~Operation() = default;

  [[nodiscard]] virtual std::unique_ptr<Operation> clone() const = 0;

  // Getters
  [[nodiscard]] virtual const Targets& getTargets() const { return targets; }
  virtual Targets& getTargets() { return targets; }
  [[nodiscard]] virtual std::size_t getNtargets() const {
    return targets.size();
  }

  [[nodiscard]] virtual const Controls& getControls() const { return controls; }
  virtual Controls& getControls() { return controls; }
  [[nodiscard]] virtual std::size_t getNcontrols() const {
    return controls.size();
  }
  [[nodiscard]] std::size_t getNqubits() const {
    return getUsedQubits().size();
  }

  [[nodiscard]] const std::vector<fp>& getParameter() const {
    return parameter;
  }
  std::vector<fp>& getParameter() { return parameter; }

  [[nodiscard]] const std::string& getName() const { return name; }
  [[nodiscard]] virtual OpType getType() const { return type; }

  [[nodiscard]] virtual auto
  getUsedQubitsPermuted(const Permutation& perm) const -> std::set<Qubit>;

  [[nodiscard]] auto getUsedQubits() const -> std::set<Qubit>;

  [[nodiscard]] std::unique_ptr<Operation> getInverted() const {
    auto op = clone();
    op->invert();
    return op;
  }

  // Setter
  virtual void setTargets(const Targets& t) { targets = t; }

  virtual void setControls(const Controls& c) {
    clearControls();
    addControls(c);
  }

  virtual void addControl(Control c) = 0;

  void addControls(const Controls& c) {
    for (const auto& control : c) {
      addControl(control);
    }
  }

  virtual void clearControls() = 0;

  virtual void removeControl(Control c) = 0;

  virtual Controls::iterator removeControl(Controls::iterator it) = 0;

  void removeControls(const Controls& c) {
    for (auto it = c.begin(); it != c.end();) {
      it = removeControl(it);
    }
  }

  virtual void setGate(const OpType g) {
    type = g;
    name = toString(g);
  }

  virtual void setParameter(const std::vector<fp>& p) { parameter = p; }

  virtual void apply(const Permutation& permutation);

  [[nodiscard]] virtual bool isUnitary() const { return true; }

  [[nodiscard]] virtual bool isStandardOperation() const { return false; }

  [[nodiscard]] virtual bool isCompoundOperation() const noexcept {
    return false;
  }

  [[nodiscard]] virtual bool isNonUnitaryOperation() const { return false; }

  [[nodiscard]] virtual bool isClassicControlledOperation() const noexcept {
    return false;
  }

  [[nodiscard]] virtual bool isSymbolicOperation() const { return false; }

  [[nodiscard]] virtual auto isDiagonalGate() const -> bool {
    // the second bit in the type is a flag that is set for diagonal gates
    return (+type & OpTypeDiag) != 0;
  }

  [[nodiscard]] virtual auto isSingleQubitGate() const -> bool {
    return !isControlled() && qc::isSingleQubitGate(type);
  }

  [[nodiscard]] virtual bool isControlled() const { return !controls.empty(); }

  [[nodiscard]] virtual bool isClifford() const { return false; }

  /**
   * @brief Checks whether a gate is global.
   * @details A StandardOperation is global if it acts on all qubits.
   * A CompoundOperation is global if all its sub-operations are
   * StandardOperations of the same type with the same parameters acting on all
   * qubits. The latter is what a QASM line like `ry(π) q;` is translated to in
   * MQT Core. All other operations are not global.
   * @return True if the operation is global, false otherwise.
   */
  [[nodiscard]] virtual bool isGlobal(size_t /* unused */) const {
    return false;
  }

  [[nodiscard]] virtual bool actsOn(const Qubit i) const {
    for (const auto& t : targets) {
      if (t == i) {
        return true;
      }
    }
    return controls.count(i) > 0;
  }

  virtual void addDepthContribution(std::vector<std::size_t>& depths) const;

  [[nodiscard]] virtual bool equals(const Operation& op,
                                    const Permutation& perm1,
                                    const Permutation& perm2) const;
  [[nodiscard]] virtual bool equals(const Operation& op) const {
    return equals(op, {}, {});
  }

  virtual std::ostream& printParameters(std::ostream& os) const;
  std::ostream& print(std::ostream& os, const std::size_t nqubits) const {
    return print(os, {}, 0, nqubits);
  }
  virtual std::ostream& print(std::ostream& os, const Permutation& permutation,
                              std::size_t prefixWidth,
                              std::size_t nqubits) const;

  void dumpOpenQASM2(std::ostream& of, const QubitIndexToRegisterMap& qubitMap,
                     const BitIndexToRegisterMap& bitMap) const {
    dumpOpenQASM(of, qubitMap, bitMap, 0, false);
  }
  void dumpOpenQASM3(std::ostream& of, const QubitIndexToRegisterMap& qubitMap,
                     const BitIndexToRegisterMap& bitMap) const {
    dumpOpenQASM(of, qubitMap, bitMap, 0, true);
  }
  virtual void dumpOpenQASM(std::ostream& of,
                            const QubitIndexToRegisterMap& qubitMap,
                            const BitIndexToRegisterMap& bitMap, size_t indent,
                            bool openQASM3) const = 0;

  /// Checks whether operation commutes with other operation on a given qubit.
  [[nodiscard]] virtual auto commutesAtQubit(const Operation& /*other*/,
                                             const Qubit& /*qubit*/) const
      -> bool {
    return false;
  }

  [[nodiscard]] virtual auto isInverseOf(const Operation& /*other*/) const
      -> bool;

  virtual void invert() = 0;

  virtual bool operator==(const Operation& rhs) const { return equals(rhs); }
  bool operator!=(const Operation& rhs) const { return !(*this == rhs); }
};
} // namespace qc

template <> struct std::hash<qc::Operation> {
  std::size_t operator()(const qc::Operation& op) const noexcept {
    std::size_t seed = 0U;
    qc::hashCombine(seed, hash<qc::OpType>{}(op.getType()));
    for (const auto& control : op.getControls()) {
      qc::hashCombine(seed, hash<qc::Qubit>{}(control.qubit));
      if (control.type == qc::Control::Type::Neg) {
        seed ^= 1ULL;
      }
    }
    for (const auto& target : op.getTargets()) {
      qc::hashCombine(seed, hash<qc::Qubit>{}(target));
    }
    for (const auto& param : op.getParameter()) {
      qc::hashCombine(seed, hash<qc::fp>{}(param));
    }
    return seed;
  }
};
