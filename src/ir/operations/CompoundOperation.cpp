/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/operations/CompoundOperation.hpp"

#include "ir/Definitions.hpp"
#include "ir/Permutation.hpp"
#include "ir/QuantumComputation.hpp"
#include "ir/Register.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/Operation.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <iterator>
#include <memory>
#include <ostream>
#include <ranges>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qc {
CompoundOperation::CompoundOperation(bool isCustom) : customGate(isCustom) {
  name = "Compound operation:";
  type = Compound;
}

CompoundOperation::CompoundOperation(
    std::vector<std::unique_ptr<Operation>>&& operations, bool isCustom)
    : CompoundOperation(isCustom) {
  // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
  ops = std::move(operations);
}

CompoundOperation::CompoundOperation(const CompoundOperation& co)
    : Operation(co), ops(co.ops.size()), customGate(co.customGate) {
  for (std::size_t i = 0; i < co.ops.size(); ++i) {
    ops[i] = co.ops[i]->clone();
  }
}

CompoundOperation& CompoundOperation::operator=(const CompoundOperation& co) {
  if (this != &co) {
    Operation::operator=(co);
    ops.resize(co.ops.size());
    for (std::size_t i = 0; i < co.ops.size(); ++i) {
      ops[i] = co.ops[i]->clone();
    }
    customGate = co.customGate;
  }
  return *this;
}

std::unique_ptr<Operation> CompoundOperation::clone() const {
  return std::make_unique<CompoundOperation>(*this);
}

std::size_t CompoundOperation::getNqubits() const {
  return getUsedQubits().size();
}

bool CompoundOperation::isNonUnitaryOperation() const {
  return std::ranges::any_of(
      ops, [](const auto& op) { return op->isNonUnitaryOperation(); });
}

bool CompoundOperation::isCompoundOperation() const noexcept { return true; }

bool CompoundOperation::isCustomGate() const noexcept { return customGate; }

bool CompoundOperation::isGlobal(const size_t nQubits) const noexcept {
  const auto& params = ops.front()->getParameter();
  const auto& t = ops.front()->getType();
  return getUsedQubits().size() == nQubits &&
         std::ranges::all_of(ops.cbegin() + 1, ops.cend(),
                             [&](const auto& operation) {
                               return operation->isStandardOperation() &&
                                      operation->getNcontrols() == 0 &&
                                      operation->getType() == t &&
                                      operation->getParameter() == params;
                             });
}

bool CompoundOperation::isSymbolicOperation() const {
  return std::ranges::any_of(
      ops, [](const auto& op) { return op->isSymbolicOperation(); });
}

void CompoundOperation::addControl(const Control c) {
  controls.insert(c);
  // we can just add the controls to each operation, as the operations will
  // check if they already act on the control qubits.
  for (const auto& op : ops) {
    op->addControl(c);
  }
}

void CompoundOperation::clearControls() {
  // we remove just our controls from nested operations
  removeControls(controls);
}

void CompoundOperation::removeControl(const Control c) {
  // first we iterate over our controls and check if we are actually allowed
  // to remove them
  if (controls.erase(c) == 0) {
    throw std::runtime_error(
        "Cannot remove control from compound operation as it "
        "is not a control.");
  }

  for (const auto& op : ops) {
    op->removeControl(c);
  }
}

Controls::iterator
CompoundOperation::removeControl(const Controls::iterator it) {
  for (const auto& op : ops) {
    op->removeControl(*it);
  }

  return controls.erase(it);
}
bool CompoundOperation::equals(const Operation& op, const Permutation& perm1,
                               const Permutation& perm2) const {
  if (const auto* comp = dynamic_cast<const CompoundOperation*>(&op)) {
    if (comp->ops.size() != ops.size()) {
      return false;
    }

    auto it = comp->ops.cbegin();
    for (const auto& operation : ops) {
      if (!operation->equals(**it, perm1, perm2)) {
        return false;
      }
      ++it;
    }
    return true;
  }
  return false;
}

bool CompoundOperation::equals(const Operation& operation) const {
  return equals(operation, {}, {});
}

std::ostream& CompoundOperation::print(std::ostream& os,
                                       const Permutation& permutation,
                                       const std::size_t prefixWidth,
                                       const std::size_t nqubits) const {
  const auto prefix = std::string(prefixWidth - 1, ' ');
  os << std::string(4 * nqubits, '-') << "\n";
  for (const auto& op : ops) {
    os << prefix << ":";
    op->print(os, permutation, prefixWidth, nqubits);
    os << "\n";
  }
  os << prefix << std::string((4 * nqubits) + 1, '-');
  return os;
}

bool CompoundOperation::actsOn(const Qubit i) const {
  return std::ranges::any_of(ops,
                             [&i](const auto& op) { return op->actsOn(i); });
}

void CompoundOperation::addDepthContribution(
    std::vector<std::size_t>& depths) const {
  for (const auto& op : ops) {
    op->addDepthContribution(depths);
  }
}

void CompoundOperation::dumpOpenQASM(std::ostream& of,
                                     const QubitIndexToRegisterMap& qubitMap,
                                     const BitIndexToRegisterMap& bitMap,
                                     const std::size_t indent,
                                     bool openQASM3) const {
  for (const auto& op : ops) {
    op->dumpOpenQASM(of, qubitMap, bitMap, indent, openQASM3);
  }
}

auto CompoundOperation::getUsedQubitsPermuted(const Permutation& perm) const
    -> std::set<Qubit> {
  std::set<Qubit> usedQubits{};
  for (const auto& op : ops) {
    usedQubits.merge(op->getUsedQubitsPermuted(perm));
  }
  return usedQubits;
}

auto CompoundOperation::commutesAtQubit(const Operation& other,
                                        const Qubit& qubit) const -> bool {
  return std::ranges::all_of(ops, [&other, &qubit](const auto& op) {
    return op->commutesAtQubit(other, qubit);
  });
}

auto CompoundOperation::isInverseOf(const Operation& other) const -> bool {
  if (other.isCompoundOperation()) {
    // cast other to CompoundOperation
    const auto& co = dynamic_cast<const CompoundOperation&>(other);
    if (size() != co.size()) {
      return false;
    }
    // here both compound operations have the same size
    if (empty()) {
      return true;
    }
    // transform compound to a QuantumComputation such that the invert method
    // and the reorderOperations method can be used to get a canonical form of
    // the compound operations
    const auto& thisUsedQubits = getUsedQubits();
    assert(!thisUsedQubits.empty());
    const auto thisMaxQubit = *std::ranges::max_element(thisUsedQubits);
    QuantumComputation thisQc(thisMaxQubit + 1);
    std::for_each(cbegin(), cend(),
                  [&](const auto& op) { thisQc.emplace_back(op->clone()); });
    const auto& otherUsedQubits = co.getUsedQubits();
    assert(!otherUsedQubits.empty());
    const auto otherMaxQubit = *std::ranges::max_element(otherUsedQubits);
    QuantumComputation otherQc(otherMaxQubit + 1);
    std::for_each(co.cbegin(), co.cend(),
                  [&](const auto& op) { otherQc.emplace_back(op->clone()); });
    thisQc.reorderOperations();
    otherQc.invert();
    otherQc.reorderOperations();
    return std::equal(
        thisQc.cbegin(), thisQc.cend(), otherQc.cbegin(),
        [](const auto& op1, const auto& op2) { return *op1 == *op2; });
  }
  return false;
}

void CompoundOperation::invert() {
  for (const auto& op : ops) {
    op->invert();
  }
  std::ranges::reverse(ops);
}

void CompoundOperation::apply(const Permutation& permutation) {
  Operation::apply(permutation);
  for (const auto& op : ops) {
    op->apply(permutation);
  }
}

void CompoundOperation::merge(CompoundOperation& op) {
  ops.reserve(ops.size() + op.size());
  ops.insert(ops.end(), std::make_move_iterator(op.begin()),
             std::make_move_iterator(op.end()));
  op.clear();
}

bool CompoundOperation::isConvertibleToSingleOperation() const {
  if (ops.size() != 1) {
    return false;
  }
  assert(ops.front() != nullptr);
  if (!ops.front()->isCompoundOperation()) {
    return true;
  }
  return dynamic_cast<const CompoundOperation&>(*ops.front())
      .isConvertibleToSingleOperation();
}

std::unique_ptr<Operation> CompoundOperation::collapseToSingleOperation() {
  assert(isConvertibleToSingleOperation());
  if (!ops.front()->isCompoundOperation()) {
    return std::move(ops.front());
  }
  return dynamic_cast<CompoundOperation&>(*ops.front())
      .collapseToSingleOperation();
}

} // namespace qc

std::size_t std::hash<qc::CompoundOperation>::operator()(
    const qc::CompoundOperation& co) const noexcept {
  std::size_t seed = 0U;
  for (const auto& op : co) {
    qc::hashCombine(seed, std::hash<qc::Operation>{}(*op));
  }
  return seed;
}
