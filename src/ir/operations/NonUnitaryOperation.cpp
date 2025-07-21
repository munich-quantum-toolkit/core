/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for
 * more information.
 */

#include "ir/operations/NonUnitaryOperation.hpp"

#include "ir/Definitions.hpp"
#include "ir/Permutation.hpp"
#include "ir/Register.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/Operation.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iomanip>
#include <ostream>
#include <ranges>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qc {
// Measurement constructor
NonUnitaryOperation::NonUnitaryOperation(std::vector<Qubit> qubitRegister,
                                         std::vector<Bit> classicalRegister)
    : classics(std::move(classicalRegister)) {
  type = Measure;
  targets = std::move(qubitRegister);
  name = toString(type);
  if (targets.size() != classics.size()) {
    throw std::invalid_argument(
        "Sizes of qubit register and classical register do not match.");
  }
}
NonUnitaryOperation::NonUnitaryOperation(const Qubit qubit, const Bit cbit)
    : classics({cbit}) {
  type = Measure;
  targets = {qubit};
  name = toString(type);
}

// General constructor
NonUnitaryOperation::NonUnitaryOperation(Targets qubits, OpType op) {
  type = op;
  targets = std::move(qubits);
  std::ranges::sort(targets);
  name = toString(type);
}

std::ostream&
NonUnitaryOperation::print(std::ostream& os, const Permutation& permutation,
                           [[maybe_unused]] const std::size_t prefixWidth,
                           const std::size_t nqubits) const {
  switch (type) {
  case Measure:
    printMeasurement(os, targets, classics, permutation, nqubits);
    break;
  case Reset:
    printReset(os, targets, permutation, nqubits);
    break;
  default:
    break;
  }
  return os;
}

namespace {
bool isWholeClassicalRegister(const BitIndexToRegisterMap& reg, const Bit start,
                              const Bit end) {
  const auto& startReg = reg.at(start).first;
  const auto& endReg = reg.at(end).first;
  return startReg == endReg && startReg.getStartIndex() == start &&
         endReg.getEndIndex() == end;
}
} // namespace

void NonUnitaryOperation::dumpOpenQASM(std::ostream& of,
                                       const QubitIndexToRegisterMap& qubitMap,
                                       const BitIndexToRegisterMap& bitMap,
                                       std::size_t indent,
                                       bool openQASM3) const {
  of << std::string(indent * OUTPUT_INDENT_SIZE, ' ');

  if (isWholeQubitRegister(qubitMap, targets.front(), targets.back()) &&
      (type != Measure ||
       isWholeClassicalRegister(bitMap, classics.front(), classics.back()))) {
    if (type == Measure && openQASM3) {
      of << bitMap.at(classics.front()).first.getName() << " = ";
    }
    of << toString(type) << " " << qubitMap.at(targets.front()).first.getName();
    if (type == Measure && !openQASM3) {
      of << " -> ";
      of << bitMap.at(classics.front()).first.getName();
    }
    of << ";\n";
    return;
  }
  auto classicsIt = classics.cbegin();
  for (const auto& q : targets) {
    const auto& qreg = qubitMap.at(q);
    if (type == Measure && openQASM3) {
      const auto& creg = bitMap.at(*classicsIt);
      of << creg.second << " = ";
    }
    of << toString(type) << " " << qreg.second;
    if (type == Measure && !openQASM3) {
      const auto& creg = bitMap.at(*classicsIt);
      of << " -> " << creg.second;
      ++classicsIt;
    }
    of << ";\n";
  }
}

bool NonUnitaryOperation::equals(const Operation& op, const Permutation& perm1,
                                 const Permutation& perm2) const {
  if (const auto* nonunitary = dynamic_cast<const NonUnitaryOperation*>(&op)) {
    if (getType() != nonunitary->getType()) {
      return false;
    }

    if (getType() == Measure) {
      // check number of qubits to be measured
      const auto nq1 = targets.size();
      const auto nq2 = nonunitary->targets.size();
      if (nq1 != nq2) {
        return false;
      }

      // these are just sanity checks and should always be fulfilled
      assert(targets.size() == classics.size());
      assert(nonunitary->targets.size() == nonunitary->classics.size());

      std::set<std::pair<Qubit, Bit>> measurements1{};
      auto qubitIt1 = targets.cbegin();
      auto classicIt1 = classics.cbegin();
      while (qubitIt1 != targets.cend()) {
        if (perm1.empty()) {
          measurements1.emplace(*qubitIt1, *classicIt1);
        } else {
          measurements1.emplace(perm1.at(*qubitIt1), *classicIt1);
        }
        ++qubitIt1;
        ++classicIt1;
      }

      std::set<std::pair<Qubit, Bit>> measurements2{};
      auto qubitIt2 = nonunitary->targets.cbegin();
      auto classicIt2 = nonunitary->classics.cbegin();
      while (qubitIt2 != nonunitary->targets.cend()) {
        if (perm2.empty()) {
          measurements2.emplace(*qubitIt2, *classicIt2);
        } else {
          measurements2.emplace(perm2.at(*qubitIt2), *classicIt2);
        }
        ++qubitIt2;
        ++classicIt2;
      }

      return measurements1 == measurements2;
    }
    return Operation::equals(op, perm1, perm2);
  }
  return false;
}

void NonUnitaryOperation::printMeasurement(std::ostream& os,
                                           const std::vector<Qubit>& q,
                                           const std::vector<Bit>& c,
                                           const Permutation& permutation,
                                           const std::size_t nqubits) {
  auto qubitIt = q.cbegin();
  auto classicIt = c.cbegin();
  if (permutation.empty()) {
    for (std::size_t i = 0; i < nqubits; ++i) {
      if (qubitIt != q.cend() && *qubitIt == i) {
        os << "\033[34m" << std::setw(4) << *classicIt << "\033[0m";
        ++qubitIt;
        ++classicIt;
      } else {
        os << std::setw(4) << "|";
      }
    }
  } else {
    for (const auto& [physical, logical] : permutation) {
      if (qubitIt != q.cend() && *qubitIt == physical) {
        os << "\033[34m" << std::setw(4) << *classicIt << "\033[0m";
        ++qubitIt;
        ++classicIt;
      } else {
        os << std::setw(4) << "|";
      }
    }
  }
}

void NonUnitaryOperation::printReset(std::ostream& os,
                                     const std::vector<Qubit>& q,
                                     const Permutation& permutation,
                                     const std::size_t nqubits) const {
  const auto actualTargets = permutation.apply(q);
  for (std::size_t i = 0; i < nqubits; ++i) {
    if (std::find(actualTargets.cbegin(), actualTargets.cend(), i) !=
        actualTargets.cend()) {
      os << "\033[31m" << std::setw(4) << shortName(type) << "\033[0m";
      continue;
    }
    os << std::setw(4) << "|";
  }
}

void NonUnitaryOperation::addDepthContribution(
    std::vector<std::size_t>& depths) const {
  for (const auto& target : getTargets()) {
    depths[target] += 1;
  }
}

void NonUnitaryOperation::apply(const Permutation& permutation) {
  getTargets() = permutation.apply(getTargets());
}
} // namespace qc
