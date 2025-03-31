/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/operations/Operation.hpp"

#include "ir/Definitions.hpp"
#include "ir/Permutation.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/OpType.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <set>
#include <vector>

namespace qc {

std::ostream& Operation::printParameters(std::ostream& os) const {
  if (isClassicControlledOperation()) {

    os << "  c[" << parameter[0];
    if (parameter.size() == 2) {
      os << "] == " << parameter[1];
    } else {
      os << "..." << (parameter[0] + parameter[1] - 1)
         << "] == " << parameter[2];
    }
    return os;
  }

  bool isZero = true;
  for (const auto& p : parameter) {
    if (p != static_cast<fp>(0)) {
      isZero = false;
      break;
    }
  }
  if (!isZero) {
    os << "  p: (" << parameter[0] << ") ";
    for (size_t j = 1; j < parameter.size(); ++j) {
      isZero = true;
      for (size_t i = j; i < parameter.size(); ++i) {
        if (parameter.at(i) != static_cast<fp>(0)) {
          isZero = false;
          break;
        }
      }
      if (isZero) {
        break;
      }
      os << "(" << parameter.at(j) << ") ";
    }
  }

  return os;
}

std::ostream& Operation::print(std::ostream& os, const Permutation& permutation,
                               [[maybe_unused]] const std::size_t prefixWidth,
                               const std::size_t nqubits) const {
  const auto precBefore = std::cout.precision(20);
  const auto& actualControls = permutation.apply(getControls());
  const auto& actualTargets = permutation.apply(getTargets());

  for (std::size_t i = 0; i < nqubits; ++i) {
    const auto q = static_cast<Qubit>(i);
    if (std::find(actualTargets.cbegin(), actualTargets.cend(), q) !=
        actualTargets.cend()) {
      if (type == ClassicControlled) {
        const auto reducedName = name.substr(2);
        os << "\033[1m\033[35m" << std::setw(4) << reducedName;
      } else if (type == Barrier) {
        os << "\033[1m\033[32m" << std::setw(4) << shortName(type);
      } else {
        os << "\033[1m\033[36m" << std::setw(4) << shortName(type);
      }
      os << "\033[0m";
      continue;
    }

    if (const auto it =
            std::find(actualControls.cbegin(), actualControls.cend(), q);
        it != actualControls.cend()) {
      if (it->type == Control::Type::Pos) {
        os << "\033[32m";
      } else {
        os << "\033[31m";
      }
      os << std::setw(4) << "c"
         << "\033[0m";
      continue;
    }

    os << std::setw(4) << "|"
       << "\033[0m";
  }

  printParameters(os);

  std::cout.precision(precBefore);

  return os;
}

bool Operation::equals(const Operation& op, const Permutation& perm1,
                       const Permutation& perm2) const {
  // check type
  if (getType() != op.getType()) {
    return false;
  }

  // check number of controls
  const auto nc1 = getNcontrols();
  const auto nc2 = op.getNcontrols();
  if (nc1 != nc2) {
    return false;
  }

  // check parameters
  const auto& param1 = getParameter();
  const auto& param2 = op.getParameter();
  if (param1 != param2) {
    return false;
  }

  if (isDiagonalGate()) {
    // check pos. controls and targets together
    const auto& usedQubits1 = getUsedQubitsPermuted(perm1);
    const auto& usedQubits2 = op.getUsedQubitsPermuted(perm2);
    if (usedQubits1 != usedQubits2) {
      return false;
    }

    std::set<Qubit> negControls1{};
    for (const auto& control : getControls()) {
      if (control.type == Control::Type::Neg) {
        negControls1.emplace(perm1.apply(control.qubit));
      }
    }
    std::set<Qubit> negControls2{};
    for (const auto& control : op.getControls()) {
      if (control.type == Control::Type::Neg) {
        negControls2.emplace(perm2.apply(control.qubit));
      }
    }
    return negControls1 == negControls2;
  }
  // check controls
  if (nc1 != 0U &&
      perm1.apply(getControls()) != perm2.apply(op.getControls())) {
    return false;
  }

  return perm1.apply(getTargets()) == perm2.apply(op.getTargets());
}

void Operation::addDepthContribution(std::vector<std::size_t>& depths) const {
  if (type == Barrier) {
    return;
  }

  std::size_t maxDepth = 0;
  for (const auto& target : getTargets()) {
    maxDepth = std::max(maxDepth, depths[target]);
  }
  for (const auto& control : getControls()) {
    maxDepth = std::max(maxDepth, depths[control.qubit]);
  }
  maxDepth += 1;
  for (const auto& target : getTargets()) {
    depths[target] = maxDepth;
  }
  for (const auto& control : getControls()) {
    depths[control.qubit] = maxDepth;
  }
}

void Operation::apply(const Permutation& permutation) {
  getTargets() = permutation.apply(getTargets());
  getControls() = permutation.apply(getControls());
}

auto Operation::isInverseOf(const Operation& other) const -> bool {
  return operator==(*other.getInverted());
}

auto Operation::getUsedQubitsPermuted(const qc::Permutation& perm) const
    -> std::set<Qubit> {
  std::set<Qubit> usedQubits;
  for (const auto& target : getTargets()) {
    usedQubits.emplace(perm.apply(target));
  }
  for (const auto& control : getControls()) {
    usedQubits.emplace(perm.apply(control.qubit));
  }
  return usedQubits;
}

auto Operation::getUsedQubits() const -> std::set<Qubit> {
  return getUsedQubitsPermuted({});
}
} // namespace qc
