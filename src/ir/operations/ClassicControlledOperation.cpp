/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/operations/ClassicControlledOperation.hpp"

#include "ir/Definitions.hpp"
#include "ir/Permutation.hpp"
#include "ir/Register.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/Operation.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace qc {

std::string toString(const ComparisonKind& kind) {
  switch (kind) {
  case Eq:
    return "==";
  case Neq:
    return "!=";
  case Lt:
    return "<";
  case Leq:
    return "<=";
  case Gt:
    return ">";
  case Geq:
    return ">=";
  default:
    unreachable();
  }
}

std::ostream& operator<<(std::ostream& os, const ComparisonKind& kind) {
  os << toString(kind);
  return os;
}

ClassicControlledOperation::ClassicControlledOperation(
    std::unique_ptr<Operation>&& operation, ClassicalRegister controlReg,
    const std::uint64_t expectedVal, const ComparisonKind kind)
    : op(std::move(operation)), controlRegister(std::move(controlReg)),
      expectedValue(expectedVal), comparisonKind(kind) {
  name = "c_" + shortName(op->getType());
  parameter.reserve(3);
  parameter.emplace_back(static_cast<fp>(controlRegister->getStartIndex()));
  parameter.emplace_back(static_cast<fp>(controlRegister->getSize()));
  parameter.emplace_back(static_cast<fp>(expectedValue));
  type = ClassicControlled;
}
ClassicControlledOperation::ClassicControlledOperation(
    std::unique_ptr<Operation>&& operation, const Bit cBit,
    const std::uint64_t expectedVal, const ComparisonKind kind)
    : op(std::move(operation)), controlBit(cBit), expectedValue(expectedVal),
      comparisonKind(kind) {
  if (expectedVal > 1) {
    throw std::invalid_argument(
        "Expected value for single bit comparison must be 0 or 1.");
  }
  name = "c_" + shortName(op->getType());
  // Canonicalize comparisons on a single bit.
  if (comparisonKind == Neq) {
    comparisonKind = Eq;
    expectedValue = 1 - expectedValue;
  }
  if (comparisonKind != Eq) {
    throw std::invalid_argument(
        "Inequality comparisons on a single bit are not supported.");
  }
  parameter.reserve(2);
  parameter.emplace_back(static_cast<fp>(cBit));
  parameter.emplace_back(static_cast<fp>(expectedValue));
  type = ClassicControlled;
}
ClassicControlledOperation::ClassicControlledOperation(
    const ClassicControlledOperation& ccop)
    : Operation(ccop), controlRegister(ccop.controlRegister),
      controlBit(ccop.controlBit), expectedValue(ccop.expectedValue) {
  op = ccop.op->clone();
}
ClassicControlledOperation&
ClassicControlledOperation::operator=(const ClassicControlledOperation& ccop) {
  if (this != &ccop) {
    Operation::operator=(ccop);
    controlRegister = ccop.controlRegister;
    controlBit = ccop.controlBit;
    expectedValue = ccop.expectedValue;
    op = ccop.op->clone();
  }
  return *this;
}

bool ClassicControlledOperation::equals(const Operation& operation,
                                        const Permutation& perm1,
                                        const Permutation& perm2) const {
  if (const auto* classic =
          dynamic_cast<const ClassicControlledOperation*>(&operation)) {
    if (controlRegister != classic->controlRegister) {
      return false;
    }

    if (controlBit != classic->controlBit) {
      return false;
    }

    if (expectedValue != classic->expectedValue ||
        comparisonKind != classic->comparisonKind) {
      return false;
    }

    return op->equals(*classic->op, perm1, perm2);
  }
  return false;
}
void ClassicControlledOperation::dumpOpenQASM(
    std::ostream& of, const QubitIndexToRegisterMap& qubitMap,
    const BitIndexToRegisterMap& bitMap, const std::size_t indent,
    const bool openQASM3) const {
  of << std::string(indent * OUTPUT_INDENT_SIZE, ' ');
  of << "if (";
  if (controlRegister.has_value()) {
    assert(!controlBit.has_value());
    of << controlRegister->getName() << " " << comparisonKind << " "
       << expectedValue;
  }
  if (controlBit.has_value()) {
    assert(!controlRegister.has_value());
    of << (expectedValue == 0 ? "!" : "") << bitMap.at(*controlBit).second;
  }
  of << ") ";
  if (openQASM3) {
    of << "{\n";
  }
  op->dumpOpenQASM(of, qubitMap, bitMap, indent + 1, openQASM3);
  if (openQASM3) {
    of << "}\n";
  }
}

ComparisonKind getInvertedComparisonKind(const ComparisonKind kind) {
  switch (kind) {
  case Lt:
    return Geq;
  case Leq:
    return Gt;
  case Gt:
    return Leq;
  case Geq:
    return Lt;
  case Eq:
    return Neq;
  case Neq:
    return Eq;
  default:
    unreachable();
  }
}
} // namespace qc
