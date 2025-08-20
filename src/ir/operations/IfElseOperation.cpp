/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/operations/IfElseOperation.hpp"

#include "ir/Definitions.hpp"
#include "ir/Permutation.hpp"
#include "ir/Register.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/Operation.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace qc {

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

IfElseOperation::IfElseOperation(std::unique_ptr<Operation>&& thenOp,
                                 std::unique_ptr<Operation>&& elseOp,
                                 const ClassicalRegister& controlRegister,
                                 const std::uint64_t expectedValue,
                                 const ComparisonKind kind)
    : thenOp(std::move(thenOp)), elseOp(std::move(elseOp)),
      controlRegister(controlRegister), expectedValueRegister(expectedValue),
      comparisonKind(kind) {
  name = "if_else";
  type = IfElse;
}

IfElseOperation::IfElseOperation(std::unique_ptr<Operation>&& thenOp,
                                 std::unique_ptr<Operation>&& elseOp,
                                 const Bit controlBit, const bool expectedValue,
                                 const ComparisonKind kind)
    : thenOp(std::move(thenOp)), elseOp(std::move(elseOp)),
      controlBit(controlBit), expectedValueBit(expectedValue),
      comparisonKind(kind) {
  // Canonicalize comparisons on a single bit
  if (comparisonKind == Neq) {
    comparisonKind = Eq;
    expectedValueBit = !expectedValueBit;
  }
  if (comparisonKind != Eq) {
    throw std::invalid_argument(
        "Inequality comparisons on a single bit are not supported.");
  }
  name = "if_else";
  type = IfElse;
}

IfElseOperation::IfElseOperation(const IfElseOperation& op)
    : Operation(op), thenOp(op.thenOp ? op.thenOp->clone() : nullptr),
      elseOp(op.elseOp ? op.elseOp->clone() : nullptr),
      controlRegister(op.controlRegister), controlBit(op.controlBit),
      expectedValueRegister(op.expectedValueRegister),
      expectedValueBit(op.expectedValueBit), comparisonKind(op.comparisonKind) {
}

IfElseOperation& IfElseOperation::operator=(const IfElseOperation& op) {
  if (this != &op) {
    Operation::operator=(op);
    thenOp = op.thenOp ? op.thenOp->clone() : nullptr;
    elseOp = op.elseOp ? op.elseOp->clone() : nullptr;
    controlRegister = op.controlRegister;
    controlBit = op.controlBit;
    expectedValueRegister = op.expectedValueRegister;
    expectedValueBit = op.expectedValueBit;
    comparisonKind = op.comparisonKind;
  }
  return *this;
}

std::ostream&
IfElseOperation::print(std::ostream& os, const Permutation& permutation,
                       [[maybe_unused]] const std::size_t prefixWidth,
                       const std::size_t nqubits) const {
  if (thenOp) {
    thenOp->print(os, permutation, prefixWidth, nqubits);
  }

  os << "  " << "\033[1m\033[35m";
  if (controlRegister.has_value()) {
    assert(!controlBit.has_value());
    os << controlRegister->getName() << " == " << expectedValueRegister;
  }
  if (controlBit.has_value()) {
    assert(!controlRegister.has_value());
    os << (expectedValueBit ? "!" : "") << "c[" << controlBit.value() << "]";
  }
  os << "\033[0m";

  return os;
}

bool IfElseOperation::equals(const Operation& operation,
                             const Permutation& perm1,
                             const Permutation& perm2) const {
  if (const auto* other = dynamic_cast<const IfElseOperation*>(&operation)) {
    if (controlRegister != other->controlRegister) {
      return false;
    }
    if (controlBit != other->controlBit) {
      return false;
    }
    if (expectedValueRegister != other->expectedValueRegister) {
      return false;
    }
    if (expectedValueBit != other->expectedValueBit) {
      return false;
    }
    if (comparisonKind != other->comparisonKind) {
      return false;
    }
    if (thenOp && other->thenOp) {
      if (!thenOp->equals(*other->thenOp, perm1, perm2)) {
        return false;
      }
    } else if (thenOp || other->thenOp) {
      return false;
    }
    if (elseOp && other->elseOp) {
      if (!elseOp->equals(*other->elseOp, perm1, perm2)) {
        return false;
      }
    } else if (elseOp || other->elseOp) {
      return false;
    }
    return true;
  }
  return false;
}

void IfElseOperation::dumpOpenQASM(std::ostream& of,
                                   const QubitIndexToRegisterMap& qubitMap,
                                   const BitIndexToRegisterMap& bitMap,
                                   const std::size_t indent,
                                   const bool openQASM3) const {
  of << std::string(indent * OUTPUT_INDENT_SIZE, ' ');
  of << "if (";
  if (controlRegister.has_value()) {
    assert(!controlBit.has_value());
    of << controlRegister->getName() << ' ' << comparisonKind << ' '
       << expectedValueRegister;
  } else if (controlBit.has_value()) {
    of << (!expectedValueBit ? "!" : "") << bitMap.at(*controlBit).second;
  }
  of << ") ";
  of << "{\n";
  if (thenOp) {
    thenOp->dumpOpenQASM(of, qubitMap, bitMap, indent + 1, openQASM3);
  }
  if (!elseOp) {
    of << "}\n";
    return;
  }
  of << "}";
  if (openQASM3) {
    of << " else {\n";
    elseOp->dumpOpenQASM(of, qubitMap, bitMap, indent + 1, openQASM3);
  } else {
    of << " if (";
    if (controlRegister.has_value()) {
      assert(!controlBit.has_value());
      of << controlRegister->getName() << ' '
         << getInvertedComparisonKind(comparisonKind) << ' '
         << expectedValueRegister;
    }
    if (controlBit.has_value()) {
      assert(!controlRegister.has_value());
      of << (expectedValueBit ? "!" : "") << bitMap.at(*controlBit).second;
    }
    of << ") ";
    of << "{\n";
    elseOp->dumpOpenQASM(of, qubitMap, bitMap, indent + 1, openQASM3);
  }
  of << "}\n";
}

} // namespace qc

std::size_t std::hash<qc::IfElseOperation>::operator()(
    qc::IfElseOperation const& op) const noexcept {
  std::size_t seed = 0U;
  if (op.getThenOp() != nullptr) {
    qc::hashCombine(seed, std::hash<qc::Operation>{}(*op.getThenOp()));
  }
  if (op.getElseOp() != nullptr) {
    qc::hashCombine(seed, std::hash<qc::Operation>{}(*op.getElseOp()));
  }
  if (const auto& reg = op.getControlRegister(); reg.has_value()) {
    assert(!op.getControlBit().has_value());
    qc::hashCombine(seed, std::hash<qc::ClassicalRegister>{}(reg.value()));
    qc::hashCombine(seed, op.getExpectedValueRegister());
  }
  if (const auto& bit = op.getControlBit(); bit.has_value()) {
    assert(!op.getControlRegister().has_value());
    qc::hashCombine(seed, bit.value());
    qc::hashCombine(seed, static_cast<std::size_t>(op.getExpectedValueBit()));
  }
  qc::hashCombine(seed, op.getComparisonKind());
  return seed;
}
