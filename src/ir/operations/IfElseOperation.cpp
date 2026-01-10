/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
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
  }
  unreachable();
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
  }
  unreachable();
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
    : thenOp_(std::move(thenOp)), elseOp_(std::move(elseOp)),
      controlRegister_(controlRegister), expectedValueRegister_(expectedValue),
      comparisonKind_(kind) {
  name = "if_else";
  type = IfElse;
  canonicalize();
}

IfElseOperation::IfElseOperation(std::unique_ptr<Operation>&& thenOp,
                                 std::unique_ptr<Operation>&& elseOp,
                                 const Bit controlBit, const bool expectedValue,
                                 const ComparisonKind kind)
    : thenOp_(std::move(thenOp)), elseOp_(std::move(elseOp)),
      controlBit_(controlBit), expectedValueBit_(expectedValue),
      comparisonKind_(kind) {
  name = "if_else";
  type = IfElse;
  canonicalize();
}

IfElseOperation::IfElseOperation(const IfElseOperation& op)
    : Operation(op), thenOp_(op.thenOp_ ? op.thenOp_->clone() : nullptr),
      elseOp_(op.elseOp_ ? op.elseOp_->clone() : nullptr),
      controlRegister_(op.controlRegister_), controlBit_(op.controlBit_),
      expectedValueRegister_(op.expectedValueRegister_),
      expectedValueBit_(op.expectedValueBit_),
      comparisonKind_(op.comparisonKind_) {}

IfElseOperation& IfElseOperation::operator=(const IfElseOperation& op) {
  if (this != &op) {
    Operation::operator=(op);
    thenOp_ = op.thenOp_ ? op.thenOp_->clone() : nullptr;
    elseOp_ = op.elseOp_ ? op.elseOp_->clone() : nullptr;
    controlRegister_ = op.controlRegister_;
    controlBit_ = op.controlBit_;
    expectedValueRegister_ = op.expectedValueRegister_;
    expectedValueBit_ = op.expectedValueBit_;
    comparisonKind_ = op.comparisonKind_;
  }
  return *this;
}

void IfElseOperation::apply(const Permutation& permutation) {
  if (thenOp_) {
    thenOp_->apply(permutation);
  }
  if (elseOp_) {
    elseOp_->apply(permutation);
  }
}

bool IfElseOperation::equals(const Operation& operation,
                             const Permutation& perm1,
                             const Permutation& perm2) const {
  if (const auto* other = dynamic_cast<const IfElseOperation*>(&operation)) {
    if (controlRegister_ != other->controlRegister_) {
      return false;
    }
    if (controlBit_ != other->controlBit_) {
      return false;
    }
    if (expectedValueRegister_ != other->expectedValueRegister_) {
      return false;
    }
    if (expectedValueBit_ != other->expectedValueBit_) {
      return false;
    }
    if (comparisonKind_ != other->comparisonKind_) {
      return false;
    }
    if (thenOp_ && other->thenOp_) {
      if (!thenOp_->equals(*other->thenOp_, perm1, perm2)) {
        return false;
      }
    } else if (thenOp_ || other->thenOp_) {
      return false;
    }
    if (elseOp_ && other->elseOp_) {
      if (!elseOp_->equals(*other->elseOp_, perm1, perm2)) {
        return false;
      }
    } else if (elseOp_ || other->elseOp_) {
      return false;
    }
    return true;
  }
  return false;
}

std::ostream&
IfElseOperation::print(std::ostream& os, const Permutation& permutation,
                       [[maybe_unused]] const std::size_t prefixWidth,
                       const std::size_t nqubits) const {
  const std::string indent(prefixWidth, ' ');

  // print condition header line
  os << indent << "\033[1m\033[35m" << "if (";
  if (controlRegister_.has_value()) {
    assert(!controlBit_.has_value());
    os << controlRegister_->getName() << ' ' << comparisonKind_ << ' '
       << expectedValueRegister_;
  } else if (controlBit_.has_value()) {
    assert(!controlRegister_.has_value());
    os << (!expectedValueBit_ ? "!" : "") << "c[" << controlBit_.value() << "]";
  }
  os << ") {\033[0m" << '\n'; // cyan brace

  // then-block
  if (thenOp_) {
    os << indent;
    thenOp_->print(os, permutation, prefixWidth, nqubits);
  }
  os << '\n';

  // else-block (only if present)
  if (elseOp_) {
    os << indent << "  \033[1m\033[35m} else {\033[0m" << '\n' << indent;
    elseOp_->print(os, permutation, prefixWidth, nqubits);
    os << '\n';
  }

  // closing brace aligned with prefixWidth
  os << indent << "  \033[1m\033[35m}\033[0m";

  return os;
}

void IfElseOperation::dumpOpenQASM(std::ostream& of,
                                   const QubitIndexToRegisterMap& qubitMap,
                                   const BitIndexToRegisterMap& bitMap,
                                   const std::size_t indent,
                                   const bool openQASM3) const {
  of << std::string(indent * OUTPUT_INDENT_SIZE, ' ');
  of << "if (";
  if (controlRegister_.has_value()) {
    assert(!controlBit_.has_value());
    of << controlRegister_->getName() << ' ' << comparisonKind_ << ' '
       << expectedValueRegister_;
  } else if (controlBit_.has_value()) {
    of << (!expectedValueBit_ ? "!" : "") << bitMap.at(*controlBit_).second;
  }
  of << ") ";
  of << "{\n";
  if (thenOp_) {
    thenOp_->dumpOpenQASM(of, qubitMap, bitMap, indent + 1, openQASM3);
  }
  if (!elseOp_) {
    of << "}\n";
    return;
  }
  of << "}";
  if (openQASM3) {
    of << " else {\n";
    elseOp_->dumpOpenQASM(of, qubitMap, bitMap, indent + 1, openQASM3);
  } else {
    of << '\n' << "if (";
    if (controlRegister_.has_value()) {
      assert(!controlBit_.has_value());
      of << controlRegister_->getName() << ' '
         << getInvertedComparisonKind(comparisonKind_) << ' '
         << expectedValueRegister_;
    }
    if (controlBit_.has_value()) {
      assert(!controlRegister_.has_value());
      of << (expectedValueBit_ ? "!" : "") << bitMap.at(*controlBit_).second;
    }
    of << ") ";
    of << "{\n";
    elseOp_->dumpOpenQASM(of, qubitMap, bitMap, indent + 1, openQASM3);
  }
  of << "}\n";
}

/**
 * @brief Canonicalizes the IfElseOperation by normalizing its internal
 * representation.
 *
 * This method ensures that the then/else branches and comparison kinds are in a
 * standard form.
 * - If the thenOp is null, swap thenOp and elseOp, and invert the comparison
 * kind.
 * - For single-bit control, only equality comparisons are supported; Neq is
 * converted to Eq with inverted expectedValueBit.
 * - If expectedValueBit is false and elseOp exists, swap thenOp and elseOp, and
 * set expectedValueBit to true.
 *
 * This normalization simplifies further processing and ensures consistent
 * behavior.
 */
void IfElseOperation::canonicalize() {
  // If thenOp is null, swap thenOp and elseOp, and invert the comparison kind.
  if (thenOp_ == nullptr) {
    std::swap(thenOp_, elseOp_);
    comparisonKind_ = getInvertedComparisonKind(comparisonKind_);
  }
  // If control is a single bit, only equality comparisons are supported.
  if (controlBit_.has_value()) {
    // Convert Neq to Eq by inverting expectedValueBit.
    if (comparisonKind_ == Neq) {
      comparisonKind_ = Eq;
      expectedValueBit_ = !expectedValueBit_;
    }
    // Throw if comparison is not Eq (after possible conversion above).
    if (comparisonKind_ != Eq) {
      throw std::invalid_argument(
          "Inequality comparisons on a single bit are not supported.");
    }
    // If expectedValueBit is false and elseOp exists, swap thenOp and elseOp,
    // and set expectedValueBit to true.
    if (!expectedValueBit_ && elseOp_ != nullptr) {
      std::swap(thenOp_, elseOp_);
      expectedValueBit_ = true;
    }
  }
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
