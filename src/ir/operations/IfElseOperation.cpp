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

#include "ir/operations/OpType.hpp"

#include <cassert>
#include <utility>

namespace qc {
IfElseOperation::IfElseOperation(std::unique_ptr<Operation>&& thenOp,
                                 std::unique_ptr<Operation>&& elseOp,
                                 ClassicalRegister controlReg,
                                 const std::uint64_t expectedVal,
                                 const ComparisonKind kind)
    : thenBranch(std::move(thenOp)), elseBranch(std::move(elseOp)),
      controlRegister(std::move(controlReg)), expectedValue(expectedVal),
      comparisonKind(kind) {
  name = "if_else";
  type = IfElse;
}

IfElseOperation::IfElseOperation(std::unique_ptr<Operation>&& thenOp,
                                 std::unique_ptr<Operation>&& elseOp, Bit cBit,
                                 const std::uint64_t expectedVal,
                                 ComparisonKind kind)
    : thenBranch(std::move(thenOp)), elseBranch(std::move(elseOp)),
      controlBit(cBit), expectedValue(expectedVal), comparisonKind(kind) {
  assert(expectedVal <= 1);
  name = "if_else";
  type = IfElse;
}

IfElseOperation::IfElseOperation(const IfElseOperation& op)
    : Operation(op),
      thenBranch(op.thenBranch ? op.thenBranch->clone() : nullptr),
      elseBranch(op.elseBranch ? op.elseBranch->clone() : nullptr),
      controlRegister(op.controlRegister), controlBit(op.controlBit),
      expectedValue(op.expectedValue), comparisonKind(op.comparisonKind) {}

IfElseOperation& IfElseOperation::operator=(const IfElseOperation& op) {
  if (this != &op) {
    Operation::operator=(op);
    thenBranch = op.thenBranch ? op.thenBranch->clone() : nullptr;
    elseBranch = op.elseBranch ? op.elseBranch->clone() : nullptr;
    controlRegister = op.controlRegister;
    controlBit = op.controlBit;
    expectedValue = op.expectedValue;
    comparisonKind = op.comparisonKind;
  }
  return *this;
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
    if (expectedValue != other->expectedValue ||
        comparisonKind != other->comparisonKind) {
      return false;
    }
    if (thenBranch && other->thenBranch) {
      if (!thenBranch->equals(*other->thenBranch, perm1, perm2)) {
        return false;
      }
    } else if (thenBranch || other->thenBranch) {
      return false;
    }
    if (elseBranch && other->elseBranch) {
      if (!elseBranch->equals(*other->elseBranch, perm1, perm2)) {
        return false;
      }
    } else if (elseBranch || other->elseBranch) {
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
       << expectedValue;
  } else if (controlBit.has_value()) {
    of << (expectedValue == 0 ? "!" : "") << bitMap.at(*controlBit).second;
  }
  of << ") ";
  if (openQASM3) {
    of << "{\n";
  }
  if (thenBranch) {
    thenBranch->dumpOpenQASM(of, qubitMap, bitMap, indent + 1, openQASM3);
  }
  if (openQASM3) {
    of << "}";
    if (elseBranch) {
      of << " else {\n";
      elseBranch->dumpOpenQASM(of, qubitMap, bitMap, indent + 1, openQASM3);
      of << "}";
    }
    of << "\n";
  }
}
} // namespace qc

std::size_t std::hash<qc::IfElseOperation>::operator()(
    qc::IfElseOperation const& op) const noexcept {
  std::size_t seed = 0U;
  if (op.getThenBranch()) {
    qc::hashCombine(seed, std::hash<qc::Operation>{}(*op.getThenBranch()));
  }
  if (op.getElseBranch()) {
    qc::hashCombine(seed, std::hash<qc::Operation>{}(*op.getElseBranch()));
  }
  if (const auto& reg = op.getControlRegister(); reg.has_value()) {
    qc::hashCombine(seed, std::hash<qc::ClassicalRegister>{}(reg.value()));
  }
  if (const auto& bit = op.getControlBit(); bit.has_value()) {
    qc::hashCombine(seed, bit.value());
  }
  qc::hashCombine(seed, op.getExpectedValue());
  qc::hashCombine(seed, op.getComparisonKind());
  return seed;
}
