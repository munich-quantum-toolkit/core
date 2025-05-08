/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qasm3/Statement.hpp"

#include "ir/operations/ClassicControlledOperation.hpp"
#include "qasm3/InstVisitor.hpp"

#include <cassert>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace qasm3 {

std::optional<qc::ComparisonKind>
getComparisonKind(const BinaryExpression::Op op) {
  switch (op) {
  case BinaryExpression::Op::LessThan:
    return qc::ComparisonKind::Lt;
  case BinaryExpression::Op::LessThanOrEqual:
    return qc::ComparisonKind::Leq;
  case BinaryExpression::Op::GreaterThan:
    return qc::ComparisonKind::Gt;
  case BinaryExpression::Op::GreaterThanOrEqual:
    return qc::ComparisonKind::Geq;
  case BinaryExpression::Op::Equal:
    return qc::ComparisonKind::Eq;
  case BinaryExpression::Op::NotEqual:
    return qc::ComparisonKind::Neq;
  default:
    return std::nullopt;
  }
}
void OutputPermutation::accept(InstVisitor* visitor) {
  visitor->visitOutputPermutation(shared_from_this());
}
void DeclarationStatement::accept(InstVisitor* visitor) {
  visitor->visitDeclarationStatement(shared_from_this());
}
void GateCallStatement::accept(InstVisitor* visitor) {
  visitor->visitGateCallStatement(shared_from_this());
}
void AssignmentStatement::accept(InstVisitor* visitor) {
  visitor->visitAssignmentStatement(shared_from_this());
}
void BarrierStatement::accept(InstVisitor* visitor) {
  visitor->visitBarrierStatement(shared_from_this());
}
void ResetStatement::accept(InstVisitor* visitor) {
  visitor->visitResetStatement(shared_from_this());
}
void IfStatement::accept(InstVisitor* visitor) {
  visitor->visitIfStatement(shared_from_this());
}

GateDeclaration::GateDeclaration(
    std::shared_ptr<DebugInfo> debug, std::string id,
    std::shared_ptr<IdentifierList> params,
    std::shared_ptr<IdentifierList> qbits,
    std::vector<std::shared_ptr<QuantumStatement>> stmts, const bool opaque)
    : Statement(std::move(debug)), identifier(std::move(id)),
      parameters(std::move(params)), qubits(std::move(qbits)),
      statements(std::move(stmts)), isOpaque(opaque) {
  if (opaque) {
    assert(statements.empty() && "Opaque gate should not have statements.");
  }
}

void GateDeclaration::accept(InstVisitor* visitor) {
  visitor->visitGateStatement(shared_from_this());
}
void VersionDeclaration::accept(InstVisitor* visitor) {
  visitor->visitVersionDeclaration(shared_from_this());
}
void InitialLayout::accept(InstVisitor* visitor) {
  visitor->visitInitialLayout(shared_from_this());
}
} // namespace qasm3
