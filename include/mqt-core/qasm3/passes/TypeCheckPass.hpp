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

#include "qasm3/InstVisitor.hpp"
#include "qasm3/Statement_fwd.hpp"
#include "qasm3/Types.hpp"
#include "qasm3/passes/CompilerPass.hpp"
#include "qasm3/passes/ConstEvalPass.hpp"

#include <map>
#include <memory>
#include <string>
#include <utility>

namespace qasm3 {
class GateOperand;
struct DebugInfo;
} // namespace qasm3

namespace qasm3::type_checking {
struct InferredType {
  bool isError;
  std::shared_ptr<ResolvedType> type;

  InferredType(const bool isErr, std::shared_ptr<ResolvedType> ty)
      : isError(isErr), type(std::move(ty)) {}

  explicit InferredType(std::shared_ptr<ResolvedType> ty)
      : isError(false), type(std::move(ty)) {}

  static InferredType error() { return InferredType{true, nullptr}; }

  [[nodiscard]] bool matches(const InferredType& other) const {
    if (isError || other.isError) {
      return true;
    }

    return *type == *other.type;
  }

  [[nodiscard]] std::string toString() const {
    if (isError) {
      return "error";
    }
    return type->toString();
  }
};

class TypeCheckPass final : public CompilerPass,
                            public InstVisitor,
                            public ExpressionVisitor<InferredType> {
  bool hasError = false;
  std::string errMessage;
  std::map<std::string, InferredType> env;
  // We need a reference to a const eval pass to evaluate types before type
  // checking.
  const_eval::ConstEvalPass* constEvalPass;

  InferredType error(const std::string& msg,
                     const std::shared_ptr<DebugInfo>& debugInfo = nullptr);

public:
  explicit TypeCheckPass(const_eval::ConstEvalPass& pass)
      : constEvalPass(&pass) {}

  ~TypeCheckPass() override = default;

  void addBuiltin(const std::string& identifier, const InferredType& ty) {
    env.emplace(identifier, ty);
  }

  void processStatement(Statement& statement) override;

  void checkIndexOperator(const IndexOperator& indexOperator);
  void checkIndexedIdentifier(const IndexedIdentifier& id);
  void checkGateOperand(const GateOperand& operand);

  // Types
  void
  visitGateStatement(std::shared_ptr<GateDeclaration> gateStatement) override;
  void visitVersionDeclaration(
      std::shared_ptr<VersionDeclaration> versionDeclaration) override;
  void visitDeclarationStatement(
      std::shared_ptr<DeclarationStatement> declarationStatement) override;
  void
  visitInitialLayout(std::shared_ptr<InitialLayout> initialLayout) override;
  void visitOutputPermutation(
      std::shared_ptr<OutputPermutation> outputPermutation) override;
  void visitGateCallStatement(
      std::shared_ptr<GateCallStatement> gateCallStatement) override;
  void visitAssignmentStatement(
      std::shared_ptr<AssignmentStatement> assignmentStatement) override;
  void visitBarrierStatement(
      std::shared_ptr<BarrierStatement> barrierStatement) override;
  void
  visitResetStatement(std::shared_ptr<ResetStatement> resetStatement) override;
  void visitIfStatement(std::shared_ptr<IfStatement> ifStatement) override;

  // Expressions
  InferredType visitBinaryExpression(
      std::shared_ptr<BinaryExpression> binaryExpression) override;
  InferredType visitUnaryExpression(
      std::shared_ptr<UnaryExpression> unaryExpression) override;
  InferredType
  visitConstantExpression(std::shared_ptr<Constant> constantInt) override;
  InferredType visitIdentifierExpression(
      std::shared_ptr<IdentifierExpression> identifierExpression) override;
  InferredType
  visitIdentifierList(std::shared_ptr<IdentifierList> identifierList) override;
  InferredType visitIndexedIdentifier(
      std::shared_ptr<IndexedIdentifier> indexedIdentifier) override;
  InferredType visitMeasureExpression(
      std::shared_ptr<MeasureExpression> measureExpression) override;
};
} // namespace qasm3::type_checking
