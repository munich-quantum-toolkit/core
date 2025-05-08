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
#include "qasm3/NestedEnvironment.hpp"
#include "qasm3/Statement_fwd.hpp"
#include "qasm3/passes/CompilerPass.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>

namespace qasm3::const_eval {
struct ConstEvalValue {
  enum Type : uint8_t {
    ConstInt,
    ConstUint,
    ConstFloat,
    ConstBool,
  } type;
  std::variant<int64_t, double, bool> value;
  size_t width;

  explicit ConstEvalValue(double val, const size_t w = 64)
      : type(ConstFloat), value(val), width(w) {}
  explicit ConstEvalValue(int64_t val, const bool isSigned, const size_t w = 64)
      : type(isSigned ? ConstInt : ConstUint), value(val), width(w) {}
  explicit ConstEvalValue(bool val) : type(ConstBool), value(val), width(1) {}

  [[nodiscard]] std::shared_ptr<Constant> toExpr() const;

  bool operator==(const ConstEvalValue& rhs) const;

  bool operator!=(const ConstEvalValue& rhs) const { return !(*this == rhs); }

  [[nodiscard]] std::string toString() const;
};

class ConstEvalPass final
    : public CompilerPass,
      public DefaultInstVisitor,
      public ExpressionVisitor<std::optional<ConstEvalValue>>,
      public TypeVisitor<std::shared_ptr<Expression>> {
  NestedEnvironment<ConstEvalValue> env;

public:
  ConstEvalPass() = default;
  ~ConstEvalPass() override = default;

  void addConst(const std::string& identifier, const ConstEvalValue& val) {
    env.emplace(identifier, val);
  }

  void addConst(const std::string& identifier, const double val) {
    env.emplace(identifier, ConstEvalValue(val));
  }

  void processStatement(Statement& statement) override;

  void pushEnv() { env.push(); }
  void popEnv() { env.pop(); }

  void visitDeclarationStatement(
      std::shared_ptr<DeclarationStatement> declarationStatement) override;
  void visitGateCallStatement(
      std::shared_ptr<GateCallStatement> gateCallStatement) override;

  std::optional<ConstEvalValue> visitBinaryExpression(
      std::shared_ptr<BinaryExpression> binaryExpression) override;
  std::optional<ConstEvalValue> visitUnaryExpression(
      std::shared_ptr<UnaryExpression> unaryExpression) override;
  std::optional<ConstEvalValue>
  visitConstantExpression(std::shared_ptr<Constant> constant) override;
  std::optional<ConstEvalValue> visitIdentifierExpression(
      std::shared_ptr<IdentifierExpression> identifierExpression) override;
  std::optional<ConstEvalValue>
  visitIdentifierList(std::shared_ptr<IdentifierList> identifierList) override;
  std::optional<ConstEvalValue> visitIndexedIdentifier(
      std::shared_ptr<IndexedIdentifier> indexedIdentifier) override;
  std::optional<ConstEvalValue> visitMeasureExpression(
      std::shared_ptr<MeasureExpression> measureExpression) override;

  std::shared_ptr<ResolvedType> visitDesignatedType(
      DesignatedType<std::shared_ptr<Expression>>* designatedType) override;
  std::shared_ptr<ResolvedType> visitUnsizedType(
      UnsizedType<std::shared_ptr<Expression>>* unsizedType) override;
  std::shared_ptr<ResolvedType>
  visitArrayType(ArrayType<std::shared_ptr<Expression>>* arrayType) override;
};
} // namespace qasm3::const_eval
