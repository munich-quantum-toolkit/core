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

#include "Statement_fwd.hpp" // IWYU pragma: export
#include "Types_fwd.hpp"
#include "ir/Permutation.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace qc {
// forward declarations
enum ComparisonKind : std::uint8_t;
} // namespace qc

namespace qasm3 {
class InstVisitor;

struct DebugInfo {
  size_t line;
  size_t column;
  std::string filename;
  std::shared_ptr<DebugInfo> parent;

  DebugInfo(const size_t l, const size_t c, std::string file,
            std::shared_ptr<DebugInfo> parentDebugInfo = nullptr)
      : line(l), column(c), filename(std::move(std::move(file))),
        parent(std::move(parentDebugInfo)) {}

  [[nodiscard]] std::string toString() const {
    return filename + ":" + std::to_string(line) + ":" + std::to_string(column);
  }
};

// Expressions
class Expression {
public:
  virtual ~Expression() = default;

  [[nodiscard]] virtual std::string getName() const = 0;
};

class DeclarationExpression final {
public:
  std::shared_ptr<Expression> expression;

  explicit DeclarationExpression(std::shared_ptr<Expression> expr)
      : expression(std::move(expr)) {}

  ~DeclarationExpression() = default;
};

class Constant final : public Expression {
  std::variant<int64_t, double, bool> val;
  bool isSigned;
  bool isFp;
  bool isBoolean;

public:
  Constant(int64_t value, const bool valueIsSigned)
      : val(value), isSigned(valueIsSigned), isFp(false), isBoolean(false) {}

  explicit Constant(double value)
      : val(value), isSigned(true), isFp(true), isBoolean(false) {}
  explicit Constant(bool value)
      : val(value), isSigned(false), isFp(false), isBoolean(true) {}

  [[nodiscard]] bool isInt() const { return !isFp; }
  [[nodiscard]] bool isSInt() const { return !isFp && isSigned; }
  [[nodiscard]] bool isUInt() const { return !isFp && !isSigned; }
  [[nodiscard]] bool isFP() const { return isFp; }
  [[nodiscard]] bool isBool() const { return isBoolean; }
  [[nodiscard]] int64_t getSInt() const { return std::get<0>(val); }
  [[nodiscard]] uint64_t getUInt() const {
    return static_cast<uint64_t>(std::get<0>(val));
  }
  [[nodiscard]] double getFP() const { return std::get<1>(val); }
  [[nodiscard]] double asFP() const {
    if (isFp) {
      return getFP();
    }
    if (isSigned) {
      return static_cast<double>(getSInt());
    }
    return static_cast<double>(getUInt());
  }
  [[nodiscard]] bool getBool() const { return std::get<2>(val); }

  [[nodiscard]] std::string getName() const override { return "Constant"; }
};

class BinaryExpression final
    : public Expression,
      public std::enable_shared_from_this<BinaryExpression> {
public:
  enum Op : uint8_t {
    Power,
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    LeftShift,
    RightShift,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    Equal,
    NotEqual,
    BitwiseAnd,
    BitwiseXor,
    BitwiseOr,
    LogicalAnd,
    LogicalOr,
  };

  Op op;
  std::shared_ptr<Expression> lhs;
  std::shared_ptr<Expression> rhs;

  BinaryExpression(const Op opcode, std::shared_ptr<Expression> l,
                   std::shared_ptr<Expression> r)
      : op(opcode), lhs(std::move(l)), rhs(std::move(r)) {}

  [[nodiscard]] std::string getName() const override { return "BinaryExpr"; }
};

std::optional<qc::ComparisonKind> getComparisonKind(BinaryExpression::Op op);

class UnaryExpression final
    : public Expression,
      public std::enable_shared_from_this<UnaryExpression> {
public:
  enum Op : uint8_t {
    BitwiseNot,
    LogicalNot,
    Negate,
    DurationOf,
    Sin,
    Cos,
    Tan,
    Exp,
    Ln,
    Sqrt,
  };

  std::shared_ptr<Expression> operand;
  Op op;

  UnaryExpression(const Op opcode, std::shared_ptr<Expression> expr)
      : operand(std::move(expr)), op(opcode) {}

  [[nodiscard]] std::string getName() const override { return "UnaryExpr"; }
};

class IdentifierExpression final
    : public Expression,
      public std::enable_shared_from_this<IdentifierExpression> {
public:
  std::string identifier;

  explicit IdentifierExpression(std::string id) : identifier(std::move(id)) {}

  [[nodiscard]] std::string getName() const override {
    return std::string{"IdentifierExpr ("} + identifier + ")";
  }
};

class IdentifierList final
    : public Expression,
      public std::enable_shared_from_this<IdentifierList> {
public:
  std::vector<std::shared_ptr<IdentifierExpression>> identifiers;

  explicit IdentifierList(
      std::vector<std::shared_ptr<IdentifierExpression>> ids)
      : identifiers(std::move(ids)) {}

  explicit IdentifierList() = default;

  [[nodiscard]] std::string getName() const override {
    return "IdentifierList";
  }
};

class IndexOperator {
public:
  std::vector<std::shared_ptr<Expression>> indexExpressions;

  explicit IndexOperator(std::vector<std::shared_ptr<Expression>> indices)
      : indexExpressions(std::move(indices)) {}
};

class IndexedIdentifier final
    : public Expression,
      public std::enable_shared_from_this<IndexedIdentifier> {
public:
  std::string identifier;
  std::vector<std::shared_ptr<IndexOperator>> indices;

  explicit IndexedIdentifier(
      std::string id, std::vector<std::shared_ptr<IndexOperator>> idxs = {})
      : identifier(std::move(id)), indices(std::move(idxs)) {}

  [[nodiscard]] std::string getName() const override {
    return std::string{"IndexedIdentifier ("} + identifier + ")";
  }
};

class GateOperand final : public Expression,
                          public std::enable_shared_from_this<GateOperand> {
public:
  std::variant<std::shared_ptr<IndexedIdentifier>, uint64_t> operand;

  explicit GateOperand(std::shared_ptr<IndexedIdentifier> id)
      : operand(std::move(id)) {}

  explicit GateOperand(const uint64_t qubit) : operand(qubit) {}

  [[nodiscard]] bool isHardwareQubit() const {
    return std::holds_alternative<uint64_t>(operand);
  }

  [[nodiscard]] uint64_t getHardwareQubit() const {
    return std::get<uint64_t>(operand);
  }

  [[nodiscard]] const std::shared_ptr<IndexedIdentifier>&
  getIdentifier() const {
    return std::get<std::shared_ptr<IndexedIdentifier>>(operand);
  }

  [[nodiscard]] std::string getName() const override {
    return isHardwareQubit() ? "$" + std::to_string(getHardwareQubit())
                             : getIdentifier()->getName();
  }
};

class MeasureExpression final
    : public Expression,
      public std::enable_shared_from_this<MeasureExpression> {
public:
  std::shared_ptr<GateOperand> gate;

  explicit MeasureExpression(std::shared_ptr<GateOperand> gateOperand)
      : gate(std::move(gateOperand)) {}

  [[nodiscard]] std::string getName() const override {
    return "MeasureExpression";
  }
};

// Statements

class Statement {
public:
  std::shared_ptr<DebugInfo> debugInfo;
  explicit Statement(std::shared_ptr<DebugInfo> debug)
      : debugInfo(std::move(debug)) {}
  virtual ~Statement() = default;

  virtual void accept(InstVisitor* visitor) = 0;
};

class QuantumStatement : public Statement {
protected:
  explicit QuantumStatement(std::shared_ptr<DebugInfo> debug)
      : Statement(std::move(debug)) {}
};

class GateDeclaration final
    : public Statement,
      public std::enable_shared_from_this<GateDeclaration> {
public:
  std::string identifier;
  std::shared_ptr<IdentifierList> parameters;
  std::shared_ptr<IdentifierList> qubits;
  std::vector<std::shared_ptr<QuantumStatement>> statements;
  bool isOpaque;

  explicit GateDeclaration(std::shared_ptr<DebugInfo> debug, std::string id,
                           std::shared_ptr<IdentifierList> params,
                           std::shared_ptr<IdentifierList> qbits,
                           std::vector<std::shared_ptr<QuantumStatement>> stmts,
                           bool opaque = false);

  void accept(InstVisitor* visitor) override;
};

class VersionDeclaration final
    : public Statement,
      public std::enable_shared_from_this<VersionDeclaration> {
public:
  double version;

  explicit VersionDeclaration(std::shared_ptr<DebugInfo> debug,
                              const double versionNum)
      : Statement(std::move(debug)), version(versionNum) {}

  void accept(InstVisitor* visitor) override;
};

class InitialLayout final : public Statement,
                            public std::enable_shared_from_this<InitialLayout> {
public:
  qc::Permutation permutation;

  explicit InitialLayout(std::shared_ptr<DebugInfo> debug, qc::Permutation perm)
      : Statement(std::move(debug)), permutation(std::move(perm)) {}

private:
  void accept(InstVisitor* visitor) override;
};

class OutputPermutation final
    : public Statement,
      public std::enable_shared_from_this<OutputPermutation> {
public:
  qc::Permutation permutation;

  explicit OutputPermutation(std::shared_ptr<DebugInfo> debug,
                             qc::Permutation perm)
      : Statement(std::move(debug)), permutation(std::move(perm)) {}

private:
  void accept(InstVisitor* visitor) override;
};

class DeclarationStatement final
    : public Statement,
      public std::enable_shared_from_this<DeclarationStatement> {
public:
  bool isConst;
  std::variant<std::shared_ptr<TypeExpr>, std::shared_ptr<ResolvedType>> type;
  std::string identifier;
  std::shared_ptr<DeclarationExpression> expression;

  DeclarationStatement(std::shared_ptr<DebugInfo> debug, const bool declIsConst,
                       std::shared_ptr<TypeExpr> ty, std::string id,
                       std::shared_ptr<DeclarationExpression> expr)
      : Statement(std::move(debug)), isConst(declIsConst), type(ty),
        identifier(std::move(id)), expression(std::move(expr)) {}

  void accept(InstVisitor* visitor) override;
};

class GateModifier : public std::enable_shared_from_this<GateModifier> {
public:
  virtual ~GateModifier() = default;
};

class InvGateModifier final
    : public GateModifier,
      public std::enable_shared_from_this<InvGateModifier> {};

class PowGateModifier final
    : public GateModifier,
      public std::enable_shared_from_this<PowGateModifier> {
public:
  std::shared_ptr<Expression> expression;

  explicit PowGateModifier(std::shared_ptr<Expression> expr)
      : expression(std::move(expr)) {}
};

class CtrlGateModifier final
    : public GateModifier,
      public std::enable_shared_from_this<CtrlGateModifier> {
public:
  bool ctrlType;
  std::shared_ptr<Expression> expression;

  explicit CtrlGateModifier(const bool ty, std::shared_ptr<Expression> expr)
      : ctrlType(ty), expression(std::move(expr)) {}
};

class GateCallStatement final
    : public QuantumStatement,
      public std::enable_shared_from_this<GateCallStatement> {
public:
  std::string identifier;
  std::vector<std::shared_ptr<GateModifier>> modifiers;
  std::vector<std::shared_ptr<Expression>> arguments;
  std::vector<std::shared_ptr<GateOperand>> operands;

  GateCallStatement(std::shared_ptr<DebugInfo> debug, std::string id,
                    std::vector<std::shared_ptr<GateModifier>> modifierList,
                    std::vector<std::shared_ptr<Expression>> argumentList,
                    std::vector<std::shared_ptr<GateOperand>> operandList)
      : QuantumStatement(std::move(debug)), identifier(std::move(id)),
        modifiers(std::move(modifierList)), arguments(std::move(argumentList)),
        operands(std::move(operandList)) {}

  void accept(InstVisitor* visitor) override;
};

class AssignmentStatement final
    : public Statement,
      public std::enable_shared_from_this<AssignmentStatement> {
public:
  enum Type : uint8_t {
    Assignment,
    PlusAssignment,
    MinusAssignment,
    TimesAssignment,
    DivAssignment,
    BitwiseAndAssignment,
    BitwiseOrAssignment,
    BitwiseNotAssignment,
    BitwiseXorAssignment,
    LeftShiftAssignment,
    RightShiftAssignment,
    ModuloAssignment,
    PowerAssignment,
  } type;
  std::shared_ptr<IndexedIdentifier> identifier;
  std::shared_ptr<DeclarationExpression> expression;

  AssignmentStatement(std::shared_ptr<DebugInfo> debug, const Type ty,
                      std::shared_ptr<IndexedIdentifier> id,
                      std::shared_ptr<DeclarationExpression> expr)
      : Statement(std::move(debug)), type(ty), identifier(std::move(id)),
        expression(std::move(expr)) {}

  void accept(InstVisitor* visitor) override;
};

class BarrierStatement final
    : public QuantumStatement,
      public std::enable_shared_from_this<BarrierStatement> {
public:
  std::vector<std::shared_ptr<GateOperand>> gates;

  explicit BarrierStatement(std::shared_ptr<DebugInfo> debug,
                            std::vector<std::shared_ptr<GateOperand>> gateList)
      : QuantumStatement(std::move(debug)), gates(std::move(gateList)) {}

  void accept(InstVisitor* visitor) override;
};

class ResetStatement final
    : public QuantumStatement,
      public std::enable_shared_from_this<ResetStatement> {
public:
  std::shared_ptr<GateOperand> gate;

  explicit ResetStatement(std::shared_ptr<DebugInfo> debug,
                          std::shared_ptr<GateOperand> g)
      : QuantumStatement(std::move(debug)), gate(std::move(g)) {}

  void accept(InstVisitor* visitor) override;
};

class IfStatement final : public Statement,
                          public std::enable_shared_from_this<IfStatement> {
public:
  std::shared_ptr<Expression> condition;
  std::vector<std::shared_ptr<Statement>> thenStatements;
  std::vector<std::shared_ptr<Statement>> elseStatements;

  IfStatement(const std::shared_ptr<Expression>& cond,
              const std::vector<std::shared_ptr<Statement>>& thenStmts,
              const std::vector<std::shared_ptr<Statement>>& elseStmts,
              std::shared_ptr<DebugInfo> debug)
      : Statement(std::move(debug)), condition(cond), thenStatements(thenStmts),
        elseStatements(elseStmts) {}

  void accept(InstVisitor* visitor) override;
};
} // namespace qasm3
