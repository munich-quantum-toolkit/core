/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include <llvm/ADT/StringRef.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace llvm {
class SourceMgr;
} // namespace llvm

namespace mlir::oq3::frontend {

using ExpressionId = std::uint32_t;
using RegisterId = std::uint32_t;
using ScalarId = std::uint32_t;
using ConditionId = std::uint32_t;
using StatementId = std::uint32_t;

/// Maximum number of leaves materialized for one structured dynamic-qubit
/// dispatch. This bounds the Cartesian expansion of multiple dynamic operands.
inline constexpr std::size_t kDynamicQubitDispatchLeafLimit = 4096;

struct SourceLocation {
  std::string filename = "<input>";
  std::uint32_t line = 1;
  std::uint32_t column = 1;
};

struct Diagnostic {
  SourceLocation location;
  std::string message;
};

enum class GatePolicy : std::uint8_t {
  Strict,
  MQTCompatibility,
};

struct FrontendOptions {
  GatePolicy gatePolicy = GatePolicy::MQTCompatibility;
};

struct AnalysisResult;
struct ParseResult;

class ParsedProgram {
public:
  ParsedProgram(ParsedProgram&&) noexcept;
  ParsedProgram& operator=(ParsedProgram&&) noexcept;
  ~ParsedProgram();

  ParsedProgram(const ParsedProgram&) = delete;
  ParsedProgram& operator=(const ParsedProgram&) = delete;

private:
  struct Impl;
  std::unique_ptr<Impl> impl;

  explicit ParsedProgram(std::unique_ptr<Impl> implementation);

  friend ParseResult parseOpenQASM(llvm::StringRef);
  friend ParseResult parseOpenQASM(llvm::SourceMgr&);
  friend AnalysisResult analyzeOpenQASM(const ParsedProgram&,
                                        const FrontendOptions&);
};

struct ParseResult {
  std::unique_ptr<ParsedProgram> program;
  std::vector<Diagnostic> diagnostics;

  [[nodiscard]] explicit operator bool() const { return program != nullptr; }
};

enum class ScalarType : std::uint8_t {
  Bool,
  Int,
  Uint,
  Float,
};

enum class ExpressionKind : std::uint8_t {
  Constant,
  GateParameter,
  Variable,
  Negate,
  ArcCos,
  ArcSin,
  ArcTan,
  Sin,
  Cos,
  Tan,
  Exp,
  Ln,
  Sqrt,
  Add,
  Subtract,
  Multiply,
  Divide,
  Modulo,
  Power,
};

struct ScalarExpression {
  ExpressionKind kind = ExpressionKind::Constant;
  ScalarType type = ScalarType::Float;
  std::variant<bool, std::int64_t, std::uint64_t, double> constant = 0.0;
  std::uint32_t parameter = 0;
  ScalarId variable = 0;
  ExpressionId lhs = 0;
  ExpressionId rhs = 0;
};

struct ScalarDeclaration {
  ScalarType type = ScalarType::Int;
  std::string name;
};

enum class RegisterKind : std::uint8_t {
  Qubit,
  Bit,
};

struct RegisterDeclaration {
  RegisterKind kind = RegisterKind::Qubit;
  std::string name;
  std::uint64_t width = 0;
  SourceLocation location;
};

enum class QubitReferenceKind : std::uint8_t {
  Register,
  GateArgument,
  Hardware,
};

struct QubitReference {
  QubitReferenceKind kind = QubitReferenceKind::Register;
  std::uint32_t symbol = 0;
  std::uint64_t index = 0;
  std::optional<ExpressionId> dynamicIndex;

  bool operator==(const QubitReference&) const = default;
};

struct BitReference {
  RegisterId reg = 0;
  std::uint64_t index = 0;
  std::optional<ExpressionId> dynamicIndex;
};

enum class ComparisonKind : std::uint8_t {
  Equal,
  NotEqual,
  Less,
  LessEqual,
  Greater,
  GreaterEqual,
};

enum class ConditionKind : std::uint8_t {
  Literal,
  Scalar,
  Bit,
  Measurement,
  Not,
  And,
  Or,
  Comparison,
};

struct ConditionExpression {
  ConditionKind kind = ConditionKind::Literal;
  SourceLocation location;
  bool literal = false;
  ScalarId scalar = 0;
  BitReference bit;
  QubitReference measurement;
  ConditionId lhs = 0;
  ConditionId rhs = 0;
  ExpressionId comparisonLhs = 0;
  ExpressionId comparisonRhs = 0;
  ComparisonKind comparison = ComparisonKind::Equal;
};

enum class ModifierKind : std::uint8_t {
  Inv,
  Ctrl,
  NegCtrl,
  Pow,
};

struct GateModifier {
  ModifierKind kind = ModifierKind::Inv;
  std::optional<ExpressionId> operand;
};

struct GateApplication {
  std::string callee;
  std::vector<ExpressionId> parameters;
  std::vector<QubitReference> qubits;
  std::vector<GateModifier> modifiers;
};

struct GateDefinition {
  std::string name;
  std::size_t parameterCount = 0;
  std::size_t qubitCount = 0;
  std::vector<StatementId> body;
  SourceLocation location;
};

struct DeclarationStatement {
  RegisterId reg = 0;
};

struct ScalarDeclarationStatement {
  ScalarId scalar = 0;
  std::optional<ExpressionId> initializer;
  std::optional<ConditionId> conditionInitializer;
};

struct ScalarAssignmentStatement {
  ScalarId scalar = 0;
  std::optional<ExpressionId> value;
  std::optional<ConditionId> condition;
};

struct BitAssignmentStatement {
  BitReference target;
  ConditionId value = 0;
};

struct MeasurementStatement {
  std::vector<BitReference> targets;
  std::vector<QubitReference> qubits;
};

struct ResetStatement {
  std::vector<QubitReference> qubits;
};

struct BarrierStatement {
  std::vector<QubitReference> qubits;
};

struct IfStatement {
  ConditionId condition = 0;
  std::vector<StatementId> thenStatements;
  std::vector<StatementId> elseStatements;
};

struct ForStatement {
  ScalarId inductionVariable = 0;
  ExpressionId start = 0;
  ExpressionId step = 0;
  ExpressionId stop = 0;
  std::vector<StatementId> body;
};

struct WhileStatement {
  ConditionId condition = 0;
  std::vector<StatementId> body;
};

using StatementData =
    std::variant<DeclarationStatement, ScalarDeclarationStatement,
                 ScalarAssignmentStatement, BitAssignmentStatement,
                 GateApplication, MeasurementStatement, ResetStatement,
                 BarrierStatement, IfStatement, ForStatement, WhileStatement>;

struct Statement {
  StatementData data;
  SourceLocation location;
};

struct TypedProgram {
  bool openQASM2 = false;
  GatePolicy gatePolicy = GatePolicy::MQTCompatibility;
  bool standardLibraryIncluded = false;
  std::vector<ScalarExpression> expressions;
  std::vector<ConditionExpression> conditions;
  std::vector<ScalarDeclaration> scalars;
  std::vector<RegisterDeclaration> registers;
  std::vector<GateDefinition> gates;
  std::vector<Statement> statements;
  std::vector<StatementId> body;
  std::vector<RegisterId> outputs;
};

struct AnalysisResult {
  std::unique_ptr<TypedProgram> program;
  std::vector<Diagnostic> diagnostics;

  [[nodiscard]] explicit operator bool() const { return program != nullptr; }
};

[[nodiscard]] ParseResult parseOpenQASM(llvm::SourceMgr& sourceMgr);

[[nodiscard]] ParseResult parseOpenQASM(llvm::StringRef source);

[[nodiscard]] AnalysisResult
analyzeOpenQASM(const ParsedProgram& program,
                const FrontendOptions& options = {});

[[nodiscard]] AnalysisResult
analyzeOpenQASM(llvm::SourceMgr& sourceMgr,
                const FrontendOptions& options = {});

[[nodiscard]] AnalysisResult
analyzeOpenQASM(llvm::StringRef source, const FrontendOptions& options = {});

} // namespace mlir::oq3::frontend
