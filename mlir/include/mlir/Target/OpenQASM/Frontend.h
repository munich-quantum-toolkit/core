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

#include <llvm/ADT/APInt.h>
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

using ExpressionId = uint32_t;
using BitVectorExpressionId = uint32_t;
using RegisterId = uint32_t;
using ScalarId = uint32_t;
using ConditionId = uint32_t;
using StatementId = uint32_t;

struct SourceLocation {
  std::string filename = "<input>";
  uint32_t line = 1;
  uint32_t column = 1;
  struct IncludeFrame {
    std::string filename;
    uint32_t line = 1;
    uint32_t column = 1;
  };
  /// Include sites from the immediate parent through the main source.
  std::vector<IncludeFrame> includeStack;
};

struct Diagnostic {
  SourceLocation location;
  std::string message;
};

enum class GatePolicy : uint8_t {
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

  friend ParseResult parseOpenQASM(llvm::StringRef source);
  friend ParseResult parseOpenQASM(llvm::SourceMgr& sourceMgr);
  friend AnalysisResult analyzeOpenQASM(const ParsedProgram& program,
                                        const FrontendOptions& options);
};

struct ParseResult {
  std::unique_ptr<ParsedProgram> program;
  std::vector<Diagnostic> diagnostics;

  [[nodiscard]] explicit operator bool() const { return program != nullptr; }
};

enum class ScalarType : uint8_t {
  Bool,
  Int,
  Uint,
  Float,
  Angle,
};

enum class ExpressionKind : uint8_t {
  Constant,
  GateParameter,
  Variable,
  Cast,
  Negate,
  ArcCos,
  ArcSin,
  ArcTan,
  Ceiling,
  Sin,
  Cos,
  Floor,
  Tan,
  Exp,
  Log,
  Sqrt,
  PopCount,
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
  std::variant<bool, int64_t, uint64_t, double> constant = 0.0;
  uint32_t parameter = 0;
  ScalarId variable = 0;
  ExpressionId lhs = 0;
  ExpressionId rhs = 0;
  BitVectorExpressionId bitVector = 0;
};

enum class BitVectorExpressionKind : uint8_t {
  Register,
  RotateLeft,
  RotateRight,
};

struct BitVectorExpression {
  BitVectorExpressionKind kind = BitVectorExpressionKind::Register;
  uint64_t width = 0;
  RegisterId reg = 0;
  BitVectorExpressionId operand = 0;
  ExpressionId distance = 0;
};

struct ScalarDeclaration {
  ScalarType type = ScalarType::Int;
  std::string name;
  SourceLocation location;
};

enum class RegisterKind : uint8_t {
  Qubit,
  Bit,
};

struct RegisterDeclaration {
  RegisterKind kind = RegisterKind::Qubit;
  std::string name;
  uint64_t width = 0;
  bool isScalar = false;
  SourceLocation location;
};

enum class QubitReferenceKind : uint8_t {
  Register,
  GateArgument,
  Hardware,
};

struct QubitReference {
  QubitReferenceKind kind = QubitReferenceKind::Register;
  uint32_t symbol = 0;
  uint64_t index = 0;
  /// A nonconstant register index proven safe by semantic analysis.
  std::optional<ExpressionId> provenIndex;

  bool operator==(const QubitReference&) const = default;
};

struct BitReference {
  RegisterId reg = 0;
  uint64_t index = 0;
  std::optional<ExpressionId> dynamicIndex;
};

enum class ComparisonKind : uint8_t {
  Equal,
  NotEqual,
  Less,
  LessEqual,
  Greater,
  GreaterEqual,
};

enum class ConditionKind : uint8_t {
  Literal,
  Scalar,
  Bit,
  Measurement,
  Not,
  And,
  Or,
  RegisterComparison,
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
  RegisterId reg = 0;
  llvm::APInt expected = llvm::APInt(1, 0);
  ExpressionId comparisonLhs = 0;
  ExpressionId comparisonRhs = 0;
  ComparisonKind comparison = ComparisonKind::Equal;
};

enum class ModifierKind : uint8_t {
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
  size_t parameterCount = 0;
  size_t qubitCount = 0;
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

struct BitVectorAssignmentStatement {
  RegisterId target = 0;
  BitVectorExpressionId value = 0;
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
  /// Whether the inclusive range has a positive constant step and all range
  /// arithmetic is proven to fit the frontend's signed 64-bit index model.
  bool provenPositiveRange = false;
  std::vector<StatementId> body;
};

struct WhileStatement {
  ConditionId condition = 0;
  std::vector<StatementId> body;
};

struct SwitchCase {
  std::vector<int64_t> labels;
  std::vector<StatementId> body;
};

struct SwitchStatement {
  ExpressionId control = 0;
  std::vector<SwitchCase> cases;
  std::vector<StatementId> defaultStatements;
};

using StatementData =
    std::variant<DeclarationStatement, ScalarDeclarationStatement,
                 ScalarAssignmentStatement, BitAssignmentStatement,
                 BitVectorAssignmentStatement, GateApplication,
                 MeasurementStatement, ResetStatement, BarrierStatement,
                 IfStatement, ForStatement, WhileStatement, SwitchStatement>;

struct Statement {
  StatementData data;
  SourceLocation location;
};

enum class OutputKind : uint8_t {
  Scalar,
  BitRegister,
};

struct ProgramOutput {
  OutputKind kind = OutputKind::Scalar;
  uint32_t symbol = 0;
};

struct TypedProgram {
  bool openQASM2 = false;
  bool stdGatesIncluded = false;
  bool qelib1Included = false;
  std::vector<ScalarExpression> expressions;
  std::vector<BitVectorExpression> bitVectorExpressions;
  std::vector<ConditionExpression> conditions;
  std::vector<ScalarDeclaration> scalars;
  std::vector<RegisterDeclaration> registers;
  std::vector<GateDefinition> gates;
  std::vector<Statement> statements;
  std::vector<StatementId> body;
  std::vector<ProgramOutput> outputs;
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
