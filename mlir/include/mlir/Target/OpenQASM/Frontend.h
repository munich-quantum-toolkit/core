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
using StatementId = std::uint32_t;

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

  friend struct ParseResult;
  friend struct AnalysisResult;
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
  Negate,
  BitwiseNot,
  LogicalNot,
  Add,
  Subtract,
  Multiply,
  Divide,
  Power,
};

struct ScalarExpression {
  ExpressionKind kind = ExpressionKind::Constant;
  ScalarType type = ScalarType::Float;
  std::variant<bool, std::int64_t, std::uint64_t, double> constant = 0.0;
  std::uint32_t parameter = 0;
  ExpressionId lhs = 0;
  ExpressionId rhs = 0;
};

enum class RegisterKind : std::uint8_t {
  Qubit,
  Bit,
  Int,
  Uint,
};

struct RegisterDeclaration {
  RegisterId id = 0;
  RegisterKind kind = RegisterKind::Qubit;
  std::string name;
  std::uint64_t width = 0;
  bool output = false;
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
};

struct BitReference {
  RegisterId reg = 0;
  std::uint64_t index = 0;
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
  SourceLocation location;
};

struct GateDefinition {
  std::string name;
  std::vector<std::string> parameterNames;
  std::vector<std::string> qubitNames;
  std::vector<GateApplication> body;
  SourceLocation location;
};

struct DeclarationStatement {
  RegisterId reg = 0;
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
  BitReference condition;
  bool negated = false;
  std::vector<StatementId> thenStatements;
  std::vector<StatementId> elseStatements;
};

using StatementData =
    std::variant<DeclarationStatement, GateApplication, MeasurementStatement,
                 ResetStatement, BarrierStatement, IfStatement>;

struct Statement {
  StatementData data;
  SourceLocation location;
};

struct TypedProgram {
  bool openQASM2 = false;
  GatePolicy gatePolicy = GatePolicy::MQTCompatibility;
  bool standardLibraryIncluded = false;
  std::vector<ScalarExpression> expressions;
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
