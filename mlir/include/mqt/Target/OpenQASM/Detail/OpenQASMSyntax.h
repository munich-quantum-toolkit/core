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

#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/SMLoc.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace mlir::openqasm::frontend::detail {

using SyntaxExpressionId = uint32_t;

/// An exact OpenQASM version, preserving the decimal minor component.
struct Version {
  uint32_t major = 0;
  uint32_t minor = 0;
};

enum class ScalarKind : uint8_t { Bool, Int, Uint, Float, Angle };

/// @defgroup ParseVocabulary Parser vocabulary
/// The vocabulary the parser hands to a sink.
///
/// Expressions use IDs in the persistent syntax arena. Gate-call arrays borrow
/// parser-local storage until the builder records the statement.

/// @ingroup ParseVocabulary
/// A (sub-)expression.
///
/// Expression kinds shared by parsing and semantic analysis.
struct Expr {
  enum class Kind : uint8_t {
    Int,
    Float,
    BitString,
    FloatCast,
    Bool,
    Identifier,
    IntCast,
    BoolCast,
    BitCast,
    UintCast,
    AngleCast,
    Index,
    Neg,
    Not,
    BitNot,
    Add,
    Sub,
    Mul,
    Div,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    And,
    Or,
    BitAnd,
    BitOr,
    BitXor,
    ShiftLeft,
    ShiftRight,
    /// Built-in math functions
    ArcCos,
    ArcSin,
    ArcTan,
    Ceiling,
    Cos,
    Exp,
    Floor,
    Log,
    Mod,
    BuiltinMod,
    PopCount,
    BuiltinPow,
    Pow,
    RotateLeft,
    RotateRight,
    Sin,
    Sqrt,
    Tan,
  };
};

/// Get the kind of the built-in math function @p name.
[[nodiscard]] inline std::optional<Expr::Kind>
getMathFunctionKind(StringRef name) {
  return llvm::StringSwitch<std::optional<Expr::Kind>>(name)
      .Case("arccos", Expr::Kind::ArcCos)
      .Case("arcsin", Expr::Kind::ArcSin)
      .Case("arctan", Expr::Kind::ArcTan)
      .Case("ceiling", Expr::Kind::Ceiling)
      .Case("cos", Expr::Kind::Cos)
      .Case("exp", Expr::Kind::Exp)
      .Case("floor", Expr::Kind::Floor)
      .Case("log", Expr::Kind::Log)
      .Case("mod", Expr::Kind::BuiltinMod)
      .Case("popcount", Expr::Kind::PopCount)
      .Case("pow", Expr::Kind::BuiltinPow)
      .Case("rotl", Expr::Kind::RotateLeft)
      .Case("rotr", Expr::Kind::RotateRight)
      .Case("sin", Expr::Kind::Sin)
      .Case("sqrt", Expr::Kind::Sqrt)
      .Case("tan", Expr::Kind::Tan)
      .Default(std::nullopt);
}

/// @ingroup ParseVocabulary
/// A gate modifier: `inv @`, `pow(e) @`, `ctrl(e) @`, or `negctrl(e) @`.
struct Modifier {
  enum class Kind : uint8_t { Inv, Pow, Ctrl, NegCtrl };
  Kind kind = Kind::Inv;
  std::optional<SyntaxExpressionId> argument = std::nullopt;
};

/// @ingroup ParseVocabulary
/// A gate operand: a (possibly indexed) identifier, or a hardware qubit.
struct Operand {
  SMLoc location;
  StringRef identifier;
  std::optional<SyntaxExpressionId> index = std::nullopt;
  std::optional<uint64_t> hardwareQubit;
};

/// A (possibly indexed) classical reference (e.g., `c` or `c[0]`).
struct BitReference {
  SMLoc location;
  StringRef identifier;
  std::optional<SyntaxExpressionId> index = std::nullopt;
};

/// @ingroup ParseVocabulary
/// A parsed gate call.
///
/// Array members are borrowed for the duration of the sink call.
struct GateCall {
  SMLoc loc;
  StringRef identifier;
  ArrayRef<Modifier> modifiers;
  ArrayRef<SyntaxExpressionId> parameters;
  ArrayRef<Operand> operands;
};

using SyntaxStatementId = uint32_t;
using SyntaxIncludeContextId = size_t;

struct SyntaxExpression {
  Expr::Kind kind = Expr::Kind::Int;
  SMLoc location;
  uint64_t integer = 0;
  double floatingPoint = 0.0;
  bool boolean = false;
  StringRef identifier;
  StringRef wideInteger;
  std::optional<uint64_t> hardwareQubit;
  std::optional<SyntaxExpressionId> lhs;
  std::optional<SyntaxExpressionId> rhs;
};

struct SyntaxGateCall {
  SMLoc location;
  StringRef identifier;
  std::vector<Modifier> modifiers;
  std::vector<SyntaxExpressionId> parameters;
  std::vector<Operand> operands;
};

struct SyntaxScalarDeclaration {
  ScalarKind kind = ScalarKind::Int;
  StringRef identifier;
  std::optional<SyntaxExpressionId> size;
  std::optional<SyntaxExpressionId> initializer;
  bool isConst = false;
  bool output = false;
};

struct SyntaxAssignment {
  BitReference target;
  SyntaxExpressionId value = 0;
};

struct SyntaxQubitDeclaration {
  StringRef identifier;
  std::optional<SyntaxExpressionId> size;
};

struct SyntaxBitDeclaration {
  StringRef identifier;
  std::optional<SyntaxExpressionId> size;
  std::optional<SyntaxExpressionId> initializer;
  bool output = false;
};

struct SyntaxMeasurement {
  std::optional<BitReference> target;
  Operand source;
};

struct SyntaxReset {
  Operand operand;
};

struct SyntaxBarrier {
  std::vector<Operand> operands;
};

struct SyntaxGateDefinition {
  StringRef identifier;
  std::vector<StringRef> parameters;
  std::vector<StringRef> qubits;
  std::vector<SyntaxStatementId> body;
};

struct SyntaxIf {
  SyntaxExpressionId condition = 0;
  std::vector<SyntaxStatementId> thenStatements;
  std::vector<SyntaxStatementId> elseStatements;
};

struct SyntaxFor {
  StringRef inductionVariable;
  bool isUnsigned = false;
  SyntaxExpressionId start = 0;
  SyntaxExpressionId step = 0;
  SyntaxExpressionId stop = 0;
  std::vector<SyntaxStatementId> body;
};

struct SyntaxBreak {};
struct SyntaxContinue {};

struct SyntaxWhile {
  SyntaxExpressionId condition = 0;
  std::vector<SyntaxStatementId> body;
};

struct SyntaxSwitchCase {
  std::vector<SyntaxExpressionId> labels;
  std::vector<SyntaxStatementId> body;
};

struct SyntaxSwitch {
  SyntaxExpressionId control = 0;
  std::vector<SyntaxSwitchCase> cases;
  std::vector<SyntaxStatementId> defaultStatements;
};

enum class StandardLibraryKind : uint8_t {
  StdGates,
  QELib1,
};

struct SyntaxStandardLibraryInclude {
  StandardLibraryKind kind = StandardLibraryKind::StdGates;
};

using SyntaxStatementData =
    std::variant<SyntaxStandardLibraryInclude, SyntaxScalarDeclaration,
                 SyntaxAssignment, SyntaxQubitDeclaration, SyntaxBitDeclaration,
                 SyntaxMeasurement, SyntaxReset, SyntaxBarrier, SyntaxGateCall,
                 SyntaxGateDefinition, SyntaxIf, SyntaxFor, SyntaxWhile,
                 SyntaxSwitch, SyntaxBreak, SyntaxContinue>;

struct SyntaxStatement {
  SMLoc location;
  SyntaxStatementData data;
};

struct SyntaxInclude {
  SMLoc location;
  StringRef filename;
  size_t bodyOffset = 0;
};

struct SyntaxIncludeContext {
  SMLoc location;
  std::optional<SyntaxIncludeContextId> parent;
};

struct SyntaxProgram {
  std::optional<Version> version;
  SMLoc versionLocation;
  std::vector<SyntaxInclude> includes;
  std::vector<SyntaxIncludeContext> includeContexts;
  std::vector<SyntaxExpression> expressions;
  std::vector<SyntaxStatement> statements;
  std::vector<SyntaxStatementId> body;
  /// Expansion-site include context for each statement in `body`.
  std::vector<std::optional<SyntaxIncludeContextId>> bodyIncludeContexts;
};

struct SyntaxDiagnostic {
  SMLoc location;
  std::string message;
};

class SyntaxBuilder {
public:
  [[nodiscard]] SyntaxExpressionId
  addExpression(const SyntaxExpression& expression);
  [[nodiscard]] LogicalResult error(SMLoc location, const Twine& message);
  [[nodiscard]] LogicalResult version(SMLoc location, Version value);
  [[nodiscard]] LogicalResult include(SMLoc location, StringRef filename);
  [[nodiscard]] SyntaxStatementId
  standardLibraryInclude(SMLoc location, StandardLibraryKind kind);
  [[nodiscard]] LogicalResult
  scalarDecl(SMLoc location, ScalarKind kind, StringRef identifier,
             std::optional<SyntaxExpressionId> size,
             std::optional<SyntaxExpressionId> initializer, bool isConst,
             bool output);
  [[nodiscard]] LogicalResult assignment(SMLoc location,
                                         const BitReference& target,
                                         SyntaxExpressionId value);
  [[nodiscard]] LogicalResult
  qubitRegister(SMLoc location, StringRef identifier,
                std::optional<SyntaxExpressionId> size);
  [[nodiscard]] LogicalResult
  classicalRegister(SMLoc location, StringRef identifier,
                    std::optional<SyntaxExpressionId> size,
                    std::optional<SyntaxExpressionId> initializer, bool output);
  [[nodiscard]] LogicalResult
  measure(SMLoc location, const BitReference* target, const Operand& source);
  [[nodiscard]] LogicalResult reset(SMLoc location, const Operand& operand);
  [[nodiscard]] LogicalResult barrier(SMLoc location,
                                      ArrayRef<Operand> operands);
  [[nodiscard]] LogicalResult gateCall(const GateCall& call);
  [[nodiscard]] LogicalResult
  gateDefinition(SMLoc location, StringRef identifier,
                 ArrayRef<StringRef> parameters, ArrayRef<StringRef> qubits,
                 function_ref<LogicalResult()> continuation);
  [[nodiscard]] LogicalResult
  ifStmt(SMLoc location, SyntaxExpressionId condition,
         function_ref<LogicalResult()> thenContinuation,
         function_ref<LogicalResult()> elseContinuation);
  [[nodiscard]] LogicalResult
  forStmt(SMLoc location, StringRef inductionVariable, bool isUnsigned,
          SyntaxExpressionId start, SyntaxExpressionId step,
          SyntaxExpressionId stop, function_ref<LogicalResult()> continuation);
  [[nodiscard]] LogicalResult breakStmt(SMLoc location);
  [[nodiscard]] LogicalResult continueStmt(SMLoc location);
  [[nodiscard]] LogicalResult
  whileStmt(SMLoc location, SyntaxExpressionId condition,
            function_ref<LogicalResult()> continuation);
  [[nodiscard]] LogicalResult
  switchStmt(SMLoc location, SyntaxExpressionId control,
             function_ref<LogicalResult()> continuation);
  [[nodiscard]] LogicalResult
  switchCase(SMLoc location, ArrayRef<SyntaxExpressionId> labels,
             function_ref<LogicalResult()> continuation);
  [[nodiscard]] LogicalResult
  switchDefault(SMLoc location, function_ref<LogicalResult()> continuation);

  [[nodiscard]] SyntaxProgram takeProgram() { return std::move(program); }
  [[nodiscard]] const std::vector<SyntaxDiagnostic>& getDiagnostics() const {
    return diagnostics;
  }
  [[nodiscard]] ArrayRef<SyntaxInclude> getIncludes() const {
    return program.includes;
  }
  [[nodiscard]] ArrayRef<SyntaxStatementId> getBody() const {
    return program.body;
  }
  void replaceBody(
      std::vector<SyntaxStatementId> body,
      std::vector<std::optional<SyntaxIncludeContextId>> includeContexts,
      std::vector<SyntaxIncludeContext> contexts);

private:
  [[nodiscard]] SyntaxGateCall copyGateCall(const GateCall& call);
  [[nodiscard]] SyntaxStatementId addStatement(SMLoc location,
                                               SyntaxStatementData data);
  [[nodiscard]] FailureOr<std::vector<SyntaxStatementId>>
  parseNestedBody(function_ref<LogicalResult()> continuation);

  SyntaxProgram program;
  std::vector<SyntaxDiagnostic> diagnostics;
  SmallVector<std::vector<SyntaxStatementId>*> bodyStack{&program.body};
  SmallVector<SyntaxSwitch*> switchStack;
  bool sawConstruct = false;
};

} // namespace mlir::openqasm::frontend::detail
