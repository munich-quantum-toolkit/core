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

#include "mlir/Target/OpenQASM/Detail/OpenQASMParser.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/SMLoc.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace mlir::oq3::frontend::detail {

using SyntaxExpressionId = std::uint32_t;
using SyntaxStatementId = std::uint32_t;

struct SyntaxExpression {
  Expr::Kind kind = Expr::Kind::Int;
  SMLoc location;
  std::uint64_t integer = 0;
  double floatingPoint = 0.0;
  bool boolean = false;
  StringRef identifier;
  std::optional<std::uint64_t> hardwareQubit;
  std::optional<SyntaxExpressionId> lhs;
  std::optional<SyntaxExpressionId> rhs;
};

struct SyntaxOperand {
  SMLoc location;
  StringRef identifier;
  std::optional<SyntaxExpressionId> index;
  std::optional<std::uint64_t> hardwareQubit;
};

struct SyntaxBitReference {
  SMLoc location;
  StringRef identifier;
  std::optional<SyntaxExpressionId> index;
};

struct SyntaxModifier {
  Modifier::Kind kind = Modifier::Kind::Inv;
  std::optional<SyntaxExpressionId> argument;
};

struct SyntaxGateCall {
  SMLoc location;
  StringRef identifier;
  std::vector<SyntaxModifier> modifiers;
  std::vector<SyntaxExpressionId> parameters;
  std::vector<SyntaxOperand> operands;
};

struct SyntaxScalarDeclaration {
  ScalarKind kind = ScalarKind::Int;
  StringRef identifier;
  std::optional<SyntaxExpressionId> initializer;
  bool isConst = false;
};

struct SyntaxAssignment {
  SyntaxBitReference target;
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
  std::optional<SyntaxBitReference> target;
  SyntaxOperand source;
};

struct SyntaxReset {
  SyntaxOperand operand;
};

struct SyntaxBarrier {
  std::vector<SyntaxOperand> operands;
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

struct SyntaxWhile {
  SyntaxExpressionId condition = 0;
  std::vector<SyntaxStatementId> body;
};

struct SyntaxStandardLibraryInclude {};

using SyntaxStatementData =
    std::variant<SyntaxStandardLibraryInclude, SyntaxScalarDeclaration,
                 SyntaxAssignment, SyntaxQubitDeclaration, SyntaxBitDeclaration,
                 SyntaxMeasurement, SyntaxReset, SyntaxBarrier, SyntaxGateCall,
                 SyntaxGateDefinition, SyntaxIf, SyntaxFor, SyntaxWhile>;

struct SyntaxStatement {
  SMLoc location;
  SyntaxStatementData data;
};

struct SyntaxInclude {
  SMLoc location;
  StringRef filename;
  std::size_t bodyOffset = 0;
};

struct SyntaxProgram {
  std::optional<Version> version;
  SMLoc versionLocation;
  std::vector<SyntaxInclude> includes;
  std::vector<SyntaxExpression> expressions;
  std::vector<SyntaxStatement> statements;
  std::vector<SyntaxStatementId> body;
};

struct SyntaxDiagnostic {
  SMLoc location;
  std::string message;
};

class SyntaxBuilder {
public:
  [[nodiscard]] LogicalResult error(SMLoc location, const Twine& message);
  [[nodiscard]] LogicalResult version(SMLoc location, Version value);
  [[nodiscard]] LogicalResult include(SMLoc location, StringRef filename);
  [[nodiscard]] SyntaxStatementId standardLibraryInclude(SMLoc location);
  [[nodiscard]] LogicalResult scalarDecl(SMLoc location, ScalarKind kind,
                                         StringRef identifier,
                                         const Expr* initializer, bool isConst);
  [[nodiscard]] LogicalResult
  assignment(SMLoc location, const BitReference& target, const Expr& value);
  [[nodiscard]] LogicalResult
  qubitRegister(SMLoc location, StringRef identifier, const Expr* size);
  [[nodiscard]] LogicalResult
  classicalRegister(SMLoc location, StringRef identifier, const Expr* size,
                    const Expr* initializer, bool output);
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
  ifStmt(SMLoc location, const Expr& condition,
         function_ref<LogicalResult()> thenContinuation,
         function_ref<LogicalResult()> elseContinuation);
  [[nodiscard]] LogicalResult
  forStmt(SMLoc location, StringRef inductionVariable, bool isUnsigned,
          const Expr& start, const Expr& step, const Expr& stop,
          function_ref<LogicalResult()> continuation);
  [[nodiscard]] LogicalResult
  whileStmt(SMLoc location, const Expr& condition,
            function_ref<LogicalResult()> continuation);

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
  void replaceBody(std::vector<SyntaxStatementId> body);

private:
  [[nodiscard]] SyntaxExpressionId copyExpression(const Expr& expression);
  [[nodiscard]] SyntaxOperand copyOperand(const Operand& operand);
  [[nodiscard]] SyntaxBitReference
  copyBitReference(const BitReference& reference);
  [[nodiscard]] SyntaxGateCall copyGateCall(const GateCall& call);
  [[nodiscard]] SyntaxStatementId addStatement(SMLoc location,
                                               SyntaxStatementData data);
  [[nodiscard]] FailureOr<std::vector<SyntaxStatementId>>
  parseNestedBody(function_ref<LogicalResult()> continuation);

  SyntaxProgram program;
  std::vector<SyntaxDiagnostic> diagnostics;
  SmallVector<std::vector<SyntaxStatementId>*> bodyStack{&program.body};
  bool sawConstruct = false;
};

} // namespace mlir::oq3::frontend::detail
