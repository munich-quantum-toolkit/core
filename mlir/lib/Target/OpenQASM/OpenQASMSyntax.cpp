/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Target/OpenQASM/Detail/OpenQASMSyntax.h"

#include "mlir/Target/OpenQASM/Detail/OpenQASMParser.h"

#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace mlir::oq3::frontend::detail {

LogicalResult SyntaxBuilder::error(SMLoc location, const Twine& message) {
  diagnostics.push_back({.location = location, .message = message.str()});
  return failure();
}

LogicalResult SyntaxBuilder::version(SMLoc location, const Version value) {
  if (program.version || sawConstruct) {
    return error(location,
                 "version declaration must be the first source construct");
  }
  program.version = value;
  program.versionLocation = location;
  sawConstruct = true;
  return success();
}

LogicalResult SyntaxBuilder::include(SMLoc location, StringRef filename) {
  program.includes.push_back({
      .location = location,
      .filename = filename,
      .bodyOffset = program.body.size(),
  });
  sawConstruct = true;
  return success();
}

SyntaxStatementId
SyntaxBuilder::standardLibraryInclude(SMLoc location,
                                      const StandardLibraryKind kind) {
  const auto id = static_cast<SyntaxStatementId>(program.statements.size());
  program.statements.push_back(
      {.location = location, .data = SyntaxStandardLibraryInclude{kind}});
  return id;
}

void SyntaxBuilder::replaceBody(
    std::vector<SyntaxStatementId> body,
    std::vector<std::optional<SyntaxIncludeContextId>> includeContexts,
    std::vector<SyntaxIncludeContext> contexts) {
  program.body = std::move(body);
  program.bodyIncludeContexts = std::move(includeContexts);
  program.includeContexts = std::move(contexts);
}

SyntaxStatementId SyntaxBuilder::addStatement(SMLoc location,
                                              SyntaxStatementData data) {
  sawConstruct = true;
  const auto id = static_cast<SyntaxStatementId>(program.statements.size());
  program.statements.push_back({.location = location, .data = std::move(data)});
  bodyStack.back()->push_back(id);
  return id;
}

SyntaxExpressionId
SyntaxBuilder::addExpression(const SyntaxExpression& expression) {
  const auto id = static_cast<SyntaxExpressionId>(program.expressions.size());
  program.expressions.push_back(expression);
  return id;
}

SyntaxGateCall SyntaxBuilder::copyGateCall(const GateCall& call) {
  return {
      .location = call.loc,
      .identifier = call.identifier,
      .modifiers = call.modifiers.vec(),
      .parameters = call.parameters.vec(),
      .operands = call.operands.vec(),
  };
}

LogicalResult
SyntaxBuilder::scalarDecl(SMLoc location, const ScalarKind kind,
                          StringRef identifier,
                          std::optional<SyntaxExpressionId> size,
                          std::optional<SyntaxExpressionId> initializer,
                          const bool isConst, const bool output) {
  SyntaxScalarDeclaration declaration{
      .kind = kind,
      .identifier = identifier,
      .size = size,
      .initializer = initializer,
      .isConst = isConst,
      .output = output,
  };
  std::ignore = addStatement(location, declaration);
  return success();
}

LogicalResult SyntaxBuilder::assignment(SMLoc location,
                                        const BitReference& target,
                                        SyntaxExpressionId value) {
  std::ignore = addStatement(location, SyntaxAssignment{
                                           .target = target,
                                           .value = value,
                                       });
  return success();
}

LogicalResult
SyntaxBuilder::qubitRegister(SMLoc location, StringRef identifier,
                             std::optional<SyntaxExpressionId> size) {
  SyntaxQubitDeclaration declaration{
      .identifier = identifier,
      .size = size,
  };
  std::ignore = addStatement(location, declaration);
  return success();
}

LogicalResult
SyntaxBuilder::classicalRegister(SMLoc location, StringRef identifier,
                                 std::optional<SyntaxExpressionId> size,
                                 std::optional<SyntaxExpressionId> initializer,
                                 const bool output) {
  SyntaxBitDeclaration declaration{
      .identifier = identifier,
      .size = size,
      .initializer = initializer,
      .output = output,
  };
  std::ignore = addStatement(location, declaration);
  return success();
}

LogicalResult SyntaxBuilder::measure(SMLoc location, const BitReference* target,
                                     const Operand& source) {
  SyntaxMeasurement measurement{
      .target = std::nullopt,
      .source = source,
  };
  if (target != nullptr) {
    measurement.target = *target;
  }
  std::ignore = addStatement(location, measurement);
  return success();
}

LogicalResult SyntaxBuilder::reset(SMLoc location, const Operand& operand) {
  std::ignore = addStatement(location, SyntaxReset{.operand = operand});
  return success();
}

LogicalResult SyntaxBuilder::barrier(SMLoc location,
                                     ArrayRef<Operand> operands) {
  std::ignore =
      addStatement(location, SyntaxBarrier{.operands = operands.vec()});
  return success();
}

LogicalResult SyntaxBuilder::gateCall(const GateCall& call) {
  std::ignore = addStatement(call.loc, copyGateCall(call));
  return success();
}

LogicalResult SyntaxBuilder::gateDefinition(
    SMLoc location, StringRef identifier, ArrayRef<StringRef> parameters,
    ArrayRef<StringRef> qubits, function_ref<LogicalResult()> continuation) {
  SyntaxGateDefinition definition{
      .identifier = identifier,
      .parameters = parameters.vec(),
      .qubits = qubits.vec(),
      .body = {},
  };
  auto body = parseNestedBody(continuation);
  if (failed(body)) {
    return failure();
  }
  definition.body = std::move(*body);
  std::ignore = addStatement(location, std::move(definition));
  return success();
}

FailureOr<std::vector<SyntaxStatementId>>
SyntaxBuilder::parseNestedBody(function_ref<LogicalResult()> continuation) {
  std::vector<SyntaxStatementId> body;
  bodyStack.push_back(&body);
  const auto result = continuation();
  bodyStack.pop_back();
  if (failed(result)) {
    return failure();
  }
  return body;
}

LogicalResult SyntaxBuilder::continueStmt(SMLoc location) {
  std::ignore = addStatement(location, SyntaxContinue{});
  return success();
}

LogicalResult SyntaxBuilder::breakStmt(SMLoc location) {
  std::ignore = addStatement(location, SyntaxBreak{});
  return success();
}

LogicalResult
SyntaxBuilder::ifStmt(SMLoc location, SyntaxExpressionId condition,
                      function_ref<LogicalResult()> thenContinuation,
                      function_ref<LogicalResult()> elseContinuation) {
  auto thenStatements = parseNestedBody(thenContinuation);
  if (failed(thenStatements)) {
    return failure();
  }
  auto elseStatements = parseNestedBody(elseContinuation);
  if (failed(elseStatements)) {
    return failure();
  }
  std::ignore =
      addStatement(location, SyntaxIf{
                                 .condition = condition,
                                 .thenStatements = std::move(*thenStatements),
                                 .elseStatements = std::move(*elseStatements),
                             });
  return success();
}

LogicalResult
SyntaxBuilder::forStmt(SMLoc location, StringRef inductionVariable,
                       const bool isUnsigned, SyntaxExpressionId start,
                       SyntaxExpressionId step, SyntaxExpressionId stop,
                       function_ref<LogicalResult()> continuation) {
  auto body = parseNestedBody(continuation);
  if (failed(body)) {
    return failure();
  }
  std::ignore =
      addStatement(location, SyntaxFor{
                                 .inductionVariable = inductionVariable,
                                 .isUnsigned = isUnsigned,
                                 .start = start,
                                 .step = step,
                                 .stop = stop,
                                 .body = std::move(*body),
                             });
  return success();
}

LogicalResult
SyntaxBuilder::whileStmt(SMLoc location, SyntaxExpressionId condition,
                         function_ref<LogicalResult()> continuation) {
  auto body = parseNestedBody(continuation);
  if (failed(body)) {
    return failure();
  }
  std::ignore = addStatement(location, SyntaxWhile{
                                           .condition = condition,
                                           .body = std::move(*body),
                                       });
  return success();
}

LogicalResult
SyntaxBuilder::switchStmt(SMLoc location, SyntaxExpressionId control,
                          function_ref<LogicalResult()> continuation) {
  SyntaxSwitch statement{
      .control = control,
      .cases = {},
      .defaultStatements = {},
  };
  switchStack.push_back(&statement);
  const auto result = continuation();
  switchStack.pop_back();
  if (failed(result)) {
    return failure();
  }
  std::ignore = addStatement(location, std::move(statement));
  return success();
}

LogicalResult
SyntaxBuilder::switchCase(SMLoc /*location*/,
                          const ArrayRef<SyntaxExpressionId> labels,
                          function_ref<LogicalResult()> continuation) {
  SyntaxSwitchCase switchCase;
  switchCase.labels = labels.vec();
  auto body = parseNestedBody(continuation);
  if (failed(body)) {
    return failure();
  }
  switchCase.body = std::move(*body);
  switchStack.back()->cases.push_back(std::move(switchCase));
  return success();
}

LogicalResult
SyntaxBuilder::switchDefault(SMLoc /*location*/,
                             function_ref<LogicalResult()> continuation) {
  auto body = parseNestedBody(continuation);
  if (failed(body)) {
    return failure();
  }
  switchStack.back()->defaultStatements = std::move(*body);
  return success();
}

} // namespace mlir::oq3::frontend::detail
