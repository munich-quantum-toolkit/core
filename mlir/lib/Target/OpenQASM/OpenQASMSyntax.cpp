/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "OpenQASMSyntax.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/Twine.h>

#include <utility>

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
  program.includes.push_back({.location = location,
                              .filename = filename,
                              .bodyOffset = program.body.size()});
  sawConstruct = true;
  return success();
}

SyntaxStatementId SyntaxBuilder::standardLibraryInclude(SMLoc location) {
  const auto id = static_cast<SyntaxStatementId>(program.statements.size());
  program.statements.push_back(
      {.location = location, .data = SyntaxStandardLibraryInclude{}});
  return id;
}

void SyntaxBuilder::replaceBody(std::vector<SyntaxStatementId> body) {
  program.body = std::move(body);
}

SyntaxStatementId SyntaxBuilder::addStatement(SMLoc location,
                                              SyntaxStatementData data) {
  sawConstruct = true;
  const auto id = static_cast<SyntaxStatementId>(program.statements.size());
  program.statements.push_back({.location = location, .data = std::move(data)});
  bodyStack.back()->push_back(id);
  return id;
}

SyntaxExpressionId SyntaxBuilder::copyExpression(const Expr& expression) {
  SyntaxExpression copy{.kind = expression.kind,
                        .location = expression.loc,
                        .integer = expression.intValue,
                        .floatingPoint = expression.floatValue,
                        .boolean = expression.boolValue,
                        .identifier = expression.identifier,
                        .hardwareQubit = expression.hardwareQubit};
  if (expression.lhs != nullptr) {
    copy.lhs = copyExpression(*expression.lhs);
  }
  if (expression.rhs != nullptr) {
    copy.rhs = copyExpression(*expression.rhs);
  }
  const auto id = static_cast<SyntaxExpressionId>(program.expressions.size());
  program.expressions.push_back(copy);
  return id;
}

SyntaxOperand SyntaxBuilder::copyOperand(const Operand& operand) {
  SyntaxOperand copy{.location = operand.loc,
                     .identifier = operand.identifier,
                     .hardwareQubit = operand.hardwareQubit};
  if (operand.index != nullptr) {
    copy.index = copyExpression(*operand.index);
  }
  return copy;
}

SyntaxBitReference
SyntaxBuilder::copyBitReference(const BitReference& reference) {
  SyntaxBitReference copy{.location = reference.loc,
                          .identifier = reference.identifier};
  if (reference.index != nullptr) {
    copy.index = copyExpression(*reference.index);
  }
  return copy;
}

SyntaxGateCall SyntaxBuilder::copyGateCall(const GateCall& call) {
  SyntaxGateCall copy{.location = call.loc, .identifier = call.identifier};
  copy.modifiers.reserve(call.modifiers.size());
  for (const auto& modifier : call.modifiers) {
    SyntaxModifier converted{.kind = modifier.kind};
    if (modifier.argument != nullptr) {
      converted.argument = copyExpression(*modifier.argument);
    }
    copy.modifiers.push_back(converted);
  }
  copy.parameters.reserve(call.parameters.size());
  for (const auto* parameter : call.parameters) {
    copy.parameters.push_back(copyExpression(*parameter));
  }
  copy.operands.reserve(call.operands.size());
  llvm::transform(call.operands, std::back_inserter(copy.operands),
                  [&](const Operand& operand) { return copyOperand(operand); });
  return copy;
}

LogicalResult SyntaxBuilder::scalarDecl(SMLoc location, const ScalarKind kind,
                                        StringRef identifier,
                                        const Expr* initializer,
                                        const bool isConst) {
  SyntaxScalarDeclaration declaration{
      .kind = kind, .identifier = identifier, .isConst = isConst};
  if (initializer != nullptr) {
    declaration.initializer = copyExpression(*initializer);
  }
  (void)addStatement(location, std::move(declaration));
  return success();
}

LogicalResult SyntaxBuilder::assignment(SMLoc location,
                                        const BitReference& target,
                                        const Expr& value) {
  (void)addStatement(location,
                     SyntaxAssignment{.target = copyBitReference(target),
                                      .value = copyExpression(value)});
  return success();
}

LogicalResult SyntaxBuilder::qubitRegister(SMLoc location, StringRef identifier,
                                           const Expr* size) {
  SyntaxQubitDeclaration declaration{.identifier = identifier};
  if (size != nullptr) {
    declaration.size = copyExpression(*size);
  }
  (void)addStatement(location, declaration);
  return success();
}

LogicalResult SyntaxBuilder::classicalRegister(SMLoc location,
                                               StringRef identifier,
                                               const Expr* size,
                                               const Expr* initializer,
                                               const bool output) {
  SyntaxBitDeclaration declaration{.identifier = identifier, .output = output};
  if (size != nullptr) {
    declaration.size = copyExpression(*size);
  }
  if (initializer != nullptr) {
    declaration.initializer = copyExpression(*initializer);
  }
  (void)addStatement(location, declaration);
  return success();
}

LogicalResult SyntaxBuilder::measure(SMLoc location, const BitReference* target,
                                     const Operand& source) {
  SyntaxMeasurement measurement{.source = copyOperand(source)};
  if (target != nullptr) {
    measurement.target = copyBitReference(*target);
  }
  (void)addStatement(location, std::move(measurement));
  return success();
}

LogicalResult SyntaxBuilder::reset(SMLoc location, const Operand& operand) {
  (void)addStatement(location, SyntaxReset{.operand = copyOperand(operand)});
  return success();
}

LogicalResult SyntaxBuilder::barrier(SMLoc location,
                                     ArrayRef<Operand> operands) {
  SyntaxBarrier barrier;
  barrier.operands.reserve(operands.size());
  llvm::transform(operands, std::back_inserter(barrier.operands),
                  [&](const Operand& operand) { return copyOperand(operand); });
  (void)addStatement(location, std::move(barrier));
  return success();
}

LogicalResult SyntaxBuilder::gateCall(const GateCall& call) {
  (void)addStatement(call.loc, copyGateCall(call));
  return success();
}

LogicalResult SyntaxBuilder::gateDefinition(
    SMLoc location, StringRef identifier, ArrayRef<StringRef> parameters,
    ArrayRef<StringRef> qubits, function_ref<LogicalResult()> continuation) {
  SyntaxGateDefinition definition{.identifier = identifier,
                                  .parameters = parameters.vec(),
                                  .qubits = qubits.vec()};
  auto body = parseNestedBody(continuation);
  if (failed(body)) {
    return failure();
  }
  definition.body = std::move(*body);
  (void)addStatement(location, std::move(definition));
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

LogicalResult
SyntaxBuilder::ifStmt(SMLoc location, const Expr& condition,
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
  (void)addStatement(location,
                     SyntaxIf{.condition = copyExpression(condition),
                              .thenStatements = std::move(*thenStatements),
                              .elseStatements = std::move(*elseStatements)});
  return success();
}

LogicalResult
SyntaxBuilder::forStmt(SMLoc location, StringRef inductionVariable,
                       const bool isUnsigned, const Expr& start,
                       const Expr& step, const Expr& stop,
                       function_ref<LogicalResult()> continuation) {
  auto body = parseNestedBody(continuation);
  if (failed(body)) {
    return failure();
  }
  (void)addStatement(location, SyntaxFor{.inductionVariable = inductionVariable,
                                         .isUnsigned = isUnsigned,
                                         .start = copyExpression(start),
                                         .step = copyExpression(step),
                                         .stop = copyExpression(stop),
                                         .body = std::move(*body)});
  return success();
}

LogicalResult
SyntaxBuilder::whileStmt(SMLoc location, const Expr& condition,
                         function_ref<LogicalResult()> continuation) {
  auto body = parseNestedBody(continuation);
  if (failed(body)) {
    return failure();
  }
  (void)addStatement(location,
                     SyntaxWhile{.condition = copyExpression(condition),
                                 .body = std::move(*body)});
  return success();
}

} // namespace mlir::oq3::frontend::detail
