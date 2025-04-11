/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qasm3/passes/TypeCheckPass.hpp"

#include "qasm3/Exception.hpp"
#include "qasm3/Statement.hpp"
#include "qasm3/Types.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

namespace qasm3::type_checking {

InferredType TypeCheckPass::error(const std::string& msg,
                                  const std::shared_ptr<DebugInfo>& debugInfo) {
  std::cerr << "Type check error: " << msg << '\n';
  if (debugInfo) {
    std::cerr << "  " << debugInfo->toString() << '\n';
  }
  hasError = true;
  errMessage = msg;
  return InferredType::error();
}

void TypeCheckPass::processStatement(Statement& statement) {
  try {
    statement.accept(this);

    if (hasError) {
      throw TypeCheckError(errMessage);
    }
  } catch (const TypeCheckError& e) {
    throw CompilerError(e.what(), statement.debugInfo);
  }
}

void TypeCheckPass::checkIndexOperator(const IndexOperator& indexOperator) {
  for (const auto& index : indexOperator.indexExpressions) {
    if (const auto type = visit(index); !type.isError && !type.type->isUint()) {
      error("Index must be an unsigned integer");
    }
  }
}

void TypeCheckPass::checkIndexedIdentifier(const IndexedIdentifier& id) {

  if (const auto it = env.find(id.identifier); it == env.end()) {
    error("Unknown identifier '" + id.identifier + "'.");
  }
  for (const auto& index : id.indices) {
    checkIndexOperator(*index);
  }
}

void TypeCheckPass::checkGateOperand(const GateOperand& operand) {
  if (operand.isHardwareQubit()) {
    return;
  }
  checkIndexedIdentifier(*operand.getIdentifier());
}

void TypeCheckPass::visitGateStatement(
    const std::shared_ptr<GateDeclaration> gateStatement) {
  // we save the current environment to restore it afterward
  const auto oldEnv = env;

  for (const auto& param : gateStatement->parameters->identifiers) {
    env.emplace(param->identifier,
                InferredType{std::dynamic_pointer_cast<ResolvedType>(
                    DesignatedType<uint64_t>::getFloatTy(64))});
  }
  for (const auto& operand : gateStatement->qubits->identifiers) {
    env.emplace(operand->identifier,
                InferredType{std::dynamic_pointer_cast<ResolvedType>(
                    DesignatedType<uint64_t>::getQubitTy(1))});
  }

  for (const auto& stmt : gateStatement->statements) {
    stmt->accept(this);
  }

  // restore the environment
  env = oldEnv;
}

void TypeCheckPass::visitVersionDeclaration(
    std::shared_ptr<VersionDeclaration> /*versionDeclaration*/) {}

void TypeCheckPass::visitDeclarationStatement(
    const std::shared_ptr<DeclarationStatement> declarationStatement) {
  // Type checking declarations is a bit involved. If the type contains a
  // designator expression, we need to resolve the statement in three steps.
  const auto typeExpr = std::get<0>(declarationStatement->type);
  // First, type-check the type itself.
  if (typeExpr->allowsDesignator() && typeExpr->getDesignator() != nullptr) {
    auto type = visit(typeExpr->getDesignator());
    if (type.isError || !type.type->isUint()) {
      error("Designator expression type check failed.",
            declarationStatement->debugInfo);
      return;
    }
  }
  // Now we know the type is valid, we can evaluate the designator expression.
  auto resolvedType =
      std::get<0>(declarationStatement->type)->accept(constEvalPass);
  if (!resolvedType) {
    throw TypeCheckError("Expression in types must be const.");
  }
  declarationStatement->type = resolvedType;

  // Lastly, we type check the actual expression
  if (declarationStatement->expression != nullptr) {
    const auto exprType = visit(declarationStatement->expression->expression);
    if (!exprType.isError && !resolvedType->fits(*exprType.type)) {
      std::stringstream ss;
      ss << "Type mismatch in declaration statement: Expected '";
      ss << resolvedType->toString();
      ss << "', found '";
      ss << exprType.type->toString();
      ss << "'.";
      error(ss.str());
    }
  }

  env.emplace(declarationStatement->identifier, InferredType{resolvedType});
}

void TypeCheckPass::visitInitialLayout(
    std::shared_ptr<InitialLayout> /*initialLayout*/) {}

void TypeCheckPass::visitOutputPermutation(
    std::shared_ptr<OutputPermutation> /*outputPermutation*/) {}

void TypeCheckPass::visitGateCallStatement(
    const std::shared_ptr<GateCallStatement> gateCallStatement) {
  for (const auto& arg : gateCallStatement->arguments) {
    visit(arg);
  }
}

void TypeCheckPass::visitAssignmentStatement(
    const std::shared_ptr<AssignmentStatement> assignmentStatement) {
  checkIndexedIdentifier(*assignmentStatement->identifier);

  const auto exprTy = visit(assignmentStatement->expression->expression);
  const auto idTy = env.find(assignmentStatement->identifier->identifier);

  if (idTy == env.end()) {
    error("Unknown identifier '" + assignmentStatement->identifier->identifier +
              "'.",
          assignmentStatement->debugInfo);
    return;
  }

  if (!idTy->second.type->fits(*exprTy.type)) {
    std::stringstream ss;
    ss << "Type mismatch in assignment. Expected '";
    ss << idTy->second.type->toString();
    ss << "', found '";
    ss << exprTy.type->toString();
    ss << "'.";
    error(ss.str(), assignmentStatement->debugInfo);
  }
}

void TypeCheckPass::visitBarrierStatement(
    const std::shared_ptr<BarrierStatement> barrierStatement) {
  for (auto& gate : barrierStatement->gates) {
    checkGateOperand(*gate);
  }
}

void TypeCheckPass::visitResetStatement(
    const std::shared_ptr<ResetStatement> resetStatement) {
  checkGateOperand(*resetStatement->gate);
}

InferredType TypeCheckPass::visitBinaryExpression(
    const std::shared_ptr<BinaryExpression> binaryExpression) {
  auto lhs = visit(binaryExpression->lhs);
  auto rhs = visit(binaryExpression->rhs);
  if (rhs.isError) {
    return rhs;
  }
  if (lhs.isError) {
    return lhs;
  }

  auto ty = lhs;
  if (lhs.type->isConvertibleToBool() && rhs.type->isConvertibleToBool()) {
    ty = InferredType{UnsizedType<uint64_t>::getBoolTy()};
  } else if (lhs.type->isNumber() && rhs.type->isNumber()) {
    if (rhs.type->isFP() || lhs.type->isUint()) {
      // coerce to float or signed int
      ty = rhs;
    }
    ty.type->setDesignator(
        std::max(lhs.type->getDesignator(), rhs.type->getDesignator()));
  } else if (lhs.type != rhs.type) {
    std::stringstream ss;
    ss << "Type mismatch in binary expression: ";
    ss << lhs.type->toString();
    ss << ", ";
    ss << rhs.type->toString();
    ss << ".";
    return error(ss.str());
  }

  switch (binaryExpression->op) {
  case BinaryExpression::Power:
  case BinaryExpression::Add:
  case BinaryExpression::Subtract:
  case BinaryExpression::Multiply:
  case BinaryExpression::Divide:
  case BinaryExpression::Modulo:
  case BinaryExpression::LeftShift:
  case BinaryExpression::RightShift:
    if (!ty.type->isNumber()) {
      return error("Cannot apply arithmetic operation to non-numeric type.");
    }
    break;
  case BinaryExpression::LessThan:
  case BinaryExpression::LessThanOrEqual:
  case BinaryExpression::GreaterThan:
  case BinaryExpression::GreaterThanOrEqual:
    // all types except for bool
    if (ty.type->isBool()) {
      return error("Cannot compare boolean types.");
    }
    return InferredType{UnsizedType<uint64_t>::getBoolTy()};
  case BinaryExpression::Equal:
  case BinaryExpression::NotEqual:
    return InferredType{UnsizedType<uint64_t>::getBoolTy()};
  case BinaryExpression::BitwiseAnd:
  case BinaryExpression::BitwiseXor:
  case BinaryExpression::BitwiseOr:
    if (!ty.type->isNumber()) {
      return error("Cannot apply bitwise operation to non-numeric type.");
    }
    break;
  case BinaryExpression::LogicalAnd:
  case BinaryExpression::LogicalOr:
    if (!ty.type->isBool()) {
      return error("Cannot apply logical operation to non-boolean type.");
    }
    break;
  }

  return ty;
}

InferredType TypeCheckPass::visitUnaryExpression(
    const std::shared_ptr<UnaryExpression> unaryExpression) {
  auto type = visit(unaryExpression->operand);

  switch (unaryExpression->op) {
  case UnaryExpression::BitwiseNot:
    if (!type.type->isNumber()) {
      return error("Cannot apply bitwise not to non-numeric type.");
    }
    break;
  case UnaryExpression::LogicalNot:
    if (!type.type->isConvertibleToBool()) {
      return error("Cannot apply logical not to non-boolean type.");
    }
    break;
  case UnaryExpression::Negate:
    break;
  case UnaryExpression::DurationOf:
    return InferredType{UnsizedType<uint64_t>::getDurationTy()};
  case UnaryExpression::Sin:
  case UnaryExpression::Cos:
  case UnaryExpression::Tan:
  case UnaryExpression::Exp:
  case UnaryExpression::Ln:
  case UnaryExpression::Sqrt:
    return InferredType{std::dynamic_pointer_cast<ResolvedType>(
        DesignatedType<uint64_t>::getFloatTy(64))};
  }

  return type;
}

InferredType TypeCheckPass::visitConstantExpression(
    const std::shared_ptr<Constant> constant) {
  if (constant->isFP()) {
    return InferredType{std::dynamic_pointer_cast<ResolvedType>(
        DesignatedType<uint64_t>::getFloatTy(64))};
  }
  if (constant->isBool()) {
    return InferredType(std::dynamic_pointer_cast<ResolvedType>(
        UnsizedType<uint64_t>::getBoolTy()));
  }
  assert(constant->isInt());

  size_t width = 32;
  if ((constant->isSInt() && constant->getSInt() > INT32_MAX) ||
      (constant->isUInt() && constant->getUInt() > UINT32_MAX)) {
    width = 64;
  }

  if (constant->isSInt()) {
    return InferredType{std::dynamic_pointer_cast<ResolvedType>(
        DesignatedType<uint64_t>::getIntTy(width))};
  }

  return InferredType{std::dynamic_pointer_cast<ResolvedType>(
      DesignatedType<uint64_t>::getUintTy(width))};
}

InferredType TypeCheckPass::visitIdentifierExpression(
    const std::shared_ptr<IdentifierExpression> identifierExpression) {
  const auto type = env.find(identifierExpression->identifier);
  if (type == env.end()) {
    error("Unknown identifier '" + identifierExpression->identifier + "'.");
    return InferredType::error();
  }
  return type->second;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=noreturn"
InferredType TypeCheckPass::visitIdentifierList(
    std::shared_ptr<IdentifierList> /*identifierList*/) {
  throw TypeCheckError("TypeCheckPass::visitIdentifierList not implemented");
}
#pragma GCC diagnostic pop

InferredType TypeCheckPass::visitIndexedIdentifier(
    const std::shared_ptr<IndexedIdentifier> indexedIdentifier) {
  auto type = visitIdentifierExpression(
      std::make_shared<IdentifierExpression>(indexedIdentifier->identifier));
  if (indexedIdentifier->indices.empty()) {
    return type;
  }
  // Assume that indexed access always results in a single element
  type.type->setDesignator(1);
  return type;
}

InferredType TypeCheckPass::visitMeasureExpression(
    const std::shared_ptr<MeasureExpression> measureExpression) {
  if (measureExpression->gate->isHardwareQubit()) {
    return InferredType{std::dynamic_pointer_cast<ResolvedType>(
        DesignatedType<uint64_t>::getBitTy(1))};
  }

  const auto indexedIdentifier = measureExpression->gate->getIdentifier();
  checkIndexedIdentifier(*indexedIdentifier);
  if (!indexedIdentifier->indices.empty()) {
    // This will need modification once we want to support index ranges.
    return InferredType{std::dynamic_pointer_cast<ResolvedType>(
        DesignatedType<uint64_t>::getBitTy(1))};
  }
  const auto it = env.find(indexedIdentifier->identifier);
  if (it == env.end()) {
    error("Unknown identifier '" + indexedIdentifier->identifier + "'.");
    return InferredType::error();
  }
  const auto width = it->second.type->getDesignator();
  return InferredType{std::dynamic_pointer_cast<ResolvedType>(
      DesignatedType<uint64_t>::getBitTy(width))};
}

void TypeCheckPass::visitIfStatement(
    const std::shared_ptr<IfStatement> ifStatement) {
  // We support ifs on bits and bools
  if (const auto ty = visit(ifStatement->condition);
      !ty.isError && !ty.type->isConvertibleToBool()) {
    error("Condition expression must be bool.");
  }

  for (const auto& stmt : ifStatement->thenStatements) {
    stmt->accept(this);
  }
  for (const auto& stmt : ifStatement->elseStatements) {
    stmt->accept(this);
  }
}
} // namespace qasm3::type_checking
