/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "OpenQASMToQCEmitter.h"

#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Target/OpenQASM/GateCatalog.h"

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OperationSupport.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace mlir::qc::detail {
namespace {

namespace frontend = oq3::frontend;
using oq3::frontend::GateCatalogEntry;

class OpenQASMToQCEmitter {
public:
  OpenQASMToQCEmitter(const oq3::frontend::TypedProgram& typedProgram,
                      MLIRContext& mlirContext)
      : program(typedProgram), context(mlirContext), builder(&context),
        registerValues(program.registers.size()),
        classicalRegisters(program.registers.size()),
        bitValues(program.registers.size()),
        scalarValues(program.scalars.size()) {
    context
        .loadDialect<qc::QCDialect, arith::ArithDialect, cf::ControlFlowDialect,
                     func::FuncDialect, math::MathDialect,
                     memref::MemRefDialect, scf::SCFDialect, ub::UBDialect>();
    builder.initialize();
  }

  OwningOpRef<ModuleOp> emit() {
    if (!preflight()) {
      return nullptr;
    }
    for (const auto statement : program.body) {
      emitStatement(statement, {}, {});
    }
    if (emissionFailed) {
      return nullptr;
    }

    SmallVector<Value> results;
    for (const auto output : program.outputs) {
      for (auto bit : bitValues[output]) {
        if (!bit) {
          llvm::errs() << "OpenQASM emission error: output register '"
                       << program.registers[output].name
                       << "' is not fully initialized.\n";
          return nullptr;
        }
        results.push_back(bit);
      }
    }
    if (results.empty()) {
      return builder.finalize();
    }
    builder.retype(ValueRange(results).getTypes());
    return builder.finalize(results);
  }

private:
  const oq3::frontend::TypedProgram& program;
  MLIRContext& context;
  qc::QCProgramBuilder builder;
  std::vector<SmallVector<Value>> registerValues;
  std::vector<std::optional<qc::QCProgramBuilder::ClassicalRegister>>
      classicalRegisters;
  std::vector<SmallVector<Value>> bitValues;
  std::vector<Value> scalarValues;
  DenseMap<const oq3::frontend::GateDefinition*, bool>
      structuredGateCapabilities;
  bool emissionFailed = false;

  enum class StateKind : std::uint8_t { Scalar, Bit };

  struct StateSlot {
    StateKind kind = StateKind::Scalar;
    std::uint32_t first = 0;
    std::uint32_t second = 0;
  };

  [[nodiscard]] Location
  getLocation(const frontend::SourceLocation& source) const {
    return FileLineColLoc::get(&context, source.filename, source.line,
                               source.column);
  }

  static constexpr std::size_t projectedEmissionLimit = 100000;

  [[nodiscard]] const oq3::frontend::GateDefinition*
  findCustomGate(const StringRef name) const {
    const auto found = llvm::find_if(
        program.gates, [&](const auto& gate) { return gate.name == name; });
    return found == program.gates.end() ? nullptr : &*found;
  }

  [[nodiscard]] bool statementsRequireStructuredControlFlow(
      const ArrayRef<oq3::frontend::StatementId> statements) {
    return llvm::any_of(statements, [&](const auto id) {
      const auto& data = program.statements.at(id).data;
      if (std::holds_alternative<oq3::frontend::ForStatement>(data) ||
          std::holds_alternative<oq3::frontend::WhileStatement>(data) ||
          std::holds_alternative<oq3::frontend::IfStatement>(data)) {
        return true;
      }
      const auto* application =
          std::get_if<oq3::frontend::GateApplication>(&data);
      const auto* callee = application == nullptr
                               ? nullptr
                               : findCustomGate(application->callee);
      return callee != nullptr && gateRequiresStructuredControlFlow(*callee);
    });
  }

  [[nodiscard]] bool
  gateRequiresStructuredControlFlow(const oq3::frontend::GateDefinition& gate) {
    if (const auto it = structuredGateCapabilities.find(&gate);
        it != structuredGateCapabilities.end()) {
      return it->second;
    }
    const bool requiresStructuredControlFlow =
        statementsRequireStructuredControlFlow(gate.body);
    structuredGateCapabilities[&gate] = requiresStructuredControlFlow;
    return requiresStructuredControlFlow;
  }

  [[nodiscard]] std::optional<bool>
  staticCondition(const frontend::ConditionId id) const {
    const auto& condition = program.conditions.at(id);
    if (condition.kind == frontend::ConditionKind::Literal) {
      return condition.literal;
    }
    return std::nullopt;
  }

  [[nodiscard]] bool reportProjectedEmissionLimit(
      const oq3::frontend::SourceLocation& source) const {
    llvm::errs() << source.filename << ':' << source.line << ':'
                 << source.column
                 << ": OpenQASM QC emission error: projected emitted "
                    "operation count exceeds the safe lowering limit.\n";
    return false;
  }

  [[nodiscard]] bool
  projectedMultiplicity(const ArrayRef<frontend::QubitReference> references,
                        const std::size_t parentMultiplicity,
                        const oq3::frontend::SourceLocation& source,
                        std::size_t& result) const {
    result = parentMultiplicity;
    for (const auto& reference : references) {
      if (!reference.dynamicIndex) {
        continue;
      }
      const auto width = static_cast<std::size_t>(
          program.registers.at(reference.symbol).width);
      if (width != 0 && result > projectedEmissionLimit / width) {
        return reportProjectedEmissionLimit(source);
      }
      result *= width;
    }
    return true;
  }

  [[nodiscard]] bool
  chargeProjectedEmission(const std::size_t amount,
                          std::size_t& projectedEmission,
                          const oq3::frontend::SourceLocation& source) const {
    if (amount > projectedEmissionLimit - projectedEmission) {
      return reportProjectedEmissionLimit(source);
    }
    projectedEmission += amount;
    return true;
  }

  [[nodiscard]] bool
  chargeConditionEmission(const frontend::ConditionId id,
                          const std::size_t multiplicity,
                          std::size_t& projectedEmission,
                          const oq3::frontend::SourceLocation& source) const {
    const auto& condition = program.conditions.at(id);
    if (condition.kind == frontend::ConditionKind::Measurement) {
      std::size_t operationMultiplicity = 0;
      return projectedMultiplicity({condition.measurement}, multiplicity,
                                   source, operationMultiplicity) &&
             chargeProjectedEmission(operationMultiplicity, projectedEmission,
                                     source);
    }
    if (condition.kind == frontend::ConditionKind::Not) {
      return chargeConditionEmission(condition.lhs, multiplicity,
                                     projectedEmission, source);
    }
    if (condition.kind == frontend::ConditionKind::And ||
        condition.kind == frontend::ConditionKind::Or) {
      return chargeConditionEmission(condition.lhs, multiplicity,
                                     projectedEmission, source) &&
             chargeConditionEmission(condition.rhs, multiplicity,
                                     projectedEmission, source);
    }
    return true;
  }

  [[nodiscard]] bool
  preflightStatements(const ArrayRef<oq3::frontend::StatementId> statements,
                      std::size_t& projectedEmission,
                      const std::size_t multiplicity = 1) {
    for (const auto id : statements) {
      const auto& statement = program.statements.at(id);
      const auto* application =
          std::get_if<oq3::frontend::GateApplication>(&statement.data);
      if (application == nullptr) {
        if (const auto* conditional =
                std::get_if<oq3::frontend::IfStatement>(&statement.data)) {
          if (!chargeConditionEmission(conditional->condition, multiplicity,
                                       projectedEmission, statement.location)) {
            return false;
          }
          if (const auto selected = staticCondition(conditional->condition)) {
            const auto& selectedStatements = *selected
                                                 ? conditional->thenStatements
                                                 : conditional->elseStatements;
            if (!preflightStatements(selectedStatements, projectedEmission,
                                     multiplicity)) {
              return false;
            }
            continue;
          }
          if (!preflightStatements(conditional->thenStatements,
                                   projectedEmission, multiplicity) ||
              !preflightStatements(conditional->elseStatements,
                                   projectedEmission, multiplicity)) {
            return false;
          }
        } else if (const auto* loop = std::get_if<oq3::frontend::ForStatement>(
                       &statement.data)) {
          if (!preflightStatements(loop->body, projectedEmission,
                                   multiplicity)) {
            return false;
          }
        } else if (const auto* loop =
                       std::get_if<oq3::frontend::WhileStatement>(
                           &statement.data)) {
          if (!chargeConditionEmission(loop->condition, multiplicity,
                                       projectedEmission, statement.location)) {
            return false;
          }
          if (!preflightStatements(loop->body, projectedEmission,
                                   multiplicity)) {
            return false;
          }
        } else if (const auto* declaration =
                       std::get_if<frontend::ScalarDeclarationStatement>(
                           &statement.data)) {
          if (declaration->conditionInitializer &&
              !chargeConditionEmission(*declaration->conditionInitializer,
                                       multiplicity, projectedEmission,
                                       statement.location)) {
            return false;
          }
        } else if (const auto* assignment =
                       std::get_if<frontend::ScalarAssignmentStatement>(
                           &statement.data)) {
          if (assignment->condition &&
              !chargeConditionEmission(*assignment->condition, multiplicity,
                                       projectedEmission, statement.location)) {
            return false;
          }
        } else if (const auto* assignment =
                       std::get_if<frontend::BitAssignmentStatement>(
                           &statement.data)) {
          if (!chargeConditionEmission(assignment->value, multiplicity,
                                       projectedEmission, statement.location)) {
            return false;
          }
        } else if (const auto* measurement =
                       std::get_if<frontend::MeasurementStatement>(
                           &statement.data)) {
          for (const auto& qubit : measurement->qubits) {
            std::size_t operationMultiplicity = 0;
            if (!projectedMultiplicity({qubit}, multiplicity,
                                       statement.location,
                                       operationMultiplicity) ||
                !chargeProjectedEmission(operationMultiplicity,
                                         projectedEmission,
                                         statement.location)) {
              return false;
            }
          }
        } else if (const auto* reset =
                       std::get_if<frontend::ResetStatement>(&statement.data)) {
          for (const auto& qubit : reset->qubits) {
            std::size_t operationMultiplicity = 0;
            if (!projectedMultiplicity({qubit}, multiplicity,
                                       statement.location,
                                       operationMultiplicity) ||
                !chargeProjectedEmission(operationMultiplicity,
                                         projectedEmission,
                                         statement.location)) {
              return false;
            }
          }
        } else if (const auto* barrier =
                       std::get_if<frontend::BarrierStatement>(
                           &statement.data)) {
          std::size_t operationMultiplicity = 0;
          if (!projectedMultiplicity(barrier->qubits, multiplicity,
                                     statement.location,
                                     operationMultiplicity) ||
              !chargeProjectedEmission(operationMultiplicity, projectedEmission,
                                       statement.location)) {
            return false;
          }
        }
        continue;
      }
      for (const auto& modifier : application->modifiers) {
        if (modifier.kind == oq3::frontend::ModifierKind::Pow) {
          const auto& source = statement.location;
          llvm::errs() << source.filename << ':' << source.line << ':'
                       << source.column
                       << ": OpenQASM QC emission error: power gate modifiers "
                          "are not supported by the QC dialect.\n";
          return false;
        }
      }
      std::size_t operationMultiplicity = 0;
      if (!projectedMultiplicity(application->qubits, multiplicity,
                                 statement.location, operationMultiplicity)) {
        return false;
      }
      const auto* gate = findCustomGate(application->callee);
      if (gate == nullptr) {
        if (!chargeProjectedEmission(operationMultiplicity, projectedEmission,
                                     statement.location)) {
          return false;
        }
        continue;
      }
      if (!application->modifiers.empty() &&
          gateRequiresStructuredControlFlow(*gate)) {
        const auto& source = statement.location;
        llvm::errs() << source.filename << ':' << source.line << ':'
                     << source.column
                     << ": OpenQASM QC emission error: modifiers on custom "
                        "gates with structured control flow are not supported "
                        "by the QC dialect.\n";
        return false;
      }
      if (!preflightStatements(gate->body, projectedEmission,
                               operationMultiplicity)) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool preflight() {
    std::size_t projectedEmission = 0;
    return preflightStatements(program.body, projectedEmission);
  }

  [[nodiscard]] Value checkedSignedResult(OpBuilder& opBuilder, Location loc,
                                          Value wide, const StringRef message) {
    auto i128 = opBuilder.getIntegerType(128);
    auto minimum = arith::ConstantIntOp::create(
        opBuilder, loc, i128, std::numeric_limits<std::int64_t>::min());
    auto maximum = arith::ConstantIntOp::create(
        opBuilder, loc, i128, std::numeric_limits<std::int64_t>::max());
    auto aboveMinimum = arith::CmpIOp::create(
        opBuilder, loc, arith::CmpIPredicate::sge, wide, minimum);
    auto belowMaximum = arith::CmpIOp::create(
        opBuilder, loc, arith::CmpIPredicate::sle, wide, maximum);
    auto fits =
        arith::AndIOp::create(opBuilder, loc, aboveMinimum, belowMaximum);
    cf::AssertOp::create(opBuilder, loc, fits, message);
    return arith::TruncIOp::create(opBuilder, loc, opBuilder.getI64Type(),
                                   wide);
  }

  [[nodiscard]] Value conditionalIntegerMultiply(OpBuilder& opBuilder,
                                                 Location loc, Value condition,
                                                 Value lhs, Value rhs,
                                                 const bool isUnsigned) {
    if (isUnsigned) {
      auto product = arith::MulIOp::create(opBuilder, loc, lhs, rhs);
      return arith::SelectOp::create(opBuilder, loc, condition, product, lhs);
    }
    auto i128 = opBuilder.getIntegerType(128);
    auto lhsWide = arith::ExtSIOp::create(opBuilder, loc, i128, lhs);
    auto rhsWide = arith::ExtSIOp::create(opBuilder, loc, i128, rhs);
    auto productWide = arith::MulIOp::create(opBuilder, loc, lhsWide, rhsWide);
    auto minimum = arith::ConstantIntOp::create(
        opBuilder, loc, i128, std::numeric_limits<std::int64_t>::min());
    auto maximum = arith::ConstantIntOp::create(
        opBuilder, loc, i128, std::numeric_limits<std::int64_t>::max());
    auto aboveMinimum = arith::CmpIOp::create(
        opBuilder, loc, arith::CmpIPredicate::sge, productWide, minimum);
    auto belowMaximum = arith::CmpIOp::create(
        opBuilder, loc, arith::CmpIPredicate::sle, productWide, maximum);
    auto fits =
        arith::AndIOp::create(opBuilder, loc, aboveMinimum, belowMaximum);
    auto notRequired = arith::XOrIOp::create(
        opBuilder, loc, condition,
        arith::ConstantIntOp::create(opBuilder, loc, 1, 1));
    auto valid = arith::OrIOp::create(opBuilder, loc, notRequired, fits);
    cf::AssertOp::create(opBuilder, loc, valid, "integer power overflows i64");
    auto product = arith::TruncIOp::create(opBuilder, loc,
                                           opBuilder.getI64Type(), productWide);
    return arith::SelectOp::create(opBuilder, loc, condition, product, lhs);
  }

  [[nodiscard]] Value emitIntegerPower(OpBuilder& opBuilder, Location loc,
                                       Value base, Value exponent,
                                       const bool isUnsigned) {
    auto zero = arith::ConstantIntOp::create(opBuilder, loc, 0, 64);
    auto one = arith::ConstantIntOp::create(opBuilder, loc, 1, 64);
    if (!isUnsigned) {
      auto nonnegative = arith::CmpIOp::create(
          opBuilder, loc, arith::CmpIPredicate::sge, exponent, zero);
      cf::AssertOp::create(opBuilder, loc, nonnegative,
                           "integer power requires a nonnegative exponent");
    }
    auto power = scf::WhileOp::create(
        opBuilder, loc,
        TypeRange{base.getType(), base.getType(), exponent.getType()},
        ValueRange{one, base, exponent},
        [&](OpBuilder& nested, Location nestedLoc, ValueRange arguments) {
          auto active = arith::CmpIOp::create(
              nested, nestedLoc, arith::CmpIPredicate::ne, arguments[2], zero);
          scf::ConditionOp::create(nested, nestedLoc, active, arguments);
        },
        [&](OpBuilder& nested, Location nestedLoc, ValueRange arguments) {
          auto lowBit =
              arith::AndIOp::create(nested, nestedLoc, arguments[2], one);
          auto odd = arith::CmpIOp::create(
              nested, nestedLoc, arith::CmpIPredicate::ne, lowBit, zero);
          auto nextResult = conditionalIntegerMultiply(
              nested, nestedLoc, odd, arguments[0], arguments[1], isUnsigned);
          auto nextExponent =
              arith::ShRUIOp::create(nested, nestedLoc, arguments[2], one);
          auto squareBase = arith::CmpIOp::create(
              nested, nestedLoc, arith::CmpIPredicate::ne, nextExponent, zero);
          auto nextBase = conditionalIntegerMultiply(nested, nestedLoc,
                                                     squareBase, arguments[1],
                                                     arguments[1], isUnsigned);
          scf::YieldOp::create(nested, nestedLoc,
                               ValueRange{nextResult, nextBase, nextExponent});
        });
    return power.getResult(0);
  }

  Value emitExpression(OpBuilder& opBuilder, const frontend::ExpressionId id,
                       ValueRange gateParameters) {
    const auto& expression = program.expressions.at(id);
    auto loc = opBuilder.getInsertionPoint() == opBuilder.getBlock()->end()
                   ? opBuilder.getUnknownLoc()
                   : opBuilder.getInsertionPoint()->getLoc();
    switch (expression.kind) {
    case frontend::ExpressionKind::Constant:
      switch (expression.type) {
      case frontend::ScalarType::Bool:
        return arith::ConstantIntOp::create(
            opBuilder, loc, std::get<bool>(expression.constant), 1);
      case frontend::ScalarType::Int:
        return arith::ConstantIntOp::create(
            opBuilder, loc, std::get<std::int64_t>(expression.constant), 64);
      case frontend::ScalarType::Uint:
        return arith::ConstantOp::create(
            opBuilder, loc,
            IntegerAttr::get(opBuilder.getI64Type(),
                             APInt(64,
                                   std::get<std::uint64_t>(expression.constant),
                                   /*isSigned=*/false)));
      case frontend::ScalarType::Float:
        return arith::ConstantFloatOp::create(
            opBuilder, loc, opBuilder.getF64Type(),
            APFloat(std::get<double>(expression.constant)));
      }
      llvm_unreachable("unknown scalar type");
    case frontend::ExpressionKind::GateParameter:
      return gateParameters[expression.parameter];
    case frontend::ExpressionKind::Variable:
      return scalarValues.at(expression.variable);
    case frontend::ExpressionKind::Negate: {
      auto operand = emitExpression(opBuilder, expression.lhs, gateParameters);
      if (isa<FloatType>(operand.getType())) {
        return arith::NegFOp::create(opBuilder, loc, operand);
      }
      if (expression.type == frontend::ScalarType::Uint) {
        auto zero = arith::ConstantIntOp::create(opBuilder, loc, 0, 64);
        return arith::SubIOp::create(opBuilder, loc, zero, operand);
      }
      auto i128 = opBuilder.getIntegerType(128);
      auto zero = arith::ConstantIntOp::create(opBuilder, loc, 0, 128);
      auto operandWide = arith::ExtSIOp::create(opBuilder, loc, i128, operand);
      auto negated = arith::SubIOp::create(opBuilder, loc, zero, operandWide);
      return checkedSignedResult(opBuilder, loc, negated,
                                 "integer negation overflows i64");
    }
    case frontend::ExpressionKind::ArcCos:
    case frontend::ExpressionKind::ArcSin:
    case frontend::ExpressionKind::ArcTan:
    case frontend::ExpressionKind::Sin:
    case frontend::ExpressionKind::Cos:
    case frontend::ExpressionKind::Tan:
    case frontend::ExpressionKind::Exp:
    case frontend::ExpressionKind::Ln:
    case frontend::ExpressionKind::Sqrt: {
      Value operand = emitExpression(opBuilder, expression.lhs, gateParameters);
      if (isa<IntegerType>(operand.getType())) {
        const auto sourceType = program.expressions.at(expression.lhs).type;
        if (sourceType == frontend::ScalarType::Uint) {
          operand = arith::UIToFPOp::create(opBuilder, loc,
                                            opBuilder.getF64Type(), operand);
        } else {
          operand = arith::SIToFPOp::create(opBuilder, loc,
                                            opBuilder.getF64Type(), operand);
        }
      }
      switch (expression.kind) {
      case frontend::ExpressionKind::ArcCos:
        return math::AcosOp::create(opBuilder, loc, operand);
      case frontend::ExpressionKind::ArcSin:
        return math::AsinOp::create(opBuilder, loc, operand);
      case frontend::ExpressionKind::ArcTan:
        return math::AtanOp::create(opBuilder, loc, operand);
      case frontend::ExpressionKind::Sin:
        return math::SinOp::create(opBuilder, loc, operand);
      case frontend::ExpressionKind::Cos:
        return math::CosOp::create(opBuilder, loc, operand);
      case frontend::ExpressionKind::Tan:
        return math::TanOp::create(opBuilder, loc, operand);
      case frontend::ExpressionKind::Exp:
        return math::ExpOp::create(opBuilder, loc, operand);
      case frontend::ExpressionKind::Ln:
        return math::LogOp::create(opBuilder, loc, operand);
      case frontend::ExpressionKind::Sqrt:
        return math::SqrtOp::create(opBuilder, loc, operand);
      default:
        llvm_unreachable("unknown scalar math function");
      }
    }
    case frontend::ExpressionKind::Add:
    case frontend::ExpressionKind::Subtract:
    case frontend::ExpressionKind::Multiply:
    case frontend::ExpressionKind::Divide:
    case frontend::ExpressionKind::Modulo:
    case frontend::ExpressionKind::Power: {
      auto lhs = emitExpression(opBuilder, expression.lhs, gateParameters);
      auto rhs = emitExpression(opBuilder, expression.rhs, gateParameters);
      if (expression.type != frontend::ScalarType::Float) {
        const bool isUnsigned = expression.type == frontend::ScalarType::Uint;
        auto zero = arith::ConstantIntOp::create(opBuilder, loc, 0, 64);
        if (expression.kind == frontend::ExpressionKind::Divide ||
            expression.kind == frontend::ExpressionKind::Modulo) {
          auto nonzero = arith::CmpIOp::create(
              opBuilder, loc, arith::CmpIPredicate::ne, rhs, zero);
          cf::AssertOp::create(opBuilder, loc, nonzero,
                               expression.kind ==
                                       frontend::ExpressionKind::Divide
                                   ? "division by zero"
                                   : "modulo by zero");
          if (!isUnsigned) {
            auto minimum = arith::ConstantIntOp::create(
                opBuilder, loc, std::numeric_limits<std::int64_t>::min(), 64);
            auto minusOne =
                arith::ConstantIntOp::create(opBuilder, loc, -1, 64);
            auto lhsIsMinimum = arith::CmpIOp::create(
                opBuilder, loc, arith::CmpIPredicate::eq, lhs, minimum);
            auto rhsIsMinusOne = arith::CmpIOp::create(
                opBuilder, loc, arith::CmpIPredicate::eq, rhs, minusOne);
            auto overflows = arith::AndIOp::create(opBuilder, loc, lhsIsMinimum,
                                                   rhsIsMinusOne);
            auto valid = arith::XOrIOp::create(
                opBuilder, loc, overflows,
                arith::ConstantIntOp::create(opBuilder, loc, 1, 1));
            cf::AssertOp::create(opBuilder, loc, valid,
                                 "integer division overflows i64");
          }
          if (expression.kind == frontend::ExpressionKind::Divide) {
            return isUnsigned ? arith::DivUIOp::create(opBuilder, loc, lhs, rhs)
                                    .getResult()
                              : arith::DivSIOp::create(opBuilder, loc, lhs, rhs)
                                    .getResult();
          }
          return isUnsigned ? arith::RemUIOp::create(opBuilder, loc, lhs, rhs)
                                  .getResult()
                            : arith::RemSIOp::create(opBuilder, loc, lhs, rhs)
                                  .getResult();
        }
        if (expression.kind == frontend::ExpressionKind::Power) {
          return emitIntegerPower(opBuilder, loc, lhs, rhs, isUnsigned);
        }
        if (isUnsigned) {
          switch (expression.kind) {
          case frontend::ExpressionKind::Add:
            return arith::AddIOp::create(opBuilder, loc, lhs, rhs);
          case frontend::ExpressionKind::Subtract:
            return arith::SubIOp::create(opBuilder, loc, lhs, rhs);
          case frontend::ExpressionKind::Multiply:
            return arith::MulIOp::create(opBuilder, loc, lhs, rhs);
          default:
            llvm_unreachable("not an unsigned integer binary expression");
          }
        }
        auto i128 = opBuilder.getIntegerType(128);
        auto lhsWide = arith::ExtSIOp::create(opBuilder, loc, i128, lhs);
        auto rhsWide = arith::ExtSIOp::create(opBuilder, loc, i128, rhs);
        Value result;
        switch (expression.kind) {
        case frontend::ExpressionKind::Add:
          result = arith::AddIOp::create(opBuilder, loc, lhsWide, rhsWide);
          break;
        case frontend::ExpressionKind::Subtract:
          result = arith::SubIOp::create(opBuilder, loc, lhsWide, rhsWide);
          break;
        case frontend::ExpressionKind::Multiply:
          result = arith::MulIOp::create(opBuilder, loc, lhsWide, rhsWide);
          break;
        default:
          llvm_unreachable("not a signed integer binary expression");
        }
        return checkedSignedResult(opBuilder, loc, result,
                                   "integer arithmetic overflows i64");
      }
      const auto toFloat = [&](Value value,
                               const frontend::ScalarType sourceType) {
        if (isa<FloatType>(value.getType())) {
          return value;
        }
        if (sourceType == frontend::ScalarType::Uint) {
          return arith::UIToFPOp::create(opBuilder, loc, opBuilder.getF64Type(),
                                         value)
              .getResult();
        }
        return arith::SIToFPOp::create(opBuilder, loc, opBuilder.getF64Type(),
                                       value)
            .getResult();
      };
      auto floatLhs = toFloat(lhs, program.expressions.at(expression.lhs).type);
      auto floatRhs = toFloat(rhs, program.expressions.at(expression.rhs).type);
      switch (expression.kind) {
      case frontend::ExpressionKind::Add:
        return arith::AddFOp::create(opBuilder, loc, floatLhs, floatRhs);
      case frontend::ExpressionKind::Subtract:
        return arith::SubFOp::create(opBuilder, loc, floatLhs, floatRhs);
      case frontend::ExpressionKind::Multiply:
        return arith::MulFOp::create(opBuilder, loc, floatLhs, floatRhs);
      case frontend::ExpressionKind::Divide:
        return arith::DivFOp::create(opBuilder, loc, floatLhs, floatRhs);
      case frontend::ExpressionKind::Modulo:
        return arith::RemFOp::create(opBuilder, loc, floatLhs, floatRhs);
      case frontend::ExpressionKind::Power:
        return math::PowFOp::create(opBuilder, loc, floatLhs, floatRhs);
      default:
        llvm_unreachable("not a floating-point binary expression");
      }
    }
    }
    llvm_unreachable("unknown scalar expression kind");
  }

  [[nodiscard]] Value emitCheckedIndex(const frontend::ExpressionId expression,
                                       const std::int64_t width,
                                       const llvm::StringRef message) {
    auto index = emitExpression(builder, expression, {});
    auto zero = builder.intConstant(0);
    auto upper = builder.intConstant(width);
    Value inBounds;
    if (program.expressions.at(expression).type == frontend::ScalarType::Uint) {
      inBounds = arith::CmpIOp::create(builder, arith::CmpIPredicate::ult,
                                       index, upper);
    } else {
      auto negative = arith::CmpIOp::create(builder, arith::CmpIPredicate::slt,
                                            index, zero);
      auto wrapped = arith::AddIOp::create(builder, index, upper);
      index = arith::SelectOp::create(builder, negative, wrapped, index);
      auto nonnegative = arith::CmpIOp::create(
          builder, arith::CmpIPredicate::sge, index, zero);
      auto belowWidth = arith::CmpIOp::create(
          builder, arith::CmpIPredicate::slt, index, upper);
      inBounds = arith::AndIOp::create(builder, nonnegative, belowWidth);
    }
    cf::AssertOp::create(builder, inBounds, message);
    return index;
  }

  Value resolveQubit(const frontend::QubitReference& reference,
                     ValueRange gateQubits) {
    switch (reference.kind) {
    case frontend::QubitReferenceKind::Register: {
      assert(!reference.dynamicIndex &&
             "dynamic qubit references require structured dispatch");
      return registerValues.at(reference.symbol)[reference.index];
    }
    case frontend::QubitReferenceKind::GateArgument:
      return gateQubits[reference.symbol];
    case frontend::QubitReferenceKind::Hardware:
      return builder.staticQubit(reference.index);
    }
    llvm_unreachable("unknown qubit reference kind");
  }

  [[nodiscard]] SmallVector<Value>
  emitDynamicQubitIndices(ArrayRef<frontend::QubitReference> references) {
    SmallVector<Value> indices(references.size());
    for (const auto [position, reference] : llvm::enumerate(references)) {
      if (!reference.dynamicIndex) {
        continue;
      }
      const auto width = static_cast<std::int64_t>(
          program.registers.at(reference.symbol).width);
      indices[position] = emitCheckedIndex(*reference.dynamicIndex, width,
                                           "dynamic qubit index out of bounds");
    }
    return indices;
  }

  void
  dispatchQubits(ArrayRef<frontend::QubitReference> references,
                 ValueRange gateQubits, ValueRange dynamicIndices,
                 llvm::function_ref<void(ValueRange)> emitResolvedOperation) {
    SmallVector<Value> resolved(references.size());
    std::function<void(std::size_t)> resolveAt;
    resolveAt = [&](const std::size_t position) {
      if (position == references.size()) {
        emitResolvedOperation(resolved);
        return;
      }

      const auto& reference = references[position];
      if (!reference.dynamicIndex) {
        resolved[position] = resolveQubit(reference, gateQubits);
        resolveAt(position + 1);
        return;
      }

      const auto& qubits = registerValues.at(reference.symbol);
      std::function<void(std::size_t)> emitCase;
      emitCase = [&](const std::size_t candidate) {
        if (candidate + 1 == qubits.size()) {
          resolved[position] = qubits[candidate];
          resolveAt(position + 1);
          return;
        }
        auto matches = arith::CmpIOp::create(
            builder, arith::CmpIPredicate::eq, dynamicIndices[position],
            builder.intConstant(static_cast<std::int64_t>(candidate)));
        auto ifOp = scf::IfOp::create(builder, TypeRange{}, matches, true);
        OpBuilder::InsertionGuard guard(builder);
        const auto emitBranch = [&](Block& block,
                                    llvm::function_ref<void()> emitBody) {
          if (!block.empty()) {
            block.back().erase();
          }
          builder.setInsertionPointToEnd(&block);
          emitBody();
          scf::YieldOp::create(builder);
        };
        emitBranch(ifOp.getThenRegion().front(), [&] {
          resolved[position] = qubits[candidate];
          resolveAt(position + 1);
        });
        emitBranch(ifOp.getElseRegion().front(),
                   [&] { emitCase(candidate + 1); });
      };
      emitCase(0);
    };
    resolveAt(0);
  }

  [[nodiscard]] Value
  emitQubitOperation(const frontend::QubitReference& reference,
                     ValueRange gateQubits,
                     llvm::function_ref<Value(Value)> emitResolvedOperation) {
    if (!reference.dynamicIndex) {
      return emitResolvedOperation(resolveQubit(reference, gateQubits));
    }

    const auto dynamicIndex = emitDynamicQubitIndices({reference}).front();
    const auto& qubits = registerValues.at(reference.symbol);
    std::function<Value(std::size_t)> emitCase;
    emitCase = [&](const std::size_t candidate) -> Value {
      if (candidate + 1 == qubits.size()) {
        return emitResolvedOperation(qubits[candidate]);
      }
      auto matches = arith::CmpIOp::create(
          builder, arith::CmpIPredicate::eq, dynamicIndex,
          builder.intConstant(static_cast<std::int64_t>(candidate)));
      auto ifOp =
          scf::IfOp::create(builder, builder.getI1Type(), matches, true);
      OpBuilder::InsertionGuard guard(builder);
      const auto emitBranch = [&](Block& block,
                                  llvm::function_ref<Value()> emitValue) {
        if (!block.empty()) {
          block.back().erase();
        }
        builder.setInsertionPointToEnd(&block);
        scf::YieldOp::create(builder, emitValue());
      };
      emitBranch(ifOp.getThenRegion().front(),
                 [&] { return emitResolvedOperation(qubits[candidate]); });
      emitBranch(ifOp.getElseRegion().front(),
                 [&] { return emitCase(candidate + 1); });
      return ifOp.getResult(0);
    };
    return emitCase(0);
  }

  static LogicalResult emitPrimitive(OpBuilder& opBuilder, const Location loc,
                                     const StringRef name,
                                     const ValueRange parameters,
                                     const ValueRange qubits) {
    const auto operationName =
        llvm::StringSwitch<StringRef>(name)
            .Case("gphase", qc::GPhaseOp::getOperationName())
            .Case("id", qc::IdOp::getOperationName())
            .Case("x", qc::XOp::getOperationName())
            .Case("y", qc::YOp::getOperationName())
            .Case("z", qc::ZOp::getOperationName())
            .Case("h", qc::HOp::getOperationName())
            .Case("s", qc::SOp::getOperationName())
            .Case("sdg", qc::SdgOp::getOperationName())
            .Case("t", qc::TOp::getOperationName())
            .Case("tdg", qc::TdgOp::getOperationName())
            .Case("sx", qc::SXOp::getOperationName())
            .Case("sxdg", qc::SXdgOp::getOperationName())
            .Case("p", qc::POp::getOperationName())
            .Case("rx", qc::RXOp::getOperationName())
            .Case("ry", qc::RYOp::getOperationName())
            .Case("rz", qc::RZOp::getOperationName())
            .Case("r", qc::ROp::getOperationName())
            .Case("u2", qc::U2Op::getOperationName())
            .Case("U", qc::UOp::getOperationName())
            .Case("swap", qc::SWAPOp::getOperationName())
            .Case("iswap", qc::iSWAPOp::getOperationName())
            .Case("dcx", qc::DCXOp::getOperationName())
            .Case("ecr", qc::ECROp::getOperationName())
            .Case("rxx", qc::RXXOp::getOperationName())
            .Case("ryy", qc::RYYOp::getOperationName())
            .Case("rzx", qc::RZXOp::getOperationName())
            .Case("rzz", qc::RZZOp::getOperationName())
            .Case("xx_plus_yy", qc::XXPlusYYOp::getOperationName())
            .Case("xx_minus_yy", qc::XXMinusYYOp::getOperationName())
            .Default({});
    if (operationName.empty()) {
      return failure();
    }
    OperationState state(loc, operationName);
    if (name == "gphase") {
      state.addOperands(parameters);
    } else {
      state.addOperands(qubits);
      state.addOperands(parameters);
    }
    opBuilder.create(state);
    return success();
  }

  LogicalResult emitResolvedGate(OpBuilder& opBuilder,
                                 const frontend::GateApplication& application,
                                 const Location loc, ValueRange parameters,
                                 ValueRange qubits) {
    if (const auto* custom = findCustomGate(application.callee)) {
      if (parameters.size() != custom->parameterCount ||
          qubits.size() != custom->qubitCount) {
        llvm::errs() << "OpenQASM QC emission error: custom-gate operands do "
                        "not match its verified declaration.\n";
        return failure();
      }
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(opBuilder.getInsertionBlock(),
                                opBuilder.getInsertionPoint());
      for (const auto statement : custom->body) {
        emitStatement(statement, parameters, qubits);
      }
      return emissionFailed ? failure() : success();
    }

    const GateCatalogEntry* catalog =
        oq3::frontend::lookupGate(application.callee);
    if (catalog == nullptr || qubits.size() < catalog->targetCount) {
      return failure();
    }
    const std::size_t controls = catalog->variadicControls
                                     ? qubits.size() - catalog->targetCount
                                     : catalog->controlCount;
    if (qubits.size() < controls + catalog->targetCount) {
      return failure();
    }
    auto primitiveParameters = parameters;
    auto controlValues = qubits.take_front(controls);
    auto targets = qubits.drop_front(controls);
    if (application.callee == "cu") {
      if (controls != 1 || parameters.size() != 4 || targets.size() != 1) {
        return failure();
      }
      qc::POp::create(opBuilder, loc, controlValues.front(), parameters.back());
      primitiveParameters = parameters.drop_back();
    }
    const auto emitCatalogPrimitive = [&](ValueRange primitiveQubits) {
      if (!catalog->inverse) {
        return emitPrimitive(opBuilder, loc, catalog->primitive,
                             primitiveParameters, primitiveQubits);
      }
      LogicalResult result = success();
      qc::InvOp::create(
          opBuilder, loc, primitiveQubits, [&](ValueRange aliases) {
            result = emitPrimitive(opBuilder, loc, catalog->primitive,
                                   primitiveParameters, aliases);
          });
      return result;
    };
    if (controls == 0) {
      return emitCatalogPrimitive(qubits);
    }
    LogicalResult result = success();
    qc::CtrlOp::create(
        opBuilder, loc, controlValues, targets,
        [&](ValueRange aliases) { result = emitCatalogPrimitive(aliases); });
    return result;
  }

  LogicalResult emitModifiers(OpBuilder& opBuilder,
                              const frontend::GateApplication& application,
                              const Location loc, ValueRange parameters,
                              ArrayRef<std::int64_t> controlCounts,
                              const std::size_t position, ValueRange qubits) {
    if (position == application.modifiers.size()) {
      return emitResolvedGate(opBuilder, application, loc, parameters, qubits);
    }
    const auto kind = application.modifiers[position].kind;
    if (kind == frontend::ModifierKind::Inv) {
      LogicalResult result = success();
      qc::InvOp::create(opBuilder, loc, qubits, [&](ValueRange aliases) {
        result = emitModifiers(opBuilder, application, loc, parameters,
                               controlCounts, position + 1, aliases);
      });
      return result;
    }
    return emitControls(opBuilder, application, loc, parameters, controlCounts,
                        position + 1, controlCounts[position], qubits);
  }

  LogicalResult emitControls(OpBuilder& opBuilder,
                             const frontend::GateApplication& application,
                             const Location loc, ValueRange parameters,
                             ArrayRef<std::int64_t> controlCounts,
                             const std::size_t nextPosition,
                             const std::size_t remainingControls,
                             ValueRange qubits) {
    if (remainingControls == 0) {
      return emitModifiers(opBuilder, application, loc, parameters,
                           controlCounts, nextPosition, qubits);
    }
    LogicalResult result = success();
    qc::CtrlOp::create(opBuilder, loc, qubits.take_front(1),
                       qubits.drop_front(1), [&](ValueRange aliases) {
                         result = emitControls(opBuilder, application, loc,
                                               parameters, controlCounts,
                                               nextPosition,
                                               remainingControls - 1, aliases);
                       });
    return result;
  }

  void emitGateApplication(OpBuilder& opBuilder,
                           const frontend::GateApplication& application,
                           const Location loc, ValueRange gateParameters,
                           ValueRange gateQubits) {
    SmallVector<Value> parameters;
    parameters.reserve(application.parameters.size());
    for (const auto expression : application.parameters) {
      Value parameter = emitExpression(opBuilder, expression, gateParameters);
      if (isa<IntegerType>(parameter.getType())) {
        if (program.expressions.at(expression).type ==
            frontend::ScalarType::Uint) {
          parameter = arith::UIToFPOp::create(
              opBuilder, loc, opBuilder.getF64Type(), parameter);
        } else {
          parameter = arith::SIToFPOp::create(
              opBuilder, loc, opBuilder.getF64Type(), parameter);
        }
      }
      parameters.push_back(parameter);
    }
    const auto dynamicIndices = emitDynamicQubitIndices(application.qubits);
    for (const auto [position, reference] :
         llvm::enumerate(application.qubits)) {
      if (reference.kind != frontend::QubitReferenceKind::Register) {
        continue;
      }
      for (const auto [previousPosition, previous] :
           llvm::enumerate(ArrayRef(application.qubits).take_front(position))) {
        if (previous.kind != frontend::QubitReferenceKind::Register ||
            previous.symbol != reference.symbol ||
            (!previous.dynamicIndex && !reference.dynamicIndex)) {
          continue;
        }
        auto previousIndex =
            previous.dynamicIndex
                ? dynamicIndices[previousPosition]
                : builder.intConstant(
                      static_cast<std::int64_t>(previous.index));
        auto currentIndex = reference.dynamicIndex
                                ? dynamicIndices[position]
                                : builder.intConstant(static_cast<std::int64_t>(
                                      reference.index));
        auto distinct = arith::CmpIOp::create(builder, arith::CmpIPredicate::ne,
                                              previousIndex, currentIndex);
        cf::AssertOp::create(builder, distinct,
                             "gate operands must not reference the same qubit");
      }
    }
    SmallVector<std::int64_t> controlCounts(application.modifiers.size(), 0);
    for (const auto [position, modifier] :
         llvm::enumerate(application.modifiers)) {
      if (modifier.kind != frontend::ModifierKind::Ctrl &&
          modifier.kind != frontend::ModifierKind::NegCtrl) {
        continue;
      }
      std::int64_t count = 1;
      if (modifier.operand) {
        auto countValue =
            emitExpression(opBuilder, *modifier.operand, gateParameters);
        auto constant = countValue.getDefiningOp<arith::ConstantIntOp>();
        if (!constant || constant.value() <= 0) {
          emissionFailed = true;
          llvm::errs() << "OpenQASM QC emission error: gate control count "
                          "must be a positive constant integer.\n";
          return;
        }
        count = constant.value();
      }
      controlCounts[position] = count;
    }

    dispatchQubits(
        application.qubits, gateQubits, dynamicIndices, [&](ValueRange qubits) {
          llvm::DenseSet<Value> distinctQubits(qubits.begin(), qubits.end());
          if (distinctQubits.size() != qubits.size()) {
            return;
          }
          std::size_t negativeOffset = 0;
          for (const auto [position, modifier] :
               llvm::enumerate(application.modifiers)) {
            if (modifier.kind == frontend::ModifierKind::Ctrl ||
                modifier.kind == frontend::ModifierKind::NegCtrl) {
              if (modifier.kind == frontend::ModifierKind::NegCtrl) {
                for (auto control :
                     qubits.slice(negativeOffset, controlCounts[position])) {
                  qc::XOp::create(opBuilder, loc, control);
                }
              }
              negativeOffset +=
                  static_cast<std::size_t>(controlCounts[position]);
            }
          }
          const auto result =
              emitModifiers(opBuilder, application, loc, parameters,
                            controlCounts, 0, qubits);
          negativeOffset = 0;
          for (const auto [position, modifier] :
               llvm::enumerate(application.modifiers)) {
            if (modifier.kind == frontend::ModifierKind::Ctrl ||
                modifier.kind == frontend::ModifierKind::NegCtrl) {
              if (modifier.kind == frontend::ModifierKind::NegCtrl) {
                for (auto control :
                     qubits.slice(negativeOffset, controlCounts[position])) {
                  qc::XOp::create(opBuilder, loc, control);
                }
              }
              negativeOffset +=
                  static_cast<std::size_t>(controlCounts[position]);
            }
          }
          if (failed(result)) {
            emissionFailed = true;
            llvm::errs() << "OpenQASM QC emission error: gate '"
                         << application.callee
                         << "' has no lowering to the QC dialect.\n";
          }
        });
  }

  [[nodiscard]] Value coerceScalar(Value value,
                                   const frontend::ScalarType source,
                                   const frontend::ScalarType target) {
    if (source == target ||
        (source == frontend::ScalarType::Int &&
         target == frontend::ScalarType::Uint) ||
        (source == frontend::ScalarType::Uint &&
         target == frontend::ScalarType::Int)) {
      return value;
    }
    if (target == frontend::ScalarType::Float) {
      if (source == frontend::ScalarType::Bool ||
          source == frontend::ScalarType::Uint) {
        return arith::UIToFPOp::create(builder, builder.getF64Type(), value);
      }
      return arith::SIToFPOp::create(builder, builder.getF64Type(), value);
    }
    if (source == frontend::ScalarType::Bool) {
      return arith::ExtUIOp::create(builder, builder.getI64Type(), value);
    }
    if (source == frontend::ScalarType::Float &&
        target == frontend::ScalarType::Uint) {
      return arith::FPToUIOp::create(builder, builder.getI64Type(), value);
    }
    if (source == frontend::ScalarType::Float) {
      return arith::FPToSIOp::create(builder, builder.getI64Type(), value);
    }
    llvm_unreachable("unsupported standard scalar conversion");
  }

  [[nodiscard]] Value readBit(const frontend::BitReference& reference) {
    auto& values = bitValues.at(reference.reg);
    if (!reference.dynamicIndex) {
      return values[reference.index];
    }

    const auto width =
        static_cast<std::int64_t>(program.registers.at(reference.reg).width);
    auto index = emitCheckedIndex(*reference.dynamicIndex, width,
                                  "dynamic classical index out of bounds");

    Value selected = values.front();
    for (std::int64_t i = 1; i < width; ++i) {
      auto isIndex = arith::CmpIOp::create(builder, arith::CmpIPredicate::eq,
                                           index, builder.intConstant(i));
      selected = arith::SelectOp::create(builder, isIndex, values[i], selected);
    }
    return selected;
  }

  [[nodiscard]] Value
  emitComparison(const frontend::ConditionExpression& condition,
                 ValueRange gateParameters) {
    auto lhs = emitExpression(builder, condition.comparisonLhs, gateParameters);
    auto rhs = emitExpression(builder, condition.comparisonRhs, gateParameters);
    const auto lhsType = program.expressions.at(condition.comparisonLhs).type;
    const auto rhsType = program.expressions.at(condition.comparisonRhs).type;
    if (lhsType == frontend::ScalarType::Float ||
        rhsType == frontend::ScalarType::Float) {
      lhs = coerceScalar(lhs, lhsType, frontend::ScalarType::Float);
      rhs = coerceScalar(rhs, rhsType, frontend::ScalarType::Float);
      const auto predicate = [&] {
        switch (condition.comparison) {
        case frontend::ComparisonKind::Equal:
          return arith::CmpFPredicate::OEQ;
        case frontend::ComparisonKind::NotEqual:
          return arith::CmpFPredicate::UNE;
        case frontend::ComparisonKind::Less:
          return arith::CmpFPredicate::OLT;
        case frontend::ComparisonKind::LessEqual:
          return arith::CmpFPredicate::OLE;
        case frontend::ComparisonKind::Greater:
          return arith::CmpFPredicate::OGT;
        case frontend::ComparisonKind::GreaterEqual:
          return arith::CmpFPredicate::OGE;
        }
        llvm_unreachable("unknown floating-point comparison");
      }();
      return arith::CmpFOp::create(builder, predicate, lhs, rhs);
    }

    const bool isUnsigned = lhsType == frontend::ScalarType::Uint ||
                            rhsType == frontend::ScalarType::Uint;
    const auto predicate = [&] {
      switch (condition.comparison) {
      case frontend::ComparisonKind::Equal:
        return arith::CmpIPredicate::eq;
      case frontend::ComparisonKind::NotEqual:
        return arith::CmpIPredicate::ne;
      case frontend::ComparisonKind::Less:
        return isUnsigned ? arith::CmpIPredicate::ult
                          : arith::CmpIPredicate::slt;
      case frontend::ComparisonKind::LessEqual:
        return isUnsigned ? arith::CmpIPredicate::ule
                          : arith::CmpIPredicate::sle;
      case frontend::ComparisonKind::Greater:
        return isUnsigned ? arith::CmpIPredicate::ugt
                          : arith::CmpIPredicate::sgt;
      case frontend::ComparisonKind::GreaterEqual:
        return isUnsigned ? arith::CmpIPredicate::uge
                          : arith::CmpIPredicate::sge;
      }
      llvm_unreachable("unknown integer comparison");
    }();
    return arith::CmpIOp::create(builder, predicate, lhs, rhs);
  }

  [[nodiscard]] Value emitCondition(const frontend::ConditionId id,
                                    ValueRange gateParameters,
                                    ValueRange gateQubits) {
    const auto& condition = program.conditions.at(id);
    switch (condition.kind) {
    case frontend::ConditionKind::Literal:
      return builder.boolConstant(condition.literal);
    case frontend::ConditionKind::Scalar:
      return scalarValues.at(condition.scalar);
    case frontend::ConditionKind::Bit:
      return readBit(condition.bit);
    case frontend::ConditionKind::Measurement:
      return emitQubitOperation(
          condition.measurement, gateQubits,
          [&](Value qubit) { return builder.measure(qubit); });
    case frontend::ConditionKind::Not:
      return arith::XOrIOp::create(
          builder, emitCondition(condition.lhs, gateParameters, gateQubits),
          builder.boolConstant(true));
    case frontend::ConditionKind::And: {
      auto lhs = emitCondition(condition.lhs, gateParameters, gateQubits);
      auto rhs = emitCondition(condition.rhs, gateParameters, gateQubits);
      return arith::AndIOp::create(builder, lhs, rhs);
    }
    case frontend::ConditionKind::Or: {
      auto lhs = emitCondition(condition.lhs, gateParameters, gateQubits);
      auto rhs = emitCondition(condition.rhs, gateParameters, gateQubits);
      return arith::OrIOp::create(builder, lhs, rhs);
    }
    case frontend::ConditionKind::Comparison:
      return emitComparison(condition, gateParameters);
    }
    llvm_unreachable("unknown condition kind");
  }

  static constexpr std::uint64_t scalarStateMask = std::uint64_t{1} << 63U;

  static std::uint64_t scalarStateKey(const frontend::ScalarId scalar) {
    return scalarStateMask | scalar;
  }

  static std::uint64_t bitStateKey(const frontend::RegisterId reg,
                                   const std::uint64_t bit) {
    return (static_cast<std::uint64_t>(reg) << 32U) | bit;
  }

  void collectMutations(const frontend::StatementId id,
                        llvm::DenseSet<std::uint64_t>& mutations) const {
    const auto& statement = program.statements.at(id);
    std::visit(
        [&](const auto& data) {
          using T = std::decay_t<decltype(data)>;
          if constexpr (std::is_same_v<T,
                                       frontend::ScalarDeclarationStatement> ||
                        std::is_same_v<T,
                                       frontend::ScalarAssignmentStatement>) {
            mutations.insert(scalarStateKey(data.scalar));
          } else if constexpr (std::is_same_v<T,
                                              frontend::MeasurementStatement>) {
            for (const auto& target : data.targets) {
              if (!target.dynamicIndex) {
                mutations.insert(bitStateKey(target.reg, target.index));
                continue;
              }
              for (std::uint64_t bit = 0;
                   bit < program.registers.at(target.reg).width; ++bit) {
                mutations.insert(bitStateKey(target.reg, bit));
              }
            }
          } else if constexpr (std::is_same_v<
                                   T, frontend::BitAssignmentStatement>) {
            if (!data.target.dynamicIndex) {
              mutations.insert(bitStateKey(data.target.reg, data.target.index));
            } else {
              for (std::uint64_t bit = 0;
                   bit < program.registers.at(data.target.reg).width; ++bit) {
                mutations.insert(bitStateKey(data.target.reg, bit));
              }
            }
          } else if constexpr (std::is_same_v<T, frontend::IfStatement>) {
            for (const auto nested : data.thenStatements) {
              collectMutations(nested, mutations);
            }
            for (const auto nested : data.elseStatements) {
              collectMutations(nested, mutations);
            }
          } else if constexpr (std::is_same_v<T, frontend::ForStatement> ||
                               std::is_same_v<T, frontend::WhileStatement>) {
            for (const auto nested : data.body) {
              collectMutations(nested, mutations);
            }
          }
        },
        statement.data);
  }

  [[nodiscard]] SmallVector<StateSlot>
  mutatedState(ArrayRef<frontend::StatementId> statements) const {
    llvm::DenseSet<std::uint64_t> mutations;
    for (const auto statement : statements) {
      collectMutations(statement, mutations);
    }
    SmallVector<StateSlot> slots;
    for (const auto [scalar, value] : llvm::enumerate(scalarValues)) {
      if (value && mutations.contains(scalarStateKey(
                       static_cast<frontend::ScalarId>(scalar)))) {
        slots.push_back({.kind = StateKind::Scalar,
                         .first = static_cast<std::uint32_t>(scalar)});
      }
    }
    for (const auto [reg, values] : llvm::enumerate(bitValues)) {
      for (const auto [bit, value] : llvm::enumerate(values)) {
        if (value && mutations.contains(bitStateKey(
                         static_cast<frontend::RegisterId>(reg), bit))) {
          slots.push_back({.kind = StateKind::Bit,
                           .first = static_cast<std::uint32_t>(reg),
                           .second = static_cast<std::uint32_t>(bit)});
        }
      }
    }
    return slots;
  }

  [[nodiscard]] SmallVector<Value>
  stateValues(ArrayRef<StateSlot> slots) const {
    SmallVector<Value> values;
    values.reserve(slots.size());
    for (const auto& slot : slots) {
      values.push_back(slot.kind == StateKind::Scalar
                           ? scalarValues.at(slot.first)
                           : bitValues.at(slot.first)[slot.second]);
    }
    return values;
  }

  void assignState(ArrayRef<StateSlot> slots, ValueRange values) {
    for (const auto [slot, value] : llvm::zip_equal(slots, values)) {
      if (slot.kind == StateKind::Scalar) {
        scalarValues.at(slot.first) = value;
      } else {
        bitValues.at(slot.first)[slot.second] = value;
      }
    }
  }

  void emitStatement(const frontend::StatementId id, ValueRange gateParameters,
                     ValueRange gateQubits) {
    const auto& statement = program.statements.at(id);
    const auto loc = getLocation(statement.location);
    builder.setLoc(loc);
    std::visit(
        [&](const auto& data) {
          using T = std::decay_t<decltype(data)>;
          if constexpr (std::is_same_v<T, frontend::DeclarationStatement>) {
            emitDeclaration(data);
          } else if constexpr (std::is_same_v<
                                   T, frontend::ScalarDeclarationStatement>) {
            emitScalarDeclaration(data, gateQubits);
          } else if constexpr (std::is_same_v<
                                   T, frontend::ScalarAssignmentStatement>) {
            emitScalarAssignment(data, gateQubits);
          } else if constexpr (std::is_same_v<
                                   T, frontend::BitAssignmentStatement>) {
            emitBitAssignment(data, gateQubits);
          } else if constexpr (std::is_same_v<T, frontend::GateApplication>) {
            emitGateApplication(builder, data, loc, gateParameters, gateQubits);
          } else if constexpr (std::is_same_v<T,
                                              frontend::MeasurementStatement>) {
            emitMeasurement(data, gateQubits);
          } else if constexpr (std::is_same_v<T, frontend::ResetStatement>) {
            for (const auto& qubit : data.qubits) {
              const auto indices = emitDynamicQubitIndices({qubit});
              dispatchQubits({qubit}, gateQubits, indices,
                             [&](ValueRange resolved) {
                               builder.reset(resolved.front());
                             });
            }
          } else if constexpr (std::is_same_v<T, frontend::BarrierStatement>) {
            const auto indices = emitDynamicQubitIndices(data.qubits);
            dispatchQubits(data.qubits, gateQubits, indices,
                           [&](ValueRange qubits) { builder.barrier(qubits); });
          } else if constexpr (std::is_same_v<T, frontend::IfStatement>) {
            emitIf(data, gateParameters, gateQubits);
          } else if constexpr (std::is_same_v<T, frontend::ForStatement>) {
            emitFor(data, gateParameters, gateQubits);
          } else if constexpr (std::is_same_v<T, frontend::WhileStatement>) {
            emitWhile(data, gateParameters, gateQubits);
          }
        },
        statement.data);
  }

  [[nodiscard]] Type scalarType(const frontend::ScalarType type) {
    switch (type) {
    case frontend::ScalarType::Bool:
      return builder.getI1Type();
    case frontend::ScalarType::Int:
    case frontend::ScalarType::Uint:
      return builder.getI64Type();
    case frontend::ScalarType::Float:
      return builder.getF64Type();
    }
    llvm_unreachable("unknown scalar type");
  }

  void
  emitScalarDeclaration(const frontend::ScalarDeclarationStatement& statement,
                        ValueRange gateQubits) {
    const auto type = program.scalars.at(statement.scalar).type;
    Value value = ub::PoisonOp::create(builder, scalarType(type)).getResult();
    if (statement.initializer) {
      const auto source = program.expressions.at(*statement.initializer).type;
      value = coerceScalar(emitExpression(builder, *statement.initializer, {}),
                           source, type);
    } else if (statement.conditionInitializer) {
      value = emitCondition(*statement.conditionInitializer, {}, gateQubits);
    }
    scalarValues.at(statement.scalar) = value;
  }

  void
  emitScalarAssignment(const frontend::ScalarAssignmentStatement& statement,
                       ValueRange gateQubits) {
    const auto type = program.scalars.at(statement.scalar).type;
    if (statement.value) {
      const auto source = program.expressions.at(*statement.value).type;
      scalarValues.at(statement.scalar) = coerceScalar(
          emitExpression(builder, *statement.value, {}), source, type);
      return;
    }
    scalarValues.at(statement.scalar) =
        emitCondition(*statement.condition, {}, gateQubits);
  }

  void emitDeclaration(const frontend::DeclarationStatement& statement) {
    const auto& declaration = program.registers.at(statement.reg);
    if (declaration.kind == frontend::RegisterKind::Qubit) {
      auto allocation = builder.allocQubitRegister(
          static_cast<std::int64_t>(declaration.width));
      registerValues[statement.reg] = std::move(allocation.qubits);
      return;
    }
    classicalRegisters[statement.reg] = builder.allocClassicalBitRegister(
        static_cast<std::int64_t>(declaration.width), declaration.name);
    bitValues[statement.reg].resize(declaration.width);
    auto poison =
        ub::PoisonOp::create(builder, builder.getI1Type()).getResult();
    llvm::fill(bitValues[statement.reg], poison);
  }

  void assignBit(const frontend::BitReference& target, Value value) {
    if (!target.dynamicIndex) {
      bitValues[target.reg][target.index] = value;
      return;
    }
    const auto width =
        static_cast<std::int64_t>(program.registers.at(target.reg).width);
    auto index = emitCheckedIndex(*target.dynamicIndex, width,
                                  "dynamic classical index out of bounds");
    for (std::int64_t bit = 0; bit < width; ++bit) {
      auto selected = arith::CmpIOp::create(builder, arith::CmpIPredicate::eq,
                                            index, builder.intConstant(bit));
      bitValues[target.reg][bit] = arith::SelectOp::create(
          builder, selected, value, bitValues[target.reg][bit]);
    }
  }

  void emitBitAssignment(const frontend::BitAssignmentStatement& assignment,
                         ValueRange gateQubits) {
    assignBit(assignment.target,
              emitCondition(assignment.value, {}, gateQubits));
  }

  void emitMeasurement(const frontend::MeasurementStatement& measurement,
                       ValueRange gateQubits) {
    if (measurement.targets.empty()) {
      for (const auto& qubit : measurement.qubits) {
        const auto indices = emitDynamicQubitIndices({qubit});
        dispatchQubits({qubit}, gateQubits, indices, [&](ValueRange resolved) {
          (void)builder.measure(resolved.front());
        });
      }
      return;
    }
    for (const auto [target, qubit] :
         llvm::zip_equal(measurement.targets, measurement.qubits)) {
      const auto& reg = classicalRegisters[target.reg];
      if (!reg) {
        llvm::errs() << "OpenQASM emission error: measurement target has no "
                        "classical storage.\n";
        return;
      }
      const auto emitMeasurement = [&](Value resolved) {
        if (target.dynamicIndex) {
          return builder.measure(resolved);
        }
        return builder.measure(resolved,
                               (*reg)[static_cast<std::int64_t>(target.index)]);
      };
      auto measured = emitQubitOperation(qubit, gateQubits, emitMeasurement);
      if (!measured) {
        return;
      }
      if (!target.dynamicIndex) {
        bitValues[target.reg][target.index] = measured;
        continue;
      }
      assignBit(target, measured);
    }
  }

  void emitIf(const frontend::IfStatement& conditional,
              ValueRange gateParameters, ValueRange gateQubits) {
    const auto& typedCondition = program.conditions.at(conditional.condition);
    if (typedCondition.kind == frontend::ConditionKind::Literal) {
      const auto& selected = typedCondition.literal
                                 ? conditional.thenStatements
                                 : conditional.elseStatements;
      for (const auto statement : selected) {
        emitStatement(statement, gateParameters, gateQubits);
      }
      return;
    }
    auto condition =
        emitCondition(conditional.condition, gateParameters, gateQubits);
    SmallVector<frontend::StatementId> nestedStatements(
        conditional.thenStatements.begin(), conditional.thenStatements.end());
    nestedStatements.append(conditional.elseStatements.begin(),
                            conditional.elseStatements.end());
    const auto slots = mutatedState(nestedStatements);
    const auto initialValues = stateValues(slots);
    const auto savedScalars = scalarValues;
    const auto savedBits = bitValues;
    const auto* thenStatements = &conditional.thenStatements;
    const auto* elseStatements = &conditional.elseStatements;
    if (slots.empty() && thenStatements->empty() && !elseStatements->empty()) {
      condition =
          arith::XOrIOp::create(builder, condition, builder.boolConstant(true));
      std::swap(thenStatements, elseStatements);
    }
    const bool withElseRegion = !elseStatements->empty() || !slots.empty();
    auto ifOp = scf::IfOp::create(builder, ValueRange(initialValues).getTypes(),
                                  condition, withElseRegion);
    OpBuilder::InsertionGuard guard(builder);
    const auto emitBranch = [&](Block& block,
                                ArrayRef<frontend::StatementId> statements) {
      scalarValues = savedScalars;
      bitValues = savedBits;
      if (!block.empty()) {
        block.back().erase();
      }
      builder.setInsertionPointToEnd(&block);
      for (const auto statement : statements) {
        emitStatement(statement, gateParameters, gateQubits);
      }
      scf::YieldOp::create(builder, stateValues(slots));
    };
    emitBranch(ifOp.getThenRegion().front(), *thenStatements);
    if (withElseRegion) {
      emitBranch(ifOp.getElseRegion().front(), *elseStatements);
    }
    scalarValues = savedScalars;
    bitValues = savedBits;
    assignState(slots, ifOp.getResults());
  }

  [[nodiscard]] Value extendRangeValue(Value value, Type targetType,
                                       const bool isUnsigned) {
    if (isUnsigned) {
      return arith::ExtUIOp::create(builder, targetType, value);
    }
    return arith::ExtSIOp::create(builder, targetType, value);
  }

  [[nodiscard]] std::optional<std::int64_t>
  constantRangeTripCount(const frontend::ForStatement& loop) const {
    const auto& startExpression = program.expressions.at(loop.start);
    const auto& stepExpression = program.expressions.at(loop.step);
    const auto& stopExpression = program.expressions.at(loop.stop);
    if (startExpression.kind != frontend::ExpressionKind::Constant ||
        stepExpression.kind != frontend::ExpressionKind::Constant ||
        stopExpression.kind != frontend::ExpressionKind::Constant) {
      return std::nullopt;
    }
    const bool unsignedEndpoints =
        startExpression.type == frontend::ScalarType::Uint ||
        stopExpression.type == frontend::ScalarType::Uint;
    const auto extendConstant = [](const frontend::ScalarExpression& expression,
                                   const bool asUnsigned) {
      const auto bits = expression.type == frontend::ScalarType::Uint
                            ? std::get<std::uint64_t>(expression.constant)
                            : static_cast<std::uint64_t>(
                                  std::get<std::int64_t>(expression.constant));
      const APInt value(64, bits);
      return asUnsigned ? value.zext(128) : value.sext(128);
    };
    const auto start = extendConstant(startExpression, unsignedEndpoints);
    const auto stop = extendConstant(stopExpression, unsignedEndpoints);
    const bool unsignedStep = stepExpression.type == frontend::ScalarType::Uint;
    const auto step = extendConstant(stepExpression, unsignedStep);
    if (step.isZero()) {
      return std::nullopt;
    }
    const bool positive = unsignedStep || !step.isNegative();
    const bool nonempty =
        positive ? (unsignedEndpoints ? start.ule(stop) : start.sle(stop))
                 : (unsignedEndpoints ? start.uge(stop) : start.sge(stop));
    if (!nonempty) {
      return 0;
    }
    const auto distance = positive ? stop - start : start - stop;
    const auto absoluteStep = positive ? step : -step;
    const auto count = distance.udiv(absoluteStep) + 1;
    const APInt maximum(128, static_cast<std::uint64_t>(
                                 std::numeric_limits<std::int64_t>::max()));
    if (count.ugt(maximum)) {
      return std::nullopt;
    }
    return static_cast<std::int64_t>(count.getZExtValue());
  }

  void emitFor(const frontend::ForStatement& loop, ValueRange gateParameters,
               ValueRange gateQubits) {
    const auto slots = mutatedState(loop.body);
    const auto initialValues = stateValues(slots);
    const auto savedScalars = scalarValues;
    const auto savedBits = bitValues;

    auto start = emitExpression(builder, loop.start, {});
    auto step = emitExpression(builder, loop.step, {});
    auto stop = emitExpression(builder, loop.stop, {});
    auto i128 = IntegerType::get(&context, 128);
    const bool unsignedEndpoints =
        program.expressions.at(loop.start).type == frontend::ScalarType::Uint ||
        program.expressions.at(loop.stop).type == frontend::ScalarType::Uint;
    auto startWide = extendRangeValue(start, i128, unsignedEndpoints);
    auto stepWide = extendRangeValue(step, i128,
                                     program.expressions.at(loop.step).type ==
                                         frontend::ScalarType::Uint);
    auto stopWide = extendRangeValue(stop, i128, unsignedEndpoints);
    auto zero = arith::ConstantIntOp::create(builder, 0, 128);
    if (const auto tripCount = constantRangeTripCount(loop)) {
      auto lowerBound = arith::ConstantIndexOp::create(builder, 0);
      auto upperBound = arith::ConstantIndexOp::create(builder, *tripCount);
      auto indexStep = arith::ConstantIndexOp::create(builder, 1);
      auto forOp = scf::ForOp::create(builder, lowerBound, upperBound,
                                      indexStep, initialValues);
      {
        OpBuilder::InsertionGuard guard(builder);
        auto* body = forOp.getBody();
        if (!body->empty()) {
          body->back().erase();
        }
        builder.setInsertionPointToEnd(body);
        scalarValues = savedScalars;
        bitValues = savedBits;
        assignState(slots, forOp.getRegionIterArgs());
        auto counter = arith::IndexCastOp::create(builder, builder.getI64Type(),
                                                  forOp.getInductionVar());
        auto counterWide = arith::ExtUIOp::create(builder, i128, counter);
        auto offset = arith::MulIOp::create(builder, counterWide, stepWide);
        auto inductionWide = arith::AddIOp::create(builder, startWide, offset);
        scalarValues.at(loop.inductionVariable) = arith::TruncIOp::create(
            builder, builder.getI64Type(), inductionWide);
        for (const auto statement : loop.body) {
          emitStatement(statement, gateParameters, gateQubits);
        }
        scf::YieldOp::create(builder, stateValues(slots));
      }
      scalarValues = savedScalars;
      bitValues = savedBits;
      assignState(slots, forOp.getResults());
      return;
    }

    if (program.expressions.at(loop.step).kind !=
        frontend::ExpressionKind::Constant) {
      auto nonzero = arith::CmpIOp::create(builder, arith::CmpIPredicate::ne,
                                           stepWide, zero);
      cf::AssertOp::create(builder, nonzero,
                           "for-loop range step must not be zero");
    }
    SmallVector<Type> resultTypes{i128};
    llvm::append_range(resultTypes, ValueRange(initialValues).getTypes());
    SmallVector<Value> operands{startWide};
    llvm::append_range(operands, initialValues);
    auto whileOp = scf::WhileOp::create(
        builder, resultTypes, operands,
        [&](OpBuilder& nested, Location loc, ValueRange arguments) {
          auto positive = arith::CmpIOp::create(
              nested, loc, arith::CmpIPredicate::sgt, stepWide, zero);
          auto ascending =
              arith::CmpIOp::create(nested, loc, arith::CmpIPredicate::sle,
                                    arguments.front(), stopWide);
          auto descending =
              arith::CmpIOp::create(nested, loc, arith::CmpIPredicate::sge,
                                    arguments.front(), stopWide);
          auto active = arith::SelectOp::create(nested, loc, positive,
                                                ascending, descending);
          scf::ConditionOp::create(nested, loc, active, arguments);
        },
        [&](OpBuilder& nested, Location, ValueRange arguments) {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPoint(nested.getInsertionBlock(),
                                    nested.getInsertionPoint());
          scalarValues = savedScalars;
          bitValues = savedBits;
          assignState(slots, arguments.drop_front());
          scalarValues.at(loop.inductionVariable) = arith::TruncIOp::create(
              builder, builder.getI64Type(), arguments.front());
          for (const auto statement : loop.body) {
            emitStatement(statement, gateParameters, gateQubits);
          }
          SmallVector<Value> yielded{
              arith::AddIOp::create(builder, arguments.front(), stepWide)};
          llvm::append_range(yielded, stateValues(slots));
          scf::YieldOp::create(builder, yielded);
        });
    scalarValues = savedScalars;
    bitValues = savedBits;
    assignState(slots, whileOp.getResults().drop_front());
  }

  void emitWhile(const frontend::WhileStatement& loop,
                 ValueRange gateParameters, ValueRange gateQubits) {
    const auto slots = mutatedState(loop.body);
    const auto initialValues = stateValues(slots);
    const auto savedScalars = scalarValues;
    const auto savedBits = bitValues;
    auto whileOp = scf::WhileOp::create(
        builder, ValueRange(initialValues).getTypes(), initialValues,
        [&](OpBuilder& nested, Location, ValueRange arguments) {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPoint(nested.getInsertionBlock(),
                                    nested.getInsertionPoint());
          scalarValues = savedScalars;
          bitValues = savedBits;
          assignState(slots, arguments);
          auto condition =
              emitCondition(loop.condition, gateParameters, gateQubits);
          scf::ConditionOp::create(builder, condition, stateValues(slots));
        },
        [&](OpBuilder& nested, Location, ValueRange arguments) {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPoint(nested.getInsertionBlock(),
                                    nested.getInsertionPoint());
          scalarValues = savedScalars;
          bitValues = savedBits;
          assignState(slots, arguments);
          for (const auto statement : loop.body) {
            emitStatement(statement, gateParameters, gateQubits);
          }
          scf::YieldOp::create(builder, stateValues(slots));
        });
    scalarValues = savedScalars;
    bitValues = savedBits;
    assignState(slots, whileOp.getResults());
  }
};

} // namespace

OwningOpRef<ModuleOp> emitOpenQASMToQC(const frontend::TypedProgram& program,
                                       MLIRContext& context) {
  return OpenQASMToQCEmitter(program, context).emit();
}

} // namespace mlir::qc::detail
