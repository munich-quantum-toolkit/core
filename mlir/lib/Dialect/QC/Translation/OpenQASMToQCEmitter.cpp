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
#include <llvm/ADT/StringMap.h>
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
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OperationSupport.h>

#include <bit>
#include <cstddef>
#include <cstdint>
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
using oq3::frontend::GateLowering;

class OpenQASMToQCEmitter {
  class EmissionBudget final : public OpBuilder::Listener {
  public:
    explicit EmissionBudget(MLIRContext& mlirContext)
        : location(UnknownLoc::get(&mlirContext)) {}

    void setLocation(const Location newLocation) { location = newLocation; }

    [[nodiscard]] bool canConstruct(const std::size_t amount) {
      if (exhausted || amount > operationLimit - operationCount) {
        report();
        return false;
      }
      return true;
    }

    [[nodiscard]] bool isExhausted() const { return exhausted; }

    void notifyOperationInserted(Operation* /*operation*/,
                                 OpBuilder::InsertPoint /*previous*/) override {
      if (exhausted) {
        return;
      }
      ++operationCount;
      if (operationCount > operationLimit) {
        report();
      }
    }

    static constexpr std::size_t operationLimit = 100000;

  private:
    std::size_t operationCount = 0;
    Location location;
    bool exhausted = false;

    void report() {
      if (exhausted) {
        return;
      }
      exhausted = true;
      emitError(location)
          << "OpenQASM QC emission error: emitted operation count exceeds the "
             "safe lowering limit";
    }
  };

public:
  OpenQASMToQCEmitter(const oq3::frontend::TypedProgram& typedProgram,
                      MLIRContext& mlirContext)
      : program(typedProgram), context(mlirContext), emissionBudget(context),
        builder(&context), registerValues(program.registers.size()),
        classicalRegisters(program.registers.size()),
        bitValues(program.registers.size()),
        scalarValues(program.scalars.size()),
        expressionEmissionCosts(program.expressions.size()) {
    context
        .loadDialect<qc::QCDialect, arith::ArithDialect, cf::ControlFlowDialect,
                     func::FuncDialect, math::MathDialect,
                     memref::MemRefDialect, scf::SCFDialect, ub::UBDialect>();
    builder.setListener(&emissionBudget);
    builder.initialize();
    for (const auto& gate : program.gates) {
      customGateIndex.try_emplace(gate.name, &gate);
    }
  }

  OwningOpRef<ModuleOp> emit() {
    if (!preflight()) {
      return nullptr;
    }
    for (const auto statement : program.body) {
      emitStatement(statement, {}, {});
      if (emissionFailed || emissionBudget.isExhausted()) {
        return nullptr;
      }
    }

    SmallVector<Value> results;
    for (const auto output : program.outputs) {
      for (auto bit : bitValues[output]) {
        if (!bit) {
          emitError(getLocation(program.registers[output].location))
              << "OpenQASM QC emission error: output register '"
              << program.registers[output].name << "' is not fully initialized";
          return nullptr;
        }
        results.push_back(bit);
      }
    }
    OwningOpRef<ModuleOp> module;
    if (results.empty()) {
      module = builder.finalize();
    } else {
      builder.retype(ValueRange(results).getTypes());
      module = builder.finalize(results);
    }
    if (emissionBudget.isExhausted()) {
      return nullptr;
    }
    return module;
  }

private:
  const oq3::frontend::TypedProgram& program;
  MLIRContext& context;
  EmissionBudget emissionBudget;
  qc::QCProgramBuilder builder;
  std::vector<SmallVector<Value>> registerValues;
  std::vector<std::optional<qc::QCProgramBuilder::ClassicalRegister>>
      classicalRegisters;
  std::vector<SmallVector<Value>> bitValues;
  std::vector<Value> scalarValues;
  mutable std::vector<std::optional<std::size_t>> expressionEmissionCosts;
  DenseMap<const oq3::frontend::GateDefinition*, bool>
      structuredGateCapabilities;
  llvm::StringMap<const oq3::frontend::GateDefinition*> customGateIndex;
  bool emissionFailed = false;

  enum class StateKind : std::uint8_t { Scalar, Bit };

  struct StateSlot {
    StateKind kind = StateKind::Scalar;
    std::uint32_t first = 0;
    std::uint32_t second = 0;
  };

  [[nodiscard]] Location
  getLocation(const frontend::SourceLocation& source) const {
    return getOpenQASMLocation(source, context);
  }

  static constexpr std::size_t projectedEmissionLimit =
      EmissionBudget::operationLimit;

  [[nodiscard]] static bool
  isExactlyRepresentableAsDouble(const std::uint64_t magnitude) {
    if (magnitude == 0) {
      return true;
    }
    auto significand = magnitude;
    while ((significand & 1U) == 0) {
      significand >>= 1U;
    }
    return std::bit_width(significand) <= std::numeric_limits<double>::digits;
  }

  [[nodiscard]] static bool
  isExactlyRepresentableAsDouble(const frontend::ScalarExpression& expression) {
    if (expression.kind != frontend::ExpressionKind::Constant) {
      return true;
    }
    if (expression.type == frontend::ScalarType::Uint) {
      return isExactlyRepresentableAsDouble(
          std::get<std::uint64_t>(expression.constant));
    }
    if (expression.type != frontend::ScalarType::Int) {
      return true;
    }
    const auto value = std::get<std::int64_t>(expression.constant);
    const auto magnitude = value < 0
                               ? static_cast<std::uint64_t>(-(value + 1)) + 1U
                               : static_cast<std::uint64_t>(value);
    return isExactlyRepresentableAsDouble(magnitude);
  }

  [[nodiscard]] const oq3::frontend::GateDefinition*
  findCustomGate(const StringRef name) const {
    return customGateIndex.lookup(name);
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
    emitError(getLocation(source))
        << "OpenQASM QC emission error: projected emitted operation count "
           "exceeds the safe lowering limit";
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
  chargeScaledEmission(const std::size_t amount, const std::size_t multiplicity,
                       std::size_t& projectedEmission,
                       const oq3::frontend::SourceLocation& source) const {
    if (multiplicity != 0 && amount > projectedEmissionLimit / multiplicity) {
      return reportProjectedEmissionLimit(source);
    }
    return chargeProjectedEmission(amount * multiplicity, projectedEmission,
                                   source);
  }

  [[nodiscard]] std::size_t
  expressionEmissionCost(const frontend::ExpressionId id) const {
    if (expressionEmissionCosts[id]) {
      return *expressionEmissionCosts[id];
    }
    const auto& expression = program.expressions.at(id);
    const auto add = [](const std::size_t lhs, const std::size_t rhs) {
      return lhs > projectedEmissionLimit || rhs > projectedEmissionLimit ||
                     lhs > projectedEmissionLimit - rhs
                 ? projectedEmissionLimit + 1
                 : lhs + rhs;
    };
    const auto remember = [&](const std::size_t cost) {
      expressionEmissionCosts[id] = cost;
      return cost;
    };
    const auto unary = [&](const std::size_t local) {
      return add(expressionEmissionCost(expression.lhs), local);
    };
    const auto binary = [&](const std::size_t local) {
      return add(add(expressionEmissionCost(expression.lhs),
                     expressionEmissionCost(expression.rhs)),
                 local);
    };
    switch (expression.kind) {
    case frontend::ExpressionKind::Constant:
      return remember(1);
    case frontend::ExpressionKind::GateParameter:
    case frontend::ExpressionKind::Variable:
      return remember(0);
    case frontend::ExpressionKind::Negate:
      if (expression.type == frontend::ScalarType::Float) {
        return remember(unary(1));
      }
      return remember(
          unary(expression.type == frontend::ScalarType::Uint ? 2 : 10));
    case frontend::ExpressionKind::ArcCos:
    case frontend::ExpressionKind::ArcSin:
    case frontend::ExpressionKind::ArcTan:
    case frontend::ExpressionKind::Sin:
    case frontend::ExpressionKind::Cos:
    case frontend::ExpressionKind::Tan:
    case frontend::ExpressionKind::Exp:
    case frontend::ExpressionKind::Ln:
    case frontend::ExpressionKind::Sqrt:
      return remember(unary(2));
    case frontend::ExpressionKind::Add:
    case frontend::ExpressionKind::Subtract:
    case frontend::ExpressionKind::Multiply:
      if (expression.type == frontend::ScalarType::Float) {
        return remember(binary(3));
      }
      return remember(
          binary(expression.type == frontend::ScalarType::Uint ? 2 : 11));
    case frontend::ExpressionKind::Divide:
    case frontend::ExpressionKind::Modulo:
      if (expression.type == frontend::ScalarType::Float) {
        return remember(binary(3));
      }
      return remember(
          binary(expression.type == frontend::ScalarType::Uint ? 5 : 13));
    case frontend::ExpressionKind::Power:
      if (expression.type == frontend::ScalarType::Float) {
        return remember(binary(3));
      }
      return remember(
          binary(expression.type == frontend::ScalarType::Uint ? 16 : 42));
    }
    llvm_unreachable("unknown scalar expression kind");
  }

  [[nodiscard]] bool
  chargeExpressionEmission(const frontend::ExpressionId id,
                           const std::size_t multiplicity,
                           std::size_t& projectedEmission,
                           const frontend::SourceLocation& source) const {
    return chargeScaledEmission(expressionEmissionCost(id), multiplicity,
                                projectedEmission, source);
  }

  [[nodiscard]] bool
  chargeDynamicBitRead(const frontend::BitReference& reference,
                       const std::size_t multiplicity,
                       std::size_t& projectedEmission,
                       const frontend::SourceLocation& source) const {
    if (!reference.dynamicIndex) {
      return true;
    }
    const auto width =
        static_cast<std::size_t>(program.registers.at(reference.reg).width);
    return chargeExpressionEmission(*reference.dynamicIndex, multiplicity,
                                    projectedEmission, source) &&
           chargeScaledEmission(9 + 3 * (width - 1), multiplicity,
                                projectedEmission, source);
  }

  [[nodiscard]] bool
  chargeDynamicDispatch(const ArrayRef<frontend::QubitReference> references,
                        const std::size_t parentMultiplicity,
                        std::size_t& projectedEmission,
                        const oq3::frontend::SourceLocation& source) const {
    auto switchMultiplicity = parentMultiplicity;
    for (const auto& reference : references) {
      if (!reference.dynamicIndex) {
        continue;
      }
      const auto width = static_cast<std::size_t>(
          program.registers.at(reference.symbol).width);
      // Checked-index normalization plus the index cast and switch operation.
      if (!chargeExpressionEmission(*reference.dynamicIndex, switchMultiplicity,
                                    projectedEmission, source) ||
          !chargeScaledEmission(9, switchMultiplicity, projectedEmission,
                                source) ||
          !chargeScaledEmission(width, switchMultiplicity, projectedEmission,
                                source)) {
        return false;
      }
      if (width != 0 && switchMultiplicity > projectedEmissionLimit / width) {
        return reportProjectedEmissionLimit(source);
      }
      switchMultiplicity *= width;
    }
    return true;
  }

  [[nodiscard]] std::size_t
  modifierEmissionCost(const frontend::GateApplication& application) const {
    auto cost = application.modifiers.size();
    for (const auto& modifier : application.modifiers) {
      if (modifier.kind == frontend::ModifierKind::Pow) {
        const auto& expression = program.expressions.at(*modifier.operand);
        if (expression.kind != frontend::ExpressionKind::Constant &&
            (expression.type == frontend::ScalarType::Int ||
             expression.type == frontend::ScalarType::Uint)) {
          cost += expression.type == frontend::ScalarType::Int ? 17 : 14;
        }
        continue;
      }
      if (modifier.kind != frontend::ModifierKind::NegCtrl) {
        continue;
      }
      std::uint64_t controls = 1;
      if (modifier.operand) {
        const auto& expression = program.expressions.at(*modifier.operand);
        controls = expression.type == frontend::ScalarType::Uint
                       ? std::get<std::uint64_t>(expression.constant)
                       : static_cast<std::uint64_t>(
                             std::get<std::int64_t>(expression.constant));
      }
      if (controls > projectedEmissionLimit / 2) {
        return projectedEmissionLimit + 1;
      }
      cost += static_cast<std::size_t>(2 * controls);
    }
    return cost;
  }

  [[nodiscard]] bool
  chargeConditionEmission(const frontend::ConditionId id,
                          const std::size_t multiplicity,
                          std::size_t& projectedEmission,
                          const oq3::frontend::SourceLocation& source) const {
    const auto& condition = program.conditions.at(id);
    if (condition.kind == frontend::ConditionKind::Measurement) {
      std::size_t operationMultiplicity = 0;
      return chargeDynamicDispatch({condition.measurement}, multiplicity,
                                   projectedEmission, source) &&
             projectedMultiplicity({condition.measurement}, multiplicity,
                                   source, operationMultiplicity) &&
             chargeProjectedEmission(operationMultiplicity, projectedEmission,
                                     source);
    }
    if (condition.kind == frontend::ConditionKind::Literal) {
      return chargeScaledEmission(1, multiplicity, projectedEmission, source);
    }
    if (condition.kind == frontend::ConditionKind::Bit) {
      return chargeDynamicBitRead(condition.bit, multiplicity,
                                  projectedEmission, source);
    }
    if (condition.kind == frontend::ConditionKind::Comparison) {
      return chargeExpressionEmission(condition.comparisonLhs, multiplicity,
                                      projectedEmission, source) &&
             chargeExpressionEmission(condition.comparisonRhs, multiplicity,
                                      projectedEmission, source) &&
             chargeScaledEmission(3, multiplicity, projectedEmission, source);
    }
    if (condition.kind == frontend::ConditionKind::Not) {
      return chargeConditionEmission(condition.lhs, multiplicity,
                                     projectedEmission, source) &&
             chargeScaledEmission(2, multiplicity, projectedEmission, source);
    }
    if (condition.kind == frontend::ConditionKind::And ||
        condition.kind == frontend::ConditionKind::Or) {
      return chargeConditionEmission(condition.lhs, multiplicity,
                                     projectedEmission, source) &&
             chargeConditionEmission(condition.rhs, multiplicity,
                                     projectedEmission, source) &&
             chargeScaledEmission(5, multiplicity, projectedEmission, source);
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
                                   projectedEmission, multiplicity) ||
              !chargeScaledEmission(5, multiplicity, projectedEmission,
                                    statement.location)) {
            return false;
          }
        } else if (const auto* loop = std::get_if<oq3::frontend::ForStatement>(
                       &statement.data)) {
          if (!chargeScaledEmission(16, multiplicity, projectedEmission,
                                    statement.location) ||
              !chargeExpressionEmission(loop->start, multiplicity,
                                        projectedEmission,
                                        statement.location) ||
              !chargeExpressionEmission(loop->step, multiplicity,
                                        projectedEmission,
                                        statement.location) ||
              !chargeExpressionEmission(loop->stop, multiplicity,
                                        projectedEmission,
                                        statement.location) ||
              !preflightStatements(loop->body, projectedEmission,
                                   multiplicity)) {
            return false;
          }
        } else if (const auto* loop =
                       std::get_if<oq3::frontend::WhileStatement>(
                           &statement.data)) {
          if (!chargeConditionEmission(loop->condition, multiplicity,
                                       projectedEmission, statement.location) ||
              !chargeScaledEmission(10, multiplicity, projectedEmission,
                                    statement.location)) {
            return false;
          }
          if (!preflightStatements(loop->body, projectedEmission,
                                   multiplicity)) {
            return false;
          }
        } else if (const auto* declaration =
                       std::get_if<frontend::ScalarDeclarationStatement>(
                           &statement.data)) {
          if (!chargeScaledEmission(2, multiplicity, projectedEmission,
                                    statement.location) ||
              (declaration->initializer &&
               !chargeExpressionEmission(*declaration->initializer,
                                         multiplicity, projectedEmission,
                                         statement.location)) ||
              (declaration->conditionInitializer &&
               !chargeConditionEmission(*declaration->conditionInitializer,
                                        multiplicity, projectedEmission,
                                        statement.location))) {
            return false;
          }
        } else if (const auto* assignment =
                       std::get_if<frontend::ScalarAssignmentStatement>(
                           &statement.data)) {
          if ((assignment->value &&
               !chargeExpressionEmission(*assignment->value, multiplicity,
                                         projectedEmission,
                                         statement.location)) ||
              (assignment->condition &&
               !chargeConditionEmission(*assignment->condition, multiplicity,
                                        projectedEmission,
                                        statement.location)) ||
              !chargeScaledEmission(1, multiplicity, projectedEmission,
                                    statement.location)) {
            return false;
          }
        } else if (const auto* assignment =
                       std::get_if<frontend::BitAssignmentStatement>(
                           &statement.data)) {
          if (!chargeConditionEmission(assignment->value, multiplicity,
                                       projectedEmission, statement.location)) {
            return false;
          }
          if (assignment->target.dynamicIndex) {
            const auto width = static_cast<std::size_t>(
                program.registers.at(assignment->target.reg).width);
            if (!chargeExpressionEmission(*assignment->target.dynamicIndex,
                                          multiplicity, projectedEmission,
                                          statement.location) ||
                !chargeScaledEmission(9 + 3 * width, multiplicity,
                                      projectedEmission, statement.location)) {
              return false;
            }
          }
        } else if (const auto* declaration =
                       std::get_if<frontend::DeclarationStatement>(
                           &statement.data)) {
          const auto& reg = program.registers.at(declaration->reg);
          const auto cost = reg.kind == frontend::RegisterKind::Qubit
                                ? 1 + 2 * static_cast<std::size_t>(reg.width)
                                : 1;
          if (!chargeScaledEmission(cost, multiplicity, projectedEmission,
                                    statement.location)) {
            return false;
          }
        } else if (const auto* measurement =
                       std::get_if<frontend::MeasurementStatement>(
                           &statement.data)) {
          for (const auto& qubit : measurement->qubits) {
            std::size_t operationMultiplicity = 0;
            if (!chargeDynamicDispatch({qubit}, multiplicity, projectedEmission,
                                       statement.location) ||
                !projectedMultiplicity({qubit}, multiplicity,
                                       statement.location,
                                       operationMultiplicity) ||
                !chargeProjectedEmission(operationMultiplicity,
                                         projectedEmission,
                                         statement.location)) {
              return false;
            }
          }
          for (const auto& target : measurement->targets) {
            if (!target.dynamicIndex) {
              continue;
            }
            const auto width = static_cast<std::size_t>(
                program.registers.at(target.reg).width);
            if (!chargeExpressionEmission(*target.dynamicIndex, multiplicity,
                                          projectedEmission,
                                          statement.location) ||
                !chargeScaledEmission(9 + 3 * width, multiplicity,
                                      projectedEmission, statement.location)) {
              return false;
            }
          }
        } else if (const auto* reset =
                       std::get_if<frontend::ResetStatement>(&statement.data)) {
          for (const auto& qubit : reset->qubits) {
            std::size_t operationMultiplicity = 0;
            if (!chargeDynamicDispatch({qubit}, multiplicity, projectedEmission,
                                       statement.location) ||
                !projectedMultiplicity({qubit}, multiplicity,
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
          if (!chargeDynamicDispatch(barrier->qubits, multiplicity,
                                     projectedEmission, statement.location) ||
              !projectedMultiplicity(barrier->qubits, multiplicity,
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
        if (modifier.kind == oq3::frontend::ModifierKind::Pow &&
            !isExactlyRepresentableAsDouble(
                program.expressions.at(*modifier.operand))) {
          emitError(getLocation(statement.location))
              << "OpenQASM QC emission error: power modifier exponent cannot "
                 "be represented exactly as an f64";
          return false;
        }
      }
      for (const auto parameter : application->parameters) {
        if (!chargeExpressionEmission(parameter, multiplicity,
                                      projectedEmission, statement.location)) {
          return false;
        }
      }
      for (const auto& modifier : application->modifiers) {
        if (modifier.operand &&
            !chargeExpressionEmission(*modifier.operand, multiplicity,
                                      projectedEmission, statement.location)) {
          return false;
        }
      }
      std::size_t operationMultiplicity = 0;
      if (!projectedMultiplicity(application->qubits, multiplicity,
                                 statement.location, operationMultiplicity) ||
          !chargeDynamicDispatch(application->qubits, multiplicity,
                                 projectedEmission, statement.location)) {
        return false;
      }
      const auto* gate = findCustomGate(application->callee);
      if (gate == nullptr) {
        auto leafCost = modifierEmissionCost(*application) + 1;
        if (const auto* catalog =
                oq3::frontend::lookupGate(application->callee)) {
          if (catalog->controlCount != 0 || catalog->variadicControls) {
            ++leafCost;
          }
          if (catalog->lowering == GateLowering::CU ||
              catalog->lowering == GateLowering::U2 ||
              catalog->lowering == GateLowering::U3 ||
              (catalog->lowering == GateLowering::BuiltinU &&
               program.openQASM2)) {
            leafCost += 4;
          } else if (catalog->lowering == GateLowering::BuiltinU) {
            leafCost += 3;
          }
        }
        if (!chargeScaledEmission(leafCost, operationMultiplicity,
                                  projectedEmission, statement.location)) {
          return false;
        }
        continue;
      }
      if (!chargeScaledEmission(modifierEmissionCost(*application),
                                operationMultiplicity, projectedEmission,
                                statement.location)) {
        return false;
      }
      if (!application->modifiers.empty() &&
          gateRequiresStructuredControlFlow(*gate)) {
        emitError(getLocation(statement.location))
            << "OpenQASM QC emission error: modifiers on custom gates with "
               "structured control flow are not supported by the QC dialect";
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
    std::size_t projectedEmission = program.registers.size() + 4;
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

  [[nodiscard]] static Value
  emitExactlyRepresentableIntegerAsF64(OpBuilder& opBuilder, Location loc,
                                       Value integer, const bool isUnsigned) {
    auto zero = arith::ConstantIntOp::create(opBuilder, loc, 0, 64);
    Value magnitude = integer;
    if (!isUnsigned) {
      auto negative = arith::CmpIOp::create(
          opBuilder, loc, arith::CmpIPredicate::slt, integer, zero);
      auto negated = arith::SubIOp::create(opBuilder, loc, zero, integer);
      magnitude =
          arith::SelectOp::create(opBuilder, loc, negative, negated, integer);
    }

    auto one = arith::ConstantIntOp::create(opBuilder, loc, 1, 64);
    auto reduced = scf::WhileOp::create(
        opBuilder, loc, TypeRange{integer.getType()}, ValueRange{magnitude},
        [&](OpBuilder& nested, Location nestedLoc, ValueRange arguments) {
          auto lowBit =
              arith::AndIOp::create(nested, nestedLoc, arguments[0], one);
          auto even = arith::CmpIOp::create(
              nested, nestedLoc, arith::CmpIPredicate::eq, lowBit, zero);
          auto nonzero = arith::CmpIOp::create(
              nested, nestedLoc, arith::CmpIPredicate::ne, arguments[0], zero);
          auto hasTrailingZero =
              arith::AndIOp::create(nested, nestedLoc, even, nonzero);
          scf::ConditionOp::create(nested, nestedLoc, hasTrailingZero,
                                   arguments);
        },
        [&](OpBuilder& nested, Location nestedLoc, ValueRange arguments) {
          auto shifted =
              arith::ShRUIOp::create(nested, nestedLoc, arguments[0], one);
          scf::YieldOp::create(nested, nestedLoc, ValueRange{shifted});
        });
    auto maximumSignificand = arith::ConstantOp::create(
        opBuilder, loc,
        IntegerAttr::get(opBuilder.getI64Type(),
                         APInt(64, (std::uint64_t{1} << 53U) - 1U)));
    auto exact =
        arith::CmpIOp::create(opBuilder, loc, arith::CmpIPredicate::ule,
                              reduced.getResult(0), maximumSignificand);
    cf::AssertOp::create(
        opBuilder, loc, exact,
        "integer power modifier exponent cannot be represented exactly as an "
        "f64");
    return isUnsigned ? arith::UIToFPOp::create(opBuilder, loc,
                                                opBuilder.getF64Type(), integer)
                            .getResult()
                      : arith::SIToFPOp::create(opBuilder, loc,
                                                opBuilder.getF64Type(), integer)
                            .getResult();
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
    const auto resolveAt = [&](auto&& self,
                               const std::size_t position) -> void {
      if (position == references.size()) {
        emitResolvedOperation(resolved);
        return;
      }

      const auto& reference = references[position];
      if (!reference.dynamicIndex) {
        resolved[position] = resolveQubit(reference, gateQubits);
        self(self, position + 1);
        return;
      }

      const auto& qubits = registerValues.at(reference.symbol);
      if (!emissionBudget.canConstruct(qubits.size() + 2)) {
        return;
      }
      SmallVector<std::int64_t> cases;
      cases.reserve(qubits.size() - 1);
      for (std::size_t candidate = 0; candidate + 1 < qubits.size();
           ++candidate) {
        cases.push_back(static_cast<std::int64_t>(candidate));
      }
      auto selector = arith::IndexCastOp::create(
          builder, builder.getIndexType(), dynamicIndices[position]);
      auto switchOp = scf::IndexSwitchOp::create(builder, TypeRange{}, selector,
                                                 cases, cases.size());
      OpBuilder::InsertionGuard guard(builder);
      const auto emitCase = [&](Region& region, const std::size_t candidate) {
        auto& block = region.emplaceBlock();
        builder.setInsertionPointToEnd(&block);
        resolved[position] = qubits[candidate];
        self(self, position + 1);
        scf::YieldOp::create(builder);
      };
      for (const auto [candidate, region] :
           llvm::enumerate(switchOp.getCaseRegions())) {
        emitCase(region, candidate);
      }
      emitCase(switchOp.getDefaultRegion(), qubits.size() - 1);
    };
    resolveAt(resolveAt, 0);
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
    if (!emissionBudget.canConstruct(qubits.size() + 2)) {
      return {};
    }
    SmallVector<std::int64_t> cases;
    cases.reserve(qubits.size() - 1);
    for (std::size_t candidate = 0; candidate + 1 < qubits.size();
         ++candidate) {
      cases.push_back(static_cast<std::int64_t>(candidate));
    }
    auto selector = arith::IndexCastOp::create(builder, builder.getIndexType(),
                                               dynamicIndex);
    auto switchOp = scf::IndexSwitchOp::create(builder, builder.getI1Type(),
                                               selector, cases, cases.size());
    OpBuilder::InsertionGuard guard(builder);
    const auto emitCase = [&](Region& region, const std::size_t candidate) {
      auto& block = region.emplaceBlock();
      builder.setInsertionPointToEnd(&block);
      scf::YieldOp::create(builder, emitResolvedOperation(qubits[candidate]));
    };
    for (const auto [candidate, region] :
         llvm::enumerate(switchOp.getCaseRegions())) {
      emitCase(region, candidate);
    }
    emitCase(switchOp.getDefaultRegion(), qubits.size() - 1);
    return switchOp.getResult(0);
  }

  static LogicalResult emitPrimitive(OpBuilder& opBuilder, const Location loc,
                                     const GateLowering lowering,
                                     const ValueRange parameters,
                                     const ValueRange qubits) {
    StringRef operationName;
    switch (lowering) {
    case GateLowering::GPhase:
      operationName = qc::GPhaseOp::getOperationName();
      break;
    case GateLowering::Id:
      operationName = qc::IdOp::getOperationName();
      break;
    case GateLowering::X:
      operationName = qc::XOp::getOperationName();
      break;
    case GateLowering::Y:
      operationName = qc::YOp::getOperationName();
      break;
    case GateLowering::Z:
      operationName = qc::ZOp::getOperationName();
      break;
    case GateLowering::H:
      operationName = qc::HOp::getOperationName();
      break;
    case GateLowering::S:
      operationName = qc::SOp::getOperationName();
      break;
    case GateLowering::Sdg:
      operationName = qc::SdgOp::getOperationName();
      break;
    case GateLowering::T:
      operationName = qc::TOp::getOperationName();
      break;
    case GateLowering::Tdg:
      operationName = qc::TdgOp::getOperationName();
      break;
    case GateLowering::SX:
      operationName = qc::SXOp::getOperationName();
      break;
    case GateLowering::SXdg:
      operationName = qc::SXdgOp::getOperationName();
      break;
    case GateLowering::P:
      operationName = qc::POp::getOperationName();
      break;
    case GateLowering::RX:
      operationName = qc::RXOp::getOperationName();
      break;
    case GateLowering::RY:
      operationName = qc::RYOp::getOperationName();
      break;
    case GateLowering::RZ:
      operationName = qc::RZOp::getOperationName();
      break;
    case GateLowering::R:
      operationName = qc::ROp::getOperationName();
      break;
    case GateLowering::U2:
      operationName = qc::U2Op::getOperationName();
      break;
    case GateLowering::U3:
      operationName = qc::UOp::getOperationName();
      break;
    case GateLowering::SWAP:
      operationName = qc::SWAPOp::getOperationName();
      break;
    case GateLowering::ISWAP:
      operationName = qc::iSWAPOp::getOperationName();
      break;
    case GateLowering::DCX:
      operationName = qc::DCXOp::getOperationName();
      break;
    case GateLowering::ECR:
      operationName = qc::ECROp::getOperationName();
      break;
    case GateLowering::RCCX:
      operationName = qc::RCCXOp::getOperationName();
      break;
    case GateLowering::RXX:
      operationName = qc::RXXOp::getOperationName();
      break;
    case GateLowering::RYY:
      operationName = qc::RYYOp::getOperationName();
      break;
    case GateLowering::RZX:
      operationName = qc::RZXOp::getOperationName();
      break;
    case GateLowering::RZZ:
      operationName = qc::RZZOp::getOperationName();
      break;
    case GateLowering::XXPlusYY:
      operationName = qc::XXPlusYYOp::getOperationName();
      break;
    case GateLowering::XXMinusYY:
      operationName = qc::XXMinusYYOp::getOperationName();
      break;
    case GateLowering::BuiltinU:
    case GateLowering::CU:
      llvm_unreachable("compound gate lowering requires a dedicated recipe");
    }
    OperationState state(loc, operationName);
    if (lowering == GateLowering::GPhase) {
      state.addOperands(parameters);
    } else {
      state.addOperands(qubits);
      state.addOperands(parameters);
    }
    opBuilder.create(state);
    return success();
  }

  static Value
  emitOpenQASM3Phase(OpBuilder& opBuilder, const Location loc,
                     const ValueRange uParameters,
                     const std::optional<Value> extraPhase = std::nullopt) {
    assert(uParameters.size() == 3);
    auto half = arith::ConstantFloatOp::create(
        opBuilder, loc, opBuilder.getF64Type(), APFloat(0.5));
    Value result = arith::MulFOp::create(opBuilder, loc, uParameters[0], half);
    if (extraPhase) {
      result = arith::AddFOp::create(opBuilder, loc, result, *extraPhase);
    }
    return result;
  }

  static Value emitOpenQASM2UPhase(OpBuilder& opBuilder, const Location loc,
                                   const ValueRange uParameters) {
    assert(uParameters.size() >= 2);
    const auto phiIndex = uParameters.size() == 2 ? 0U : 1U;
    const auto lambdaIndex = uParameters.size() == 2 ? 1U : 2U;
    auto sum = arith::AddFOp::create(opBuilder, loc, uParameters[phiIndex],
                                     uParameters[lambdaIndex]);
    auto negativeHalf = arith::ConstantFloatOp::create(
        opBuilder, loc, opBuilder.getF64Type(), APFloat(-0.5));
    return arith::MulFOp::create(opBuilder, loc, sum, negativeHalf);
  }

  LogicalResult emitResolvedGate(OpBuilder& opBuilder,
                                 const frontend::GateApplication& application,
                                 const Location loc, ValueRange parameters,
                                 ValueRange qubits) {
    if (const auto* custom = findCustomGate(application.callee)) {
      if (parameters.size() != custom->parameterCount ||
          qubits.size() != custom->qubitCount) {
        emitError(loc)
            << "OpenQASM QC emission error: custom-gate operands do not match "
               "its verified declaration";
        return failure();
      }
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(opBuilder.getInsertionBlock(),
                                opBuilder.getInsertionPoint());
      for (const auto statement : custom->body) {
        emitStatement(statement, parameters, qubits);
        if (emissionFailed || emissionBudget.isExhausted()) {
          return failure();
        }
      }
      return success();
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
    auto controlValues = qubits.take_front(controls);
    auto targets = qubits.drop_front(controls);
    if (catalog->lowering == GateLowering::CU) {
      if (controls != 1 || parameters.size() != 4 || targets.size() != 1) {
        return failure();
      }
      auto relativePhase = emitOpenQASM3Phase(
          opBuilder, loc, parameters.take_front(3), parameters.back());
      qc::POp::create(opBuilder, loc, controlValues.front(), relativePhase);
      LogicalResult result = success();
      qc::CtrlOp::create(
          opBuilder, loc, controlValues, targets, [&](ValueRange aliases) {
            result = emitPrimitive(opBuilder, loc, GateLowering::U3,
                                   parameters.take_front(3), aliases);
          });
      return result;
    }

    const auto emitCatalogLowering = [&](ValueRange primitiveQubits) {
      const auto emitBody = [&](ValueRange bodyQubits) {
        if (catalog->lowering == GateLowering::BuiltinU ||
            catalog->lowering == GateLowering::U2 ||
            catalog->lowering == GateLowering::U3) {
          Value phase;
          if (catalog->lowering == GateLowering::BuiltinU &&
              !program.openQASM2) {
            phase = emitOpenQASM3Phase(opBuilder, loc, parameters);
          } else {
            phase = emitOpenQASM2UPhase(opBuilder, loc, parameters);
          }
          qc::GPhaseOp::create(opBuilder, loc, phase);
          const auto primitive = catalog->lowering == GateLowering::U2
                                     ? GateLowering::U2
                                     : GateLowering::U3;
          return emitPrimitive(opBuilder, loc, primitive, parameters,
                               bodyQubits);
        }
        return emitPrimitive(opBuilder, loc, catalog->lowering, parameters,
                             bodyQubits);
      };
      if (!catalog->inverse) {
        return emitBody(primitiveQubits);
      }
      LogicalResult result = success();
      qc::InvOp::create(
          opBuilder, loc, primitiveQubits,
          [&](ValueRange aliases) { result = emitBody(aliases); });
      return result;
    };
    if (controls == 0) {
      return emitCatalogLowering(qubits);
    }
    LogicalResult result = success();
    qc::CtrlOp::create(
        opBuilder, loc, controlValues, targets,
        [&](ValueRange aliases) { result = emitCatalogLowering(aliases); });
    return result;
  }

  LogicalResult
  emitModifiers(OpBuilder& opBuilder,
                const frontend::GateApplication& application,
                const Location loc, ValueRange parameters,
                ArrayRef<std::int64_t> controlCounts,
                ArrayRef<std::variant<double, Value>> modifierOperands,
                const std::size_t position, ValueRange qubits) {
    if (position == application.modifiers.size()) {
      return emitResolvedGate(opBuilder, application, loc, parameters, qubits);
    }
    const auto kind = application.modifiers[position].kind;
    if (kind == frontend::ModifierKind::Inv) {
      LogicalResult result = success();
      qc::InvOp::create(opBuilder, loc, qubits, [&](ValueRange aliases) {
        result = emitModifiers(opBuilder, application, loc, parameters,
                               controlCounts, modifierOperands, position + 1,
                               aliases);
      });
      return result;
    }
    if (kind == frontend::ModifierKind::Pow) {
      LogicalResult result = success();
      qc::PowOp::create(opBuilder, loc, modifierOperands[position], qubits,
                        [&](ValueRange aliases) {
                          result = emitModifiers(opBuilder, application, loc,
                                                 parameters, controlCounts,
                                                 modifierOperands, position + 1,
                                                 aliases);
                        });
      return result;
    }
    const auto count = static_cast<std::size_t>(controlCounts[position]);
    LogicalResult result = success();
    qc::CtrlOp::create(opBuilder, loc, qubits.take_front(count),
                       qubits.drop_front(count), [&](ValueRange aliases) {
                         result = emitModifiers(opBuilder, application, loc,
                                                parameters, controlCounts,
                                                modifierOperands, position + 1,
                                                aliases);
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
    SmallVector<std::variant<double, Value>> modifierOperands(
        application.modifiers.size());
    for (const auto [position, modifier] :
         llvm::enumerate(application.modifiers)) {
      if (modifier.kind == frontend::ModifierKind::Pow) {
        const auto& expression = program.expressions.at(*modifier.operand);
        if (expression.kind == frontend::ExpressionKind::Constant) {
          switch (expression.type) {
          case frontend::ScalarType::Int:
            modifierOperands[position] = static_cast<double>(
                std::get<std::int64_t>(expression.constant));
            break;
          case frontend::ScalarType::Uint:
            modifierOperands[position] = static_cast<double>(
                std::get<std::uint64_t>(expression.constant));
            break;
          case frontend::ScalarType::Float:
            modifierOperands[position] = std::get<double>(expression.constant);
            break;
          case frontend::ScalarType::Bool:
            llvm_unreachable("boolean power modifiers fail semantic analysis");
          }
          continue;
        }
        auto exponent =
            emitExpression(opBuilder, *modifier.operand, gateParameters);
        if (isa<IntegerType>(exponent.getType())) {
          exponent = emitExactlyRepresentableIntegerAsF64(
              opBuilder, loc, exponent,
              expression.type == frontend::ScalarType::Uint);
        }
        modifierOperands[position] = exponent;
        continue;
      }
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
          emitError(loc) << "OpenQASM QC emission error: gate control count "
                            "must be a positive constant integer";
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
                            controlCounts, modifierOperands, 0, qubits);
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
            emitError(loc) << "OpenQASM QC emission error: gate '"
                           << application.callee
                           << "' has no lowering to the QC dialect";
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
    if (!emissionBudget.canConstruct(3 * static_cast<std::size_t>(width - 1))) {
      return {};
    }

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
      auto ifOp = scf::IfOp::create(builder, builder.getI1Type(), lhs, true);
      OpBuilder::InsertionGuard guard(builder);
      auto& thenBlock = ifOp.getThenRegion().front();
      if (!thenBlock.empty()) {
        thenBlock.back().erase();
      }
      builder.setInsertionPointToEnd(&thenBlock);
      scf::YieldOp::create(
          builder, emitCondition(condition.rhs, gateParameters, gateQubits));
      auto& elseBlock = ifOp.getElseRegion().front();
      if (!elseBlock.empty()) {
        elseBlock.back().erase();
      }
      builder.setInsertionPointToEnd(&elseBlock);
      scf::YieldOp::create(builder, builder.boolConstant(false));
      return ifOp.getResult(0);
    }
    case frontend::ConditionKind::Or: {
      auto lhs = emitCondition(condition.lhs, gateParameters, gateQubits);
      auto ifOp = scf::IfOp::create(builder, builder.getI1Type(), lhs, true);
      OpBuilder::InsertionGuard guard(builder);
      auto& thenBlock = ifOp.getThenRegion().front();
      if (!thenBlock.empty()) {
        thenBlock.back().erase();
      }
      builder.setInsertionPointToEnd(&thenBlock);
      scf::YieldOp::create(builder, builder.boolConstant(true));
      auto& elseBlock = ifOp.getElseRegion().front();
      if (!elseBlock.empty()) {
        elseBlock.back().erase();
      }
      builder.setInsertionPointToEnd(&elseBlock);
      scf::YieldOp::create(
          builder, emitCondition(condition.rhs, gateParameters, gateQubits));
      return ifOp.getResult(0);
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
    if (emissionFailed || emissionBudget.isExhausted()) {
      return;
    }
    const auto& statement = program.statements.at(id);
    const auto loc = getLocation(statement.location);
    builder.setLoc(loc);
    emissionBudget.setLocation(loc);
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
            emitMeasurement(data, loc, gateQubits);
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
      const auto width = static_cast<std::size_t>(declaration.width);
      if (!emissionBudget.canConstruct(1 + 2 * width)) {
        return;
      }
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
    if (!emissionBudget.canConstruct(3 * static_cast<std::size_t>(width))) {
      return;
    }
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
                       Location loc, ValueRange gateQubits) {
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
        emitError(loc) << "OpenQASM QC emission error: measurement target has "
                          "no classical storage";
        emissionFailed = true;
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

Location getOpenQASMLocation(const frontend::SourceLocation& source,
                             MLIRContext& context) {
  Location location = FileLineColLoc::get(&context, source.filename,
                                          source.line, source.column);
  for (const auto& frame : source.includeStack) {
    auto caller =
        FileLineColLoc::get(&context, frame.filename, frame.line, frame.column);
    location = CallSiteLoc::get(location, caller);
  }
  return location;
}

OwningOpRef<ModuleOp> emitOpenQASMToQC(const frontend::TypedProgram& program,
                                       MLIRContext& context) {
  return OpenQASMToQCEmitter(program, context).emit();
}

} // namespace mlir::qc::detail
