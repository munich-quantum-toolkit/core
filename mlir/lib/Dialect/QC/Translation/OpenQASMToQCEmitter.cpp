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

#include "mlir/Dialect/CBit/IR/CBitAttributes.h"
#include "mlir/Dialect/CBit/IR/CBitOps.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QC/Translation/StandardGate.h"
#include "mlir/Target/OpenQASM/Frontend.h"
#include "mlir/Target/OpenQASM/GateCatalog.h"

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
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
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Support/LLVM.h>

#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
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

    [[nodiscard]] bool canConstruct(const size_t amount) {
      if (exhausted || amount > OPERATION_LIMIT - operationCount) {
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
      if (operationCount > OPERATION_LIMIT) {
        report();
      }
    }

    static constexpr size_t OPERATION_LIMIT = 10'000'000;

  private:
    size_t operationCount = 0;
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
        builder(&context), qubitValues(program.registers.size()),
        classicalRegisters(program.registers.size()),
        scalarValues(program.scalars.size()),
        expressionEmissionCosts(program.expressions.size()),
        bitVectorExpressionEmissionCosts(program.bitVectorExpressions.size()) {
    context
        .loadDialect<qc::QCDialect, arith::ArithDialect, cf::ControlFlowDialect,
                     func::FuncDialect, LLVM::LLVMDialect, math::MathDialect,
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
      if (output.kind == frontend::OutputKind::Scalar) {
        auto value = scalarValues.at(output.symbol);
        if (!value) {
          emitError(getLocation(program.scalars[output.symbol].location))
              << "OpenQASM QC emission error: output scalar '"
              << program.scalars[output.symbol].name << "' has no value";
          return nullptr;
        }
        results.push_back(value);
        continue;
      }
      const auto outputRegister =
          static_cast<frontend::RegisterId>(output.symbol);
      auto reg = classicalRegisters[outputRegister];
      if (!reg) {
        emitError(getLocation(program.registers[outputRegister].location))
            << "OpenQASM QC emission error: output register '"
            << program.registers[outputRegister].name
            << "' has no classical storage";
        return nullptr;
      }
      results.push_back(reg);
    }
    OwningOpRef<ModuleOp> moduleOp;
    if (results.empty()) {
      moduleOp = builder.finalize();
    } else {
      builder.retype(ValueRange(results).getTypes());
      moduleOp = builder.finalize(results);
    }
    if (emissionBudget.isExhausted()) {
      return nullptr;
    }
    return moduleOp;
  }

private:
  // The engine cannot exist without the program and context that outlive it.
  const oq3::frontend::TypedProgram& program;
  MLIRContext& context;
  EmissionBudget emissionBudget;
  qc::QCProgramBuilder builder;
  std::vector<Value> qubitValues;
  std::vector<Value> classicalRegisters;
  std::vector<Value> scalarValues;
  llvm::DenseMap<frontend::ScalarId, Value> provenInductionValues;
  mutable std::vector<std::optional<size_t>> expressionEmissionCosts;
  mutable std::vector<std::optional<size_t>> bitVectorExpressionEmissionCosts;
  DenseMap<const oq3::frontend::GateDefinition*, bool>
      structuredGateCapabilities;
  llvm::StringMap<const oq3::frontend::GateDefinition*> customGateIndex;
  bool emissionFailed = false;

  using StateSlot = frontend::ScalarId;

  [[nodiscard]] Location
  getLocation(const frontend::SourceLocation& source) const {
    return getOpenQASMLocation(source, context);
  }

  static constexpr size_t PROJECTED_EMISSION_LIMIT =
      EmissionBudget::OPERATION_LIMIT;

  [[nodiscard]] static bool
  isExactlyRepresentableAsDouble(const uint64_t magnitude) {
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
          std::get<uint64_t>(expression.constant));
    }
    if (expression.type != frontend::ScalarType::Int) {
      return true;
    }
    const auto value = std::get<int64_t>(expression.constant);
    const auto magnitude = value < 0 ? static_cast<uint64_t>(-(value + 1)) + 1U
                                     : static_cast<uint64_t>(value);
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
          std::holds_alternative<oq3::frontend::IfStatement>(data) ||
          std::holds_alternative<oq3::frontend::SwitchStatement>(data)) {
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
  chargeProjectedEmission(const size_t amount, size_t& projectedEmission,
                          const oq3::frontend::SourceLocation& source) const {
    if (amount > PROJECTED_EMISSION_LIMIT - projectedEmission) {
      return reportProjectedEmissionLimit(source);
    }
    projectedEmission += amount;
    return true;
  }

  [[nodiscard]] bool
  chargeScaledEmission(const size_t amount, const size_t multiplicity,
                       size_t& projectedEmission,
                       const oq3::frontend::SourceLocation& source) const {
    if (multiplicity != 0 && amount > PROJECTED_EMISSION_LIMIT / multiplicity) {
      return reportProjectedEmissionLimit(source);
    }
    return chargeProjectedEmission(amount * multiplicity, projectedEmission,
                                   source);
  }

  [[nodiscard]] size_t
  expressionEmissionCost(const frontend::ExpressionId id) const {
    if (expressionEmissionCosts[id]) {
      return *expressionEmissionCosts[id];
    }
    const auto& expression = program.expressions.at(id);
    const auto add = [](const size_t lhs, const size_t rhs) {
      return lhs > PROJECTED_EMISSION_LIMIT || rhs > PROJECTED_EMISSION_LIMIT ||
                     lhs > PROJECTED_EMISSION_LIMIT - rhs
                 ? PROJECTED_EMISSION_LIMIT + 1
                 : lhs + rhs;
    };
    const auto remember = [&](const size_t cost) {
      expressionEmissionCosts[id] = cost;
      return cost;
    };
    const auto unary = [&](const size_t local) {
      return add(expressionEmissionCost(expression.lhs), local);
    };
    const auto binary = [&](const size_t local) {
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
    case frontend::ExpressionKind::Cast:
      return remember(unary(1));
    case frontend::ExpressionKind::Negate:
      if (expression.type == frontend::ScalarType::Float ||
          expression.type == frontend::ScalarType::Angle) {
        return remember(unary(1));
      }
      return remember(
          unary(expression.type == frontend::ScalarType::Uint ? 2 : 10));
    case frontend::ExpressionKind::ArcCos:
    case frontend::ExpressionKind::ArcSin:
    case frontend::ExpressionKind::ArcTan:
    case frontend::ExpressionKind::Ceiling:
    case frontend::ExpressionKind::Sin:
    case frontend::ExpressionKind::Cos:
    case frontend::ExpressionKind::Floor:
    case frontend::ExpressionKind::Tan:
    case frontend::ExpressionKind::Exp:
    case frontend::ExpressionKind::Log:
    case frontend::ExpressionKind::Sqrt:
      return remember(unary(2));
    case frontend::ExpressionKind::PopCount: {
      const auto& bitVector =
          program.bitVectorExpressions.at(expression.bitVector);
      return remember(add(bitVectorExpressionEmissionCost(expression.bitVector),
                          (4 * static_cast<size_t>(bitVector.width)) + 2));
    }
    case frontend::ExpressionKind::Add:
    case frontend::ExpressionKind::Subtract:
    case frontend::ExpressionKind::Multiply:
      if (expression.type == frontend::ScalarType::Float ||
          expression.type == frontend::ScalarType::Angle) {
        return remember(binary(3));
      }
      return remember(
          binary(expression.type == frontend::ScalarType::Uint ? 2 : 11));
    case frontend::ExpressionKind::Divide:
    case frontend::ExpressionKind::Modulo:
      if (expression.type == frontend::ScalarType::Float ||
          expression.type == frontend::ScalarType::Angle) {
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

  Value emitProvenIndexExpression(OpBuilder& opBuilder,
                                  const frontend::ExpressionId id) {
    const auto& expression = program.expressions.at(id);
    auto loc = opBuilder.getInsertionPoint() == opBuilder.getBlock()->end()
                   ? opBuilder.getUnknownLoc()
                   : opBuilder.getInsertionPoint()->getLoc();
    switch (expression.kind) {
    case frontend::ExpressionKind::Constant: {
      const auto value =
          expression.type == frontend::ScalarType::Uint
              ? static_cast<int64_t>(std::get<uint64_t>(expression.constant))
              : std::get<int64_t>(expression.constant);
      return arith::ConstantIndexOp::create(opBuilder, loc, value);
    }
    case frontend::ExpressionKind::Variable: {
      const auto induction = provenInductionValues.find(expression.variable);
      if (induction == provenInductionValues.end()) {
        llvm_unreachable("proven index refers to an inactive induction");
      }
      return induction->second;
    }
    case frontend::ExpressionKind::Cast:
      return emitProvenIndexExpression(opBuilder, expression.lhs);
    case frontend::ExpressionKind::Negate: {
      auto zero = arith::ConstantIndexOp::create(opBuilder, loc, 0);
      auto operand = emitProvenIndexExpression(opBuilder, expression.lhs);
      return arith::SubIOp::create(opBuilder, loc, zero, operand);
    }
    case frontend::ExpressionKind::Add:
    case frontend::ExpressionKind::Subtract:
    case frontend::ExpressionKind::Multiply: {
      auto lhs = emitProvenIndexExpression(opBuilder, expression.lhs);
      auto rhs = emitProvenIndexExpression(opBuilder, expression.rhs);
      switch (expression.kind) {
      case frontend::ExpressionKind::Add:
        return arith::AddIOp::create(opBuilder, loc, lhs, rhs);
      case frontend::ExpressionKind::Subtract:
        return arith::SubIOp::create(opBuilder, loc, lhs, rhs);
      case frontend::ExpressionKind::Multiply:
        return arith::MulIOp::create(opBuilder, loc, lhs, rhs);
      default:
        llvm_unreachable("not a proven affine binary expression");
      }
    }
    default:
      llvm_unreachable("semantic analysis produced a non-affine expression");
    }
  }

  [[nodiscard]] size_t bitVectorExpressionEmissionCost(
      const frontend::BitVectorExpressionId id) const {
    if (bitVectorExpressionEmissionCosts[id]) {
      return *bitVectorExpressionEmissionCosts[id];
    }
    const auto& expression = program.bitVectorExpressions.at(id);
    const auto remember = [&](const size_t cost) {
      bitVectorExpressionEmissionCosts[id] = cost;
      return cost;
    };
    if (expression.kind == frontend::BitVectorExpressionKind::Register) {
      const auto width = static_cast<size_t>(expression.width);
      return remember(width > PROJECTED_EMISSION_LIMIT / 2
                          ? PROJECTED_EMISSION_LIMIT + 1
                          : 2 * width);
    }
    const auto operand = bitVectorExpressionEmissionCost(expression.operand);
    if (program.expressions.at(expression.distance).kind ==
        frontend::ExpressionKind::Constant) {
      return remember(operand > PROJECTED_EMISSION_LIMIT - 2
                          ? PROJECTED_EMISSION_LIMIT + 1
                          : operand + 2);
    }
    const auto width = static_cast<size_t>(expression.width);
    const auto local = (8 * width) + 8;
    const auto distance = expressionEmissionCost(expression.distance);
    if (operand > PROJECTED_EMISSION_LIMIT ||
        distance > PROJECTED_EMISSION_LIMIT ||
        operand > PROJECTED_EMISSION_LIMIT - distance ||
        operand + distance > PROJECTED_EMISSION_LIMIT - local) {
      return remember(PROJECTED_EMISSION_LIMIT + 1);
    }
    return remember(operand + distance + local);
  }

  [[nodiscard]] bool
  chargeExpressionEmission(const frontend::ExpressionId id,
                           const size_t multiplicity, size_t& projectedEmission,
                           const frontend::SourceLocation& source) const {
    return chargeScaledEmission(expressionEmissionCost(id), multiplicity,
                                projectedEmission, source);
  }

  [[nodiscard]] bool
  chargeDynamicBitRead(const frontend::BitReference& reference,
                       const size_t multiplicity, size_t& projectedEmission,
                       const frontend::SourceLocation& source) const {
    if (!reference.dynamicIndex) {
      return chargeScaledEmission(2, multiplicity, projectedEmission, source);
    }
    return chargeExpressionEmission(*reference.dynamicIndex, multiplicity,
                                    projectedEmission, source) &&
           chargeScaledEmission(11, multiplicity, projectedEmission, source);
  }

  [[nodiscard]] bool
  chargeQubitAccesses(const ArrayRef<frontend::QubitReference> references,
                      const size_t multiplicity, size_t& projectedEmission,
                      const oq3::frontend::SourceLocation& source) const {
    for (const auto& reference : references) {
      if (reference.kind != frontend::QubitReferenceKind::Register ||
          program.registers.at(reference.symbol).isScalar) {
        continue;
      }

      if (reference.provenIndex &&
          !chargeExpressionEmission(*reference.provenIndex, multiplicity,
                                    projectedEmission, source)) {
        return false;
      }
      /// Each access emits an index value and a load. Semantic analysis has
      /// already proved a nonconstant index's bounds and uniqueness.
      if (!chargeScaledEmission(2, multiplicity, projectedEmission, source)) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] size_t
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
      uint64_t controls = 1;
      if (modifier.operand) {
        const auto& expression = program.expressions.at(*modifier.operand);
        controls =
            expression.type == frontend::ScalarType::Uint
                ? std::get<uint64_t>(expression.constant)
                : static_cast<uint64_t>(std::get<int64_t>(expression.constant));
      }
      if (controls > PROJECTED_EMISSION_LIMIT / 2) {
        return PROJECTED_EMISSION_LIMIT + 1;
      }
      cost += static_cast<size_t>(2 * controls);
    }
    return cost;
  }

  [[nodiscard]] bool
  chargeConditionEmission(const frontend::ConditionId id,
                          const size_t multiplicity, size_t& projectedEmission,
                          const oq3::frontend::SourceLocation& source) const {
    const auto& condition = program.conditions.at(id);
    if (condition.kind == frontend::ConditionKind::Measurement) {
      return chargeQubitAccesses({condition.measurement}, multiplicity,
                                 projectedEmission, source) &&
             chargeScaledEmission(1, multiplicity, projectedEmission, source);
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
                      size_t& projectedEmission,
                      const size_t multiplicity = 1) {
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
          const size_t localCost = loop->provenPositiveRange ? 7 : 16;
          if (!chargeScaledEmission(localCost, multiplicity, projectedEmission,
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
        } else if (const auto* switchStatement =
                       std::get_if<oq3::frontend::SwitchStatement>(
                           &statement.data)) {
          if (!chargeExpressionEmission(switchStatement->control, multiplicity,
                                        projectedEmission,
                                        statement.location) ||
              !chargeScaledEmission(3, multiplicity, projectedEmission,
                                    statement.location)) {
            return false;
          }
          for (const auto& switchCase : switchStatement->cases) {
            const auto labelCount = switchCase.labels.size();
            if (!chargeScaledEmission(labelCount, multiplicity,
                                      projectedEmission, statement.location)) {
              return false;
            }
            if (labelCount != 0 &&
                multiplicity > PROJECTED_EMISSION_LIMIT / labelCount) {
              return reportProjectedEmissionLimit(statement.location);
            }
            const auto caseMultiplicity = multiplicity * labelCount;
            if (!preflightStatements(switchCase.body, projectedEmission,
                                     caseMultiplicity)) {
              return false;
            }
          }
          if (!preflightStatements(switchStatement->defaultStatements,
                                   projectedEmission, multiplicity)) {
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
                                       projectedEmission, statement.location) ||
              (!assignment->target.dynamicIndex &&
               !chargeScaledEmission(2, multiplicity, projectedEmission,
                                     statement.location))) {
            return false;
          }
          if (assignment->target.dynamicIndex) {
            const auto width = static_cast<size_t>(
                program.registers.at(assignment->target.reg).width);
            if (!chargeExpressionEmission(*assignment->target.dynamicIndex,
                                          multiplicity, projectedEmission,
                                          statement.location) ||
                !chargeScaledEmission(9 + (3 * width), multiplicity,
                                      projectedEmission, statement.location)) {
              return false;
            }
          }
        } else if (const auto* assignment =
                       std::get_if<frontend::BitVectorAssignmentStatement>(
                           &statement.data)) {
          if (!chargeScaledEmission(
                  bitVectorExpressionEmissionCost(assignment->value),
                  multiplicity, projectedEmission, statement.location) ||
              !chargeScaledEmission(
                  2 * static_cast<size_t>(
                          program.registers.at(assignment->target).width),
                  multiplicity, projectedEmission, statement.location)) {
            return false;
          }
        } else if (const auto* declaration =
                       std::get_if<frontend::DeclarationStatement>(
                           &statement.data)) {
          if (!chargeScaledEmission(1, multiplicity, projectedEmission,
                                    statement.location)) {
            return false;
          }
        } else if (const auto* measurement =
                       std::get_if<frontend::MeasurementStatement>(
                           &statement.data)) {
          for (const auto& qubit : measurement->qubits) {
            if (!chargeQubitAccesses({qubit}, multiplicity, projectedEmission,
                                     statement.location) ||
                !chargeScaledEmission(1, multiplicity, projectedEmission,
                                      statement.location)) {
              return false;
            }
          }
          for (const auto& target : measurement->targets) {
            if (!target.dynamicIndex &&
                !chargeScaledEmission(2, multiplicity, projectedEmission,
                                      statement.location)) {
              return false;
            }
            if (!target.dynamicIndex) {
              continue;
            }
            const auto width =
                static_cast<size_t>(program.registers.at(target.reg).width);
            if (!chargeExpressionEmission(*target.dynamicIndex, multiplicity,
                                          projectedEmission,
                                          statement.location) ||
                !chargeScaledEmission(9 + (3 * width), multiplicity,
                                      projectedEmission, statement.location)) {
              return false;
            }
          }
        } else if (const auto* reset =
                       std::get_if<frontend::ResetStatement>(&statement.data)) {
          for (const auto& qubit : reset->qubits) {
            if (!chargeQubitAccesses({qubit}, multiplicity, projectedEmission,
                                     statement.location) ||
                !chargeScaledEmission(1, multiplicity, projectedEmission,
                                      statement.location)) {
              return false;
            }
          }
        } else if (const auto* barrier =
                       std::get_if<frontend::BarrierStatement>(
                           &statement.data)) {
          if (!chargeQubitAccesses(barrier->qubits, multiplicity,
                                   projectedEmission, statement.location) ||
              !chargeScaledEmission(1, multiplicity, projectedEmission,
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
      if (!chargeQubitAccesses(application->qubits, multiplicity,
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
        if (!chargeScaledEmission(leafCost, multiplicity, projectedEmission,
                                  statement.location)) {
          return false;
        }
        continue;
      }
      if (!chargeScaledEmission(modifierEmissionCost(*application),
                                multiplicity, projectedEmission,
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
      if (!preflightStatements(gate->body, projectedEmission, multiplicity)) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool preflight() {
    const bool hasDeclaredQubits =
        llvm::any_of(program.registers, [](const auto& declaration) {
          return declaration.kind == frontend::RegisterKind::Qubit;
        });
    if (hasDeclaredQubits) {
      for (const auto& condition : program.conditions) {
        if (condition.kind == frontend::ConditionKind::Measurement &&
            condition.measurement.kind ==
                frontend::QubitReferenceKind::Hardware) {
          emitError(getLocation(condition.location))
              << "OpenQASM QC emission error: mixing physical and declared "
                 "qubits is not supported by the QC target";
          return false;
        }
      }
      for (const auto& statement : program.statements) {
        const auto containsHardwareQubit = std::visit(
            [](const auto& data) {
              using T = std::decay_t<decltype(data)>;
              if constexpr (std::is_same_v<T, frontend::GateApplication> ||
                            std::is_same_v<T, frontend::MeasurementStatement> ||
                            std::is_same_v<T, frontend::ResetStatement> ||
                            std::is_same_v<T, frontend::BarrierStatement>) {
                return llvm::any_of(data.qubits, [](const auto& qubit) {
                  return qubit.kind == frontend::QubitReferenceKind::Hardware;
                });
              }
              return false;
            },
            statement.data);
        if (containsHardwareQubit) {
          emitError(getLocation(statement.location))
              << "OpenQASM QC emission error: mixing physical and declared "
                 "qubits is not supported by the QC target";
          return false;
        }
      }
    }
    size_t projectedEmission = program.registers.size() + 4;
    return preflightStatements(program.body, projectedEmission);
  }

  [[nodiscard]] static Value checkedSignedResult(OpBuilder& opBuilder,
                                                 Location loc, Value wide,
                                                 const StringRef message) {
    auto i128 = opBuilder.getIntegerType(128);
    auto minimum = arith::ConstantIntOp::create(
        opBuilder, loc, i128, std::numeric_limits<int64_t>::min());
    auto maximum = arith::ConstantIntOp::create(
        opBuilder, loc, i128, std::numeric_limits<int64_t>::max());
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

  [[nodiscard]] static Value conditionalIntegerMultiply(OpBuilder& opBuilder,
                                                        Location loc,
                                                        Value condition,
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
        opBuilder, loc, i128, std::numeric_limits<int64_t>::min());
    auto maximum = arith::ConstantIntOp::create(
        opBuilder, loc, i128, std::numeric_limits<int64_t>::max());
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

  [[nodiscard]] static Value emitIntegerPower(OpBuilder& opBuilder,
                                              Location loc, Value base,
                                              Value exponent,
                                              const bool resultIsUnsigned,
                                              const bool exponentIsUnsigned) {
    auto zero = arith::ConstantIntOp::create(opBuilder, loc, 0, 64);
    auto one = arith::ConstantIntOp::create(opBuilder, loc, 1, 64);
    if (!exponentIsUnsigned) {
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
          auto nextResult =
              conditionalIntegerMultiply(nested, nestedLoc, odd, arguments[0],
                                         arguments[1], resultIsUnsigned);
          auto nextExponent =
              arith::ShRUIOp::create(nested, nestedLoc, arguments[2], one);
          auto squareBase = arith::CmpIOp::create(
              nested, nestedLoc, arith::CmpIPredicate::ne, nextExponent, zero);
          auto nextBase = conditionalIntegerMultiply(
              nested, nestedLoc, squareBase, arguments[1], arguments[1],
              resultIsUnsigned);
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
                         APInt(64, (uint64_t{1} << 53U) - 1U)));
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

  [[nodiscard]] static Value packBits(OpBuilder& opBuilder,
                                      ArrayRef<Value> bits) {
    assert(!bits.empty());
    const auto width = static_cast<unsigned>(bits.size());
    auto packedType = opBuilder.getIntegerType(width);
    auto loc = UnknownLoc::get(opBuilder.getContext());
    Value packed =
        width == 1
            ? bits.front()
            : arith::ExtUIOp::create(opBuilder, loc, packedType, bits.front());
    for (unsigned bit = 1; bit < width; ++bit) {
      auto extended =
          arith::ExtUIOp::create(opBuilder, loc, packedType, bits[bit]);
      auto shift = arith::ConstantIntOp::create(opBuilder, loc, bit, width);
      auto shifted = arith::ShLIOp::create(opBuilder, loc, extended, shift);
      packed = arith::OrIOp::create(opBuilder, loc, packed, shifted);
    }
    return packed;
  }

  [[nodiscard]] static SmallVector<Value>
  unpackBits(OpBuilder& opBuilder, Value packed, const uint64_t width) {
    SmallVector<Value> bits;
    bits.reserve(width);
    if (width == 1) {
      bits.push_back(packed);
      return bits;
    }
    auto loc = UnknownLoc::get(opBuilder.getContext());
    for (uint64_t bit = 0; bit < width; ++bit) {
      Value selected = packed;
      if (bit != 0) {
        auto shift = arith::ConstantIntOp::create(opBuilder, loc,
                                                  static_cast<int64_t>(bit),
                                                  static_cast<unsigned>(width));
        selected = arith::ShRUIOp::create(opBuilder, loc, packed, shift);
      }
      bits.push_back(arith::TruncIOp::create(opBuilder, loc,
                                             opBuilder.getI1Type(), selected));
    }
    return bits;
  }

  struct EmittedBitVector {
    uint64_t width = 0;
    SmallVector<Value> bits;
    Value packed;
  };

  static Value ensurePacked(OpBuilder& opBuilder, EmittedBitVector& value) {
    if (!value.packed) {
      value.packed = packBits(opBuilder, value.bits);
    }
    return value.packed;
  }

  static ArrayRef<Value> ensureBits(OpBuilder& opBuilder,
                                    EmittedBitVector& value) {
    if (value.bits.empty()) {
      value.bits = unpackBits(opBuilder, value.packed, value.width);
    }
    return value.bits;
  }

  [[nodiscard]] EmittedBitVector
  emitBitVectorExpression(OpBuilder& opBuilder,
                          const frontend::BitVectorExpressionId id) {
    const auto& expression = program.bitVectorExpressions.at(id);
    auto loc = UnknownLoc::get(opBuilder.getContext());
    if (expression.kind == frontend::BitVectorExpressionKind::Register) {
      SmallVector<Value> bits;
      bits.reserve(expression.width);
      auto reg = classicalRegisters.at(expression.reg);
      assert(reg && "semantic analysis must declare bit registers before use");
      for (uint64_t bit = 0; bit < expression.width; ++bit) {
        auto index = arith::ConstantIndexOp::create(opBuilder, loc,
                                                    static_cast<int64_t>(bit));
        bits.push_back(cbit::LoadOp::create(opBuilder, loc,
                                            opBuilder.getI1Type(), reg, index));
      }
      return {.width = expression.width, .bits = std::move(bits)};
    }
    auto operand = emitBitVectorExpression(opBuilder, expression.operand);
    const auto& distanceExpression =
        program.expressions.at(expression.distance);
    if (distanceExpression.kind == frontend::ExpressionKind::Constant) {
      const auto distance = std::get<int64_t>(distanceExpression.constant);
      const auto width = static_cast<int64_t>(expression.width);
      auto normalized = distance % width;
      if (normalized < 0) {
        normalized += width;
      }
      if (operand.bits.empty()) {
        if (normalized == 0) {
          return operand;
        }
        auto shift = arith::ConstantIntOp::create(
            opBuilder, loc, normalized,
            static_cast<unsigned>(expression.width));
        Value rotated =
            expression.kind == frontend::BitVectorExpressionKind::RotateLeft
                ? LLVM::FshlOp::create(opBuilder, loc, operand.packed,
                                       operand.packed, shift)
                      .getResult()
                : LLVM::FshrOp::create(opBuilder, loc, operand.packed,
                                       operand.packed, shift)
                      .getResult();
        return {.width = expression.width, .packed = rotated};
      }
      const auto bits = ensureBits(opBuilder, operand);
      SmallVector<Value> rotated(expression.width);
      for (uint64_t bit = 0; bit < expression.width; ++bit) {
        const auto source =
            expression.kind == frontend::BitVectorExpressionKind::RotateLeft
                ? (bit + expression.width - static_cast<uint64_t>(normalized)) %
                      expression.width
                : (bit + static_cast<uint64_t>(normalized)) % expression.width;
        rotated[bit] = bits[source];
      }
      return {.width = expression.width, .bits = std::move(rotated)};
    }

    auto distance = emitExpression(opBuilder, expression.distance, {});
    auto widthConstant = arith::ConstantIntOp::create(
        opBuilder, loc, static_cast<int64_t>(expression.width), 64);
    auto remainder =
        arith::RemSIOp::create(opBuilder, loc, distance, widthConstant);
    auto positive =
        arith::AddIOp::create(opBuilder, loc, remainder, widthConstant);
    auto normalized =
        arith::RemSIOp::create(opBuilder, loc, positive, widthConstant);
    auto packed = ensurePacked(opBuilder, operand);
    auto packedType =
        opBuilder.getIntegerType(static_cast<unsigned>(expression.width));
    Value shift = normalized;
    if (expression.width < 64) {
      shift = arith::TruncIOp::create(opBuilder, loc, packedType, normalized);
    } else if (expression.width > 64) {
      shift = arith::ExtUIOp::create(opBuilder, loc, packedType, normalized);
    }
    Value rotated =
        expression.kind == frontend::BitVectorExpressionKind::RotateLeft
            ? LLVM::FshlOp::create(opBuilder, loc, packed, packed, shift)
                  .getResult()
            : LLVM::FshrOp::create(opBuilder, loc, packed, packed, shift)
                  .getResult();
    return {.width = expression.width, .packed = rotated};
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
            opBuilder, loc,
            static_cast<int64_t>(std::get<bool>(expression.constant)), 1);
      case frontend::ScalarType::Int:
        return arith::ConstantIntOp::create(
            opBuilder, loc, std::get<int64_t>(expression.constant), 64);
      case frontend::ScalarType::Uint:
        return arith::ConstantOp::create(
            opBuilder, loc,
            IntegerAttr::get(opBuilder.getI64Type(),
                             APInt(64, std::get<uint64_t>(expression.constant),
                                   /*isSigned=*/false)));
      case frontend::ScalarType::Float:
      case frontend::ScalarType::Angle:
        return arith::ConstantFloatOp::create(
            opBuilder, loc, opBuilder.getF64Type(),
            APFloat(std::get<double>(expression.constant)));
      }
      llvm_unreachable("unknown scalar type");
    case frontend::ExpressionKind::GateParameter:
      return gateParameters[expression.parameter];
    case frontend::ExpressionKind::Variable:
      return scalarValues.at(expression.variable);
    case frontend::ExpressionKind::Cast: {
      auto operand = emitExpression(opBuilder, expression.lhs, gateParameters);
      return emitScalarCast(opBuilder, loc, operand,
                            program.expressions.at(expression.lhs).type,
                            expression.type);
    }
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
    case frontend::ExpressionKind::Ceiling:
    case frontend::ExpressionKind::Sin:
    case frontend::ExpressionKind::Cos:
    case frontend::ExpressionKind::Floor:
    case frontend::ExpressionKind::Tan:
    case frontend::ExpressionKind::Exp:
    case frontend::ExpressionKind::Log:
    case frontend::ExpressionKind::Sqrt: {
      Value operand = emitExpression(opBuilder, expression.lhs, gateParameters);
      assert(isa<FloatType>(operand.getType()) &&
             "semantic analysis must normalize math operands");
      switch (expression.kind) {
      case frontend::ExpressionKind::ArcCos:
        return math::AcosOp::create(opBuilder, loc, operand);
      case frontend::ExpressionKind::ArcSin:
        return math::AsinOp::create(opBuilder, loc, operand);
      case frontend::ExpressionKind::ArcTan:
        return math::AtanOp::create(opBuilder, loc, operand);
      case frontend::ExpressionKind::Ceiling:
        return math::CeilOp::create(opBuilder, loc, operand);
      case frontend::ExpressionKind::Sin:
        return math::SinOp::create(opBuilder, loc, operand);
      case frontend::ExpressionKind::Cos:
        return math::CosOp::create(opBuilder, loc, operand);
      case frontend::ExpressionKind::Floor:
        return math::FloorOp::create(opBuilder, loc, operand);
      case frontend::ExpressionKind::Tan:
        return math::TanOp::create(opBuilder, loc, operand);
      case frontend::ExpressionKind::Exp:
        return math::ExpOp::create(opBuilder, loc, operand);
      case frontend::ExpressionKind::Log:
        return math::LogOp::create(opBuilder, loc, operand);
      case frontend::ExpressionKind::Sqrt:
        return math::SqrtOp::create(opBuilder, loc, operand);
      default:
        llvm_unreachable("unknown scalar math function");
      }
    }
    case frontend::ExpressionKind::PopCount: {
      const auto& bitVector =
          program.bitVectorExpressions.at(expression.bitVector);
      auto value = emitBitVectorExpression(opBuilder, expression.bitVector);
      auto packed = ensurePacked(opBuilder, value);
      auto count = math::CtPopOp::create(opBuilder, loc, packed);
      if (bitVector.width < 64) {
        return arith::ExtUIOp::create(opBuilder, loc, opBuilder.getI64Type(),
                                      count);
      }
      if (bitVector.width > 64) {
        return arith::TruncIOp::create(opBuilder, loc, opBuilder.getI64Type(),
                                       count);
      }
      return count;
    }
    case frontend::ExpressionKind::Add:
    case frontend::ExpressionKind::Subtract:
    case frontend::ExpressionKind::Multiply:
    case frontend::ExpressionKind::Divide:
    case frontend::ExpressionKind::Modulo:
    case frontend::ExpressionKind::Power: {
      auto lhs = emitExpression(opBuilder, expression.lhs, gateParameters);
      auto rhs = emitExpression(opBuilder, expression.rhs, gateParameters);
      if (expression.type != frontend::ScalarType::Float &&
          expression.type != frontend::ScalarType::Angle) {
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
                opBuilder, loc, std::numeric_limits<int64_t>::min(), 64);
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
          return emitIntegerPower(opBuilder, loc, lhs, rhs, isUnsigned,
                                  program.expressions.at(expression.rhs).type ==
                                      frontend::ScalarType::Uint);
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
      switch (expression.kind) {
      case frontend::ExpressionKind::Add:
        return arith::AddFOp::create(opBuilder, loc, lhs, rhs);
      case frontend::ExpressionKind::Subtract:
        return arith::SubFOp::create(opBuilder, loc, lhs, rhs);
      case frontend::ExpressionKind::Multiply:
        return arith::MulFOp::create(opBuilder, loc, lhs, rhs);
      case frontend::ExpressionKind::Divide:
        return arith::DivFOp::create(opBuilder, loc, lhs, rhs);
      case frontend::ExpressionKind::Modulo:
        return arith::RemFOp::create(opBuilder, loc, lhs, rhs);
      case frontend::ExpressionKind::Power:
        return math::PowFOp::create(opBuilder, loc, lhs, rhs);
      default:
        llvm_unreachable("not a floating-point binary expression");
      }
    }
    }
    llvm_unreachable("unknown scalar expression kind");
  }

  [[nodiscard]] Value emitCheckedIndex(const frontend::ExpressionId expression,
                                       const int64_t width,
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
                     ValueRange gateQubits, Value index = {}) {
    switch (reference.kind) {
    case frontend::QubitReferenceKind::Register: {
      const auto& declaration = program.registers.at(reference.symbol);
      if (declaration.isScalar) {
        return qubitValues.at(reference.symbol);
      }
      assert(index && "register qubit references require an index");
      return builder.loadQubit(qubitValues.at(reference.symbol), index);
    }
    case frontend::QubitReferenceKind::GateArgument:
      return gateQubits[reference.symbol];
    case frontend::QubitReferenceKind::Hardware:
      return builder.staticQubit(reference.index);
    }
    llvm_unreachable("unknown qubit reference kind");
  }

  [[nodiscard]] SmallVector<Value>
  emitQubitIndices(ArrayRef<frontend::QubitReference> references) {
    SmallVector<Value> indices(references.size());
    for (const auto [position, reference] : llvm::enumerate(references)) {
      if (reference.kind != frontend::QubitReferenceKind::Register ||
          program.registers.at(reference.symbol).isScalar) {
        continue;
      }
      if (!reference.provenIndex) {
        indices[position] = arith::ConstantIndexOp::create(
            builder, static_cast<int64_t>(reference.index));
        continue;
      }
      indices[position] =
          emitProvenIndexExpression(builder, *reference.provenIndex);
    }
    return indices;
  }

  [[nodiscard]] SmallVector<Value>
  resolveQubits(ArrayRef<frontend::QubitReference> references,
                ValueRange gateQubits, ValueRange indices) {
    SmallVector<Value> resolved;
    resolved.reserve(references.size());
    for (const auto [position, reference] : llvm::enumerate(references)) {
      resolved.push_back(
          resolveQubit(reference, gateQubits, indices[position]));
    }
    return resolved;
  }

  [[nodiscard]] Value
  emitQubitOperation(const frontend::QubitReference& reference,
                     ValueRange gateQubits,
                     llvm::function_ref<Value(Value)> emitResolvedOperation) {
    const auto indices = emitQubitIndices({reference});
    return emitResolvedOperation(
        resolveQubit(reference, gateQubits, indices.front()));
  }

  static LogicalResult emitPrimitive(OpBuilder& opBuilder, const Location loc,
                                     const GateLowering lowering,
                                     ValueRange parameters, ValueRange qubits) {
    return qc::emitStandardGate(opBuilder, loc, lowering, parameters, qubits);
  }

  static Value
  emitOpenQASM3Phase(OpBuilder& opBuilder, const Location loc,
                     ValueRange uParameters,
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
                                   ValueRange uParameters) {
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
    const size_t controls = catalog->variadicControls
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
                ArrayRef<int64_t> controlCounts,
                ArrayRef<std::variant<double, Value>> modifierOperands,
                const size_t position, ValueRange qubits) {
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
    const auto count = static_cast<size_t>(controlCounts[position]);
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
    const auto qubitIndices = emitQubitIndices(application.qubits);
    SmallVector<int64_t> controlCounts(application.modifiers.size(), 0);
    SmallVector<std::variant<double, Value>> modifierOperands(
        application.modifiers.size());
    for (const auto [position, modifier] :
         llvm::enumerate(application.modifiers)) {
      if (modifier.kind == frontend::ModifierKind::Pow) {
        const auto& expression = program.expressions.at(*modifier.operand);
        if (expression.kind == frontend::ExpressionKind::Constant) {
          switch (expression.type) {
          case frontend::ScalarType::Int:
            modifierOperands[position] =
                static_cast<double>(std::get<int64_t>(expression.constant));
            break;
          case frontend::ScalarType::Uint:
            modifierOperands[position] =
                static_cast<double>(std::get<uint64_t>(expression.constant));
            break;
          case frontend::ScalarType::Float:
            modifierOperands[position] = std::get<double>(expression.constant);
            break;
          case frontend::ScalarType::Angle:
          case frontend::ScalarType::Bool:
            llvm_unreachable(
                "boolean and angle power modifiers fail semantic analysis");
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
      int64_t count = 1;
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

    const auto qubits =
        resolveQubits(application.qubits, gateQubits, qubitIndices);
    size_t negativeOffset = 0;
    for (const auto [position, modifier] :
         llvm::enumerate(application.modifiers)) {
      if (modifier.kind == frontend::ModifierKind::Ctrl ||
          modifier.kind == frontend::ModifierKind::NegCtrl) {
        if (modifier.kind == frontend::ModifierKind::NegCtrl) {
          for (auto control : ValueRange(qubits).slice(
                   negativeOffset, controlCounts[position])) {
            qc::XOp::create(opBuilder, loc, control);
          }
        }
        negativeOffset += static_cast<size_t>(controlCounts[position]);
      }
    }
    const auto result =
        emitModifiers(opBuilder, application, loc, parameters, controlCounts,
                      modifierOperands, 0, qubits);
    negativeOffset = 0;
    for (const auto [position, modifier] :
         llvm::enumerate(application.modifiers)) {
      if (modifier.kind == frontend::ModifierKind::Ctrl ||
          modifier.kind == frontend::ModifierKind::NegCtrl) {
        if (modifier.kind == frontend::ModifierKind::NegCtrl) {
          for (auto control : ValueRange(qubits).slice(
                   negativeOffset, controlCounts[position])) {
            qc::XOp::create(opBuilder, loc, control);
          }
        }
        negativeOffset += static_cast<size_t>(controlCounts[position]);
      }
    }
    if (failed(result)) {
      emissionFailed = true;
      emitError(loc) << "OpenQASM QC emission error: gate '"
                     << application.callee
                     << "' has no lowering to the QC dialect";
    }
  }

  [[nodiscard]] static Value emitScalarCast(OpBuilder& opBuilder,
                                            const Location loc, Value value,
                                            const frontend::ScalarType source,
                                            const frontend::ScalarType target) {
    if (source == target ||
        (source == frontend::ScalarType::Int &&
         target == frontend::ScalarType::Uint) ||
        (source == frontend::ScalarType::Uint &&
         target == frontend::ScalarType::Int) ||
        ((source == frontend::ScalarType::Float ||
          source == frontend::ScalarType::Angle) &&
         (target == frontend::ScalarType::Float ||
          target == frontend::ScalarType::Angle))) {
      return value;
    }
    if (target == frontend::ScalarType::Float ||
        target == frontend::ScalarType::Angle) {
      if (source == frontend::ScalarType::Bool ||
          source == frontend::ScalarType::Uint) {
        return arith::UIToFPOp::create(opBuilder, loc, opBuilder.getF64Type(),
                                       value);
      }
      return arith::SIToFPOp::create(opBuilder, loc, opBuilder.getF64Type(),
                                     value);
    }
    if (source == frontend::ScalarType::Bool) {
      return arith::ExtUIOp::create(opBuilder, loc, opBuilder.getI64Type(),
                                    value);
    }
    if ((source == frontend::ScalarType::Float ||
         source == frontend::ScalarType::Angle) &&
        target == frontend::ScalarType::Uint) {
      return arith::FPToUIOp::create(opBuilder, loc, opBuilder.getI64Type(),
                                     value);
    }
    if (source == frontend::ScalarType::Float ||
        source == frontend::ScalarType::Angle) {
      return arith::FPToSIOp::create(opBuilder, loc, opBuilder.getI64Type(),
                                     value);
    }
    llvm_unreachable("unsupported standard scalar conversion");
  }

  [[nodiscard]] Value readBit(const frontend::BitReference& reference) {
    auto reg = classicalRegisters.at(reference.reg);
    assert(reg && "semantic analysis must declare bit registers before use");
    if (!reference.dynamicIndex) {
      return builder.loadClassicalBit(reg,
                                      static_cast<int64_t>(reference.index));
    }

    const auto width =
        static_cast<int64_t>(program.registers.at(reference.reg).width);
    auto index = emitCheckedIndex(*reference.dynamicIndex, width,
                                  "dynamic classical index out of bounds");
    auto registerIndex =
        arith::IndexCastOp::create(builder, builder.getIndexType(), index);
    return builder.loadClassicalBit(reg, registerIndex.getResult());
  }

  [[nodiscard]] Value
  emitComparison(const frontend::ConditionExpression& condition,
                 ValueRange gateParameters) {
    auto lhs = emitExpression(builder, condition.comparisonLhs, gateParameters);
    auto rhs = emitExpression(builder, condition.comparisonRhs, gateParameters);
    const auto lhsType = program.expressions.at(condition.comparisonLhs).type;
    const auto rhsType = program.expressions.at(condition.comparisonRhs).type;
    assert(lhsType == rhsType &&
           "semantic analysis must normalize comparison operands");
    if (lhsType == frontend::ScalarType::Float ||
        lhsType == frontend::ScalarType::Angle) {
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

    const bool isUnsigned = lhsType == frontend::ScalarType::Uint;
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

  static void recordMutation(const StateSlot slot,
                             llvm::DenseSet<StateSlot>& mutationKeys,
                             SmallVectorImpl<StateSlot>& mutations) {
    if (mutationKeys.insert(slot).second) {
      mutations.push_back(slot);
    }
  }

  void collectMutations(const frontend::StatementId id,
                        llvm::DenseSet<StateSlot>& mutationKeys,
                        SmallVectorImpl<StateSlot>& mutations) const {
    const auto& statement = program.statements.at(id);
    std::visit(
        [&](const auto& data) {
          using T = std::decay_t<decltype(data)>;
          if constexpr (std::is_same_v<T,
                                       frontend::ScalarDeclarationStatement> ||
                        std::is_same_v<T,
                                       frontend::ScalarAssignmentStatement>) {
            recordMutation(data.scalar, mutationKeys, mutations);
          } else if constexpr (std::is_same_v<T, frontend::IfStatement>) {
            for (const auto nested : data.thenStatements) {
              collectMutations(nested, mutationKeys, mutations);
            }
            for (const auto nested : data.elseStatements) {
              collectMutations(nested, mutationKeys, mutations);
            }
          } else if constexpr (std::is_same_v<T, frontend::ForStatement> ||
                               std::is_same_v<T, frontend::WhileStatement>) {
            for (const auto nested : data.body) {
              collectMutations(nested, mutationKeys, mutations);
            }
          } else if constexpr (std::is_same_v<T, frontend::SwitchStatement>) {
            for (const auto& switchCase : data.cases) {
              for (const auto nested : switchCase.body) {
                collectMutations(nested, mutationKeys, mutations);
              }
            }
            for (const auto nested : data.defaultStatements) {
              collectMutations(nested, mutationKeys, mutations);
            }
          }
        },
        statement.data);
  }

  [[nodiscard]] SmallVector<StateSlot>
  mutatedState(ArrayRef<frontend::StatementId> statements) const {
    llvm::DenseSet<StateSlot> mutationKeys;
    SmallVector<StateSlot> mutations;
    for (const auto statement : statements) {
      collectMutations(statement, mutationKeys, mutations);
    }
    llvm::sort(mutations);
    SmallVector<StateSlot> slots;
    slots.reserve(mutations.size());
    for (const auto slot : mutations) {
      auto value = scalarValues.at(slot);
      if (value) {
        slots.push_back(slot);
      }
    }
    return slots;
  }

  [[nodiscard]] SmallVector<Value>
  stateValues(ArrayRef<StateSlot> slots) const {
    SmallVector<Value> values;
    values.reserve(slots.size());
    for (const auto& slot : slots) {
      values.push_back(scalarValues.at(slot));
    }
    return values;
  }

  void assignState(ArrayRef<StateSlot> slots, ValueRange values) {
    for (auto [slot, value] : llvm::zip_equal(slots, values)) {
      scalarValues.at(slot) = value;
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
          } else if constexpr (std::is_same_v<
                                   T, frontend::BitVectorAssignmentStatement>) {
            emitBitVectorAssignment(data);
          } else if constexpr (std::is_same_v<T, frontend::GateApplication>) {
            emitGateApplication(builder, data, loc, gateParameters, gateQubits);
          } else if constexpr (std::is_same_v<T,
                                              frontend::MeasurementStatement>) {
            emitMeasurement(data, gateQubits);
          } else if constexpr (std::is_same_v<T, frontend::ResetStatement>) {
            for (const auto& qubit : data.qubits) {
              const auto indices = emitQubitIndices({qubit});
              builder.reset(resolveQubit(qubit, gateQubits, indices.front()));
            }
          } else if constexpr (std::is_same_v<T, frontend::BarrierStatement>) {
            const auto indices = emitQubitIndices(data.qubits);
            builder.barrier(resolveQubits(data.qubits, gateQubits, indices));
          } else if constexpr (std::is_same_v<T, frontend::IfStatement>) {
            emitIf(data, gateParameters, gateQubits);
          } else if constexpr (std::is_same_v<T, frontend::ForStatement>) {
            emitFor(data, gateParameters, gateQubits);
          } else if constexpr (std::is_same_v<T, frontend::WhileStatement>) {
            emitWhile(data, gateParameters, gateQubits);
          } else if constexpr (std::is_same_v<T, frontend::SwitchStatement>) {
            emitSwitch(data, gateParameters, gateQubits);
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
    case frontend::ScalarType::Angle:
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
      value = emitExpression(builder, *statement.initializer, {});
    } else if (statement.conditionInitializer) {
      value = emitCondition(*statement.conditionInitializer, {}, gateQubits);
    }
    scalarValues.at(statement.scalar) = value;
  }

  void
  emitScalarAssignment(const frontend::ScalarAssignmentStatement& statement,
                       ValueRange gateQubits) {
    if (statement.value) {
      scalarValues.at(statement.scalar) =
          emitExpression(builder, *statement.value, {});
      return;
    }
    scalarValues.at(statement.scalar) =
        emitCondition(*statement.condition, {}, gateQubits);
  }

  void emitDeclaration(const frontend::DeclarationStatement& statement) {
    const auto& declaration = program.registers.at(statement.reg);
    if (declaration.kind == frontend::RegisterKind::Qubit) {
      if (declaration.isScalar) {
        if (!emissionBudget.canConstruct(1)) {
          return;
        }
        qubitValues[statement.reg] = builder.allocQubit();
        return;
      }
      if (!emissionBudget.canConstruct(1)) {
        return;
      }
      qubitValues[statement.reg] = builder.allocQubitRegisterStorage(
          static_cast<int64_t>(declaration.width), declaration.name);
      return;
    }

    if (!emissionBudget.canConstruct(1)) {
      return;
    }
    classicalRegisters[statement.reg] = builder.allocClassicalBitRegister(
        static_cast<int64_t>(declaration.width), declaration.name,
        program.openQASM2 ? cbit::Initialization::Zero
                          : cbit::Initialization::Undefined);
  }

  void assignBit(const frontend::BitReference& target, Value value) {
    auto reg = classicalRegisters[target.reg];
    assert(reg && "semantic analysis must declare bit registers before use");
    if (!target.dynamicIndex) {
      builder.storeClassicalBit(value, reg, static_cast<int64_t>(target.index));
      return;
    }
    const auto width =
        static_cast<int64_t>(program.registers.at(target.reg).width);
    auto index = emitCheckedIndex(*target.dynamicIndex, width,
                                  "dynamic classical index out of bounds");
    auto registerIndex =
        arith::IndexCastOp::create(builder, builder.getIndexType(), index);
    builder.storeClassicalBit(value, reg, registerIndex.getResult());
  }

  void emitBitAssignment(const frontend::BitAssignmentStatement& assignment,
                         ValueRange gateQubits) {
    assignBit(assignment.target,
              emitCondition(assignment.value, {}, gateQubits));
  }

  void emitBitVectorAssignment(
      const frontend::BitVectorAssignmentStatement& assignment) {
    auto value = emitBitVectorExpression(builder, assignment.value);
    const auto bits = ensureBits(builder, value);
    auto reg = classicalRegisters[assignment.target];
    assert(reg && "semantic analysis must declare bit registers before use");
    for (auto [index, bit] : llvm::enumerate(bits)) {
      builder.storeClassicalBit(bit, reg, static_cast<int64_t>(index));
    }
  }

  void emitMeasurement(const frontend::MeasurementStatement& measurement,
                       ValueRange gateQubits) {
    if (measurement.targets.empty()) {
      for (const auto& qubit : measurement.qubits) {
        const auto indices = emitQubitIndices({qubit});
        std::ignore =
            builder.measure(resolveQubit(qubit, gateQubits, indices.front()));
      }
      return;
    }
    for (const auto [target, qubit] :
         llvm::zip_equal(measurement.targets, measurement.qubits)) {
      const auto emitMeasurement = [&](Value resolved) {
        return builder.measure(resolved);
      };
      auto measured = emitQubitOperation(qubit, gateQubits, emitMeasurement);
      if (!measured) {
        return;
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
    assignState(slots, ifOp.getResults());
  }

  [[nodiscard]] Value extendRangeValue(Value value, Type targetType,
                                       const bool isUnsigned) {
    if (isUnsigned) {
      return arith::ExtUIOp::create(builder, targetType, value);
    }
    return arith::ExtSIOp::create(builder, targetType, value);
  }

  [[nodiscard]] std::optional<int64_t>
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
      const auto bits =
          expression.type == frontend::ScalarType::Uint
              ? std::get<uint64_t>(expression.constant)
              : static_cast<uint64_t>(std::get<int64_t>(expression.constant));
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
    bool nonempty = false;
    if (positive) {
      nonempty = unsignedEndpoints ? start.ule(stop) : start.sle(stop);
    } else {
      nonempty = unsignedEndpoints ? start.uge(stop) : start.sge(stop);
    }
    if (!nonempty) {
      return 0;
    }
    const auto distance = positive ? stop - start : start - stop;
    const auto absoluteStep = positive ? step : -step;
    const auto count = distance.udiv(absoluteStep) + 1;
    const APInt maximum(
        128, static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
    if (count.ugt(maximum)) {
      return std::nullopt;
    }
    return static_cast<int64_t>(count.getZExtValue());
  }

  void emitFor(const frontend::ForStatement& loop, ValueRange gateParameters,
               ValueRange gateQubits) {
    const auto slots = mutatedState(loop.body);
    const auto initialValues = stateValues(slots);
    const auto savedScalars = scalarValues;

    if (loop.provenPositiveRange) {
      auto start = emitProvenIndexExpression(builder, loop.start);
      auto step = emitProvenIndexExpression(builder, loop.step);
      auto stop = emitProvenIndexExpression(builder, loop.stop);
      auto exclusiveStop = arith::AddIOp::create(
          builder, stop, arith::ConstantIndexOp::create(builder, 1));
      auto forOp = scf::ForOp::create(builder, start, exclusiveStop, step,
                                      initialValues);
      {
        OpBuilder::InsertionGuard guard(builder);
        auto* body = forOp.getBody();
        if (!body->empty()) {
          body->back().erase();
        }
        builder.setInsertionPointToEnd(body);
        scalarValues = savedScalars;
        assignState(slots, forOp.getRegionIterArgs());
        provenInductionValues[loop.inductionVariable] = forOp.getInductionVar();
        scalarValues.at(loop.inductionVariable) = arith::IndexCastOp::create(
            builder, builder.getI64Type(), forOp.getInductionVar());
        for (const auto statement : loop.body) {
          emitStatement(statement, gateParameters, gateQubits);
        }
        scf::YieldOp::create(builder, stateValues(slots));
      }
      scalarValues = savedScalars;
      provenInductionValues.erase(loop.inductionVariable);
      assignState(slots, forOp.getResults());
      return;
    }

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
    assignState(slots, whileOp.getResults().drop_front());
  }

  void emitWhile(const frontend::WhileStatement& loop,
                 ValueRange gateParameters, ValueRange gateQubits) {
    const auto slots = mutatedState(loop.body);
    const auto initialValues = stateValues(slots);
    const auto savedScalars = scalarValues;
    auto whileOp = scf::WhileOp::create(
        builder, ValueRange(initialValues).getTypes(), initialValues,
        [&](OpBuilder& nested, Location, ValueRange arguments) {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPoint(nested.getInsertionBlock(),
                                    nested.getInsertionPoint());
          scalarValues = savedScalars;
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
          assignState(slots, arguments);
          for (const auto statement : loop.body) {
            emitStatement(statement, gateParameters, gateQubits);
          }
          scf::YieldOp::create(builder, stateValues(slots));
        });
    scalarValues = savedScalars;
    assignState(slots, whileOp.getResults());
  }

  void emitSwitch(const frontend::SwitchStatement& switchStatement,
                  ValueRange gateParameters, ValueRange gateQubits) {
    SmallVector<frontend::StatementId> nestedStatements;
    SmallVector<int64_t> labels;
    for (const auto& switchCase : switchStatement.cases) {
      llvm::append_range(nestedStatements, switchCase.body);
      llvm::append_range(labels, switchCase.labels);
    }
    llvm::append_range(nestedStatements, switchStatement.defaultStatements);
    const auto slots = mutatedState(nestedStatements);
    const auto initialValues = stateValues(slots);
    const auto savedScalars = scalarValues;

    auto control = emitExpression(builder, switchStatement.control, {});
    auto selector =
        arith::IndexCastOp::create(builder, builder.getIndexType(), control);
    auto switchOp = scf::IndexSwitchOp::create(
        builder, ValueRange(initialValues).getTypes(), selector, labels,
        labels.size());
    OpBuilder::InsertionGuard guard(builder);
    const auto emitBranch =
        [&](Region& region, const ArrayRef<frontend::StatementId> statements) {
          auto& block = region.emplaceBlock();
          builder.setInsertionPointToEnd(&block);
          scalarValues = savedScalars;
          for (const auto statement : statements) {
            emitStatement(statement, gateParameters, gateQubits);
          }
          scf::YieldOp::create(builder, stateValues(slots));
        };
    size_t region = 0;
    for (const auto& switchCase : switchStatement.cases) {
      for ([[maybe_unused]] const auto label : switchCase.labels) {
        emitBranch(switchOp.getCaseRegions()[region++], switchCase.body);
      }
    }
    emitBranch(switchOp.getDefaultRegion(), switchStatement.defaultStatements);
    scalarValues = savedScalars;
    assignState(slots, switchOp.getResults());
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
