/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Target/OpenQASM/OpenQASM.h"

#include "mlir/Dialect/OQ3/IR/GateCatalog.h"
#include "mlir/Dialect/OQ3/IR/OQ3Dialect.h"
#include "mlir/Dialect/OQ3/IR/OQ3Ops.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCOps.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
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
#include <mlir/IR/Verifier.h>

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

namespace mlir::oq3 {
namespace {

class OQ3Emitter {
public:
  OQ3Emitter(const frontend::TypedProgram& typedProgram,
             MLIRContext& mlirContext)
      : program(typedProgram), context(mlirContext), builder(&context),
        registerValues(program.registers.size()),
        classicalRegisters(program.registers.size()),
        bitValues(program.registers.size()),
        scalarValues(program.scalars.size()) {
    context.loadDialect<OQ3Dialect, qc::QCDialect, arith::ArithDialect,
                        cf::ControlFlowDialect, func::FuncDialect,
                        math::MathDialect, memref::MemRefDialect,
                        scf::SCFDialect, ub::UBDialect>();
    builder.initialize();
  }

  OwningOpRef<ModuleOp> emit() {
    emitGateSymbols();
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
  const frontend::TypedProgram& program;
  MLIRContext& context;
  qc::QCProgramBuilder builder;
  std::vector<SmallVector<Value>> registerValues;
  std::vector<std::optional<qc::QCProgramBuilder::ClassicalRegister>>
      classicalRegisters;
  std::vector<SmallVector<Value>> bitValues;
  std::vector<Value> scalarValues;
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

  [[nodiscard]] ModuleOp getModule() const {
    return builder.getInsertionBlock()
        ->getParentOp()
        ->getParentOfType<ModuleOp>();
  }

  static FunctionType gateType(MLIRContext& context,
                               const std::size_t parameters,
                               const std::size_t qubits) {
    SmallVector<Type> inputs(parameters, Float64Type::get(&context));
    inputs.append(qubits, qc::QubitType::get(&context));
    return FunctionType::get(&context, inputs, {});
  }

  void emitGateSymbols() {
    OpBuilder symbolBuilder(&context);
    symbolBuilder.setInsertionPointToStart(getModule().getBody());
    for (const auto& gate : getGateCatalog()) {
      if (gate.availability != GateAvailability::Language &&
          program.gatePolicy == frontend::GatePolicy::Strict &&
          (gate.availability != GateAvailability::StandardLibrary ||
           !program.standardLibraryIncluded)) {
        continue;
      }
      GateDeclOp::create(
          symbolBuilder, symbolBuilder.getUnknownLoc(), gate.name,
          gateType(context, gate.parameterCount, gate.qubitCount()));
    }
    for (const auto& definition : program.gates) {
      emitGateDefinition(symbolBuilder, definition);
    }
  }

  void emitGateDefinition(OpBuilder& symbolBuilder,
                          const frontend::GateDefinition& definition) {
    auto loc = getLocation(definition.location);
    auto type =
        gateType(context, definition.parameterCount, definition.qubitCount);
    OperationState state(loc, GateOp::getOperationName());
    state.addAttribute(SymbolTable::getSymbolAttrName(),
                       symbolBuilder.getStringAttr(definition.name));
    state.addAttribute("function_type", TypeAttr::get(type));
    state.addRegion();
    auto gate = cast<GateOp>(symbolBuilder.create(state));
    auto* block = new Block();
    gate.getBody().push_back(block);
    for (auto input : type.getInputs()) {
      block->addArgument(input, loc);
    }

    OpBuilder bodyBuilder = OpBuilder::atBlockBegin(block);
    const auto parameterCount = definition.parameterCount;
    auto parameters = block->getArguments().take_front(parameterCount);
    auto qubits = block->getArguments().drop_front(parameterCount);
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(block);
      for (const auto statement : definition.body) {
        emitStatement(statement, parameters, qubits);
      }
    }
    YieldOp::create(bodyBuilder, loc);
  }

  [[nodiscard]] Value emitCheckedSignedResult(OpBuilder& opBuilder,
                                              const Location loc,
                                              Value wideResult,
                                              const StringRef message) {
    auto minimum = arith::ConstantIntOp::create(
        opBuilder, loc, std::numeric_limits<std::int64_t>::min(), 128);
    auto maximum = arith::ConstantIntOp::create(
        opBuilder, loc, std::numeric_limits<std::int64_t>::max(), 128);
    auto aboveMinimum = arith::CmpIOp::create(
        opBuilder, loc, arith::CmpIPredicate::sge, wideResult, minimum);
    auto belowMaximum = arith::CmpIOp::create(
        opBuilder, loc, arith::CmpIPredicate::sle, wideResult, maximum);
    auto inRange =
        arith::AndIOp::create(opBuilder, loc, aboveMinimum, belowMaximum);
    cf::AssertOp::create(opBuilder, loc, inRange, message);
    return arith::TruncIOp::create(opBuilder, loc, opBuilder.getI64Type(),
                                   wideResult);
  }

  [[nodiscard]] Value emitCheckedSignedPower(OpBuilder& opBuilder,
                                             const Location loc, Value base,
                                             Value exponent) {
    auto zero = arith::ConstantIntOp::create(opBuilder, loc, 0, 64);
    auto one = arith::ConstantIntOp::create(opBuilder, loc, 1, 64);
    auto nonnegative = arith::CmpIOp::create(
        opBuilder, loc, arith::CmpIPredicate::sge, exponent, zero);
    cf::AssertOp::create(opBuilder, loc, nonnegative,
                         "integer power requires a nonnegative exponent");
    auto valid = arith::ConstantIntOp::create(opBuilder, loc, 1, 1);
    SmallVector<Value> initial{one, base, exponent, valid};
    auto loop = scf::WhileOp::create(
        opBuilder, loc, ValueRange(initial).getTypes(), initial,
        [&](OpBuilder& nested, Location nestedLoc, ValueRange arguments) {
          auto active = arith::CmpIOp::create(
              nested, nestedLoc, arith::CmpIPredicate::ne, arguments[2], zero);
          scf::ConditionOp::create(nested, nestedLoc, active, arguments);
        },
        [&](OpBuilder& nested, Location nestedLoc, ValueRange arguments) {
          auto i128 = nested.getIntegerType(128);
          const auto checkedProduct = [&](Value lhs, Value rhs) {
            auto lhsWide = arith::ExtSIOp::create(nested, nestedLoc, i128, lhs);
            auto rhsWide = arith::ExtSIOp::create(nested, nestedLoc, i128, rhs);
            auto product =
                arith::MulIOp::create(nested, nestedLoc, lhsWide, rhsWide);
            auto narrowed = arith::TruncIOp::create(
                nested, nestedLoc, nested.getI64Type(), product);
            auto restored =
                arith::ExtSIOp::create(nested, nestedLoc, i128, narrowed);
            auto fits = arith::CmpIOp::create(
                nested, nestedLoc, arith::CmpIPredicate::eq, product, restored);
            return std::pair<Value, Value>{narrowed, fits};
          };

          auto [multiplied, multiplicationFits] =
              checkedProduct(arguments[0], arguments[1]);
          auto lowBit =
              arith::AndIOp::create(nested, nestedLoc, arguments[2], one);
          auto odd = arith::CmpIOp::create(
              nested, nestedLoc, arith::CmpIPredicate::ne, lowBit, zero);
          auto nextResult = arith::SelectOp::create(nested, nestedLoc, odd,
                                                    multiplied, arguments[0]);
          auto resultFits = arith::SelectOp::create(nested, nestedLoc, odd,
                                                    multiplicationFits, valid);

          auto nextExponent =
              arith::ShRUIOp::create(nested, nestedLoc, arguments[2], one);
          auto [squared, squareFits] =
              checkedProduct(arguments[1], arguments[1]);
          auto needsSquare = arith::CmpIOp::create(
              nested, nestedLoc, arith::CmpIPredicate::ne, nextExponent, zero);
          auto requiredSquareFits = arith::SelectOp::create(
              nested, nestedLoc, needsSquare, squareFits, valid);
          auto checks = arith::AndIOp::create(nested, nestedLoc, resultFits,
                                              requiredSquareFits);
          auto allValid =
              arith::AndIOp::create(nested, nestedLoc, arguments[3], checks);
          scf::YieldOp::create(
              nested, nestedLoc,
              ValueRange{nextResult, squared, nextExponent, allValid});
        });
    cf::AssertOp::create(opBuilder, loc, loop.getResult(3),
                         "integer power overflows i64");
    return loop.getResult(0);
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
      auto zero = arith::ConstantIntOp::create(opBuilder, loc, 0, 64);
      if (expression.type == frontend::ScalarType::Int) {
        auto minimum = arith::ConstantIntOp::create(
            opBuilder, loc, std::numeric_limits<std::int64_t>::min(), 64);
        auto safe = arith::CmpIOp::create(
            opBuilder, loc, arith::CmpIPredicate::ne, operand, minimum);
        cf::AssertOp::create(opBuilder, loc, safe,
                             "integer negation overflows i64");
      }
      return arith::SubIOp::create(opBuilder, loc, zero, operand);
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
      if (expression.type == frontend::ScalarType::Float) {
        const auto toFloat = [&](Value value,
                                 const frontend::ScalarType sourceType) {
          if (isa<FloatType>(value.getType())) {
            return value;
          }
          if (sourceType == frontend::ScalarType::Uint) {
            return arith::UIToFPOp::create(opBuilder, loc,
                                           opBuilder.getF64Type(), value)
                .getResult();
          }
          return arith::SIToFPOp::create(opBuilder, loc, opBuilder.getF64Type(),
                                         value)
              .getResult();
        };
        auto floatLhs =
            toFloat(lhs, program.expressions.at(expression.lhs).type);
        auto floatRhs =
            toFloat(rhs, program.expressions.at(expression.rhs).type);
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
      switch (expression.kind) {
      case frontend::ExpressionKind::Add:
      case frontend::ExpressionKind::Subtract:
      case frontend::ExpressionKind::Multiply:
        if (expression.type == frontend::ScalarType::Uint) {
          if (expression.kind == frontend::ExpressionKind::Add) {
            return arith::AddIOp::create(opBuilder, loc, lhs, rhs);
          }
          if (expression.kind == frontend::ExpressionKind::Subtract) {
            return arith::SubIOp::create(opBuilder, loc, lhs, rhs);
          }
          return arith::MulIOp::create(opBuilder, loc, lhs, rhs);
        }
        {
          auto i128 = opBuilder.getIntegerType(128);
          auto lhsWide = arith::ExtSIOp::create(opBuilder, loc, i128, lhs);
          auto rhsWide = arith::ExtSIOp::create(opBuilder, loc, i128, rhs);
          Value result;
          if (expression.kind == frontend::ExpressionKind::Add) {
            result = arith::AddIOp::create(opBuilder, loc, lhsWide, rhsWide);
          } else if (expression.kind == frontend::ExpressionKind::Subtract) {
            result = arith::SubIOp::create(opBuilder, loc, lhsWide, rhsWide);
          } else {
            result = arith::MulIOp::create(opBuilder, loc, lhsWide, rhsWide);
          }
          return emitCheckedSignedResult(opBuilder, loc, result,
                                         "integer arithmetic overflows i64");
        }
      case frontend::ExpressionKind::Divide:
      case frontend::ExpressionKind::Modulo: {
        auto zero = arith::ConstantIntOp::create(opBuilder, loc, 0, 64);
        auto nonzero = arith::CmpIOp::create(
            opBuilder, loc, arith::CmpIPredicate::ne, rhs, zero);
        cf::AssertOp::create(opBuilder, loc, nonzero,
                             expression.kind == frontend::ExpressionKind::Divide
                                 ? "integer division by zero"
                                 : "integer modulo by zero");
        if (expression.type == frontend::ScalarType::Uint) {
          if (expression.kind == frontend::ExpressionKind::Divide) {
            return arith::DivUIOp::create(opBuilder, loc, lhs, rhs);
          }
          return arith::RemUIOp::create(opBuilder, loc, lhs, rhs);
        }
        auto minimum = arith::ConstantIntOp::create(
            opBuilder, loc, std::numeric_limits<std::int64_t>::min(), 64);
        auto negativeOne = arith::ConstantIntOp::create(opBuilder, loc, -1, 64);
        auto isMinimum = arith::CmpIOp::create(
            opBuilder, loc, arith::CmpIPredicate::eq, lhs, minimum);
        auto isNegativeOne = arith::CmpIOp::create(
            opBuilder, loc, arith::CmpIPredicate::eq, rhs, negativeOne);
        auto overflows =
            arith::AndIOp::create(opBuilder, loc, isMinimum, isNegativeOne);
        auto safe = arith::XOrIOp::create(
            opBuilder, loc, overflows,
            arith::ConstantIntOp::create(opBuilder, loc, 1, 1));
        cf::AssertOp::create(opBuilder, loc, safe,
                             "signed integer division overflows i64");
        if (expression.kind == frontend::ExpressionKind::Divide) {
          return arith::DivSIOp::create(opBuilder, loc, lhs, rhs);
        }
        return arith::RemSIOp::create(opBuilder, loc, lhs, rhs);
      }
      case frontend::ExpressionKind::Power:
        if (expression.type == frontend::ScalarType::Int) {
          return emitCheckedSignedPower(opBuilder, loc, lhs, rhs);
        }
        return math::IPowIOp::create(opBuilder, loc, lhs, rhs);
      default:
        llvm_unreachable("not an integer binary expression");
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

  [[nodiscard]] bool
  validateDynamicDispatchCost(ArrayRef<frontend::QubitReference> references) {
    std::size_t leaves = 1;
    for (const auto& reference : references) {
      if (!reference.dynamicIndex) {
        continue;
      }
      const auto width = program.registers.at(reference.symbol).width;
      if (width > frontend::kDynamicQubitDispatchLeafLimit / leaves) {
        llvm::errs()
            << "OpenQASM emission error: dynamic qubit selection exceeds the "
               "structured-dispatch expansion budget.\n";
        emissionFailed = true;
        return false;
      }
      leaves *= static_cast<std::size_t>(width);
    }
    return true;
  }

  void
  dispatchQubits(ArrayRef<frontend::QubitReference> references,
                 ValueRange gateQubits, ValueRange dynamicIndices,
                 llvm::function_ref<void(ValueRange)> emitResolvedOperation) {
    if (!validateDynamicDispatchCost(references)) {
      return;
    }
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
    if (!validateDynamicDispatchCost({reference})) {
      return {};
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
    SmallVector<Value> modifierOperands;
    SmallVector<std::int32_t> modifierKinds;
    SmallVector<std::int32_t> modifierIndices;
    for (const auto& modifier : application.modifiers) {
      modifierKinds.push_back(static_cast<std::int32_t>(modifier.kind));
      if (!modifier.operand) {
        modifierIndices.push_back(-1);
        continue;
      }
      modifierIndices.push_back(
          static_cast<std::int32_t>(modifierOperands.size()));
      modifierOperands.push_back(
          emitExpression(opBuilder, *modifier.operand, gateParameters));
    }

    dispatchQubits(
        application.qubits, gateQubits, dynamicIndices, [&](ValueRange qubits) {
          llvm::DenseSet<Value> distinctQubits(qubits.begin(), qubits.end());
          if (distinctQubits.size() != qubits.size()) {
            return;
          }
          OperationState state(loc, ApplyGateOp::getOperationName());
          state.addOperands(parameters);
          state.addOperands(qubits);
          state.addOperands(modifierOperands);
          state.addAttribute(
              "callee", FlatSymbolRefAttr::get(&context, application.callee));
          state.addAttribute("modifier_kinds",
                             DenseI32ArrayAttr::get(&context, modifierKinds));
          state.addAttribute("modifier_operand_indices",
                             DenseI32ArrayAttr::get(&context, modifierIndices));
          state.addAttribute(
              "operandSegmentSizes",
              DenseI32ArrayAttr::get(
                  &context,
                  {static_cast<std::int32_t>(parameters.size()),
                   static_cast<std::int32_t>(qubits.size()),
                   static_cast<std::int32_t>(modifierOperands.size())}));
          opBuilder.create(state);
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
          return arith::CmpFPredicate::OLT; // spellchecker:disable-line
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
    auto one = arith::ConstantIntOp::create(builder, 1, 128);
    auto nonzero = arith::CmpIOp::create(builder, arith::CmpIPredicate::ne,
                                         stepWide, zero);
    cf::AssertOp::create(builder, nonzero,
                         "for-loop range step must not be zero");
    auto positive = arith::CmpIOp::create(builder, arith::CmpIPredicate::sgt,
                                          stepWide, zero);
    auto ascending = arith::CmpIOp::create(builder, arith::CmpIPredicate::sle,
                                           startWide, stopWide);
    auto descending = arith::CmpIOp::create(builder, arith::CmpIPredicate::sge,
                                            startWide, stopWide);
    auto active =
        arith::SelectOp::create(builder, positive, ascending, descending);
    auto ascendingDistance =
        arith::SubIOp::create(builder, stopWide, startWide);
    auto descendingDistance =
        arith::SubIOp::create(builder, startWide, stopWide);
    auto distance = arith::SelectOp::create(
        builder, positive, ascendingDistance, descendingDistance);
    auto negativeStep = arith::SubIOp::create(builder, zero, stepWide);
    auto absoluteStep =
        arith::SelectOp::create(builder, positive, stepWide, negativeStep);
    auto quotient = arith::DivUIOp::create(builder, distance, absoluteStep);
    auto activeCount = arith::AddIOp::create(builder, quotient, one);
    auto count = arith::SelectOp::create(builder, active, activeCount, zero);
    auto maxCount = arith::ConstantIntOp::create(
        builder, std::numeric_limits<std::int64_t>::max(), 128);
    auto countFits = arith::CmpIOp::create(builder, arith::CmpIPredicate::ule,
                                           count, maxCount);
    cf::AssertOp::create(builder, countFits,
                         "for-loop iteration count exceeds index range");
    auto countI64 =
        arith::TruncIOp::create(builder, builder.getI64Type(), count);
    auto upperBound =
        arith::IndexCastOp::create(builder, builder.getIndexType(), countI64);
    auto lowerBound = arith::ConstantIndexOp::create(builder, 0);
    auto indexStep = arith::ConstantIndexOp::create(builder, 1);

    auto forOp = scf::ForOp::create(builder, lowerBound, upperBound, indexStep,
                                    initialValues);
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
      scalarValues.at(loop.inductionVariable) =
          arith::TruncIOp::create(builder, builder.getI64Type(), inductionWide);
      for (const auto statement : loop.body) {
        emitStatement(statement, gateParameters, gateQubits);
      }
      scf::YieldOp::create(builder, stateValues(slots));
    }
    scalarValues = savedScalars;
    bitValues = savedBits;
    assignState(slots, forOp.getResults());
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

void printDiagnostics(const std::vector<frontend::Diagnostic>& diagnostics) {
  for (const auto& diagnostic : diagnostics) {
    llvm::errs() << diagnostic.location.filename << ':'
                 << diagnostic.location.line << ':'
                 << diagnostic.location.column
                 << ": OpenQASM frontend error: " << diagnostic.message << '\n';
  }
}

} // namespace

OwningOpRef<ModuleOp> emitOQ3(const frontend::TypedProgram& program,
                              MLIRContext& context) {
  auto module = OQ3Emitter(program, context).emit();
  if (module && failed(verify(*module))) {
    llvm::errs() << "OpenQASM emission produced invalid OQ3 IR.\n";
    return nullptr;
  }
  return module;
}

OwningOpRef<ModuleOp>
translateOpenQASMToOQ3(llvm::SourceMgr& sourceMgr, MLIRContext& context,
                       const OpenQASMTranslationOptions& options) {
  auto analyzed = frontend::analyzeOpenQASM(sourceMgr, options.frontend);
  if (!analyzed) {
    printDiagnostics(analyzed.diagnostics);
    return nullptr;
  }
  return emitOQ3(*analyzed.program, context);
}

OwningOpRef<ModuleOp>
translateOpenQASMToOQ3(const llvm::StringRef source, MLIRContext& context,
                       const OpenQASMTranslationOptions& options) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBufferCopy(source),
                               llvm::SMLoc());
  return translateOpenQASMToOQ3(sourceMgr, context, options);
}

} // namespace mlir::oq3
