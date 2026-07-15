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

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Verifier.h>

#include <cstddef>
#include <cstdint>
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
        bitValues(program.registers.size()) {
    context.loadDialect<OQ3Dialect, qc::QCDialect, arith::ArithDialect,
                        func::FuncDialect, math::MathDialect,
                        memref::MemRefDialect, scf::SCFDialect>();
    builder.initialize();
  }

  OwningOpRef<ModuleOp> emit() {
    emitGateSymbols();
    for (const auto statement : program.body) {
      emitStatement(statement, {});
    }

    SmallVector<Value> results;
    for (const auto output : program.outputs) {
      for (const Value bit : bitValues[output]) {
        if (!bit) {
          llvm::errs() << "OpenQASM emission error: output register '"
                       << program.registers[output].name
                       << "' is not fully measured.\n";
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
    const auto loc = getLocation(definition.location);
    const auto type = gateType(context, definition.parameterNames.size(),
                               definition.qubitNames.size());
    OperationState state(loc, GateOp::getOperationName());
    state.addAttribute(SymbolTable::getSymbolAttrName(),
                       symbolBuilder.getStringAttr(definition.name));
    state.addAttribute("function_type", TypeAttr::get(type));
    state.addRegion();
    auto gate = cast<GateOp>(symbolBuilder.create(state));
    auto* block = new Block();
    gate.getBody().push_back(block);
    for (const Type input : type.getInputs()) {
      block->addArgument(input, loc);
    }

    OpBuilder bodyBuilder = OpBuilder::atBlockBegin(block);
    const auto parameterCount = definition.parameterNames.size();
    const auto parameters = block->getArguments().take_front(parameterCount);
    const auto qubits = block->getArguments().drop_front(parameterCount);
    for (const auto& application : definition.body) {
      emitGateApplication(bodyBuilder, application, parameters, qubits);
    }
    YieldOp::create(bodyBuilder, loc);
    (void)gate;
  }

  Value emitExpression(OpBuilder& opBuilder, const frontend::ExpressionId id,
                       const ValueRange gateParameters) {
    const auto& expression = program.expressions.at(id);
    const auto loc =
        opBuilder.getInsertionPoint() == opBuilder.getBlock()->end()
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
        return arith::ConstantIntOp::create(
            opBuilder, loc,
            static_cast<std::int64_t>(
                std::get<std::uint64_t>(expression.constant)),
            64);
      case frontend::ScalarType::Float:
        return arith::ConstantFloatOp::create(
            opBuilder, loc, opBuilder.getF64Type(),
            APFloat(std::get<double>(expression.constant)));
      }
      llvm_unreachable("unknown scalar type");
    case frontend::ExpressionKind::GateParameter:
      return gateParameters[expression.parameter];
    case frontend::ExpressionKind::Negate: {
      const Value operand =
          emitExpression(opBuilder, expression.lhs, gateParameters);
      if (isa<FloatType>(operand.getType())) {
        return arith::NegFOp::create(opBuilder, loc, operand);
      }
      const Value zero = arith::ConstantIntOp::create(opBuilder, loc, 0, 64);
      return arith::SubIOp::create(opBuilder, loc, zero, operand);
    }
    case frontend::ExpressionKind::BitwiseNot: {
      const Value operand =
          emitExpression(opBuilder, expression.lhs, gateParameters);
      const Value allOnes =
          arith::ConstantIntOp::create(opBuilder, loc, -1, 64);
      return arith::XOrIOp::create(opBuilder, loc, operand, allOnes);
    }
    case frontend::ExpressionKind::LogicalNot: {
      const Value operand =
          emitExpression(opBuilder, expression.lhs, gateParameters);
      const Value one = arith::ConstantIntOp::create(opBuilder, loc, 1, 1);
      return arith::XOrIOp::create(opBuilder, loc, operand, one);
    }
    case frontend::ExpressionKind::Add:
    case frontend::ExpressionKind::Subtract:
    case frontend::ExpressionKind::Multiply:
    case frontend::ExpressionKind::Divide:
    case frontend::ExpressionKind::Power: {
      const Value lhs =
          emitExpression(opBuilder, expression.lhs, gateParameters);
      const Value rhs =
          emitExpression(opBuilder, expression.rhs, gateParameters);
      if (expression.type == frontend::ScalarType::Float) {
        const auto toFloat = [&](const Value value,
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
        const Value floatLhs =
            toFloat(lhs, program.expressions.at(expression.lhs).type);
        const Value floatRhs =
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
        case frontend::ExpressionKind::Power:
          return math::PowFOp::create(opBuilder, loc, floatLhs, floatRhs);
        default:
          llvm_unreachable("not a floating-point binary expression");
        }
      }
      switch (expression.kind) {
      case frontend::ExpressionKind::Add:
        return arith::AddIOp::create(opBuilder, loc, lhs, rhs);
      case frontend::ExpressionKind::Subtract:
        return arith::SubIOp::create(opBuilder, loc, lhs, rhs);
      case frontend::ExpressionKind::Multiply:
        return arith::MulIOp::create(opBuilder, loc, lhs, rhs);
      case frontend::ExpressionKind::Divide:
        if (expression.type == frontend::ScalarType::Uint) {
          return arith::DivUIOp::create(opBuilder, loc, lhs, rhs);
        }
        return arith::DivSIOp::create(opBuilder, loc, lhs, rhs);
      case frontend::ExpressionKind::Power:
        return math::IPowIOp::create(opBuilder, loc, lhs, rhs);
      default:
        llvm_unreachable("not an integer binary expression");
      }
    }
    }
    llvm_unreachable("unknown scalar expression kind");
  }

  Value resolveQubit(const frontend::QubitReference& reference,
                     const ValueRange gateQubits) {
    switch (reference.kind) {
    case frontend::QubitReferenceKind::Register:
      return registerValues.at(reference.symbol)[reference.index];
    case frontend::QubitReferenceKind::GateArgument:
      return gateQubits[reference.symbol];
    case frontend::QubitReferenceKind::Hardware:
      return builder.staticQubit(reference.index);
    }
    llvm_unreachable("unknown qubit reference kind");
  }

  void emitGateApplication(OpBuilder& opBuilder,
                           const frontend::GateApplication& application,
                           const ValueRange gateParameters,
                           const ValueRange gateQubits) {
    const Location loc = getLocation(application.location);
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
    SmallVector<Value> qubits;
    qubits.reserve(application.qubits.size());
    for (const auto& reference : application.qubits) {
      qubits.push_back(resolveQubit(reference, gateQubits));
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

    OperationState state(loc, ApplyGateOp::getOperationName());
    state.addOperands(parameters);
    state.addOperands(qubits);
    state.addOperands(modifierOperands);
    state.addAttribute("callee",
                       FlatSymbolRefAttr::get(&context, application.callee));
    state.addAttribute("modifier_kinds",
                       DenseI32ArrayAttr::get(&context, modifierKinds));
    state.addAttribute("modifier_operand_indices",
                       DenseI32ArrayAttr::get(&context, modifierIndices));
    state.addAttribute(
        "operandSegmentSizes",
        DenseI32ArrayAttr::get(
            &context, {static_cast<std::int32_t>(parameters.size()),
                       static_cast<std::int32_t>(qubits.size()),
                       static_cast<std::int32_t>(modifierOperands.size())}));
    opBuilder.create(state);
  }

  void emitStatement(const frontend::StatementId id,
                     const ValueRange gateQubits) {
    const auto& statement = program.statements.at(id);
    builder.setLoc(getLocation(statement.location));
    std::visit(
        [&](const auto& data) {
          using T = std::decay_t<decltype(data)>;
          if constexpr (std::is_same_v<T, frontend::DeclarationStatement>) {
            emitDeclaration(data);
          } else if constexpr (std::is_same_v<T, frontend::GateApplication>) {
            emitGateApplication(builder, data, {}, gateQubits);
          } else if constexpr (std::is_same_v<T,
                                              frontend::MeasurementStatement>) {
            emitMeasurement(data, gateQubits);
          } else if constexpr (std::is_same_v<T, frontend::ResetStatement>) {
            for (const auto& qubit : data.qubits) {
              builder.reset(resolveQubit(qubit, gateQubits));
            }
          } else if constexpr (std::is_same_v<T, frontend::BarrierStatement>) {
            SmallVector<Value> qubits;
            for (const auto& qubit : data.qubits) {
              qubits.push_back(resolveQubit(qubit, gateQubits));
            }
            builder.barrier(qubits);
          } else if constexpr (std::is_same_v<T, frontend::IfStatement>) {
            emitIf(data, gateQubits);
          }
        },
        statement.data);
  }

  void emitDeclaration(const frontend::DeclarationStatement& statement) {
    const auto& declaration = program.registers.at(statement.reg);
    if (declaration.kind == frontend::RegisterKind::Qubit) {
      registerValues[statement.reg] =
          builder
              .allocQubitRegister(static_cast<std::int64_t>(declaration.width))
              .qubits;
      return;
    }
    classicalRegisters[statement.reg] = builder.allocClassicalBitRegister(
        static_cast<std::int64_t>(declaration.width), declaration.name);
    bitValues[statement.reg].resize(declaration.width);
  }

  void emitMeasurement(const frontend::MeasurementStatement& measurement,
                       const ValueRange gateQubits) {
    for (const auto [target, qubit] :
         llvm::zip_equal(measurement.targets, measurement.qubits)) {
      const auto& reg = classicalRegisters[target.reg];
      if (!reg) {
        llvm::errs() << "OpenQASM emission error: measurement target has no "
                        "classical storage.\n";
        return;
      }
      bitValues[target.reg][target.index] =
          builder.measure(resolveQubit(qubit, gateQubits),
                          (*reg)[static_cast<std::int64_t>(target.index)]);
    }
  }

  void emitIf(const frontend::IfStatement& conditional,
              const ValueRange gateQubits) {
    Value condition =
        bitValues[conditional.condition.reg][conditional.condition.index];
    const bool invertForEmptyThen = conditional.thenStatements.empty();
    if (conditional.negated != invertForEmptyThen) {
      condition =
          arith::XOrIOp::create(builder, condition, builder.boolConstant(true));
    }
    const auto& thenStatements = invertForEmptyThen
                                     ? conditional.elseStatements
                                     : conditional.thenStatements;
    const auto& elseStatements = invertForEmptyThen
                                     ? conditional.thenStatements
                                     : conditional.elseStatements;
    auto ifOp = scf::IfOp::create(builder, condition,
                                  /*withElseRegion=*/!elseStatements.empty());
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    for (const auto statement : thenStatements) {
      emitStatement(statement, gateQubits);
    }
    if (!elseStatements.empty()) {
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
      for (const auto statement : elseStatements) {
        emitStatement(statement, gateQubits);
      }
    }
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
