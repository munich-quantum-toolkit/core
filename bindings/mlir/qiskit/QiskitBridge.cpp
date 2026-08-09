/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "QiskitBridge.h"

#include "Dispatcher.h"
#include "QiskitAdapter.h"
#include "jeff/IR/JeffDialect.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCInterfaces.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSet.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace mqt::bindings::qiskit {

class QCProgramAccess {
public:
  [[nodiscard]] static mlir::QCProgram
  create(std::shared_ptr<mlir::MLIRContext> context,
         mlir::OwningOpRef<mlir::ModuleOp> module) {
    return mlir::QCProgram(mlir::QCProgram::Storage{
        .context = std::move(context), .mod = std::move(module)});
  }

  [[nodiscard]] static mlir::ModuleOp module(const mlir::QCProgram& program) {
    return program.mod();
  }
};

namespace {

constexpr llvm::StringLiteral QUANTUM_REGISTERS_ATTR =
    "mqt.qiskit.quantum_registers";
constexpr llvm::StringLiteral CLASSICAL_REGISTERS_ATTR =
    "mqt.qiskit.classical_registers";
constexpr llvm::StringLiteral PARAMETER_NAME_ATTR = "mqt.qiskit.parameter_name";

using ParameterValue = std::variant<double, mlir::Value>;

struct Symbol {
  std::size_t index;
  ClassicalType type;
  std::uint32_t width;
};

using Symbols = llvm::StringMap<Symbol>;
using LocalParameters = llvm::StringMap<mlir::Value>;

[[nodiscard]] std::shared_ptr<mlir::MLIRContext> createContext() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::qc::QCDialect, mlir::qco::QCODialect,
                  mlir::qtensor::QTensorDialect, mlir::arith::ArithDialect,
                  mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
                  mlir::scf::SCFDialect, mlir::LLVM::LLVMDialect,
                  mlir::memref::MemRefDialect, mlir::jeff::JeffDialect>();
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  return context;
}

void addSymbol(const std::string_view name, const ClassicalType type,
               const std::uint32_t width, Symbols& symbols) {
  const auto found = symbols.find(name);
  if (found != symbols.end()) {
    if (found->second.type != type || found->second.width != width) {
      throw std::runtime_error("Qiskit symbol '" + std::string(name) +
                               "' is used with incompatible types");
    }
    return;
  }
  symbols.insert(
      {name, Symbol{.index = symbols.size(), .type = type, .width = width}});
}

void collectParameter(const Parameter& parameter, Symbols& symbols,
                      const llvm::StringSet<>& localParameters) {
  if (parameter.kind == ParameterKind::Number) {
    return;
  }
  if (localParameters.contains(parameter.text)) {
    return;
  }
  if (parameter.kind == ParameterKind::Expression) {
    throw std::runtime_error(
        "Qiskit compiler bridge supports numeric parameters and bare "
        "parameter symbols; unsupported expression '" +
        parameter.text + "'");
  }
  if (parameter.text.empty()) {
    throw std::runtime_error("Qiskit returned an empty parameter symbol");
  }
  addSymbol(parameter.text, ClassicalType::Float, 64U, symbols);
}

void collectExpressionSymbols(const Expression& expression, Symbols& symbols) {
  if (expression.kind == ExpressionKind::Variable) {
    addSymbol(expression.variableName, expression.type, expression.width,
              symbols);
  }
  if (expression.left) {
    collectExpressionSymbols(*expression.left, symbols);
  }
  if (expression.right) {
    collectExpressionSymbols(*expression.right, symbols);
  }
}

[[nodiscard]] ParameterValue
parameterValue(const std::string_view text, const Symbols& symbols,
               const llvm::ArrayRef<mlir::Value> arguments,
               const LocalParameters& localParameters) {
  if (const auto local = localParameters.find(text);
      local != localParameters.end()) {
    return local->second;
  }
  const auto symbol = symbols.find(text);
  if (symbol == symbols.end()) {
    throw std::runtime_error("internal error: unregistered Qiskit parameter '" +
                             std::string(text) + "'");
  }
  return arguments[symbol->second.index];
}

[[nodiscard]] ParameterValue
parameterValue(const Parameter& parameter, const Symbols& symbols,
               const llvm::ArrayRef<mlir::Value> arguments,
               const LocalParameters& localParameters) {
  if (parameter.kind == ParameterKind::Number) {
    if (!std::isfinite(parameter.number)) {
      throw std::runtime_error("Qiskit returned a non-finite parameter");
    }
    return parameter.number;
  }
  if (const auto local = localParameters.find(parameter.text);
      local != localParameters.end()) {
    return local->second;
  }
  if (parameter.kind == ParameterKind::Expression) {
    throw std::runtime_error(
        "Qiskit compiler bridge supports numeric parameters and bare "
        "parameter symbols; unsupported expression '" +
        parameter.text + "'");
  }
  return parameterValue(parameter.text, symbols, arguments, localParameters);
}

void requireArity(const Instruction& instruction, const std::size_t qubits,
                  const std::size_t parameters) {
  if (instruction.qubits.size() != qubits ||
      instruction.parameters.size() != parameters) {
    throw std::runtime_error("Qiskit instruction '" + instruction.name +
                             "' has an unsupported operand arity");
  }
}

void emitBaseGate(mlir::qc::QCProgramBuilder& builder,
                  const std::string_view name, const mlir::ValueRange qubits,
                  const llvm::ArrayRef<ParameterValue> parameters) {
  if (name == "id") {
    builder.id(qubits[0]);
  } else if (name == "x") {
    builder.x(qubits[0]);
  } else if (name == "y") {
    builder.y(qubits[0]);
  } else if (name == "z") {
    builder.z(qubits[0]);
  } else if (name == "h") {
    builder.h(qubits[0]);
  } else if (name == "s") {
    builder.s(qubits[0]);
  } else if (name == "sdg") {
    builder.sdg(qubits[0]);
  } else if (name == "t") {
    builder.t(qubits[0]);
  } else if (name == "tdg") {
    builder.tdg(qubits[0]);
  } else if (name == "sx") {
    builder.sx(qubits[0]);
  } else if (name == "sxdg") {
    builder.sxdg(qubits[0]);
  } else if (name == "rx") {
    builder.rx(parameters[0], qubits[0]);
  } else if (name == "ry") {
    builder.ry(parameters[0], qubits[0]);
  } else if (name == "rz") {
    builder.rz(parameters[0], qubits[0]);
  } else if (name == "p" || name == "u1") {
    builder.p(parameters[0], qubits[0]);
  } else if (name == "r") {
    builder.r(parameters[0], parameters[1], qubits[0]);
  } else if (name == "u2") {
    builder.u2(parameters[0], parameters[1], qubits[0]);
  } else if (name == "u" || name == "u3") {
    builder.u(parameters[0], parameters[1], parameters[2], qubits[0]);
  } else if (name == "swap") {
    builder.swap(qubits[0], qubits[1]);
  } else if (name == "iswap") {
    builder.iswap(qubits[0], qubits[1]);
  } else if (name == "dcx") {
    builder.dcx(qubits[0], qubits[1]);
  } else if (name == "ecr") {
    builder.ecr(qubits[0], qubits[1]);
  } else if (name == "rxx") {
    builder.rxx(parameters[0], qubits[0], qubits[1]);
  } else if (name == "ryy") {
    builder.ryy(parameters[0], qubits[0], qubits[1]);
  } else if (name == "rzx") {
    builder.rzx(parameters[0], qubits[0], qubits[1]);
  } else if (name == "rzz") {
    builder.rzz(parameters[0], qubits[0], qubits[1]);
  } else if (name == "xx_plus_yy") {
    builder.xx_plus_yy(parameters[0], parameters[1], qubits[0], qubits[1]);
  } else if (name == "xx_minus_yy") {
    builder.xx_minus_yy(parameters[0], parameters[1], qubits[0], qubits[1]);
  } else if (name == "rccx") {
    builder.rccx(qubits[0], qubits[1], qubits[2]);
  } else {
    throw std::runtime_error("unsupported Qiskit standard gate '" +
                             std::string(name) + "'");
  }
}

void emitControlledGate(mlir::qc::QCProgramBuilder& builder,
                        const std::string_view baseName,
                        const mlir::ValueRange controls,
                        const mlir::ValueRange targets,
                        const llvm::ArrayRef<ParameterValue> parameters) {
  builder.ctrl(controls, targets, [&](const mlir::ValueRange targetArguments) {
    emitBaseGate(builder, baseName, targetArguments, parameters);
  });
}

void emitRC3X(mlir::qc::QCProgramBuilder& builder,
              const mlir::ValueRange qubits) {
  constexpr std::size_t dimension = 16U;
  std::vector<std::complex<double>> matrix(dimension * dimension);
  for (std::size_t index = 0; index < dimension; ++index) {
    matrix[index * dimension + index] = 1.0;
  }
  matrix[3U * dimension + 3U] = {0.0, 1.0};
  matrix[7U * dimension + 7U] = 0.0;
  matrix[7U * dimension + 15U] = 1.0;
  matrix[11U * dimension + 11U] = {0.0, -1.0};
  matrix[15U * dimension + 15U] = 0.0;
  matrix[15U * dimension + 7U] = -1.0;
  const auto type = mlir::RankedTensorType::get(
      {dimension, dimension}, mlir::ComplexType::get(builder.getF64Type()));
  builder.unitary(qubits,
                  mlir::DenseElementsAttr::get(
                      type, llvm::ArrayRef<std::complex<double>>(matrix)));
}

void emitGate(mlir::qc::QCProgramBuilder& builder,
              const Instruction& instruction,
              const llvm::ArrayRef<mlir::Value> allQubits,
              const llvm::ArrayRef<std::uint32_t> qubitMap,
              const Symbols& symbols,
              const llvm::ArrayRef<mlir::Value> arguments,
              const LocalParameters& localParameters) {
  llvm::SmallVector<mlir::Value> qubits;
  qubits.reserve(instruction.qubits.size());
  for (const auto index : instruction.qubits) {
    if (index >= qubitMap.size() || qubitMap[index] >= allQubits.size()) {
      throw std::runtime_error(
          "Qiskit instruction references an invalid qubit");
    }
    qubits.push_back(allQubits[qubitMap[index]]);
  }
  llvm::SmallVector<ParameterValue> parameters;
  parameters.reserve(instruction.parameters.size());
  for (const auto& parameter : instruction.parameters) {
    parameters.push_back(
        parameterValue(parameter, symbols, arguments, localParameters));
  }

  const auto& name = instruction.name;
  const llvm::ArrayRef<mlir::Value> qubitRange(qubits);
  if (name == "global_phase") {
    requireArity(instruction, 0, 1);
    builder.gphase(parameters[0]);
    return;
  }
  if (name == "cx" || name == "cy" || name == "cz" || name == "ch" ||
      name == "cs" || name == "csdg" || name == "csx") {
    requireArity(instruction, 2, 0);
    emitControlledGate(builder, name.substr(1), qubitRange.take_front(1),
                       qubitRange.take_back(1), parameters);
    return;
  }
  if (name == "cp" || name == "crx" || name == "cry" || name == "crz") {
    requireArity(instruction, 2, 1);
    const auto base =
        name == "cp" ? std::string_view{"p"} : std::string_view{name}.substr(1);
    emitControlledGate(builder, base, qubitRange.take_front(1),
                       qubitRange.take_back(1), parameters);
    return;
  }
  if (name == "ccx" || name == "ccz") {
    requireArity(instruction, 3, 0);
    emitControlledGate(builder, name.substr(2), qubitRange.take_front(2),
                       qubitRange.take_back(1), parameters);
    return;
  }
  if (name == "cswap") {
    requireArity(instruction, 3, 0);
    emitControlledGate(builder, "swap", qubitRange.take_front(1),
                       qubitRange.take_back(2), parameters);
    return;
  }
  if (name == "cu") {
    requireArity(instruction, 2, 4);
    builder.ctrl(qubitRange.take_front(1), qubitRange.take_back(1),
                 [&](const mlir::ValueRange targetArguments) {
                   builder.gphase(parameters[3]);
                   builder.u(parameters[0], parameters[1], parameters[2],
                             targetArguments[0]);
                 });
    return;
  }
  if (name == "cu1" || name == "cu3") {
    requireArity(instruction, 2, name == "cu1" ? 1U : 3U);
    emitControlledGate(builder, name == "cu1" ? "p" : "u",
                       qubitRange.take_front(1), qubitRange.take_back(1),
                       parameters);
    return;
  }
  if (name == "mcx" || name == "c3sx") {
    requireArity(instruction, 4, 0);
    emitControlledGate(builder, name == "mcx" ? "x" : "sx",
                       qubitRange.take_front(3), qubitRange.take_back(1),
                       parameters);
    return;
  }
  if (name == "rcccx") {
    requireArity(instruction, 4, 0);
    emitRC3X(builder, qubitRange);
    return;
  }

  static const llvm::StringMap<std::pair<std::size_t, std::size_t>> arities = {
      {"id", {1, 0}},    {"x", {1, 0}},          {"y", {1, 0}},
      {"z", {1, 0}},     {"h", {1, 0}},          {"s", {1, 0}},
      {"sdg", {1, 0}},   {"t", {1, 0}},          {"tdg", {1, 0}},
      {"sx", {1, 0}},    {"sxdg", {1, 0}},       {"rx", {1, 1}},
      {"ry", {1, 1}},    {"rz", {1, 1}},         {"p", {1, 1}},
      {"u1", {1, 1}},    {"r", {1, 2}},          {"u2", {1, 2}},
      {"u", {1, 3}},     {"u3", {1, 3}},         {"swap", {2, 0}},
      {"iswap", {2, 0}}, {"dcx", {2, 0}},        {"ecr", {2, 0}},
      {"rxx", {2, 1}},   {"ryy", {2, 1}},        {"rzx", {2, 1}},
      {"rzz", {2, 1}},   {"xx_plus_yy", {2, 2}}, {"xx_minus_yy", {2, 2}},
      {"rccx", {3, 0}},
  };
  const auto arity = arities.find(name);
  if (arity == arities.end()) {
    throw std::runtime_error("unsupported Qiskit standard gate '" + name + "'");
  }
  requireArity(instruction, arity->second.first, arity->second.second);
  emitBaseGate(builder, name, qubits, parameters);
}

[[nodiscard]] mlir::ArrayAttr registerAttributes(mlir::OpBuilder& builder,
                                                 const CircuitView& circuit,
                                                 const bool quantum) {
  llvm::SmallVector<mlir::Attribute> result;
  const auto count =
      quantum ? circuit.numQuantumRegisters() : circuit.numClassicalRegisters();
  result.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    const auto reg = quantum ? circuit.quantumRegister(index)
                             : circuit.classicalRegister(index);
    llvm::SmallVector<std::int32_t> bits;
    bits.reserve(reg.bits.size());
    for (const auto bit : reg.bits) {
      if (bit > static_cast<std::uint32_t>(
                    std::numeric_limits<std::int32_t>::max())) {
        throw std::runtime_error("Qiskit register bit index is too large");
      }
      bits.push_back(static_cast<std::int32_t>(bit));
    }
    result.push_back(builder.getDictionaryAttr({
        builder.getNamedAttr("name", builder.getStringAttr(reg.name)),
        builder.getNamedAttr("bits", builder.getDenseI32ArrayAttr(bits)),
    }));
  }
  return builder.getArrayAttr(result);
}

void collectCircuitSymbols(const CircuitView& circuit, Symbols& symbols,
                           llvm::StringSet<> localParameters);

void collectControlFlowSymbols(const ControlFlowView& controlFlow,
                               Symbols& symbols,
                               llvm::StringSet<> localParameters) {
  switch (controlFlow.kind()) {
  case ControlFlowKind::Box:
    throw std::runtime_error("Qiskit box instructions are not supported");
  case ControlFlowKind::Break:
    throw std::runtime_error("Qiskit break instructions are not supported");
  case ControlFlowKind::Continue:
    throw std::runtime_error("Qiskit continue instructions are not supported");
  case ControlFlowKind::IfElse:
  case ControlFlowKind::While: {
    const auto condition = controlFlow.condition();
    if (condition.expression) {
      collectExpressionSymbols(*condition.expression, symbols);
    }
    break;
  }
  case ControlFlowKind::For: {
    const auto loop = controlFlow.loop();
    if (loop.parameter) {
      localParameters.insert(*loop.parameter);
    }
    break;
  }
  case ControlFlowKind::Switch: {
    const auto target = controlFlow.switchTarget();
    if (target.expression) {
      collectExpressionSymbols(*target.expression, symbols);
    }
    break;
  }
  }
  for (std::size_t index = 0; index < controlFlow.numBlocks(); ++index) {
    const auto block = controlFlow.block(index);
    collectCircuitSymbols(*block, symbols, localParameters);
  }
}

void collectCircuitSymbols(const CircuitView& circuit, Symbols& symbols,
                           llvm::StringSet<> localParameters) {
  collectParameter(circuit.globalPhase(), symbols, localParameters);
  for (std::size_t index = 0; index < circuit.numInstructions(); ++index) {
    const auto instruction = circuit.instruction(index);
    for (const auto& parameter : instruction.parameters) {
      collectParameter(parameter, symbols, localParameters);
    }
    if (instruction.kind == OperationKind::ControlFlow) {
      const auto controlFlow = circuit.controlFlow(index);
      collectControlFlowSymbols(*controlFlow, symbols, localParameters);
    }
  }
}

[[nodiscard]] mlir::Type expressionType(mlir::OpBuilder& builder,
                                        const ClassicalType type,
                                        const std::uint32_t width) {
  switch (type) {
  case ClassicalType::Bool:
    return builder.getI1Type();
  case ClassicalType::Uint:
    if (width == 0U || width > 64U) {
      throw std::runtime_error(
          "Qiskit unsigned classical values must be between 1 and 64 bits");
    }
    return builder.getIntegerType(width);
  case ClassicalType::Float:
    return builder.getF64Type();
  }
  throw std::runtime_error("unknown normalized Qiskit classical type");
}

[[nodiscard]] mlir::Value integerConstant(mlir::ImplicitLocOpBuilder& builder,
                                          const std::uint32_t width,
                                          const std::uint64_t value) {
  const auto type = builder.getIntegerType(width);
  const auto attribute =
      builder.getIntegerAttr(type, llvm::APInt(width, value, false));
  return mlir::arith::ConstantOp::create(builder, attribute).getResult();
}

[[nodiscard]] mlir::Value castInteger(mlir::ImplicitLocOpBuilder& builder,
                                      const mlir::Value value,
                                      const mlir::IntegerType target) {
  const auto source = llvm::dyn_cast<mlir::IntegerType>(value.getType());
  if (!source) {
    throw std::runtime_error(
        "Qiskit classical integer cast has a non-integer operand");
  }
  if (source == target) {
    return value;
  }
  if (source.getWidth() < target.getWidth()) {
    return mlir::arith::ExtUIOp::create(builder, target, value).getResult();
  }
  return mlir::arith::TruncIOp::create(builder, target, value).getResult();
}

[[nodiscard]] mlir::Value
emitExpression(mlir::qc::QCProgramBuilder& builder,
               const Expression& expression, const Symbols& symbols,
               const llvm::ArrayRef<mlir::Value> arguments) {
  const auto resultType =
      expressionType(builder, expression.type, expression.width);
  switch (expression.kind) {
  case ExpressionKind::Value:
    switch (expression.type) {
    case ClassicalType::Bool:
      return builder.boolConstant(expression.boolValue);
    case ClassicalType::Uint:
      return integerConstant(builder, expression.width, expression.uintValue);
    case ClassicalType::Float:
      return builder.floatConstant(expression.floatValue);
    }
    break;
  case ExpressionKind::Variable: {
    const auto symbol = symbols.find(expression.variableName);
    if (symbol == symbols.end()) {
      throw std::runtime_error("unregistered Qiskit classical variable '" +
                               expression.variableName + "'");
    }
    return arguments[symbol->second.index];
  }
  case ExpressionKind::Cast: {
    const auto operand =
        emitExpression(builder, *expression.left, symbols, arguments);
    if (operand.getType() == resultType) {
      return operand;
    }
    if (const auto target = llvm::dyn_cast<mlir::IntegerType>(resultType)) {
      if (llvm::isa<mlir::IntegerType>(operand.getType())) {
        return castInteger(builder, operand, target);
      }
      if (operand.getType().isF64()) {
        return mlir::arith::FPToUIOp::create(builder, target, operand)
            .getResult();
      }
    }
    if (resultType.isF64() && llvm::isa<mlir::IntegerType>(operand.getType())) {
      return mlir::arith::UIToFPOp::create(builder, resultType, operand)
          .getResult();
    }
    throw std::runtime_error("unsupported Qiskit classical-expression cast");
  }
  case ExpressionKind::Unary: {
    const auto operand =
        emitExpression(builder, *expression.left, symbols, arguments);
    switch (expression.unaryOperation) {
    case UnaryOperation::BitNot: {
      const auto type = llvm::dyn_cast<mlir::IntegerType>(operand.getType());
      if (!type) {
        throw std::runtime_error(
            "Qiskit bitwise not requires an integer operand");
      }
      auto ones = mlir::arith::ConstantOp::create(
          builder, builder.getIntegerAttr(
                       type, llvm::APInt::getAllOnes(type.getWidth())));
      return mlir::arith::XOrIOp::create(builder, operand, ones.getResult())
          .getResult();
    }
    case UnaryOperation::LogicNot:
      if (!operand.getType().isInteger(1)) {
        throw std::runtime_error(
            "Qiskit logical not requires a Boolean operand");
      }
      return mlir::arith::XOrIOp::create(builder, operand,
                                         builder.boolConstant(true))
          .getResult();
    case UnaryOperation::Negate:
      if (operand.getType().isF64()) {
        return mlir::arith::NegFOp::create(builder, operand).getResult();
      }
      if (const auto type =
              llvm::dyn_cast<mlir::IntegerType>(operand.getType())) {
        return mlir::arith::SubIOp::create(
                   builder, integerConstant(builder, type.getWidth(), 0U),
                   operand)
            .getResult();
      }
      throw std::runtime_error(
          "Qiskit arithmetic negation has an unsupported type");
    }
    break;
  }
  case ExpressionKind::Binary: {
    auto left = emitExpression(builder, *expression.left, symbols, arguments);
    auto right = emitExpression(builder, *expression.right, symbols, arguments);
    const auto comparison = [&]() -> std::optional<mlir::Value> {
      std::optional<mlir::arith::CmpIPredicate> integerPredicate;
      std::optional<mlir::arith::CmpFPredicate> floatPredicate;
      switch (expression.binaryOperation) {
      case BinaryOperation::Equal:
        integerPredicate = mlir::arith::CmpIPredicate::eq;
        floatPredicate = mlir::arith::CmpFPredicate::OEQ;
        break;
      case BinaryOperation::NotEqual:
        integerPredicate = mlir::arith::CmpIPredicate::ne;
        floatPredicate = mlir::arith::CmpFPredicate::UNE;
        break;
      case BinaryOperation::Less:
        integerPredicate = mlir::arith::CmpIPredicate::ult;
        floatPredicate = mlir::arith::CmpFPredicate::OLT;
        break;
      case BinaryOperation::LessEqual:
        integerPredicate = mlir::arith::CmpIPredicate::ule;
        floatPredicate = mlir::arith::CmpFPredicate::OLE;
        break;
      case BinaryOperation::Greater:
        integerPredicate = mlir::arith::CmpIPredicate::ugt;
        floatPredicate = mlir::arith::CmpFPredicate::OGT;
        break;
      case BinaryOperation::GreaterEqual:
        integerPredicate = mlir::arith::CmpIPredicate::uge;
        floatPredicate = mlir::arith::CmpFPredicate::OGE;
        break;
      default:
        return std::nullopt;
      }
      if (left.getType().isF64() && right.getType().isF64()) {
        return mlir::arith::CmpFOp::create(builder, *floatPredicate, left,
                                           right)
            .getResult();
      }
      if (left.getType() != right.getType() ||
          !llvm::isa<mlir::IntegerType>(left.getType())) {
        throw std::runtime_error(
            "Qiskit classical comparison has incompatible operand types");
      }
      return mlir::arith::CmpIOp::create(builder, *integerPredicate, left,
                                         right)
          .getResult();
    }();
    if (comparison) {
      return *comparison;
    }
    if (left.getType().isF64() && right.getType().isF64()) {
      switch (expression.binaryOperation) {
      case BinaryOperation::Add:
        return mlir::arith::AddFOp::create(builder, left, right).getResult();
      case BinaryOperation::Subtract:
        return mlir::arith::SubFOp::create(builder, left, right).getResult();
      case BinaryOperation::Multiply:
        return mlir::arith::MulFOp::create(builder, left, right).getResult();
      case BinaryOperation::Divide:
        return mlir::arith::DivFOp::create(builder, left, right).getResult();
      default:
        throw std::runtime_error(
            "unsupported floating-point Qiskit classical operation");
      }
    }
    const auto integerType = llvm::dyn_cast<mlir::IntegerType>(left.getType());
    if (!integerType) {
      throw std::runtime_error(
          "Qiskit classical operation requires integer operands");
    }
    right = castInteger(builder, right, integerType);
    switch (expression.binaryOperation) {
    case BinaryOperation::BitAnd:
    case BinaryOperation::LogicAnd:
      return mlir::arith::AndIOp::create(builder, left, right).getResult();
    case BinaryOperation::BitOr:
    case BinaryOperation::LogicOr:
      return mlir::arith::OrIOp::create(builder, left, right).getResult();
    case BinaryOperation::BitXor:
      return mlir::arith::XOrIOp::create(builder, left, right).getResult();
    case BinaryOperation::ShiftLeft:
      return mlir::arith::ShLIOp::create(builder, left, right).getResult();
    case BinaryOperation::ShiftRight:
      return mlir::arith::ShRUIOp::create(builder, left, right).getResult();
    case BinaryOperation::Add:
      return mlir::arith::AddIOp::create(builder, left, right).getResult();
    case BinaryOperation::Subtract:
      return mlir::arith::SubIOp::create(builder, left, right).getResult();
    case BinaryOperation::Multiply:
      return mlir::arith::MulIOp::create(builder, left, right).getResult();
    case BinaryOperation::Divide:
      return mlir::arith::DivUIOp::create(builder, left, right).getResult();
    default:
      break;
    }
    throw std::runtime_error("unsupported Qiskit classical binary operation");
  }
  case ExpressionKind::Index: {
    const auto target =
        emitExpression(builder, *expression.left, symbols, arguments);
    auto index = emitExpression(builder, *expression.right, symbols, arguments);
    const auto targetType = llvm::dyn_cast<mlir::IntegerType>(target.getType());
    if (!targetType) {
      throw std::runtime_error(
          "Qiskit index expressions require a Uint target");
    }
    index = castInteger(builder, index, targetType);
    const auto shifted =
        mlir::arith::ShRUIOp::create(builder, target, index).getResult();
    return castInteger(builder, shifted,
                       llvm::cast<mlir::IntegerType>(resultType));
  }
  }
  throw std::runtime_error("unsupported normalized Qiskit expression");
}

[[nodiscard]] mlir::Value loadClassicalBit(mlir::qc::QCProgramBuilder& builder,
                                           const mlir::Value classicalStorage,
                                           const std::uint32_t index) {
  if (!classicalStorage) {
    throw std::runtime_error(
        "Qiskit control flow references a circuit without classical bits");
  }
  auto position = mlir::arith::ConstantIndexOp::create(builder, index);
  return mlir::memref::LoadOp::create(builder, classicalStorage,
                                      mlir::ValueRange{position.getResult()})
      .getResult();
}

[[nodiscard]] mlir::Value packRegister(mlir::qc::QCProgramBuilder& builder,
                                       const mlir::Value classicalStorage,
                                       const Register& reg) {
  if (reg.bits.empty() || reg.bits.size() > 64U) {
    throw std::runtime_error(
        "Qiskit classical registers must contain between 1 and 64 bits");
  }
  const auto width = static_cast<std::uint32_t>(reg.bits.size());
  const auto type = builder.getIntegerType(width);
  auto packed = integerConstant(builder, width, 0U);
  for (std::size_t index = 0; index < reg.bits.size(); ++index) {
    auto bit = castInteger(
        builder, loadClassicalBit(builder, classicalStorage, reg.bits[index]),
        type);
    if (index != 0U) {
      bit = mlir::arith::ShLIOp::create(builder, bit,
                                        integerConstant(builder, width, index))
                .getResult();
    }
    packed = mlir::arith::OrIOp::create(builder, packed, bit).getResult();
  }
  return packed;
}

[[nodiscard]] mlir::Value
emitCondition(mlir::qc::QCProgramBuilder& builder,
              const ClassicalTarget& target, const mlir::Value classicalStorage,
              const Symbols& symbols,
              const llvm::ArrayRef<mlir::Value> arguments) {
  switch (target.kind) {
  case ClassicalTargetKind::ClassicalBit: {
    const auto actual = loadClassicalBit(builder, classicalStorage, target.bit);
    return mlir::arith::CmpIOp::create(builder, mlir::arith::CmpIPredicate::eq,
                                       actual,
                                       builder.boolConstant(target.expectedBit))
        .getResult();
  }
  case ClassicalTargetKind::ClassicalRegister: {
    const auto actual = packRegister(builder, classicalStorage, target.reg);
    const auto expected =
        integerConstant(builder, target.width, target.expectedRegister);
    return mlir::arith::CmpIOp::create(builder, mlir::arith::CmpIPredicate::eq,
                                       actual, expected)
        .getResult();
  }
  case ClassicalTargetKind::Expression: {
    const auto condition =
        emitExpression(builder, *target.expression, symbols, arguments);
    if (!condition.getType().isInteger(1)) {
      throw std::runtime_error(
          "Qiskit control-flow condition expression must have Boolean type");
    }
    return condition;
  }
  }
  throw std::runtime_error("unknown normalized Qiskit condition type");
}

[[nodiscard]] mlir::Value
emitSwitchTarget(mlir::qc::QCProgramBuilder& builder,
                 const ClassicalTarget& target,
                 const mlir::Value classicalStorage, const Symbols& symbols,
                 const llvm::ArrayRef<mlir::Value> arguments) {
  mlir::Value value;
  switch (target.kind) {
  case ClassicalTargetKind::ClassicalBit:
    value = loadClassicalBit(builder, classicalStorage, target.bit);
    break;
  case ClassicalTargetKind::ClassicalRegister:
    value = packRegister(builder, classicalStorage, target.reg);
    break;
  case ClassicalTargetKind::Expression:
    value = emitExpression(builder, *target.expression, symbols, arguments);
    break;
  }
  if (!llvm::isa<mlir::IntegerType>(value.getType())) {
    throw std::runtime_error("Qiskit switch targets must be Boolean or Uint");
  }
  return mlir::arith::IndexCastOp::create(builder, builder.getIndexType(),
                                          value)
      .getResult();
}

void translateCircuit(mlir::qc::QCProgramBuilder& builder,
                      const CircuitView& circuit,
                      llvm::ArrayRef<std::uint32_t> qubitMap,
                      llvm::ArrayRef<std::uint32_t> clbitMap,
                      llvm::ArrayRef<mlir::Value> allQubits,
                      mlir::Value classicalStorage, const Symbols& symbols,
                      llvm::ArrayRef<mlir::Value> arguments,
                      LocalParameters localParameters);

[[nodiscard]] std::int64_t rangeLength(const Loop& loop) {
  if (loop.step == 0) {
    throw std::runtime_error("Qiskit for-loop range has a zero step");
  }
  if ((loop.step > 0 && loop.start >= loop.stop) ||
      (loop.step < 0 && loop.start <= loop.stop)) {
    return 0;
  }
  const auto distance = loop.step > 0
                            ? static_cast<std::uint64_t>(loop.stop) -
                                  static_cast<std::uint64_t>(loop.start)
                            : static_cast<std::uint64_t>(loop.start) -
                                  static_cast<std::uint64_t>(loop.stop);
  const auto magnitude =
      loop.step > 0 ? static_cast<std::uint64_t>(loop.step)
                    : std::uint64_t{0} - static_cast<std::uint64_t>(loop.step);
  const auto count = (distance - 1U) / magnitude + 1U;
  if (count >
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::runtime_error(
        "Qiskit for-loop range is too large to represent safely");
  }
  return static_cast<std::int64_t>(count);
}

void requireExactLoopParameter(const std::int64_t value) {
  constexpr std::int64_t MAX_EXACT_DOUBLE_INTEGER = std::int64_t{1} << 53U;
  if (value < -MAX_EXACT_DOUBLE_INTEGER || value > MAX_EXACT_DOUBLE_INTEGER) {
    throw std::runtime_error(
        "Qiskit loop parameter integer cannot be represented exactly as f64");
  }
}

[[nodiscard]] mlir::Value
loopParameterValue(mlir::qc::QCProgramBuilder& builder,
                   const mlir::Value iteration, const Loop& loop) {
  const auto counter =
      mlir::arith::IndexCastOp::create(builder, builder.getI64Type(), iteration)
          .getResult();
  const auto offset = mlir::arith::MulIOp::create(
                          builder, counter, builder.intConstant(loop.step))
                          .getResult();
  const auto value = mlir::arith::AddIOp::create(
                         builder, builder.intConstant(loop.start), offset)
                         .getResult();
  return mlir::arith::SIToFPOp::create(builder, builder.getF64Type(), value)
      .getResult();
}

void translateControlFlow(mlir::qc::QCProgramBuilder& builder,
                          const ControlFlowView& controlFlow,
                          llvm::ArrayRef<mlir::Value> allQubits,
                          mlir::Value classicalStorage, const Symbols& symbols,
                          llvm::ArrayRef<mlir::Value> arguments,
                          const LocalParameters& localParameters) {
  const auto qubitMap = controlFlow.qubitMap();
  const auto clbitMap = controlFlow.clbitMap();
  const auto translateBlock = [&](const CircuitView& block,
                                  LocalParameters parameters) {
    translateCircuit(builder, block, qubitMap, clbitMap, allQubits,
                     classicalStorage, symbols, arguments,
                     std::move(parameters));
  };

  switch (controlFlow.kind()) {
  case ControlFlowKind::Box:
    throw std::runtime_error("Qiskit box instructions are not supported");
  case ControlFlowKind::Break:
    throw std::runtime_error("Qiskit break instructions are not supported");
  case ControlFlowKind::Continue:
    throw std::runtime_error("Qiskit continue instructions are not supported");
  case ControlFlowKind::IfElse: {
    if (controlFlow.numBlocks() < 1U || controlFlow.numBlocks() > 2U) {
      throw std::runtime_error(
          "Qiskit if/else has an invalid number of blocks");
    }
    const auto condition = controlFlow.condition();
    const auto value =
        emitCondition(builder, condition, classicalStorage, symbols, arguments);
    const auto thenBlock = controlFlow.block(0);
    if (controlFlow.numBlocks() == 1U) {
      builder.scfIf(value,
                    [&] { translateBlock(*thenBlock, localParameters); });
      return;
    }
    const auto elseBlock = controlFlow.block(1);
    builder.scfIf(
        value, [&] { translateBlock(*thenBlock, localParameters); },
        [&] { translateBlock(*elseBlock, localParameters); });
    return;
  }
  case ControlFlowKind::While: {
    if (controlFlow.numBlocks() != 1U) {
      throw std::runtime_error("Qiskit while loop has an invalid block count");
    }
    const auto condition = controlFlow.condition();
    const auto body = controlFlow.block(0);
    builder.scfWhile(
        [&] {
          builder.scfCondition(emitCondition(
              builder, condition, classicalStorage, symbols, arguments));
        },
        [&] { translateBlock(*body, localParameters); });
    return;
  }
  case ControlFlowKind::For: {
    if (controlFlow.numBlocks() != 1U) {
      throw std::runtime_error("Qiskit for loop has an invalid block count");
    }
    const auto loop = controlFlow.loop();
    const auto body = controlFlow.block(0);
    if (!loop.isRange) {
      for (const auto value : loop.values) {
        auto parameters = localParameters;
        if (loop.parameter) {
          requireExactLoopParameter(value);
          parameters[*loop.parameter] =
              builder.floatConstant(static_cast<double>(value));
        }
        translateBlock(*body, std::move(parameters));
      }
      return;
    }
    const auto count = rangeLength(loop);
    if (loop.parameter && count != 0) {
      requireExactLoopParameter(loop.start);
      requireExactLoopParameter(loop.step > 0 ? loop.stop - 1 : loop.stop + 1);
    }
    builder.scfFor(0, count, 1, [&](const mlir::Value iteration) {
      auto parameters = localParameters;
      if (loop.parameter) {
        parameters[*loop.parameter] =
            loopParameterValue(builder, iteration, loop);
      }
      translateBlock(*body, std::move(parameters));
    });
    return;
  }
  case ControlFlowKind::Switch: {
    const auto cases = controlFlow.switchCases();
    if (cases.size() != controlFlow.numBlocks()) {
      throw std::runtime_error(
          "Qiskit switch case metadata does not match its blocks");
    }
    const auto target = emitSwitchTarget(builder, controlFlow.switchTarget(),
                                         classicalStorage, symbols, arguments);
    std::vector<std::unique_ptr<CircuitView>> blocks;
    blocks.reserve(controlFlow.numBlocks());
    for (std::size_t index = 0; index < controlFlow.numBlocks(); ++index) {
      blocks.push_back(controlFlow.block(index));
    }
    llvm::SmallVector<std::int64_t> labels;
    llvm::SmallVector<std::function<void()>> ownedBodies;
    std::optional<std::size_t> defaultBlock;
    for (std::size_t caseIndex = 0; caseIndex < cases.size(); ++caseIndex) {
      if (cases[caseIndex].isDefault) {
        if (defaultBlock) {
          throw std::runtime_error(
              "Qiskit switch has more than one default case");
        }
        defaultBlock = caseIndex;
      }
      for (const auto label : cases[caseIndex].labels) {
        if (label > static_cast<std::uint64_t>(
                        std::numeric_limits<std::int64_t>::max())) {
          throw std::runtime_error("Qiskit switch label cannot be represented "
                                   "safely by scf.index_switch");
        }
        labels.push_back(static_cast<std::int64_t>(label));
        ownedBodies.emplace_back([&, caseIndex] {
          translateBlock(*blocks[caseIndex], localParameters);
        });
      }
    }
    llvm::SmallVector<llvm::function_ref<void()>> bodies;
    bodies.reserve(ownedBodies.size());
    for (auto& body : ownedBodies) {
      bodies.emplace_back(body);
    }
    std::function<void()> ownedDefault = [&] {
      if (defaultBlock) {
        translateBlock(*blocks[*defaultBlock], localParameters);
      }
    };
    const llvm::function_ref<void()> defaultBody(ownedDefault);
    builder.scfIndexSwitch(target, labels, bodies, defaultBody);
    return;
  }
  }
}

void translateCircuit(mlir::qc::QCProgramBuilder& builder,
                      const CircuitView& circuit,
                      const llvm::ArrayRef<std::uint32_t> qubitMap,
                      const llvm::ArrayRef<std::uint32_t> clbitMap,
                      const llvm::ArrayRef<mlir::Value> allQubits,
                      const mlir::Value classicalStorage,
                      const Symbols& symbols,
                      const llvm::ArrayRef<mlir::Value> arguments,
                      LocalParameters localParameters) {
  builder.gphase(parameterValue(circuit.globalPhase(), symbols, arguments,
                                localParameters));
  const auto getQubit = [&](const std::uint32_t local) {
    if (local >= qubitMap.size() || qubitMap[local] >= allQubits.size()) {
      throw std::runtime_error(
          "Qiskit instruction references an invalid mapped qubit");
    }
    return allQubits[qubitMap[local]];
  };
  const auto getClbit = [&](const std::uint32_t local) {
    if (local >= clbitMap.size()) {
      throw std::runtime_error(
          "Qiskit instruction references an invalid mapped classical bit");
    }
    return clbitMap[local];
  };

  for (std::size_t index = 0; index < circuit.numInstructions(); ++index) {
    const auto instruction = circuit.instruction(index);
    switch (instruction.kind) {
    case OperationKind::Gate:
      emitGate(builder, instruction, allQubits, qubitMap, symbols, arguments,
               localParameters);
      break;
    case OperationKind::Barrier: {
      llvm::SmallVector<mlir::Value> operands;
      for (const auto qubit : instruction.qubits) {
        operands.push_back(getQubit(qubit));
      }
      builder.barrier(operands);
      break;
    }
    case OperationKind::Measure:
      requireArity(instruction, 1, 0);
      if (instruction.clbits.size() != 1U || !classicalStorage) {
        throw std::runtime_error(
            "Qiskit measurement has an invalid classical destination");
      }
      builder.measure(
          getQubit(instruction.qubits[0]), classicalStorage,
          static_cast<std::int64_t>(getClbit(instruction.clbits[0])));
      break;
    case OperationKind::Reset:
      requireArity(instruction, 1, 0);
      builder.reset(getQubit(instruction.qubits[0]));
      break;
    case OperationKind::Unitary: {
      llvm::SmallVector<mlir::Value> operands;
      for (const auto qubit : instruction.qubits) {
        operands.push_back(getQubit(qubit));
      }
      if (operands.size() >=
          static_cast<std::size_t>(std::numeric_limits<std::int64_t>::digits)) {
        throw std::runtime_error("Qiskit unitary is too large to represent");
      }
      const auto dimension = std::int64_t{1} << operands.size();
      const auto type = mlir::RankedTensorType::get(
          {dimension, dimension}, mlir::ComplexType::get(builder.getF64Type()));
      const auto values = circuit.unitary(index);
      builder.unitary(operands,
                      mlir::DenseElementsAttr::get(
                          type, llvm::ArrayRef<std::complex<double>>(values)));
      break;
    }
    case OperationKind::ControlFlow: {
      const auto controlFlow = circuit.controlFlow(index);
      translateControlFlow(builder, *controlFlow, allQubits, classicalStorage,
                           symbols, arguments, localParameters);
      break;
    }
    case OperationKind::Delay:
      throw std::runtime_error("Qiskit delay instructions are not supported");
    case OperationKind::Unknown:
      throw std::runtime_error("unsupported custom Qiskit instruction '" +
                               instruction.name + "'");
    }
  }
}

struct ExportedInstruction {
  enum class Kind : std::uint8_t { Gate, Measure, Reset, Barrier, Unitary };
  Kind kind = Kind::Gate;
  std::string name;
  std::vector<std::uint32_t> qubits;
  std::vector<std::uint32_t> clbits;
  std::vector<Parameter> parameters;
  std::vector<std::complex<double>> matrix;
};

[[nodiscard]] Parameter exportParameter(const mlir::Value value,
                                        mlir::func::FuncOp function) {
  if (const auto number = mlir::utils::valueToDouble(value)) {
    if (!std::isfinite(*number)) {
      throw std::runtime_error("cannot export a non-finite QC parameter");
    }
    return {.kind = ParameterKind::Number, .number = *number};
  }
  const auto argument = llvm::dyn_cast<mlir::BlockArgument>(value);
  if (!argument || argument.getOwner() != &function.getBody().front()) {
    throw std::runtime_error("QC to Qiskit export supports only numeric and "
                             "entry-argument parameters");
  }
  const auto name = function.getArgAttrOfType<mlir::StringAttr>(
      argument.getArgNumber(), PARAMETER_NAME_ATTR);
  if (!name) {
    throw std::runtime_error(
        "QC entry parameter is missing Qiskit symbol metadata");
  }
  return {.kind = ParameterKind::Symbol, .text = name.str()};
}

[[nodiscard]] std::uint32_t checkedIndex(const std::int64_t index,
                                         const std::string_view kind) {
  if (index < 0 || static_cast<std::uint64_t>(index) >
                       std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error(std::string(kind) +
                             " index cannot be represented by Qiskit");
  }
  return static_cast<std::uint32_t>(index);
}

[[nodiscard]] std::uint32_t checkedAdd(const std::uint32_t left,
                                       const std::uint32_t right,
                                       const std::string_view kind) {
  if (right > std::numeric_limits<std::uint32_t>::max() - left) {
    throw std::runtime_error(std::string(kind) +
                             " count cannot be represented by Qiskit");
  }
  return left + right;
}

struct ExportState {
  llvm::DenseMap<mlir::Value, std::uint32_t> qubits;
  llvm::DenseMap<mlir::Value, std::uint32_t> quantumBases;
  llvm::DenseMap<mlir::Value, std::uint32_t> quantumSizes;
  llvm::DenseMap<mlir::Value, std::uint32_t> classicalBases;
  llvm::DenseMap<mlir::Value, std::uint32_t> classicalSizes;
  std::vector<ExportedInstruction> instructions;
  Parameter globalPhase;
  std::uint32_t numQubits = 0;
  std::uint32_t numClbits = 0;
};

[[nodiscard]] std::vector<std::uint32_t>
mapQubits(const mlir::ValueRange values,
          const llvm::DenseMap<mlir::Value, std::uint32_t>& qubits) {
  std::vector<std::uint32_t> result;
  result.reserve(values.size());
  for (const auto value : values) {
    const auto found = qubits.find(value);
    if (found == qubits.end()) {
      throw std::runtime_error(
          "QC to Qiskit export could not resolve a qubit operand");
    }
    result.push_back(found->second);
  }
  return result;
}

[[nodiscard]] ExportedInstruction collectUnitaryInstruction(
    mlir::Operation& operation,
    const llvm::DenseMap<mlir::Value, std::uint32_t>& qubits,
    mlir::func::FuncOp function);

[[nodiscard]] std::vector<mlir::Operation*>
modifierBodyOperations(mlir::Region& region) {
  if (!llvm::hasSingleElement(region)) {
    throw std::runtime_error(
        "QC to Qiskit export requires single-block modifier regions");
  }
  std::vector<mlir::Operation*> operations;
  for (auto& operation : region.front()) {
    if (!llvm::isa<mlir::qc::YieldOp, mlir::arith::ConstantOp>(operation)) {
      operations.push_back(&operation);
    }
  }
  return operations;
}

[[nodiscard]] llvm::DenseMap<mlir::Value, std::uint32_t>
modifierQubitMap(const llvm::DenseMap<mlir::Value, std::uint32_t>& outer,
                 mlir::Block& block, mlir::ValueRange operands) {
  if (block.getNumArguments() != operands.size()) {
    throw std::runtime_error(
        "QC modifier block arguments do not match its qubit operands");
  }
  auto result = outer;
  for (const auto [argument, operand] :
       llvm::zip_equal(block.getArguments(), operands)) {
    const auto mapped = outer.find(operand);
    if (mapped == outer.end()) {
      throw std::runtime_error(
          "QC to Qiskit export could not resolve a modifier qubit");
    }
    result[argument] = mapped->second;
  }
  return result;
}

[[nodiscard]] std::string controlledGateName(const std::string_view base,
                                             const std::size_t controls) {
  if (controls == 1U) {
    static const llvm::StringMap<std::string> names = {
        {"h", "ch"}, {"x", "cx"},     {"y", "cy"},   {"z", "cz"},
        {"p", "cp"}, {"rx", "crx"},   {"ry", "cry"}, {"rz", "crz"},
        {"s", "cs"}, {"sdg", "csdg"}, {"sx", "csx"}, {"swap", "cswap"},
    };
    if (const auto name = names.find(base); name != names.end()) {
      return name->second;
    }
  }
  if (controls == 2U && (base == "x" || base == "z")) {
    return "cc" + std::string(base);
  }
  if (controls == 3U && base == "x") {
    return "mcx";
  }
  if (controls == 3U && base == "sx") {
    return "c3sx";
  }
  if (controls == 1U && base == "u") {
    return "cu3";
  }
  throw std::runtime_error("QC control modifier has no QkCircuit standard-gate "
                           "equivalent");
}

void invertGate(ExportedInstruction& instruction) {
  static const llvm::StringMap<std::string> inverseNames = {
      {"id", "id"},   {"x", "x"},     {"y", "y"},         {"z", "z"},
      {"h", "h"},     {"s", "sdg"},   {"sdg", "s"},       {"t", "tdg"},
      {"tdg", "t"},   {"sx", "sxdg"}, {"sxdg", "sx"},     {"swap", "swap"},
      {"cx", "cx"},   {"cy", "cy"},   {"cz", "cz"},       {"ch", "ch"},
      {"ccx", "ccx"}, {"ccz", "ccz"}, {"cswap", "cswap"}, {"ecr", "ecr"},
  };
  if (const auto inverse = inverseNames.find(instruction.name);
      inverse != inverseNames.end()) {
    instruction.name = inverse->second;
    return;
  }
  static const llvm::StringSet<> negateFirst = {
      "p",   "rx",  "ry",  "rz",  "rxx", "ryy",
      "rzz", "rzx", "crx", "cry", "crz", "cp",
  };
  if (negateFirst.contains(instruction.name) &&
      !instruction.parameters.empty()) {
    auto& parameter = instruction.parameters.front();
    if (parameter.kind != ParameterKind::Number) {
      throw std::runtime_error(
          "QC inverse modifier would require a composite Qiskit parameter");
    }
    parameter.number = -parameter.number;
    return;
  }
  if ((instruction.name == "u" || instruction.name == "u3") &&
      instruction.parameters.size() == 3U) {
    std::array<double, 3> values{};
    for (std::size_t index = 0; index < values.size(); ++index) {
      if (instruction.parameters[index].kind != ParameterKind::Number) {
        throw std::runtime_error(
            "QC inverse modifier would require composite Qiskit parameters");
      }
      values[index] = instruction.parameters[index].number;
    }
    instruction.parameters = {
        {.kind = ParameterKind::Number, .number = -values[0]},
        {.kind = ParameterKind::Number, .number = -values[2]},
        {.kind = ParameterKind::Number, .number = -values[1]},
    };
    return;
  }
  throw std::runtime_error(
      "QC inverse modifier has no supported QkCircuit gate equivalent");
}

[[nodiscard]] ExportedInstruction collectUnitaryInstruction(
    mlir::Operation& operation,
    const llvm::DenseMap<mlir::Value, std::uint32_t>& qubits,
    const mlir::func::FuncOp function) {
  if (auto control = llvm::dyn_cast<mlir::qc::CtrlOp>(operation)) {
    auto bodyOperations = modifierBodyOperations(control.getRegion());
    const auto controls = mapQubits(control.getControls(), qubits);
    auto nestedMap = modifierQubitMap(qubits, control.getRegion().front(),
                                      control.getTargets());
    if (controls.size() == 1U && bodyOperations.size() == 2U &&
        llvm::isa<mlir::qc::GPhaseOp>(*bodyOperations[0]) &&
        llvm::isa<mlir::qc::UOp>(*bodyOperations[1])) {
      auto phase = llvm::cast<mlir::qc::GPhaseOp>(*bodyOperations[0]);
      auto unitary = llvm::cast<mlir::qc::UOp>(*bodyOperations[1]);
      auto targets = mapQubits(unitary.getTargets(), nestedMap);
      if (targets.size() != 1U) {
        throw std::runtime_error("QC controlled-U modifier has invalid arity");
      }
      ExportedInstruction result{.kind = ExportedInstruction::Kind::Gate,
                                 .name = "cu",
                                 .qubits = {controls.front(), targets.front()}};
      for (const auto parameter : unitary.getParameters()) {
        result.parameters.push_back(exportParameter(parameter, function));
      }
      result.parameters.push_back(exportParameter(phase.getTheta(), function));
      return result;
    }
    if (bodyOperations.size() != 1U) {
      throw std::runtime_error(
          "QC control export requires one standard gate in the modifier body");
    }
    auto result =
        collectUnitaryInstruction(*bodyOperations.front(), nestedMap, function);
    result.name = controlledGateName(result.name, controls.size());
    result.qubits.insert(result.qubits.begin(), controls.begin(),
                         controls.end());
    return result;
  }
  if (auto inverse = llvm::dyn_cast<mlir::qc::InvOp>(operation)) {
    auto bodyOperations = modifierBodyOperations(inverse.getRegion());
    if (bodyOperations.size() != 1U) {
      throw std::runtime_error(
          "QC inverse export requires one standard gate in the modifier body");
    }
    auto nestedMap = modifierQubitMap(qubits, inverse.getRegion().front(),
                                      inverse.getQubits());
    auto result =
        collectUnitaryInstruction(*bodyOperations.front(), nestedMap, function);
    invertGate(result);
    return result;
  }
  if (auto power = llvm::dyn_cast<mlir::qc::PowOp>(operation)) {
    auto bodyOperations = modifierBodyOperations(power.getRegion());
    if (bodyOperations.size() != 1U) {
      throw std::runtime_error(
          "QC power export requires one standard gate in the modifier body");
    }
    const auto exponent = exportParameter(power.getExponent(), function);
    if (exponent.kind != ParameterKind::Number ||
        (exponent.number != 1.0 && exponent.number != -1.0)) {
      throw std::runtime_error(
          "QC power export supports only constant exponents 1 and -1");
    }
    auto nestedMap =
        modifierQubitMap(qubits, power.getRegion().front(), power.getQubits());
    auto result =
        collectUnitaryInstruction(*bodyOperations.front(), nestedMap, function);
    if (exponent.number == -1.0) {
      invertGate(result);
    }
    return result;
  }
  auto gate = llvm::dyn_cast<mlir::qc::UnitaryOpInterface>(operation);
  if (!gate ||
      llvm::isa<mlir::qc::GPhaseOp, mlir::qc::BarrierOp, mlir::qc::UnitaryOp>(
          operation)) {
    throw std::runtime_error(
        "QC modifier body is not a constructible standard Qiskit gate");
  }
  ExportedInstruction result{.kind = ExportedInstruction::Kind::Gate,
                             .name = gate.getBaseSymbol().str(),
                             .qubits = mapQubits(gate.getTargets(), qubits)};
  for (const auto parameter : gate.getParameters()) {
    result.parameters.push_back(exportParameter(parameter, function));
  }
  return result;
}

void collectResources(mlir::func::FuncOp function, ExportState& state) {
  for (auto& operation : function.getBody().front()) {
    if (auto staticQubit = llvm::dyn_cast<mlir::qc::StaticOp>(operation)) {
      const auto index = checkedIndex(staticQubit.getIndex(), "qubit");
      if (index == std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("qubit count cannot be represented by Qiskit");
      }
      state.qubits[staticQubit.getQubit()] = index;
      state.numQubits = std::max(state.numQubits, index + 1U);
    }
  }
  for (auto& operation : function.getBody().front()) {
    if (auto alloc = llvm::dyn_cast<mlir::qc::AllocOp>(operation)) {
      state.qubits[alloc.getResult()] = state.numQubits;
      state.numQubits = checkedAdd(state.numQubits, 1U, "qubit");
      continue;
    }
    auto alloc = llvm::dyn_cast<mlir::memref::AllocOp>(operation);
    if (!alloc) {
      continue;
    }
    const auto type = alloc.getType();
    if (type.getRank() != 1 || type.isDynamicDim(0)) {
      continue;
    }
    if (llvm::isa<mlir::qc::QubitType>(type.getElementType())) {
      const auto size = checkedIndex(type.getShape()[0], "qubit-register size");
      state.quantumBases[alloc.getResult()] = state.numQubits;
      state.quantumSizes[alloc.getResult()] = size;
      state.numQubits = checkedAdd(state.numQubits, size, "qubit");
    } else if (type.getElementType().isInteger(1)) {
      const auto size =
          checkedIndex(type.getShape()[0], "classical-register size");
      state.classicalBases[alloc.getResult()] = state.numClbits;
      state.classicalSizes[alloc.getResult()] = size;
      state.numClbits = checkedAdd(state.numClbits, size, "classical-bit");
    }
  }
  for (auto& operation : function.getBody().front()) {
    auto load = llvm::dyn_cast<mlir::memref::LoadOp>(operation);
    if (!load || !llvm::isa<mlir::qc::QubitType>(load.getResult().getType()) ||
        load.getIndices().size() != 1U) {
      continue;
    }
    const auto index = mlir::getConstantIntValue(load.getIndices().front());
    if (!index) {
      throw std::runtime_error(
          "QC to Qiskit export does not support dynamic qubit indices");
    }
    const auto base = state.quantumBases.find(load.getMemRef());
    if (base == state.quantumBases.end()) {
      throw std::runtime_error(
          "QC to Qiskit export could not resolve a qubit-register allocation");
    }
    const auto size = state.quantumSizes.find(load.getMemRef());
    const auto checked = checkedIndex(*index, "qubit");
    if (size == state.quantumSizes.end() || checked >= size->second) {
      throw std::runtime_error(
          "QC to Qiskit export encountered an out-of-bounds qubit index");
    }
    state.qubits[load.getResult()] = checkedAdd(base->second, checked, "qubit");
  }
}

void collectFlatInstructions(mlir::func::FuncOp function, ExportState& state) {
  for (auto& operation : function.getBody().front()) {
    if (llvm::isa<mlir::arith::ConstantOp, mlir::memref::AllocOp,
                  mlir::memref::LoadOp, mlir::memref::DeallocOp,
                  mlir::qc::AllocOp, mlir::qc::DeallocOp, mlir::qc::StaticOp,
                  mlir::func::ReturnOp>(operation)) {
      continue;
    }
    if (auto store = llvm::dyn_cast<mlir::memref::StoreOp>(operation)) {
      if (llvm::isa_and_nonnull<mlir::qc::MeasureOp>(
              store.getValueToStore().getDefiningOp())) {
        continue;
      }
      throw std::runtime_error(
          "QC to Qiskit export does not support classical execution");
    }
    if (auto phase = llvm::dyn_cast<mlir::qc::GPhaseOp>(operation)) {
      if (state.globalPhase.kind != ParameterKind::Number ||
          state.globalPhase.number != 0.0) {
        throw std::runtime_error(
            "QC to Qiskit export requires normalized global phase");
      }
      state.globalPhase = exportParameter(phase.getTheta(), function);
      continue;
    }
    if (auto measure = llvm::dyn_cast<mlir::qc::MeasureOp>(operation)) {
      mlir::memref::StoreOp destination;
      for (auto& use : measure.getResult().getUses()) {
        if (const auto store =
                llvm::dyn_cast<mlir::memref::StoreOp>(use.getOwner())) {
          if (destination) {
            throw std::runtime_error(
                "QC measurement has more than one classical destination");
          }
          destination = store;
        }
      }
      if (!destination || destination.getIndices().size() != 1U) {
        throw std::runtime_error(
            "QC measurement is missing a static classical destination");
      }
      const auto base = state.classicalBases.find(destination.getMemref());
      const auto index =
          mlir::getConstantIntValue(destination.getIndices().front());
      if (base == state.classicalBases.end() || !index) {
        throw std::runtime_error(
            "QC measurement uses an unsupported classical destination");
      }
      const auto size = state.classicalSizes.find(destination.getMemref());
      const auto checked = checkedIndex(*index, "classical-bit");
      if (size == state.classicalSizes.end() || checked >= size->second) {
        throw std::runtime_error(
            "QC measurement uses an out-of-bounds classical destination");
      }
      state.instructions.push_back(
          {.kind = ExportedInstruction::Kind::Measure,
           .qubits = mapQubits(measure.getQubit(), state.qubits),
           .clbits = {checkedAdd(base->second, checked, "classical-bit")}});
      continue;
    }
    if (auto reset = llvm::dyn_cast<mlir::qc::ResetOp>(operation)) {
      state.instructions.push_back(
          {.kind = ExportedInstruction::Kind::Reset,
           .qubits = mapQubits(reset.getQubit(), state.qubits)});
      continue;
    }
    if (auto barrier = llvm::dyn_cast<mlir::qc::BarrierOp>(operation)) {
      state.instructions.push_back(
          {.kind = ExportedInstruction::Kind::Barrier,
           .qubits = mapQubits(barrier.getQubits(), state.qubits)});
      continue;
    }
    if (auto unitary = llvm::dyn_cast<mlir::qc::UnitaryOp>(operation)) {
      const auto matrix =
          llvm::cast<mlir::DenseElementsAttr>(unitary.getMatrix());
      std::vector<std::complex<double>> values;
      values.reserve(matrix.size());
      llvm::append_range(values, matrix.getValues<std::complex<double>>());
      state.instructions.push_back(
          {.kind = ExportedInstruction::Kind::Unitary,
           .qubits = mapQubits(unitary.getQubits(), state.qubits),
           .matrix = std::move(values)});
      continue;
    }
    if (llvm::isa<mlir::scf::IfOp, mlir::scf::WhileOp, mlir::scf::ForOp,
                  mlir::scf::IndexSwitchOp>(operation)) {
      throw std::runtime_error(
          "QC to Qiskit export cannot construct structured control flow "
          "through the Qiskit 2.5 C API");
    }
    if (llvm::isa<mlir::qc::UnitaryOpInterface>(operation)) {
      state.instructions.push_back(
          collectUnitaryInstruction(operation, state.qubits, function));
      continue;
    }
    throw std::runtime_error("unsupported QC operation in Qiskit export: " +
                             operation.getName().getStringRef().str());
  }
}

[[nodiscard]] std::vector<Register>
readRegisterMetadata(mlir::ModuleOp module, const llvm::StringRef name) {
  const auto attributes = module->getAttrOfType<mlir::ArrayAttr>(name);
  if (!attributes) {
    return {};
  }
  std::vector<Register> result;
  result.reserve(attributes.size());
  for (const auto attribute : attributes) {
    const auto dictionary = llvm::dyn_cast<mlir::DictionaryAttr>(attribute);
    const auto registerName =
        dictionary ? dictionary.getAs<mlir::StringAttr>("name") : nullptr;
    const auto bits = dictionary
                          ? dictionary.getAs<mlir::DenseI32ArrayAttr>("bits")
                          : nullptr;
    if (!dictionary || !registerName || !bits) {
      throw std::runtime_error("invalid Qiskit register metadata in QC module");
    }
    Register reg{.name = registerName.str()};
    reg.bits.reserve(bits.size());
    for (const auto bit : bits.asArrayRef()) {
      reg.bits.push_back(checkedIndex(bit, "register bit"));
    }
    result.push_back(std::move(reg));
  }
  return result;
}

[[nodiscard]] std::uint32_t
validateRegisterLayout(const std::vector<Register>& registers,
                       const std::uint32_t total, const std::string_view kind) {
  std::vector<bool> inRegister(total, false);
  for (const auto& reg : registers) {
    if (reg.bits.empty()) {
      throw std::runtime_error("Qiskit does not support empty " +
                               std::string(kind) + " registers");
    }
    for (const auto bit : reg.bits) {
      if (bit >= total || inRegister[bit]) {
        throw std::runtime_error(
            "Qiskit 2.5 C-API export requires disjoint register membership");
      }
      inRegister[bit] = true;
    }
  }
  const auto firstRegistered =
      std::find(inRegister.begin(), inRegister.end(), true);
  const auto loose =
      static_cast<std::uint32_t>(firstRegistered - inRegister.begin());
  std::uint32_t expected = loose;
  for (const auto& reg : registers) {
    for (const auto bit : reg.bits) {
      if (bit != expected) {
        throw std::runtime_error(
            "Qiskit 2.5 C-API export requires loose bits before contiguous "
            "registers");
      }
      ++expected;
    }
  }
  if (expected != total) {
    throw std::runtime_error("Qiskit 2.5 C-API export requires loose bits "
                             "before contiguous registers");
  }
  return loose;
}

} // namespace

mlir::QCProgram importCircuit(PyObject* circuit) {
  auto adapter = selectAdapter();
  auto view = adapter->openCircuit(circuit);

  Symbols symbols;
  collectCircuitSymbols(*view, symbols, {});

  auto context = createContext();
  mlir::qc::QCProgramBuilder builder(context.get());
  llvm::SmallVector<mlir::Type> parameterTypes(symbols.size());
  for (const auto& symbol : symbols) {
    parameterTypes[symbol.second.index] =
        expressionType(builder, symbol.second.type, symbol.second.width);
  }
  llvm::SmallVector<mlir::Type> resultTypes;
  if (view->numClbits() == 0U) {
    resultTypes.push_back(builder.getI64Type());
  } else {
    resultTypes.push_back(mlir::MemRefType::get(
        {static_cast<std::int64_t>(view->numClbits())}, builder.getI1Type()));
  }
  auto arguments = builder.initialize(parameterTypes, resultTypes);
  if (!arguments.empty()) {
    auto function = llvm::cast<mlir::func::FuncOp>(
        arguments.front().getParentBlock()->getParentOp());
    for (const auto& symbol : symbols) {
      function.setArgAttr(static_cast<unsigned int>(symbol.second.index),
                          PARAMETER_NAME_ATTR,
                          builder.getStringAttr(symbol.first()));
    }
  }

  llvm::SmallVector<mlir::Value> qubits;
  if (view->numQubits() > 0U) {
    const auto quantumRegister = builder.allocQubitRegister(
        static_cast<std::int64_t>(view->numQubits()));
    qubits = quantumRegister.qubits;
  }
  mlir::Value classicalStorage;
  if (view->numClbits() > 0U) {
    classicalStorage = builder.allocClassicalBitRegister(
        static_cast<std::int64_t>(view->numClbits()), "");
  }

  std::vector<std::uint32_t> qubitMap(view->numQubits());
  std::vector<std::uint32_t> clbitMap(view->numClbits());
  std::iota(qubitMap.begin(), qubitMap.end(), 0U);
  std::iota(clbitMap.begin(), clbitMap.end(), 0U);
  translateCircuit(builder, *view, qubitMap, clbitMap, qubits, classicalStorage,
                   symbols, arguments, {});

  auto module = classicalStorage
                    ? builder.finalize(mlir::ValueRange{classicalStorage})
                    : builder.finalize();
  mlir::OpBuilder attributeBuilder(context.get());
  (*module)->setAttr(QUANTUM_REGISTERS_ATTR,
                     registerAttributes(attributeBuilder, *view, true));
  (*module)->setAttr(CLASSICAL_REGISTERS_ATTR,
                     registerAttributes(attributeBuilder, *view, false));
  return QCProgramAccess::create(std::move(context), std::move(module));
}

PyObject* exportCircuit(const mlir::QCProgram& program) {
  auto module = QCProgramAccess::module(program);
  const auto functions = module.getOps<mlir::func::FuncOp>();
  if (functions.empty() || !llvm::hasSingleElement(functions)) {
    throw std::runtime_error(
        "QC to Qiskit export requires exactly one entry function");
  }
  auto function = *functions.begin();
  if (function.getBody().empty() ||
      !llvm::hasSingleElement(function.getBody())) {
    throw std::runtime_error(
        "QC to Qiskit export requires a single-block entry function");
  }
  llvm::StringSet<> parameterNames;
  for (const auto argument : function.getArguments()) {
    const auto name = function.getArgAttrOfType<mlir::StringAttr>(
        argument.getArgNumber(), PARAMETER_NAME_ATTR);
    if (!argument.getType().isF64() || !name || name.getValue().empty()) {
      throw std::runtime_error(
          "QC to Qiskit export supports only named f64 parameter arguments");
    }
    if (!parameterNames.insert(name.getValue()).second) {
      throw std::runtime_error(
          "QC to Qiskit export requires unique parameter names");
    }
  }

  ExportState state;
  collectResources(function, state);
  collectFlatInstructions(function, state);
  const auto quantumRegisters =
      readRegisterMetadata(module, QUANTUM_REGISTERS_ATTR);
  const auto classicalRegisters =
      readRegisterMetadata(module, CLASSICAL_REGISTERS_ATTR);
  const auto looseQubits =
      validateRegisterLayout(quantumRegisters, state.numQubits, "quantum");
  const auto looseClbits =
      validateRegisterLayout(classicalRegisters, state.numClbits, "classical");

  auto adapter = selectAdapter();
  auto writer = adapter->createCircuit(looseQubits, looseClbits);
  for (const auto& reg : quantumRegisters) {
    writer->addQuantumRegister(reg.name,
                               static_cast<std::uint32_t>(reg.bits.size()));
  }
  for (const auto& reg : classicalRegisters) {
    writer->addClassicalRegister(reg.name,
                                 static_cast<std::uint32_t>(reg.bits.size()));
  }
  writer->setGlobalPhase(state.globalPhase);
  for (const auto& instruction : state.instructions) {
    switch (instruction.kind) {
    case ExportedInstruction::Kind::Gate:
      writer->addGate(instruction.name, instruction.qubits,
                      instruction.parameters);
      break;
    case ExportedInstruction::Kind::Measure:
      writer->addMeasure(instruction.qubits.at(0), instruction.clbits.at(0));
      break;
    case ExportedInstruction::Kind::Reset:
      writer->addReset(instruction.qubits.at(0));
      break;
    case ExportedInstruction::Kind::Barrier:
      writer->addBarrier(instruction.qubits);
      break;
    case ExportedInstruction::Kind::Unitary:
      writer->addUnitary(instruction.matrix, instruction.qubits);
      break;
    }
  }
  return writer->finish();
}

bool compilerBridgeAvailable() {
  try {
    return hasSupportedAdapter(inspectInstalledVersion());
  } catch (const std::runtime_error&) {
    return false;
  }
}

} // namespace mqt::bindings::qiskit
