/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

// Keep the public declaration visible so this definition is type-checked.
#include "Qiskit.h" // IWYU pragma: keep
#include "QiskitTranslation.h"
#include "QiskitVersion.h"
#include "jeff/IR/JeffDialect.h"
#include "mlir/Compiler/Programs.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/Translation/StandardGate.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Dialect/Utils/DenseUnitary.h"

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <nanobind/nanobind.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace mqt::bindings::qiskit {
namespace {

using ParameterValue = std::variant<double, mlir::Value>;

using LocalParameters = llvm::StringMap<mlir::Value>;

constexpr size_t MAX_DEFINITION_DEPTH = 64U;
constexpr size_t MAX_CONTROL_FLOW_DEPTH = 64U;
constexpr size_t MAX_EXPANDED_OPERATIONS = 10'000'000U;

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

void validateParameter(const Parameter& parameter,
                       const llvm::StringSet<>& localParameters) {
  if (parameter.number) {
    if (!std::isfinite(*parameter.number)) {
      throw std::runtime_error("Qiskit returned a non-finite parameter");
    }
    return;
  }
  if (localParameters.contains(parameter.text)) {
    return;
  }
  throw std::runtime_error(
      "Qiskit circuit import does not support free symbolic parameter '" +
      parameter.text + "'");
}

[[nodiscard]] ParameterValue
parameterValue(const std::string_view text,
               const LocalParameters& localParameters) {
  if (const auto local = localParameters.find(text);
      local != localParameters.end()) {
    return local->second;
  }
  throw std::runtime_error(
      "Qiskit circuit import does not support free symbolic parameter '" +
      std::string(text) + "'");
}

[[nodiscard]] ParameterValue
parameterValue(const Parameter& parameter,
               const LocalParameters& localParameters) {
  if (parameter.number) {
    if (!std::isfinite(*parameter.number)) {
      throw std::runtime_error("Qiskit returned a non-finite parameter");
    }
    return *parameter.number;
  }
  return parameterValue(parameter.text, localParameters);
}

void requireArity(const Instruction& instruction, const size_t qubits,
                  const size_t parameters) {
  if (instruction.qubits.size() != qubits ||
      instruction.parameters.size() != parameters) {
    throw std::runtime_error("Qiskit instruction '" + instruction.name +
                             "' has an unsupported operand arity");
  }
}

using GateArity = std::pair<size_t, size_t>;

struct ModifiedQubitArity {
  size_t controls;
  size_t targets;
};

[[nodiscard]] size_t modifierControlCount(const Instruction& instruction) {
  size_t controls = 0U;
  for (const auto& modifier : instruction.modifiers) {
    if (modifier.kind != GateModifierKind::Control) {
      continue;
    }
    if (modifier.numControls > std::numeric_limits<size_t>::max() - controls) {
      throw std::runtime_error("Qiskit control count is too large");
    }
    controls += modifier.numControls;
  }
  return controls;
}

[[nodiscard]] ModifiedQubitArity
modifiedQubitArity(const Instruction& instruction, const size_t targets) {
  const auto controls = modifierControlCount(instruction);
  if (targets > std::numeric_limits<size_t>::max() - controls ||
      instruction.qubits.size() != controls + targets) {
    throw std::runtime_error("Qiskit instruction '" + instruction.name +
                             "' has an unsupported modified operand arity");
  }
  return {.controls = controls, .targets = targets};
}

[[nodiscard]] ModifiedQubitArity
denseUnitaryArity(const Instruction& instruction) {
  if (!instruction.parameters.empty() || !instruction.clbits.empty()) {
    throw std::runtime_error(
        "Qiskit unitary instruction has an unsupported operand arity");
  }
  const auto controls = modifierControlCount(instruction);
  if (instruction.qubits.size() <= controls) {
    throw std::runtime_error(
        "Qiskit unitary instruction has an unsupported operand arity");
  }
  const auto targets = instruction.qubits.size() - controls;
  if (targets > mlir::utils::MAX_DENSE_UNITARY_QUBITS) {
    throw std::runtime_error(
        "Qiskit unitary supports at most " +
        std::to_string(mlir::utils::MAX_DENSE_UNITARY_QUBITS) + " qubits");
  }
  return {.controls = controls, .targets = targets};
}

[[nodiscard]] std::optional<GateArity>
gateArity(const Instruction& instruction) {
  if (!instruction.standardGate) {
    return std::nullopt;
  }
  const auto& descriptor =
      mlir::qc::getStandardGateDescriptor(instruction.standardGate->gate);
  return GateArity{instruction.standardGate->controls +
                       descriptor.controlCount + descriptor.targetCount,
                   descriptor.parameterCount};
}

[[nodiscard]] mlir::Value floatConstant(mlir::ImplicitLocOpBuilder& builder,
                                        const double value) {
  return mlir::arith::ConstantOp::create(builder,
                                         builder.getF64FloatAttr(value))
      .getResult();
}

void emitBaseGate(mlir::qc::QCProgramBuilder& builder,
                  const mlir::qc::StandardGate gate,
                  const mlir::ValueRange qubits,
                  const llvm::ArrayRef<ParameterValue> parameters) {
  if (gate == mlir::qc::StandardGate::CU ||
      gate == mlir::qc::StandardGate::BuiltinU) {
    throw std::runtime_error(
        "Qiskit standard gate requires a compound emission recipe");
  }
  llvm::SmallVector<mlir::Value> parameterValues;
  parameterValues.reserve(parameters.size());
  for (const auto& parameter : parameters) {
    parameterValues.push_back(
        std::holds_alternative<double>(parameter)
            ? floatConstant(builder, std::get<double>(parameter))
            : std::get<mlir::Value>(parameter));
  }
  if (failed(mlir::qc::emitStandardGate(builder, builder.getLoc(), gate,
                                        parameterValues, qubits))) {
    throw std::runtime_error(
        "Qiskit instruction has an unsupported operand arity");
  }
}

void emitControlledGate(mlir::qc::QCProgramBuilder& builder,
                        const mlir::qc::StandardGate gate,
                        const mlir::ValueRange controls,
                        const mlir::ValueRange targets,
                        const llvm::ArrayRef<ParameterValue> parameters) {
  builder.ctrl(controls, targets, [&](const mlir::ValueRange targetArguments) {
    emitBaseGate(builder, gate, targetArguments, parameters);
  });
}

void emitStandardGate(mlir::qc::QCProgramBuilder& builder,
                      const Instruction& instruction, mlir::ValueRange qubits,
                      llvm::ArrayRef<ParameterValue> parameters);

template <typename EmitBase>
void emitModifiedOperation(mlir::qc::QCProgramBuilder& builder,
                           const Instruction& instruction,
                           const mlir::ValueRange qubits,
                           const ModifiedQubitArity arity,
                           const LocalParameters& localParameters,
                           EmitBase&& emitBase) {
  const auto targets = qubits.drop_front(arity.controls);
  const auto emitModifiers =
      [&](auto&& self, const size_t count,
          const mlir::ValueRange targetArguments) -> void {
    if (count == 0U) {
      emitBase(targetArguments);
      return;
    }
    const auto& modifier = instruction.modifiers[count - 1U];
    switch (modifier.kind) {
    case GateModifierKind::Control:
      // Closed controls commute with the other supported Qiskit modifiers and
      // are represented together by the outer qc.ctrl below.
      self(self, count - 1U, targetArguments);
      return;
    case GateModifierKind::Inverse:
      builder.inv(targetArguments, [&](const mlir::ValueRange innerArguments) {
        self(self, count - 1U, innerArguments);
      });
      return;
    case GateModifierKind::Power: {
      const auto exponent = parameterValue(modifier.exponent, localParameters);
      builder.pow(exponent, targetArguments,
                  [&](const mlir::ValueRange innerArguments) {
                    self(self, count - 1U, innerArguments);
                  });
      return;
    }
    }
    throw std::runtime_error("unknown normalized Qiskit gate modifier");
  };

  if (arity.controls == 0U) {
    emitModifiers(emitModifiers, instruction.modifiers.size(), targets);
    return;
  }
  builder.ctrl(qubits.take_front(arity.controls), targets,
               [&](const mlir::ValueRange targetArguments) {
                 emitModifiers(emitModifiers, instruction.modifiers.size(),
                               targetArguments);
               });
}

void emitModifiedGate(mlir::qc::QCProgramBuilder& builder,
                      const Instruction& instruction,
                      const mlir::ValueRange qubits,
                      const llvm::ArrayRef<ParameterValue> parameters,
                      const LocalParameters& localParameters) {
  const auto arity = gateArity(instruction);
  if (!arity) {
    throw std::runtime_error("unsupported modified Qiskit standard gate '" +
                             instruction.name + "'");
  }
  if (instruction.parameters.size() != arity->second) {
    throw std::runtime_error("Qiskit instruction '" + instruction.name +
                             "' has an unsupported modified operand arity");
  }
  emitModifiedOperation(
      builder, instruction, qubits,
      modifiedQubitArity(instruction, arity->first), localParameters,
      [&](const mlir::ValueRange targetArguments) {
        emitStandardGate(builder, instruction, targetArguments, parameters);
      });
}

void emitStandardGate(mlir::qc::QCProgramBuilder& builder,
                      const Instruction& instruction,
                      const mlir::ValueRange qubits,
                      const llvm::ArrayRef<ParameterValue> parameters) {
  const auto& name = instruction.name;
  const auto arity = gateArity(instruction);
  if (!arity || !instruction.standardGate) {
    throw std::runtime_error("unsupported Qiskit standard gate '" + name + "'");
  }
  if (qubits.size() != arity->first || parameters.size() != arity->second) {
    throw std::runtime_error("Qiskit instruction '" + name +
                             "' has an unsupported operand arity");
  }
  const auto mapping = *instruction.standardGate;
  if (mapping.gate == mlir::qc::StandardGate::GPhase) {
    builder.gphase(parameters[0]);
  } else if (mapping.gate == mlir::qc::StandardGate::CU) {
    builder.ctrl(qubits.take_front(1), qubits.take_back(1),
                 [&](const mlir::ValueRange targetArguments) {
                   builder.gphase(parameters[3]);
                   builder.u(parameters[0], parameters[1], parameters[2],
                             targetArguments[0]);
                 });
  } else if (mapping.controls != 0U) {
    emitControlledGate(builder, mapping.gate,
                       qubits.take_front(mapping.controls),
                       qubits.drop_front(mapping.controls), parameters);
  } else {
    emitBaseGate(builder, mapping.gate, qubits, parameters);
  }
}

void emitGate(mlir::qc::QCProgramBuilder& builder,
              const Instruction& instruction,
              const llvm::ArrayRef<mlir::Value> allQubits,
              const llvm::ArrayRef<uint32_t> qubitMap,
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
    parameters.push_back(parameterValue(parameter, localParameters));
  }

  const llvm::ArrayRef<mlir::Value> qubitRange(qubits);
  if (!instruction.modifiers.empty()) {
    emitModifiedGate(builder, instruction, qubitRange, parameters,
                     localParameters);
    return;
  }
  emitStandardGate(builder, instruction, qubitRange, parameters);
}

[[nodiscard]] std::vector<Register>
circuitRegisters(const CircuitReader& circuit, const bool quantum) {
  std::vector<Register> result;
  const auto count =
      quantum ? circuit.numQuantumRegisters() : circuit.numClassicalRegisters();
  result.reserve(count);
  for (size_t index = 0; index < count; ++index) {
    result.push_back(quantum ? circuit.quantumRegister(index)
                             : circuit.classicalRegister(index));
  }
  return result;
}

[[nodiscard]] mlir::Type expressionType(mlir::OpBuilder& builder,
                                        const ClassicalType type,
                                        const uint32_t width) {
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
                                          const uint32_t width,
                                          const uint64_t value) {
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

[[nodiscard]] mlir::Value emitExpression(mlir::qc::QCProgramBuilder& builder,
                                         const Expression& expression) {
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
      return floatConstant(builder, expression.floatValue);
    }
    break;
  case ExpressionKind::Cast: {
    const auto operand = emitExpression(builder, *expression.left);
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
    const auto operand = emitExpression(builder, *expression.left);
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
    auto left = emitExpression(builder, *expression.left);
    auto right = emitExpression(builder, *expression.right);
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
    if (expression.binaryOperation == BinaryOperation::ShiftLeft ||
        expression.binaryOperation == BinaryOperation::ShiftRight) {
      const auto shiftType = llvm::dyn_cast<mlir::IntegerType>(right.getType());
      if (!shiftType || shiftType.getWidth() > integerType.getWidth()) {
        throw std::runtime_error(
            "Qiskit circuit import does not support a shift amount wider "
            "than its integer operand");
      }
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
    const auto target = emitExpression(builder, *expression.left);
    auto index = emitExpression(builder, *expression.right);
    const auto targetType = llvm::dyn_cast<mlir::IntegerType>(target.getType());
    if (!targetType) {
      throw std::runtime_error(
          "Qiskit index expressions require a Uint target");
    }
    const auto indexType = llvm::dyn_cast<mlir::IntegerType>(index.getType());
    if (!indexType || indexType.getWidth() > targetType.getWidth()) {
      throw std::runtime_error(
          "Qiskit circuit import does not support an index wider than its "
          "integer operand");
    }
    index = castInteger(builder, index, targetType);
    const auto shifted =
        mlir::arith::ShRUIOp::create(builder, target, index).getResult();
    const auto integerResult = llvm::dyn_cast<mlir::IntegerType>(resultType);
    if (!integerResult) {
      throw std::runtime_error(
          "Qiskit index expressions must produce a Boolean or Uint value");
    }
    return castInteger(builder, shifted, integerResult);
  }
  }
  throw std::runtime_error("unsupported normalized Qiskit expression");
}

struct ClassicalBitRef {
  mlir::Value storage;
  int64_t index;
};

[[nodiscard]] mlir::Value
loadClassicalBit(mlir::qc::QCProgramBuilder& builder,
                 const llvm::ArrayRef<ClassicalBitRef> classicalBits,
                 const llvm::ArrayRef<uint32_t> rootClbitMap,
                 const uint32_t index) {
  if (index >= rootClbitMap.size() ||
      rootClbitMap[index] >= classicalBits.size()) {
    throw std::runtime_error(
        "Qiskit control flow references an invalid classical bit");
  }
  const auto& bit = classicalBits[rootClbitMap[index]];
  auto position = mlir::arith::ConstantIndexOp::create(builder, bit.index);
  return mlir::memref::LoadOp::create(builder, bit.storage,
                                      mlir::ValueRange{position.getResult()})
      .getResult();
}

[[nodiscard]] mlir::Value
packRegister(mlir::qc::QCProgramBuilder& builder,
             const llvm::ArrayRef<ClassicalBitRef> classicalBits,
             const llvm::ArrayRef<uint32_t> rootClbitMap, const Register& reg) {
  if (reg.bits.empty() || reg.bits.size() > 64U) {
    throw std::runtime_error(
        "Qiskit classical registers must contain between 1 and 64 bits");
  }
  const auto width = static_cast<uint32_t>(reg.bits.size());
  const auto type = builder.getIntegerType(width);
  auto packed = integerConstant(builder, width, 0U);
  for (size_t index = 0; index < reg.bits.size(); ++index) {
    auto bit = castInteger(
        builder,
        loadClassicalBit(builder, classicalBits, rootClbitMap, reg.bits[index]),
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
              const ClassicalTarget& target,
              const llvm::ArrayRef<ClassicalBitRef> classicalBits,
              const llvm::ArrayRef<uint32_t> rootClbitMap) {
  switch (target.kind) {
  case ClassicalTargetKind::ClassicalBit: {
    const auto actual =
        loadClassicalBit(builder, classicalBits, rootClbitMap, target.bit);
    return mlir::arith::CmpIOp::create(builder, mlir::arith::CmpIPredicate::eq,
                                       actual,
                                       builder.boolConstant(target.expectedBit))
        .getResult();
  }
  case ClassicalTargetKind::ClassicalRegister: {
    const auto actual = castInteger(
        builder, packRegister(builder, classicalBits, rootClbitMap, target.reg),
        builder.getIntegerType(target.width));
    const auto expected =
        integerConstant(builder, target.width, target.expectedRegister);
    return mlir::arith::CmpIOp::create(builder, mlir::arith::CmpIPredicate::eq,
                                       actual, expected)
        .getResult();
  }
  case ClassicalTargetKind::Expression: {
    const auto condition = emitExpression(builder, *target.expression);
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
                 const llvm::ArrayRef<ClassicalBitRef> classicalBits,
                 const llvm::ArrayRef<uint32_t> rootClbitMap) {
  mlir::Value value;
  switch (target.kind) {
  case ClassicalTargetKind::ClassicalBit:
    value = loadClassicalBit(builder, classicalBits, rootClbitMap, target.bit);
    break;
  case ClassicalTargetKind::ClassicalRegister:
    value = packRegister(builder, classicalBits, rootClbitMap, target.reg);
    break;
  case ClassicalTargetKind::Expression:
    value = emitExpression(builder, *target.expression);
    break;
  }
  if (!llvm::isa<mlir::IntegerType>(value.getType())) {
    throw std::runtime_error("Qiskit switch targets must be Boolean or Uint");
  }
  return mlir::arith::IndexCastUIOp::create(builder, builder.getIndexType(),
                                            value)
      .getResult();
}

void translateCircuit(mlir::qc::QCProgramBuilder& builder,
                      const CircuitReader& circuit,
                      llvm::ArrayRef<uint32_t> qubitMap,
                      llvm::ArrayRef<uint32_t> clbitMap,
                      llvm::ArrayRef<uint32_t> rootQubitMap,
                      llvm::ArrayRef<uint32_t> rootClbitMap,
                      llvm::ArrayRef<mlir::Value> allQubits,
                      llvm::ArrayRef<ClassicalBitRef> classicalBits,
                      const LocalParameters& localParameters,
                      size_t definitionDepth, size_t controlFlowDepth);

[[nodiscard]] int64_t rangeLength(const Loop& loop) {
  if (loop.step == 0) {
    throw std::runtime_error("Qiskit for-loop range has a zero step");
  }
  if ((loop.step > 0 && loop.start >= loop.stop) ||
      (loop.step < 0 && loop.start <= loop.stop)) {
    return 0;
  }
  const auto distance = loop.step > 0 ? static_cast<uint64_t>(loop.stop) -
                                            static_cast<uint64_t>(loop.start)
                                      : static_cast<uint64_t>(loop.start) -
                                            static_cast<uint64_t>(loop.stop);
  const auto magnitude = loop.step > 0
                             ? static_cast<uint64_t>(loop.step)
                             : static_cast<uint64_t>(-(loop.step + 1)) + 1U;
  const auto count = ((distance - 1U) / magnitude) + 1U;
  if (count > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    throw std::runtime_error(
        "Qiskit for-loop range is too large to represent safely");
  }
  return static_cast<int64_t>(count);
}

void requireExactLoopParameter(const int64_t value) {
  constexpr auto maxExactDoubleInteger = static_cast<int64_t>(1ULL << 53U);
  if (value < -maxExactDoubleInteger || value > maxExactDoubleInteger) {
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
                          const ControlFlowReader& controlFlow,
                          llvm::ArrayRef<mlir::Value> allQubits,
                          llvm::ArrayRef<ClassicalBitRef> classicalBits,
                          llvm::ArrayRef<uint32_t> rootQubitMap,
                          llvm::ArrayRef<uint32_t> rootClbitMap,
                          const LocalParameters& localParameters,
                          const size_t definitionDepth,
                          const size_t controlFlowDepth) {
  if (controlFlowDepth >= MAX_CONTROL_FLOW_DEPTH) {
    throw std::runtime_error(
        "Qiskit control flow exceeds the nesting limit of 64");
  }
  const auto mapToGlobal = [](const std::vector<uint32_t>& localToRoot,
                              const llvm::ArrayRef<uint32_t> rootToGlobal,
                              const std::string_view kind) {
    std::vector<uint32_t> result;
    result.reserve(localToRoot.size());
    for (const auto root : localToRoot) {
      if (root >= rootToGlobal.size()) {
        throw std::runtime_error("Qiskit control flow references an invalid " +
                                 std::string(kind));
      }
      result.push_back(rootToGlobal[root]);
    }
    return result;
  };
  const auto qubitMap =
      mapToGlobal(controlFlow.qubitMap(), rootQubitMap, "qubit");
  const auto clbitMap =
      mapToGlobal(controlFlow.clbitMap(), rootClbitMap, "classical bit");
  const auto translateBlock = [&](const CircuitReader& block,
                                  const LocalParameters& parameters) {
    translateCircuit(builder, block, qubitMap, clbitMap, rootQubitMap,
                     rootClbitMap, allQubits, classicalBits, parameters,
                     definitionDepth, controlFlowDepth + 1U);
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
        emitCondition(builder, condition, classicalBits, rootClbitMap);
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
          builder.scfCondition(
              emitCondition(builder, condition, classicalBits, rootClbitMap));
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
              floatConstant(builder, static_cast<double>(value));
        }
        translateBlock(*body, parameters);
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
      translateBlock(*body, parameters);
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
                                         classicalBits, rootClbitMap);
    std::vector<std::unique_ptr<CircuitReader>> blocks;
    blocks.reserve(controlFlow.numBlocks());
    for (size_t index = 0; index < controlFlow.numBlocks(); ++index) {
      blocks.push_back(controlFlow.block(index));
    }
    llvm::SmallVector<int64_t> labels;
    llvm::SmallVector<std::function<void()>> ownedBodies;
    std::optional<size_t> defaultBlock;
    for (size_t caseIndex = 0; caseIndex < cases.size(); ++caseIndex) {
      if (cases[caseIndex].isDefault) {
        if (defaultBlock) {
          throw std::runtime_error(
              "Qiskit switch has more than one default case");
        }
        defaultBlock = caseIndex;
      }
      for (const auto label : cases[caseIndex].labels) {
        if (label >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
          throw std::runtime_error("Qiskit switch label cannot be represented "
                                   "safely by scf.index_switch");
        }
        labels.push_back(static_cast<int64_t>(label));
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
                      const CircuitReader& circuit,
                      const llvm::ArrayRef<uint32_t> qubitMap,
                      const llvm::ArrayRef<uint32_t> clbitMap,
                      const llvm::ArrayRef<uint32_t> rootQubitMap,
                      const llvm::ArrayRef<uint32_t> rootClbitMap,
                      const llvm::ArrayRef<mlir::Value> allQubits,
                      const llvm::ArrayRef<ClassicalBitRef> classicalBits,
                      const LocalParameters& localParameters,
                      const size_t definitionDepth,
                      const size_t controlFlowDepth) {
  builder.gphase(parameterValue(circuit.globalPhase(), localParameters));
  const auto getQubit = [&](const uint32_t local) {
    if (local >= qubitMap.size() || qubitMap[local] >= allQubits.size()) {
      throw std::runtime_error(
          "Qiskit instruction references an invalid mapped qubit");
    }
    return allQubits[qubitMap[local]];
  };
  const auto getClbit = [&](const uint32_t local) {
    if (local >= clbitMap.size() || clbitMap[local] >= classicalBits.size()) {
      throw std::runtime_error(
          "Qiskit instruction references an invalid mapped classical bit");
    }
    return clbitMap[local];
  };
  const auto translateDefinition = [&](const size_t index,
                                       const Instruction& instruction) {
    if (definitionDepth >= MAX_DEFINITION_DEPTH) {
      throw std::runtime_error(
          "Qiskit instruction definitions exceed the nesting limit of 64");
    }
    auto definition = circuit.definition(index);
    std::vector<uint32_t> definitionQubits;
    definitionQubits.reserve(instruction.qubits.size());
    for (const auto qubit : instruction.qubits) {
      if (qubit >= qubitMap.size()) {
        throw std::runtime_error(
            "Qiskit instruction definition references an invalid qubit");
      }
      definitionQubits.push_back(qubitMap[qubit]);
    }
    std::vector<uint32_t> definitionClbits;
    definitionClbits.reserve(instruction.clbits.size());
    for (const auto clbit : instruction.clbits) {
      definitionClbits.push_back(getClbit(clbit));
    }
    translateCircuit(builder, *definition, definitionQubits, definitionClbits,
                     definitionQubits, definitionClbits, allQubits,
                     classicalBits, localParameters, definitionDepth + 1U,
                     controlFlowDepth);
  };

  for (size_t index = 0; index < circuit.numInstructions(); ++index) {
    const auto instruction = circuit.instruction(index);
    switch (instruction.kind) {
    case OperationKind::Gate:
      if (instruction.standardGate) {
        emitGate(builder, instruction, allQubits, qubitMap, localParameters);
      } else {
        translateDefinition(index, instruction);
      }
      break;
    case OperationKind::Barrier: {
      llvm::SmallVector<mlir::Value> operands;
      for (const auto qubit : instruction.qubits) {
        operands.push_back(getQubit(qubit));
      }
      builder.barrier(operands);
      break;
    }
    case OperationKind::Measure: {
      requireArity(instruction, 1, 0);
      if (instruction.clbits.size() != 1U) {
        throw std::runtime_error(
            "Qiskit measurement has an invalid classical destination");
      }
      const auto& destination = classicalBits[getClbit(instruction.clbits[0])];
      builder.measure(getQubit(instruction.qubits[0]), destination.storage,
                      destination.index);
      break;
    }
    case OperationKind::Reset:
      requireArity(instruction, 1, 0);
      builder.reset(getQubit(instruction.qubits[0]));
      break;
    case OperationKind::Unitary: {
      llvm::SmallVector<mlir::Value> operands;
      operands.reserve(instruction.qubits.size());
      for (const auto qubit : instruction.qubits) {
        operands.push_back(getQubit(qubit));
      }
      const auto arity = denseUnitaryArity(instruction);
      const auto dimension = int64_t{1} << arity.targets;
      const auto type = mlir::RankedTensorType::get(
          {dimension, dimension}, mlir::ComplexType::get(builder.getF64Type()));
      const auto values =
          reverseQubitOrder(circuit.unitary(index), arity.targets);
      const auto matrix = mlir::DenseElementsAttr::get(
          type, llvm::ArrayRef<std::complex<double>>(values));
      emitModifiedOperation(builder, instruction, operands, arity,
                            localParameters,
                            [&](const mlir::ValueRange targetArguments) {
                              builder.unitary(targetArguments, matrix);
                            });
    } break;
    case OperationKind::ControlFlow: {
      const auto controlFlow = circuit.controlFlow(index);
      translateControlFlow(builder, *controlFlow, allQubits, classicalBits,
                           rootQubitMap, rootClbitMap, localParameters,
                           definitionDepth, controlFlowDepth);
      break;
    }
    case OperationKind::Delay:
      throw std::runtime_error("Qiskit delay instructions are not supported");
    case OperationKind::Unknown:
      translateDefinition(index, instruction);
      break;
    }
  }
}

struct ExpansionSummary {
  size_t operations = 0U;
  size_t definitionDepth = 0U;
  size_t controlFlowDepth = 0U;
};

struct ExpansionCountState {
  llvm::DenseMap<uintptr_t, ExpansionSummary> definitions;
  llvm::DenseSet<uintptr_t> activeDefinitions;
};

void addExpandedOperations(size_t& total, const size_t additional) {
  if (additional > MAX_EXPANDED_OPERATIONS - total) {
    throw std::runtime_error(
        "Qiskit instruction expansion exceeds 10000000 operations");
  }
  total += additional;
}

[[nodiscard]] size_t repeatedOperations(const size_t operations,
                                        const size_t repetitions) {
  if (repetitions != 0U && operations > MAX_EXPANDED_OPERATIONS / repetitions) {
    throw std::runtime_error(
        "Qiskit instruction expansion exceeds 10000000 operations");
  }
  return operations * repetitions;
}

[[nodiscard]] ExpansionSummary
expansionSummary(const CircuitReader& circuit, ExpansionCountState& state,
                 const size_t definitionDepth = 0U,
                 const size_t controlFlowDepth = 0U) {
  ExpansionSummary result;
  for (size_t index = 0; index < circuit.numInstructions(); ++index) {
    addExpandedOperations(result.operations, 1U);
    const auto instruction = circuit.instruction(index);
    if ((instruction.kind == OperationKind::Gate &&
         !instruction.standardGate) ||
        instruction.kind == OperationKind::Unknown) {
      if (!instruction.modifiers.empty()) {
        throw std::runtime_error(
            "Qiskit circuit import does not support modifiers on custom "
            "instructions");
      }
      const auto identity = circuit.definitionIdentity(index);
      if (identity == 0U) {
        throw std::runtime_error("Qiskit instruction '" + instruction.name +
                                 "' has no circuit definition");
      }
      if (!state.activeDefinitions.insert(identity).second) {
        throw std::runtime_error(
            "Qiskit instruction definitions contain a cycle");
      }
      ExpansionSummary definitionSummary;
      try {
        if (definitionDepth >= MAX_DEFINITION_DEPTH) {
          throw std::runtime_error(
              "Qiskit instruction definitions exceed the nesting limit of 64");
        }
        const auto definition = circuit.definition(index);
        if (definition->numQubits() != instruction.qubits.size() ||
            definition->numClbits() != instruction.clbits.size()) {
          throw std::runtime_error("Qiskit instruction '" + instruction.name +
                                   "' does not match its definition arity");
        }
        if (const auto cached = state.definitions.find(identity);
            cached != state.definitions.end()) {
          definitionSummary = cached->second;
        } else {
          definitionSummary = expansionSummary(
              *definition, state, definitionDepth + 1U, controlFlowDepth);
          state.definitions.insert({identity, definitionSummary});
        }
      } catch (...) {
        state.activeDefinitions.erase(identity);
        throw;
      }
      state.activeDefinitions.erase(identity);
      if (definitionSummary.definitionDepth >=
          MAX_DEFINITION_DEPTH - definitionDepth) {
        throw std::runtime_error(
            "Qiskit instruction definitions exceed the nesting limit of 64");
      }
      result.definitionDepth = std::max(result.definitionDepth,
                                        definitionSummary.definitionDepth + 1U);
      if (definitionSummary.controlFlowDepth >
          MAX_CONTROL_FLOW_DEPTH - controlFlowDepth) {
        throw std::runtime_error(
            "Qiskit control flow exceeds the nesting limit of 64");
      }
      result.controlFlowDepth =
          std::max(result.controlFlowDepth, definitionSummary.controlFlowDepth);
      addExpandedOperations(result.operations, definitionSummary.operations);
      continue;
    }
    if (instruction.kind != OperationKind::ControlFlow) {
      continue;
    }
    if (controlFlowDepth >= MAX_CONTROL_FLOW_DEPTH) {
      throw std::runtime_error(
          "Qiskit control flow exceeds the nesting limit of 64");
    }
    const auto controlFlow = circuit.controlFlow(index);
    size_t repetitions = 1U;
    if (controlFlow->kind() == ControlFlowKind::For) {
      const auto loop = controlFlow->loop();
      if (!loop.isRange) {
        repetitions = loop.values.size();
      }
    }
    for (size_t blockIndex = 0; blockIndex < controlFlow->numBlocks();
         ++blockIndex) {
      const auto blockSummary =
          expansionSummary(*controlFlow->block(blockIndex), state,
                           definitionDepth, controlFlowDepth + 1U);
      result.definitionDepth =
          std::max(result.definitionDepth, blockSummary.definitionDepth);
      result.controlFlowDepth =
          std::max(result.controlFlowDepth, blockSummary.controlFlowDepth + 1U);
      addExpandedOperations(
          result.operations,
          repeatedOperations(blockSummary.operations, repetitions));
    }
  }
  return result;
}

void validateCircuit(const CircuitReader& circuit,
                     const llvm::StringSet<>& localParameters,
                     uint32_t rootQubits, uint32_t rootClbits,
                     size_t definitionDepth, size_t controlFlowDepth);

void validateExpression(const Expression& expression) {
  if (expression.type == ClassicalType::Uint &&
      (expression.width == 0U || expression.width > 64U)) {
    throw std::runtime_error(
        "Qiskit unsigned classical values must be between 1 and 64 bits");
  }
  const auto requireOperand = [](const std::unique_ptr<Expression>& operand) {
    if (!operand) {
      throw std::runtime_error(
          "Qiskit classical expression has a missing operand");
    }
    validateExpression(*operand);
  };
  switch (expression.kind) {
  case ExpressionKind::Value:
    return;
  case ExpressionKind::Unary:
  case ExpressionKind::Cast:
    requireOperand(expression.left);
    return;
  case ExpressionKind::Binary:
  case ExpressionKind::Index:
    requireOperand(expression.left);
    requireOperand(expression.right);
    return;
  }
}

void validateTarget(const ClassicalTarget& target, const uint32_t rootClbits) {
  switch (target.kind) {
  case ClassicalTargetKind::ClassicalBit:
    if (target.bit >= rootClbits) {
      throw std::runtime_error(
          "Qiskit control flow references an invalid classical bit");
    }
    return;
  case ClassicalTargetKind::ClassicalRegister:
    if (target.reg.bits.empty() || target.reg.bits.size() > 64U) {
      throw std::runtime_error(
          "Qiskit control-flow registers must contain between 1 and 64 bits");
    }
    for (const auto bit : target.reg.bits) {
      if (bit >= rootClbits) {
        throw std::runtime_error(
            "Qiskit control flow references an invalid classical bit");
      }
    }
    return;
  case ClassicalTargetKind::Expression:
    if (!target.expression) {
      throw std::runtime_error(
          "Qiskit control flow contains an empty classical expression");
    }
    validateExpression(*target.expression);
    return;
  }
}

void validateControlFlow(const ControlFlowReader& controlFlow,
                         llvm::StringSet<> localParameters,
                         const uint32_t rootQubits, const uint32_t rootClbits,
                         const size_t definitionDepth,
                         const size_t controlFlowDepth) {
  if (controlFlowDepth >= MAX_CONTROL_FLOW_DEPTH) {
    throw std::runtime_error(
        "Qiskit control flow exceeds the nesting limit of 64");
  }
  const auto blockCount = controlFlow.numBlocks();
  switch (controlFlow.kind()) {
  case ControlFlowKind::Box:
    throw std::runtime_error("Qiskit box instructions are not supported");
  case ControlFlowKind::Break:
    throw std::runtime_error("Qiskit break instructions are not supported");
  case ControlFlowKind::Continue:
    throw std::runtime_error("Qiskit continue instructions are not supported");
  case ControlFlowKind::IfElse: {
    if (blockCount < 1U || blockCount > 2U) {
      throw std::runtime_error("Qiskit if/else has an invalid block count");
    }
    const auto condition = controlFlow.condition();
    validateTarget(condition, rootClbits);
    if (condition.kind == ClassicalTargetKind::Expression &&
        condition.expression->type != ClassicalType::Bool) {
      throw std::runtime_error(
          "Qiskit control-flow condition expression must have Boolean type");
    }
    break;
  }
  case ControlFlowKind::While: {
    if (blockCount != 1U) {
      throw std::runtime_error("Qiskit while loop has an invalid block count");
    }
    const auto condition = controlFlow.condition();
    validateTarget(condition, rootClbits);
    if (condition.kind == ClassicalTargetKind::Expression &&
        condition.expression->type != ClassicalType::Bool) {
      throw std::runtime_error(
          "Qiskit control-flow condition expression must have Boolean type");
    }
    break;
  }
  case ControlFlowKind::For: {
    if (blockCount != 1U) {
      throw std::runtime_error("Qiskit for loop has an invalid block count");
    }
    const auto loop = controlFlow.loop();
    if (loop.isRange) {
      const auto count = rangeLength(loop);
      if (loop.parameter && count != 0) {
        requireExactLoopParameter(loop.start);
        requireExactLoopParameter(loop.step > 0 ? loop.stop - 1
                                                : loop.stop + 1);
      }
    } else if (loop.parameter) {
      for (const auto value : loop.values) {
        requireExactLoopParameter(value);
      }
    }
    if (loop.parameter) {
      localParameters.insert(*loop.parameter);
    }
    break;
  }
  case ControlFlowKind::Switch: {
    const auto cases = controlFlow.switchCases();
    if (cases.size() != blockCount) {
      throw std::runtime_error(
          "Qiskit switch case metadata does not match its blocks");
    }
    auto hasDefault = false;
    for (const auto& switchCase : cases) {
      if (switchCase.isDefault && std::exchange(hasDefault, true)) {
        throw std::runtime_error(
            "Qiskit switch has more than one default case");
      }
      for (const auto label : switchCase.labels) {
        if (label >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
          throw std::runtime_error("Qiskit switch label cannot be represented "
                                   "safely by scf.index_switch");
        }
      }
    }
    const auto target = controlFlow.switchTarget();
    validateTarget(target, rootClbits);
    if (target.kind == ClassicalTargetKind::Expression &&
        target.expression->type == ClassicalType::Float) {
      throw std::runtime_error("Qiskit switch targets must be Boolean or Uint");
    }
    break;
  }
  }

  const auto qubitMap = controlFlow.qubitMap();
  const auto clbitMap = controlFlow.clbitMap();
  for (const auto qubit : qubitMap) {
    if (qubit >= rootQubits) {
      throw std::runtime_error(
          "Qiskit control flow references an invalid qubit");
    }
  }
  for (const auto clbit : clbitMap) {
    if (clbit >= rootClbits) {
      throw std::runtime_error(
          "Qiskit control flow references an invalid classical bit");
    }
  }
  for (size_t blockIndex = 0; blockIndex < blockCount; ++blockIndex) {
    const auto block = controlFlow.block(blockIndex);
    if (block->numQubits() != qubitMap.size() ||
        block->numClbits() != clbitMap.size()) {
      throw std::runtime_error(
          "Qiskit control-flow block operands do not match its bit mapping");
    }
    validateCircuit(*block, localParameters, rootQubits, rootClbits,
                    definitionDepth, controlFlowDepth + 1U);
  }
}

void validateDefinition(const CircuitReader& circuit, const size_t index,
                        const llvm::StringSet<>& localParameters,
                        const size_t definitionDepth,
                        const size_t controlFlowDepth) {
  if (definitionDepth >= MAX_DEFINITION_DEPTH) {
    throw std::runtime_error(
        "Qiskit instruction definitions exceed the nesting limit of 64");
  }
  const auto definition = circuit.definition(index);
  validateCircuit(*definition, localParameters, definition->numQubits(),
                  definition->numClbits(), definitionDepth + 1U,
                  controlFlowDepth);
}

void validateCircuit(const CircuitReader& circuit,
                     const llvm::StringSet<>& localParameters,
                     const uint32_t rootQubits, const uint32_t rootClbits,
                     const size_t definitionDepth,
                     const size_t controlFlowDepth) {
  if (circuit.hasClassicalVariables()) {
    throw std::runtime_error(
        "Qiskit circuit import does not support standalone classical "
        "variables");
  }
  static_cast<void>(validateRegisterLayout(circuitRegisters(circuit, true),
                                           circuit.numQubits(), "quantum"));
  static_cast<void>(validateRegisterLayout(circuitRegisters(circuit, false),
                                           circuit.numClbits(), "classical"));
  validateParameter(circuit.globalPhase(), localParameters);

  for (size_t index = 0; index < circuit.numInstructions(); ++index) {
    const auto instruction = circuit.instruction(index);
    for (const auto qubit : instruction.qubits) {
      if (qubit >= circuit.numQubits()) {
        throw std::runtime_error(
            "Qiskit instruction references an invalid qubit");
      }
    }
    for (const auto clbit : instruction.clbits) {
      if (clbit >= circuit.numClbits()) {
        throw std::runtime_error(
            "Qiskit instruction references an invalid classical bit");
      }
    }
    for (const auto& parameter : instruction.parameters) {
      validateParameter(parameter, localParameters);
    }
    for (const auto& modifier : instruction.modifiers) {
      if (modifier.kind == GateModifierKind::Power) {
        validateParameter(modifier.exponent, localParameters);
      }
    }

    switch (instruction.kind) {
    case OperationKind::Gate:
      if (const auto arity = gateArity(instruction)) {
        size_t modifierControls = 0U;
        for (const auto& modifier : instruction.modifiers) {
          if (modifier.kind == GateModifierKind::Control) {
            if (modifier.numControls >
                std::numeric_limits<size_t>::max() - modifierControls) {
              throw std::runtime_error("Qiskit control count is too large");
            }
            modifierControls += modifier.numControls;
          }
        }
        if (instruction.qubits.size() != arity->first + modifierControls ||
            instruction.parameters.size() != arity->second) {
          throw std::runtime_error("Qiskit instruction '" + instruction.name +
                                   "' has an unsupported operand arity");
        }
        break;
      }
      if (!instruction.modifiers.empty()) {
        throw std::runtime_error(
            "Qiskit circuit import does not support modifiers on custom "
            "instructions");
      }
      validateDefinition(circuit, index, localParameters, definitionDepth,
                         controlFlowDepth);
      break;
    case OperationKind::Unknown:
      if (!instruction.modifiers.empty()) {
        throw std::runtime_error(
            "Qiskit circuit import does not support modifiers on custom "
            "instructions");
      }
      validateDefinition(circuit, index, localParameters, definitionDepth,
                         controlFlowDepth);
      break;
    case OperationKind::Barrier:
      if (!instruction.parameters.empty() || !instruction.clbits.empty()) {
        throw std::runtime_error("Qiskit barrier has an invalid operand arity");
      }
      break;
    case OperationKind::Measure:
      requireArity(instruction, 1U, 0U);
      if (instruction.clbits.size() != 1U) {
        throw std::runtime_error(
            "Qiskit measurement has an invalid classical destination");
      }
      break;
    case OperationKind::Reset:
      requireArity(instruction, 1U, 0U);
      if (!instruction.clbits.empty()) {
        throw std::runtime_error("Qiskit reset has an invalid operand arity");
      }
      break;
    case OperationKind::Unitary:
      static_cast<void>(denseUnitaryArity(instruction));
      break;
    case OperationKind::ControlFlow: {
      const auto controlFlow = circuit.controlFlow(index);
      validateControlFlow(*controlFlow, localParameters, rootQubits, rootClbits,
                          definitionDepth, controlFlowDepth);
      break;
    }
    case OperationKind::Delay:
      throw std::runtime_error("Qiskit delay instructions are not supported");
    }
  }
}

} // namespace

mlir::QCProgram importCircuit(const nb::handle circuit) {
  auto translation = selectTranslation();
  auto view = translation->openCircuit(circuit);

  ExpansionCountState expansion;
  static_cast<void>(expansionSummary(*view, expansion));
  validateCircuit(*view, {}, view->numQubits(), view->numClbits(), 0U, 0U);
  const auto quantumRegisters = circuitRegisters(*view, true);
  const auto classicalRegisters = circuitRegisters(*view, false);
  const auto looseQubits =
      validateRegisterLayout(quantumRegisters, view->numQubits(), "quantum");
  const auto looseClbits = validateRegisterLayout(
      classicalRegisters, view->numClbits(), "classical");

  auto context = createContext();
  mlir::qc::QCProgramBuilder builder(context.get());
  llvm::SmallVector<mlir::Type> resultTypes;
  if (view->numClbits() == 0U) {
    resultTypes.push_back(builder.getI64Type());
  } else {
    if (looseClbits != 0U) {
      resultTypes.push_back(mlir::MemRefType::get(
          {static_cast<int64_t>(looseClbits)}, builder.getI1Type()));
    }
    for (const auto& reg : classicalRegisters) {
      resultTypes.push_back(mlir::MemRefType::get(
          {static_cast<int64_t>(reg.bits.size())}, builder.getI1Type()));
    }
  }
  builder.initialize(resultTypes);

  llvm::SmallVector<mlir::Value> qubits;
  qubits.reserve(view->numQubits());
  if (looseQubits != 0U) {
    const auto loose =
        builder.allocQubitRegister(static_cast<int64_t>(looseQubits));
    llvm::append_range(qubits, loose.qubits);
  }
  for (const auto& reg : quantumRegisters) {
    const auto allocated = builder.allocQubitRegister(
        static_cast<int64_t>(reg.bits.size()), reg.name);
    llvm::append_range(qubits, allocated.qubits);
  }

  llvm::SmallVector<ClassicalBitRef> classicalBits;
  llvm::SmallVector<mlir::Value> classicalStorage;
  classicalBits.reserve(view->numClbits());
  const auto allocateClassical = [&](const uint32_t size,
                                     const std::string_view name) {
    const auto storage =
        builder.allocClassicalBitRegister(static_cast<int64_t>(size), name);
    classicalStorage.push_back(storage);
    for (uint32_t index = 0U; index < size; ++index) {
      classicalBits.push_back(
          {.storage = storage, .index = static_cast<int64_t>(index)});
    }
  };
  if (looseClbits != 0U) {
    allocateClassical(looseClbits, "");
  }
  for (const auto& reg : classicalRegisters) {
    allocateClassical(static_cast<uint32_t>(reg.bits.size()), reg.name);
  }

  std::vector<uint32_t> qubitMap(view->numQubits());
  std::vector<uint32_t> clbitMap(view->numClbits());
  std::iota(qubitMap.begin(), qubitMap.end(), 0U);
  std::iota(clbitMap.begin(), clbitMap.end(), 0U);
  translateCircuit(builder, *view, qubitMap, clbitMap, qubitMap, clbitMap,
                   qubits, classicalBits, {}, 0U, 0U);

  auto moduleOp = classicalStorage.empty() ? builder.finalize()
                                           : builder.finalize(classicalStorage);
  auto program = mlir::QCProgram::fromModule(context, std::move(moduleOp));
  if (!program) {
    throw std::runtime_error(
        "Qiskit circuit import produced an invalid QC program");
  }
  return std::move(*program);
}

} // namespace mqt::bindings::qiskit
