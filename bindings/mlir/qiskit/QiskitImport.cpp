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
#include "mlir/Dialect/CBit/IR/CBitDialect.h"
#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/MQT/Utils/DenseUnitary.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/Translation/StandardGate.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"

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
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
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
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace mqt::bindings::qiskit {

using ParameterValue = std::variant<double, mlir::Value>;

using LocalParameters = llvm::StringMap<mlir::Value>;
using GlobalParameters = llvm::StringMap<mlir::Value>;
using ValidationParameters = llvm::StringMap<Parameter>;

constexpr size_t MAX_DEFINITION_DEPTH = 64U;
constexpr size_t MAX_CONTROL_FLOW_DEPTH = 64U;
constexpr size_t MAX_EXPANDED_OPERATIONS = 10'000'000U;

[[nodiscard]] static mlir::Value
floatConstant(mlir::ImplicitLocOpBuilder& builder, double value);

[[nodiscard]] static mlir::DictionaryAttr
parameterGroupAttribute(mlir::Builder& builder, const ParameterGroup& group) {
  if (group.index >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      group.size > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    throw std::runtime_error(
        "Qiskit parameter-vector metadata cannot be represented by MLIR");
  }
  return builder.getDictionaryAttr({
      builder.getNamedAttr("identity", builder.getStringAttr(group.identity)),
      builder.getNamedAttr("name", builder.getStringAttr(group.name)),
      builder.getNamedAttr("index", builder.getI64IntegerAttr(
                                        static_cast<int64_t>(group.index))),
      builder.getNamedAttr(
          "size", builder.getI64IntegerAttr(static_cast<int64_t>(group.size))),
  });
}

[[noreturn]] static void throwImportedParameterExpressionSizeError() {
  throw std::runtime_error(
      "Qiskit parameter expression exceeds the supported " +
      std::to_string(MAX_PARAMETER_EXPRESSION_NODES) + "-node size");
}

[[noreturn]] static void throwImportedParameterExpressionDepthError() {
  throw std::runtime_error(
      "Qiskit parameter expression exceeds the supported " +
      std::to_string(MAX_PARAMETER_EXPRESSION_DEPTH) + "-level nesting depth");
}

[[nodiscard]] static std::shared_ptr<mlir::MLIRContext> createContext() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::cbit::CBitDialect, mlir::mqt::MQTDialect,
                  mlir::qc::QCDialect, mlir::qco::QCODialect,
                  mlir::qtensor::QTensorDialect, mlir::arith::ArithDialect,
                  mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
                  mlir::math::MathDialect, mlir::scf::SCFDialect,
                  mlir::LLVM::LLVMDialect, mlir::memref::MemRefDialect,
                  mlir::jeff::JeffDialect>();
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  return context;
}

static void validateParameterImpl(const Parameter& parameter,
                                  const ValidationParameters& localParameters,
                                  const ValidationParameters& freeParameters,
                                  const size_t depth, size_t& nodes) {
  if (depth > MAX_PARAMETER_EXPRESSION_DEPTH) {
    throwImportedParameterExpressionDepthError();
  }
  if (++nodes > MAX_PARAMETER_EXPRESSION_NODES) {
    throwImportedParameterExpressionSizeError();
  }
  if (const auto* number = parameter.getNumber()) {
    if (!std::isfinite(number->value)) {
      throw std::runtime_error("Qiskit returned a non-finite parameter");
    }
    return;
  }
  if (const auto* symbol = parameter.getSymbol()) {
    if (symbol->name.empty()) {
      throw std::runtime_error(
          "Qiskit returned a parameter with invalid symbol metadata");
    }
    const auto validateKnownSymbol = [&](const ValidationParameters& known) {
      const auto found = known.find(symbol->name);
      if (found == known.end()) {
        return false;
      }
      const auto* expected = found->second.getSymbol();
      if (expected == nullptr || expected->group != symbol->group) {
        throw std::runtime_error("Qiskit parameter symbol '" + symbol->name +
                                 "' has conflicting group metadata");
      }
      return true;
    };
    if (validateKnownSymbol(localParameters) ||
        validateKnownSymbol(freeParameters)) {
      return;
    }
    throw std::runtime_error("Qiskit parameter symbol '" + symbol->name +
                             "' is not defined in this circuit scope");
  }
  if (const auto* unary = parameter.getUnary()) {
    validateParameterImpl(*unary->operand, localParameters, freeParameters,
                          depth + 1U, nodes);
    return;
  }
  if (const auto* binary = parameter.getBinary()) {
    validateParameterImpl(*binary->left, localParameters, freeParameters,
                          depth + 1U, nodes);
    validateParameterImpl(*binary->right, localParameters, freeParameters,
                          depth + 1U, nodes);
    return;
  }
  throw std::runtime_error("unknown normalized Qiskit parameter expression");
}

static void validateParameter(const Parameter& parameter,
                              const ValidationParameters& localParameters,
                              const ValidationParameters& freeParameters) {
  size_t nodes = 0U;
  validateParameterImpl(parameter, localParameters, freeParameters, 1U, nodes);
}

[[nodiscard]] static mlir::Value
materializeParameterValue(mlir::qc::QCProgramBuilder& builder,
                          const ParameterValue& parameter) {
  return std::holds_alternative<double>(parameter)
             ? floatConstant(builder, std::get<double>(parameter))
             : std::get<mlir::Value>(parameter);
}

[[nodiscard]] static ParameterValue
parameterValueImpl(mlir::qc::QCProgramBuilder& builder,
                   const Parameter& parameter,
                   const LocalParameters& localParameters,
                   const GlobalParameters& globalParameters, const size_t depth,
                   size_t& nodes) {
  if (depth > MAX_PARAMETER_EXPRESSION_DEPTH) {
    throwImportedParameterExpressionDepthError();
  }
  if (++nodes > MAX_PARAMETER_EXPRESSION_NODES) {
    throwImportedParameterExpressionSizeError();
  }
  if (const auto* number = parameter.getNumber()) {
    return number->value;
  }
  if (const auto* symbol = parameter.getSymbol()) {
    if (const auto local = localParameters.find(symbol->name);
        local != localParameters.end()) {
      return local->second;
    }
    if (const auto global = globalParameters.find(symbol->name);
        global != globalParameters.end()) {
      return global->second;
    }
    throw std::runtime_error("Qiskit parameter symbol '" + symbol->name +
                             "' is not defined in this circuit scope");
  }

  if (const auto* unary = parameter.getUnary()) {
    if (unary->operation == UnaryParameterKind::Conjugate) {
      /// QC scalar parameters are real-valued, so conjugation is the identity.
      return parameterValueImpl(builder, *unary->operand, localParameters,
                                globalParameters, depth + 1U, nodes);
    }
    auto operand = materializeParameterValue(
        builder, parameterValueImpl(builder, *unary->operand, localParameters,
                                    globalParameters, depth + 1U, nodes));
    switch (unary->operation) {
    case UnaryParameterKind::Negate:
      return mlir::arith::NegFOp::create(builder, operand).getResult();
    case UnaryParameterKind::Sin:
      return mlir::math::SinOp::create(builder, operand).getResult();
    case UnaryParameterKind::Cos:
      return mlir::math::CosOp::create(builder, operand).getResult();
    case UnaryParameterKind::Tan:
      return mlir::math::TanOp::create(builder, operand).getResult();
    case UnaryParameterKind::ArcSin:
      return mlir::math::AsinOp::create(builder, operand).getResult();
    case UnaryParameterKind::ArcCos:
      return mlir::math::AcosOp::create(builder, operand).getResult();
    case UnaryParameterKind::ArcTan:
      return mlir::math::AtanOp::create(builder, operand).getResult();
    case UnaryParameterKind::Exp:
      return mlir::math::ExpOp::create(builder, operand).getResult();
    case UnaryParameterKind::Log:
      return mlir::math::LogOp::create(builder, operand).getResult();
    case UnaryParameterKind::Abs:
      return mlir::math::AbsFOp::create(builder, operand).getResult();
    case UnaryParameterKind::Conjugate:
      break;
    }
  }

  if (const auto* binary = parameter.getBinary()) {
    auto left = materializeParameterValue(
        builder, parameterValueImpl(builder, *binary->left, localParameters,
                                    globalParameters, depth + 1U, nodes));
    auto right = materializeParameterValue(
        builder, parameterValueImpl(builder, *binary->right, localParameters,
                                    globalParameters, depth + 1U, nodes));
    switch (binary->operation) {
    case BinaryParameterKind::Add:
      return mlir::arith::AddFOp::create(builder, left, right).getResult();
    case BinaryParameterKind::Subtract:
      return mlir::arith::SubFOp::create(builder, left, right).getResult();
    case BinaryParameterKind::Multiply:
      return mlir::arith::MulFOp::create(builder, left, right).getResult();
    case BinaryParameterKind::Divide:
      return mlir::arith::DivFOp::create(builder, left, right).getResult();
    case BinaryParameterKind::Power:
      return mlir::math::PowFOp::create(builder, left, right).getResult();
    }
  }
  throw std::runtime_error("unknown normalized Qiskit parameter expression");
}

[[nodiscard]] static ParameterValue
parameterValue(mlir::qc::QCProgramBuilder& builder, const Parameter& parameter,
               const LocalParameters& localParameters,
               const GlobalParameters& globalParameters) {
  size_t nodes = 0U;
  return parameterValueImpl(builder, parameter, localParameters,
                            globalParameters, 1U, nodes);
}

static void requireArity(const Instruction& instruction, const size_t qubits,
                         const size_t parameters) {
  if (instruction.qubits.size() != qubits ||
      instruction.parameters.size() != parameters) {
    throw std::runtime_error("Qiskit instruction '" + instruction.name +
                             "' has an unsupported operand arity");
  }
}

using GateArity = std::pair<size_t, size_t>;

namespace {
struct ModifiedQubitArity {
  size_t controls;
  size_t targets;
};
} // namespace

[[nodiscard]] static size_t
modifierControlCount(const Instruction& instruction) {
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

[[nodiscard]] static ModifiedQubitArity
modifiedQubitArity(const Instruction& instruction, const size_t targets) {
  const auto controls = modifierControlCount(instruction);
  if (targets > std::numeric_limits<size_t>::max() - controls ||
      instruction.qubits.size() != controls + targets) {
    throw std::runtime_error("Qiskit instruction '" + instruction.name +
                             "' has an unsupported modified operand arity");
  }
  return {.controls = controls, .targets = targets};
}

[[nodiscard]] static ModifiedQubitArity
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
  if (targets > mlir::mqt::MAX_DENSE_UNITARY_QUBITS) {
    throw std::runtime_error(
        "Qiskit unitary supports at most " +
        std::to_string(mlir::mqt::MAX_DENSE_UNITARY_QUBITS) + " qubits");
  }
  return {.controls = controls, .targets = targets};
}

[[nodiscard]] static std::optional<GateArity>
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

static void emitBaseGate(mlir::qc::QCProgramBuilder& builder,
                         const mlir::qc::StandardGate gate,
                         mlir::ValueRange qubits,
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

static void
emitControlledGate(mlir::qc::QCProgramBuilder& builder,
                   const mlir::qc::StandardGate gate, mlir::ValueRange controls,
                   mlir::ValueRange targets,
                   const llvm::ArrayRef<ParameterValue> parameters) {
  builder.ctrl(controls, targets, [&](mlir::ValueRange targetArguments) {
    emitBaseGate(builder, gate, targetArguments, parameters);
  });
}

static void emitStandardGate(mlir::qc::QCProgramBuilder& builder,
                             const Instruction& instruction,
                             mlir::ValueRange qubits,
                             llvm::ArrayRef<ParameterValue> parameters);

static void
emitModifiedOperation(mlir::qc::QCProgramBuilder& builder,
                      const Instruction& instruction, mlir::ValueRange qubits,
                      const ModifiedQubitArity arity,
                      const LocalParameters& localParameters,
                      const GlobalParameters& globalParameters,
                      llvm::function_ref<void(mlir::ValueRange)> emitBase) {
  auto targets = qubits.drop_front(arity.controls);
  const auto emitModifiers = [&](auto&& self, const size_t count,
                                 mlir::ValueRange targetArguments) -> void {
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
      builder.inv(targetArguments, [&](mlir::ValueRange innerArguments) {
        self(self, count - 1U, innerArguments);
      });
      return;
    case GateModifierKind::Power: {
      const auto exponent = parameterValue(builder, modifier.exponent,
                                           localParameters, globalParameters);
      builder.pow(exponent, targetArguments,
                  [&](mlir::ValueRange innerArguments) {
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
               [&](mlir::ValueRange targetArguments) {
                 emitModifiers(emitModifiers, instruction.modifiers.size(),
                               targetArguments);
               });
}

static void emitModifiedGate(mlir::qc::QCProgramBuilder& builder,
                             const Instruction& instruction,
                             mlir::ValueRange qubits,
                             const llvm::ArrayRef<ParameterValue> parameters,
                             const LocalParameters& localParameters,
                             const GlobalParameters& globalParameters) {
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
      globalParameters, [&](mlir::ValueRange targetArguments) {
        emitStandardGate(builder, instruction, targetArguments, parameters);
      });
}

void emitStandardGate(mlir::qc::QCProgramBuilder& builder,
                      const Instruction& instruction, mlir::ValueRange qubits,
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
                 [&](mlir::ValueRange targetArguments) {
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

static void emitGate(mlir::qc::QCProgramBuilder& builder,
                     const Instruction& instruction,
                     const llvm::ArrayRef<mlir::Value> allQubits,
                     const llvm::ArrayRef<uint32_t> qubitMap,
                     const LocalParameters& localParameters,
                     const GlobalParameters& globalParameters) {
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
        parameterValue(builder, parameter, localParameters, globalParameters));
  }

  const llvm::ArrayRef<mlir::Value> qubitRange(qubits);
  if (!instruction.modifiers.empty()) {
    emitModifiedGate(builder, instruction, qubitRange, parameters,
                     localParameters, globalParameters);
    return;
  }
  emitStandardGate(builder, instruction, qubitRange, parameters);
}

[[nodiscard]] static std::vector<Register>
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

[[nodiscard]] static mlir::Type expressionType(mlir::OpBuilder& builder,
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

[[nodiscard]] static mlir::Value
integerConstant(mlir::ImplicitLocOpBuilder& builder, const uint32_t width,
                const uint64_t value) {
  const auto type = builder.getIntegerType(width);
  const auto attribute =
      builder.getIntegerAttr(type, llvm::APInt(width, value, false));
  return mlir::arith::ConstantOp::create(builder, attribute).getResult();
}

[[nodiscard]] static mlir::Value
castInteger(mlir::ImplicitLocOpBuilder& builder, mlir::Value value,
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

namespace {
struct ClassicalBitRef {
  mlir::Value storage;
  int64_t index;
};
} // namespace

[[nodiscard]] static mlir::Value
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
  return builder.loadClassicalBit(bit.storage, bit.index);
}

[[nodiscard]] static mlir::Value
packRegister(mlir::qc::QCProgramBuilder& builder,
             const llvm::ArrayRef<ClassicalBitRef> classicalBits,
             const llvm::ArrayRef<uint32_t> rootClbitMap, const Register& reg) {
  if (reg.bits.empty() || reg.bits.size() > 64U) {
    throw std::runtime_error(
        "Qiskit classical registers must contain between 1 and 64 bits");
  }
  const auto width = static_cast<uint32_t>(reg.bits.size());
  const auto type = builder.getIntegerType(width);
  llvm::SmallVector<mlir::Value> terms;
  terms.reserve(reg.bits.size());
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
    terms.push_back(bit);
  }
  while (terms.size() > 1U) {
    const auto reducedSize = (terms.size() + 1U) / 2U;
    for (size_t index = 0U; index < reducedSize; ++index) {
      const auto left = 2U * index;
      if (left + 1U < terms.size()) {
        terms[index] =
            mlir::arith::OrIOp::create(builder, terms[left], terms[left + 1U])
                .getResult();
      } else {
        terms[index] = terms[left];
      }
    }
    terms.resize(reducedSize);
  }
  return terms.front();
}

[[nodiscard]] static mlir::Value
emitExpression(mlir::qc::QCProgramBuilder& builder,
               const Expression& expression,
               const llvm::ArrayRef<ClassicalBitRef> classicalBits,
               const llvm::ArrayRef<uint32_t> rootClbitMap) {
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
  case ExpressionKind::ClassicalBit:
    return loadClassicalBit(builder, classicalBits, rootClbitMap,
                            expression.bit);
  case ExpressionKind::ClassicalRegister: {
    const auto target = llvm::dyn_cast<mlir::IntegerType>(resultType);
    if (!target) {
      throw std::runtime_error(
          "Qiskit classical-register expressions must have Uint type");
    }
    return castInteger(
        builder,
        packRegister(builder, classicalBits, rootClbitMap, expression.reg),
        target);
  }
  case ExpressionKind::Cast: {
    auto operand =
        emitExpression(builder, *expression.left, classicalBits, rootClbitMap);
    if (operand.getType() == resultType) {
      return operand;
    }
    if (expression.type == ClassicalType::Bool) {
      if (const auto source =
              llvm::dyn_cast<mlir::IntegerType>(operand.getType())) {
        return mlir::arith::CmpIOp::create(
                   builder, mlir::arith::CmpIPredicate::ne, operand,
                   integerConstant(builder, source.getWidth(), 0U))
            .getResult();
      }
      if (operand.getType().isF64()) {
        return mlir::arith::CmpFOp::create(builder,
                                           mlir::arith::CmpFPredicate::UNE,
                                           operand, floatConstant(builder, 0.0))
            .getResult();
      }
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
    auto operand =
        emitExpression(builder, *expression.left, classicalBits, rootClbitMap);
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
    auto left =
        emitExpression(builder, *expression.left, classicalBits, rootClbitMap);
    if (expression.binaryOperation == BinaryOperation::LogicAnd ||
        expression.binaryOperation == BinaryOperation::LogicOr) {
      if (!left.getType().isInteger(1)) {
        throw std::runtime_error(
            "Qiskit logical operation requires Boolean operands");
      }
      const auto emitRight = [&]() {
        auto right = emitExpression(builder, *expression.right, classicalBits,
                                    rootClbitMap);
        if (!right.getType().isInteger(1)) {
          throw std::runtime_error(
              "Qiskit logical operation requires Boolean operands");
        }
        return right;
      };
      const auto isAnd =
          expression.binaryOperation == BinaryOperation::LogicAnd;
      return mlir::scf::IfOp::create(
                 builder, left,
                 [&](mlir::OpBuilder&, mlir::Location) {
                   mlir::scf::YieldOp::create(
                       builder,
                       isAnd ? emitRight() : builder.boolConstant(true));
                 },
                 [&](mlir::OpBuilder&, mlir::Location) {
                   mlir::scf::YieldOp::create(
                       builder,
                       isAnd ? builder.boolConstant(false) : emitRight());
                 })
          .getResult(0);
    }
    auto right =
        emitExpression(builder, *expression.right, classicalBits, rootClbitMap);
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
      return mlir::arith::AndIOp::create(builder, left, right).getResult();
    case BinaryOperation::BitOr:
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
    auto target =
        emitExpression(builder, *expression.left, classicalBits, rootClbitMap);
    auto index =
        emitExpression(builder, *expression.right, classicalBits, rootClbitMap);
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
    auto shifted =
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

[[nodiscard]] static mlir::Value
emitCondition(mlir::qc::QCProgramBuilder& builder,
              const ClassicalTarget& target,
              const llvm::ArrayRef<ClassicalBitRef> classicalBits,
              const llvm::ArrayRef<uint32_t> rootClbitMap) {
  switch (target.kind) {
  case ClassicalTargetKind::ClassicalBit: {
    auto actual =
        loadClassicalBit(builder, classicalBits, rootClbitMap, target.bit);
    return mlir::arith::CmpIOp::create(builder, mlir::arith::CmpIPredicate::eq,
                                       actual,
                                       builder.boolConstant(target.expectedBit))
        .getResult();
  }
  case ClassicalTargetKind::ClassicalRegister: {
    auto actual = castInteger(
        builder, packRegister(builder, classicalBits, rootClbitMap, target.reg),
        builder.getIntegerType(target.width));
    auto expected =
        integerConstant(builder, target.width, target.expectedRegister);
    return mlir::arith::CmpIOp::create(builder, mlir::arith::CmpIPredicate::eq,
                                       actual, expected)
        .getResult();
  }
  case ClassicalTargetKind::Expression: {
    auto condition = emitExpression(builder, *target.expression, classicalBits,
                                    rootClbitMap);
    if (!condition.getType().isInteger(1)) {
      throw std::runtime_error(
          "Qiskit control-flow condition expression must have Boolean type");
    }
    return condition;
  }
  }
  throw std::runtime_error("unknown normalized Qiskit condition type");
}

[[nodiscard]] static mlir::Value
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
    value = emitExpression(builder, *target.expression, classicalBits,
                           rootClbitMap);
    break;
  }
  if (!llvm::isa<mlir::IntegerType>(value.getType())) {
    throw std::runtime_error("Qiskit switch targets must be Boolean or Uint");
  }
  return mlir::arith::IndexCastUIOp::create(builder, builder.getIndexType(),
                                            value)
      .getResult();
}

static void translateCircuit(mlir::qc::QCProgramBuilder& builder,
                             const CircuitReader& circuit,
                             llvm::ArrayRef<uint32_t> qubitMap,
                             llvm::ArrayRef<uint32_t> clbitMap,
                             llvm::ArrayRef<uint32_t> rootQubitMap,
                             llvm::ArrayRef<uint32_t> rootClbitMap,
                             llvm::ArrayRef<mlir::Value> allQubits,
                             llvm::ArrayRef<ClassicalBitRef> classicalBits,
                             const LocalParameters& localParameters,
                             const GlobalParameters& globalParameters,
                             size_t definitionDepth, size_t controlFlowDepth);

[[nodiscard]] static int64_t rangeLength(const Loop& loop) {
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

static void requireExactLoopParameter(const int64_t value) {
  constexpr auto maxExactDoubleInteger = static_cast<int64_t>(1ULL << 53U);
  if (value < -maxExactDoubleInteger || value > maxExactDoubleInteger) {
    throw std::runtime_error(
        "Qiskit loop parameter integer cannot be represented exactly as f64");
  }
}

[[nodiscard]] static mlir::Value
loopParameterValue(mlir::qc::QCProgramBuilder& builder, mlir::Value iteration,
                   const Loop& loop) {
  auto counter =
      mlir::arith::IndexCastOp::create(builder, builder.getI64Type(), iteration)
          .getResult();
  auto offset = mlir::arith::MulIOp::create(builder, counter,
                                            builder.intConstant(loop.step))
                    .getResult();
  auto value = mlir::arith::AddIOp::create(
                   builder, builder.intConstant(loop.start), offset)
                   .getResult();
  return mlir::arith::SIToFPOp::create(builder, builder.getF64Type(), value)
      .getResult();
}

static void translateControlFlow(mlir::qc::QCProgramBuilder& builder,
                                 const ControlFlowReader& controlFlow,
                                 llvm::ArrayRef<mlir::Value> allQubits,
                                 llvm::ArrayRef<ClassicalBitRef> classicalBits,
                                 llvm::ArrayRef<uint32_t> rootQubitMap,
                                 llvm::ArrayRef<uint32_t> rootClbitMap,
                                 const LocalParameters& localParameters,
                                 const GlobalParameters& globalParameters,
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
                     globalParameters, definitionDepth, controlFlowDepth + 1U);
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
    auto value = emitCondition(builder, condition, classicalBits, rootClbitMap);
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
          const auto* symbol = loop.parameter->getSymbol();
          if (symbol == nullptr) {
            throw std::runtime_error(
                "Qiskit for-loop parameter is not a symbol");
          }
          parameters[symbol->name] =
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
    auto* const containingBlock = builder.getInsertionBlock();
    builder.scfFor(0, count, 1, [&](mlir::Value iteration) {
      auto parameters = localParameters;
      if (loop.parameter) {
        const auto* symbol = loop.parameter->getSymbol();
        if (symbol == nullptr) {
          throw std::runtime_error("Qiskit for-loop parameter is not a symbol");
        }
        parameters[symbol->name] = loopParameterValue(builder, iteration, loop);
      }
      translateBlock(*body, parameters);
    });
    if (loop.parameter) {
      const auto* symbol = loop.parameter->getSymbol();
      if (symbol != nullptr && symbol->group) {
        mlir::cast<mlir::scf::ForOp>(&containingBlock->back())
            ->setAttr(
                mlir::mqt::MQTDialect::ParameterGroupAttrHelper::getNameStr(),
                parameterGroupAttribute(builder, *symbol->group));
      }
    }
    return;
  }
  case ControlFlowKind::Switch: {
    const auto cases = controlFlow.switchCases();
    if (cases.size() != controlFlow.numBlocks()) {
      throw std::runtime_error(
          "Qiskit switch case metadata does not match its blocks");
    }
    auto target = emitSwitchTarget(builder, controlFlow.switchTarget(),
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
                      const GlobalParameters& globalParameters,
                      const size_t definitionDepth,
                      const size_t controlFlowDepth) {
  builder.gphase(parameterValue(builder, circuit.globalPhase(), localParameters,
                                globalParameters));
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
                     classicalBits, localParameters, globalParameters,
                     definitionDepth + 1U, controlFlowDepth);
  };

  for (size_t index = 0; index < circuit.numInstructions(); ++index) {
    const auto instruction = circuit.instruction(index);
    switch (instruction.kind) {
    case OperationKind::Gate:
      if (instruction.standardGate) {
        emitGate(builder, instruction, allQubits, qubitMap, localParameters,
                 globalParameters);
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
      auto targets = std::span{operands}.subspan(arity.controls);
      std::ranges::reverse(targets);
      const auto dimension = int64_t{1} << arity.targets;
      const auto type = mlir::RankedTensorType::get(
          {dimension, dimension}, mlir::ComplexType::get(builder.getF64Type()));
      const auto values = circuit.unitary(index);
      const auto matrix = mlir::DenseElementsAttr::get(
          type, llvm::ArrayRef<std::complex<double>>(values));
      emitModifiedOperation(builder, instruction, operands, arity,
                            localParameters, globalParameters,
                            [&](mlir::ValueRange targetArguments) {
                              builder.unitary(targetArguments, matrix);
                            });
    } break;
    case OperationKind::ControlFlow: {
      const auto controlFlow = circuit.controlFlow(index);
      translateControlFlow(builder, *controlFlow, allQubits, classicalBits,
                           rootQubitMap, rootClbitMap, localParameters,
                           globalParameters, definitionDepth, controlFlowDepth);
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

namespace {
struct ExpansionSummary {
  size_t operations = 0U;
  size_t definitionDepth = 0U;
  size_t controlFlowDepth = 0U;
};

struct ExpansionCountState {
  llvm::DenseMap<uintptr_t, ExpansionSummary> definitions;
  llvm::DenseSet<uintptr_t> activeDefinitions;
};
} // namespace

static void addExpandedOperations(size_t& total, const size_t additional) {
  if (additional > MAX_EXPANDED_OPERATIONS - total) {
    throw std::runtime_error(
        "Qiskit instruction expansion exceeds 10000000 operations");
  }
  total += additional;
}

[[nodiscard]] static size_t repeatedOperations(const size_t operations,
                                               const size_t repetitions) {
  if (repetitions != 0U && operations > MAX_EXPANDED_OPERATIONS / repetitions) {
    throw std::runtime_error(
        "Qiskit instruction expansion exceeds 10000000 operations");
  }
  return operations * repetitions;
}

[[nodiscard]] static ExpansionSummary
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

static void validateCircuit(const CircuitReader& circuit,
                            const ValidationParameters& localParameters,
                            const ValidationParameters& freeParameters,
                            llvm::StringSet<>& parameterNames,
                            uint32_t rootQubits, uint32_t rootClbits,
                            size_t definitionDepth, size_t controlFlowDepth);

static void validateExpression(const Expression& expression,
                               const uint32_t rootClbits) {
  if ((expression.type == ClassicalType::Bool && expression.width != 1U) ||
      (expression.type == ClassicalType::Uint &&
       (expression.width == 0U || expression.width > 64U)) ||
      (expression.type == ClassicalType::Float && expression.width != 64U)) {
    throw std::runtime_error(
        "Qiskit classical expression has an invalid type width");
  }
  const auto requireOperand = [&](const std::unique_ptr<Expression>& operand) {
    if (!operand) {
      throw std::runtime_error(
          "Qiskit classical expression has a missing operand");
    }
    validateExpression(*operand, rootClbits);
  };
  const auto sameType = [](const Expression& first, const Expression& second) {
    return first.type == second.type && first.width == second.width;
  };
  const auto hasType = [](const Expression& value, const ClassicalType type) {
    return value.type == type;
  };
  const auto requireCompatible = [](const bool compatible) {
    if (!compatible) {
      throw std::runtime_error(
          "Qiskit classical expression has incompatible operator and operand "
          "types");
    }
  };
  switch (expression.kind) {
  case ExpressionKind::Value:
    return;
  case ExpressionKind::ClassicalBit:
    if (expression.type != ClassicalType::Bool || expression.width != 1U ||
        expression.bit >= rootClbits) {
      throw std::runtime_error(
          "Qiskit classical-bit expression has an invalid reference");
    }
    return;
  case ExpressionKind::ClassicalRegister: {
    if (expression.type != ClassicalType::Uint || expression.reg.bits.empty() ||
        expression.reg.bits.size() > 64U ||
        expression.width < expression.reg.bits.size()) {
      throw std::runtime_error(
          "Qiskit classical-register expression has an invalid type");
    }
    llvm::DenseSet<uint32_t> seen;
    for (const auto bit : expression.reg.bits) {
      if (bit >= rootClbits || !seen.insert(bit).second) {
        throw std::runtime_error(
            "Qiskit classical-register expression has an invalid bit");
      }
    }
    return;
  }
  case ExpressionKind::Unary: {
    requireOperand(expression.left);
    const auto& operand = *expression.left;
    switch (expression.unaryOperation) {
    case UnaryOperation::BitNot:
      requireCompatible((hasType(operand, ClassicalType::Bool) ||
                         hasType(operand, ClassicalType::Uint)) &&
                        sameType(expression, operand));
      return;
    case UnaryOperation::LogicNot:
      requireCompatible(hasType(expression, ClassicalType::Bool) &&
                        hasType(operand, ClassicalType::Bool));
      return;
    case UnaryOperation::Negate:
      requireCompatible(hasType(expression, ClassicalType::Float) &&
                        hasType(operand, ClassicalType::Float));
      return;
    }
    return;
  }
  case ExpressionKind::Cast:
    requireOperand(expression.left);
    return;
  case ExpressionKind::Binary: {
    requireOperand(expression.left);
    requireOperand(expression.right);
    const auto& left = *expression.left;
    const auto& right = *expression.right;
    switch (expression.binaryOperation) {
    case BinaryOperation::BitAnd:
    case BinaryOperation::BitOr:
    case BinaryOperation::BitXor:
      requireCompatible(sameType(left, right) && sameType(expression, left) &&
                        (hasType(left, ClassicalType::Bool) ||
                         hasType(left, ClassicalType::Uint)));
      return;
    case BinaryOperation::LogicAnd:
    case BinaryOperation::LogicOr:
      requireCompatible(hasType(expression, ClassicalType::Bool) &&
                        hasType(left, ClassicalType::Bool) &&
                        hasType(right, ClassicalType::Bool));
      return;
    case BinaryOperation::Equal:
    case BinaryOperation::NotEqual:
      requireCompatible(hasType(expression, ClassicalType::Bool) &&
                        sameType(left, right));
      return;
    case BinaryOperation::Less:
    case BinaryOperation::LessEqual:
    case BinaryOperation::Greater:
    case BinaryOperation::GreaterEqual:
      requireCompatible(hasType(expression, ClassicalType::Bool) &&
                        sameType(left, right) &&
                        (hasType(left, ClassicalType::Uint) ||
                         hasType(left, ClassicalType::Float)));
      return;
    case BinaryOperation::ShiftLeft:
    case BinaryOperation::ShiftRight:
      requireCompatible(hasType(left, ClassicalType::Uint) &&
                        hasType(right, ClassicalType::Uint) &&
                        sameType(expression, left));
      return;
    case BinaryOperation::Add:
    case BinaryOperation::Subtract:
    case BinaryOperation::Multiply:
    case BinaryOperation::Divide:
      requireCompatible(sameType(left, right) && sameType(expression, left) &&
                        (hasType(left, ClassicalType::Uint) ||
                         hasType(left, ClassicalType::Float)));
      return;
    }
    return;
  }
  case ExpressionKind::Index:
    requireOperand(expression.left);
    requireOperand(expression.right);
    requireCompatible(hasType(expression, ClassicalType::Bool) &&
                      hasType(*expression.left, ClassicalType::Uint) &&
                      hasType(*expression.right, ClassicalType::Uint));
    return;
  }
}

static void validateTarget(const ClassicalTarget& target,
                           const uint32_t rootClbits) {
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
    validateExpression(*target.expression, rootClbits);
    return;
  }
}

static void validateControlFlow(const ControlFlowReader& controlFlow,
                                ValidationParameters localParameters,
                                const ValidationParameters& freeParameters,
                                llvm::StringSet<>& parameterNames,
                                const uint32_t rootQubits,
                                const uint32_t rootClbits,
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
      const auto* symbol = loop.parameter->getSymbol();
      if (symbol == nullptr || symbol->name.empty()) {
        throw std::runtime_error(
            "Qiskit for-loop parameter has invalid symbol metadata");
      }
      if (!parameterNames.insert(symbol->name).second) {
        throw std::runtime_error(
            "Qiskit circuit contains distinct parameters with the same name");
      }
      localParameters[symbol->name] = *loop.parameter;
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
    validateCircuit(*block, localParameters, freeParameters, parameterNames,
                    rootQubits, rootClbits, definitionDepth,
                    controlFlowDepth + 1U);
  }
}

static void validateDefinition(const CircuitReader& circuit, const size_t index,
                               const ValidationParameters& localParameters,
                               const ValidationParameters& freeParameters,
                               llvm::StringSet<>& parameterNames,
                               const size_t definitionDepth,
                               const size_t controlFlowDepth) {
  if (definitionDepth >= MAX_DEFINITION_DEPTH) {
    throw std::runtime_error(
        "Qiskit instruction definitions exceed the nesting limit of 64");
  }
  const auto definition = circuit.definition(index);
  validateCircuit(*definition, localParameters, freeParameters, parameterNames,
                  definition->numQubits(), definition->numClbits(),
                  definitionDepth + 1U, controlFlowDepth);
}

void validateCircuit(const CircuitReader& circuit,
                     const ValidationParameters& localParameters,
                     const ValidationParameters& freeParameters,
                     llvm::StringSet<>& parameterNames,
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
  validateParameter(circuit.globalPhase(), localParameters, freeParameters);

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
      validateParameter(parameter, localParameters, freeParameters);
    }
    for (const auto& modifier : instruction.modifiers) {
      if (modifier.kind == GateModifierKind::Power) {
        validateParameter(modifier.exponent, localParameters, freeParameters);
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
      validateDefinition(circuit, index, localParameters, freeParameters,
                         parameterNames, definitionDepth, controlFlowDepth);
      break;
    case OperationKind::Unknown:
      if (!instruction.modifiers.empty()) {
        throw std::runtime_error(
            "Qiskit circuit import does not support modifiers on custom "
            "instructions");
      }
      validateDefinition(circuit, index, localParameters, freeParameters,
                         parameterNames, definitionDepth, controlFlowDepth);
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
      validateControlFlow(*controlFlow, localParameters, freeParameters,
                          parameterNames, rootQubits, rootClbits,
                          definitionDepth, controlFlowDepth);
      break;
    }
    case OperationKind::Delay:
      throw std::runtime_error("Qiskit delay instructions are not supported");
    }
  }
}

mlir::QCProgram importCircuit(const nb::handle circuit) {
  auto translation = selectTranslation();
  auto view = translation->openCircuit(circuit);
  const auto freeParameters = view->parameters();
  ValidationParameters freeParameterSymbols;
  llvm::StringSet<> parameterNames;
  ParameterGroupRegistry parameterGroups;
  for (const auto& parameter : freeParameters) {
    const auto* symbol = parameter.getSymbol();
    if (symbol == nullptr || symbol->name.empty()) {
      throw std::runtime_error(
          "Qiskit circuit returned an invalid free parameter");
    }
    if (!parameterNames.insert(symbol->name).second) {
      throw std::runtime_error(
          "Qiskit circuit contains distinct parameters with the same name");
    }
    if (symbol->group) {
      const auto& group = *symbol->group;
      if (group.index >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::runtime_error(
            "Qiskit parameter-vector index cannot be represented by MLIR");
      }
      parameterGroups.add(group);
    }
    freeParameterSymbols.try_emplace(symbol->name, parameter);
  }

  ExpansionCountState expansion;
  static_cast<void>(expansionSummary(*view, expansion));
  validateCircuit(*view, {}, freeParameterSymbols, parameterNames,
                  view->numQubits(), view->numClbits(), 0U, 0U);
  const auto quantumRegisters = circuitRegisters(*view, true);
  const auto classicalRegisters = circuitRegisters(*view, false);
  for (const auto& reg :
       llvm::concat<const Register>(quantumRegisters, classicalRegisters)) {
    if (!parameterNames.insert(reg.name).second) {
      throw std::runtime_error(
          "Qiskit circuit requires unique parameter and register names");
    }
  }
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
      resultTypes.push_back(mlir::cbit::RegisterType::get(
          context.get(), static_cast<int64_t>(looseClbits)));
    }
    for (const auto& reg : classicalRegisters) {
      resultTypes.push_back(mlir::cbit::RegisterType::get(
          context.get(), static_cast<int64_t>(reg.bits.size())));
    }
  }
  builder.initialize(resultTypes);
  auto function = llvm::cast<mlir::func::FuncOp>(
      builder.getInsertionBlock()->getParentOp());
  GlobalParameters globalParameters;
  for (const auto& parameter : freeParameters) {
    const auto* symbol = parameter.getSymbol();
    llvm::SmallVector<mlir::NamedAttribute> argumentAttributes{
        builder.getNamedAttr(
            mlir::mqt::MQTDialect::InputNameAttrHelper::getNameStr(),
            builder.getStringAttr(symbol->name))};
    if (symbol->group) {
      argumentAttributes.push_back(builder.getNamedAttr(
          mlir::mqt::MQTDialect::ParameterGroupAttrHelper::getNameStr(),
          parameterGroupAttribute(builder, *symbol->group)));
    }
    const auto index = function.getNumArguments();
    // MLIR types are handles. Converting FloatType to Type keeps the same
    // storage and does not slice object state.
    const mlir::Type parameterType = builder.getF64Type();
    if (failed(function.insertArgument(
            index, parameterType, builder.getDictionaryAttr(argumentAttributes),
            builder.getLoc()))) {
      throw std::runtime_error(
          "failed to create a compiler input for a Qiskit parameter");
    }
    globalParameters[symbol->name] = function.getArgument(index);
  }

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
    auto storage =
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
                   qubits, classicalBits, {}, globalParameters, 0U, 0U);

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
