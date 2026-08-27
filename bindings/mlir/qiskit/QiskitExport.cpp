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
#include "mlir/Compiler/Programs.h"
#include "mlir/Compiler/Target.h"
#include "mlir/Dialect/CBit/IR/CBitAttributes.h"
#include "mlir/Dialect/CBit/IR/CBitDialect.h"
#include "mlir/Dialect/CBit/IR/CBitOps.h"
#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/MQT/Utils/ConstantFolding.h"
#include "mlir/Dialect/MQT/Utils/Parameters.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCInterfaces.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QC/Translation/StandardGate.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/WalkResult.h>
#include <nanobind/nanobind.h>

#include <algorithm>
#include <array>
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
#include <vector>

namespace mqt::bindings::qiskit {

constexpr size_t MAX_EXPORT_CONTROL_FLOW_DEPTH = 64U;
constexpr size_t MAX_EXPORT_EXPRESSION_DEPTH = 64U;
constexpr size_t MAX_EXPORT_EXPRESSION_NODES = 4096U;

namespace {
struct ExportedControlFlow;

struct ExportedInstruction {
  enum class Kind : uint8_t {
    Gate,
    Measure,
    Reset,
    Barrier,
    Unitary,
    ControlFlow,
  };
  Kind kind = Kind::Gate;
  StandardGateMapping gate;
  std::vector<uint32_t> qubits;
  std::vector<uint32_t> clbits;
  std::vector<Parameter> parameters;
  std::vector<std::complex<double>> matrix;
  uint32_t unitaryControls = 0;
  std::unique_ptr<ExportedControlFlow> controlFlow;
};

using ExportedParameters = llvm::DenseMap<mlir::Value, Parameter>;

struct ExportedCircuit {
  Parameter globalPhase = Parameter::number(0.0);
  std::vector<ExportedInstruction> instructions;
};

struct ExportedControlFlow {
  ControlFlowKind kind = ControlFlowKind::IfElse;
  ClassicalTarget target;
  Loop loop;
  std::vector<SwitchCase> switchCases;
  std::vector<ExportedCircuit> blocks;
};
} // namespace

[[noreturn]] static void throwExportedParameterExpressionSizeError() {
  throw std::runtime_error("QC parameter expression exceeds the supported " +
                           std::to_string(MAX_PARAMETER_EXPRESSION_NODES) +
                           "-node size");
}

[[noreturn]] static void throwExportedParameterExpressionDepthError() {
  throw std::runtime_error("QC parameter expression exceeds the supported " +
                           std::to_string(MAX_PARAMETER_EXPRESSION_DEPTH) +
                           "-level nesting depth");
}

[[nodiscard]] static Parameter numberParameter(const double value) {
  return Parameter::number(value);
}

[[nodiscard]] static Parameter unaryParameter(const UnaryParameterKind kind,
                                              Parameter operand) {
  return Parameter::unary(kind, std::move(operand));
}

[[nodiscard]] static Parameter binaryParameter(const BinaryParameterKind kind,
                                               Parameter left,
                                               Parameter right) {
  return Parameter::binary(kind, std::move(left), std::move(right));
}

[[nodiscard]] static Parameter
exportParameterImpl(mlir::Value value, ExportedParameters& parameters,
                    const size_t depth, size_t& nodes) {
  if (depth > MAX_PARAMETER_EXPRESSION_DEPTH) {
    throwExportedParameterExpressionDepthError();
  }
  if (const auto cached = parameters.find(value); cached != parameters.end()) {
    return cached->second;
  }
  if (++nodes > MAX_PARAMETER_EXPRESSION_NODES) {
    throwExportedParameterExpressionSizeError();
  }
  if (const auto number = mlir::mqt::valueToDouble(value)) {
    auto result = numberParameter(*number);
    parameters.try_emplace(value, result);
    return result;
  }
  if (!value.getType().isF64()) {
    throw std::runtime_error(
        "Qiskit circuit export requires f64 scalar parameters");
  }
  auto* const operation = value.getDefiningOp();
  if (operation == nullptr || operation->getNumResults() != 1U ||
      operation->getResult(0) != value) {
    throw std::runtime_error(
        "Qiskit circuit export cannot resolve an unnamed scalar parameter");
  }
  const auto unary = [&](const UnaryParameterKind kind) {
    if (operation->getNumOperands() != 1U) {
      throw std::runtime_error("QC parameter operation '" +
                               operation->getName().getStringRef().str() +
                               "' has invalid arity");
    }
    return unaryParameter(kind,
                          exportParameterImpl(operation->getOperand(0),
                                              parameters, depth + 1U, nodes));
  };
  const auto binary = [&](const BinaryParameterKind kind) {
    if (operation->getNumOperands() != 2U) {
      throw std::runtime_error("QC parameter operation '" +
                               operation->getName().getStringRef().str() +
                               "' has invalid arity");
    }
    auto left = exportParameterImpl(operation->getOperand(0), parameters,
                                    depth + 1U, nodes);
    auto right = exportParameterImpl(operation->getOperand(1), parameters,
                                     depth + 1U, nodes);
    return binaryParameter(kind, std::move(left), std::move(right));
  };

  Parameter result;
  if (llvm::isa<mlir::arith::AddFOp>(*operation)) {
    result = binary(BinaryParameterKind::Add);
  } else if (llvm::isa<mlir::arith::SubFOp>(*operation)) {
    result = binary(BinaryParameterKind::Subtract);
  } else if (llvm::isa<mlir::arith::MulFOp>(*operation)) {
    result = binary(BinaryParameterKind::Multiply);
  } else if (llvm::isa<mlir::arith::DivFOp>(*operation)) {
    result = binary(BinaryParameterKind::Divide);
  } else if (llvm::isa<mlir::math::PowFOp>(*operation)) {
    result = binary(BinaryParameterKind::Power);
  } else if (llvm::isa<mlir::arith::NegFOp>(*operation)) {
    result = unary(UnaryParameterKind::Negate);
  } else if (llvm::isa<mlir::math::SinOp>(*operation)) {
    result = unary(UnaryParameterKind::Sin);
  } else if (llvm::isa<mlir::math::CosOp>(*operation)) {
    result = unary(UnaryParameterKind::Cos);
  } else if (llvm::isa<mlir::math::TanOp>(*operation)) {
    result = unary(UnaryParameterKind::Tan);
  } else if (llvm::isa<mlir::math::AsinOp>(*operation)) {
    result = unary(UnaryParameterKind::ArcSin);
  } else if (llvm::isa<mlir::math::AcosOp>(*operation)) {
    result = unary(UnaryParameterKind::ArcCos);
  } else if (llvm::isa<mlir::math::AtanOp>(*operation)) {
    result = unary(UnaryParameterKind::ArcTan);
  } else if (llvm::isa<mlir::math::ExpOp>(*operation)) {
    result = unary(UnaryParameterKind::Exp);
  } else if (llvm::isa<mlir::math::LogOp>(*operation)) {
    result = unary(UnaryParameterKind::Log);
  } else if (llvm::isa<mlir::math::AbsFOp>(*operation)) {
    result = unary(UnaryParameterKind::Abs);
  } else {
    throw std::runtime_error(
        "Qiskit circuit export does not support scalar parameter operation '" +
        operation->getName().getStringRef().str() + "'");
  }
  parameters.try_emplace(value, result);
  return result;
}

[[nodiscard]] static Parameter exportParameter(mlir::Value value,
                                               ExportedParameters& parameters) {
  size_t nodes = 0U;
  return exportParameterImpl(value, parameters, 1U, nodes);
}

static void validateExportParameterImpl(const Parameter& parameter,
                                        const size_t depth, size_t& nodes,
                                        llvm::StringSet<>& names) {
  if (depth > MAX_PARAMETER_EXPRESSION_DEPTH) {
    throwExportedParameterExpressionDepthError();
  }
  if (++nodes > MAX_PARAMETER_EXPRESSION_NODES) {
    throwExportedParameterExpressionSizeError();
  }
  if (parameter.getNumber() != nullptr) {
    return;
  }
  if (const auto* symbol = parameter.getSymbol()) {
    if (symbol->name.empty()) {
      throw std::runtime_error("QC parameter symbol has an invalid name");
    }
    if (symbol->name.find('\0') != std::string::npos) {
      throw std::runtime_error(
          "QC parameter symbol name contains a null character");
    }
    names.insert(symbol->name);
    return;
  }
  if (const auto* unary = parameter.getUnary()) {
    validateExportParameterImpl(*unary->operand, depth + 1U, nodes, names);
    return;
  }
  if (const auto* binary = parameter.getBinary()) {
    validateExportParameterImpl(*binary->left, depth + 1U, nodes, names);
    validateExportParameterImpl(*binary->right, depth + 1U, nodes, names);
    return;
  }
  throw std::runtime_error("unknown QC parameter expression");
}

static void validateExportParameter(const Parameter& parameter,
                                    llvm::StringSet<>& names) {
  size_t nodes = 0U;
  validateExportParameterImpl(parameter, 1U, nodes, names);
}

[[nodiscard]] static bool
isParameterExpressionOperation(mlir::Operation& operation) {
  return llvm::isa<mlir::arith::AddFOp, mlir::arith::SubFOp,
                   mlir::arith::MulFOp, mlir::arith::DivFOp,
                   mlir::arith::NegFOp, mlir::math::PowFOp, mlir::math::SinOp,
                   mlir::math::CosOp, mlir::math::TanOp, mlir::math::AsinOp,
                   mlir::math::AcosOp, mlir::math::AtanOp, mlir::math::ExpOp,
                   mlir::math::LogOp, mlir::math::AbsFOp>(operation);
}

[[nodiscard]] static uint32_t checkedIndex(const int64_t index,
                                           const std::string_view kind) {
  if (index < 0 ||
      std::cmp_greater(index, std::numeric_limits<uint32_t>::max())) {
    throw std::runtime_error(std::string(kind) +
                             " index cannot be represented by Qiskit");
  }
  return static_cast<uint32_t>(index);
}

[[nodiscard]] static uint32_t checkedIndex(const uint64_t index,
                                           const std::string_view kind) {
  if (index > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error(std::string(kind) +
                             " index cannot be represented by Qiskit");
  }
  return static_cast<uint32_t>(index);
}

[[nodiscard]] static mlir::CompilerTarget::SiteId
checkedTargetSiteId(const uint64_t index) {
  using SiteId = mlir::CompilerTarget::SiteId;
  if (!std::in_range<SiteId>(index)) {
    throw std::runtime_error(
        "QC static qubit index cannot be represented by a compiler target "
        "site ID");
  }
  return static_cast<SiteId>(index);
}

[[nodiscard]] static uint32_t checkedAdd(const uint32_t left,
                                         const uint32_t right,
                                         const std::string_view kind) {
  if (right > std::numeric_limits<uint32_t>::max() - left) {
    throw std::runtime_error(std::string(kind) +
                             " count cannot be represented by Qiskit");
  }
  return left + right;
}

namespace {
struct ExportState {
  struct ClassicalRegisterInfo {
    uint32_t base;
    uint32_t size;
    mlir::cbit::Initialization initialization;
  };

  llvm::DenseMap<mlir::Value, uint32_t> qubits;
  llvm::DenseMap<mlir::Value, uint32_t> quantumBases;
  llvm::DenseMap<mlir::Value, uint32_t> quantumSizes;
  llvm::DenseMap<mlir::Value, ClassicalRegisterInfo> classicalRegisterInfo;
  llvm::DenseMap<mlir::Value, llvm::DenseSet<uint32_t>> unconditionalWrites;
  llvm::DenseMap<mlir::Value, llvm::DenseSet<uint32_t>> measurementDestinations;
  llvm::DenseMap<mlir::Value, uint32_t> measurementResultBits;
  llvm::DenseSet<mlir::Operation*> expressionOperations;
  std::vector<Register> quantumRegisters;
  std::vector<Register> classicalRegisters;
  ExportedParameters parameters;
  std::vector<Parameter> inputParameters;
  llvm::StringSet<> parameterNames;
  ParameterGroupRegistry parameterGroups;
  size_t nextLoopParameter = 0U;
  uint32_t numQubits = 0;
  uint32_t numClbits = 0;
};
} // namespace

[[nodiscard]] static ParameterGroup
parameterGroup(const mlir::Attribute attribute) {
  const auto metadata = llvm::dyn_cast<mlir::DictionaryAttr>(attribute);
  if (!metadata || metadata.size() != 4U) {
    throw std::runtime_error(
        "Qiskit circuit export requires complete and valid parameter-group "
        "metadata");
  }
  const auto identity = metadata.getAs<mlir::StringAttr>("identity");
  const auto name = metadata.getAs<mlir::StringAttr>("name");
  const auto index = metadata.getAs<mlir::IntegerAttr>("index");
  const auto size = metadata.getAs<mlir::IntegerAttr>("size");
  if (!identity || !name || !index || !size || identity.getValue().empty() ||
      identity.getValue().contains('\0') || name.getValue().contains('\0') ||
      !index.getType().isInteger(64) || index.getInt() < 0 ||
      !size.getType().isInteger(64) || size.getInt() < 0) {
    throw std::runtime_error(
        "Qiskit circuit export requires complete and valid parameter-group "
        "metadata");
  }
  return {
      .identity = identity.str(),
      .name = name.str(),
      .index = static_cast<uint64_t>(index.getInt()),
      .size = static_cast<uint64_t>(size.getInt()),
  };
}

[[nodiscard]] static bool parameterUsesName(const Parameter& parameter,
                                            const std::string_view name) {
  if (const auto* symbol = parameter.getSymbol()) {
    return symbol->name == name;
  }
  if (const auto* unary = parameter.getUnary()) {
    return parameterUsesName(*unary->operand, name);
  }
  if (const auto* binary = parameter.getBinary()) {
    return parameterUsesName(*binary->left, name) ||
           parameterUsesName(*binary->right, name);
  }
  return false;
}

[[nodiscard]] static bool
circuitUsesParameterName(const ExportedCircuit& circuit,
                         const std::string_view name) {
  if (parameterUsesName(circuit.globalPhase, name)) {
    return true;
  }
  for (const auto& instruction : circuit.instructions) {
    if (llvm::any_of(instruction.parameters, [&](const auto& parameter) {
          return parameterUsesName(parameter, name);
        })) {
      return true;
    }
    if (!instruction.controlFlow) {
      continue;
    }
    if (llvm::any_of(instruction.controlFlow->blocks, [&](const auto& block) {
          return circuitUsesParameterName(block, name);
        })) {
      return true;
    }
  }
  return false;
}

static void validateExportParameters(const ExportedCircuit& circuit,
                                     llvm::StringSet<>& usedNames) {
  const auto validate = [&](const Parameter& parameter) {
    validateExportParameter(parameter, usedNames);
  };
  validate(circuit.globalPhase);
  for (const auto& instruction : circuit.instructions) {
    for (const auto& parameter : instruction.parameters) {
      validate(parameter);
    }
    if (!instruction.controlFlow) {
      continue;
    }
    if (instruction.controlFlow->loop.parameter) {
      validate(*instruction.controlFlow->loop.parameter);
    }
    for (const auto& block : instruction.controlFlow->blocks) {
      validateExportParameters(block, usedNames);
    }
  }
}

static void validateExportParameters(const ExportedCircuit& circuit,
                                     const std::vector<Parameter>& inputs) {
  llvm::StringSet<> usedNames;
  validateExportParameters(circuit, usedNames);
  for (const auto& input : inputs) {
    const auto* symbol = input.getSymbol();
    if (symbol == nullptr) {
      throw std::runtime_error("QC program input is not a parameter symbol");
    }
    if (!usedNames.contains(symbol->name)) {
      throw std::runtime_error(
          "Qiskit circuit export cannot preserve unused named f64 program "
          "input '" +
          symbol->name + "'");
    }
  }
}

static void collectParameters(mlir::func::FuncOp function, ExportState& state) {
  for (const auto [index, argument] :
       llvm::enumerate(function.getArguments())) {
    const auto name = function.getArgAttrOfType<mlir::StringAttr>(
        index, mlir::mqt::MQTDialect::InputNameAttrHelper::getNameStr());
    if (!argument.getType().isF64() || !name) {
      throw std::runtime_error(
          "Qiskit circuit export requires named f64 program inputs");
    }
    if (name.getValue().contains('\0')) {
      throw std::runtime_error(
          "Qiskit circuit export does not support parameter names with null "
          "characters");
    }
    if (!state.parameterNames.insert(name.getValue()).second) {
      throw std::runtime_error(
          "Qiskit circuit export requires unique parameter names");
    }

    const auto groupAttribute = function.getArgAttr(
        index, mlir::mqt::MQTDialect::ParameterGroupAttrHelper::getNameStr());
    std::optional<ParameterGroup> group;
    if (groupAttribute) {
      group = parameterGroup(groupAttribute);
      if (name.getValue() !=
          group->name + "[" + std::to_string(group->index) + "]") {
        throw std::runtime_error(
            "Qiskit parameter input name does not match its group and index");
      }
      state.parameterGroups.add(*group);
    }
    auto parameter = Parameter::symbol(name.str(), std::move(group));
    state.parameters[argument] = parameter;
    state.inputParameters.push_back(std::move(parameter));
  }
}

static void addGlobalPhase(ExportedCircuit& circuit, const Parameter& phase) {
  if (const auto* number = phase.getNumber()) {
    if (const auto* globalNumber = circuit.globalPhase.getNumber()) {
      const auto sum = globalNumber->value + number->value;
      if (!std::isfinite(sum)) {
        throw std::runtime_error(
            "QC global phase cannot be represented by Qiskit");
      }
      circuit.globalPhase = Parameter::number(sum);
      return;
    }
    if (std::abs(number->value) <= mlir::mqt::PARAMETER_COMPARISON_TOLERANCE) {
      return;
    }
  } else if (const auto* globalNumber = circuit.globalPhase.getNumber();
             globalNumber != nullptr &&
             std::abs(globalNumber->value) <=
                 mlir::mqt::PARAMETER_COMPARISON_TOLERANCE) {
    circuit.globalPhase = phase;
    return;
  }
  circuit.globalPhase = binaryParameter(BinaryParameterKind::Add,
                                        std::move(circuit.globalPhase), phase);
}

[[nodiscard]] static std::vector<uint32_t>
mapQubits(mlir::ValueRange values,
          const llvm::DenseMap<mlir::Value, uint32_t>& qubits) {
  std::vector<uint32_t> result;
  result.reserve(values.size());
  for (auto value : values) {
    const auto found = qubits.find(value);
    if (found == qubits.end()) {
      throw std::runtime_error(
          "QC to Qiskit export could not resolve a qubit operand");
    }
    result.push_back(found->second);
  }
  return result;
}

[[nodiscard]] static ExportedInstruction
collectUnitaryInstruction(mlir::Operation& operation,
                          const llvm::DenseMap<mlir::Value, uint32_t>& qubits,
                          ExportedParameters& parameters);

[[nodiscard]] static std::vector<mlir::Operation*>
modifierBodyOperations(mlir::Region& region) {
  if (!llvm::hasSingleElement(region)) {
    throw std::runtime_error(
        "QC to Qiskit export requires single-block modifier regions");
  }
  std::vector<mlir::Operation*> operations;
  for (auto& operation : region.front()) {
    if (!llvm::isa<mlir::qc::YieldOp, mlir::arith::ConstantOp>(operation) &&
        !isParameterExpressionOperation(operation)) {
      operations.push_back(&operation);
    }
  }
  return operations;
}

[[nodiscard]] static llvm::DenseMap<mlir::Value, uint32_t>
modifierQubitMap(const llvm::DenseMap<mlir::Value, uint32_t>& outer,
                 mlir::Block& block, mlir::ValueRange operands) {
  if (block.getNumArguments() != operands.size()) {
    throw std::runtime_error(
        "QC modifier block arguments do not match its qubit operands");
  }
  auto result = outer;
  for (auto [argument, operand] :
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

static void invertGate(ExportedInstruction& instruction) {
  if (instruction.kind == ExportedInstruction::Kind::Unitary) {
    if (instruction.unitaryControls > instruction.qubits.size()) {
      throw std::runtime_error("QC unitary has an invalid control count");
    }
    const auto numTargets =
        instruction.qubits.size() - instruction.unitaryControls;
    if (numTargets >= std::numeric_limits<size_t>::digits / 2U) {
      throw std::runtime_error("QC unitary matrix is too large to represent");
    }
    const auto dimension = size_t{1} << numTargets;
    if (dimension * dimension != instruction.matrix.size()) {
      throw std::runtime_error("QC unitary matrix has an invalid dimension");
    }
    auto source = instruction.matrix;
    for (size_t row = 0U; row < dimension; ++row) {
      for (size_t column = 0U; column < dimension; ++column) {
        instruction.matrix[(row * dimension) + column] =
            std::conj(source[(column * dimension) + row]);
      }
    }
    return;
  }
  using Gate = mlir::qc::StandardGate;
  switch (instruction.gate.gate) {
  case Gate::Id:
  case Gate::X:
  case Gate::Y:
  case Gate::Z:
  case Gate::H:
  case Gate::SWAP:
  case Gate::ECR:
    return;
  case Gate::S:
    instruction.gate.gate = Gate::Sdg;
    return;
  case Gate::Sdg:
    instruction.gate.gate = Gate::S;
    return;
  case Gate::T:
    instruction.gate.gate = Gate::Tdg;
    return;
  case Gate::Tdg:
    instruction.gate.gate = Gate::T;
    return;
  case Gate::SX:
    instruction.gate.gate = Gate::SXdg;
    return;
  case Gate::SXdg:
    instruction.gate.gate = Gate::SX;
    return;
  default:
    break;
  }

  if (instruction.gate.gate == Gate::P || instruction.gate.gate == Gate::RX ||
      instruction.gate.gate == Gate::RY || instruction.gate.gate == Gate::RZ ||
      instruction.gate.gate == Gate::RXX ||
      instruction.gate.gate == Gate::RYY ||
      instruction.gate.gate == Gate::RZZ ||
      instruction.gate.gate == Gate::RZX) {
    if (instruction.parameters.empty()) {
      throw std::runtime_error("QC inverse modifier has invalid arity");
    }
    instruction.parameters.front() = unaryParameter(
        UnaryParameterKind::Negate, std::move(instruction.parameters.front()));
    return;
  }
  if (instruction.gate.gate == Gate::U3 &&
      instruction.parameters.size() == 3U) {
    auto parameters = std::move(instruction.parameters);
    instruction.parameters = {
        unaryParameter(UnaryParameterKind::Negate, std::move(parameters[0])),
        unaryParameter(UnaryParameterKind::Negate, std::move(parameters[2])),
        unaryParameter(UnaryParameterKind::Negate, std::move(parameters[1]))};
    return;
  }
  throw std::runtime_error(
      "QC inverse modifier has no supported Qiskit gate equivalent");
}

[[nodiscard]] ExportedInstruction
collectUnitaryInstruction(mlir::Operation& operation,
                          const llvm::DenseMap<mlir::Value, uint32_t>& qubits,
                          ExportedParameters& parameters) {
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
                                 .gate = {mlir::qc::StandardGate::CU, 0},
                                 .qubits = {controls.front(), targets.front()}};
      for (auto parameter : unitary.getParameters()) {
        result.parameters.push_back(exportParameter(parameter, parameters));
      }
      result.parameters.push_back(
          exportParameter(phase.getTheta(), parameters));
      return result;
    }
    if (bodyOperations.size() != 1U) {
      throw std::runtime_error(
          "QC control export requires one standard gate in the modifier body");
    }
    auto result = collectUnitaryInstruction(*bodyOperations.front(), nestedMap,
                                            parameters);
    auto& numControls = result.kind == ExportedInstruction::Kind::Unitary
                            ? result.unitaryControls
                            : result.gate.controls;
    if (std::cmp_greater(controls.size(),
                         std::numeric_limits<uint32_t>::max() - numControls)) {
      throw std::runtime_error("QC control count cannot be represented");
    }
    numControls += static_cast<uint32_t>(controls.size());
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
    auto result = collectUnitaryInstruction(*bodyOperations.front(), nestedMap,
                                            parameters);
    invertGate(result);
    return result;
  }
  if (auto power = llvm::dyn_cast<mlir::qc::PowOp>(operation)) {
    auto bodyOperations = modifierBodyOperations(power.getRegion());
    if (bodyOperations.size() != 1U) {
      throw std::runtime_error(
          "QC power export requires one standard gate in the modifier body");
    }
    const auto exponent = exportParameter(power.getExponent(), parameters);
    const auto* number = exponent.getNumber();
    if (number == nullptr || (number->value != 1.0 && number->value != -1.0)) {
      throw std::runtime_error(
          "QC power export supports only constant exponents 1 and -1");
    }
    auto nestedMap =
        modifierQubitMap(qubits, power.getRegion().front(), power.getQubits());
    auto result = collectUnitaryInstruction(*bodyOperations.front(), nestedMap,
                                            parameters);
    if (number->value == -1.0) {
      invertGate(result);
    }
    return result;
  }
  if (auto unitary = llvm::dyn_cast<mlir::qc::UnitaryOp>(operation)) {
    const auto matrix =
        llvm::cast<mlir::DenseElementsAttr>(unitary.getMatrix());
    std::vector<std::complex<double>> values;
    values.reserve(matrix.size());
    llvm::append_range(values, matrix.getValues<std::complex<double>>());
    auto targetQubits = mapQubits(unitary.getQubits(), qubits);
    std::ranges::reverse(targetQubits);
    return {.kind = ExportedInstruction::Kind::Unitary,
            .qubits = std::move(targetQubits),
            .matrix = std::move(values)};
  }
  auto gate = llvm::dyn_cast<mlir::qc::UnitaryOpInterface>(operation);
  if (!gate || llvm::isa<mlir::qc::GPhaseOp, mlir::qc::BarrierOp>(operation)) {
    throw std::runtime_error(
        "QC modifier body is not a constructible standard Qiskit gate");
  }
  ExportedInstruction result{.kind = ExportedInstruction::Kind::Gate,
                             .qubits = mapQubits(gate.getTargets(), qubits)};
  const auto* descriptor =
      mlir::qc::lookupStandardGateByOperationSymbol(gate.getBaseSymbol());
  if (descriptor == nullptr ||
      descriptor->gate == mlir::qc::StandardGate::GPhase ||
      descriptor->gate == mlir::qc::StandardGate::BuiltinU ||
      descriptor->gate == mlir::qc::StandardGate::CU) {
    throw std::runtime_error(
        "QC operation has no constructible standard Qiskit gate");
  }
  result.gate.gate = descriptor->gate;
  for (auto parameter : gate.getParameters()) {
    result.parameters.push_back(exportParameter(parameter, parameters));
  }
  return result;
}

static void collectResources(mlir::func::FuncOp function, ExportState& state,
                             const mlir::CompilerTarget* const target) {
  llvm::DenseSet<uint32_t> staticIndices;
  for (auto& operation : function.getBody().front()) {
    if (auto staticQubit = llvm::dyn_cast<mlir::qc::StaticOp>(operation)) {
      uint32_t index = 0;
      if (target != nullptr) {
        const auto vertex =
            target->vertexForSite(checkedTargetSiteId(staticQubit.getIndex()));
        if (!vertex) {
          throw std::runtime_error(
              "QC static qubit is not a site of the supplied compiler target");
        }
        index = checkedIndex(static_cast<uint64_t>(*vertex), "qubit");
      } else {
        index = checkedIndex(staticQubit.getIndex(), "qubit");
      }
      if (index == std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("qubit count cannot be represented by Qiskit");
      }
      if (!staticIndices.insert(index).second) {
        throw std::runtime_error(
            "QC to Qiskit export does not support aliased static qubits");
      }
      state.qubits[staticQubit.getQubit()] = index;
      state.numQubits = std::max(state.numQubits, index + 1U);
    }
  }
  for (auto& operation : function.getBody().front()) {
    if (auto alloc = llvm::dyn_cast<mlir::qc::AllocOp>(operation)) {
      if (target != nullptr) {
        throw std::runtime_error(
            "target-aware Qiskit export requires statically mapped qubits");
      }
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
      throw std::runtime_error(
          "QC to Qiskit export supports only static one-dimensional resource "
          "allocations");
    }
    if (llvm::isa<mlir::qc::QubitType>(type.getElementType())) {
      if (target != nullptr) {
        throw std::runtime_error(
            "target-aware Qiskit export requires statically mapped qubits");
      }
      const auto size = checkedIndex(type.getShape()[0], "qubit-register size");
      state.quantumBases[alloc.getResult()] = state.numQubits;
      state.quantumSizes[alloc.getResult()] = size;
      if (const auto name = operation.getAttrOfType<mlir::StringAttr>(
              mlir::mqt::MQTDialect::RegisterNameAttrHelper::getNameStr())) {
        Register reg{.name = name.str()};
        reg.bits.resize(size);
        std::iota(reg.bits.begin(), reg.bits.end(), state.numQubits);
        state.quantumRegisters.push_back(std::move(reg));
      }
      state.numQubits = checkedAdd(state.numQubits, size, "qubit");
    } else {
      throw std::runtime_error(
          "QC to Qiskit export encountered an unsupported memory allocation");
    }
  }
  function.walk([&](mlir::memref::LoadOp load) {
    if (!llvm::isa<mlir::qc::QubitType>(load.getResult().getType()) ||
        load.getIndices().size() != 1U) {
      return;
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
  });

  auto returnOp =
      llvm::dyn_cast<mlir::func::ReturnOp>(function.getBody().front().back());
  if (!returnOp) {
    throw std::runtime_error(
        "QC to Qiskit export requires an entry-function return");
  }
  if (returnOp.getNumOperands() == 1U) {
    auto result = returnOp.getOperand(0);
    const auto sentinel = mlir::getConstantIntValue(result);
    if (result.getType().isInteger(64) && sentinel && *sentinel == 0) {
      return;
    }
  }
  llvm::DenseSet<mlir::Value> returnedRegisters;
  for (auto result : returnOp.getOperands()) {
    const auto type =
        llvm::dyn_cast<mlir::cbit::RegisterType>(result.getType());
    if (!type) {
      throw std::runtime_error(
          "QC to Qiskit export supports only CBit function return values");
    }
    if (!returnedRegisters.insert(result).second) {
      throw std::runtime_error(
          "QC to Qiskit export does not support duplicate result registers");
    }
    auto alloc = result.getDefiningOp<mlir::cbit::AllocOp>();
    if (!alloc || alloc->getBlock() != &function.getBody().front()) {
      throw std::runtime_error(
          "QC to Qiskit export requires direct result-register allocations");
    }
    const auto size = checkedIndex(type.getWidth(), "classical-register size");
    state.classicalRegisterInfo[result] = {.base = state.numClbits,
                                           .size = size,
                                           .initialization =
                                               alloc.getInitialization()};
    if (const auto name = alloc->getAttrOfType<mlir::StringAttr>(
            mlir::mqt::MQTDialect::RegisterNameAttrHelper::getNameStr())) {
      Register reg{.name = name.str()};
      reg.bits.resize(size);
      std::iota(reg.bits.begin(), reg.bits.end(), state.numClbits);
      state.classicalRegisters.push_back(std::move(reg));
    }
    state.numClbits = checkedAdd(state.numClbits, size, "classical-bit");
  }
}

[[nodiscard]] static std::optional<uint64_t>
constantUnsignedInteger(mlir::Value value) {
  auto constant = value.getDefiningOp<mlir::arith::ConstantOp>();
  const auto integer =
      constant ? llvm::dyn_cast<mlir::IntegerAttr>(constant.getValue())
               : mlir::IntegerAttr{};
  if (!integer || integer.getValue().getBitWidth() > 64U) {
    return std::nullopt;
  }
  return integer.getValue().getZExtValue();
}

static void setExpressionType(Expression& expression, const mlir::Type type) {
  if (type.isInteger(1)) {
    expression.type = ClassicalType::Bool;
    expression.width = 1U;
    return;
  }
  if (const auto integer = llvm::dyn_cast<mlir::IntegerType>(type)) {
    if (integer.getWidth() == 0U || integer.getWidth() > 64U) {
      throw std::runtime_error(
          "Qiskit unsigned classical values must be between 1 and 64 bits");
    }
    expression.type = ClassicalType::Uint;
    expression.width = integer.getWidth();
    return;
  }
  if (type.isF64()) {
    expression.type = ClassicalType::Float;
    expression.width = 64U;
    return;
  }
  throw std::runtime_error(
      "Qiskit classical expressions support only Bool, Uint, and Float");
}

[[nodiscard]] static uint32_t classicalBitIndex(mlir::cbit::LoadOp load,
                                                const ExportState& state) {
  if (!load.getResult().getType().isInteger(1)) {
    throw std::runtime_error(
        "Qiskit classical expressions require a static classical-bit load");
  }
  const auto info = state.classicalRegisterInfo.find(load.getReg());
  const auto index = mlir::getConstantIntValue(load.getIndex());
  if (info == state.classicalRegisterInfo.end() || !index) {
    throw std::runtime_error(
        "Qiskit classical expressions could not resolve a classical bit");
  }
  const auto checked = checkedIndex(*index, "classical-bit");
  if (checked >= info->second.size) {
    throw std::runtime_error(
        "Qiskit classical expression uses an out-of-bounds classical bit");
  }
  if (info->second.initialization != mlir::cbit::Initialization::Zero) {
    const auto written = state.unconditionalWrites.find(load.getReg());
    if (written == state.unconditionalWrites.end() ||
        !written->second.contains(checked)) {
      throw std::runtime_error(
          "Qiskit classical expression loads an undefined classical bit "
          "before an unconditional measurement write");
    }
  }
  return checkedAdd(info->second.base, checked, "classical-bit");
}

[[noreturn]] static void throwClassicalExpressionSizeError() {
  throw std::runtime_error(
      "QC classical expression exceeds the size limit of 4096 nodes");
}

[[noreturn]] static void throwClassicalExpressionDepthError() {
  throw std::runtime_error(
      "QC classical expressions exceed the nesting limit of 64");
}

static void countExpressionNode(size_t& nodeCount) {
  if (++nodeCount > MAX_EXPORT_EXPRESSION_NODES) {
    throwClassicalExpressionSizeError();
  }
}

namespace {
struct PackedRegister {
  Register reg;
  llvm::SmallPtrSet<mlir::Operation*, 16> operations;
};
} // namespace

[[nodiscard]] static std::optional<PackedRegister>
matchPackedRegister(mlir::Value value, ExportState& state,
                    mlir::Block& evaluationBlock);

[[nodiscard]] static std::unique_ptr<Expression>
exportExpressionImpl(mlir::Value value, ExportState& state,
                     mlir::Block& evaluationBlock, const size_t depth,
                     size_t& nodeCount) {
  if (depth >= MAX_EXPORT_EXPRESSION_DEPTH) {
    throwClassicalExpressionDepthError();
  }
  countExpressionNode(nodeCount);
  auto* operation = value.getDefiningOp();
  if (operation == nullptr) {
    throw std::runtime_error(
        "Qiskit classical expressions cannot capture an SSA block argument");
  }
  if (!llvm::isa<mlir::arith::ConstantOp>(operation) &&
      operation->getBlock() != &evaluationBlock) {
    throw std::runtime_error(
        "Qiskit classical expressions cannot capture a computed SSA value "
        "across a control-flow region");
  }

  auto result = std::make_unique<Expression>();
  setExpressionType(*result, value.getType());
  if (const auto measured = state.measurementResultBits.find(value);
      measured != state.measurementResultBits.end()) {
    result->kind = ExpressionKind::ClassicalBit;
    result->bit = measured->second;
    return result;
  }
  if (result->type == ClassicalType::Uint) {
    if (auto packed = matchPackedRegister(value, state, evaluationBlock)) {
      result->kind = ExpressionKind::ClassicalRegister;
      result->reg = std::move(packed->reg);
      state.expressionOperations.insert(packed->operations.begin(),
                                        packed->operations.end());
      return result;
    }
  }
  if (auto constant = llvm::dyn_cast<mlir::arith::ConstantOp>(operation)) {
    result->kind = ExpressionKind::Value;
    if (const auto integer =
            llvm::dyn_cast<mlir::IntegerAttr>(constant.getValue())) {
      if (result->type == ClassicalType::Bool) {
        result->boolValue = !integer.getValue().isZero();
      } else if (result->type == ClassicalType::Uint) {
        result->uintValue = integer.getValue().getZExtValue();
      } else {
        throw std::runtime_error(
            "Qiskit Float expressions require a floating-point constant");
      }
      return result;
    }
    const auto floating = llvm::dyn_cast<mlir::FloatAttr>(constant.getValue());
    if (!floating || result->type != ClassicalType::Float) {
      throw std::runtime_error(
          "Qiskit classical expression contains an unsupported constant");
    }
    result->floatValue = floating.getValueAsDouble();
    if (!std::isfinite(result->floatValue)) {
      throw std::runtime_error(
          "Qiskit classical floating-point literals must be finite");
    }
    return result;
  }
  if (auto load = llvm::dyn_cast<mlir::cbit::LoadOp>(operation)) {
    result->kind = ExpressionKind::ClassicalBit;
    result->bit = classicalBitIndex(load, state);
    state.expressionOperations.insert(operation);
    return result;
  }
  if (auto ifOp = llvm::dyn_cast<mlir::scf::IfOp>(operation)) {
    if (ifOp.getNumResults() != 1U || !value.getType().isInteger(1) ||
        ifOp.getElseRegion().empty()) {
      throw std::runtime_error(
          "Qiskit classical expressions support only canonical "
          "short-circuit Boolean scf.if results");
    }
    auto& thenBlock = ifOp.getThenRegion().front();
    auto& elseBlock = ifOp.getElseRegion().front();
    auto thenYield = llvm::cast<mlir::scf::YieldOp>(thenBlock.getTerminator());
    auto elseYield = llvm::cast<mlir::scf::YieldOp>(elseBlock.getTerminator());
    auto thenValue = thenYield.getOperand(0);
    auto elseValue = elseYield.getOperand(0);
    mlir::Value right;
    if (mlir::matchPattern(elseValue, mlir::m_Zero())) {
      result->binaryOperation = BinaryOperation::LogicAnd;
      right = thenValue;
    } else if (mlir::matchPattern(thenValue, mlir::m_One())) {
      result->binaryOperation = BinaryOperation::LogicOr;
      right = elseValue;
    } else {
      throw std::runtime_error(
          "Qiskit classical expressions support only canonical "
          "short-circuit Boolean scf.if results");
    }
    auto condition = exportExpressionImpl(
        ifOp.getCondition(), state, *ifOp->getBlock(), depth + 1U, nodeCount);
    auto rightExpression = exportExpressionImpl(
        right, state,
        result->binaryOperation == BinaryOperation::LogicAnd ? thenBlock
                                                             : elseBlock,
        depth + 1U, nodeCount);
    const auto validateBranch = [&](mlir::Block& branch) {
      for (auto& nested : branch.without_terminator()) {
        if (!llvm::isa<mlir::arith::ConstantOp>(nested) &&
            !state.expressionOperations.contains(&nested)) {
          throw std::runtime_error(
              "Qiskit Boolean scf.if expressions must be side-effect free");
        }
      }
    };
    validateBranch(thenBlock);
    validateBranch(elseBlock);
    state.expressionOperations.insert(operation);
    result->kind = ExpressionKind::Binary;
    result->left = std::move(condition);
    result->right = std::move(rightExpression);
    return result;
  }

  const auto unary = [&](const ExpressionKind kind, mlir::Value operand) {
    result->kind = kind;
    result->left = exportExpressionImpl(operand, state, evaluationBlock,
                                        depth + 1U, nodeCount);
    state.expressionOperations.insert(operation);
    return std::move(result);
  };
  const auto binary = [&](const BinaryOperation kind, mlir::Value left,
                          mlir::Value right) {
    result->kind = ExpressionKind::Binary;
    result->binaryOperation = kind;
    result->left = exportExpressionImpl(left, state, evaluationBlock,
                                        depth + 1U, nodeCount);
    result->right = exportExpressionImpl(right, state, evaluationBlock,
                                         depth + 1U, nodeCount);
    state.expressionOperations.insert(operation);
    return std::move(result);
  };

  if (llvm::isa<mlir::arith::ExtUIOp, mlir::arith::UIToFPOp,
                mlir::arith::FPToUIOp>(operation)) {
    return unary(ExpressionKind::Cast, operation->getOperand(0));
  }
  if (auto cast = llvm::dyn_cast<mlir::arith::TruncIOp>(operation)) {
    if (!cast.getType().isInteger(1)) {
      return unary(ExpressionKind::Cast, cast.getIn());
    }
    result->kind = ExpressionKind::Index;
    if (auto shift = cast.getIn().getDefiningOp<mlir::arith::ShRUIOp>()) {
      result->left = exportExpressionImpl(
          shift.getLhs(), state, evaluationBlock, depth + 1U, nodeCount);
      result->right = exportExpressionImpl(
          shift.getRhs(), state, evaluationBlock, depth + 1U, nodeCount);
      state.expressionOperations.insert(shift);
    } else {
      result->left = exportExpressionImpl(cast.getIn(), state, evaluationBlock,
                                          depth + 1U, nodeCount);
      countExpressionNode(nodeCount);
      auto zero = std::make_unique<Expression>();
      setExpressionType(*zero, cast.getIn().getType());
      zero->kind = ExpressionKind::Value;
      zero->uintValue = 0U;
      result->right = std::move(zero);
    }
    state.expressionOperations.insert(operation);
    return result;
  }
  if (auto cast = llvm::dyn_cast<mlir::arith::IndexCastUIOp>(operation)) {
    state.expressionOperations.insert(operation);
    return exportExpressionImpl(cast.getIn(), state, evaluationBlock,
                                depth + 1U, nodeCount);
  }
  if (auto op = llvm::dyn_cast<mlir::arith::CmpIOp>(operation)) {
    auto kind = BinaryOperation::Equal;
    switch (op.getPredicate()) {
    case mlir::arith::CmpIPredicate::eq:
      kind = BinaryOperation::Equal;
      break;
    case mlir::arith::CmpIPredicate::ne:
      kind = BinaryOperation::NotEqual;
      break;
    case mlir::arith::CmpIPredicate::ult:
      kind = BinaryOperation::Less;
      break;
    case mlir::arith::CmpIPredicate::ule:
      kind = BinaryOperation::LessEqual;
      break;
    case mlir::arith::CmpIPredicate::ugt:
      kind = BinaryOperation::Greater;
      break;
    case mlir::arith::CmpIPredicate::uge:
      kind = BinaryOperation::GreaterEqual;
      break;
    default:
      throw std::runtime_error(
          "Qiskit Uint expressions do not support signed comparisons");
    }
    return binary(kind, op.getLhs(), op.getRhs());
  }
  if (auto op = llvm::dyn_cast<mlir::arith::CmpFOp>(operation)) {
    auto kind = BinaryOperation::Equal;
    switch (op.getPredicate()) {
    case mlir::arith::CmpFPredicate::OEQ:
      kind = BinaryOperation::Equal;
      break;
    case mlir::arith::CmpFPredicate::UNE:
      kind = BinaryOperation::NotEqual;
      break;
    case mlir::arith::CmpFPredicate::OLT:
      kind = BinaryOperation::Less;
      break;
    case mlir::arith::CmpFPredicate::OLE:
      kind = BinaryOperation::LessEqual;
      break;
    case mlir::arith::CmpFPredicate::OGT:
      kind = BinaryOperation::Greater;
      break;
    case mlir::arith::CmpFPredicate::OGE:
      kind = BinaryOperation::GreaterEqual;
      break;
    default:
      throw std::runtime_error(
          "Qiskit Float expressions require ordered comparisons");
    }
    return binary(kind, op.getLhs(), op.getRhs());
  }
  if (auto op = llvm::dyn_cast<mlir::arith::AndIOp>(operation)) {
    return binary(value.getType().isInteger(1) ? BinaryOperation::LogicAnd
                                               : BinaryOperation::BitAnd,
                  op.getLhs(), op.getRhs());
  }
  if (auto op = llvm::dyn_cast<mlir::arith::OrIOp>(operation)) {
    return binary(value.getType().isInteger(1) ? BinaryOperation::LogicOr
                                               : BinaryOperation::BitOr,
                  op.getLhs(), op.getRhs());
  }
  if (auto op = llvm::dyn_cast<mlir::arith::XOrIOp>(operation)) {
    return binary(BinaryOperation::BitXor, op.getLhs(), op.getRhs());
  }
  if (auto op = llvm::dyn_cast<mlir::arith::ShLIOp>(operation)) {
    return binary(BinaryOperation::ShiftLeft, op.getLhs(), op.getRhs());
  }
  if (auto op = llvm::dyn_cast<mlir::arith::ShRUIOp>(operation)) {
    return binary(BinaryOperation::ShiftRight, op.getLhs(), op.getRhs());
  }
  if (llvm::isa<mlir::arith::AddIOp, mlir::arith::AddFOp>(operation)) {
    return binary(BinaryOperation::Add, operation->getOperand(0),
                  operation->getOperand(1));
  }
  if (llvm::isa<mlir::arith::SubIOp, mlir::arith::SubFOp>(operation)) {
    return binary(BinaryOperation::Subtract, operation->getOperand(0),
                  operation->getOperand(1));
  }
  if (llvm::isa<mlir::arith::MulIOp, mlir::arith::MulFOp>(operation)) {
    return binary(BinaryOperation::Multiply, operation->getOperand(0),
                  operation->getOperand(1));
  }
  if (llvm::isa<mlir::arith::DivUIOp, mlir::arith::DivFOp>(operation)) {
    return binary(BinaryOperation::Divide, operation->getOperand(0),
                  operation->getOperand(1));
  }
  if (auto op = llvm::dyn_cast<mlir::arith::NegFOp>(operation)) {
    result->unaryOperation = UnaryOperation::Negate;
    return unary(ExpressionKind::Unary, op.getOperand());
  }
  throw std::runtime_error(
      "unsupported QC classical operation in Qiskit export: " +
      operation->getName().getStringRef().str());
}

static void validateExpressionDepth(const Expression& expression,
                                    const size_t depth = 0U) {
  if (depth >= MAX_EXPORT_EXPRESSION_DEPTH) {
    throwClassicalExpressionDepthError();
  }
  if (expression.left) {
    validateExpressionDepth(*expression.left, depth + 1U);
  }
  if (expression.right) {
    validateExpressionDepth(*expression.right, depth + 1U);
  }
}

[[nodiscard]] static std::unique_ptr<Expression>
exportExpression(mlir::Value value, ExportState& state,
                 mlir::Block& evaluationBlock) {
  size_t nodeCount = 0U;
  auto result =
      exportExpressionImpl(value, state, evaluationBlock, 0U, nodeCount);
  validateExpressionDepth(*result);
  return result;
}

[[nodiscard]] std::optional<PackedRegister>
matchPackedRegister(mlir::Value value, ExportState& state,
                    mlir::Block& evaluationBlock) {
  auto type = llvm::dyn_cast<mlir::IntegerType>(value.getType());
  if (!type || type.getWidth() == 0U || type.getWidth() > 64U) {
    return std::nullopt;
  }
  std::vector<std::optional<uint32_t>> bits(type.getWidth());
  llvm::SmallPtrSet<mlir::Operation*, 16> operations;
  size_t nodeCount = 0U;
  const std::function<bool(mlir::Value, uint32_t, size_t)> collect =
      [&](mlir::Value current, const uint32_t shift, const size_t depth) {
        if (depth >= MAX_EXPORT_EXPRESSION_DEPTH ||
            ++nodeCount > MAX_EXPORT_EXPRESSION_NODES) {
          return false;
        }
        auto* operation = current.getDefiningOp();
        if (operation == nullptr) {
          return false;
        }
        if (auto constant =
                llvm::dyn_cast<mlir::arith::ConstantOp>(operation)) {
          const auto integer =
              llvm::dyn_cast<mlir::IntegerAttr>(constant.getValue());
          return integer && integer.getValue().isZero();
        }
        if (operation->getBlock() != &evaluationBlock) {
          return false;
        }
        if (auto op = llvm::dyn_cast<mlir::arith::OrIOp>(operation)) {
          operations.insert(operation);
          return collect(op.getLhs(), shift, depth + 1U) &&
                 collect(op.getRhs(), shift, depth + 1U);
        }
        if (auto op = llvm::dyn_cast<mlir::arith::ShLIOp>(operation)) {
          const auto amount = constantUnsignedInteger(op.getRhs());
          if (!amount || *amount >= bits.size() ||
              *amount > std::numeric_limits<uint32_t>::max() - shift) {
            return false;
          }
          operations.insert(operation);
          return collect(op.getLhs(), shift + static_cast<uint32_t>(*amount),
                         depth + 1U);
        }
        if (auto op = llvm::dyn_cast<mlir::arith::ExtUIOp>(operation)) {
          operations.insert(operation);
          return collect(op.getIn(), shift, depth + 1U);
        }
        auto load = llvm::dyn_cast<mlir::cbit::LoadOp>(operation);
        if (!load || shift >= bits.size() || bits[shift]) {
          return false;
        }
        bits[shift] = classicalBitIndex(load, state);
        operations.insert(operation);
        return true;
      };
  if (!collect(value, 0U, 0U) ||
      llvm::any_of(bits, [](const auto& bit) { return !bit.has_value(); })) {
    return std::nullopt;
  }
  Register reg;
  reg.bits.reserve(bits.size());
  llvm::DenseSet<uint32_t> seenBits;
  for (const auto bit : bits) {
    if (!seenBits.insert(*bit).second) {
      return std::nullopt;
    }
    reg.bits.push_back(*bit);
  }
  for (const auto& candidate : state.classicalRegisters) {
    if (candidate.bits == reg.bits) {
      reg.name = candidate.name;
      break;
    }
  }
  return PackedRegister{.reg = std::move(reg),
                        .operations = std::move(operations)};
}

static void acceptPackedRegister(PackedRegister& packed, ExportState& state) {
  state.expressionOperations.insert(packed.operations.begin(),
                                    packed.operations.end());
}

[[nodiscard]] static bool storesToValueRecursively(mlir::Operation& operation,
                                                   mlir::Value value) {
  return operation
      .walk([&](mlir::cbit::StoreOp store) {
        return store.getReg() == value ? mlir::WalkResult::interrupt()
                                       : mlir::WalkResult::advance();
      })
      .wasInterrupted();
}

static void validateClassicalSnapshot(mlir::Value expression,
                                      mlir::Operation& consumer) {
  llvm::DenseSet<mlir::Value> visited;
  llvm::SmallVector<mlir::cbit::LoadOp> loads;
  llvm::SmallVector<mlir::Value, 16> worklist{expression};
  while (!worklist.empty()) {
    auto value = worklist.pop_back_val();
    if (!visited.insert(value).second) {
      continue;
    }
    if (visited.size() > MAX_EXPORT_EXPRESSION_NODES) {
      throwClassicalExpressionSizeError();
    }
    auto* operation = value.getDefiningOp();
    if (operation == nullptr) {
      continue;
    }
    if (auto load = llvm::dyn_cast<mlir::cbit::LoadOp>(operation)) {
      loads.push_back(load);
      continue;
    }
    if (auto ifOp = llvm::dyn_cast<mlir::scf::IfOp>(operation)) {
      const auto resultIndex =
          llvm::cast<mlir::OpResult>(value).getResultNumber();
      for (auto& region : ifOp->getRegions()) {
        auto yield =
            llvm::cast<mlir::scf::YieldOp>(region.front().getTerminator());
        worklist.push_back(yield.getOperand(resultIndex));
      }
    }
    worklist.append(operation->operand_begin(), operation->operand_end());
  }
  for (auto load : loads) {
    mlir::Operation* anchor = load;
    auto* anchorBlock = load->getBlock();
    while (anchorBlock != consumer.getBlock()) {
      auto* parent = anchorBlock->getParentOp();
      auto parentIf = llvm::dyn_cast_if_present<mlir::scf::IfOp>(parent);
      if (!parentIf || parentIf.getNumResults() == 0U) {
        throw std::runtime_error(
            "Qiskit control-flow expressions cannot capture a classical "
            "snapshot across a region");
      }
      anchor = parent;
      anchorBlock = parent->getBlock();
    }
    if (!anchor->isBeforeInBlock(&consumer)) {
      throw std::runtime_error(
          "Qiskit control-flow expressions cannot capture a classical "
          "snapshot across a region");
    }
    for (auto* operation = anchor->getNextNode(); operation != &consumer;
         operation = operation->getNextNode()) {
      if (operation == nullptr) {
        throw std::runtime_error(
            "Qiskit control-flow expression does not dominate its consumer");
      }
      if (auto store = llvm::dyn_cast<mlir::cbit::StoreOp>(operation);
          store && store.getReg() == load.getReg()) {
        throw std::runtime_error(
            "Qiskit control-flow export cannot preserve a stale classical "
            "snapshot");
      }
      if (operation->getNumRegions() != 0U &&
          storesToValueRecursively(*operation, load.getReg())) {
        throw std::runtime_error(
            "Qiskit control-flow export cannot preserve a classical "
            "snapshot across nested control flow");
      }
    }
  }
}

[[nodiscard]] static ClassicalTarget
exportCondition(mlir::Value value, ExportState& state,
                mlir::Block& evaluationBlock, mlir::Operation& consumer) {
  if (!value.getType().isInteger(1)) {
    throw std::runtime_error(
        "Qiskit control-flow conditions must have Boolean type");
  }
  validateClassicalSnapshot(value, consumer);
  if (auto comparison = value.getDefiningOp<mlir::arith::CmpIOp>();
      comparison &&
      comparison.getPredicate() == mlir::arith::CmpIPredicate::eq) {
    for (auto [actual, expected] :
         std::array{std::pair{comparison.getLhs(), comparison.getRhs()},
                    std::pair{comparison.getRhs(), comparison.getLhs()}}) {
      const auto constant = constantUnsignedInteger(expected);
      if (!constant) {
        continue;
      }
      if (auto load = actual.getDefiningOp<mlir::cbit::LoadOp>();
          load && actual.getType().isInteger(1) && *constant <= 1U) {
        state.expressionOperations.insert(comparison);
        state.expressionOperations.insert(load);
        return {.kind = ClassicalTargetKind::ClassicalBit,
                .bit = classicalBitIndex(load, state),
                .expectedBit = *constant != 0U};
      }
      if (auto packed = matchPackedRegister(actual, state, evaluationBlock)) {
        if (packed->reg.bits.size() != 64U &&
            *constant >= (uint64_t{1} << packed->reg.bits.size())) {
          continue;
        }
        state.expressionOperations.insert(comparison);
        acceptPackedRegister(*packed, state);
        return {.kind = ClassicalTargetKind::ClassicalRegister,
                .reg = std::move(packed->reg),
                .expectedRegister = *constant,
                .width =
                    llvm::cast<mlir::IntegerType>(actual.getType()).getWidth()};
      }
    }
  }
  ClassicalTarget target{.kind = ClassicalTargetKind::Expression};
  target.expression = exportExpression(value, state, evaluationBlock);
  return target;
}

[[nodiscard]] static ClassicalTarget
exportSwitchTarget(mlir::Value value, ExportState& state,
                   mlir::Block& evaluationBlock, mlir::Operation& consumer) {
  validateClassicalSnapshot(value, consumer);
  if (auto cast = value.getDefiningOp<mlir::arith::IndexCastUIOp>()) {
    state.expressionOperations.insert(cast);
    value = cast.getIn();
  } else if (value.getType().isIndex()) {
    if (const auto constant = constantUnsignedInteger(value)) {
      auto expression = std::make_unique<Expression>();
      expression->kind = ExpressionKind::Value;
      expression->type = ClassicalType::Uint;
      expression->width = 64U;
      expression->uintValue = *constant;
      return {.kind = ClassicalTargetKind::Expression,
              .width = 64U,
              .expression = std::move(expression)};
    }
    throw std::runtime_error(
        "Qiskit switch targets require a constant index or an unsigned "
        "integer-to-index cast");
  }
  if (auto load = value.getDefiningOp<mlir::cbit::LoadOp>();
      load && value.getType().isInteger(1)) {
    state.expressionOperations.insert(load);
    return {.kind = ClassicalTargetKind::ClassicalBit,
            .bit = classicalBitIndex(load, state)};
  }
  if (auto packed = matchPackedRegister(value, state, evaluationBlock)) {
    acceptPackedRegister(*packed, state);
    return {.kind = ClassicalTargetKind::ClassicalRegister,
            .reg = std::move(packed->reg),
            .width = llvm::cast<mlir::IntegerType>(value.getType()).getWidth()};
  }
  ClassicalTarget target{.kind = ClassicalTargetKind::Expression};
  target.expression = exportExpression(value, state, evaluationBlock);
  if (target.expression->type == ClassicalType::Float) {
    throw std::runtime_error("Qiskit switch targets must be Boolean or Uint");
  }
  target.width = target.expression->width;
  return target;
}

[[nodiscard]] static int64_t checkedAffine(const int64_t multiplier,
                                           const int64_t value,
                                           const int64_t offset,
                                           const std::string_view kind) {
  const llvm::APInt wideMultiplier(128U, static_cast<uint64_t>(multiplier),
                                   true);
  const llvm::APInt wideValue(128U, static_cast<uint64_t>(value), true);
  const llvm::APInt wideOffset(128U, static_cast<uint64_t>(offset), true);
  const auto result = (wideMultiplier * wideValue) + wideOffset;
  if (!result.isSignedIntN(64U)) {
    throw std::runtime_error(std::string(kind) +
                             " cannot be represented safely by Qiskit");
  }
  return result.getSExtValue();
}

[[nodiscard]] static uint64_t
rangeLength(const int64_t lower, const int64_t upper, const int64_t step) {
  if (lower >= upper) {
    return 0U;
  }
  const auto distance =
      static_cast<uint64_t>(upper) - static_cast<uint64_t>(lower);
  return ((distance - 1U) / static_cast<uint64_t>(step)) + 1U;
}

namespace {
struct LoopParameterProjection {
  mlir::Value value;
  int64_t multiplier = 1;
  int64_t offset = 0;
  llvm::SmallPtrSet<mlir::Operation*, 4> operations;
};
} // namespace

[[nodiscard]] static mlir::Operation* uniqueUser(mlir::Value value) {
  return value.hasOneUse() ? *value.getUsers().begin() : nullptr;
}

[[nodiscard]] static std::optional<LoopParameterProjection>
matchLoopParameterProjection(mlir::scf::ForOp loop) {
  auto* castOperation = uniqueUser(loop.getInductionVar());
  auto cast =
      llvm::dyn_cast_if_present<mlir::arith::IndexCastOp>(castOperation);
  if (!cast || !cast.getOut().getType().isInteger(64)) {
    return std::nullopt;
  }
  LoopParameterProjection projection;
  projection.operations.insert(castOperation);
  auto current = cast.getOut();

  if (auto* user = uniqueUser(current)) {
    if (auto multiply = llvm::dyn_cast<mlir::arith::MulIOp>(user)) {
      auto other =
          multiply.getLhs() == current ? multiply.getRhs() : multiply.getLhs();
      const auto constant = mlir::getConstantIntValue(other);
      if (!constant) {
        return std::nullopt;
      }
      projection.multiplier = *constant;
      projection.operations.insert(user);
      current = multiply.getResult();
    }
  }
  if (auto* user = uniqueUser(current)) {
    if (auto add = llvm::dyn_cast<mlir::arith::AddIOp>(user)) {
      auto other = add.getLhs() == current ? add.getRhs() : add.getLhs();
      const auto constant = mlir::getConstantIntValue(other);
      if (!constant) {
        return std::nullopt;
      }
      projection.offset = *constant;
      projection.operations.insert(user);
      current = add.getResult();
    }
  }
  auto* conversionOperation = uniqueUser(current);
  auto conversion =
      llvm::dyn_cast_if_present<mlir::arith::SIToFPOp>(conversionOperation);
  if (!conversion || !conversion.getOut().getType().isF64()) {
    return std::nullopt;
  }
  projection.operations.insert(conversionOperation);
  projection.value = conversion.getOut();
  return projection;
}

[[nodiscard]] static ExportedCircuit
collectBlock(mlir::Block& block, ExportState& state, size_t controlFlowDepth);

static void validateControlFlowDepth(const size_t controlFlowDepth) {
  if (controlFlowDepth >= MAX_EXPORT_CONTROL_FLOW_DEPTH) {
    throw std::runtime_error("QC control flow exceeds the nesting limit of 64");
  }
}

[[nodiscard]] static bool isFusableMeasurementStore(mlir::qc::MeasureOp measure,
                                                    mlir::cbit::StoreOp store) {
  if (store.getValue() != measure.getResult() ||
      measure->getBlock() != store->getBlock()) {
    return false;
  }
  for (auto* operation = measure->getNextNode(); operation != store;
       operation = operation->getNextNode()) {
    if (operation == nullptr ||
        !llvm::isa<mlir::arith::ConstantOp>(operation)) {
      return false;
    }
  }
  return true;
}

static void validateExpressionBlock(mlir::Block& block,
                                    const ExportState& state) {
  for (auto& operation : block.without_terminator()) {
    if (llvm::isa<mlir::arith::ConstantOp>(operation) ||
        state.expressionOperations.contains(&operation)) {
      continue;
    }
    throw std::runtime_error(
        "Qiskit while-loop condition regions must contain only classical "
        "expression operations");
  }
}

[[nodiscard]] static std::unique_ptr<ExportedControlFlow>
collectIf(mlir::scf::IfOp ifOp, ExportState& state,
          const size_t controlFlowDepth) {
  validateControlFlowDepth(controlFlowDepth);
  auto result = std::make_unique<ExportedControlFlow>();
  result->kind = ControlFlowKind::IfElse;
  result->target = exportCondition(ifOp.getCondition(), state,
                                   *ifOp->getBlock(), *ifOp.getOperation());
  result->blocks.push_back(
      collectBlock(ifOp.getThenRegion().front(), state, controlFlowDepth + 1U));
  if (!ifOp.getElseRegion().empty()) {
    result->blocks.push_back(collectBlock(ifOp.getElseRegion().front(), state,
                                          controlFlowDepth + 1U));
  }
  return result;
}

[[nodiscard]] static std::unique_ptr<ExportedControlFlow>
collectFor(mlir::scf::ForOp loop, ExportState& state,
           const size_t controlFlowDepth) {
  if (!loop.getInitArgs().empty() || loop.getNumResults() != 0U) {
    throw std::runtime_error(
        "Qiskit for-loop export does not support loop-carried values");
  }
  validateControlFlowDepth(controlFlowDepth);
  const auto lower = mlir::getConstantIntValue(loop.getLowerBound());
  const auto upper = mlir::getConstantIntValue(loop.getUpperBound());
  const auto step = mlir::getConstantIntValue(loop.getStep());
  if (!lower || !upper || !step || *step <= 0) {
    throw std::runtime_error(
        "Qiskit for-loop export requires constant bounds and a positive step");
  }

  auto result = std::make_unique<ExportedControlFlow>();
  result->kind = ControlFlowKind::For;
  result->loop = {
      .isRange = true, .start = *lower, .stop = *upper, .step = *step};
  std::optional<ParameterGroup> sourceGroup;
  if (const auto attribute = loop->getAttr(
          mlir::mqt::MQTDialect::ParameterGroupAttrHelper::getNameStr())) {
    sourceGroup = parameterGroup(attribute);
    state.parameterGroups.add(*sourceGroup);
  }
  std::optional<LoopParameterProjection> projection;
  std::optional<Parameter> loopParameter;
  std::string loopParameterName;
  if (!loop.getInductionVar().use_empty()) {
    projection = matchLoopParameterProjection(loop);
    if (!projection) {
      throw std::runtime_error(
          "Qiskit for-loop export supports only a loop induction value used "
          "as an f64 gate parameter");
    }
    state.expressionOperations.insert(projection->operations.begin(),
                                      projection->operations.end());
    if (!projection->value.use_empty()) {
      size_t identity = 0U;
      do {
        identity = state.nextLoopParameter++;
        loopParameterName = "_mqt_loop_" + std::to_string(identity);
      } while (state.parameterNames.contains(loopParameterName));
      state.parameterNames.insert(loopParameterName);
      loopParameter = Parameter::symbol(loopParameterName, sourceGroup);
      state.parameters[projection->value] = *loopParameter;
    }
  }
  auto body = collectBlock(*loop.getBody(), state, controlFlowDepth + 1U);
  if (projection && loopParameter &&
      circuitUsesParameterName(body, loopParameterName)) {
    result->loop.parameter = *loopParameter;
    const auto count = rangeLength(*lower, *upper, *step);
    if (count != 0U) {
      result->loop.start =
          checkedAffine(projection->multiplier, *lower, projection->offset,
                        "scf.for induction start");
      result->loop.step = checkedAffine(projection->multiplier, *step, 0,
                                        "scf.for induction step");
      if (result->loop.step == 0) {
        throw std::runtime_error(
            "Qiskit for-loop export cannot represent a constant induction "
            "projection");
      }
      if (count > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::runtime_error(
            "scf.for iteration count is too large for Qiskit");
      }
      result->loop.stop =
          checkedAffine(result->loop.step, static_cast<int64_t>(count),
                        result->loop.start, "scf.for induction stop");
    }
  }
  result->blocks.push_back(std::move(body));
  return result;
}

[[nodiscard]] static std::unique_ptr<ExportedControlFlow>
collectWhile(mlir::scf::WhileOp loop, ExportState& state,
             const size_t controlFlowDepth) {
  validateControlFlowDepth(controlFlowDepth);
  auto& before = loop.getBefore().front();
  auto& after = loop.getAfter().front();
  auto condition =
      llvm::dyn_cast<mlir::scf::ConditionOp>(before.getTerminator());
  auto yield = llvm::dyn_cast<mlir::scf::YieldOp>(after.getTerminator());
  if (!loop.getInits().empty() || loop.getNumResults() != 0U ||
      before.getNumArguments() != 0U || after.getNumArguments() != 0U ||
      !condition || !condition.getArgs().empty() || !yield ||
      yield.getNumOperands() != 0U) {
    throw std::runtime_error(
        "Qiskit while-loop export does not support loop-carried values");
  }
  auto result = std::make_unique<ExportedControlFlow>();
  result->kind = ControlFlowKind::While;
  result->target = exportCondition(condition.getCondition(), state, before,
                                   *condition.getOperation());
  validateExpressionBlock(before, state);
  result->blocks.push_back(collectBlock(after, state, controlFlowDepth + 1U));
  return result;
}

[[nodiscard]] static std::unique_ptr<ExportedControlFlow>
collectSwitch(mlir::scf::IndexSwitchOp switchOp, ExportState& state,
              const size_t controlFlowDepth) {
  if (switchOp.getNumResults() != 0U) {
    throw std::runtime_error(
        "Qiskit switch export does not support SSA results");
  }
  validateControlFlowDepth(controlFlowDepth);
  auto result = std::make_unique<ExportedControlFlow>();
  result->kind = ControlFlowKind::Switch;
  result->target =
      exportSwitchTarget(switchOp.getArg(), state, *switchOp->getBlock(),
                         *switchOp.getOperation());
  const uint32_t targetWidth = result->target.width;
  for (const auto [index, label] : llvm::enumerate(switchOp.getCases())) {
    if (label < 0) {
      throw std::runtime_error(
          "Qiskit switch export does not support negative case labels");
    }
    if (targetWidth < 64U &&
        static_cast<uint64_t>(label) >= (uint64_t{1} << targetWidth)) {
      throw std::runtime_error("Qiskit switch case label " +
                               std::to_string(label) + " does not fit the " +
                               std::to_string(targetWidth) + "-bit target");
    }
    result->switchCases.push_back({.labels = {static_cast<uint64_t>(label)}});
    result->blocks.push_back(
        collectBlock(switchOp.getCaseRegions()[index].front(), state,
                     controlFlowDepth + 1U));
  }
  result->switchCases.push_back({.isDefault = true});
  result->blocks.push_back(collectBlock(switchOp.getDefaultRegion().front(),
                                        state, controlFlowDepth + 1U));
  return result;
}

[[nodiscard]] ExportedCircuit collectBlock(mlir::Block& block,
                                           ExportState& state,
                                           const size_t controlFlowDepth) {
  const bool topLevel = controlFlowDepth == 0U;
  ExportedCircuit circuit;
  llvm::SmallVector<mlir::Operation*> deferredExpressions;
  for (auto& operation : block) {
    if (llvm::isa<mlir::arith::ConstantOp>(operation) ||
        isParameterExpressionOperation(operation)) {
      continue;
    }
    if (llvm::isa<mlir::cbit::AllocOp, mlir::memref::AllocOp, mlir::qc::AllocOp,
                  mlir::qc::DeallocOp, mlir::qc::StaticOp,
                  mlir::func::ReturnOp>(operation)) {
      if (!topLevel) {
        throw std::runtime_error(
            "Qiskit control-flow blocks cannot allocate or release circuit "
            "resources");
      }
      continue;
    }
    if (auto load = llvm::dyn_cast<mlir::memref::LoadOp>(operation)) {
      if (state.qubits.contains(load.getResult())) {
        continue;
      }
      deferredExpressions.push_back(&operation);
      continue;
    }
    if (auto load = llvm::dyn_cast<mlir::cbit::LoadOp>(operation)) {
      static_cast<void>(classicalBitIndex(load, state));
      deferredExpressions.push_back(&operation);
      continue;
    }
    if (auto dealloc = llvm::dyn_cast<mlir::memref::DeallocOp>(operation)) {
      if (topLevel && state.quantumBases.contains(dealloc.getMemref())) {
        continue;
      }
      throw std::runtime_error(
          "QC to Qiskit export encountered an unsupported memory deallocation");
    }
    if (auto store = llvm::dyn_cast<mlir::cbit::StoreOp>(operation)) {
      if (!store.getValue().getDefiningOp<mlir::qc::MeasureOp>()) {
        throw std::runtime_error(
            "QC to Qiskit export does not support non-measurement classical "
            "stores");
      }
      continue;
    }
    if (auto phase = llvm::dyn_cast<mlir::qc::GPhaseOp>(operation)) {
      addGlobalPhase(circuit,
                     exportParameter(phase.getTheta(), state.parameters));
      continue;
    }
    if (auto measure = llvm::dyn_cast<mlir::qc::MeasureOp>(operation)) {
      mlir::cbit::StoreOp destination;
      for (auto& use : measure.getResult().getUses()) {
        if (auto store = llvm::dyn_cast<mlir::cbit::StoreOp>(use.getOwner())) {
          if (destination) {
            throw std::runtime_error(
                "QC measurement has more than one classical destination");
          }
          destination = store;
        }
      }
      if (!destination) {
        throw std::runtime_error(
            "QC measurement is missing a static classical destination");
      }
      const auto info = state.classicalRegisterInfo.find(destination.getReg());
      const auto index = mlir::getConstantIntValue(destination.getIndex());
      if (info == state.classicalRegisterInfo.end()) {
        throw std::runtime_error(
            "QC measurement uses an unsupported classical destination");
      }
      if (!index) {
        throw std::runtime_error(
            "QC measurement uses a dynamic classical destination");
      }
      if (!isFusableMeasurementStore(measure, destination)) {
        throw std::runtime_error(
            "QC measurement destination must follow the measurement in the "
            "same block");
      }
      const auto checked = checkedIndex(*index, "classical-bit");
      if (checked >= info->second.size) {
        throw std::runtime_error(
            "QC measurement uses an out-of-bounds classical destination");
      }
      if (!state.measurementDestinations[destination.getReg()]
               .insert(checked)
               .second) {
        throw std::runtime_error(
            "QC to Qiskit export does not support duplicate classical "
            "destinations");
      }
      if (topLevel) {
        state.unconditionalWrites[destination.getReg()].insert(checked);
      }
      const auto destinationBit =
          checkedAdd(info->second.base, checked, "classical-bit");
      circuit.instructions.push_back(
          {.kind = ExportedInstruction::Kind::Measure,
           .qubits = mapQubits(measure.getQubit(), state.qubits),
           .clbits = {destinationBit}});
      state.measurementResultBits.try_emplace(measure.getResult(),
                                              destinationBit);
      continue;
    }
    if (auto reset = llvm::dyn_cast<mlir::qc::ResetOp>(operation)) {
      circuit.instructions.push_back(
          {.kind = ExportedInstruction::Kind::Reset,
           .qubits = mapQubits(reset.getQubit(), state.qubits)});
      continue;
    }
    if (auto barrier = llvm::dyn_cast<mlir::qc::BarrierOp>(operation)) {
      circuit.instructions.push_back(
          {.kind = ExportedInstruction::Kind::Barrier,
           .qubits = mapQubits(barrier.getQubits(), state.qubits)});
      continue;
    }
    if (llvm::isa<mlir::qc::UnitaryOp>(operation)) {
      circuit.instructions.push_back(
          collectUnitaryInstruction(operation, state.qubits, state.parameters));
      continue;
    }
    if (auto ifOp = llvm::dyn_cast<mlir::scf::IfOp>(operation)) {
      if (ifOp.getNumResults() != 0U) {
        if (ifOp.getNumResults() != 1U ||
            !ifOp.getResult(0).getType().isInteger(1)) {
          throw std::runtime_error(
              "Qiskit if/else export supports only one canonical "
              "short-circuit Boolean SSA result");
        }
        deferredExpressions.push_back(&operation);
        continue;
      }
      circuit.instructions.push_back(
          {.kind = ExportedInstruction::Kind::ControlFlow,
           .controlFlow = collectIf(ifOp, state, controlFlowDepth)});
      continue;
    }
    if (auto loop = llvm::dyn_cast<mlir::scf::ForOp>(operation)) {
      circuit.instructions.push_back(
          {.kind = ExportedInstruction::Kind::ControlFlow,
           .controlFlow = collectFor(loop, state, controlFlowDepth)});
      continue;
    }
    if (auto loop = llvm::dyn_cast<mlir::scf::WhileOp>(operation)) {
      circuit.instructions.push_back(
          {.kind = ExportedInstruction::Kind::ControlFlow,
           .controlFlow = collectWhile(loop, state, controlFlowDepth)});
      continue;
    }
    if (auto switchOp = llvm::dyn_cast<mlir::scf::IndexSwitchOp>(operation)) {
      circuit.instructions.push_back(
          {.kind = ExportedInstruction::Kind::ControlFlow,
           .controlFlow = collectSwitch(switchOp, state, controlFlowDepth)});
      continue;
    }
    if (llvm::isa<mlir::qc::UnitaryOpInterface>(operation)) {
      circuit.instructions.push_back(
          collectUnitaryInstruction(operation, state.qubits, state.parameters));
      continue;
    }
    if (llvm::isa<mlir::scf::YieldOp>(operation)) {
      auto yield = llvm::cast<mlir::scf::YieldOp>(operation);
      if (yield.getNumOperands() != 0U) {
        throw std::runtime_error(
            "Qiskit control-flow export does not support yielded SSA values");
      }
      continue;
    }
    if (operation.getDialect() ==
        operation.getContext()->getLoadedDialect<mlir::arith::ArithDialect>()) {
      deferredExpressions.push_back(&operation);
      continue;
    }
    if (operation.getNumResults() == 1U &&
        operation.getResult(0).getType().isF64()) {
      throw std::runtime_error("Qiskit circuit export does not support scalar "
                               "parameter operation '" +
                               operation.getName().getStringRef().str() + "'");
    }
    throw std::runtime_error("unsupported QC operation in Qiskit export: " +
                             operation.getName().getStringRef().str());
  }
  for (auto* operation : deferredExpressions) {
    if (!state.expressionOperations.contains(operation)) {
      throw std::runtime_error(
          "QC to Qiskit export found classical execution outside a supported "
          "control-flow expression");
    }
  }
  return circuit;
}

static void
validateConstructibleGates(const ExportedCircuit& circuit,
                           const VersionedTranslation& translation) {
  for (const auto& instruction : circuit.instructions) {
    if (instruction.kind == ExportedInstruction::Kind::Gate &&
        !translation.supportsGate(instruction.gate)) {
      const auto& descriptor =
          mlir::qc::getStandardGateDescriptor(instruction.gate.gate);
      throw std::runtime_error(
          "Qiskit output cannot construct standard gate '" +
          descriptor.operationSymbol.str() + "' with " +
          std::to_string(instruction.gate.controls) + " controls");
    }
    if (instruction.kind != ExportedInstruction::Kind::ControlFlow) {
      continue;
    }
    for (const auto& block : instruction.controlFlow->blocks) {
      validateConstructibleGates(block, translation);
    }
  }
}

static void emitCircuit(ExportedCircuit& circuit, CircuitWriter& writer,
                        const VersionedTranslation& translation,
                        const uint32_t numQubits, const uint32_t numClbits) {
  writer.setGlobalPhase(circuit.globalPhase);
  for (auto& instruction : circuit.instructions) {
    switch (instruction.kind) {
    case ExportedInstruction::Kind::Gate:
      writer.addGate(instruction.gate, instruction.qubits,
                     instruction.parameters);
      break;
    case ExportedInstruction::Kind::Measure:
      writer.addMeasure(instruction.qubits.at(0), instruction.clbits.at(0));
      break;
    case ExportedInstruction::Kind::Reset:
      writer.addReset(instruction.qubits.at(0));
      break;
    case ExportedInstruction::Kind::Barrier:
      writer.addBarrier(instruction.qubits);
      break;
    case ExportedInstruction::Kind::Unitary:
      writer.addUnitary(instruction.matrix, instruction.qubits,
                        instruction.unitaryControls);
      break;
    case ExportedInstruction::Kind::ControlFlow: {
      auto& control = *instruction.controlFlow;
      std::vector<std::unique_ptr<CircuitWriter>> blocks;
      blocks.reserve(control.blocks.size());
      for (auto& block : control.blocks) {
        auto blockWriter = translation.createCircuit(numQubits, numClbits);
        emitCircuit(block, *blockWriter, translation, numQubits, numClbits);
        blocks.push_back(std::move(blockWriter));
      }
      writer.addControlFlow(control.kind, std::move(control.target),
                            std::move(control.loop),
                            std::move(control.switchCases), std::move(blocks));
      break;
    }
    }
  }
}

nb::object exportCircuit(const mlir::QCProgram& program,
                         const mlir::CompilerTarget* const target) {
  auto moduleOp = program.module();
  const auto functions = moduleOp.getOps<mlir::func::FuncOp>();
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
  ExportState state;
  collectParameters(function, state);
  if (target != nullptr) {
    state.numQubits = checkedIndex(static_cast<uint64_t>(target->numQubits()),
                                   "target qubit count");
  }
  collectResources(function, state, target);
  auto circuit = collectBlock(function.getBody().front(), state, 0U);
  for (const auto& [reg, info] : state.classicalRegisterInfo) {
    if (info.initialization == mlir::cbit::Initialization::Zero) {
      continue;
    }
    const auto written = state.unconditionalWrites.find(reg);
    if (written == state.unconditionalWrites.end() ||
        written->second.size() != info.size) {
      throw std::runtime_error(
          "QC to Qiskit export cannot return undefined classical bits");
    }
  }
  validateExportParameters(circuit, state.inputParameters);
  if (target != nullptr) {
    Register reg{.name = "q"};
    reg.bits.resize(state.numQubits);
    std::iota(reg.bits.begin(), reg.bits.end(), 0U);
    state.quantumRegisters.push_back(std::move(reg));
  }
  const auto looseQubits = validateRegisterLayout(state.quantumRegisters,
                                                  state.numQubits, "quantum");
  const auto looseClbits = validateRegisterLayout(state.classicalRegisters,
                                                  state.numClbits, "classical");

  auto translation = selectTranslation();
  validateConstructibleGates(circuit, *translation);
  auto writer = translation->createCircuit(looseQubits, looseClbits);
  for (const auto& reg : state.quantumRegisters) {
    writer->addQuantumRegister(reg.name,
                               static_cast<uint32_t>(reg.bits.size()));
  }
  for (const auto& reg : state.classicalRegisters) {
    writer->addClassicalRegister(reg.name,
                                 static_cast<uint32_t>(reg.bits.size()));
  }
  emitCircuit(circuit, *writer, *translation, state.numQubits, state.numClbits);
  return writer->finish();
}

} // namespace mqt::bindings::qiskit
