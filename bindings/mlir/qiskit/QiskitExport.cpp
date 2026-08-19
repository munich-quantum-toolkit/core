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
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCInterfaces.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QC/Translation/StandardGate.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <nanobind/nanobind.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
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
namespace {

struct ExportedInstruction {
  enum class Kind : uint8_t {
    Gate,
    Measure,
    Reset,
    Barrier,
    Unitary,
  };
  Kind kind = Kind::Gate;
  StandardGateMapping gate;
  std::vector<uint32_t> qubits;
  std::vector<uint32_t> clbits;
  std::vector<Parameter> parameters;
  std::vector<std::complex<double>> matrix;
  uint32_t unitaryControls = 0;
};

using ExportedParameters = llvm::DenseMap<mlir::Value, Parameter>;

[[noreturn]] void throwExportedParameterExpressionSizeError() {
  throw std::runtime_error("QC parameter expression exceeds the supported " +
                           std::to_string(MAX_PARAMETER_EXPRESSION_NODES) +
                           "-node size");
}

[[noreturn]] void throwExportedParameterExpressionDepthError() {
  throw std::runtime_error("QC parameter expression exceeds the supported " +
                           std::to_string(MAX_PARAMETER_EXPRESSION_DEPTH) +
                           "-level nesting depth");
}

[[noreturn]] void throwExportedParameterGroupSizeError() {
  throw std::runtime_error("Qiskit parameter vectors support at most " +
                           std::to_string(MAX_PARAMETER_GROUP_SIZE) +
                           " elements");
}

[[noreturn]] void throwExportedParameterGroupTotalSizeError() {
  throw std::runtime_error("Qiskit circuit export supports at most " +
                           std::to_string(MAX_PARAMETER_GROUP_SIZE) +
                           " elements across all distinct parameter vectors");
}

[[nodiscard]] Parameter numberParameter(const double value) {
  return {.kind = ParameterKind::Number, .number = value};
}

[[nodiscard]] Parameter unaryParameter(const ParameterKind kind,
                                       Parameter operand) {
  return {.kind = kind,
          .left = std::make_shared<const Parameter>(std::move(operand))};
}

[[nodiscard]] Parameter binaryParameter(const ParameterKind kind,
                                        Parameter left, Parameter right) {
  return {.kind = kind,
          .left = std::make_shared<const Parameter>(std::move(left)),
          .right = std::make_shared<const Parameter>(std::move(right))};
}

[[nodiscard]] std::string
parameterGroupElementName(const ParameterGroup& group) {
  return group.name + "[" + std::to_string(group.index) + "]";
}

[[nodiscard]] Parameter exportParameterImpl(mlir::Value value,
                                            ExportedParameters& parameters,
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
  if (const auto number = mlir::utils::valueToDouble(value)) {
    if (!std::isfinite(*number)) {
      throw std::runtime_error("cannot export a non-finite QC parameter");
    }
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
  const auto unary = [&](const ParameterKind kind) {
    if (operation->getNumOperands() != 1U) {
      throw std::runtime_error("QC parameter operation '" +
                               operation->getName().getStringRef().str() +
                               "' has invalid arity");
    }
    return unaryParameter(kind,
                          exportParameterImpl(operation->getOperand(0),
                                              parameters, depth + 1U, nodes));
  };
  const auto binary = [&](const ParameterKind kind) {
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
    result = binary(ParameterKind::Add);
  } else if (llvm::isa<mlir::arith::SubFOp>(*operation)) {
    result = binary(ParameterKind::Subtract);
  } else if (llvm::isa<mlir::arith::MulFOp>(*operation)) {
    result = binary(ParameterKind::Multiply);
  } else if (llvm::isa<mlir::arith::DivFOp>(*operation)) {
    result = binary(ParameterKind::Divide);
  } else if (llvm::isa<mlir::math::PowFOp>(*operation)) {
    result = binary(ParameterKind::Power);
  } else if (llvm::isa<mlir::arith::NegFOp>(*operation)) {
    result = unary(ParameterKind::Negate);
  } else if (llvm::isa<mlir::math::SinOp>(*operation)) {
    result = unary(ParameterKind::Sin);
  } else if (llvm::isa<mlir::math::CosOp>(*operation)) {
    result = unary(ParameterKind::Cos);
  } else if (llvm::isa<mlir::math::TanOp>(*operation)) {
    result = unary(ParameterKind::Tan);
  } else if (llvm::isa<mlir::math::AsinOp>(*operation)) {
    result = unary(ParameterKind::ArcSin);
  } else if (llvm::isa<mlir::math::AcosOp>(*operation)) {
    result = unary(ParameterKind::ArcCos);
  } else if (llvm::isa<mlir::math::AtanOp>(*operation)) {
    result = unary(ParameterKind::ArcTan);
  } else if (llvm::isa<mlir::math::ExpOp>(*operation)) {
    result = unary(ParameterKind::Exp);
  } else if (llvm::isa<mlir::math::LogOp>(*operation)) {
    result = unary(ParameterKind::Log);
  } else if (llvm::isa<mlir::math::AbsFOp>(*operation)) {
    result = unary(ParameterKind::Abs);
  } else {
    throw std::runtime_error(
        "Qiskit circuit export does not support scalar parameter operation '" +
        operation->getName().getStringRef().str() + "'");
  }
  parameters.try_emplace(value, result);
  return result;
}

[[nodiscard]] Parameter exportParameter(const mlir::Value value,
                                        ExportedParameters& parameters) {
  size_t nodes = 0U;
  return exportParameterImpl(value, parameters, 1U, nodes);
}

void validateExportParameterImpl(const Parameter& parameter, const size_t depth,
                                 size_t& nodes) {
  if (depth > MAX_PARAMETER_EXPRESSION_DEPTH) {
    throwExportedParameterExpressionDepthError();
  }
  if (++nodes > MAX_PARAMETER_EXPRESSION_NODES) {
    throwExportedParameterExpressionSizeError();
  }
  const auto requireLeaf = [&] {
    if (parameter.left || parameter.right) {
      throw std::runtime_error(
          "QC parameter-expression leaf has unexpected operands");
    }
  };
  const auto requireUnary = [&] {
    if (!parameter.left || parameter.right) {
      throw std::runtime_error(
          "QC unary parameter expression has invalid operands");
    }
    validateExportParameterImpl(*parameter.left, depth + 1U, nodes);
  };
  const auto requireBinary = [&] {
    if (!parameter.left || !parameter.right) {
      throw std::runtime_error(
          "QC binary parameter expression has missing operands");
    }
    validateExportParameterImpl(*parameter.left, depth + 1U, nodes);
    validateExportParameterImpl(*parameter.right, depth + 1U, nodes);
  };
  switch (parameter.kind) {
  case ParameterKind::Number:
    requireLeaf();
    if (parameter.group) {
      throw std::runtime_error(
          "numeric QC parameter has unexpected input-group metadata");
    }
    if (!std::isfinite(parameter.number)) {
      throw std::runtime_error("cannot export a non-finite QC parameter");
    }
    return;
  case ParameterKind::Symbol:
    requireLeaf();
    if (parameter.text.empty() || parameter.identity.empty()) {
      throw std::runtime_error(
          "QC parameter symbol has invalid identity metadata");
    }
    if (parameter.text.find('\0') != std::string::npos ||
        parameter.identity.find('\0') != std::string::npos) {
      throw std::runtime_error(
          "QC parameter symbol metadata contains a null character");
    }
    if (parameter.group && parameter.group->size > MAX_PARAMETER_GROUP_SIZE) {
      throwExportedParameterGroupSizeError();
    }
    if (parameter.group &&
        (parameter.group->identity.empty() ||
         parameter.group->identity.find('\0') != std::string::npos ||
         parameter.group->name.find('\0') != std::string::npos ||
         parameter.group->index >=
             static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
         parameter.group->size >
             static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
         parameter.text != parameterGroupElementName(*parameter.group))) {
      throw std::runtime_error(
          "QC parameter symbol has invalid input-group metadata");
    }
    return;
  case ParameterKind::Add:
  case ParameterKind::Subtract:
  case ParameterKind::Multiply:
  case ParameterKind::Divide:
  case ParameterKind::Power:
    requireBinary();
    return;
  case ParameterKind::Negate:
  case ParameterKind::Sin:
  case ParameterKind::Cos:
  case ParameterKind::Tan:
  case ParameterKind::ArcSin:
  case ParameterKind::ArcCos:
  case ParameterKind::ArcTan:
  case ParameterKind::Exp:
  case ParameterKind::Log:
  case ParameterKind::Abs:
  case ParameterKind::Conjugate:
    requireUnary();
    return;
  }
  throw std::runtime_error("unknown QC parameter expression kind");
}

void validateExportParameter(const Parameter& parameter) {
  size_t nodes = 0U;
  validateExportParameterImpl(parameter, 1U, nodes);
}

[[nodiscard]] bool isParameterExpressionOperation(mlir::Operation& operation) {
  return llvm::isa<mlir::arith::AddFOp, mlir::arith::SubFOp,
                   mlir::arith::MulFOp, mlir::arith::DivFOp,
                   mlir::arith::NegFOp, mlir::math::PowFOp, mlir::math::SinOp,
                   mlir::math::CosOp, mlir::math::TanOp, mlir::math::AsinOp,
                   mlir::math::AcosOp, mlir::math::AtanOp, mlir::math::ExpOp,
                   mlir::math::LogOp, mlir::math::AbsFOp>(operation);
}

[[nodiscard]] uint32_t checkedIndex(const int64_t index,
                                    const std::string_view kind) {
  if (index < 0 ||
      std::cmp_greater(index, std::numeric_limits<uint32_t>::max())) {
    throw std::runtime_error(std::string(kind) +
                             " index cannot be represented by Qiskit");
  }
  return static_cast<uint32_t>(index);
}

[[nodiscard]] uint32_t checkedIndex(const uint64_t index,
                                    const std::string_view kind) {
  if (index > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error(std::string(kind) +
                             " index cannot be represented by Qiskit");
  }
  return static_cast<uint32_t>(index);
}

[[nodiscard]] mlir::CompilerTarget::SiteId
checkedTargetSiteId(const uint64_t index) {
  using SiteId = mlir::CompilerTarget::SiteId;
  if (!std::in_range<SiteId>(index)) {
    throw std::runtime_error(
        "QC static qubit index cannot be represented by a compiler target "
        "site ID");
  }
  return static_cast<SiteId>(index);
}

[[nodiscard]] uint32_t checkedAdd(const uint32_t left, const uint32_t right,
                                  const std::string_view kind) {
  if (right > std::numeric_limits<uint32_t>::max() - left) {
    throw std::runtime_error(std::string(kind) +
                             " count cannot be represented by Qiskit");
  }
  return left + right;
}

struct ExportState {
  llvm::DenseMap<mlir::Value, uint32_t> qubits;
  llvm::DenseMap<mlir::Value, uint32_t> quantumBases;
  llvm::DenseMap<mlir::Value, uint32_t> quantumSizes;
  llvm::DenseMap<mlir::Value, uint32_t> classicalBases;
  llvm::DenseMap<mlir::Value, uint32_t> classicalSizes;
  std::vector<ExportedInstruction> instructions;
  std::vector<Register> quantumRegisters;
  std::vector<Register> classicalRegisters;
  ExportedParameters parameters;
  Parameter globalPhase{.kind = ParameterKind::Number, .number = 0.0};
  uint32_t numQubits = 0;
  uint32_t numClbits = 0;
};

void validateExportParameters(const ExportState& state) {
  validateExportParameter(state.globalPhase);
  for (const auto& instruction : state.instructions) {
    for (const auto& parameter : instruction.parameters) {
      validateExportParameter(parameter);
    }
  }
}

void collectParameters(mlir::func::FuncOp function, ExportState& state) {
  llvm::StringSet<> names;
  llvm::StringMap<ParameterGroup> groups;
  uint64_t totalParameterGroupSize = 0U;
  for (const auto [index, argument] :
       llvm::enumerate(function.getArguments())) {
    const auto name = function.getArgAttrOfType<mlir::StringAttr>(
        index, mlir::utils::INPUT_NAME_ATTR);
    if (!argument.getType().isF64() || !name || name.getValue().empty()) {
      throw std::runtime_error(
          "Qiskit circuit export requires named f64 program inputs");
    }
    if (name.getValue().contains('\0')) {
      throw std::runtime_error(
          "Qiskit circuit export does not support parameter names with null "
          "characters");
    }
    if (!names.insert(name.getValue()).second) {
      throw std::runtime_error(
          "Qiskit circuit export requires unique parameter names");
    }
    const auto groupIdentityAttribute =
        function.getArgAttr(index, mlir::utils::INPUT_GROUP_ATTR);
    const auto groupNameAttribute =
        function.getArgAttr(index, mlir::utils::INPUT_GROUP_NAME_ATTR);
    const auto groupIndexAttribute =
        function.getArgAttr(index, mlir::utils::INPUT_GROUP_INDEX_ATTR);
    const auto groupSizeAttribute =
        function.getArgAttr(index, mlir::utils::INPUT_GROUP_SIZE_ATTR);
    const auto hasGroup = static_cast<bool>(groupIdentityAttribute) ||
                          static_cast<bool>(groupNameAttribute) ||
                          static_cast<bool>(groupIndexAttribute) ||
                          static_cast<bool>(groupSizeAttribute);
    std::optional<ParameterGroup> group;
    if (hasGroup) {
      const auto groupIdentity =
          llvm::dyn_cast_if_present<mlir::StringAttr>(groupIdentityAttribute);
      const auto groupName =
          llvm::dyn_cast_if_present<mlir::StringAttr>(groupNameAttribute);
      const auto groupIndex =
          llvm::dyn_cast_if_present<mlir::IntegerAttr>(groupIndexAttribute);
      const auto groupSize =
          llvm::dyn_cast_if_present<mlir::IntegerAttr>(groupSizeAttribute);
      if (!groupIdentity || groupIdentity.getValue().empty() || !groupName ||
          !groupIndex || !groupIndex.getType().isInteger(64) ||
          groupIndex.getInt() < 0 || !groupSize ||
          !groupSize.getType().isInteger(64) || groupSize.getInt() < 0 ||
          groupIdentity.getValue().contains('\0') ||
          groupName.getValue().contains('\0')) {
        throw std::runtime_error(
            "Qiskit circuit export requires complete and valid input-group "
            "metadata");
      }
      group =
          ParameterGroup{.identity = groupIdentity.str(),
                         .name = groupName.str(),
                         .index = static_cast<uint64_t>(groupIndex.getInt()),
                         .size = static_cast<uint64_t>(groupSize.getInt())};
      if (group->size == 0U || group->index >= group->size) {
        throw std::runtime_error(
            "Qiskit circuit export requires complete and valid input-group "
            "metadata");
      }
      if (group->size > MAX_PARAMETER_GROUP_SIZE) {
        throwExportedParameterGroupSizeError();
      }
      if (group->size >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error(
            "Qiskit circuit export input-group size cannot be represented by "
            "the Python runtime");
      }
      if (name.getValue() != parameterGroupElementName(*group)) {
        throw std::runtime_error(
            "Qiskit circuit export requires an input-group element name that "
            "matches its group and index");
      }
      const auto [knownGroup, inserted] =
          groups.try_emplace(group->identity, *group);
      if (inserted) {
        if (group->size > MAX_PARAMETER_GROUP_SIZE - totalParameterGroupSize) {
          throwExportedParameterGroupTotalSizeError();
        }
        totalParameterGroupSize += group->size;
      } else {
        if (knownGroup->second.name != group->name ||
            knownGroup->second.size != group->size) {
          throw std::runtime_error(
              "Qiskit circuit export found conflicting metadata for one "
              "input group");
        }
      }
    }
    state.parameters[argument] = {
        .kind = ParameterKind::Symbol,
        .text = name.str(),
        .identity = "input:" + std::to_string(index),
        .group = std::move(group),
    };
  }
}

void addGlobalPhase(ExportState& state, const Parameter& phase) {
  if (phase.kind == ParameterKind::Number) {
    if (!std::isfinite(phase.number)) {
      throw std::runtime_error(
          "QC global phase cannot be represented by Qiskit");
    }
    if (state.globalPhase.kind == ParameterKind::Number) {
      state.globalPhase.number += phase.number;
      if (!std::isfinite(state.globalPhase.number)) {
        throw std::runtime_error(
            "QC global phase cannot be represented by Qiskit");
      }
      return;
    }
    if (std::abs(phase.number) <= mlir::utils::TOLERANCE) {
      return;
    }
  } else if (state.globalPhase.kind == ParameterKind::Number &&
             std::abs(state.globalPhase.number) <= mlir::utils::TOLERANCE) {
    state.globalPhase = phase;
    return;
  }
  state.globalPhase =
      binaryParameter(ParameterKind::Add, std::move(state.globalPhase), phase);
}

[[nodiscard]] std::vector<uint32_t>
mapQubits(const mlir::ValueRange values,
          const llvm::DenseMap<mlir::Value, uint32_t>& qubits) {
  std::vector<uint32_t> result;
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

[[nodiscard]] ExportedInstruction
collectUnitaryInstruction(mlir::Operation& operation,
                          const llvm::DenseMap<mlir::Value, uint32_t>& qubits,
                          ExportedParameters& parameters);

[[nodiscard]] std::vector<mlir::Operation*>
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

[[nodiscard]] llvm::DenseMap<mlir::Value, uint32_t>
modifierQubitMap(const llvm::DenseMap<mlir::Value, uint32_t>& outer,
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

void invertGate(ExportedInstruction& instruction) {
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
        ParameterKind::Negate, std::move(instruction.parameters.front()));
    return;
  }
  if (instruction.gate.gate == Gate::U3 &&
      instruction.parameters.size() == 3U) {
    auto parameters = std::move(instruction.parameters);
    instruction.parameters = {
        unaryParameter(ParameterKind::Negate, std::move(parameters[0])),
        unaryParameter(ParameterKind::Negate, std::move(parameters[2])),
        unaryParameter(ParameterKind::Negate, std::move(parameters[1]))};
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
      for (const auto parameter : unitary.getParameters()) {
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
    if (exponent.kind != ParameterKind::Number ||
        (exponent.number != 1.0 && exponent.number != -1.0)) {
      throw std::runtime_error(
          "QC power export supports only constant exponents 1 and -1");
    }
    auto nestedMap =
        modifierQubitMap(qubits, power.getRegion().front(), power.getQubits());
    auto result = collectUnitaryInstruction(*bodyOperations.front(), nestedMap,
                                            parameters);
    if (exponent.number == -1.0) {
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
  for (const auto parameter : gate.getParameters()) {
    result.parameters.push_back(exportParameter(parameter, parameters));
  }
  return result;
}

void collectResources(mlir::func::FuncOp function, ExportState& state,
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
              mlir::utils::QUBIT_REGISTER_NAME_ATTR)) {
        Register reg{.name = name.str()};
        reg.bits.resize(size);
        std::iota(reg.bits.begin(), reg.bits.end(), state.numQubits);
        state.quantumRegisters.push_back(std::move(reg));
      }
      state.numQubits = checkedAdd(state.numQubits, size, "qubit");
    } else if (type.getElementType().isInteger(1)) {
      const auto size =
          checkedIndex(type.getShape()[0], "classical-register size");
      state.classicalBases[alloc.getResult()] = state.numClbits;
      state.classicalSizes[alloc.getResult()] = size;
      if (const auto name = operation.getAttrOfType<mlir::StringAttr>(
              mlir::utils::CLASSICAL_REGISTER_NAME_ATTR)) {
        Register reg{.name = name.str()};
        reg.bits.resize(size);
        std::iota(reg.bits.begin(), reg.bits.end(), state.numClbits);
        state.classicalRegisters.push_back(std::move(reg));
      }
      state.numClbits = checkedAdd(state.numClbits, size, "classical-bit");
    } else {
      throw std::runtime_error(
          "QC to Qiskit export encountered an unsupported memory allocation");
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

[[nodiscard]] std::optional<uint32_t>
initialClassicalZeroStoreIndex(mlir::memref::StoreOp store,
                               mlir::Value registerValue,
                               const ExportState& state) {
  if (store.getMemref() != registerValue || store.getIndices().size() != 1U ||
      !mlir::matchPattern(store.getValueToStore(), mlir::m_Zero())) {
    return std::nullopt;
  }

  const auto size = state.classicalSizes.find(registerValue);
  const auto index = mlir::getConstantIntValue(store.getIndices().front());
  if (size == state.classicalSizes.end() || !index || *index < 0 ||
      std::cmp_greater_equal(*index, size->second)) {
    return std::nullopt;
  }
  return static_cast<uint32_t>(*index);
}

void collectFlatInstructions(mlir::func::FuncOp function, ExportState& state) {
  mlir::Value initializationRegister;
  llvm::DenseSet<uint32_t> initializedClassicalBits;
  for (auto& operation : function.getBody().front()) {
    if (auto alloc = llvm::dyn_cast<mlir::memref::AllocOp>(operation)) {
      initializationRegister = {};
      initializedClassicalBits.clear();
      if (state.classicalBases.contains(alloc.getResult())) {
        initializationRegister = alloc.getResult();
      }
      continue;
    }
    if (llvm::isa<mlir::arith::ConstantOp>(operation) ||
        isParameterExpressionOperation(operation)) {
      continue;
    }
    if (auto load = llvm::dyn_cast<mlir::memref::LoadOp>(operation)) {
      initializationRegister = {};
      if (state.qubits.contains(load.getResult())) {
        continue;
      }
      throw std::runtime_error(
          "QC to Qiskit export does not support classical or unknown memory "
          "loads");
    }
    if (auto dealloc = llvm::dyn_cast<mlir::memref::DeallocOp>(operation)) {
      initializationRegister = {};
      if (state.quantumBases.contains(dealloc.getMemref()) ||
          state.classicalBases.contains(dealloc.getMemref())) {
        continue;
      }
      throw std::runtime_error(
          "QC to Qiskit export encountered an unsupported memory deallocation");
    }
    if (auto poison = llvm::dyn_cast<mlir::ub::PoisonOp>(operation)) {
      initializationRegister = {};
      if (llvm::all_of(poison->getResults(), [](const mlir::Value result) {
            return result.use_empty();
          })) {
        continue;
      }
      throw std::runtime_error(
          "QC to Qiskit export does not support used poison values");
    }
    if (auto store = llvm::dyn_cast<mlir::memref::StoreOp>(operation)) {
      if (llvm::isa_and_nonnull<mlir::qc::MeasureOp>(
              store.getValueToStore().getDefiningOp())) {
        initializationRegister = {};
        continue;
      }
      if (initializationRegister) {
        const auto index = initialClassicalZeroStoreIndex(
            store, initializationRegister, state);
        if (index && initializedClassicalBits.insert(*index).second) {
          continue;
        }
      }
      initializationRegister = {};
      throw std::runtime_error(
          "QC to Qiskit export does not support classical execution");
    }
    initializationRegister = {};
    if (llvm::isa<mlir::qc::AllocOp, mlir::qc::DeallocOp, mlir::qc::StaticOp,
                  mlir::func::ReturnOp>(operation)) {
      continue;
    }
    if (auto phase = llvm::dyn_cast<mlir::qc::GPhaseOp>(operation)) {
      addGlobalPhase(state,
                     exportParameter(phase.getTheta(), state.parameters));
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
    if (llvm::isa<mlir::qc::UnitaryOp>(operation)) {
      state.instructions.push_back(
          collectUnitaryInstruction(operation, state.qubits, state.parameters));
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
          collectUnitaryInstruction(operation, state.qubits, state.parameters));
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
}

} // namespace

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
  collectFlatInstructions(function, state);
  validateExportParameters(state);
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
  for (const auto& instruction : state.instructions) {
    if (instruction.kind != ExportedInstruction::Kind::Gate ||
        translation->supportsGate(instruction.gate)) {
      continue;
    }
    const auto& descriptor =
        mlir::qc::getStandardGateDescriptor(instruction.gate.gate);
    throw std::runtime_error("Qiskit output cannot construct standard gate '" +
                             descriptor.operationSymbol.str() + "' with " +
                             std::to_string(instruction.gate.controls) +
                             " controls");
  }
  auto writer = translation->createCircuit(looseQubits, looseClbits);
  for (const auto& reg : state.quantumRegisters) {
    writer->addQuantumRegister(reg.name,
                               static_cast<uint32_t>(reg.bits.size()));
  }
  for (const auto& reg : state.classicalRegisters) {
    writer->addClassicalRegister(reg.name,
                                 static_cast<uint32_t>(reg.bits.size()));
  }
  writer->setGlobalPhase(state.globalPhase);
  for (const auto& instruction : state.instructions) {
    switch (instruction.kind) {
    case ExportedInstruction::Kind::Gate:
      writer->addGate(instruction.gate, instruction.qubits,
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
      writer->addUnitary(instruction.matrix, instruction.qubits,
                         instruction.unitaryControls);
      break;
    }
  }
  return writer->finish();
}

} // namespace mqt::bindings::qiskit
