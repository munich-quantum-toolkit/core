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
#include <llvm/ADT/Sequence.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/BuiltinAttributes.h>
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
namespace {

constexpr size_t MAX_EXPORT_CONTROL_FLOW_DEPTH = 64U;
constexpr size_t MAX_EXPORT_EXPRESSION_NODES = 4096U;

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

struct ExportedCircuit {
  Parameter globalPhase;
  std::vector<ExportedInstruction> instructions;
};

struct ExportedControlFlow {
  ControlFlowKind kind = ControlFlowKind::IfElse;
  ClassicalTarget target;
  Loop loop;
  std::vector<SwitchCase> switchCases;
  std::vector<ExportedCircuit> blocks;
  std::vector<uint32_t> qubits;
  std::vector<uint32_t> clbits;
};

struct ExportScope {
  llvm::DenseMap<mlir::Value, std::string> loopParameters;
};

[[nodiscard]] Parameter exportParameter(const mlir::Value value,
                                        const ExportScope& scope) {
  if (const auto number = mlir::utils::valueToDouble(value)) {
    if (!std::isfinite(*number)) {
      throw std::runtime_error("cannot export a non-finite QC parameter");
    }
    return {.number = number};
  }
  if (const auto found = scope.loopParameters.find(value);
      found != scope.loopParameters.end()) {
    return {.number = std::nullopt, .text = found->second};
  }
  throw std::runtime_error(
      "Qiskit circuit export supports only numeric and loop-local "
      "parameters");
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
  std::vector<Register> quantumRegisters;
  std::vector<Register> classicalRegisters;
  llvm::DenseSet<mlir::Operation*> expressionOperations;
  size_t nextLoopParameter = 0U;
  uint32_t numQubits = 0;
  uint32_t numClbits = 0;
};

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
                          const ExportScope& scope);

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
    if (!instruction.parameters.front().number) {
      throw std::runtime_error(
          "QC inverse export does not support a symbolic gate parameter");
    }
    *instruction.parameters.front().number =
        -*instruction.parameters.front().number;
    return;
  }
  if (instruction.gate.gate == Gate::U3 &&
      instruction.parameters.size() == 3U) {
    if (!instruction.parameters[0].number ||
        !instruction.parameters[1].number ||
        !instruction.parameters[2].number) {
      throw std::runtime_error(
          "QC inverse export does not support symbolic U parameters");
    }
    const std::array values{*instruction.parameters[0].number,
                            *instruction.parameters[1].number,
                            *instruction.parameters[2].number};
    instruction.parameters = {
        {.number = -values[0]}, {.number = -values[2]}, {.number = -values[1]}};
    return;
  }
  throw std::runtime_error(
      "QC inverse modifier has no supported Qiskit gate equivalent");
}

[[nodiscard]] ExportedInstruction
collectUnitaryInstruction(mlir::Operation& operation,
                          const llvm::DenseMap<mlir::Value, uint32_t>& qubits,
                          const ExportScope& scope) {
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
        result.parameters.push_back(exportParameter(parameter, scope));
      }
      result.parameters.push_back(exportParameter(phase.getTheta(), scope));
      return result;
    }
    if (bodyOperations.size() != 1U) {
      throw std::runtime_error(
          "QC control export requires one standard gate in the modifier body");
    }
    auto result =
        collectUnitaryInstruction(*bodyOperations.front(), nestedMap, scope);
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
    auto result =
        collectUnitaryInstruction(*bodyOperations.front(), nestedMap, scope);
    invertGate(result);
    return result;
  }
  if (auto power = llvm::dyn_cast<mlir::qc::PowOp>(operation)) {
    auto bodyOperations = modifierBodyOperations(power.getRegion());
    if (bodyOperations.size() != 1U) {
      throw std::runtime_error(
          "QC power export requires one standard gate in the modifier body");
    }
    const auto exponent = exportParameter(power.getExponent(), scope);
    if (!exponent.number ||
        (*exponent.number != 1.0 && *exponent.number != -1.0)) {
      throw std::runtime_error(
          "QC power export supports only constant exponents 1 and -1");
    }
    auto nestedMap =
        modifierQubitMap(qubits, power.getRegion().front(), power.getQubits());
    auto result =
        collectUnitaryInstruction(*bodyOperations.front(), nestedMap, scope);
    if (*exponent.number == -1.0) {
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
    result.parameters.push_back(exportParameter(parameter, scope));
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
}

[[nodiscard]] std::optional<uint64_t>
constantUnsignedInteger(const mlir::Value value) {
  auto constant = value.getDefiningOp<mlir::arith::ConstantOp>();
  const auto integer =
      constant ? llvm::dyn_cast<mlir::IntegerAttr>(constant.getValue())
               : mlir::IntegerAttr{};
  if (!integer || integer.getValue().getBitWidth() > 64U) {
    return std::nullopt;
  }
  return integer.getValue().getZExtValue();
}

void setExpressionType(Expression& expression, const mlir::Type type) {
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

[[nodiscard]] uint32_t classicalBitIndex(mlir::memref::LoadOp load,
                                         const ExportState& state) {
  if (!load.getResult().getType().isInteger(1) ||
      load.getIndices().size() != 1U) {
    throw std::runtime_error(
        "Qiskit classical expressions require a static classical-bit load");
  }
  const auto base = state.classicalBases.find(load.getMemRef());
  const auto size = state.classicalSizes.find(load.getMemRef());
  const auto index = mlir::getConstantIntValue(load.getIndices().front());
  if (base == state.classicalBases.end() ||
      size == state.classicalSizes.end() || !index) {
    throw std::runtime_error(
        "Qiskit classical expressions could not resolve a classical bit");
  }
  const auto checked = checkedIndex(*index, "classical-bit");
  if (checked >= size->second) {
    throw std::runtime_error(
        "Qiskit classical expression uses an out-of-bounds classical bit");
  }
  return checkedAdd(base->second, checked, "classical-bit");
}

[[nodiscard]] std::unique_ptr<Expression>
makeBooleanUnary(const UnaryOperation operation,
                 std::unique_ptr<Expression> operand, size_t& nodeCount) {
  if (++nodeCount > MAX_EXPORT_EXPRESSION_NODES) {
    throw std::runtime_error(
        "QC classical expression exceeds the size limit of 4096 nodes");
  }
  auto result = std::make_unique<Expression>();
  result->kind = ExpressionKind::Unary;
  result->type = ClassicalType::Bool;
  result->width = 1U;
  result->unaryOperation = operation;
  result->left = std::move(operand);
  return result;
}

[[nodiscard]] std::unique_ptr<Expression>
makeBooleanBinary(const BinaryOperation operation,
                  std::unique_ptr<Expression> left,
                  std::unique_ptr<Expression> right, size_t& nodeCount) {
  if (++nodeCount > MAX_EXPORT_EXPRESSION_NODES) {
    throw std::runtime_error(
        "QC classical expression exceeds the size limit of 4096 nodes");
  }
  auto result = std::make_unique<Expression>();
  result->kind = ExpressionKind::Binary;
  result->type = ClassicalType::Bool;
  result->width = 1U;
  result->binaryOperation = operation;
  result->left = std::move(left);
  result->right = std::move(right);
  return result;
}

[[nodiscard]] std::optional<bool>
constantBoolean(const std::unique_ptr<Expression>& expression) {
  if (expression && expression->kind == ExpressionKind::Value &&
      expression->type == ClassicalType::Bool) {
    return expression->boolValue;
  }
  return std::nullopt;
}

[[nodiscard]] std::unique_ptr<Expression>
cloneExpression(const Expression& expression, size_t& nodeCount) {
  if (++nodeCount > MAX_EXPORT_EXPRESSION_NODES) {
    throw std::runtime_error(
        "QC classical expression exceeds the size limit of 4096 nodes");
  }
  auto result = std::make_unique<Expression>();
  result->kind = expression.kind;
  result->type = expression.type;
  result->width = expression.width;
  result->binaryOperation = expression.binaryOperation;
  result->unaryOperation = expression.unaryOperation;
  result->boolValue = expression.boolValue;
  result->uintValue = expression.uintValue;
  result->floatValue = expression.floatValue;
  result->bit = expression.bit;
  result->reg = expression.reg;
  if (expression.left) {
    result->left = cloneExpression(*expression.left, nodeCount);
  }
  if (expression.right) {
    result->right = cloneExpression(*expression.right, nodeCount);
  }
  return result;
}

[[nodiscard]] std::unique_ptr<Expression>
makeBooleanSelect(std::unique_ptr<Expression> condition,
                  std::unique_ptr<Expression> thenValue,
                  std::unique_ptr<Expression> elseValue, size_t& nodeCount) {
  const auto thenConstant = constantBoolean(thenValue);
  const auto elseConstant = constantBoolean(elseValue);
  if (thenConstant && elseConstant) {
    if (*thenConstant == *elseConstant) {
      return std::move(thenValue);
    }
    if (*thenConstant) {
      return condition;
    }
    return makeBooleanUnary(UnaryOperation::LogicNot, std::move(condition),
                            nodeCount);
  }
  if (elseConstant && !*elseConstant) {
    return makeBooleanBinary(BinaryOperation::LogicAnd, std::move(condition),
                             std::move(thenValue), nodeCount);
  }
  if (elseConstant && *elseConstant) {
    return makeBooleanBinary(BinaryOperation::LogicOr,
                             makeBooleanUnary(UnaryOperation::LogicNot,
                                              std::move(condition), nodeCount),
                             std::move(thenValue), nodeCount);
  }
  if (thenConstant && *thenConstant) {
    return makeBooleanBinary(BinaryOperation::LogicOr, std::move(condition),
                             std::move(elseValue), nodeCount);
  }
  if (thenConstant && !*thenConstant) {
    return makeBooleanBinary(BinaryOperation::LogicAnd,
                             makeBooleanUnary(UnaryOperation::LogicNot,
                                              std::move(condition), nodeCount),
                             std::move(elseValue), nodeCount);
  }
  auto negated =
      makeBooleanUnary(UnaryOperation::LogicNot,
                       cloneExpression(*condition, nodeCount), nodeCount);
  return makeBooleanBinary(
      BinaryOperation::LogicOr,
      makeBooleanBinary(BinaryOperation::LogicAnd, std::move(condition),
                        std::move(thenValue), nodeCount),
      makeBooleanBinary(BinaryOperation::LogicAnd, std::move(negated),
                        std::move(elseValue), nodeCount),
      nodeCount);
}

[[nodiscard]] std::unique_ptr<Expression>
exportExpressionImpl(mlir::Value value, ExportState& state,
                     mlir::Block& evaluationBlock, const size_t depth,
                     size_t& nodeCount) {
  if (depth >= MAX_EXPORT_CONTROL_FLOW_DEPTH) {
    throw std::runtime_error(
        "QC classical expressions exceed the nesting limit of 64");
  }
  if (++nodeCount > MAX_EXPORT_EXPRESSION_NODES) {
    throw std::runtime_error(
        "QC classical expression exceeds the size limit of 4096 nodes");
  }
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
      state.expressionOperations.insert(operation);
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
    state.expressionOperations.insert(operation);
    return result;
  }
  if (auto load = llvm::dyn_cast<mlir::memref::LoadOp>(operation)) {
    result->kind = ExpressionKind::ClassicalBit;
    result->bit = classicalBitIndex(load, state);
    state.expressionOperations.insert(operation);
    return result;
  }
  if (auto ifOp = llvm::dyn_cast<mlir::scf::IfOp>(operation)) {
    if (ifOp.getNumResults() == 0U ||
        !llvm::all_of(
            ifOp.getResultTypes(),
            [](const mlir::Type type) { return type.isInteger(1); }) ||
        ifOp.getElseRegion().empty()) {
      throw std::runtime_error(
          "Qiskit classical expressions support only Boolean scf.if "
          "results with an else branch");
    }
    const auto opResult = llvm::dyn_cast<mlir::OpResult>(value);
    if (!opResult || opResult.getOwner() != operation) {
      throw std::runtime_error(
          "Qiskit classical expression does not refer to an scf.if result");
    }
    const size_t resultIndex = opResult.getResultNumber();
    auto& thenBlock = ifOp.getThenRegion().front();
    auto& elseBlock = ifOp.getElseRegion().front();
    auto thenYield =
        llvm::dyn_cast<mlir::scf::YieldOp>(thenBlock.getTerminator());
    auto elseYield =
        llvm::dyn_cast<mlir::scf::YieldOp>(elseBlock.getTerminator());
    if (!thenYield || !elseYield ||
        thenYield.getNumOperands() != ifOp.getNumResults() ||
        elseYield.getNumOperands() != ifOp.getNumResults()) {
      throw std::runtime_error(
          "Qiskit Boolean scf.if expressions require one yielded value per "
          "result in each branch");
    }
    auto condition = exportExpressionImpl(
        ifOp.getCondition(), state, *ifOp->getBlock(), depth + 1U, nodeCount);
    std::unique_ptr<Expression> thenValue;
    std::unique_ptr<Expression> elseValue;
    for (const size_t index : llvm::seq(ifOp.getNumResults())) {
      auto currentThen = exportExpressionImpl(
          thenYield.getOperand(index), state, thenBlock, depth + 1U, nodeCount);
      auto currentElse = exportExpressionImpl(
          elseYield.getOperand(index), state, elseBlock, depth + 1U, nodeCount);
      if (index == resultIndex) {
        thenValue = std::move(currentThen);
        elseValue = std::move(currentElse);
      }
    }
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
    if (!thenValue || !elseValue) {
      throw std::runtime_error(
          "Qiskit classical expression refers to an invalid scf.if result");
    }
    state.expressionOperations.insert(operation);
    return makeBooleanSelect(std::move(condition), std::move(thenValue),
                             std::move(elseValue), nodeCount);
  }

  const auto unary = [&](const ExpressionKind kind, const mlir::Value operand) {
    result->kind = kind;
    result->left = exportExpressionImpl(operand, state, evaluationBlock,
                                        depth + 1U, nodeCount);
    state.expressionOperations.insert(operation);
    return std::move(result);
  };
  const auto binary = [&](const BinaryOperation kind, const mlir::Value left,
                          const mlir::Value right) {
    result->kind = ExpressionKind::Binary;
    result->binaryOperation = kind;
    result->left = exportExpressionImpl(left, state, evaluationBlock,
                                        depth + 1U, nodeCount);
    result->right = exportExpressionImpl(right, state, evaluationBlock,
                                         depth + 1U, nodeCount);
    state.expressionOperations.insert(operation);
    return std::move(result);
  };

  if (auto cast = llvm::dyn_cast<mlir::arith::ExtUIOp>(operation)) {
    return unary(ExpressionKind::Cast, cast.getIn());
  }
  if (auto cast = llvm::dyn_cast<mlir::arith::TruncIOp>(operation)) {
    return unary(ExpressionKind::Cast, cast.getIn());
  }
  if (auto cast = llvm::dyn_cast<mlir::arith::UIToFPOp>(operation)) {
    return unary(ExpressionKind::Cast, cast.getIn());
  }
  if (auto cast = llvm::dyn_cast<mlir::arith::FPToUIOp>(operation)) {
    return unary(ExpressionKind::Cast, cast.getIn());
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
  if (auto op = llvm::dyn_cast<mlir::arith::AddIOp>(operation)) {
    return binary(BinaryOperation::Add, op.getLhs(), op.getRhs());
  }
  if (auto op = llvm::dyn_cast<mlir::arith::SubIOp>(operation)) {
    return binary(BinaryOperation::Subtract, op.getLhs(), op.getRhs());
  }
  if (auto op = llvm::dyn_cast<mlir::arith::MulIOp>(operation)) {
    return binary(BinaryOperation::Multiply, op.getLhs(), op.getRhs());
  }
  if (auto op = llvm::dyn_cast<mlir::arith::DivUIOp>(operation)) {
    return binary(BinaryOperation::Divide, op.getLhs(), op.getRhs());
  }
  if (auto op = llvm::dyn_cast<mlir::arith::AddFOp>(operation)) {
    return binary(BinaryOperation::Add, op.getLhs(), op.getRhs());
  }
  if (auto op = llvm::dyn_cast<mlir::arith::SubFOp>(operation)) {
    return binary(BinaryOperation::Subtract, op.getLhs(), op.getRhs());
  }
  if (auto op = llvm::dyn_cast<mlir::arith::MulFOp>(operation)) {
    return binary(BinaryOperation::Multiply, op.getLhs(), op.getRhs());
  }
  if (auto op = llvm::dyn_cast<mlir::arith::DivFOp>(operation)) {
    return binary(BinaryOperation::Divide, op.getLhs(), op.getRhs());
  }
  if (auto op = llvm::dyn_cast<mlir::arith::NegFOp>(operation)) {
    result->unaryOperation = UnaryOperation::Negate;
    return unary(ExpressionKind::Unary, op.getOperand());
  }
  throw std::runtime_error(
      "unsupported QC classical operation in Qiskit export: " +
      operation->getName().getStringRef().str());
}

[[nodiscard]] std::unique_ptr<Expression>
exportExpression(mlir::Value value, ExportState& state,
                 mlir::Block& evaluationBlock) {
  size_t nodeCount = 0U;
  return exportExpressionImpl(value, state, evaluationBlock, 0U, nodeCount);
}

struct PackedRegister {
  Register reg;
  llvm::SmallPtrSet<mlir::Operation*, 16> operations;
};

[[nodiscard]] std::optional<PackedRegister>
matchPackedRegister(mlir::Value value, ExportState& state,
                    mlir::Block& evaluationBlock) {
  auto type = llvm::dyn_cast<mlir::IntegerType>(value.getType());
  if (!type || type.getWidth() == 0U || type.getWidth() > 64U) {
    return std::nullopt;
  }
  std::vector<std::optional<uint32_t>> bits(type.getWidth());
  llvm::SmallPtrSet<mlir::Operation*, 16> operations;
  const std::function<bool(mlir::Value, uint32_t)> collect =
      [&](const mlir::Value current, const uint32_t shift) {
        auto* operation = current.getDefiningOp();
        if (operation == nullptr) {
          return false;
        }
        if (auto constant =
                llvm::dyn_cast<mlir::arith::ConstantOp>(operation)) {
          const auto integer =
              llvm::dyn_cast<mlir::IntegerAttr>(constant.getValue());
          if (!integer || !integer.getValue().isZero()) {
            return false;
          }
          operations.insert(operation);
          return true;
        }
        if (operation->getBlock() != &evaluationBlock) {
          return false;
        }
        if (auto op = llvm::dyn_cast<mlir::arith::OrIOp>(operation)) {
          operations.insert(operation);
          return collect(op.getLhs(), shift) && collect(op.getRhs(), shift);
        }
        if (auto op = llvm::dyn_cast<mlir::arith::ShLIOp>(operation)) {
          const auto amount = constantUnsignedInteger(op.getRhs());
          if (!amount || *amount >= bits.size() ||
              *amount > std::numeric_limits<uint32_t>::max() - shift) {
            return false;
          }
          operations.insert(operation);
          return collect(op.getLhs(), shift + static_cast<uint32_t>(*amount));
        }
        if (auto op = llvm::dyn_cast<mlir::arith::ExtUIOp>(operation)) {
          operations.insert(operation);
          return collect(op.getIn(), shift);
        }
        auto load = llvm::dyn_cast<mlir::memref::LoadOp>(operation);
        if (!load || shift >= bits.size() || bits[shift]) {
          return false;
        }
        try {
          bits[shift] = classicalBitIndex(load, state);
        } catch (const std::runtime_error&) {
          return false;
        }
        operations.insert(operation);
        return true;
      };
  if (!collect(value, 0U) ||
      llvm::any_of(bits, [](const auto& bit) { return !bit.has_value(); })) {
    return std::nullopt;
  }
  Register reg;
  reg.bits.reserve(bits.size());
  for (const auto bit : bits) {
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

void acceptPackedRegister(PackedRegister& packed, ExportState& state) {
  state.expressionOperations.insert(packed.operations.begin(),
                                    packed.operations.end());
}

[[nodiscard]] bool storesToValueRecursively(mlir::Operation& operation,
                                            const mlir::Value value) {
  bool stores = false;
  operation.walk([&](mlir::Operation* nested) {
    if (auto store = llvm::dyn_cast<mlir::memref::StoreOp>(nested);
        store && store.getMemRef() == value) {
      stores = true;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return stores;
}

void validateClassicalSnapshot(const mlir::Value expression,
                               mlir::Operation& consumer) {
  llvm::DenseSet<mlir::Value> visited;
  llvm::SmallVector<mlir::memref::LoadOp> loads;
  const std::function<void(mlir::Value)> collectLoads =
      [&](const mlir::Value value) {
        if (!visited.insert(value).second) {
          return;
        }
        auto* operation = value.getDefiningOp();
        if (operation == nullptr) {
          return;
        }
        if (auto load = llvm::dyn_cast<mlir::memref::LoadOp>(operation)) {
          loads.push_back(load);
          return;
        }
        if (auto ifOp = llvm::dyn_cast<mlir::scf::IfOp>(operation);
            ifOp && ifOp.getNumResults() != 0U) {
          for (auto& region : ifOp->getRegions()) {
            if (region.empty()) {
              continue;
            }
            if (auto yield = llvm::dyn_cast<mlir::scf::YieldOp>(
                    region.front().getTerminator())) {
              for (const auto yielded : yield.getOperands()) {
                collectLoads(yielded);
              }
            }
          }
        }
        for (const auto operand : operation->getOperands()) {
          collectLoads(operand);
        }
      };
  collectLoads(expression);
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
      if (auto store = llvm::dyn_cast<mlir::memref::StoreOp>(operation);
          store && store.getMemRef() == load.getMemRef()) {
        throw std::runtime_error(
            "Qiskit control-flow export cannot preserve a stale classical "
            "snapshot");
      }
      if (operation->getNumRegions() != 0U &&
          storesToValueRecursively(*operation, load.getMemRef())) {
        throw std::runtime_error(
            "Qiskit control-flow export cannot preserve a classical "
            "snapshot across nested control flow");
      }
    }
  }
}

[[nodiscard]] ClassicalTarget exportCondition(mlir::Value value,
                                              ExportState& state,
                                              mlir::Block& evaluationBlock,
                                              mlir::Operation& consumer) {
  if (!value.getType().isInteger(1)) {
    throw std::runtime_error(
        "Qiskit control-flow conditions must have Boolean type");
  }
  validateClassicalSnapshot(value, consumer);
  if (auto comparison = value.getDefiningOp<mlir::arith::CmpIOp>();
      comparison &&
      comparison.getPredicate() == mlir::arith::CmpIPredicate::eq) {
    for (const auto [actual, expected] :
         std::array{std::pair{comparison.getLhs(), comparison.getRhs()},
                    std::pair{comparison.getRhs(), comparison.getLhs()}}) {
      const auto constant = constantUnsignedInteger(expected);
      if (!constant) {
        continue;
      }
      if (auto load = actual.getDefiningOp<mlir::memref::LoadOp>();
          load && actual.getType().isInteger(1) && *constant <= 1U) {
        state.expressionOperations.insert(comparison);
        state.expressionOperations.insert(expected.getDefiningOp());
        state.expressionOperations.insert(load);
        return {.kind = ClassicalTargetKind::ClassicalBit,
                .bit = classicalBitIndex(load, state),
                .expectedBit = *constant != 0U};
      }
      if (auto packed = matchPackedRegister(actual, state, evaluationBlock)) {
        if (*constant >= (packed->reg.bits.size() == 64U
                              ? std::numeric_limits<uint64_t>::max()
                              : uint64_t{1} << packed->reg.bits.size()) &&
            packed->reg.bits.size() != 64U) {
          continue;
        }
        state.expressionOperations.insert(comparison);
        state.expressionOperations.insert(expected.getDefiningOp());
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

[[nodiscard]] ClassicalTarget exportSwitchTarget(mlir::Value value,
                                                 ExportState& state,
                                                 mlir::Block& evaluationBlock,
                                                 mlir::Operation& consumer) {
  validateClassicalSnapshot(value, consumer);
  if (auto cast = value.getDefiningOp<mlir::arith::IndexCastUIOp>()) {
    state.expressionOperations.insert(cast);
    value = cast.getIn();
  }
  if (auto load = value.getDefiningOp<mlir::memref::LoadOp>();
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
  return target;
}

[[nodiscard]] int64_t signedIntegerConstant(const mlir::Value value,
                                            const std::string_view kind) {
  auto constant = value.getDefiningOp<mlir::arith::ConstantOp>();
  const auto integer =
      constant ? llvm::dyn_cast<mlir::IntegerAttr>(constant.getValue())
               : mlir::IntegerAttr{};
  if (!integer || integer.getValue().getBitWidth() > 64U) {
    throw std::runtime_error(std::string(kind) + " must be a constant i64");
  }
  return integer.getValue().getSExtValue();
}

[[nodiscard]] int64_t checkedAffine(const int64_t multiplier,
                                    const int64_t value, const int64_t offset,
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

[[nodiscard]] uint64_t rangeLength(const int64_t lower, const int64_t upper,
                                   const int64_t step) {
  if (step <= 0) {
    throw std::runtime_error(
        "QC to Qiskit export requires a positive scf.for step");
  }
  if (lower >= upper) {
    return 0U;
  }
  const llvm::APInt lowerWide(65U, static_cast<uint64_t>(lower), true);
  const llvm::APInt upperWide(65U, static_cast<uint64_t>(upper), true);
  const llvm::APInt stepWide(65U, static_cast<uint64_t>(step), true);
  const auto count = ((upperWide - lowerWide - 1U).udiv(stepWide)) + 1U;
  if (count.getActiveBits() > 64U) {
    throw std::runtime_error("scf.for iteration count is too large for Qiskit");
  }
  return count.getZExtValue();
}

struct LoopParameterProjection {
  mlir::Value value;
  int64_t multiplier = 1;
  int64_t offset = 0;
  llvm::SmallPtrSet<mlir::Operation*, 4> operations;
};

[[nodiscard]] mlir::Operation* uniqueUser(const mlir::Value value) {
  return value.hasOneUse() ? *value.getUsers().begin() : nullptr;
}

[[nodiscard]] std::optional<LoopParameterProjection>
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
      const auto other =
          multiply.getLhs() == current ? multiply.getRhs() : multiply.getLhs();
      auto constant = other.getDefiningOp<mlir::arith::ConstantOp>();
      if (!constant) {
        return std::nullopt;
      }
      projection.multiplier =
          signedIntegerConstant(other, "scf.for induction multiplier");
      projection.operations.insert(user);
      current = multiply.getResult();
    }
  }
  if (auto* user = uniqueUser(current)) {
    if (auto add = llvm::dyn_cast<mlir::arith::AddIOp>(user)) {
      const auto other = add.getLhs() == current ? add.getRhs() : add.getLhs();
      auto constant = other.getDefiningOp<mlir::arith::ConstantOp>();
      if (!constant) {
        return std::nullopt;
      }
      projection.offset =
          signedIntegerConstant(other, "scf.for induction offset");
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

void setGlobalPhase(ExportedCircuit& circuit, Parameter phase) {
  if (circuit.globalPhase.number && phase.number) {
    *circuit.globalPhase.number += *phase.number;
    if (!std::isfinite(*circuit.globalPhase.number)) {
      throw std::runtime_error(
          "QC global phase cannot be represented by Qiskit");
    }
    return;
  }
  if (circuit.globalPhase.number && *circuit.globalPhase.number == 0.0) {
    circuit.globalPhase = std::move(phase);
    return;
  }
  if (phase.number && *phase.number == 0.0) {
    return;
  }
  throw std::runtime_error(
      "Qiskit circuit export cannot combine symbolic global phases");
}

[[nodiscard]] ExportedCircuit
collectBlock(mlir::Block& block, ExportState& state, const ExportScope& scope,
             size_t controlFlowDepth, bool topLevel);

[[nodiscard]] std::vector<uint32_t> allIndices(const uint32_t size) {
  std::vector<uint32_t> result(size);
  std::iota(result.begin(), result.end(), 0U);
  return result;
}

[[nodiscard]] std::optional<uint32_t>
initialClassicalZeroStoreIndex(mlir::memref::StoreOp store,
                               const mlir::Value registerValue,
                               const ExportState& state) {
  auto constant =
      store.getValueToStore().getDefiningOp<mlir::arith::ConstantOp>();
  const auto integer =
      constant ? llvm::dyn_cast<mlir::IntegerAttr>(constant.getValue())
               : mlir::IntegerAttr{};
  if (store.getMemRef() != registerValue || !integer ||
      !store.getValueToStore().getType().isInteger(1) ||
      !integer.getValue().isZero() || store.getIndices().size() != 1U) {
    return std::nullopt;
  }
  const auto base = state.classicalBases.find(store.getMemRef());
  const auto size = state.classicalSizes.find(store.getMemRef());
  const auto index = mlir::getConstantIntValue(store.getIndices().front());
  if (base == state.classicalBases.end() ||
      size == state.classicalSizes.end() || !index || *index < 0 ||
      std::cmp_greater_equal(*index, size->second)) {
    return std::nullopt;
  }
  return static_cast<uint32_t>(*index);
}

[[nodiscard]] bool
isImplicitClassicalZeroInitialization(mlir::memref::StoreOp store,
                                      const ExportState& state) {
  const auto current =
      initialClassicalZeroStoreIndex(store, store.getMemRef(), state);
  auto allocation = store.getMemRef().getDefiningOp<mlir::memref::AllocOp>();
  if (!current || !allocation || allocation->getBlock() != store->getBlock()) {
    return false;
  }

  llvm::DenseSet<uint32_t> initialized;
  for (auto* operation = allocation->getNextNode(); operation != store;
       operation = operation->getNextNode()) {
    if (operation == nullptr) {
      return false;
    }
    if (llvm::isa<mlir::arith::ConstantOp>(operation)) {
      continue;
    }
    auto previous = llvm::dyn_cast<mlir::memref::StoreOp>(operation);
    const auto index =
        previous
            ? initialClassicalZeroStoreIndex(previous, store.getMemRef(), state)
            : std::nullopt;
    if (!index || !initialized.insert(*index).second) {
      return false;
    }
  }
  return initialized.insert(*current).second;
}

[[nodiscard]] bool isFusableMeasurementStore(mlir::qc::MeasureOp measure,
                                             mlir::memref::StoreOp store) {
  if (!measure.getResult().hasOneUse() ||
      store.getValueToStore() != measure.getResult() ||
      measure->getBlock() != store->getBlock() ||
      store.getIndices().size() != 1U) {
    return false;
  }
  const auto index = mlir::getConstantIntValue(store.getIndices().front());
  if (!index) {
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

void validateExpressionBlock(mlir::Block& block, const ExportState& state) {
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

[[nodiscard]] std::unique_ptr<ExportedControlFlow>
collectIf(mlir::scf::IfOp ifOp, ExportState& state, const ExportScope& scope,
          const size_t controlFlowDepth) {
  if (ifOp.getNumResults() != 0U) {
    throw std::runtime_error(
        "Qiskit if/else export does not support SSA results");
  }
  if (controlFlowDepth >= MAX_EXPORT_CONTROL_FLOW_DEPTH) {
    throw std::runtime_error("QC control flow exceeds the nesting limit of 64");
  }
  auto result = std::make_unique<ExportedControlFlow>();
  result->kind = ControlFlowKind::IfElse;
  result->target = exportCondition(ifOp.getCondition(), state,
                                   *ifOp->getBlock(), *ifOp.getOperation());
  result->blocks.push_back(collectBlock(ifOp.getThenRegion().front(), state,
                                        scope, controlFlowDepth + 1U, false));
  if (!ifOp.getElseRegion().empty()) {
    result->blocks.push_back(collectBlock(ifOp.getElseRegion().front(), state,
                                          scope, controlFlowDepth + 1U, false));
  }
  result->qubits = allIndices(state.numQubits);
  result->clbits = allIndices(state.numClbits);
  return result;
}

[[nodiscard]] std::unique_ptr<ExportedControlFlow>
collectFor(mlir::scf::ForOp loop, ExportState& state, const ExportScope& scope,
           const size_t controlFlowDepth) {
  if (!loop.getInitArgs().empty() || loop.getNumResults() != 0U) {
    throw std::runtime_error(
        "Qiskit for-loop export does not support loop-carried values");
  }
  if (controlFlowDepth >= MAX_EXPORT_CONTROL_FLOW_DEPTH) {
    throw std::runtime_error("QC control flow exceeds the nesting limit of 64");
  }
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
  auto bodyScope = scope;
  if (!loop.getInductionVar().use_empty()) {
    auto projection = matchLoopParameterProjection(loop);
    if (!projection) {
      throw std::runtime_error(
          "Qiskit for-loop export supports only a loop induction value used "
          "as an f64 gate parameter");
    }
    state.expressionOperations.insert(projection->operations.begin(),
                                      projection->operations.end());
    if (projection->value.use_empty()) {
      result->blocks.push_back(collectBlock(*loop.getBody(), state, bodyScope,
                                            controlFlowDepth + 1U, false));
      result->qubits = allIndices(state.numQubits);
      result->clbits = allIndices(state.numClbits);
      return result;
    }
    const auto symbol =
        "_mqt_loop_" + std::to_string(state.nextLoopParameter++);
    result->loop.parameter = symbol;
    bodyScope.loopParameters[projection->value] = symbol;
    const auto count = rangeLength(*lower, *upper, *step);
    if (count == 0U) {
      result->loop.start = 0;
      result->loop.stop = 0;
      result->loop.step = 1;
    } else {
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
  result->blocks.push_back(collectBlock(*loop.getBody(), state, bodyScope,
                                        controlFlowDepth + 1U, false));
  result->qubits = allIndices(state.numQubits);
  result->clbits = allIndices(state.numClbits);
  return result;
}

[[nodiscard]] uint32_t switchTargetWidth(const ClassicalTarget& target) {
  switch (target.kind) {
  case ClassicalTargetKind::ClassicalBit:
    return 1U;
  case ClassicalTargetKind::ClassicalRegister:
    return target.width;
  case ClassicalTargetKind::Expression:
    if (target.expression) {
      return target.expression->width;
    }
    break;
  }
  throw std::runtime_error("Qiskit switch export has no target expression");
}

[[nodiscard]] std::unique_ptr<ExportedControlFlow>
collectWhile(mlir::scf::WhileOp loop, ExportState& state,
             const ExportScope& scope, const size_t controlFlowDepth) {
  if (controlFlowDepth >= MAX_EXPORT_CONTROL_FLOW_DEPTH) {
    throw std::runtime_error("QC control flow exceeds the nesting limit of 64");
  }
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
  result->blocks.push_back(
      collectBlock(after, state, scope, controlFlowDepth + 1U, false));
  result->qubits = allIndices(state.numQubits);
  result->clbits = allIndices(state.numClbits);
  return result;
}

[[nodiscard]] std::unique_ptr<ExportedControlFlow>
collectSwitch(mlir::scf::IndexSwitchOp switchOp, ExportState& state,
              const ExportScope& scope, const size_t controlFlowDepth) {
  if (switchOp.getNumResults() != 0U) {
    throw std::runtime_error(
        "Qiskit switch export does not support SSA results");
  }
  if (controlFlowDepth >= MAX_EXPORT_CONTROL_FLOW_DEPTH) {
    throw std::runtime_error("QC control flow exceeds the nesting limit of 64");
  }
  auto result = std::make_unique<ExportedControlFlow>();
  result->kind = ControlFlowKind::Switch;
  result->target =
      exportSwitchTarget(switchOp.getArg(), state, *switchOp->getBlock(),
                         *switchOp.getOperation());
  const uint32_t targetWidth = switchTargetWidth(result->target);
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
        collectBlock(switchOp.getCaseRegions()[index].front(), state, scope,
                     controlFlowDepth + 1U, false));
  }
  result->switchCases.push_back({.isDefault = true});
  result->blocks.push_back(collectBlock(switchOp.getDefaultRegion().front(),
                                        state, scope, controlFlowDepth + 1U,
                                        false));
  result->qubits = allIndices(state.numQubits);
  result->clbits = allIndices(state.numClbits);
  return result;
}

[[nodiscard]] ExportedCircuit
collectBlock(mlir::Block& block, ExportState& state, const ExportScope& scope,
             const size_t controlFlowDepth, const bool topLevel) {
  ExportedCircuit circuit;
  llvm::SmallVector<mlir::Operation*> deferredExpressions;
  for (auto& operation : block) {
    if (llvm::isa<mlir::arith::ConstantOp, mlir::memref::AllocOp,
                  mlir::qc::AllocOp, mlir::qc::DeallocOp, mlir::qc::StaticOp,
                  mlir::func::ReturnOp>(operation)) {
      if (!topLevel && llvm::isa<mlir::memref::AllocOp, mlir::qc::AllocOp,
                                 mlir::qc::DeallocOp, mlir::qc::StaticOp,
                                 mlir::func::ReturnOp>(operation)) {
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
    if (auto dealloc = llvm::dyn_cast<mlir::memref::DeallocOp>(operation)) {
      if (topLevel && (state.quantumBases.contains(dealloc.getMemref()) ||
                       state.classicalBases.contains(dealloc.getMemref()))) {
        continue;
      }
      throw std::runtime_error(
          "QC to Qiskit export encountered an unsupported memory deallocation");
    }
    if (auto poison = llvm::dyn_cast<mlir::ub::PoisonOp>(operation)) {
      if (llvm::all_of(poison->getResults(), [](const mlir::Value result) {
            return result.use_empty();
          })) {
        continue;
      }
      throw std::runtime_error(
          "QC to Qiskit export does not support used poison values");
    }
    if (auto store = llvm::dyn_cast<mlir::memref::StoreOp>(operation)) {
      if (topLevel && isImplicitClassicalZeroInitialization(store, state)) {
        continue;
      }
      if (auto measure =
              store.getValueToStore().getDefiningOp<mlir::qc::MeasureOp>();
          measure && isFusableMeasurementStore(measure, store)) {
        continue;
      }
      throw std::runtime_error(
          "QC to Qiskit export does not support classical execution");
    }
    if (auto phase = llvm::dyn_cast<mlir::qc::GPhaseOp>(operation)) {
      setGlobalPhase(circuit, exportParameter(phase.getTheta(), scope));
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
      if (!isFusableMeasurementStore(measure, destination)) {
        throw std::runtime_error(
            "QC measurement destination must follow the measurement in the "
            "same block");
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
      circuit.instructions.push_back(
          {.kind = ExportedInstruction::Kind::Measure,
           .qubits = mapQubits(measure.getQubit(), state.qubits),
           .clbits = {checkedAdd(base->second, checked, "classical-bit")}});
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
          collectUnitaryInstruction(operation, state.qubits, scope));
      continue;
    }
    if (auto ifOp = llvm::dyn_cast<mlir::scf::IfOp>(operation)) {
      if (ifOp.getNumResults() != 0U) {
        if (!llvm::all_of(ifOp.getResultTypes(), [](const mlir::Type type) {
              return type.isInteger(1);
            })) {
          throw std::runtime_error(
              "Qiskit if/else export does not support SSA results except as "
              "a Boolean classical expression");
        }
        deferredExpressions.push_back(&operation);
        continue;
      }
      circuit.instructions.push_back(
          {.kind = ExportedInstruction::Kind::ControlFlow,
           .controlFlow = collectIf(ifOp, state, scope, controlFlowDepth)});
      continue;
    }
    if (auto loop = llvm::dyn_cast<mlir::scf::ForOp>(operation)) {
      circuit.instructions.push_back(
          {.kind = ExportedInstruction::Kind::ControlFlow,
           .controlFlow = collectFor(loop, state, scope, controlFlowDepth)});
      continue;
    }
    if (auto loop = llvm::dyn_cast<mlir::scf::WhileOp>(operation)) {
      circuit.instructions.push_back(
          {.kind = ExportedInstruction::Kind::ControlFlow,
           .controlFlow = collectWhile(loop, state, scope, controlFlowDepth)});
      continue;
    }
    if (auto switchOp = llvm::dyn_cast<mlir::scf::IndexSwitchOp>(operation)) {
      circuit.instructions.push_back(
          {.kind = ExportedInstruction::Kind::ControlFlow,
           .controlFlow =
               collectSwitch(switchOp, state, scope, controlFlowDepth)});
      continue;
    }
    if (llvm::isa<mlir::qc::UnitaryOpInterface>(operation)) {
      circuit.instructions.push_back(
          collectUnitaryInstruction(operation, state.qubits, scope));
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

void validateConstructibleGates(const ExportedCircuit& circuit,
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
    if (instruction.kind != ExportedInstruction::Kind::ControlFlow ||
        !instruction.controlFlow) {
      continue;
    }
    for (const auto& block : instruction.controlFlow->blocks) {
      validateConstructibleGates(block, translation);
    }
  }
}

void emitCircuit(ExportedCircuit& circuit, CircuitWriter& writer,
                 const VersionedTranslation& translation) {
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
      if (!instruction.controlFlow) {
        throw std::runtime_error(
            "Qiskit export encountered an empty control-flow plan");
      }
      auto& control = *instruction.controlFlow;
      std::vector<std::unique_ptr<CircuitWriter>> blocks;
      blocks.reserve(control.blocks.size());
      for (auto& block : control.blocks) {
        auto blockWriter = translation.createCircuit(
            static_cast<uint32_t>(control.qubits.size()),
            static_cast<uint32_t>(control.clbits.size()));
        emitCircuit(block, *blockWriter, translation);
        blocks.push_back(std::move(blockWriter));
      }
      writer.addControlFlow(control.kind, std::move(control.target),
                            std::move(control.loop),
                            std::move(control.switchCases), std::move(blocks),
                            control.qubits, control.clbits);
      break;
    }
    }
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
  if (!function.getArguments().empty()) {
    throw std::runtime_error(
        "Qiskit circuit export does not support symbolic or runtime inputs");
  }

  ExportState state;
  if (target != nullptr) {
    state.numQubits = checkedIndex(static_cast<uint64_t>(target->numQubits()),
                                   "target qubit count");
  }
  collectResources(function, state, target);
  const ExportScope rootScope;
  auto circuit =
      collectBlock(function.getBody().front(), state, rootScope, 0U, true);
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
  emitCircuit(circuit, *writer, *translation);
  return writer->finish();
}

} // namespace mqt::bindings::qiskit
