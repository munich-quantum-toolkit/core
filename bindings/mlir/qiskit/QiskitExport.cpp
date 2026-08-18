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
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCInterfaces.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QC/Translation/StandardGate.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <nanobind/nanobind.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
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
  std::vector<double> parameters;
  std::vector<std::complex<double>> matrix;
  uint32_t unitaryControls = 0;
};

[[nodiscard]] double exportParameter(const mlir::Value value) {
  if (const auto number = mlir::utils::valueToDouble(value)) {
    if (!std::isfinite(*number)) {
      throw std::runtime_error("cannot export a non-finite QC parameter");
    }
    return *number;
  }
  throw std::runtime_error(
      "Qiskit circuit export supports only numeric parameters");
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
  llvm::DenseMap<mlir::Value, mlir::cbit::Initialization>
      classicalInitializations;
  std::vector<ExportedInstruction> instructions;
  std::vector<Register> quantumRegisters;
  std::vector<Register> classicalRegisters;
  double globalPhase = 0.0;
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
                          const llvm::DenseMap<mlir::Value, uint32_t>& qubits);

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
    instruction.parameters.front() = -instruction.parameters.front();
    return;
  }
  if (instruction.gate.gate == Gate::U3 &&
      instruction.parameters.size() == 3U) {
    const std::array values{instruction.parameters[0],
                            instruction.parameters[1],
                            instruction.parameters[2]};
    instruction.parameters = {-values[0], -values[2], -values[1]};
    return;
  }
  throw std::runtime_error(
      "QC inverse modifier has no supported Qiskit gate equivalent");
}

[[nodiscard]] ExportedInstruction
collectUnitaryInstruction(mlir::Operation& operation,
                          const llvm::DenseMap<mlir::Value, uint32_t>& qubits) {
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
        result.parameters.push_back(exportParameter(parameter));
      }
      result.parameters.push_back(exportParameter(phase.getTheta()));
      return result;
    }
    if (bodyOperations.size() != 1U) {
      throw std::runtime_error(
          "QC control export requires one standard gate in the modifier body");
    }
    auto result = collectUnitaryInstruction(*bodyOperations.front(), nestedMap);
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
    auto result = collectUnitaryInstruction(*bodyOperations.front(), nestedMap);
    invertGate(result);
    return result;
  }
  if (auto power = llvm::dyn_cast<mlir::qc::PowOp>(operation)) {
    auto bodyOperations = modifierBodyOperations(power.getRegion());
    if (bodyOperations.size() != 1U) {
      throw std::runtime_error(
          "QC power export requires one standard gate in the modifier body");
    }
    const auto exponent = exportParameter(power.getExponent());
    if (exponent != 1.0 && exponent != -1.0) {
      throw std::runtime_error(
          "QC power export supports only constant exponents 1 and -1");
    }
    auto nestedMap =
        modifierQubitMap(qubits, power.getRegion().front(), power.getQubits());
    auto result = collectUnitaryInstruction(*bodyOperations.front(), nestedMap);
    if (exponent == -1.0) {
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
    result.parameters.push_back(exportParameter(parameter));
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

  auto returnOp =
      llvm::dyn_cast<mlir::func::ReturnOp>(function.getBody().front().back());
  if (!returnOp) {
    throw std::runtime_error(
        "QC to Qiskit export requires an entry-function return");
  }
  llvm::DenseSet<mlir::Value> returnedRegisters;
  for (const auto result : returnOp.getOperands()) {
    const auto type =
        llvm::dyn_cast<mlir::cbit::RegisterType>(result.getType());
    if (!type) {
      continue;
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
    state.classicalBases[result] = state.numClbits;
    state.classicalSizes[result] = size;
    state.classicalInitializations[result] = alloc.getInitialization();
    if (const auto name = alloc.getSourceNameAttr()) {
      Register reg{.name = name.str()};
      reg.bits.resize(size);
      std::iota(reg.bits.begin(), reg.bits.end(), state.numClbits);
      state.classicalRegisters.push_back(std::move(reg));
    }
    state.numClbits = checkedAdd(state.numClbits, size, "classical-bit");
  }
}

void collectFlatInstructions(mlir::func::FuncOp function, ExportState& state) {
  llvm::DenseMap<mlir::Value, llvm::DenseSet<uint32_t>> writtenBits;
  llvm::DenseMap<mlir::Operation*, mlir::cbit::StoreOp> measurementDestinations;

  for (auto store : function.getBody().front().getOps<mlir::cbit::StoreOp>()) {
    auto measure = store.getValue().getDefiningOp<mlir::qc::MeasureOp>();
    if (!measure) {
      throw std::runtime_error(
          "QC to Qiskit export does not support non-measurement classical "
          "stores");
    }
    const auto size = state.classicalSizes.find(store.getReg());
    const auto index = mlir::getConstantIntValue(store.getIndex());
    if (size == state.classicalSizes.end()) {
      throw std::runtime_error(
          "QC measurement stores to a classical register that is not "
          "returned");
    }
    if (!index) {
      throw std::runtime_error(
          "QC measurement uses a dynamic classical destination");
    }
    const auto checked = checkedIndex(*index, "classical-bit");
    if (checked >= size->second) {
      throw std::runtime_error(
          "QC measurement uses an out-of-bounds classical destination");
    }
    if (!writtenBits[store.getReg()].insert(checked).second) {
      throw std::runtime_error(
          "QC to Qiskit export does not support duplicate classical "
          "destinations");
    }
    if (!measurementDestinations.try_emplace(measure.getOperation(), store)
             .second) {
      throw std::runtime_error(
          "QC measurement has more than one classical destination");
    }
  }

  for (auto& operation : function.getBody().front()) {
    if (llvm::isa<mlir::cbit::AllocOp>(operation)) {
      continue;
    }
    if (auto alloc = llvm::dyn_cast<mlir::memref::AllocOp>(operation)) {
      if (state.quantumBases.contains(alloc.getResult())) {
        continue;
      }
      throw std::runtime_error(
          "QC to Qiskit export encountered an unsupported memory allocation");
    }
    if (llvm::isa<mlir::arith::ConstantOp>(operation)) {
      continue;
    }
    if (auto load = llvm::dyn_cast<mlir::memref::LoadOp>(operation)) {
      if (state.qubits.contains(load.getResult())) {
        continue;
      }
      throw std::runtime_error(
          "QC to Qiskit export does not support classical or unknown memory "
          "loads");
    }
    if (auto dealloc = llvm::dyn_cast<mlir::memref::DeallocOp>(operation)) {
      if (state.quantumBases.contains(dealloc.getMemref())) {
        continue;
      }
      throw std::runtime_error(
          "QC to Qiskit export encountered an unsupported memory deallocation");
    }
    if (llvm::isa<mlir::cbit::LoadOp>(operation)) {
      throw std::runtime_error(
          "QC to Qiskit export does not support classical loads or control "
          "flow");
    }
    if (llvm::isa<mlir::cbit::StoreOp>(operation)) {
      continue;
    }
    if (llvm::isa<mlir::qc::AllocOp, mlir::qc::DeallocOp, mlir::qc::StaticOp,
                  mlir::func::ReturnOp>(operation)) {
      continue;
    }
    if (auto phase = llvm::dyn_cast<mlir::qc::GPhaseOp>(operation)) {
      state.globalPhase += exportParameter(phase.getTheta());
      if (!std::isfinite(state.globalPhase)) {
        throw std::runtime_error(
            "QC global phase cannot be represented by Qiskit");
      }
      continue;
    }
    if (auto measure = llvm::dyn_cast<mlir::qc::MeasureOp>(operation)) {
      const auto destination =
          measurementDestinations.find(measure.getOperation());
      if (destination == measurementDestinations.end()) {
        throw std::runtime_error(
            "QC measurement is missing a static classical destination");
      }
      auto store = destination->second;
      const auto base = state.classicalBases.find(store.getReg());
      const auto index = mlir::getConstantIntValue(store.getIndex());
      if (base == state.classicalBases.end() || !index) {
        throw std::runtime_error(
            "QC measurement uses an unsupported classical destination");
      }
      const auto size = state.classicalSizes.find(store.getReg());
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
          collectUnitaryInstruction(operation, state.qubits));
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
          collectUnitaryInstruction(operation, state.qubits));
      continue;
    }
    throw std::runtime_error("unsupported QC operation in Qiskit export: " +
                             operation.getName().getStringRef().str());
  }

  for (const auto& [reg, initialization] : state.classicalInitializations) {
    if (initialization == mlir::cbit::Initialization::Zero) {
      continue;
    }
    const auto size = state.classicalSizes.lookup(reg);
    if (writtenBits[reg].size() != size) {
      throw std::runtime_error(
          "QC to Qiskit export cannot return undefined classical bits");
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
  collectFlatInstructions(function, state);
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
