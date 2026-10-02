/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Compiler/Programs.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QC/IR/QCInterfaces.h"
#include "mqt/Dialect/QC/IR/QCOps.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>

namespace mlir {
static QuantumProgramInfo inspectProgram(ModuleOp moduleOp) {
  QuantumProgramInfo info;
  auto entry = mqt::getEntryPoint(moduleOp);
  if (entry && llvm::none_of(entry.getArgumentTypes(), [](Type type) {
        return isa<qc::QubitType, qco::QubitType>(type) ||
               (isa<ShapedType>(type) &&
                isa<qc::QubitType, qco::QubitType>(
                    cast<ShapedType>(type).getElementType()));
      })) {
    info.numQubits = 0;
  }
  const auto addAllocation = [&](const uint64_t size) {
    if (!info.numQubits) {
      return;
    }
    if (size > std::numeric_limits<uint64_t>::max() - *info.numQubits) {
      info.numQubits.reset();
    } else {
      *info.numQubits += size;
    }
  };
  moduleOp.walk<WalkOrder::PreOrder>([&](Operation* op) {
    if (isa<ModuleOp>(op) && op != moduleOp) {
      return WalkResult::skip();
    }
    info.hasControlFlow |= isa<BranchOpInterface, RegionBranchOpInterface>(op);
    if (auto qubit = dyn_cast<qc::StaticOp>(op)) {
      info.staticQubits.push_back(qubit.getIndex());
    } else if (auto qubit = dyn_cast<qco::StaticOp>(op)) {
      info.staticQubits.push_back(qubit.getIndex());
    } else if (isa<qc::AllocOp, qco::AllocOp>(op)) {
      addAllocation(1);
    } else {
      ShapedType type;
      if (auto alloc = dyn_cast<memref::AllocOp>(op);
          alloc && isa<qc::QubitType>(alloc.getType().getElementType())) {
        type = alloc.getType();
      } else if (auto alloc = dyn_cast<qtensor::AllocOp>(op)) {
        type = alloc.getType();
      }
      if (type) {
        if (type.hasStaticShape()) {
          addAllocation(static_cast<uint64_t>(type.getNumElements()));
        } else {
          info.numQubits.reset();
        }
      }
    }
    return WalkResult::advance();
  });
  llvm::sort(info.staticQubits);
  const auto duplicates = std::ranges::unique(info.staticQubits);
  info.staticQubits.erase(duplicates.begin(), duplicates.end());
  if (info.numQubits && !info.staticQubits.empty()) {
    info.numQubits = info.staticQubits.size();
  }
  return info;
}
QuantumProgramInfo QCProgram::inspect() const { return inspectProgram(mod()); }
QuantumProgramInfo QCOProgram::inspect() const { return inspectProgram(mod()); }

static void forEachGate(ModuleOp moduleOp,
                        function_ref<void(qc::UnitaryOpInterface)> visit) {
  auto entryPoint = mqt::getEntryPoint(moduleOp);
  if (!entryPoint) {
    return;
  }
  entryPoint.walk<WalkOrder::PreOrder>([&](qc::UnitaryOpInterface op) {
    if (!isa<qc::BarrierOp>(op)) {
      visit(op);
    }
    return isa<qc::CtrlOp, qc::InvOp, qc::PowOp>(op) ? WalkResult::skip()
                                                     : WalkResult::advance();
  });
}

namespace {

struct RegisterDepths {
  size_t dynamic = 0;
  DenseMap<int64_t, size_t> constants;

  void mergeMax(const RegisterDepths& other) {
    dynamic = std::max(dynamic, other.dynamic);
    for (const auto& [index, depth] : other.constants) {
      constants[index] = std::max(constants[index], depth);
    }
  }

  [[nodiscard]] size_t maximum() const {
    auto result = dynamic;
    for (const auto& entry : constants) {
      result = std::max(result, entry.second);
    }
    return result;
  }
};

struct GateDepthState {
  bool resolved = true;
  DenseMap<Value, size_t> values;
  DenseMap<uint64_t, size_t> staticQubits;
  DenseMap<Value, RegisterDepths> registers;

  void mergeMax(const GateDepthState& other) {
    resolved &= other.resolved;
    for (const auto& [value, depth] : other.values) {
      values[value] = std::max(values[value], depth);
    }
    for (const auto& [index, depth] : other.staticQubits) {
      staticQubits[index] = std::max(staticQubits[index], depth);
    }
    for (const auto& [value, depths] : other.registers) {
      registers[value].mergeMax(depths);
    }
  }

  [[nodiscard]] size_t maximum() const {
    size_t result = 0;
    for (const auto& entry : values) {
      result = std::max(result, entry.second);
    }
    for (const auto& entry : staticQubits) {
      result = std::max(result, entry.second);
    }
    for (const auto& entry : registers) {
      result = std::max(result, entry.second.maximum());
    }
    return result;
  }

  [[nodiscard]] size_t get(Value qubit) {
    if (auto staticOp = qubit.getDefiningOp<qc::StaticOp>()) {
      return staticQubits[staticOp.getIndex()];
    }
    if (auto loadOp = qubit.getDefiningOp<memref::LoadOp>();
        loadOp && isa<qc::QubitType>(loadOp.getType())) {
      // A view or stored reference may alias another resource. Do not report
      // a depth unless this is an owning register with read-only references.
      auto reg = loadOp.getMemref();
      const auto [position, inserted] = registers.try_emplace(reg);
      if (inserted &&
          (!reg.getDefiningOp<memref::AllocOp>() ||
           llvm::any_of(reg.getUsers(), [](Operation* user) {
             return !isa<memref::LoadOp, memref::DeallocOp, memref::DimOp>(
                 user);
           }))) {
        resolved = false;
        return 0;
      }
      auto& depths = position->second;
      if (loadOp.getIndices().size() == 1) {
        if (const auto index =
                getConstantIntValue(loadOp.getIndices().front())) {
          return std::max(depths.dynamic, depths.constants[*index]);
        }
      }
      return depths.maximum();
    }
    if (!qubit.getDefiningOp<qc::AllocOp>()) {
      resolved = false;
    }
    return values[qubit];
  }

  void set(Value qubit, const size_t depth) {
    if (auto staticOp = qubit.getDefiningOp<qc::StaticOp>()) {
      staticQubits[staticOp.getIndex()] = depth;
      return;
    }
    if (auto loadOp = qubit.getDefiningOp<memref::LoadOp>();
        loadOp && isa<qc::QubitType>(loadOp.getType())) {
      auto& depths = registers[loadOp.getMemref()];
      if (loadOp.getIndices().size() == 1) {
        if (const auto index =
                getConstantIntValue(loadOp.getIndices().front())) {
          depths.constants[*index] = depth;
          return;
        }
      }
      depths.dynamic = depth;
      return;
    }
    values[qubit] = depth;
  }
};

} // namespace

static void updateGateDepth(qc::UnitaryOpInterface gate,
                            GateDepthState& state) {
  if (isa<qc::BarrierOp>(gate) || gate.getNumQubits() == 0) {
    return;
  }
  size_t depth = 0;
  for (auto qubit : gate.getQubits()) {
    depth = std::max(depth, state.get(qubit));
  }
  ++depth;
  for (auto qubit : gate.getQubits()) {
    state.set(qubit, depth);
  }
}

static void calculateRegionDepth(Region& region, GateDepthState& state,
                                 unsigned nesting = 0);

static void calculateOperationDepth(Operation& operation, GateDepthState& state,
                                    unsigned nesting) {
  if (auto gate = dyn_cast<qc::UnitaryOpInterface>(&operation)) {
    updateGateDepth(gate, state);
    return;
  }
  if (isa<scf::IfOp, scf::IndexSwitchOp>(&operation)) {
    GateDepthState merged = state;
    for (auto& region : operation.getRegions()) {
      auto branchState = state;
      calculateRegionDepth(region, branchState, nesting + 1);
      merged.mergeMax(branchState);
    }
    state = std::move(merged);
    return;
  }
  if (isa<BranchOpInterface>(&operation) ||
      (isa<RegionBranchOpInterface>(&operation) &&
       !isa<scf::ForOp, scf::WhileOp>(&operation))) {
    state.resolved = false;
    return;
  }
  for (auto& region : operation.getRegions()) {
    calculateRegionDepth(region, state, nesting + 1);
  }
}

static void calculateRegionDepth(Region& region, GateDepthState& state,
                                 unsigned nesting) {
  if (nesting >= 128) {
    state.resolved = false;
    return;
  }
  for (auto& block : region) {
    for (auto& operation : block) {
      if (!state.resolved) {
        return;
      }
      calculateOperationDepth(operation, state, nesting);
    }
  }
}

size_t QCProgram::numGates() const {
  size_t count = 0;
  forEachGate(mod(), [&](qc::UnitaryOpInterface) { ++count; });
  return count;
}

size_t QCProgram::numSingleQubitGates() const {
  size_t count = 0;
  forEachGate(mod(),
              [&](qc::UnitaryOpInterface op) { count += op.isSingleQubit(); });
  return count;
}

size_t QCProgram::numTwoQubitGates() const {
  size_t count = 0;
  forEachGate(mod(),
              [&](qc::UnitaryOpInterface op) { count += op.isTwoQubit(); });
  return count;
}

std::map<std::string, size_t> QCProgram::gateCounts() const {
  std::map<std::string, size_t> counts;
  forEachGate(mod(), [&](qc::UnitaryOpInterface op) {
    ++counts[op.getBaseSymbol().str()];
  });
  return counts;
}

std::optional<size_t> QCProgram::staticDepth() const {
  GateDepthState state;
  auto entryPoint = mqt::getEntryPoint(mod());
  if (!entryPoint) {
    return std::nullopt;
  }
  for (auto& region : entryPoint->getRegions()) {
    calculateRegionDepth(region, state);
  }
  return state.resolved ? std::optional{state.maximum()} : std::nullopt;
}

} // namespace mlir
