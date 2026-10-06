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
#include "mqt/Dialect/MQT/Utils/Modifiers.h"
#include "mqt/Dialect/QC/IR/QCInterfaces.h"
#include "mqt/Dialect/QC/IR/QCOps.h"
#include "mqt/Dialect/QCO/IR/QCOInterfaces.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>

namespace mlir {
template <class Gate> static std::string gateName(Gate op) {
  auto symbol = op.getBaseSymbol().str();
  if (!isa<qc::CtrlOp, qco::CtrlOp, qc::InvOp, qco::InvOp, qc::PowOp,
           qco::PowOp>(op)) {
    return symbol;
  }
  auto inner = mqt::getSoleBodyUnitary<Gate>(op->getRegion(0).front());
  if (!inner || inner.getNumQubits() != op.getNumTargets()) {
    return symbol;
  }
  const auto name = gateName(inner);
  if (isa<qc::CtrlOp, qco::CtrlOp>(op)) {
    if (inner->getNumRegions() == 0 && !isa<qc::CallOp, qco::CallOp>(inner)) {
      return std::string(op.getNumControls(), 'c') + name;
    }
    const auto controls = op.getNumControls();
    return "ctrl(" + (controls == 1 ? "" : std::to_string(controls) + ",") +
           name + ")";
  }
  return symbol + "(" + name + ")";
}

static std::map<std::string, size_t>
countOperationsIf(Operation* root, function_ref<bool(Operation*)> predicate) {
  std::map<std::string, size_t> counts;
  if (root != nullptr) {
    root->walk([&](Operation* op) {
      if (predicate(op)) {
        ++counts[op->getName().getStringRef().str()];
      }
    });
  }
  return counts;
}

static bool isControlFlow(Operation* op) {
  return isa<BranchOpInterface, RegionBranchOpInterface>(op);
}

std::map<std::string, size_t> Program::operationCounts() const {
  return countOperationsIf(mod(), [](Operation*) { return true; });
}

template <class Gate>
static void forEachGate(ModuleOp moduleOp, function_ref<void(Gate)> visit) {
  auto entryPoint = mqt::getEntryPoint(moduleOp);
  if (!entryPoint) {
    return;
  }
  entryPoint.walk<WalkOrder::PreOrder>([&](Gate op) {
    if (!isa<qc::BarrierOp, qco::BarrierOp>(op)) {
      visit(op);
    }
    /// Count each unitary operation atomically, including modifiers.
    return WalkResult::skip();
  });
}

template <class Gate>
static QuantumProgramInfo inspectProgram(ModuleOp moduleOp) {
  QuantumProgramInfo info;
  auto entry = mqt::getEntryPoint(moduleOp);
  info.operationCounts =
      countOperationsIf(moduleOp, [](Operation*) { return true; });
  info.controlFlowCounts = countOperationsIf(entry, isControlFlow);
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
    info.hasControlFlow |= isControlFlow(op);
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
        if (const auto size = type.hasStaticShape() ? type.tryGetNumElements()
                                                    : std::nullopt) {
          addAllocation(static_cast<uint64_t>(*size));
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
  forEachGate<Gate>(moduleOp, [&](Gate op) {
    ++info.numGates;
    const auto numQubits = op.getNumQubits();
    info.numSingleQubitGates += numQubits == 1;
    info.numTwoQubitGates += numQubits == 2;
    ++info.gateCounts[gateName(op)];
  });
  return info;
}
QuantumProgramInfo QCProgram::inspect() const {
  return inspectProgram<qc::UnitaryOpInterface>(mod());
}
QuantumProgramInfo QCOProgram::inspect() const {
  return inspectProgram<qco::UnitaryOpInterface>(mod());
}

template <class Gate>
static size_t countGatesIf(ModuleOp moduleOp,
                           function_ref<bool(Gate)> predicate) {
  size_t count = 0;
  forEachGate<Gate>(moduleOp, [&](Gate op) { count += predicate(op); });
  return count;
}

template <class Gate>
static std::map<std::string, size_t> countGatesByName(ModuleOp moduleOp) {
  std::map<std::string, size_t> counts;
  forEachGate<Gate>(moduleOp, [&](Gate op) { ++counts[gateName(op)]; });
  return counts;
}

size_t QCProgram::numGates() const {
  return countGatesIf<qc::UnitaryOpInterface>(mod(), [](auto) { return true; });
}

size_t QCProgram::numSingleQubitGates() const {
  return countGatesIf<qc::UnitaryOpInterface>(
      mod(), [](auto op) { return op.isSingleQubit(); });
}

size_t QCProgram::numTwoQubitGates() const {
  return countGatesIf<qc::UnitaryOpInterface>(
      mod(), [](auto op) { return op.isTwoQubit(); });
}

std::map<std::string, size_t> QCProgram::gateCounts() const {
  return countGatesByName<qc::UnitaryOpInterface>(mod());
}

std::map<std::string, size_t> QCProgram::controlFlowCounts() const {
  return countOperationsIf(mqt::getEntryPoint(mod()), isControlFlow);
}

size_t QCOProgram::numGates() const {
  return countGatesIf<qco::UnitaryOpInterface>(mod(),
                                               [](auto) { return true; });
}

size_t QCOProgram::numSingleQubitGates() const {
  return countGatesIf<qco::UnitaryOpInterface>(
      mod(), [](auto op) { return op.isSingleQubit(); });
}

size_t QCOProgram::numTwoQubitGates() const {
  return countGatesIf<qco::UnitaryOpInterface>(
      mod(), [](auto op) { return op.isTwoQubit(); });
}

std::map<std::string, size_t> QCOProgram::gateCounts() const {
  return countGatesByName<qco::UnitaryOpInterface>(mod());
}

std::map<std::string, size_t> QCOProgram::controlFlowCounts() const {
  return countOperationsIf(mqt::getEntryPoint(mod()), isControlFlow);
}
} // namespace mlir
