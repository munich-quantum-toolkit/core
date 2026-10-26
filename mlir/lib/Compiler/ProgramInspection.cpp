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
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <optional>
#include <string>
#include <utility>

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

static std::string gateName(Operation* op) {
  return TypeSwitch<Operation*, std::string>(op)
      .Case<qc::UnitaryOpInterface, qco::UnitaryOpInterface>(
          [](auto gate) { return gateName(gate); })
      .Default([](Operation* other) {
        return other->getName().stripDialect().str();
      });
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

static bool isMeasurement(Operation* op) {
  return llvm::isa<qc::MeasureOp>(op) || llvm::isa<qco::MeasureOp>(op);
}

static bool isReset(Operation* op) {
  return llvm::isa<qc::ResetOp>(op) || llvm::isa<qco::ResetOp>(op);
}

static bool isBarrier(Operation* op) {
  return llvm::isa<qc::BarrierOp>(op) || llvm::isa<qco::BarrierOp>(op);
}

static bool isGateModifier(Operation* op) {
  return llvm::isa<qc::YieldOp>(op) || llvm::isa<qc::CtrlOp>(op) ||
         llvm::isa<qc::InvOp>(op) || llvm::isa<qc::PowOp>(op) ||
         llvm::isa<qco::YieldOp>(op) || llvm::isa<qco::CtrlOp>(op) ||
         llvm::isa<qco::InvOp>(op) || llvm::isa<qco::PowOp>(op);
}

static bool isCallOp(Operation* op) {
  return llvm::isa<qc::CallOp>(op) || llvm::isa<qco::CallOp>(op);
}

static bool isStandardGate(Operation* op) {
  return (llvm::isa<qc::UnitaryOpInterface>(op) ||
          llvm::isa<qco::UnitaryOpInterface>(op)) &&
         !isBarrier(op) && !isGateModifier(op) && !isCallOp(op);
}

static bool isLoop(Operation* op) {
  return llvm::isa<mlir::LoopLikeOpInterface>(op);
}

static bool isControlFlow(Operation* op) {
  return isa<BranchOpInterface, RegionBranchOpInterface>(op);
}

std::map<std::string, size_t> Program::operationCounts() const {
  return countOperationsIf(mod(), [](Operation*) { return true; });
}

static void forEachGate(ModuleOp moduleOp,
                        function_ref<void(Operation*, size_t)> visit) {
  auto entryPoint = mqt::getEntryPoint(moduleOp);
  if (!entryPoint) {
    return;
  }
  entryPoint.walk<WalkOrder::PreOrder>([&](Operation* op) {
    if (isa<qc::BarrierOp, qco::BarrierOp>(op)) {
      return WalkResult::skip();
    }
    size_t arity = 1;
    if (auto gate = dyn_cast<qc::UnitaryOpInterface>(op)) {
      arity = gate.getNumQubits();
    } else if (auto gate = dyn_cast<qco::UnitaryOpInterface>(op)) {
      arity = gate.getNumQubits();
    } else if (!isa<qc::MeasureOp, qco::MeasureOp, qc::ResetOp, qco::ResetOp>(
                   op)) {
      return WalkResult::advance();
    }
    visit(op, arity);
    // Count each gate atomically, including modifiers.
    return WalkResult::skip();
  });
}

static std::optional<std::pair<Value, int64_t>> getQubitKey(Value qubit) {
  auto memref = qubit.getDefiningOp<memref::LoadOp>();
  if (!memref) {
    return std::nullopt;
  }
  std::optional<int64_t> index =
      getConstantIntValue(memref.getIndices().front());
  if (!index) {
    return std::nullopt;
  }
  return std::make_pair(memref.getMemRef(), *index);
}

// Set all qubits to their highest current depth, plus one if `addLayer`.
static void
walkDepth(ValueRange inputQubits, ValueRange outputQubits, bool addLayer,
          llvm::DenseMap<std::pair<Value, int64_t>, size_t>& depthOf) {
  size_t maxDepth = 0;
  for (auto qi : inputQubits) {
    if (auto key = getQubitKey(qi)) {
      maxDepth = std::max(maxDepth, depthOf.lookup(*key));
    }
  }
  if (addLayer) {
    maxDepth += 1;
  }
  for (auto qo : outputQubits) {
    if (auto key = getQubitKey(qo)) {
      depthOf[*key] = maxDepth;
    }
  }
}

static std::optional<size_t> computeDepth(ModuleOp moduleOp) {
  auto entryPoint = mqt::getEntryPoint(moduleOp);

  if (!entryPoint) {
    return std::nullopt;
  }

  llvm::DenseMap<std::pair<Value, int64_t>, size_t> depthOf;
  bool hasInvalidOperation = false;

  entryPoint.walk<WalkOrder::PreOrder>([&](Operation* op) {
    if (isStandardGate(op) || isGateModifier(op)) {
      auto gate = dyn_cast<qc::UnitaryOpInterface>(op);
      if (gate) {
        walkDepth(gate.getQubits(), gate.getQubits(), /*addLayer=*/true,
                  depthOf);
      }
      // QCO
    } else if (isMeasurement(op)) {
      auto measure = dyn_cast<qc::MeasureOp>(op);
      if (measure) {
        walkDepth(measure.getQubit(), measure.getQubit(), /*addLayer=*/true,
                  depthOf);
      }
      // QCO
    } else if (isReset(op)) {
      auto reset = dyn_cast<qc::ResetOp>(op);
      if (reset) {
        walkDepth(reset.getQubit(), reset.getQubit(), /*addLayer=*/true,
                  depthOf);
      }
      // QCO
    } else if (isBarrier(op)) {
      auto barrier = dyn_cast<qc::BarrierOp>(op);
      if (barrier) {
        walkDepth(barrier.getQubits(), barrier.getQubits(), /*addLayer=*/false,
                  depthOf);
      }
      // QCO
    } else if (isControlFlow(op)) {
      // Here can be logic of how to handle control flow
      hasInvalidOperation = true;
      return WalkResult::interrupt();
    } else if (isLoop(op)) {
      // Here can be logic of how to handle loops
      hasInvalidOperation = true;
      return WalkResult::interrupt();
    } else if (isCallOp(op)) {
      // Here can be logic of how to handle function calls
      hasInvalidOperation = true;
      return WalkResult::interrupt();
    } else {
      return WalkResult::advance();
    }
    return WalkResult::skip();
  });

  size_t depth = 0;
  for (const auto& [key, _depth] : depthOf) {
    depth = std::max(depth, _depth);
  }

  if (hasInvalidOperation) {
    return std::nullopt;
  }

  return depth;
}

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
  forEachGate(moduleOp, [&](Operation* op, const size_t numQubits) {
    ++info.numGates;
    info.numSingleQubitGates += numQubits == 1;
    info.numTwoQubitGates += numQubits == 2;
    ++info.gateCounts[gateName(op)];
  });

  info.depth = computeDepth(moduleOp);

  return info;
}
QuantumProgramInfo QCProgram::inspect() const { return inspectProgram(mod()); }
QuantumProgramInfo QCOProgram::inspect() const { return inspectProgram(mod()); }

static size_t countGatesIf(ModuleOp moduleOp,
                           function_ref<bool(size_t)> predicate) {
  size_t count = 0;
  forEachGate(moduleOp, [&](Operation*, const size_t arity) {
    count += predicate(arity);
  });
  return count;
}

static std::map<std::string, size_t> countGatesByName(ModuleOp moduleOp) {
  std::map<std::string, size_t> counts;
  forEachGate(moduleOp, [&](Operation* op, size_t) { ++counts[gateName(op)]; });
  return counts;
}

size_t QCProgram::numGates() const {
  return countGatesIf(mod(), [](size_t) { return true; });
}

size_t QCProgram::numSingleQubitGates() const {
  return countGatesIf(mod(), [](const size_t arity) { return arity == 1; });
}

size_t QCProgram::numTwoQubitGates() const {
  return countGatesIf(mod(), [](const size_t arity) { return arity == 2; });
}

std::map<std::string, size_t> QCProgram::gateCounts() const {
  return countGatesByName(mod());
}

std::map<std::string, size_t> QCProgram::controlFlowCounts() const {
  return countOperationsIf(mqt::getEntryPoint(mod()), isControlFlow);
}

std::optional<size_t> QCProgram::depth() const { return computeDepth(mod()); }

size_t QCOProgram::numGates() const {
  return countGatesIf(mod(), [](size_t) { return true; });
}

size_t QCOProgram::numSingleQubitGates() const {
  return countGatesIf(mod(), [](const size_t arity) { return arity == 1; });
}

size_t QCOProgram::numTwoQubitGates() const {
  return countGatesIf(mod(), [](const size_t arity) { return arity == 2; });
}

std::map<std::string, size_t> QCOProgram::gateCounts() const {
  return countGatesByName(mod());
}

std::map<std::string, size_t> QCOProgram::controlFlowCounts() const {
  return countOperationsIf(mqt::getEntryPoint(mod()), isControlFlow);
}

std::optional<size_t> QCOProgram::depth() const { return computeDepth(mod()); }

} // namespace mlir
