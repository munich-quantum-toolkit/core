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
#include "mqt/Dialect/CBit/IR/CBitOps.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/MQT/Utils/Modifiers.h"
#include "mqt/Dialect/QC/IR/QCInterfaces.h"
#include "mqt/Dialect/QC/IR/QCOps.h"
#include "mqt/Dialect/QCO/IR/QCOInterfaces.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
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
  return llvm::isa<qc::CtrlOp>(op) || llvm::isa<qc::InvOp>(op) ||
         llvm::isa<qc::PowOp>(op) || llvm::isa<qco::CtrlOp>(op) ||
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

// A key is the register value and the index within it.
using Key = std::pair<Value, int64_t>;

// Set all wires/keys to their highest current depth, plus one if `addLayer`.
static void bumpLayer(ArrayRef<Key> keys, bool addLayer,
                      llvm::DenseMap<Key, size_t>& depthOf) {
  size_t maxDepth = 0;
  for (const Key& key : keys) {
    maxDepth = std::max(maxDepth, depthOf.lookup(key));
  }
  if (addLayer) {
    maxDepth += 1;
  }
  for (const Key& key : keys) {
    depthOf[key] = maxDepth;
  }
}

// Add a layer over the keys behind `inputs`. Inputs and outputs correspond by
// position: in QC they are the same values, in QCO each output continues the
// register slot of its input.
static void increaseDepth(ValueRange inputs, ValueRange outputs, bool addLayer,
                          llvm::DenseMap<Key, size_t>& depthOf,
                          llvm::DenseMap<Value, Key>& keyOf) {
  SmallVector<Key> keys;
  for (auto [in, out] : llvm::zip_equal(inputs, outputs)) {
    if (auto it = keyOf.find(in); it != keyOf.end()) {
      Key key = it->second;
      keys.push_back(key);
      keyOf[out] = key;
    }
  }
  bumpLayer(keys, addLayer, depthOf);
}

// A measurement is one layer over its qubit and the bit it is stored to.
// Classical Bits are only tracked through measurements, by following the result
// to its `cbit.store`/`cbit::StoreOp`.
static void increaseDepthMeasurement(Value qubitIn, Value qubitOut,
                                     Value result,
                                     llvm::DenseMap<Key, size_t>& depthOf,
                                     llvm::DenseMap<Value, Key>& keyOf) {
  SmallVector<Key> keys;
  if (auto it = keyOf.find(qubitIn); it != keyOf.end()) {
    Key qubit = it->second;
    keys.push_back(qubit);
    keyOf[qubitOut] = qubit;
  }

  for (Operation* user : result.getUsers()) {
    if (auto store = dyn_cast<cbit::StoreOp>(user)) {
      if (auto index = getConstantIntValue(store.getIndex())) {
        Key bit = std::make_pair(Value(store.getReg()), *index);
        keys.push_back(bit);
        keyOf[result] = bit;
      }
    }
  }
  bumpLayer(keys, true, depthOf);
}

// Function to compute the depth of a QCProgram or QCOProgram.
static std::optional<size_t> computeDepth(ModuleOp moduleOp) {
  auto entryPoint = mqt::getEntryPoint(moduleOp);

  if (!entryPoint) {
    return std::nullopt;
  }

  // Map to track the depth of wires/keys
  llvm::DenseMap<Key, size_t> depthOf;

  // Map to track which key/wire a value is coming from
  llvm::DenseMap<Value, Key> keyOf;

  // QCO: Which register a register is coming from
  llvm::DenseMap<Value, Value> registerOf;
  const auto rootOf = [&](Value tensor) {
    Value root = registerOf.lookup(tensor);
    return root ? root : tensor;
  };

  WalkResult result = entryPoint.walk<WalkOrder::PreOrder>([&](Operation* op) {
    if (isStandardGate(op) || isGateModifier(op)) {
      if (auto gate = dyn_cast<qc::UnitaryOpInterface>(op)) {
        increaseDepth(gate.getQubits(), gate.getQubits(), true, depthOf, keyOf);
      } else if (auto gate = dyn_cast<qco::UnitaryOpInterface>(op)) {
        increaseDepth(gate.getInputQubits(), gate.getOutputQubits(), true,
                      depthOf, keyOf);
      }
    } else if (isMeasurement(op)) {
      if (auto measure = dyn_cast<qc::MeasureOp>(op)) {
        increaseDepthMeasurement(measure.getQubit(), measure.getQubit(),
                                 measure.getResult(), depthOf, keyOf);
      } else if (auto measure = dyn_cast<qco::MeasureOp>(op)) {
        increaseDepthMeasurement(measure.getQubitIn(), measure.getQubitOut(),
                                 measure.getResult(), depthOf, keyOf);
      }
    } else if (isReset(op)) {
      if (auto reset = dyn_cast<qc::ResetOp>(op)) {
        increaseDepth(reset.getQubit(), reset.getQubit(), true, depthOf, keyOf);
      } else if (auto reset = dyn_cast<qco::ResetOp>(op)) {
        increaseDepth(reset.getQubitIn(), reset.getQubitOut(), true, depthOf,
                      keyOf);
      }
    } else if (isBarrier(op)) {
      if (auto barrier = dyn_cast<qc::BarrierOp>(op)) {
        increaseDepth(barrier.getQubits(), barrier.getQubits(), false, depthOf,
                      keyOf);
      } else if (auto barrier = dyn_cast<qco::BarrierOp>(op)) {
        increaseDepth(barrier.getQubitsIn(), barrier.getQubitsOut(), false,
                      depthOf, keyOf);
      }
    }

    // Operations that set keys to keep track
    else if (auto load = dyn_cast<memref::LoadOp>(op)) {
      // QC qubits keep their value, so the load fixes the slot for all uses.
      if (load.getIndices().size() == 1) {
        if (auto index = getConstantIntValue(load.getIndices().front())) {
          keyOf[load.getResult()] = std::make_pair(load.getMemRef(), *index);
        }
      }
    } else if (auto load = dyn_cast<cbit::LoadOp>(op)) {
      // A loaded bit stands for its register slot
      if (auto index = getConstantIntValue(load.getIndex())) {
        keyOf[load.getResult()] = std::make_pair(load.getReg(), *index);
      }
    } else if (auto extract = dyn_cast<qtensor::ExtractOp>(op)) {
      // QCO: Extraction assigns new variables to a new tensor
      Value reg = rootOf(extract.getTensor());
      registerOf[extract.getOutTensor()] = reg;
      if (auto index = getConstantIntValue(extract.getIndex())) {
        keyOf[extract.getResult()] = std::make_pair(reg, *index);
      }
    } else if (auto insert = dyn_cast<qtensor::InsertOp>(op)) {
      // QCO: Insertion changes the tensor, so we keep track.
      registerOf[insert.getResult()] = rootOf(insert.getDest());
    }

    else if (isControlFlow(op)) {
      if (auto ifop = dyn_cast<scf::IfOp>(op)) {
        SmallVector<Key> keys;
        Value cond = ifop.getCondition();
        Key condKey = keyOf.lookup(cond);
        keys.push_back(condKey);

        ifop.getThenRegion().walk([&](Operation* inner) {
          for (Value operand : inner->getOperands()) {
            if (auto load = operand.getDefiningOp<memref::LoadOp>()) {
              if (auto index = getConstantIntValue(load.getIndices().front())) {
                keys.push_back(std::make_pair(load.getMemRef(), *index));
              }
            }
          }
        });

        bumpLayer(keys, true, depthOf);
      } else if (auto ifop = dyn_cast<qco::IfOp>(op)) {
        SmallVector<Key> keys;
        keys.push_back(keyOf.lookup(ifop.getCondition()));

        // The block arguments and the results continue the registers passed in.
        for (auto [in, arg, out] :
             llvm::zip_equal(ifop.getQubits(), ifop.thenBlock()->getArguments(),
                             ifop.getLinearResults())) {
          registerOf[arg] = rootOf(in);
          registerOf[out] = rootOf(in);
        }

        ifop.getThenRegion().walk([&](Operation* inner) {
          if (auto extract = dyn_cast<qtensor::ExtractOp>(inner)) {
            Value reg = rootOf(extract.getTensor());
            registerOf[extract.getOutTensor()] = reg;
            if (auto index = getConstantIntValue(extract.getIndex())) {
              keys.push_back(std::make_pair(reg, *index));
            }
          } else if (auto insert = dyn_cast<qtensor::InsertOp>(inner)) {
            registerOf[insert.getResult()] = rootOf(insert.getDest());
          }
        });

        bumpLayer(keys, true, depthOf);
      } else {
        // Complex control flow
        return WalkResult::interrupt();
      }
    } else if (isLoop(op) || isCallOp(op)) {
      // Unsupported operations
      return WalkResult::interrupt();
    } else {
      // Important to advance for operations not matching a branch
      return WalkResult::advance();
    }
    return WalkResult::skip();
  });

  size_t depth = 0;
  for (const auto& [key, _depth] : depthOf) {
    depth = std::max(depth, _depth);
  }

  if (result.wasInterrupted()) {
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
