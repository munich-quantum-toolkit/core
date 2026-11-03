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

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
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
  return isa<qc::MeasureOp, qco::MeasureOp>(op);
}

static bool isReset(Operation* op) {
  return isa<qc::ResetOp, qco::ResetOp>(op);
}

static bool isBarrier(Operation* op) {
  return isa<qc::BarrierOp, qco::BarrierOp>(op);
}

static bool isGateModifier(Operation* op) {
  return isa<qc::CtrlOp, qc::InvOp, qc::PowOp, qco::CtrlOp, qco::InvOp,
             qco::PowOp>(op);
}

static bool isCallOp(Operation* op) { return isa<CallOpInterface>(op); }

static bool isStandardGate(Operation* op) {
  return isa<qc::UnitaryOpInterface, qco::UnitaryOpInterface>(op) &&
         !isBarrier(op) && !isGateModifier(op) && !isCallOp(op);
}

static bool isLoop(Operation* op) { return isa<LoopLikeOpInterface>(op); }

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

// The key of a register slot, if its index is a known constant.
static std::optional<Key> slotKey(Value reg, Value index) {
  if (auto constant = getConstantIntValue(index)) {
    return Key(reg, *constant);
  }
  return std::nullopt;
}

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

// Add the key of `in` to `keys` and continue its wire at `out`.
static void continueWire(Value in, Value out, SmallVectorImpl<Key>& keys,
                         llvm::DenseMap<Value, Key>& keyOf) {
  if (auto it = keyOf.find(in); it != keyOf.end()) {
    keys.push_back(it->second);
    keyOf[out] = keys.back();
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
    continueWire(in, out, keys, keyOf);
  }
  bumpLayer(keys, addLayer, depthOf);
}

// A measurement is one layer over its qubit and the bit it is stored to.
// The bit is found by following the result to its `cbit.store`.
static void increaseDepthMeasurement(Value qubitIn, Value qubitOut,
                                     Value result,
                                     llvm::DenseMap<Key, size_t>& depthOf,
                                     llvm::DenseMap<Value, Key>& keyOf) {
  SmallVector<Key> keys;
  continueWire(qubitIn, qubitOut, keys, keyOf);

  for (Operation* user : result.getUsers()) {
    if (auto store = dyn_cast<cbit::StoreOp>(user)) {
      if (auto bit = slotKey(store.getReg(), store.getIndex())) {
        keys.push_back(*bit);
        keyOf[result] = *bit;
      }
    }
  }
  bumpLayer(keys, true, depthOf);
}

// Add the keys of the classical bits that a condition depends on. Frontends may
// derive the condition from loaded bits, for example with `arith.cmpi`.
static LogicalResult addConditionKeys(Value cond, SmallVectorImpl<Key>& keys,
                                      const llvm::DenseMap<Value, Key>& keyOf) {
  if (auto it = keyOf.find(cond); it != keyOf.end()) {
    keys.push_back(it->second);
    return success();
  }
  llvm::SetVector<Operation*> slice;
  if (failed(getBackwardSlice(cond, &slice))) {
    return failure();
  }
  const size_t numKeys = keys.size();
  for (Operation* op : slice) {
    if (auto load = dyn_cast<cbit::LoadOp>(op)) {
      auto it = keyOf.find(load.getResult());
      if (it == keyOf.end()) {
        return failure();
      }
      keys.push_back(it->second);
    }
  }
  return success(keys.size() > numKeys);
}

// The register that a tensor continues, or the tensor itself.
static Value rootOf(Value tensor,
                    const llvm::DenseMap<Value, Value>& registerOf) {
  Value root = registerOf.lookup(tensor);
  return root ? root : tensor;
}

static bool isQubitStore(Operation* op) {
  auto store = dyn_cast<memref::StoreOp>(op);
  return store && isa<qc::QubitType>(store.getValue().getType());
}

// Record which wire the value or register of `op` belongs to. This never
// changes a depth, so it is safe in both the main walk and body walks.
static void trackWire(Operation* op, llvm::DenseMap<Value, Key>& keyOf,
                      llvm::DenseMap<Value, Value>& registerOf,
                      llvm::DenseMap<Key, Key>& qubitInSlot) {
  if (auto load = dyn_cast<memref::LoadOp>(op)) {
    // QC qubits keep their value, so the load fixes the slot for all uses.
    if (isa<qc::QubitType>(load.getType()) && load.getIndices().size() == 1) {
      if (auto slot = slotKey(load.getMemRef(), load.getIndices().front())) {
        auto held = qubitInSlot.find(*slot);
        keyOf[load.getResult()] =
            held != qubitInSlot.end() ? held->second : *slot;
      }
    }
  } else if (isQubitStore(op)) {
    auto store = cast<memref::StoreOp>(op);
    if (store.getIndices().size() == 1) {
      if (auto slot = slotKey(store.getMemRef(), store.getIndices().front())) {
        if (auto it = keyOf.find(store.getValue()); it != keyOf.end()) {
          qubitInSlot[*slot] = it->second;
        }
      }
    }
  } else if (auto load = dyn_cast<cbit::LoadOp>(op)) {
    // A loaded bit stands for its register slot
    if (auto bit = slotKey(load.getReg(), load.getIndex())) {
      keyOf[load.getResult()] = *bit;
    }
  } else if (auto extract = dyn_cast<qtensor::ExtractOp>(op)) {
    // QCO: Extraction assigns new variables to a new tensor
    Value reg = rootOf(extract.getTensor(), registerOf);
    registerOf[extract.getOutTensor()] = reg;
    if (auto slot = slotKey(reg, extract.getIndex())) {
      keyOf[extract.getResult()] = *slot;
    }
  } else if (auto insert = dyn_cast<qtensor::InsertOp>(op)) {
    // QCO: Insertion changes the tensor, so we keep track.
    registerOf[insert.getResult()] = rootOf(insert.getDest(), registerOf);
  } else if (isa<qc::AllocOp, qco::AllocOp>(op)) {
    Value qubit = op->getResult(0);
    keyOf[qubit] = Key(qubit, 0);
  } else {
    // Equal indices are the same physical qubit, so the key has no register.
    TypeSwitch<Operation*>(op).Case<qc::StaticOp, qco::StaticOp>(
        [&](auto qubit) {
          keyOf[qubit.getResult()] =
              Key(Value(), static_cast<int64_t>(qubit.getIndex()));
        });
  }
}

// Whether depth does not support `op` inside a conditional or modifier body.
// A store in a branch may not run, so the qubit in its slot would be unknown.
static bool isUnsupportedInBody(Operation* op) {
  return isControlFlow(op) || isLoop(op) || isCallOp(op) || isQubitStore(op);
}

// Whether the body of a operation holds an unsupported operation.
static bool containsUnsupportedOp(Operation* op) {
  for (Region& region : op->getRegions()) {
    WalkResult result = region.walk([](Operation* inner) {
      return isUnsupportedInBody(inner) ? WalkResult::interrupt()
                                        : WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      return true;
    }
  }
  return false;
}

// Add the keys of all wires used in the branches of a conditional. The bodies
// add no layers themselves, as the conditional is one layer. Fails if a body
// holds an unsupported operation.
static LogicalResult addBodyKeys(Operation* conditional,
                                 SmallVectorImpl<Key>& keys,
                                 llvm::DenseMap<Value, Key>& keyOf,
                                 llvm::DenseMap<Value, Value>& registerOf,
                                 llvm::DenseMap<Key, Key>& qubitInSlot) {
  for (Region& region : conditional->getRegions()) {
    WalkResult result = region.walk([&](Operation* inner) {
      if (isUnsupportedInBody(inner)) {
        return WalkResult::interrupt();
      }
      trackWire(inner, keyOf, registerOf, qubitInSlot);
      for (Value operand : inner->getOperands()) {
        if (auto it = keyOf.find(operand); it != keyOf.end()) {
          keys.push_back(it->second);
        }
      }
      if (auto store = dyn_cast<cbit::StoreOp>(inner)) {
        if (auto bit = slotKey(store.getReg(), store.getIndex())) {
          keys.push_back(*bit);
        }
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      return failure();
    }
  }
  return success();
}

// Compute the depth by assigning every qubit and bit to a wire, and every
// operation to the layer after the deepest wire it uses.
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

  // QC: Which qubit a register slot holds after a store
  llvm::DenseMap<Key, Key> qubitInSlot;

  WalkResult result = entryPoint.walk<WalkOrder::PreOrder>([&](Operation* op) {
    trackWire(op, keyOf, registerOf, qubitInSlot);
    // A qubit without a wire has no known depth, so the result would be wrong.
    for (Value operand : op->getOperands()) {
      if (isa<qc::QubitType, qco::QubitType>(operand.getType()) &&
          !keyOf.contains(operand)) {
        return WalkResult::interrupt();
      }
    }
    if (isGateModifier(op) && containsUnsupportedOp(op)) {
      return WalkResult::interrupt();
    }
    if (isStandardGate(op) || isGateModifier(op) || isBarrier(op)) {
      // Barriers only align their qubits.
      const bool addLayer = !isBarrier(op);
      if (auto gate = dyn_cast<qc::UnitaryOpInterface>(op)) {
        increaseDepth(gate.getQubits(), gate.getQubits(), addLayer, depthOf,
                      keyOf);
      } else if (auto gate = dyn_cast<qco::UnitaryOpInterface>(op)) {
        increaseDepth(gate.getInputQubits(), gate.getOutputQubits(), addLayer,
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
    } else if (isControlFlow(op)) {
      if (auto ifop = dyn_cast<scf::IfOp>(op)) {
        SmallVector<Key> keys;
        if (failed(addConditionKeys(ifop.getCondition(), keys, keyOf)) ||
            failed(addBodyKeys(ifop, keys, keyOf, registerOf, qubitInSlot))) {
          return WalkResult::interrupt();
        }
        bumpLayer(keys, true, depthOf);
        return WalkResult::skip();
      }
      if (auto ifop = dyn_cast<qco::IfOp>(op)) {
        // The block arguments and the results continue the registers and
        // qubits passed in.
        for (auto [in, thenArg, elseArg, out] : llvm::zip_equal(
                 ifop.getQubits(), ifop.thenBlock()->getArguments(),
                 ifop.elseBlock()->getArguments(), ifop.getLinearResults())) {
          Value reg = rootOf(in, registerOf);
          registerOf[thenArg] = reg;
          registerOf[elseArg] = reg;
          registerOf[out] = reg;
          if (auto it = keyOf.find(in); it != keyOf.end()) {
            Key key = it->second;
            keyOf[thenArg] = key;
            keyOf[elseArg] = key;
            keyOf[out] = key;
          }
        }
        SmallVector<Key> keys;
        if (failed(addConditionKeys(ifop.getCondition(), keys, keyOf)) ||
            failed(addBodyKeys(ifop, keys, keyOf, registerOf, qubitInSlot))) {
          return WalkResult::interrupt();
        }
        bumpLayer(keys, true, depthOf);
        return WalkResult::skip();
      }
      // Complex control flow
      return WalkResult::interrupt();
    } else if (isLoop(op) || isCallOp(op)) {
      // Unsupported operations
      return WalkResult::interrupt();
    } else {
      // Important to advance for operations not matching a branch
      return WalkResult::advance();
    }
    return WalkResult::skip();
  });

  if (result.wasInterrupted()) {
    return std::nullopt;
  }

  size_t depth = 0;
  for (const auto& [key, nodeDepth] : depthOf) {
    depth = std::max(depth, nodeDepth);
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

QuantumProgramAnalysis::QuantumProgramAnalysis(ModuleOp moduleOp)
    : info_(inspectProgram(moduleOp)) {}

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
