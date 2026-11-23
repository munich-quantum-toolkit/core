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
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/RegionUtils.h"

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

// A wire is the register value and the index within it.
using Wire = std::pair<Value, int64_t>;

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

// Whether `type` is a qubit or a register of qubits, such as a QC memref or a
// QCO tensor.
static bool holdsQubits(Type type) {
  if (auto shaped = dyn_cast<ShapedType>(type)) {
    type = shaped.getElementType();
  }
  return isa<qc::QubitType, qco::QubitType>(type);
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

// The wire of a register slot, if its index is a known constant.
static std::optional<Wire> slotWire(Value reg, Value index) {
  if (auto constant = getConstantIntValue(index)) {
    return Wire(reg, *constant);
  }
  return std::nullopt;
}

// Whether `ifOp` only computes classical values, such as a lowered `&&`.
static bool isClassicalIf(scf::IfOp ifOp) {
  if (ifOp.getNumResults() == 0) {
    return false;
  }
  return !ifOp->walk([&](Operation* inner) {
                return llvm::any_of(inner->getOperandTypes(), holdsQubits) ||
                               llvm::any_of(inner->getResultTypes(),
                                            holdsQubits)
                           ? WalkResult::interrupt()
                           : WalkResult::advance();
              })
              .wasInterrupted();
}

static bool isQubitStore(Operation* op) {
  auto store = dyn_cast<memref::StoreOp>(op);
  return store && isa<qc::QubitType>(store.getValue().getType());
}

// The slot of a QC qubit register access, if it has one constant index.
static std::optional<Wire> qubitSlot(Value buffer, ValueRange indices) {
  while (auto cast = buffer.getDefiningOp<memref::CastOp>()) {
    buffer = cast.getSource();
  }
  if (indices.size() != 1 || buffer.getDefiningOp<ViewLikeOpInterface>()) {
    return std::nullopt;
  }
  return slotWire(buffer, indices.front());
}

// Whether depth does not support `op` inside a conditional or modifier body.
static bool isUnsupportedInBody(Operation* op) {
  return isControlFlow(op) || isLoop(op) || isCallOp(op);
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

// The highest current depth of `wires`.
static size_t maxDepth(ArrayRef<Wire> wires,
                       const llvm::DenseMap<Wire, size_t>& depthOf) {
  size_t depth = 0;
  for (const Wire& wire : wires) {
    depth = std::max(depth, depthOf.lookup(wire));
  }
  return depth;
}

// Set all wires to their highest current depth, plus one if `addLayer`.
static void bumpLayer(ArrayRef<Wire> wires, bool addLayer,
                      llvm::DenseMap<Wire, size_t>& depthOf) {
  const size_t depth = maxDepth(wires, depthOf) + (addLayer ? 1 : 0);
  for (const Wire& wire : wires) {
    depthOf[wire] = depth;
  }
}

// Add the wire of `in` to `wires` and continue its wire at `out`.
static void continueWire(Value in, Value out, SmallVectorImpl<Wire>& wires,
                         llvm::DenseMap<Value, Wire>& wireOf) {
  if (auto it = wireOf.find(in); it != wireOf.end()) {
    wires.push_back(it->second);
    wireOf[out] = wires.back();
  }
}

// Add a layer over the wires behind `inputs` and the `classical` bit wires.
// Inputs and outputs correspond by position: in QC they are the same values, in
// QCO each output continues the register slot of its input.
static void increaseDepth(ValueRange inputs, ValueRange outputs, bool addLayer,
                          llvm::DenseMap<Wire, size_t>& depthOf,
                          llvm::DenseMap<Value, Wire>& wireOf,
                          ArrayRef<Wire> classical = {}) {
  SmallVector<Wire> wires(classical);
  for (auto [in, out] : llvm::zip_equal(inputs, outputs)) {
    continueWire(in, out, wires, wireOf);
  }
  bumpLayer(wires, addLayer, depthOf);
}

// A measurement is one layer over its qubit and the bit it is stored to.
// The bit is found by following the result to its `cbit.store`. A result that
// is never stored gets its own wire, so values derived from it still wait for
// the measurement.
static void increaseDepthMeasurement(Value qubitIn, Value qubitOut,
                                     Value result,
                                     llvm::DenseMap<Wire, size_t>& depthOf,
                                     llvm::DenseMap<Value, Wire>& wireOf) {
  SmallVector<Wire> wires;
  continueWire(qubitIn, qubitOut, wires, wireOf);

  for (Operation* user : result.getUsers()) {
    if (auto store = dyn_cast<cbit::StoreOp>(user)) {
      if (auto bit = slotWire(store.getReg(), store.getIndex())) {
        wires.push_back(*bit);
        wireOf[result] = *bit;
      }
    }
  }
  if (!wireOf.contains(result)) {
    wires.emplace_back(result, 0);
    wireOf[result] = wires.back();
  }
  bumpLayer(wires, true, depthOf);
}

// Add the wires of the classical bits that `value` depends on. Frontends may
// derive values from loaded bits, for example with `arith.cmpi`, or read whole
// registers with `cbit.read`. Fails if a bit has no known wire.
static LogicalResult
addClassicalWires(Value value, SmallVectorImpl<Wire>& wires,
                  const llvm::DenseMap<Value, Wire>& wireOf) {
  if (auto it = wireOf.find(value); it != wireOf.end()) {
    wires.push_back(it->second);
    return success();
  }
  if (isa<BlockArgument>(value)) {
    return success();
  }
  BackwardSliceOptions options;
  options.inclusive = true;
  options.omitBlockArguments = true;
  options.omitUsesFromAbove = false;
  llvm::SetVector<Operation*> slice;
  if (failed(getBackwardSlice(value, &slice, options))) {
    return failure();
  }
  for (Operation* op : slice) {
    WalkResult result = op->walk([&](Operation* inner) {
      if (auto load = dyn_cast<cbit::LoadOp>(inner)) {
        auto it = wireOf.find(load.getResult());
        if (it == wireOf.end()) {
          return WalkResult::interrupt();
        }
        wires.push_back(it->second);
      } else if (auto read = dyn_cast<cbit::ReadOp>(inner)) {
        const unsigned width = read.getType().getWidth();
        for (unsigned index = 0; index < width; ++index) {
          wires.emplace_back(read.getReg(), index);
        }
      } else if (isMeasurement(inner)) {
        Value bit = isa<qc::MeasureOp>(inner)
                        ? cast<qc::MeasureOp>(inner).getResult()
                        : cast<qco::MeasureOp>(inner).getResult();
        auto it = wireOf.find(bit);
        if (it == wireOf.end()) {
          return WalkResult::interrupt();
        }
        wires.push_back(it->second);
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      return failure();
    }
  }
  return success();
}

// Like addClassicalWires(), but a condition must depend on at least one bit.
static LogicalResult
addConditionWires(Value cond, SmallVectorImpl<Wire>& wires,
                  const llvm::DenseMap<Value, Wire>& wireOf) {
  const size_t numWires = wires.size();
  if (failed(addClassicalWires(cond, wires, wireOf))) {
    return failure();
  }
  return success(wires.size() > numWires);
}

// Add the bit wires behind the classical operands of `op`, including values
// that its body uses from above.
static LogicalResult
addClassicalOperandWires(Operation* op, SmallVectorImpl<Wire>& wires,
                         const llvm::DenseMap<Value, Wire>& wireOf) {
  llvm::SetVector<Value> values;
  values.insert(op->operand_begin(), op->operand_end());
  getUsedValuesDefinedAbove(op->getRegions(), values);
  for (Value value : values) {
    // Qubit wires are tracked separately; only classical values read bits.
    if (!holdsQubits(value.getType()) &&
        failed(addClassicalWires(value, wires, wireOf))) {
      return failure();
    }
  }
  return success();
}

// The register that a tensor continues, or the tensor itself.
static Value rootOf(Value tensor,
                    const llvm::DenseMap<Value, Value>& registerOf) {
  Value root = registerOf.lookup(tensor);
  return root ? root : tensor;
}

// Record which wire the value or register of `op` belongs to. This never
// changes a depth, so it is safe in both the main walk and body walks.
// A value whose wire is unknown gets no entry; its first use then fails. Fails
// only where a wire would otherwise be lost without such a use.
static LogicalResult trackWire(Operation* op,
                               llvm::DenseMap<Value, Wire>& wireOf,
                               llvm::DenseMap<Value, Value>& registerOf,
                               llvm::DenseMap<Wire, Wire>& qubitInSlot) {
  if (auto load = dyn_cast<memref::LoadOp>(op)) {
    // QC qubits keep their value, so the load fixes the slot for all uses.
    if (!isa<qc::QubitType>(load.getType())) {
      return success();
    }
    if (auto slot = qubitSlot(load.getMemRef(), load.getIndices())) {
      auto held = qubitInSlot.find(*slot);
      wireOf[load.getResult()] =
          held != qubitInSlot.end() ? held->second : *slot;
    }
  } else if (isQubitStore(op)) {
    // An unknown slot would lose which qubit later loads refer to.
    auto store = cast<memref::StoreOp>(op);
    auto slot = qubitSlot(store.getMemRef(), store.getIndices());
    if (!slot) {
      return failure();
    }
    if (auto it = wireOf.find(store.getValue()); it != wireOf.end()) {
      qubitInSlot[*slot] = it->second;
    }
  } else if (auto load = dyn_cast<cbit::LoadOp>(op)) {
    // A loaded bit stands for its register slot
    if (auto bit = slotWire(load.getReg(), load.getIndex())) {
      wireOf[load.getResult()] = *bit;
    }
  } else if (auto extract = dyn_cast<qtensor::ExtractOp>(op)) {
    // QCO: Extraction assigns new variables to a new tensor
    Value reg = rootOf(extract.getTensor(), registerOf);
    registerOf[extract.getOutTensor()] = reg;
    if (auto slot = slotWire(reg, extract.getIndex())) {
      auto held = qubitInSlot.find(*slot);
      wireOf[extract.getResult()] =
          held != qubitInSlot.end() ? held->second : *slot;
    }
  } else if (auto pack = dyn_cast<qtensor::FromElementsOp>(op)) {
    // QCO: Each slot of a packed tensor holds the wire of its element.
    for (auto [index, element] : llvm::enumerate(pack.getElements())) {
      if (auto it = wireOf.find(element); it != wireOf.end()) {
        qubitInSlot[Wire(pack.getResult(), index)] = it->second;
      }
    }
  } else if (auto insert = dyn_cast<qtensor::InsertOp>(op)) {
    // QCO: Insertion changes the tensor, so we keep track.
    registerOf[insert.getResult()] = rootOf(insert.getDest(), registerOf);
  } else if (isa<qc::AllocOp, qco::AllocOp>(op)) {
    Value qubit = op->getResult(0);
    wireOf[qubit] = Wire(qubit, 0);
  } else if (auto qubit = dyn_cast<qc::StaticOp>(op)) {
    // Equal indices are the same physical qubit, so the wire has no register.
    wireOf[qubit.getResult()] =
        Wire(Value(), static_cast<int64_t>(qubit.getIndex()));
  } else if (auto qubit = dyn_cast<qco::StaticOp>(op)) {
    wireOf[qubit.getResult()] =
        Wire(Value(), static_cast<int64_t>(qubit.getIndex()));
  }
  return success();
}

// Add all wires used in the branches of a conditional. The bodies add no
// layers themselves, as the conditional is one layer. Fails if a body holds an
// unsupported operation or a qubit without a known wire.
static LogicalResult addBodyWires(Operation* conditional,
                                  SmallVectorImpl<Wire>& wires,
                                  llvm::DenseMap<Value, Wire>& wireOf,
                                  llvm::DenseMap<Value, Value>& registerOf,
                                  llvm::DenseMap<Wire, Wire>& qubitInSlot) {
  for (Region& region : conditional->getRegions()) {
    WalkResult result = region.walk<WalkOrder::PreOrder>([&](Operation* inner) {
      if (isUnsupportedInBody(inner) ||
          (isGateModifier(inner) && containsUnsupportedOp(inner))) {
        return WalkResult::interrupt();
      }
      if (failed(trackWire(inner, wireOf, registerOf, qubitInSlot))) {
        return WalkResult::interrupt();
      }
      for (Value operand : inner->getOperands()) {
        if (auto it = wireOf.find(operand); it != wireOf.end()) {
          wires.push_back(it->second);
        } else if (isa<qc::QubitType, qco::QubitType>(operand.getType())) {
          return WalkResult::interrupt();
        }
      }
      if (auto store = dyn_cast<cbit::StoreOp>(inner)) {
        if (auto bit = slotWire(store.getReg(), store.getIndex())) {
          wires.push_back(*bit);
        }
      }
      // QCO: Outputs continue the wires of their inputs, so later operations
      // in the body resolve.
      if (auto gate = dyn_cast<qco::UnitaryOpInterface>(inner)) {
        for (auto [in, out] :
             llvm::zip_equal(gate.getInputQubits(), gate.getOutputQubits())) {
          continueWire(in, out, wires, wireOf);
        }
      } else if (auto measure = dyn_cast<qco::MeasureOp>(inner)) {
        continueWire(measure.getQubitIn(), measure.getQubitOut(), wires,
                     wireOf);
      } else if (auto reset = dyn_cast<qco::ResetOp>(inner)) {
        continueWire(reset.getQubitIn(), reset.getQubitOut(), wires, wireOf);
      }
      // A modifier's operands are its wires; its body uses block arguments
      // that alias them and have no wires.
      return isGateModifier(inner) ? WalkResult::skip() : WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      return failure();
    }
  }
  return success();
}

// A conditional is one layer over the bits of `cond`, the bits behind its
// classical operands, and every wire that its branches use.
static LogicalResult
addConditionalLayer(Operation* conditional, Value cond,
                    llvm::DenseMap<Wire, size_t>& depthOf,
                    llvm::DenseMap<Value, Wire>& wireOf,
                    llvm::DenseMap<Value, Value>& registerOf,
                    llvm::DenseMap<Wire, Wire>& qubitInSlot) {
  SmallVector<Wire> wires;
  if (failed(addConditionWires(cond, wires, wireOf)) ||
      failed(addClassicalOperandWires(conditional, wires, wireOf)) ||
      failed(
          addBodyWires(conditional, wires, wireOf, registerOf, qubitInSlot))) {
    return failure();
  }
  bumpLayer(wires, true, depthOf);
  return success();
}

// Compute the depth by assigning every qubit and bit to a wire, and every
// operation to the layer after the deepest wire it uses.
static std::optional<size_t> computeDepth(ModuleOp moduleOp) {
  auto entryPoint = mqt::getEntryPoint(moduleOp);
  if (!entryPoint) {
    return std::nullopt;
  }

  // The depth that each wire has reached.
  llvm::DenseMap<Wire, size_t> depthOf;

  // The wire that each qubit or bit value belongs to.
  llvm::DenseMap<Value, Wire> wireOf;

  // QCO: The register that each tensor continues.
  llvm::DenseMap<Value, Value> registerOf;

  // The wire of the qubit that a register slot holds after a QC store or a QCO
  // packing.
  llvm::DenseMap<Wire, Wire> qubitInSlot;

  WalkResult result = entryPoint.walk<WalkOrder::PreOrder>([&](Operation* op) {
    if (failed(trackWire(op, wireOf, registerOf, qubitInSlot))) {
      return WalkResult::interrupt();
    }
    // A qubit without a wire has no known depth, so the result would be wrong.
    for (Value operand : op->getOperands()) {
      if (isa<qc::QubitType, qco::QubitType>(operand.getType()) &&
          !wireOf.contains(operand)) {
        return WalkResult::interrupt();
      }
    }
    if (isGateModifier(op) && containsUnsupportedOp(op)) {
      return WalkResult::interrupt();
    }
    if (isStandardGate(op) || isGateModifier(op) || isBarrier(op)) {
      // A gate on no qubit, such as a global phase, has no layer and must not
      // delay the bits its operands read.
      if (llvm::none_of(op->getOperandTypes(), [](Type type) {
            return isa<qc::QubitType, qco::QubitType>(type);
          })) {
        return WalkResult::skip();
      }
      // Barriers only align their qubits.
      const bool addLayer = !isBarrier(op);
      // Classical operands, such as angles, wait for the bits they read.
      SmallVector<Wire> classical;
      if (failed(addClassicalOperandWires(op, classical, wireOf))) {
        return WalkResult::interrupt();
      }
      if (auto gate = dyn_cast<qc::UnitaryOpInterface>(op)) {
        increaseDepth(gate.getQubits(), gate.getQubits(), addLayer, depthOf,
                      wireOf, classical);
      } else if (auto gate = dyn_cast<qco::UnitaryOpInterface>(op)) {
        increaseDepth(gate.getInputQubits(), gate.getOutputQubits(), addLayer,
                      depthOf, wireOf, classical);
      }
    } else if (auto store = dyn_cast<cbit::StoreOp>(op)) {
      // c1 becomes ready once the stored value is; a copy adds no layer.
      auto bit = slotWire(store.getReg(), store.getIndex());
      if (!bit) {
        return WalkResult::interrupt();
      }
      SmallVector<Wire> sources;
      if (failed(addClassicalWires(store.getValue(), sources, wireOf))) {
        return WalkResult::interrupt();
      }
      const size_t ready = maxDepth(sources, depthOf);
      size_t& depth = depthOf[*bit];
      depth = std::max(depth, ready);
    } else if (auto write = dyn_cast<cbit::WriteOp>(op)) {
      // A register write, such as `d = c` or `d = ~c`, adds no layer. Like in
      // Qiskit, every written bit waits for all bits that the value reads.
      const unsigned width = write.getValue().getType().getWidth();
      SmallVector<Wire> sources;
      if (failed(addClassicalWires(write.getValue(), sources, wireOf))) {
        return WalkResult::interrupt();
      }
      const size_t ready = maxDepth(sources, depthOf);
      for (unsigned index = 0; index < width; ++index) {
        size_t& depth = depthOf[Wire(write.getReg(), index)];
        depth = std::max(depth, ready);
      }
    } else if (isMeasurement(op)) {
      if (auto measure = dyn_cast<qc::MeasureOp>(op)) {
        increaseDepthMeasurement(measure.getQubit(), measure.getQubit(),
                                 measure.getResult(), depthOf, wireOf);
      } else if (auto measure = dyn_cast<qco::MeasureOp>(op)) {
        increaseDepthMeasurement(measure.getQubitIn(), measure.getQubitOut(),
                                 measure.getResult(), depthOf, wireOf);
      }
    } else if (isReset(op)) {
      if (auto reset = dyn_cast<qc::ResetOp>(op)) {
        increaseDepth(reset.getQubit(), reset.getQubit(), true, depthOf,
                      wireOf);
      } else if (auto reset = dyn_cast<qco::ResetOp>(op)) {
        increaseDepth(reset.getQubitIn(), reset.getQubitOut(), true, depthOf,
                      wireOf);
      }
    } else if (isControlFlow(op)) {
      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        if (isClassicalIf(ifOp)) {
          // Track the bits that the body reads for the condition using it.
          return WalkResult::advance();
        }
        return succeeded(addConditionalLayer(ifOp, ifOp.getCondition(), depthOf,
                                             wireOf, registerOf, qubitInSlot))
                   ? WalkResult::skip()
                   : WalkResult::interrupt();
      }
      if (auto ifOp = dyn_cast<qco::IfOp>(op)) {
        // The block arguments and the results continue the registers and
        // qubits passed in.
        for (auto [in, thenArg, elseArg, out] : llvm::zip_equal(
                 ifOp.getQubits(), ifOp.thenBlock()->getArguments(),
                 ifOp.elseBlock()->getArguments(), ifOp.getLinearResults())) {
          Value reg = rootOf(in, registerOf);
          registerOf[thenArg] = reg;
          registerOf[elseArg] = reg;
          registerOf[out] = reg;
          if (auto it = wireOf.find(in); it != wireOf.end()) {
            Wire wire = it->second;
            wireOf[thenArg] = wire;
            wireOf[elseArg] = wire;
            wireOf[out] = wire;
          }
        }
        return succeeded(addConditionalLayer(ifOp, ifOp.getCondition(), depthOf,
                                             wireOf, registerOf, qubitInSlot))
                   ? WalkResult::skip()
                   : WalkResult::interrupt();
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
  for (const auto& [wire, wireDepth] : depthOf) {
    depth = std::max(depth, wireDepth);
  }
  return depth;
}

static QuantumProgramInfo inspectProgram(ModuleOp moduleOp) {
  QuantumProgramInfo info;
  auto entry = mqt::getEntryPoint(moduleOp);
  info.operationCounts =
      countOperationsIf(moduleOp, [](Operation*) { return true; });
  info.controlFlowCounts = countOperationsIf(entry, isControlFlow);
  if (entry && llvm::none_of(entry.getArgumentTypes(), holdsQubits)) {
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
