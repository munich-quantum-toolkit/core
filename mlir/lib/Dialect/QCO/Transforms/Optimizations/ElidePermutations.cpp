/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/MQT/IR/QubitLayout.h"
#include "mqt/Dialect/QCO/IR/QCOInterfaces.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mqt/Dialect/QCO/Transforms/Passes.h"
#include "mqt/Dialect/QCO/Utils/FunctionUtils.h"
#include "mqt/Dialect/QCO/Utils/Layout.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <tuple>
#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_ELIDEPERMUTATIONS
#define GEN_PASS_DEF_ELIDETERMINALSWAPSPASS
#include "mqt/Dialect/QCO/Transforms/Passes.h.inc"

using Resource = size_t;
using Slot = std::pair<int64_t, Value>;
// Region summaries use quantum argument order, independent of discovery order.
// A slot of -1 denotes a scalar argument; tensor slots are nonnegative.
using Port = std::pair<unsigned, int64_t>;
using Permutation = DenseMap<Port, Port>;
using RegionEffects = DenseMap<Operation*, unsigned>;
constexpr unsigned HAS_SWAP = 1;
constexpr unsigned OBSERVES = 2;
constexpr unsigned CAPTURES = 4;

static SmallVector<Value> quantumValues(ValueRange values) {
  return llvm::filter_to_vector(
      values, [](Value value) { return isLinearQubitType(value.getType()); });
}

static Slot slotKey(OpFoldResult index) {
  if (auto constant = getConstantIntValue(index)) {
    return {*constant, {}};
  }
  return {0, cast<Value>(index)};
}

static bool isTerminalBookkeeping(Operation* op) {
  auto insert = dyn_cast<qtensor::InsertOp>(op);
  auto extract = dyn_cast<qtensor::ExtractOp>(op);
  return (!insert || getConstantIntValue(insert.getIndex())) &&
         (!extract || getConstantIntValue(extract.getIndex())) &&
         op->getNumRegions() == 0 &&
         (isa<MeasureOp, SinkOp, qtensor::InsertOp, qtensor::ExtractOp,
              qtensor::DeallocOp>(op) ||
          (!llvm::any_of(op->getOperandTypes(), isLinearQubitType) &&
           !llvm::any_of(op->getResultTypes(), isLinearQubitType)));
}

static Operation* firstTerminalSwap(func::FuncOp entry) {
  Operation* first = nullptr;
  if (!entry || !entry.getBody().hasOneBlock()) {
    return first;
  }
  bool hasLaterGate = false;
  for (auto& op : llvm::reverse(entry.getBody().front().without_terminator())) {
    if (isa<SWAPOp>(op)) {
      first = &op;
      hasLaterGate = true;
      continue;
    }
    if (auto gate = dyn_cast<UnitaryOpInterface>(op);
        gate && gate.getNumQubits() == 1 && op.getNumRegions() == 0) {
      hasLaterGate = true;
      continue;
    }
    if ((hasLaterGate && isa<MeasureOp, SinkOp, qtensor::DeallocOp>(op)) ||
        !isTerminalBookkeeping(&op)) {
      break;
    }
  }
  return first;
}

namespace {

/// Track logical states separately from the resources carrying them.
///
/// Tensor frontiers stay in physical slot order, just like the router's scalar
/// wires.
class PermutationTracker {
  struct Wire {
    Value value;
    std::optional<size_t> tensor;
    OpFoldResult index;
    std::optional<Port> port;
    std::optional<int64_t> source;
    bool fixed = false;
  };
  struct Tensor {
    Value value;
    DenseMap<Slot, Resource> slots;
    llvm::SmallSetVector<Resource, 4> dirty;
    std::optional<unsigned> argument;
    DenseI64ArrayAttr sources;
    bool fixed = false;
  };

  IRRewriter& rewriter_;
  Block& block_;
  const RegionEffects& effects_;
  Operation* firstEligibleSwap_;
  Layout<Resource> layout_;
  SmallVector<Wire> wires_;
  SmallVector<Tensor> tensors_;
  DenseMap<Value, Resource> scalars_;
  DenseMap<Value, size_t> owners_;
  // Entries may be stale. Draining queues avoids linear SetVector removals.
  llvm::SmallSetVector<Resource, 8> dirty_;
  SmallVector<Operation*> erased_;
  SmallVector<Value> arguments_;
  Operation* terminalStart_ = nullptr;
  bool terminal_ = false;

  Resource addWire(Value value, std::optional<Port> port = std::nullopt,
                   bool fixed = false) {
    const auto id = layout_.appendIdentity();
    wires_.push_back({
        .value = value,
        .tensor = {},
        .index = {},
        .port = port,
        .source = {},
        .fixed = fixed,
    });
    if (value) {
      scalars_[value] = id;
    }
    return id;
  }

  size_t addTensor(Value value, std::optional<unsigned> argument = std::nullopt,
                   bool fixed = false) {
    const auto owner = tensors_.size();
    auto& tensor = tensors_.emplace_back();
    tensor.value = value;
    tensor.argument = argument;
    tensor.fixed = fixed;
    if (auto* root = value.getDefiningOp()) {
      tensors_.back().sources =
          root->getAttrOfType<DenseI64ArrayAttr>(mqt::kSourceQubitIndicesAttr);
    }
    owners_[value] = owner;
    return owner;
  }

  Resource slot(size_t owner, OpFoldResult index) {
    auto& tensor = tensors_[owner];
    const auto key = slotKey(index);
    if (auto found = tensor.slots.find(key); found != tensor.slots.end()) {
      return found->second;
    }
    std::optional<Port> port;
    if (tensor.argument && !key.second) {
      port = Port{*tensor.argument, key.first};
    }
    const auto id = addWire({}, port, tensor.fixed);
    auto& wire = wires_[id];
    wire.tensor = owner;
    wire.index = index;
    if (tensor.sources && !key.second && key.first >= 0 &&
        key.first < tensor.sources.size()) {
      wire.source = tensor.sources[key.first];
    }
    tensor.slots[key] = id;
    return id;
  }

  void exchange(Resource a, Resource b) {
    const auto first = layout_.getProgramIndex(a);
    const auto second = layout_.getProgramIndex(b);
    layout_.swap(a, b);
    for (const auto id : {first, second}) {
      if (layout_.getHardwareIndex(id) != id) {
        dirty_.insert(id);
        if (wires_[id].tensor) {
          tensors_[*wires_[id].tensor].dirty.insert(id);
        }
      }
    }
  }

  Value indexValue(Wire& wire) {
    auto value = getValueOrCreateConstantIndexOp(
        rewriter_, rewriter_.getInsertionPoint()->getLoc(), wire.index);
    wire.index = value;
    return value;
  }

  Value unpack(Resource id) {
    auto& wire = wires_[id];
    if (!wire.value) {
      assert(wire.tensor && "a live scalar resource must have an SSA value");
      auto& tensor = tensors_[*wire.tensor];
      auto extract = qtensor::ExtractOp::create(
          rewriter_, rewriter_.getInsertionPoint()->getLoc(), tensor.value,
          indexValue(wire));
      tensor.value = extract.getOutTensor();
      wire.value = extract.getResult();
    }
    return wire.value;
  }

  void pack(Resource id) {
    auto& wire = wires_[id];
    auto& tensor = tensors_[*wire.tensor];
    tensor.value = qtensor::InsertOp::create(
        rewriter_, rewriter_.getInsertionPoint()->getLoc(), wire.value,
        tensor.value, indexValue(wire));
    wire.value = {};
  }

  void restore(Resource id, Resource destination) {
    const auto other = layout_.getHardwareIndex(id);
    if (other == destination) {
      return;
    }
    const bool packed = !wires_[destination].value;
    const bool otherPacked = !wires_[other].value;
    auto a = unpack(destination);
    auto b = unpack(other);
    auto swap = SWAPOp::create(rewriter_,
                               rewriter_.getInsertionPoint()->getLoc(), a, b);
    wires_[destination].value = swap.getQubit0Out();
    wires_[other].value = swap.getQubit1Out();
    exchange(destination, other);
    if (packed) {
      pack(destination);
    }
    if (otherPacked) {
      pack(other);
    }
  }

  void restore(Resource id) { restore(id, id); }

  void restoreTensor(size_t owner) {
    auto& dirty = tensors_[owner].dirty;
    while (!dirty.empty()) {
      restore(dirty.pop_back_val());
    }
  }

  void restoreAll(const Permutation& entry = Permutation{}) {
    DenseMap<Resource, Resource> targets;
    for (const auto& [from, to] : entry) {
      const auto id = resource(from);
      targets[id] = resource(to);
      dirty_.insert(id);
    }
    for (const auto id : llvm::to_vector(dirty_)) {
      auto target = targets.find(id);
      restore(id, target == targets.end() ? id : target->second);
    }
  }

  void fence(Value value) {
    if (isa<QubitType>(value.getType())) {
      restore(scalars_.lookup(value));
    } else {
      restoreTensor(owners_.lookup(value));
    }
  }

  Value current(Value value) {
    if (isa<QubitType>(value.getType())) {
      return unpack(layout_.getHardwareIndex(scalars_.lookup(value)));
    }
    return tensors_[owners_.lookup(value)].value;
  }

  Resource resource(Port port) {
    auto argument = arguments_[port.first];
    if (port.second == -1) {
      return scalars_.lookup(argument);
    }
    return slot(owners_.lookup(argument), rewriter_.getIndexAttr(port.second));
  }

  void import(const Permutation& permutation) {
    SmallVector<std::pair<Resource, Resource>> targets;
    for (const auto& [from, to] : permutation) {
      targets.emplace_back(resource(from), resource(to));
    }
    for (const auto [from, to] : targets) {
      exchange(layout_.getHardwareIndex(from), to);
    }
  }

  Permutation summary() {
    // Local allocations and dynamic slots cannot name a region output port.
    for (const auto id : llvm::to_vector(dirty_)) {
      if (!wires_[id].port || !wires_[layout_.getHardwareIndex(id)].port) {
        restore(id);
      }
    }
    Permutation result;
    for (const auto id : dirty_) {
      if (layout_.getHardwareIndex(id) == id) {
        continue;
      }
      result.try_emplace(*wires_[id].port,
                         *wires_[layout_.getHardwareIndex(id)].port);
    }
    return result;
  }

  void finish(bool normalize, const Permutation& entry = Permutation{}) {
    rewriter_.setInsertionPoint(block_.getTerminator());
    if (normalize) {
      restoreAll(entry);
    }
    // Structured/call ABIs return resources in positional order. The summary
    // carries the logical permutation separately, as in the router.
    auto* terminator = block_.getTerminator();
    auto yielded = getYieldedValues(block_);
    const auto offset = terminator->getNumOperands() - yielded.size();
    for (auto [index, value] : llvm::enumerate(yielded)) {
      if (isa<QubitType>(value.getType())) {
        terminator->setOperand(offset + index, unpack(scalars_.lookup(value)));
      } else if (isLinearQubitType(value.getType())) {
        terminator->setOperand(offset + index, current(value));
      }
    }

    for (auto* op : llvm::reverse(erased_)) {
      rewriter_.eraseOp(op);
    }
    erased_.clear();
  }

  void visitExtract(qtensor::ExtractOp op) {
    const auto owner = owners_.lookup(op.getTensor());
    if (!getConstantIntValue(op.getIndex())) {
      restoreTensor(owner);
    }
    const auto logical = slot(owner, op.getIndex());
    const auto physical = layout_.getHardwareIndex(logical);
    scalars_[op.getResult()] = logical;
    owners_[op.getOutTensor()] = owner;
    if (physical != logical) {
      unpack(physical);
      erased_.push_back(op);
      return;
    }
    // The slot still holds its own state; keep the access.
    auto& tensor = tensors_[owner];
    assert(!wires_[logical].value && "a tensor slot cannot be extracted twice");
    op.getTensorMutable().assign(tensor.value);
    tensor.value = op.getOutTensor();
    wires_[logical].value = op.getResult();
  }

  void visitInsert(qtensor::InsertOp op) {
    const auto owner = owners_.lookup(op.getDest());
    const auto logical = scalars_.lookup(op.getScalar());
    auto physical = layout_.getHardwareIndex(logical);
    // Dynamic slots and independently allocated scalars retain their explicit
    // ownership boundary. Other tensor slots can be addressed directly.
    if (!getConstantIntValue(op.getIndex()) || !wires_[physical].tensor) {
      restore(logical);
      physical = logical;
    }
    if (!wires_[physical].tensor) {
      // An opaque operation may return a borrowed qubit. Insertion names its
      // original slot and therefore recovers its resource provenance.
      const auto original = slot(owner, op.getIndex());
      wires_[physical].tensor = owner;
      wires_[physical].index = op.getIndex();
      wires_[physical].port = wires_[original].port;
      wires_[physical].source = wires_[original].source;
      wires_[physical].fixed = wires_[original].fixed;
      tensors_[owner].slots[slotKey(op.getIndex())] = physical;
    }
    owners_[op.getResult()] = owner;
    if (physical != logical) {
      pack(physical);
      erased_.push_back(op);
      return;
    }
    // Insertion restores the qubit's own slot; keep the access.
    auto& tensor = tensors_[owner];
    auto& wire = wires_[physical];
    op.getScalarMutable().assign(wire.value);
    op.getDestMutable().assign(tensor.value);
    tensor.value = op.getResult();
    wire.value = {};
  }

  void visitFromElements(qtensor::FromElementsOp op) {
    SmallVector<Resource> resources;
    for (auto element : op.getElements()) {
      const auto id = scalars_.lookup(element);
      restore(id);
      resources.push_back(id);
    }
    const auto owner = addTensor(op.getResult());
    for (auto [index, id] : llvm::enumerate(resources)) {
      op->setOperand(index, wires_[id].value);
      auto& wire = wires_[id];
      wire.tensor = owner;
      wire.index = rewriter_.getIndexAttr(static_cast<int64_t>(index));
      wire.value = {};
      tensors_[owner].slots[{static_cast<int64_t>(index), {}}] = id;
      tensors_[owner].fixed |= wire.fixed;
    }
  }

  void enterTerminal() {
    terminal_ = true;
    // Retain only static cycles whose final packing/disposal cannot require a
    // missing partner. Cross-owner cycles are reconciled before measurements.
    for (const auto id : llvm::to_vector(dirty_)) {
      const auto other = layout_.getHardwareIndex(id);
      if (!wires_[id].source || !wires_[other].source ||
          wires_[id].tensor != wires_[other].tensor) {
        restore(id);
      }
    }
  }

  void fenceCaptures(Operation* op) {
    SmallVector<OpOperand*> captures;
    visitUsedValuesDefinedAbove(op->getRegions(), [&](OpOperand* operand) {
      if (isLinearQubitType(operand->get().getType())) {
        captures.push_back(operand);
      }
    });
    for (auto* operand : captures) {
      auto value = operand->get();
      fence(value);
      operand->set(current(value));
    }
  }

  Permutation tensorEntry(ArrayRef<Value> inputs) {
    DenseMap<size_t, unsigned> ports;
    SmallVector<Resource> pending;
    for (auto [index, input] : llvm::enumerate(inputs)) {
      if (!isa<QubitType>(input.getType())) {
        const auto owner = owners_.lookup(input);
        ports[owner] = static_cast<unsigned>(index);
        llvm::append_range(pending, tensors_[owner].dirty);
        tensors_[owner].dirty.clear();
      }
    }
    const auto port = [&](Resource id) -> std::optional<Port> {
      const auto& wire = wires_[id];
      if (!wire.tensor || !ports.contains(*wire.tensor)) {
        return std::nullopt;
      }
      if (auto index = getConstantIntValue(wire.index)) {
        return Port{ports.lookup(*wire.tensor), *index};
      }
      return std::nullopt;
    };
    for (const auto id : pending) {
      if (!port(id)) {
        restore(id);
        continue;
      }
      while (!port(layout_.getHardwareIndex(id))) {
        restore(layout_.getHardwareIndex(id));
      }
    }
    Permutation entry;
    for (const auto id : pending) {
      const auto physical = layout_.getHardwareIndex(id);
      if (physical != id) {
        tensors_[*wires_[id].tensor].dirty.insert(id);
        entry.try_emplace(*port(id), *port(physical));
      }
    }
    return entry;
  }

  void visitRegions(Operation* op) {
    auto branch = cast<RegionBranchOpInterface>(op);
    auto inputs = quantumValues(
        branch.getEntrySuccessorOperands(RegionSuccessor(&op->getRegion(0))));
    // Measurements can be terminal to consumers such as statevector simulation.
    // Reconcile before entering an observing region, never after its
    // measurement.
    if ((effects_.lookup(op) & OBSERVES) != 0) {
      for (auto input : inputs) {
        fence(input);
      }
    }
    const auto incoming = tensorEntry(inputs);
    SmallVector<Resource> scalarInputs;
    SmallVector<bool> fixed;
    for (auto input : inputs) {
      if (isa<QubitType>(input.getType())) {
        const auto physical = layout_.getHardwareIndex(scalars_.lookup(input));
        scalarInputs.push_back(physical);
        fixed.push_back(wires_[physical].fixed);
      } else {
        const auto owner = owners_.lookup(input);
        fixed.push_back(tensors_[owner].fixed);
      }
    }
    for (auto& operand : op->getOpOperands()) {
      if (isLinearQubitType(operand.get().getType())) {
        operand.set(current(operand.get()));
      }
    }

    if ((effects_.lookup(op) & HAS_SWAP) != 0 || !incoming.empty()) {
      SmallVector<PermutationTracker, 0> children;
      children.reserve(op->getNumRegions());
      SmallVector<Permutation> summaries;
      for (auto& region : op->getRegions()) {
        auto& block = region.front();
        auto& child = children.emplace_back(rewriter_, block, effects_,
                                            firstEligibleSwap_);
        for (auto [argument, isFixed] : llvm::zip(child.arguments_, fixed)) {
          if (isa<QubitType>(argument.getType())) {
            child.wires_[child.scalars_.lookup(argument)].fixed = isFixed;
          } else {
            child.tensors_[child.owners_.lookup(argument)].fixed = isFixed;
          }
        }
        rewriter_.setInsertionPointToStart(&block);
        child.import(isa<scf::WhileOp>(op) && !summaries.empty()
                         ? summaries.front()
                         : incoming);
        child.run();
        rewriter_.setInsertionPoint(block.getTerminator());
        summaries.push_back(isa<IfOp, IndexSwitchOp>(op) ||
                                    (isa<scf::WhileOp>(op) && summaries.empty())
                                ? child.summary()
                                : Permutation{});
      }
      Permutation exit;
      if (isa<scf::WhileOp>(op)) {
        exit = summaries.front();
        children[0].finish(false);
        children[1].finish(true, incoming);
      } else if (isa<IfOp, IndexSwitchOp>(op) && llvm::all_equal(summaries)) {
        exit = summaries.front();
        for (auto& child : children) {
          child.finish(false);
        }
      } else {
        for (auto& child : children) {
          child.finish(true, incoming);
        }
        exit = incoming;
      }
      rewriter_.setInsertionPoint(op);
      const auto physicalPort = [&](Port port) {
        auto input = inputs[port.first];
        if (port.second == -1) {
          return layout_.getHardwareIndex(scalars_.lookup(input));
        }
        return slot(owners_.lookup(input), rewriter_.getIndexAttr(port.second));
      };
      // Include incoming entries that the body cancelled back to identity.
      auto destinations = exit;
      for (const auto& [from, unused] : incoming) {
        destinations.try_emplace(from, from);
      }
      SmallVector<std::pair<Resource, Resource>> targets;
      for (const auto& [from, to] : destinations) {
        const auto logical = from.second == -1
                                 ? scalars_.lookup(inputs[from.first])
                                 : physicalPort(from);
        targets.emplace_back(logical, physicalPort(to));
      }
      for (const auto [from, to] : targets) {
        exchange(layout_.getHardwareIndex(from), to);
      }
    }
    size_t scalar = 0;
    for (auto [input, output] :
         llvm::zip(inputs, quantumValues(op->getResults()))) {
      if (isa<QubitType>(output.getType())) {
        scalars_[output] = scalars_.lookup(input);
        wires_[scalarInputs[scalar++]].value = output;
      } else {
        const auto owner = owners_.lookup(input);
        owners_[output] = owner;
        tensors_[owner].value = output;
      }
    }
  }

  void visit(Operation* op) {
    const bool handled =
        TypeSwitch<Operation*, bool>(op)
            .Case([&](qtensor::AllocOp alloc) {
              addTensor(alloc.getResult());
              return true;
            })
            .Case([&](qtensor::ExtractOp extract) {
              visitExtract(extract);
              return true;
            })
            .Case([&](qtensor::InsertOp insert) {
              visitInsert(insert);
              return true;
            })
            .Case([&](qtensor::FromElementsOp elements) {
              visitFromElements(elements);
              return true;
            })
            .Case<IfOp, IndexSwitchOp, scf::ForOp, scf::WhileOp>([&](auto) {
              if ((effects_.lookup(op) & CAPTURES) != 0) {
                return false;
              }
              visitRegions(op);
              return true;
            })
            .Default(false);
    if (handled) {
      return;
    }
    // Other regions, including modifier bodies, keep their wiring.
    if (op->getNumRegions() != 0) {
      fenceCaptures(op);
    }
    const auto inputs = quantumValues(op->getOperands());
    const auto outputs = quantumValues(op->getResults());
    if (auto swap = dyn_cast<SWAPOp>(op)) {
      const auto a = scalars_.lookup(inputs[0]);
      const auto b = scalars_.lookup(inputs[1]);
      if (!wires_[a].fixed && !wires_[b].fixed &&
          firstEligibleSwap_ == nullptr) {
        scalars_[outputs[0]] = a;
        scalars_[outputs[1]] = b;
        exchange(layout_.getHardwareIndex(a), layout_.getHardwareIndex(b));
        erased_.push_back(swap);
        return;
      }
    }
    // Gates act on whichever resource holds their logical state. Terminal
    // observations and disposal may too; the output permutation records it.
    if (!isa<UnitaryOpInterface, ResetOp>(op) &&
        !(terminal_ && isa<MeasureOp, SinkOp, qtensor::DeallocOp>(op))) {
      for (auto input : inputs) {
        fence(input);
      }
    }
    for (auto& operand : op->getOpOperands()) {
      if (isLinearQubitType(operand.get().getType())) {
        operand.set(current(operand.get()));
      }
    }
    // Valid QCO continues gate and call operands positionally.
    if (isa<UnitaryOpInterface, ResetOp, MeasureOp, func::CallOp>(op) &&
        inputs.size() == outputs.size()) {
      for (auto [input, output] : llvm::zip_equal(inputs, outputs)) {
        if (isa<QubitType>(output.getType())) {
          const auto logical = scalars_.lookup(input);
          scalars_[output] = logical;
          wires_[layout_.getHardwareIndex(logical)].value = output;
        } else {
          const auto owner = owners_.lookup(input);
          owners_[output] = owner;
          tensors_[owner].value = output;
        }
      }
      return;
    }
    // Unknown outputs may refer to fixed physical resources.
    for (auto output : outputs) {
      if (isa<QubitType>(output.getType())) {
        const auto id = addWire(output, {}, !isa<AllocOp>(op));
        if (auto sources = op->getAttrOfType<DenseI64ArrayAttr>(
                mqt::kSourceQubitIndicesAttr);
            sources && sources.size() == 1) {
          wires_[id].source = sources[0];
        }
      } else {
        addTensor(output, {}, true);
      }
    }
  }

public:
  PermutationTracker(IRRewriter& rewriter, Block& block,
                     const RegionEffects& effects,
                     Operation* firstEligibleSwap = nullptr,
                     bool trackOutput = false)
      : rewriter_(rewriter), block_(block), effects_(effects),
        firstEligibleSwap_(firstEligibleSwap),
        arguments_(quantumValues(block.getArguments())) {
    for (auto [index, argument] : llvm::enumerate(arguments_)) {
      if (isa<QubitType>(argument.getType())) {
        addWire(argument, Port{index, -1});
      } else {
        addTensor(argument, index);
      }
    }
    if (trackOutput) {
      for (auto& op : llvm::reverse(block)) {
        if (!isTerminalBookkeeping(&op)) {
          break;
        }
        terminalStart_ = &op;
      }
    }
  }

  void run() {
    for (auto& op : llvm::make_early_inc_range(block_.without_terminator())) {
      rewriter_.setInsertionPoint(&op);
      if (&op == firstEligibleSwap_) {
        // The suffix has no regions; earlier children cannot reach this SWAP.
        firstEligibleSwap_ = nullptr;
      }
      if (&op == terminalStart_) {
        enterTerminal();
      }
      visit(&op);
    }
  }

  void finishFunction(SmallVectorImpl<int64_t>* output) {
    if (output != nullptr && terminal_) {
      for (const auto id : dirty_) {
        if (layout_.getHardwareIndex(id) == id) {
          continue;
        }
        // The verifier bounds source labels by the module's source count.
        const auto source = *wires_[id].source;
        assert(std::cmp_less(source, output->size()) &&
               "source label exceeds the source count");
        (*output)[source] = *wires_[layout_.getHardwareIndex(id)].source;
      }
      finish(false);
    } else {
      finish(true);
    }
  }
};

} // namespace

/// Compose an absorbed output permutation into the circuit-wire layout.
///
/// Entry `j` of @p permutation is the source wire that holds output `j`.
static LogicalResult recordOutputPermutation(ModuleOp moduleOp,
                                             ArrayRef<int64_t> permutation) {
  auto layout =
      mqt::getCircuitLayout(moduleOp, static_cast<int64_t>(permutation.size()));
  if (failed(layout)) {
    return failure();
  }
  std::vector<int64_t> routing(permutation.size());
  for (auto [wire, destination] : llvm::enumerate(routing)) {
    destination =
        permutation[layout->routing ? (*layout->routing)[wire] : wire];
  }
  layout->routing = std::move(routing);
  moduleOp->setAttr("mqt.layout", layout->toAttr(moduleOp.getContext()));
  return success();
}

/// Elide SWAPs, or only terminal SWAPs of prepared programs.
static LogicalResult elidePermutations(ModuleOp moduleOp, bool terminalOnly) {
  auto entry = mqt::getEntryPoint(moduleOp);
  auto count = moduleOp->getAttrOfType<IntegerAttr>(mqt::kSourceQubitCountAttr);
  Operation* firstEligibleSwap = nullptr;
  if (terminalOnly) {
    // Only a recorded output permutation can absorb terminal SWAPs.
    firstEligibleSwap = count ? firstTerminalSwap(entry) : nullptr;
    if (firstEligibleSwap == nullptr) {
      return success();
    }
  } else if (!moduleOp.walk([](SWAPOp) { return WalkResult::interrupt(); })
                  .wasInterrupted()) {
    return success();
  }
  // Valid QCO continues region and call operands positionally, so a region
  // needs special handling only when it contains a SWAP, observes its inputs,
  // or captures quantum values that no region argument carries.
  RegionEffects effects;
  moduleOp.walk([&](Operation* op) {
    for (auto input : op->getOperands()) {
      if (!isLinearQubitType(input.getType())) {
        continue;
      }
      auto* scope = input.getParentRegion()->getParentOp();
      for (auto* parent = op->getParentOp(); parent != scope;
           parent = parent->getParentOp()) {
        effects[parent] |= CAPTURES;
      }
    }
    const unsigned flags = isa<SWAPOp>(op)                    ? HAS_SWAP
                           : isa<MeasureOp, func::CallOp>(op) ? OBSERVES
                                                              : 0;
    for (auto* parent = op; flags != 0 && parent != nullptr;
         parent = parent->getParentOp()) {
      auto& known = effects[parent];
      if ((known & flags) == flags) {
        break;
      }
      known |= flags;
    }
  });
  SmallVector<int64_t> output;
  IRRewriter rewriter(moduleOp.getContext());
  for (auto function : moduleOp.getOps<func::FuncOp>()) {
    if (!function.getBody().hasOneBlock() ||
        (effects.lookup(function) & HAS_SWAP) == 0 ||
        (terminalOnly && function != entry)) {
      continue;
    }
    const bool track = count && function == entry;
    if (track) {
      output.resize(static_cast<size_t>(count.getInt()));
      std::iota(output.begin(), output.end(), int64_t{0});
    }
    PermutationTracker tracker(rewriter, function.getBody().front(), effects,
                               firstEligibleSwap, track);
    tracker.run();
    tracker.finishFunction(track ? &output : nullptr);
  }
  if (llvm::any_of(llvm::enumerate(output), [](auto entry) {
        return static_cast<int64_t>(entry.index()) != entry.value();
      })) {
    return recordOutputPermutation(moduleOp, output);
  }
  return success();
}

namespace {

struct ElidePermutations final
    : impl::ElidePermutationsBase<ElidePermutations> {
protected:
  void runOnOperation() override {
    if (failed(elidePermutations(getOperation(), /*terminalOnly=*/false))) {
      signalPassFailure();
    }
  }
};

struct ElideTerminalSwapsPass final
    : impl::ElideTerminalSwapsPassBase<ElideTerminalSwapsPass> {
protected:
  void runOnOperation() override {
    if (failed(elidePermutations(getOperation(), /*terminalOnly=*/true))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::qco
