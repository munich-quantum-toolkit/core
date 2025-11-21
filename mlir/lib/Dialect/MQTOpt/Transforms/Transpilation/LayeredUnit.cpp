/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/LayeredUnit.h"

#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Common.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Layout.h"

#include <cstddef>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Compiler.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Support/LLVM.h>

namespace mqt::ir::opt {
namespace {

/// @brief Map to handle multi-qubit gates when traversing the def-use chain.
class SynchronizationMap {
  /// @brief Maps operations to to-be-released iterators.
  DenseMap<Operation*, SmallVector<Wire, 0>> onHold;

  /// @brief Maps operations to ref counts. An op can be released whenever the
  /// count reaches zero.
  DenseMap<Operation*, std::size_t> refCount;

public:
  /// @returns true iff. the operation is contained in the map.
  bool contains(Operation* op) const { return onHold.contains(op); }

  /// @brief Add op with respective wire and ref count to map.
  void add(Operation* op, Wire wire, const std::size_t cnt) {
    onHold.try_emplace(op, SmallVector<Wire>{wire});
    /// Decrease the cnt by one because the op was visited when adding.
    refCount.try_emplace(op, cnt - 1);
  }

  /// @brief Decrement ref count of op and potentially release its iterators.
  std::optional<SmallVector<Wire, 0>> visit(Operation* op, Wire wire) {
    assert(refCount.contains(op) && "expected sync map to contain op");

    /// Add iterator for later release.
    onHold[op].push_back(wire);

    /// Release iterators whenever the ref count reaches zero.
    if (--refCount[op] == 0) {
      return onHold[op];
    }

    return std::nullopt;
  }
};

SmallVector<Wire, 2> skipTwoQubitBlock(ArrayRef<Wire> wires, OpLayer& opLayer) {
  assert(wires.size() == 2 && "expected two wires");

  WireIterator end;
  auto [it0, index0] = wires[0];
  auto [it1, index1] = wires[1];
  while (it0 != end && it1 != end) {
    Operation* op0 = *it0;
    if (!isa<UnitaryInterface>(op0) || isa<BarrierOp>(op0)) {
      break;
    }

    Operation* op1 = *it1;
    if (!isa<UnitaryInterface>(op1) || isa<BarrierOp>(op1)) {
      break;
    }

    UnitaryInterface u0 = cast<UnitaryInterface>(op0);

    /// Advance for single qubit gate on wire 0.
    if (!isTwoQubitGate(u0)) {
      opLayer.addOp(u0);
      ++it0;
      continue;
    }

    UnitaryInterface u1 = cast<UnitaryInterface>(op1);

    /// Advance for single qubit gate on wire 1.
    if (!isTwoQubitGate(u1)) {
      opLayer.addOp(u1);
      ++it1;
      continue;
    }

    /// Stop if the wires reach different two qubit gates.
    if (op0 != op1) {
      break;
    }

    /// Advance if u0 == u1.
    opLayer.addOp(u1);

    ++it0;
    ++it1;
  }

  return {Wire(it0, index0), Wire(it1, index1)};
}
} // namespace

SmallVector<LayeredUnit, 3> LayeredUnit::next() {
  if (divider_ == nullptr) {
    return {};
  }

  SmallVector<LayeredUnit, 3> units;
  TypeSwitch<Operation*>(divider_)
      .Case<scf::ForOp>([&](scf::ForOp op) {
        /// Copy layout.
        Layout forLayout(layout_);

        /// Forward out-of-loop and in-loop values.
        const auto nqubits = layout_.getNumQubits();
        const auto initArgs = op.getInitArgs().take_front(nqubits);
        const auto results = op.getResults().take_front(nqubits);
        const auto iterArgs = op.getRegionIterArgs().take_front(nqubits);
        for (const auto [arg, res, iter] :
             llvm::zip(initArgs, results, iterArgs)) {
          layout_.remapQubitValue(arg, res);
          forLayout.remapQubitValue(arg, iter);
        }

        units.emplace_back(std::move(layout_), region_, restore_);
        units.emplace_back(std::move(forLayout), &op.getRegion(), true);
      })
      .Case<scf::IfOp>([&](scf::IfOp op) {
        units.emplace_back(layout_, &op.getThenRegion(), true);
        units.emplace_back(layout_, &op.getElseRegion(), true);

        /// Forward results.
        const auto results =
            op->getResults().take_front(layout_.getNumQubits());
        for (const auto [in, out] :
             llvm::zip(layout_.getHardwareQubits(), results)) {
          layout_.remapQubitValue(in, out);
        }

        units.emplace_back(layout_, region_, restore_);
      })
      .Default(
          [](auto) { throw std::runtime_error("invalid 'next' operation"); });

  return units;
}

SlidingWindow LayeredUnit::slidingWindow(std::size_t nlookahead) const {
  return SlidingWindow(gateLayers, opLayers, nlookahead);
}

#ifndef NDEBUG
LLVM_DUMP_METHOD void LayeredUnit::dump(llvm::raw_ostream& os) const {
  os << "schedule: gate layers=\n";
  for (const auto [i, layer] : llvm::enumerate(gateLayers)) {
    os << '\t' << i << ": ";
    os << "gates= ";
    if (!layer.empty()) {
      for (const auto [prog0, prog1] : layer) {
        os << "(" << prog0 << "," << prog1 << "), ";
      }
    } else {
      os << "(), ";
    }
    os << '\n';
  }

  os << "schedule: op layers=\n";
  for (const auto [i, layer] : llvm::enumerate(opLayers)) {
    os << '\t' << i << ": ";
    os << "#ops= " << layer.ops.size();
    if (!layer.ops.empty()) {
      os << " anchor= " << layer.anchor->getLoc();
    }
    os << '\n';
  }
  if (divider_ != nullptr) {
    os << "schedule: followUp= " << divider_->getLoc() << '\n';
  }
}
#endif

void LayeredUnit::init(const Layout& layout, Region* region) {
  SynchronizationMap sync;

  SmallVector<Wire, 0> curr;
  SmallVector<Wire, 0> next;
  curr.reserve(layout.getNumQubits());
  next.reserve(layout.getNumQubits());

  for (const auto q : layout.getHardwareQubits()) {
    curr.emplace_back(WireIterator(q, region), layout.lookupProgramIndex(q));
  }

  while (true) {

    /// Advance each wire until (>=2)-qubit gates are found, collect the indices
    /// of the respective two-qubit gates, and prepare iterators for next
    /// iteration.

    GateLayer gateLayer;
    OpLayer opLayer;

    bool haltOnWire{};

    for (auto [it, index] : curr) {
      while (it != WireIterator::end()) {
        haltOnWire =
            TypeSwitch<Operation*, bool>(*it)
                .Case<UnitaryInterface>([&](UnitaryInterface op) {
                  const auto nins = op.getInQubits().size() +
                                    op.getPosCtrlInQubits().size() +
                                    op.getNegCtrlInQubits().size();

                  /// Skip over one-qubit gates. Note: Might be a BarrierOp.
                  if (nins == 1) {
                    opLayer.addOp(op);
                    ++it;
                    return false;
                  }

                  /// Otherwise, add it to the sync map.
                  if (!sync.contains(op)) {
                    sync.add(op, Wire(++it, index), nins);
                    return true;
                  }

                  if (const auto iterators =
                          sync.visit(op, Wire(++it, index))) {
                    opLayer.addOp(op);

                    if (!isa<BarrierOp>(op)) { // Is ready two-qubit unitary?
                      gateLayer.emplace_back((*iterators)[0].index,
                                             (*iterators)[1].index);
                      next.append(skipTwoQubitBlock(*iterators, opLayer));
                    } else {
                      next.append(*iterators);
                    }
                  }

                  return true;
                })
                .Case<ResetOp>([&](ResetOp op) {
                  opLayer.addOp(op);
                  ++it;
                  return false;
                })
                .Case<MeasureOp>([&](MeasureOp op) {
                  opLayer.addOp(op);
                  ++it;
                  return false;
                })
                .Case<scf::YieldOp>([&](scf::YieldOp yield) {
                  if (!sync.contains(yield)) {
                    sync.add(yield, Wire(++it, index), layout.getNumQubits());
                    return true;
                  }

                  if (const auto iterators =
                          sync.visit(yield, Wire(++it, index))) {
                    opLayer.addOp(yield);
                  }

                  return true;
                })
                .Case<RegionBranchOpInterface>([&](RegionBranchOpInterface op) {
                  if (!sync.contains(op)) {
                    sync.add(op, Wire(++it, index), layout.getNumQubits());
                    return true;
                  }

                  if (const auto iterators =
                          sync.visit(op, Wire(++it, index))) {
                    divider_ = op;
                  }
                  return true;
                })
                .Default([](auto) {
                  llvm_unreachable("unhandled operation");
                  return true;
                });

        if (haltOnWire) {
          break;
        }
      }

      if (divider_ != nullptr) {
        break;
      }
    }

    /// If there is no gates to route, merge the last layer with this one and
    /// keep the anchor the same. Otherwise, add the layer.

    if (gateLayer.empty()) {
      if (!opLayer.empty()) {
        if (opLayers.empty()) {
          gateLayers.emplace_back();
          opLayers.emplace_back();
        }
        opLayers.back().ops.append(opLayer.ops);
        if (opLayers.back().anchor == nullptr) {
          opLayers.back().anchor = opLayer.anchor;
        }
      }

    } else if (!opLayer.empty()) {
      gateLayers.emplace_back(gateLayer);
      opLayers.emplace_back(opLayer);
    }

    /// Prepare next iteration or stop.
    curr.swap(next);
    next.clear();

    if (curr.empty() || divider_ != nullptr) {
      break;
    }
  };
}
} // namespace mqt::ir::opt
