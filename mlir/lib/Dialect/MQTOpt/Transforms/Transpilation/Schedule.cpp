/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Schedule.h"

#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/WireIterator.h"

#include "llvm/ADT/ArrayRef.h"

#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>

namespace mqt::ir::opt {

namespace {
struct Wire {
  Wire(const WireIterator& it, QubitIndex index) : it(it), index(index) {}

  WireIterator it;
  QubitIndex index;
};

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

SmallVector<Wire, 2> skipTwoQubitBlock(ArrayRef<Wire> wires,
                                       Schedule::OpLayer& opLayer) {
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

/**
 * @brief Advance each wire until (>=2)-qubit gates are found, collect the
 * indices of the respective two-qubit gates, and prepare iterators for next
 * iteration.
 */
void collectLayerAndAdvance(ArrayRef<Wire> wires, SynchronizationMap& sync,
                            Schedule& s, const std::size_t nqubits,
                            SmallVector<Wire, 0>& next) {
  Schedule::GateLayer gateLayer;
  Schedule::OpLayer opLayer;

  for (auto [it, index] : wires) {
    while (it != WireIterator::end()) {
      const bool stop =
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

                if (const auto iterators = sync.visit(op, Wire(++it, index))) {
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
                  sync.add(yield, Wire(++it, index), nqubits);
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
                  sync.add(op, Wire(++it, index), nqubits);
                  return true;
                }

                if (const auto iterators = sync.visit(op, Wire(++it, index))) {
                  opLayer.addOp(op);
                  next.append(*iterators);
                }
                return true;
              })
              .Default([](auto) {
                llvm_unreachable("unhandled operation");
                return true;
              });

      if (stop) {
        break;
      }
    }
  }

  /// If there is no gates to route, merge the last layer with this one and
  /// keep the anchor the same. Otherwise, add the layer.
  if (gateLayer.empty()) {
    s.gateLayers.back().append(gateLayer);
    s.opLayers.back().ops.append(opLayer.ops);
  } else {
    s.gateLayers.emplace_back(gateLayer);
    s.opLayers.emplace_back(opLayer);
  }
}
} // namespace

MutableArrayRef<Schedule::GateLayer>
Schedule::getWindow(const std::size_t start, const std::size_t nlookahead) {
  const size_t sz = gateLayers.size();
  const size_t len = std::min(1 + nlookahead, sz - start);
  return MutableArrayRef<Schedule::GateLayer>(gateLayers).slice(start, len);
}

#ifndef NDEBUG
LLVM_DUMP_METHOD void Schedule::dump(llvm::raw_ostream& os) const {
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
}
#endif

Schedule getSchedule(const Layout& layout, Region& region) {
  Schedule s;
  SynchronizationMap sync;

  SmallVector<Wire, 0> curr;
  SmallVector<Wire, 0> next;

  const auto nqubits = layout.getNumQubits();
  curr.reserve(nqubits);
  next.reserve(nqubits);

  for (auto [hw, q] : llvm::enumerate(layout.getHardwareQubits())) {
    curr.emplace_back(WireIterator(q, &region), layout.getProgramIndex(hw));
  }

  while (!curr.empty()) {
    collectLayerAndAdvance(curr, sync, s, nqubits, next);
    curr.swap(next);
    next.clear();
  };

  LLVM_DEBUG(s.dump());

  return s;
}
} // namespace mqt::ir::opt
