/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Unit.h"

#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Common.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Router.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstddef>
#include <llvm/Support/Compiler.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Support/LLVM.h>

namespace mqt::ir::opt {
namespace {

/**
 * @brief Insert SWAP ops at the rewriter's insertion point.
 *
 * @param location The location of the inserted SWAP ops.
 * @param swaps The hardware indices of the SWAPs.
 * @param layout The current layout.
 * @param rewriter The pattern rewriter.
 */
void insertSWAPs(Location location, ArrayRef<QubitIndexPair> swaps,
                 Layout& layout, PatternRewriter& rewriter) {
  for (const auto [hw0, hw1] : swaps) {
    const Value in0 = layout.lookupHardwareValue(hw0);
    const Value in1 = layout.lookupHardwareValue(hw1);
    [[maybe_unused]] const auto [prog0, prog1] =
        layout.getProgramIndices(hw0, hw1);

    LLVM_DEBUG({
      llvm::dbgs() << llvm::format(
          "insertSWAPs: swap= p%d:h%d, p%d:h%d <- p%d:h%d, p%d:h%d\n", prog1,
          hw0, prog0, hw1, prog0, hw0, prog1, hw1);
    });

    auto swap = createSwap(location, in0, in1, rewriter);
    const auto [out0, out1] = getOuts(swap);

    rewriter.setInsertionPointAfter(swap);
    replaceAllUsesInRegionAndChildrenExcept(in0, out1, swap->getParentRegion(),
                                            swap, rewriter);
    replaceAllUsesInRegionAndChildrenExcept(in1, out0, swap->getParentRegion(),
                                            swap, rewriter);

    layout.swap(in0, in1);
    layout.remapQubitValue(in0, out0);
    layout.remapQubitValue(in1, out1);
  }
}

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

MutableArrayRef<GateLayer>
Schedule::window(const std::size_t start, const std::size_t nlookahead) const {
  const size_t sz = gateLayers.size();
  const size_t len = std::min(1 + nlookahead, sz - start);
  return MutableArrayRef<GateLayer>(gateLayers).slice(start, len);
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
  if (next != nullptr) {
    os << "schedule: followUp= " << next->getLoc() << '\n';
  }
}
#endif

void Unit::schedule() {
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
                    s.next = op;
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

      if (s.next != nullptr) {
        break;
      }
    }

    /// If there is no gates to route, merge the last layer with this one and
    /// keep the anchor the same. Otherwise, add the layer.

    if (gateLayer.empty()) {
      if (!opLayer.empty()) {
        if (s.opLayers.empty()) {
          s.gateLayers.emplace_back();
          s.opLayers.emplace_back();
        }
        s.opLayers.back().ops.append(opLayer.ops);
        if (s.opLayers.back().anchor == nullptr) {
          s.opLayers.back().anchor = opLayer.anchor;
        }
      }

    } else if (!opLayer.empty()) {
      s.gateLayers.emplace_back(gateLayer);
      s.opLayers.emplace_back(opLayer);
    }

    /// Prepare next iteration or stop.
    curr.swap(next);
    next.clear();

    if (curr.empty() || s.next != nullptr) {
      break;
    }
  };

  LLVM_DEBUG(s.dump());
}

void Unit::route(const AStarHeuristicRouter& router, std::size_t nlookahead,
                 const Architecture& arch, PatternRewriter& rewriter) {
  SmallVector<QubitIndexPair> history;
  for (std::size_t i = 0; i < s.gateLayers.size(); ++i) {
    const auto opLayer = s.opLayers[i];

    rewriter.setInsertionPoint(opLayer.anchor);
    const auto swaps = router.route(s.window(i, nlookahead), layout, arch);
    if (!swaps) {
      throw std::runtime_error("A* failed to find a valid SWAP sequence");
    }

    if (!swaps->empty()) {
      history.append(*swaps);
      insertSWAPs(opLayer.anchor->getLoc(), *swaps, layout, rewriter);
      /// *(ctx.stats.numSwaps) += swaps->size();
    }

    for (Operation* curr : opLayer.ops) {
      rewriter.setInsertionPoint(curr);

      /// Re-order to fix any SSA Dominance issues.
      if (i + 1 < s.gateLayers.size()) {
        rewriter.moveOpBefore(curr, s.opLayers[i + 1].anchor);
      }

      TypeSwitch<Operation*>(curr)
          .Case<UnitaryInterface>([&](UnitaryInterface op) {
            if (isa<SWAPOp>(op)) {
              const auto ins = getIns(op);
              layout.swap(ins.first, ins.second);
              history.push_back({layout.lookupHardwareIndex(ins.first),
                                 layout.lookupHardwareIndex(ins.second)});
            }
            remap(op, layout);
          })
          .Case<ResetOp>([&](ResetOp op) { remap(op, layout); })
          .Case<MeasureOp>([&](MeasureOp op) { remap(op, layout); })
          .Case<scf::YieldOp>([&](scf::YieldOp op) {
            if (restore) {
              rewriter.setInsertionPointAfter(op->getPrevNode());
              insertSWAPs(op.getLoc(), llvm::to_vector(llvm::reverse(history)),
                          layout, rewriter);
            }
          })
          .Default(
              [](auto) { llvm_unreachable("unhandled 'curr' operation"); });
    }
  }
}

SmallVector<Unit, 3> Unit::advance() {
  if (s.next == nullptr) {
    return {};
  }

  SmallVector<Unit, 3> next;

  TypeSwitch<Operation*>(s.next)
      .Case<scf::ForOp>([&](scf::ForOp op) {
        /// Copy layout.
        Layout forLayout(layout);

        /// Forward out-of-loop and in-loop values.
        const auto nqubits = layout.getNumQubits();
        const auto initArgs = op.getInitArgs().take_front(nqubits);
        const auto results = op.getResults().take_front(nqubits);
        const auto iterArgs = op.getRegionIterArgs().take_front(nqubits);
        for (const auto [arg, res, iter] :
             llvm::zip(initArgs, results, iterArgs)) {
          layout.remapQubitValue(arg, res);
          forLayout.remapQubitValue(arg, iter);
        }

        next.emplace_back(std::move(layout), region, restore);
        next.emplace_back(std::move(forLayout), &op.getRegion(), true);
      })
      .Case<scf::IfOp>([&](scf::IfOp op) {
        next.emplace_back(layout, &op.getThenRegion(), true);
        next.emplace_back(layout, &op.getElseRegion(), true);

        /// Forward results.
        const auto results = op->getResults().take_front(layout.getNumQubits());
        for (const auto [in, out] :
             llvm::zip(layout.getHardwareQubits(), results)) {
          layout.remapQubitValue(in, out);
        }

        next.emplace_back(layout, region, restore);
      })
      .Default(
          [](auto) { throw std::runtime_error("invalid 'next' operation"); });

  return next;
}

} // namespace mqt::ir::opt
