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

#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/IR/WireIterator.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Common.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Layout.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Unit.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Compiler.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Region.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <optional>
#include <utility>

namespace mqt::ir::opt {
namespace {

/// @brief A wire links a WireIterator to a program index.
struct Wire {
  Wire(const WireIterator& it, uint32_t index) : it(it), index(index) {}
  WireIterator it;
  uint32_t index;
};

/// @brief Map to handle multi-qubit gates when traversing the def-use chain.
class SynchronizationMap {
public:
  /// @returns true iff. the operation is contained in the map.
  bool contains(mlir::Operation* op) const { return onHold.contains(op); }

  /// @brief Add op with respective wire and ref count to map.
  void add(mlir::Operation* op, Wire wire, const std::size_t cnt) {
    onHold.try_emplace(op, mlir::SmallVector<Wire>{wire});
    /// Decrease the cnt by one because the op was visited when adding.
    refCount.try_emplace(op, cnt - 1);
  }

  /// @brief Decrement ref count of op and potentially release its iterators.
  std::optional<mlir::SmallVector<Wire, 0>> visit(mlir::Operation* op,
                                                  Wire wire) {
    assert(refCount.contains(op) && "expected sync map to contain op");

    /// Add iterator for later release.
    onHold[op].push_back(wire);

    /// Release iterators whenever the ref count reaches zero.
    if (--refCount[op] == 0) {
      return onHold[op];
    }

    return std::nullopt;
  }

private:
  /// @brief Maps operations to to-be-released iterators.
  mlir::DenseMap<mlir::Operation*, mlir::SmallVector<Wire, 0>> onHold;
  /// @brief Maps operations to ref counts.
  mlir::DenseMap<mlir::Operation*, std::size_t> refCount;
};
} // namespace

static mlir::SmallVector<Wire, 2> skipTwoQubitBlock(mlir::ArrayRef<Wire> wires,
                                                    Layer& opLayer) {
  assert(wires.size() == 2 && "expected two wires");

  auto [it0, index0] = wires[0];
  auto [it1, index1] = wires[1];
  while (it0 != std::default_sentinel && it1 != std::default_sentinel) {
    mlir::Operation* op0 = *it0;
    if (!mlir::isa<UnitaryInterface>(op0) || mlir::isa<BarrierOp>(op0)) {
      break;
    }

    mlir::Operation* op1 = *it1;
    if (!mlir::isa<UnitaryInterface>(op1) || mlir::isa<BarrierOp>(op1)) {
      break;
    }

    const UnitaryInterface u0 = cast<UnitaryInterface>(op0);

    /// Advance for single qubit gate on wire 0.
    if (!isTwoQubitGate(u0)) {
      opLayer.addOp(u0);
      ++it0;
      continue;
    }

    const UnitaryInterface u1 = cast<UnitaryInterface>(op1);

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

LayeredUnit LayeredUnit::fromEntryPointFunction(mlir::func::FuncOp func,
                                                const std::size_t nqubits) {
  Layout layout(nqubits);
  for_each(func.getOps<QubitOp>(), [&](QubitOp op) {
    layout.add(op.getIndex(), op.getIndex(), op.getQubit());
  });
  return {std::move(layout), &func.getBody()};
}

LayeredUnit::LayeredUnit(Layout layout, mlir::Region* region)
    : Unit(std::move(layout), region) {
  SynchronizationMap sync;

  mlir::SmallVector<Wire, 0> curr;
  mlir::SmallVector<Wire, 0> next;
  curr.reserve(layout_.getNumQubits());
  next.reserve(layout_.getNumQubits());

  for (const auto q : layout_.getHardwareQubits()) {
    /// Increment the iterator here to skip the defining operation.
    curr.emplace_back(++WireIterator(q, region_),
                      layout_.lookupProgramIndex(q));
  }

  while (true) {

    /// Advance each wire until (>=2)-qubit gates are found, collect the indices
    /// of the respective two-qubit gates, and prepare iterators for next
    /// iteration.

    Layer layer;

    bool haltOnWire{};

    for (const auto& wire : curr) {
      auto it = wire.it;
      const auto index = wire.index;
      while (it != std::default_sentinel) {
        haltOnWire =
            mlir::TypeSwitch<mlir::Operation*, bool>(*it)
                .Case<UnitaryInterface>([&](UnitaryInterface op) {
                  const auto nins = op.getInQubits().size() +
                                    op.getPosCtrlInQubits().size() +
                                    op.getNegCtrlInQubits().size();

                  /// Skip over one-qubit gates. Note: Might be a BarrierOp.
                  if (nins == 1) {
                    layer.addOp(op);
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
                    layer.addOp(op);

                    // Is ready two-qubit unitary?
                    if (!mlir::isa<BarrierOp>(op)) {
                      layer.twoQubitProgs.emplace_back((*iterators)[0].index,
                                                       (*iterators)[1].index);
                      next.append(skipTwoQubitBlock(*iterators, layer));
                    } else {
                      next.append(*iterators);
                    }
                  }

                  return true;
                })
                .Case<ResetOp, MeasureOp>([&](auto op) {
                  layer.addOp(op);
                  ++it;
                  return false;
                })
                .Case<mlir::scf::YieldOp>([&](mlir::scf::YieldOp yield) {
                  if (!sync.contains(yield)) {
                    sync.add(yield, Wire(++it, index), layout_.getNumQubits());
                    return true;
                  }

                  if (const auto iterators =
                          sync.visit(yield, Wire(++it, index))) {
                    layer.addOp(yield);
                  }

                  return true;
                })
                .Case<mlir::RegionBranchOpInterface>(
                    [&](mlir::RegionBranchOpInterface op) {
                      if (!sync.contains(op)) {
                        sync.add(op, Wire(++it, index), layout_.getNumQubits());
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

    if (!layer.hasZeroOps()) {
      if (!layer.hasZero2QOps() || layers_.empty()) {
        layers_.emplace_back(layer);
      } else {
        /// If there is no gates to route, merge into the previous layer.
        layers_.back().ops.append(layer.ops);
        if (layers_.back().anchor == nullptr) {
          layers_.back().anchor = layer.anchor;
        }
      }
    }

    /// Prepare next iteration or stop.
    curr.swap(next);
    next.clear();

    if (curr.empty() || divider_ != nullptr) {
      break;
    }
  };
}

mlir::SmallVector<LayeredUnit, 3> LayeredUnit::nextImpl() {
  if (divider_ == nullptr) {
    return {};
  }

  mlir::SmallVector<LayeredUnit, 3> units;
  mlir::TypeSwitch<mlir::Operation*>(divider_)
      .Case<mlir::scf::ForOp>([&](mlir::scf::ForOp op) {
        Layout forLayout(layout_); // Copy layout.
        forLayout.remapToLoopBody(op);
        layout_.remapToLoopResults(op);
        units.emplace_back(std::move(layout_), region_);
        units.emplace_back(std::move(forLayout), &op.getRegion());
      })
      .Case<mlir::scf::IfOp>([&](mlir::scf::IfOp op) {
        units.emplace_back(layout_, &op.getThenRegion());
        units.emplace_back(layout_, &op.getElseRegion());
        layout_.remapIfResults(op);
        units.emplace_back(layout_, region_);
      })
      .Default([](auto) { llvm_unreachable("invalid 'next' operation"); });

  return units;
}

#ifndef NDEBUG
LLVM_DUMP_METHOD void LayeredUnit::dump(llvm::raw_ostream& os) const {
  os << "schedule: layers=\n";
  for (const auto [i, layer] : llvm::enumerate(layers_)) {
    os << '\t' << '[' << i << "]:\n";
    os << "\t #ops= " << layer.ops.size();
    if (!layer.ops.empty()) {
      os << " anchor= " << layer.anchor->getLoc();
    }
    os << '\n';
    os << "\t gates= ";
    if (!layer.hasZero2QOps()) {
      for (const auto [prog0, prog1] : layer.twoQubitProgs) {
        os << "(" << prog0 << "," << prog1 << "), ";
      }
    } else {
      os << "(), ";
    }
    os << '\n';
  }
  if (divider_ != nullptr) {
    os << "schedule: followUp= " << divider_->getLoc() << '\n';
  }
}
#endif
} // namespace mqt::ir::opt
