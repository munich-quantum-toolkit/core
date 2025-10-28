/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Common.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Layout.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <array>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <optional>

#define DEBUG_TYPE "route-sc"

namespace mqt::ir::opt {

/**
 * @brief A vector of gates.
 */
using Layer = SmallVector<QubitIndexPair>;

/**
 * @brief A vector of layers.
 * [0]=current, [1]=lookahead (optional), >=2 future layers
 */
using Layers = SmallVector<Layer>;

/**
 * @brief A scheduler divides the circuit into routable sections.
 */
struct SchedulerBase {
  virtual ~SchedulerBase() = default;
  [[nodiscard]] virtual Layers schedule(UnitaryInterface op,
                                        Layout layout) const = 0;
};

/**
 * @brief A sequential scheduler simply returns the given op.
 */
struct SequentialOpScheduler final : SchedulerBase {
  [[nodiscard]] Layers schedule(UnitaryInterface op,
                                Layout layout) const override {
    const auto [in0, in1] = getIns(op);
    return {{{layout.lookupProgramIndex(in0), layout.lookupProgramIndex(in1)}}};
  }
};

/**
 * @brief A crawl scheduler "crawls" the DAG for all gates that can be executed
 * in parallel, i.e., they act on different qubits.
 */
struct ParallelOpScheduler final : SchedulerBase {
  explicit ParallelOpScheduler(const std::size_t nlookahead)
      : nlayers_(1 + nlookahead) {}

  [[nodiscard]] Layers schedule(UnitaryInterface op,
                                Layout layout) const override {
    Layers layers;
    layers.reserve(nlayers_);

    /// Worklist of active qubits.
    SmallVector<Value> wl;
    SmallVector<Value> nextWl;
    wl.reserve(layout.getHardwareQubits().size());
    nextWl.reserve(layout.getHardwareQubits().size());

    // Initialize worklist.
    llvm::copy_if(layout.getHardwareQubits(), std::back_inserter(wl),
                  [](Value q) { return !q.use_empty(); });

    /// Set of two-qubit gates seen at least once.
    llvm::SmallDenseSet<Operation*, 32> openTwoQubit;

    /// Vector of two-qubit gates seen twice.
    SmallVector<Operation*, 32> readyTwoQubit;

    while (!wl.empty() && layers.size() < nlayers_) {
      for (const Value q : wl) {
        const auto opt =
            advanceDefUseUntilTwoQubitGate(q, op->getParentRegion(), layout);

        if (!opt) {
          continue;
        }

        const auto& [qNext, gate] = opt.value();

        if (q != qNext) {
          layout.remapQubitValue(q, qNext);
        }

        /// If we've seen a two-qubit gate twice, we move it from "open" to
        /// "ready".
        if (!openTwoQubit.insert(gate).second) {
          readyTwoQubit.push_back(gate);
          openTwoQubit.erase(gate);
          continue;
        }
      }

      if (readyTwoQubit.empty()) {
        break;
      }

      nextWl.clear();
      layers.emplace_back();
      layers.back().reserve(readyTwoQubit.size());

      for (const auto& op : readyTwoQubit) {
        const UnitaryInterface u = dyn_cast<UnitaryInterface>(op);
        const auto [in0, in1] = getIns(u);
        const auto [out0, out1] = getOuts(u);

        layout.remapQubitValue(in0, out0);
        layout.remapQubitValue(in1, out1);

        layers.back().emplace_back(layout.lookupProgramIndex(out0),
                                   layout.lookupProgramIndex(out1));

        if (!out0.use_empty()) {
          nextWl.push_back(out0);
        }

        if (!out1.use_empty()) {
          nextWl.push_back(out1);
        }
      }

      /// Prepare for next iteration.
      readyTwoQubit.clear();
      wl = std::move(nextWl);
    }

    LLVM_DEBUG({
      llvm::dbgs() << "schedule: layers=\n";
      for (const auto [i, layer] : llvm::enumerate(layers)) {
        llvm::dbgs() << '\t' << i << "= ";
        for (const auto [prog0, prog1] : layer) {
          llvm::dbgs() << "(" << prog0 << "," << prog1 << "), ";
        }
        llvm::dbgs() << '\n';
      }
    });

    return layers;
  }

private:
  static std::optional<std::pair<Value, Operation*>>
  advanceDefUseUntilTwoQubitGate(const Value q, const Region* region,
                                 const Layout& layout) {
    Value head = q;
    Operation* twoQubitOp = nullptr;
    while (true) {
      if (head.use_empty()) { // No two-qubit gate found.
        return std::nullopt;
      }

      if (twoQubitOp != nullptr) { // Two-qubit gate found.
        return std::make_pair(head, twoQubitOp);
      }

      Operation* user = getUserInRegion(head, region);
      if (user == nullptr) { // No two-qubit gate found.
        return std::nullopt;
      }

      bool endOfRegion = false;
      TypeSwitch<Operation*>(user)
          /// MQT
          /// BarrierOp is a UnitaryInterface, however, requires special care.
          .Case<BarrierOp>([&](BarrierOp op) {
            for (const auto [in, out] :
                 llvm::zip_equal(op.getInQubits(), op.getOutQubits())) {
              if (in == head) {
                head = out;
                return;
              }
            }
            llvm_unreachable("head must be in barrier");
          })
          .Case<UnitaryInterface>([&](UnitaryInterface op) {
            if (isTwoQubitGate(op)) {
              twoQubitOp = op;
              return;
            }

            head = op.getOutQubits().front();
          })
          .Case<ResetOp>([&](ResetOp op) { head = op.getOutQubit(); })
          .Case<MeasureOp>([&](MeasureOp op) { head = op.getOutQubit(); })
          /// SCF
          /// The scf functions assume that the first n results are the
          /// hardw. qubits. We can use 'q' to get the hardware index
          /// because the def-use chain keeps the index constant.
          .Case<scf::ForOp>([&](scf::ForOp op) {
            head = op->getResult(layout.lookupHardwareIndex(q));
          })
          .Case<scf::IfOp>([&](scf::IfOp op) {
            head = op->getResult(layout.lookupHardwareIndex(q));
          })
          .Case<scf::YieldOp>([&](scf::YieldOp) { endOfRegion = true; })
          .Default([&](Operation* op) {
            LLVM_DEBUG({
              llvm::dbgs() << "unknown operation in def-use chain: ";
              op->dump();
            });
            llvm_unreachable("unknown operation in def-use chain");
          });

      if (endOfRegion) {
        return std::nullopt;
      }
    }

    return std::nullopt;
  }

  static Operation* getUserInRegion(const Value v, const Region* region) {
    for (Operation* user : v.getUsers()) {
      if (user->getParentRegion() == region) {
        return user;
      }
    }
    return nullptr;
  }

  std::size_t nlayers_;
};
} // namespace mqt::ir::opt
