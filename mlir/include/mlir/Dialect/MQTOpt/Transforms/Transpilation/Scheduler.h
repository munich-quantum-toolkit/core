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

#include "llvm/Support/Casting.h"

#include <array>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#define DEBUG_TYPE "route-sc"

namespace mqt::ir::opt {

/**
 * @brief A vector of gates.
 */
using Layer = mlir::SmallVector<QubitIndexPair>;

/**
 * @brief A vector of layers.
 * [0]=current, [1]=lookahead (optional), >=2 future layers
 */
using Layers = mlir::SmallVector<Layer>;

/**
 * @brief A scheduler divides the circuit into routable sections.
 */
struct SchedulerBase {
  virtual ~SchedulerBase() = default;
  [[nodiscard]] virtual Layers schedule(UnitaryInterface op,
                                        const Layout& layout) const = 0;
};

/**
 * @brief A sequential scheduler simply returns the given op.
 */
struct SequentialOpScheduler final : SchedulerBase {
  [[nodiscard]] Layers schedule(UnitaryInterface op,
                                const Layout& layout) const override {
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
                                const Layout& layout) const override {
    Layers layers(nlayers_);
    Layout layoutCopy(layout);

    const auto* region = op->getParentRegion();

    for (Layer& layer : layers) {
      mlir::DenseSet<mlir::Operation*> seenTwoQubit;
      mlir::SmallVector<UnitaryInterface> readyTwoQubit;

      const auto qubits = layoutCopy.getHardwareQubits();
      const auto nqubits = qubits.size();

      /// The maximum amount of two-qubit gates in a layer is nqubits / 2.
      /// Assuming sparsity we half this value: nqubits / (2 * 2)
      seenTwoQubit.reserve(nqubits / 4);
      readyTwoQubit.reserve(nqubits / 4);

      for (const mlir::Value q : qubits) {
        bool stop = false;
        mlir::Value prev = q;
        mlir::Value head = q;

        while (!head.use_empty() && !stop) {
          mlir::Operation* user = getUserInRegion(head, region);
          if (user == nullptr) {
            break;
          }

          /// TypeSwitch performs sequential dyn_cast checks.
          /// Hence, always put most frequent ops first.

          mlir::TypeSwitch<mlir::Operation*, void>(user)
              .Case<UnitaryInterface>([&](UnitaryInterface op) {
                if (mlir::isa<BarrierOp>(op)) {
                  for (const auto [in, out] :
                       llvm::zip_equal(op.getInQubits(), op.getOutQubits())) {
                    if (in == head) {
                      head = out;
                      break;
                    }
                  }
                  return;
                }

                /// Insert two-qubit gates into seen-set.
                /// If this is the second encounter, the gate is ready.
                if (isTwoQubitGate(op)) {
                  if (!seenTwoQubit.insert(op.getOperation()).second) {
                    readyTwoQubit.emplace_back(op);
                  }
                  stop = true;
                  return;
                }

                head = op.getOutQubits().front();
              })
              .Case<ResetOp>([&](ResetOp op) { head = op.getOutQubit(); })
              .Case<MeasureOp>([&](MeasureOp op) { head = op.getOutQubit(); })
              .Case<mlir::scf::ForOp>([&](mlir::scf::ForOp op) {
                /// This assumes that the first n results are the hardw. qubits.
                head = op->getResult(layoutCopy.lookupHardwareIndex(head));
              })
              .Case<mlir::scf::IfOp>([&](mlir::scf::IfOp op) {
                /// This assumes that the first n results are the hardw. qubits.
                head = op->getResult(layoutCopy.lookupHardwareIndex(head));
              })
              .Default([&](mlir::Operation*) { stop = true; });

          if (prev != head) {
            layoutCopy.remapQubitValue(prev, head);
            prev = head;
          }
        }
      }

      if (readyTwoQubit.empty()) {
        break;
      }

      for (const UnitaryInterface op : readyTwoQubit) {
        const auto [in0, in1] = getIns(op);
        const auto [out0, out1] = getOuts(op);
        layoutCopy.remapQubitValue(in0, out0);
        layoutCopy.remapQubitValue(in1, out1);

        layer.emplace_back(layoutCopy.lookupProgramIndex(out0),
                           layoutCopy.lookupProgramIndex(out1));

        // skipTwoQubitBlock(out0, out1, layoutCopy, region);
      }
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
  static void skipTwoQubitBlock(const Value q0, const Value q1, Layout& layout,
                                const Region* region) {

    std::array<Value, 2> current{q0, q1};
    while (true) {
      std::array<UnitaryInterface, 2> gates{};

      for (const auto [i, tail] : llvm::enumerate(current)) {
        Value head = tail;

        while (!head.use_empty()) {
          mlir::Operation* user = getUserInRegion(head, region);
          if (user == nullptr) {
            break;
          }

          auto gate = mlir::dyn_cast<UnitaryInterface>(user);
          if (!gate) {
            break;
          }

          if (llvm::isa<BarrierOp>(gate)) {
            break;
          }

          if (isTwoQubitGate(gate)) {
            gates[i] = gate;
            break;
          }

          head = gate.getOutQubits().front();
        }

        if (tail != head) {
          layout.remapQubitValue(tail, head);
        }
      }

      if (gates[0] == nullptr || gates[1] == nullptr) {
        break;
      }

      if (gates[0] != gates[1]) {
        break;
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "skipTwoQubitBlock: skip= " << gates[0]->getName() << '\n');
      const auto [in0, in1] = getIns(gates[0]);
      const auto [out0, out1] = getOuts(gates[0]);
      layout.remapQubitValue(in0, out0);
      layout.remapQubitValue(in1, out1);
      current = {out0, out1};
    }
  }

  static mlir::Operation* getUserInRegion(const mlir::Value v,
                                          const mlir::Region* region) {
    for (mlir::Operation* user : v.getUsers()) {
      if (user->getParentRegion() == region) {
        return user;
      }
    }
    return nullptr;
  }

  std::size_t nlayers_;
};
} // namespace mqt::ir::opt
