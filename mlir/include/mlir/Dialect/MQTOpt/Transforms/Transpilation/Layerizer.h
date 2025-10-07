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
 * @brief A layerizer divides the circuit into routable sections.
 */
struct LayerizerBase {
  virtual ~LayerizerBase() = default;
  [[nodiscard]] virtual Layers
  layerize(UnitaryInterface op, const Layout<QubitIndex>& layout) const = 0;
};

/**
 * @brief A one-op layerizer simply returns the given op.
 */
struct OneOpLayerizer final : LayerizerBase {
  [[nodiscard]] Layers
  layerize(UnitaryInterface op,
           const Layout<QubitIndex>& layout) const override {
    const auto [in0, in1] = getIns(op);
    return {{{layout.lookupProgramIndex(in0), layout.lookupProgramIndex(in1)}}};
  }
};

/**
 * @brief A crawl layerizer "crawls" the DAG for all gates that can be executed
 * in parallel, i.e., they don't depend each others results.
 */
struct CrawlLayerizer final : LayerizerBase {
  std::size_t nlookahead = 1;

  [[nodiscard]] Layers
  layerize(UnitaryInterface op,
           const Layout<QubitIndex>& layout) const override {
    Layout<QubitIndex> copy(layout);
    Layers layers(1 + nlookahead);

    const mlir::Region* region = op->getParentRegion();
    const mlir::ArrayRef<mlir::Value> qubits = copy.getHardwareQubits();
    const std::size_t nqubits = qubits.size();

    for (Layer& layer : layers) {
      mlir::DenseSet<UnitaryInterface> seenTwoQubit;
      mlir::SmallVector<UnitaryInterface> readyToQubit;

      /// The maximum amount of two-qubit gates in a layer is nqubits / 2.
      /// Assuming sparsity we half this value: nqubits / (2 * 2)
      seenTwoQubit.reserve(nqubits / 4);
      readyToQubit.reserve(nqubits / 4);

      for (const mlir::Value q : qubits) {
        bool stop = false;
        mlir::Value prev = q;
        mlir::Value head = q;

        while (!head.use_empty() && !stop) {
          mlir::Operation* user = getUserInRegion(head, region);
          if (user == nullptr) {
            break;
          }

          mlir::TypeSwitch<mlir::Operation*>(user)
              .Case<mlir::scf::ForOp>([&](mlir::scf::ForOp op) {
                head = op->getResult(copy.lookupHardwareIndex(head));
              })
              .Case<mlir::scf::IfOp>([&](mlir::scf::IfOp op) {
                head = op->getResult(copy.lookupHardwareIndex(head));
              })
              .Case<ResetOp>([&](ResetOp op) { head = op.getOutQubit(); })
              .Case<MeasureOp>([&](MeasureOp op) { head = op.getOutQubit(); })
              .Case<UnitaryInterface>([&](UnitaryInterface op) {
                if (mlir::isa<GPhaseOp>(op)) {
                  stop = true;
                  return;
                }

                /// Insert two-qubit gates into seen-set.
                /// If this is the second encounter, the gate is ready.
                if (isTwoQubitGate(op)) {
                  if (!seenTwoQubit.insert(op).second) {
                    readyToQubit.emplace_back(op);
                  }
                  stop = true;
                  return;
                }

                head = op.getOutQubits().front();
              })
              .Default([&](mlir::Operation*) { stop = true; });

          if (prev != head) {
            copy.remapQubitValue(prev, head);
            prev = head;
          }
        }
      }

      for (const UnitaryInterface op : readyToQubit) {
        const auto [in0, in1] = getIns(op);
        const auto [out0, out1] = getOuts(op);
        layer.emplace_back(copy.lookupProgramIndex(in0),
                           copy.lookupProgramIndex(in1));
        copy.remapQubitValue(in0, out0);
        copy.remapQubitValue(in1, out1);
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "crawl layerizer: layers=\n";
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
  static mlir::Operation* getUserInRegion(const mlir::Value v,
                                          const mlir::Region* region) {
    for (mlir::Operation* user : v.getUsers()) {
      if (user->getParentRegion() == region) {
        return user;
      }
    }
    return nullptr;
  }
};
} // namespace mqt::ir::opt
