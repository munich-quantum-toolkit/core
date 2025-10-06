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
#include "mlir/IR/Value.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Support/LLVM.h>
#include <utility>

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
  layerize([[maybe_unused]] UnitaryInterface op,
           const Layout<QubitIndex>& layout) const override {
    Layout<QubitIndex> copy(layout);
    Layers layers(1 + nlookahead);

    for (Layer& layer : layers) {
      mlir::DenseSet<UnitaryInterface> visited;
      mlir::SmallVector<UnitaryInterface> gates;

      for (const mlir::Value curr : copy.getHardwareQubits()) {
        mlir::Value next = curr;
        while (!next.getUsers().empty()) {

          mlir::Operation* user = *next.getUsers().begin();
          const bool stop =
              mlir::TypeSwitch<mlir::Operation*, bool>(user)
                  .Case<mlir::scf::ForOp>([&](auto op) {
                    next = op->getResult(copy.lookupHardwareIndex(next));
                    return false;
                  })
                  .Case<mlir::scf::IfOp>([&](auto op) {
                    next = op->getResult(copy.lookupHardwareIndex(next));
                    return false;
                  })
                  .Case<ResetOp>([&](auto op) {
                    next = op.getOutQubit();
                    return false;
                  })
                  .Case<MeasureOp>([&](auto op) {
                    next = op.getOutQubit();
                    return false;
                  })
                  .Case<UnitaryInterface>([&](auto op) {
                    if (mlir::isa<GPhaseOp>(op)) {
                      return true;
                    }

                    if (isTwoQubitGate(op)) {
                      /// If we visit a two-qubit gate twice, it is "ready".
                      if (visited.contains(op)) {
                        gates.emplace_back(op);
                        return true;
                      }

                      visited.insert(op);
                      return true;
                    }

                    next = op.getOutQubits().front();
                    return false;
                  })
                  .Default([&](mlir::Operation*) { return true; });

          if (stop) {
            if (next != curr) {
              copy.remapQubitValue(curr, next);
            }
            break;
          }
        }
      }

      for (const UnitaryInterface op : gates) {
        const auto [in0, in1] = getIns(op);
        const auto [out0, out1] = getOuts(op);
        layer.emplace_back(copy.lookupProgramIndex(in0),
                           copy.lookupProgramIndex(in1));
        copy.remapQubitValue(in0, out0);
        copy.remapQubitValue(in1, out1);
      }
    }

    return layers;
  }
};
} // namespace mqt::ir::opt
