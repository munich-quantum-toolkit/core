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

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Support/LLVM.h>
#include <utility>

namespace mqt::ir::opt {

using LayerizerResult = mlir::SmallVector<QubitIndexPair>;

/**
 * @brief A layerizer divides the circuit into routable sections.
 */
struct LayerizerBase {
  virtual ~LayerizerBase() = default;
  [[nodiscard]] virtual LayerizerResult
  layerize(UnitaryInterface op, const Layout<QubitIndex>& layout) const = 0;
};

/**
 * @brief A one-op layerizer simply returns the given op.
 */
struct OneOpLayerizer final : LayerizerBase {
  [[nodiscard]] LayerizerResult
  layerize(UnitaryInterface op,
           const Layout<QubitIndex>& layout) const override {
    const auto [in0, in1] = getIns(op);
    return {{layout.lookupProgramIndex(in0), layout.lookupProgramIndex(in1)}};
  }
};

/**
 * @brief A crawl layerizer "crawls" the DAG for all gates that can be executed
 * in parallel, i.e., they don't depend each others results.
 */
struct CrawlLayerizer final : LayerizerBase {
  [[nodiscard]] LayerizerResult
  layerize([[maybe_unused]] UnitaryInterface op,
           const Layout<QubitIndex>& layout) const override {
    Layout<QubitIndex> copy(layout);
    mlir::DenseSet<UnitaryInterface> candidates;

    for (const mlir::Value in : copy.getHardwareQubits()) {

      mlir::Value curr = in;
      while (!curr.getUsers().empty()) {
        mlir::Operation* user = *curr.getUsers().begin();

        using SwitchResult = std::pair<mlir::Value, bool>;
        auto [next, stop] =
            mlir::TypeSwitch<mlir::Operation*, SwitchResult>(user)
                .Case<mlir::scf::ForOp>([&](auto op) {
                  const QubitIndex hw = copy.lookupHardwareIndex(curr);
                  return std::make_pair(op->getResult(hw), false);
                })
                .Case<mlir::scf::IfOp>([&](auto op) {
                  const QubitIndex hw = copy.lookupHardwareIndex(curr);
                  return std::make_pair(op->getResult(hw), false);
                })
                .Case<ResetOp>([&](auto op) {
                  return std::make_pair(op.getOutQubit(), false);
                })
                .Case<MeasureOp>([&](auto op) {
                  return std::make_pair(op.getOutQubit(), false);
                })
                .Case<UnitaryInterface>([&](auto op) {
                  if (isTwoQubitGate(op)) {
                    candidates.insert(op);
                    return std::make_pair(curr, true);
                  }

                  const bool isZeroOp = mlir::isa<GPhaseOp>(op);
                  return !isZeroOp
                             ? std::make_pair(op.getOutQubits().front(), false)
                             : std::make_pair(curr, true);
                })
                .Default([&](mlir::Operation*) {
                  return std::make_pair(curr, true);
                });

        if (stop) {
          break;
        }

        if (next != curr) {
          copy.remapQubitValue(curr, next);
          curr = next;
        }
      }
    }

    LayerizerResult gates;
    for (UnitaryInterface op : candidates) {
      const auto [in0, in1] = getIns(op);
      if (copy.contains(in0) && copy.contains(in1)) {
        gates.push_back(
            {copy.lookupProgramIndex(in0), copy.lookupProgramIndex(in1)});
      }
    }

    return gates;
  }
};
} // namespace mqt::ir::opt
