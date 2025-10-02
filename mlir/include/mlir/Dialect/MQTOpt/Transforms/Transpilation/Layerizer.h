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

#include <mlir/Support/LLVM.h>

namespace mqt::ir::opt {

using LayerizerResult = mlir::SmallVector<QubitIndexPair>;

/**
 * @brief A layerizer divides the circuit into routable sections.
 */
struct LayerizerBase {
  virtual ~LayerizerBase() = default;
  [[nodiscard]] virtual LayerizerResult
  layerize(UnitaryInterface op, const Layout<QubitIndex>& layout) = 0;
};

/**
 * @brief A one-op layerizer simply returns the given op.
 */
struct OneOpLayerizer : LayerizerBase {
  [[nodiscard]] LayerizerResult
  layerize(UnitaryInterface op, const Layout<QubitIndex>& layout) final {
    const auto [in0, in1] = getIns(op);
    return {{layout.lookupProgramIndex(in0), layout.lookupProgramIndex(in1)}};
  }
};

/**
 * @brief A crawl layerizer "crawls" the DAG for all gates that can be executed
 * in parallel, i.e., they don't depend each others results.
 */
struct CrawlLayerizer : LayerizerBase {
  [[nodiscard]] LayerizerResult
  layerize([[maybe_unused]] UnitaryInterface op,
           const Layout<QubitIndex>& layout) final {
    Layout<QubitIndex> copy(layout);
    mlir::DenseSet<UnitaryInterface> candidates;

    for (const mlir::Value in : copy.getHardwareQubits()) {
      mlir::Value out = in;

      while (!out.getUsers().empty()) {
        mlir::Operation* user = *out.getUsers().begin();

        if (auto op = dyn_cast<ResetOp>(user)) {
          out = op.getOutQubit();
          continue;
        }

        if (auto op = dyn_cast<UnitaryInterface>(user)) {
          if (isTwoQubitGate(op)) {
            candidates.insert(op);
            break;
          }

          if (!dyn_cast<GPhaseOp>(user)) {
            out = op.getOutQubits().front();
          }

          continue;
        }

        if (auto measure = dyn_cast<MeasureOp>(user)) {
          out = measure.getOutQubit();
          continue;
        }

        break;
      }

      if (in != out) {
        copy.remapQubitValue(in, out);
      }
    }

    llvm::SmallVector<QubitIndexPair> gates;
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
