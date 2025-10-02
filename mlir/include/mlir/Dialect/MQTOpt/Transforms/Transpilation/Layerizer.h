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

using LayerizerResult = SmallVector<QubitIndexPair>;

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
    return {{layout.lookupProgram(in0), layout.lookupProgram(in1)}};
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
    DenseSet<UnitaryInterface> candidates;

    for (const Value in : copy.getHardwareQubits()) {
      Value out = in;

      while (!out.getUsers().empty()) {
        Operation* user = *out.getUsers().begin();

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
        gates.push_back({copy.lookupProgram(in0), copy.lookupProgram(in1)});
      }
    }

    return gates;
  }
};
} // namespace mqt::ir::opt
