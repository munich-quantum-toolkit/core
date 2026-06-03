/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/IR/QCOUnitaryMatrixInterfaces.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Euler.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QCO/Utils/WireIterator.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <Eigen/Core>
#include <mlir/Dialect/Arith/IR/Arith.h> // IWYU pragma: keep (Passes.h.inc)
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <iterator>
#include <optional>
#include <ranges>
#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_FUSESINGLEQUBITUNITARYRUNS
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

static bool isFuseCandidate(UnitaryOpInterface op) {
  if (!op || !op.isSingleQubit()) {
    return false;
  }
  return isa<UnitaryMatrixOpInterface>(op.getOperation());
}

static std::optional<Eigen::Matrix2cd> getConstMatrix(UnitaryOpInterface op) {
  auto matrixOp = dyn_cast<UnitaryMatrixOpInterface>(op.getOperation());
  if (!matrixOp) {
    return std::nullopt;
  }
  Eigen::Matrix2cd m;
  if (!matrixOp.getUnitaryMatrix2x2(m)) {
    return std::nullopt;
  }
  return m;
}

/// Compose a run of unitary ops (execution order) into a single matrix.
static std::optional<Eigen::Matrix2cd>
composeRun(ArrayRef<UnitaryOpInterface> run) {
  Eigen::Matrix2cd composed = Eigen::Matrix2cd::Identity();
  for (auto op : run) {
    auto m = getConstMatrix(op);
    if (!m) {
      return std::nullopt;
    }
    // Execution order: first op applied first => multiply on the left.
    composed = (*m) * composed;
  }
  return composed;
}

namespace {

struct FuseSingleQubitUnitaryRunsPass final
    : impl::FuseSingleQubitUnitaryRunsBase<FuseSingleQubitUnitaryRunsPass> {
  using Base::Base;

  explicit FuseSingleQubitUnitaryRunsPass(
      FuseSingleQubitUnitaryRunsOptions options)
      : Base(std::move(options)) {}

protected:
  void runOnOperation() override {
    auto module = getOperation();

    const auto parsed = decomposition::parseEulerBasis(this->basis);
    if (!parsed) {
      module.emitError() << "Invalid Euler basis '" << this->basis
                         << "'. Expected one of: zyz, zxz, xzx, xyx, u, zsxx.";
      signalPassFailure();
      return;
    }

    SmallVector<Value, 16> wireStarts;
    module.walk([&](AllocOp op) { wireStarts.emplace_back(op.getResult()); });
    module.walk([&](StaticOp op) { wireStarts.emplace_back(op.getQubit()); });
    module.walk([&](qtensor::ExtractOp op) {
      wireStarts.emplace_back(op.getResult());
    });
    module.walk([&](func::FuncOp func) {
      if (func.empty()) {
        return;
      }
      for (BlockArgument arg : func.getBody().front().getArguments()) {
        if (isa<QubitType>(arg.getType())) {
          wireStarts.emplace_back(arg);
        }
      }
    });

    // Collect runs first, rewrite afterwards.
    SmallVector<SmallVector<UnitaryOpInterface, 8>, 16> runs;
    DenseSet<Operation*> seen;

    auto flushRun = [&](SmallVector<UnitaryOpInterface, 8>& current) {
      if (current.size() > 1) {
        runs.emplace_back(std::move(current));
        current = SmallVector<UnitaryOpInterface, 8>();
      } else {
        current.clear();
      }
    };

    for (Value start : wireStarts) {
      if (!start) {
        continue;
      }

      SmallVector<UnitaryOpInterface, 8> current;
      Block* currentBlock = nullptr;

      WireIterator it(start);
      ++it; // Move to the first op on the wire.
      for (; it != std::default_sentinel; ++it) {
        Operation* op = *it;

        if (currentBlock == nullptr) {
          currentBlock = op->getBlock();
        }
        if (op->getBlock() != currentBlock) {
          break;
        }

        if (seen.contains(op)) {
          // Wire may be reached from multiple starts; flush any partial run.
          flushRun(current);
          continue;
        }
        if (!isa<UnitaryOpInterface>(op)) {
          flushRun(current);
          continue;
        }

        auto iface = cast<UnitaryOpInterface>(op);
        if (!isFuseCandidate(iface) || !getConstMatrix(iface).has_value()) {
          flushRun(current);
          continue;
        }

        current.emplace_back(iface);
        seen.insert(op);
      }

      flushRun(current);
    }

    for (auto& run : runs) {
      if (run.empty() || run.front().getOperation()->getParentOp() == nullptr) {
        continue;
      }

      auto composed = composeRun(run);
      if (!composed) {
        continue;
      }

      OpBuilder builder(run.front().getOperation());
      Value qubit = decomposition::synthesizeUnitary1QEuler(
          builder, run.front().getLoc(), run.front().getInputTarget(0),
          *composed, *parsed);

      run.back().getOutputTarget(0).replaceAllUsesWith(qubit);
      for (auto& it : std::ranges::reverse_view(run)) {
        it.getOperation()->erase();
      }
    }
  }
};

} // namespace

} // namespace mlir::qco
