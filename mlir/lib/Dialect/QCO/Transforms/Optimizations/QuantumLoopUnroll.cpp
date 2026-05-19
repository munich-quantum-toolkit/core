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
#include "mlir/Dialect/QCO/Transforms/Passes.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Utils/Utils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>

#include <cassert>
#include <tuple>

namespace mlir::qco {

#define GEN_PASS_DEF_QUANTUMLOOPUNROLL
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

/**
 * @brief Predicate for quantum loops.
 * @details A quantum loop is a `scf.for` operation that has at least one qubit
 * or qtensor value as init argument.
 * @param loop The loop to test.
 * @returns true, if the loop is a quantum loop.
 */
static bool isQuantumLoop(scf::ForOp loop) {
  return llvm::any_of(loop.getInitArgs(), [](Value arg) {
    const auto type = arg.getType();
    return isa<QubitType>(type) ||
           (isa<RankedTensorType>(type) &&
            isa<QubitType>(cast<RankedTensorType>(type).getElementType()));
  });
}

/**
 * @brief Post-order collect all quantum loops inside a module.
 * @param m The module.
 * @return A vector of quantum `scf.for` loops.
 */
static SmallVector<scf::ForOp> collectQuantumLoops(FunctionOpInterface func) {
  SmallVector<scf::ForOp> loops;
  std::ignore = func.walk<WalkOrder::PostOrder>([&](scf::ForOp loop) {
    if (!isQuantumLoop(loop)) {
      return WalkResult::advance();
    }

    loops.emplace_back(loop);
    return WalkResult::advance();
  });
  return loops;
}

namespace {

/**
 * @brief Unroll bounded quantum loops.
 */
struct QuantumLoopUnroll final
    : impl::QuantumLoopUnrollBase<QuantumLoopUnroll> {
  using QuantumLoopUnrollBase::QuantumLoopUnrollBase;

protected:
  void runOnOperation() override {
    assert(unrollFactor >= -1 && "runOnOperation: invalid unroll factor");

    // Note that the built-in loop-unrolling utilities initialize
    // `IRRewriter`s using the context of the loop operation and automatically
    // rewrite the IR. This is the reason why we don't use patterns here.

    // A unroll-factor of zero is a no-op.
    if (unrollFactor == 0) {
      return;
    }

    // If the unroll factor is larger than zero, partially unroll the loops.
    if (unrollFactor > 0) {
      for (auto loop : collectQuantumLoops(getOperation())) {
        if (failed(loopUnrollByFactor(loop, unrollFactor))) {
          loop.emitError() << "failed to unroll with factor " +
                                  Twine(unrollFactor);
          signalPassFailure();
          return;
        }
      }

      return;
    }

    // Otherwise, fully unroll all loops.
    for (auto loop : collectQuantumLoops(getOperation())) {
      if (failed(loopUnrollFull(loop))) {
        loop.emitError() << "failed to fully unroll";
        signalPassFailure();
        return;
      }
    }
  }
};
} // namespace
} // namespace mlir::qco
