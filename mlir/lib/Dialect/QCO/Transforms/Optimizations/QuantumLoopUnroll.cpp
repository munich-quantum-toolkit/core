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
#include "mlir/Dialect/QCO/QCOUtils.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Utils/Utils.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Verifier.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <cstdint>
#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_QUANTUMLOOPUNROLL
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

/// Keep verifier-valid loop bounds from turning this optimization into an
/// unbounded allocation request.
static constexpr uint64_t K_MAX_QUANTUM_LOOP_UNROLL_FACTOR = 4096;
static constexpr uint64_t K_MAX_QUANTUM_LOOP_EXPANDED_OPERATIONS = 100000;

/**
 * @brief Predicate for quantum loops.
 * @details A quantum loop is a `scf.for` operation that has at least one qubit
 * or qtensor value as init argument.
 * @param loop The loop to test.
 * @returns true, if the loop is a quantum loop.
 */
static bool isQuantumLoop(scf::ForOp loop) {
  return llvm::any_of(loop.getInitArgs(), [](Value arg) {
    if (isa<QubitType>(arg.getType())) {
      return true;
    }
    if (const auto tensorTy = dyn_cast<RankedTensorType>(arg.getType())) {
      return isa<QubitType>(tensorTy.getElementType());
    }
    return false;
  });
}

/**
 * @brief Post-order collect all quantum loops in a function.
 * @param func The function to collect quantum loops from.
 * @return A vector of quantum `scf.for` loops.
 */
static SmallVector<scf::ForOp> collectQuantumLoops(FunctionOpInterface func) {
  SmallVector<scf::ForOp> loops;
  func.walk<WalkOrder::PostOrder>([&](scf::ForOp loop) {
    if (isQuantumLoop(loop)) {
      loops.emplace_back(loop);
    }
  });
  return loops;
}

/** @brief Whether a loop body only yields its iteration arguments unchanged. */
static bool hasIdentityYieldOnlyBody(scf::ForOp loop) {
  if (!llvm::hasSingleElement(loop.getBody()->getOperations())) {
    return false;
  }
  auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
  return yield && llvm::equal(yield.getResults(), loop.getRegionIterArgs());
}

/** @brief Check the projected unrolled IR size before cloning or rewriting. */
static LogicalResult verifyUnrollExpansionBudget(FunctionOpInterface func,
                                                 int64_t unrollFactor) {
  uint64_t projectedOperations = 0;
  SmallVector<std::pair<Operation*, uint64_t>> worklist;
  worklist.emplace_back(func.getOperation(), 1);

  while (!worklist.empty()) {
    const auto [operation, multiplier] = worklist.pop_back_val();
    if (multiplier >
        K_MAX_QUANTUM_LOOP_EXPANDED_OPERATIONS - projectedOperations) {
      return operation->emitError()
             << "quantum loop unrolling would exceed the limit of "
             << K_MAX_QUANTUM_LOOP_EXPANDED_OPERATIONS
             << " projected operations";
    }
    projectedOperations += multiplier;

    uint64_t nestedMultiplier = multiplier;
    if (auto loop = dyn_cast<scf::ForOp>(operation);
        loop && isQuantumLoop(loop)) {
      if (hasIdentityYieldOnlyBody(loop)) {
        nestedMultiplier = 0;
      } else {
        uint64_t factor = 0;
        if (unrollFactor == -1) {
          const auto tripCount = loop.getStaticTripCount();
          if (!tripCount) {
            // Nested bounds may become constant after an enclosing loop is
            // unrolled. Account for their current body once and recheck the
            // complete budget before every subsequent unrolling round.
            factor = 1;
          } else {
            factor = tripCount->getLimitedValue(
                K_MAX_QUANTUM_LOOP_UNROLL_FACTOR + 1);
          }
        } else {
          factor = static_cast<uint64_t>(unrollFactor);
        }
        if (factor > K_MAX_QUANTUM_LOOP_UNROLL_FACTOR) {
          return loop.emitError() << "quantum loop unroll factor " << factor
                                  << " exceeds the limit of "
                                  << K_MAX_QUANTUM_LOOP_UNROLL_FACTOR;
        }
        if (factor != 0 &&
            nestedMultiplier >
                K_MAX_QUANTUM_LOOP_EXPANDED_OPERATIONS / factor) {
          return loop.emitError()
                 << "quantum loop unrolling would exceed the limit of "
                 << K_MAX_QUANTUM_LOOP_EXPANDED_OPERATIONS
                 << " projected operations";
        }
        nestedMultiplier *= factor;
      }
    }

    if (nestedMultiplier == 0) {
      continue;
    }
    for (Region& region : operation->getRegions()) {
      for (Block& block : region) {
        for (Operation& nested : block) {
          worklist.emplace_back(&nested, nestedMultiplier);
        }
      }
    }
  }
  return success();
}

/** @brief Unroll all selected loops in @p func. */
static LogicalResult unrollQuantumLoops(FunctionOpInterface func,
                                        int64_t unrollFactor) {
  if (unrollFactor == -1) {
    while (true) {
      if (failed(verifyUnrollExpansionBudget(func, unrollFactor))) {
        return failure();
      }
      auto loops = collectQuantumLoops(func);
      if (loops.empty()) {
        return success();
      }

      bool changed = false;
      for (auto loop : loops) {
        if (hasIdentityYieldOnlyBody(loop)) {
          loop.replaceAllUsesWith(loop.getInitArgs());
          loop.erase();
          changed = true;
          continue;
        }

        const auto tripCount = loop.getStaticTripCount();
        if (!tripCount) {
          continue;
        }
        if (tripCount->isZero()) {
          loop.replaceAllUsesWith(loop.getInitArgs());
          loop.erase();
          changed = true;
          continue;
        }

        if (failed(loopUnrollFull(loop))) {
          loop.emitError() << "failed to fully unroll";
          return failure();
        }
        changed = true;
      }

      if (!changed) {
        loops.front().emitError()
            << "cannot fully unroll a quantum loop without a static trip "
               "count";
        return failure();
      }

      if (failed(applyPatternsGreedily(func,
                                       RewritePatternSet(func.getContext())))) {
        return failure();
      }
    }
  }

  for (auto loop : collectQuantumLoops(func)) {
    if (failed(loopUnrollByFactor(loop, unrollFactor))) {
      loop.emitError() << "failed to unroll with factor " + Twine(unrollFactor);
      return failure();
    }
  }
  return success();
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
    if (unrollFactor < -1) {
      getOperation()->emitError()
          << "invalid unroll factor " << Twine(unrollFactor);
      signalPassFailure();
      return;
    }
    if (std::cmp_greater(static_cast<int64_t>(unrollFactor),
                         K_MAX_QUANTUM_LOOP_UNROLL_FACTOR)) {
      getOperation()->emitError()
          << "quantum loop unroll factor " << Twine(unrollFactor)
          << " exceeds the limit of " << K_MAX_QUANTUM_LOOP_UNROLL_FACTOR;
      signalPassFailure();
      return;
    }

    // Note that the built-in loop-unrolling utilities initialize
    // `IRRewriter`s using the context of the loop operation and automatically
    // rewrite the IR. This is the reason why we don't use patterns here.

    // An unroll-factor of zero or one is a no-op.
    if (unrollFactor == 0 || unrollFactor == 1) {
      return;
    }

    if (collectQuantumLoops(getOperation()).empty()) {
      return;
    }

    if (failed(verifyUnrollExpansionBudget(getOperation(), unrollFactor))) {
      signalPassFailure();
      return;
    }

    // Perform the transformation on a clone first. Besides validating all
    // selected loop bounds, this keeps the source function untouched if any
    // loop cannot be unrolled.
    OwningOpRef<ModuleOp> transformedModule =
        ModuleOp::create(getOperation()->getLoc());
    Operation* transformedOperation = getOperation()->clone();
    transformedModule->push_back(transformedOperation);
    if (failed(unrollQuantumLoops(
            cast<FunctionOpInterface>(transformedOperation), unrollFactor))) {
      signalPassFailure();
      return;
    }
    // Symbol references remain unchanged, so verify the function without
    // requiring sibling symbols to be present in the temporary module.
    if (failed(verify(transformedOperation)) ||
        failed(qco::verifyLinearity(transformedOperation))) {
      getOperation()->emitError("quantum loop unrolling produced invalid IR");
      signalPassFailure();
      return;
    }

    for (auto [originalRegion, transformedRegion] :
         llvm::zip_equal(getOperation()->getRegions(),
                         transformedOperation->getRegions())) {
      originalRegion.takeBody(transformedRegion);
    }
  }
};
} // namespace
} // namespace mlir::qco
