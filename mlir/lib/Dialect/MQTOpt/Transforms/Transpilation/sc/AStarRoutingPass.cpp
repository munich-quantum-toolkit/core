/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Architecture.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Common.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Layout.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Router.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Unit.h"

#include <cassert>
#include <cstddef>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/Statistic.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/LogicalResult.h>
#include <memory>
#include <mlir/Analysis/TopologicalSortUtils.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Rewrite/PatternApplicator.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>
#include <queue>
#include <utility>

#define DEBUG_TYPE "route-astar-sc"

namespace mqt::ir::opt {

#define GEN_PASS_DEF_ASTARROUTINGPASSSC
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc"

namespace {
using namespace mlir;

/// @brief A composite datastructure for LLVM Statistics.
struct Statistics {
  llvm::Statistic* numSwaps;
};

/// @brief A composite datastructure for pass parameters.
struct Params {
  /// @brief The amount of lookahead layers.
  std::size_t nlookahead;
  /// @brief The alpha factor in the heuristic function.
  float alpha;
  /// @brief The lambda decay factor in the heuristic function.
  float lambda;
};

/// @brief Commonly passed parameters for the routing functions.
struct RoutingContext {
  /// @brief The targeted architecture.
  std::unique_ptr<Architecture> arch;
  /// @brief LLVM/MLIR statistics.
  Statistics stats;
  /// @brief A pattern rewriter.
  PatternRewriter rewriter;
  /// @brief The A*-search based router.
  AStarHeuristicRouter router;
  /// @brief The amount of lookahead layers.
  std::size_t nlookahead;
};

LogicalResult processFunction(func::FuncOp func, RoutingContext& ctx) {
  /// Collect entry layout.
  Layout layout(ctx.arch->nqubits());
  for_each(func.getOps<QubitOp>(), [&](QubitOp op) {
    layout.add(op.getIndex(), op.getIndex(), op.getQubit());
  });

  std::queue<Unit> units;
  units.emplace(std::move(layout), &func.getBody());

  for (; !units.empty(); units.pop()) {
    Unit& unit = units.front();
    unit.schedule();
    unit.route(ctx.router, ctx.nlookahead, *ctx.arch, ctx.rewriter);
    for (auto next : unit.advance()) {
      units.emplace(next);
    }
  }

  return success();
}

/**
 * @brief Route the given module for the targeted architecture using A*-search.
 * Processes each entry_point function separately.
 */
LogicalResult route(ModuleOp module, std::unique_ptr<Architecture> arch,
                    Params& params, Statistics& stats) {
  const HeuristicWeights weights(params.alpha, params.lambda,
                                 params.nlookahead);
  RoutingContext ctx{.arch = std::move(arch),
                     .stats = stats,
                     .rewriter = PatternRewriter(module->getContext()),
                     .router = AStarHeuristicRouter(weights),
                     .nlookahead = params.nlookahead};

  for (auto func : module.getOps<func::FuncOp>()) {
    LLVM_DEBUG(llvm::dbgs() << "handleFunc: " << func.getSymName() << '\n');

    if (!isEntryPoint(func)) {
      LLVM_DEBUG(llvm::dbgs() << "\tskip non entry\n");
      return success(); // Ignore non entry_point functions for now.
    }

    if (failed(processFunction(func, ctx))) {
      return failure();
    }
  }

  return success();
}

/**
 * @brief Routes the program by dividing the circuit into layers of parallel
 * two-qubit gates and iteratively searches and inserts SWAPs for each layer
 * using A*-search.
 */
struct AStarRoutingPassSC final
    : impl::AStarRoutingPassSCBase<AStarRoutingPassSC> {
  using AStarRoutingPassSCBase<AStarRoutingPassSC>::AStarRoutingPassSCBase;

  void runOnOperation() override {
    if (preflight().failed()) {
      signalPassFailure();
      return;
    }

    auto arch = getArchitecture(archName);
    if (!arch) {
      emitError(UnknownLoc::get(&getContext()))
          << "unsupported architecture '" << archName << "'";
      signalPassFailure();
      return;
    }

    Statistics stats{.numSwaps = &numSwaps};
    Params params{.nlookahead = nlookahead, .alpha = alpha, .lambda = lambda};
    if (route(getOperation(), std::move(arch), params, stats).failed()) {
      signalPassFailure();
    };
  }

private:
  LogicalResult preflight() {
    if (archName.empty()) {
      return emitError(UnknownLoc::get(&getContext()),
                       "required option 'arch' not provided");
    }

    return success();
  }
};

} // namespace
} // namespace mqt::ir::opt
