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
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Stack.h"

#include <cassert>
#include <cstddef>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>
#include <vector>

#define DEBUG_TYPE "routing-verification-sc"

namespace mqt::ir::opt {

#define GEN_PASS_DEF_ROUTINGVERIFICATIONSCPASS
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc"

namespace {
using namespace mlir;

/**
 * @brief The necessary datastructures for verification.
 */
struct VerificationContext {
  explicit VerificationContext(Architecture& arch) : arch(&arch) {}

  Architecture* arch;
  LayoutStack<Layout> stack{};
};

/**
 * @brief Push new state onto the stack. Skip non entry-point functions.
 */
WalkResult handleFunc(func::FuncOp op, VerificationContext& ctx) {
  if (!isEntryPoint(op)) {
    return WalkResult::skip();
  }

  /// Function body state.
  ctx.stack.emplace(ctx.arch->nqubits());

  return WalkResult::advance();
}

/**
 * @brief Defines the end of a region: Pop the top of the stack.
 */
WalkResult handleReturn(VerificationContext& ctx) {
  ctx.stack.pop();
  return WalkResult::advance();
}

/**
 * @brief Prepares state for nested regions: Pushes a copy of the state on
 * the stack. Forwards all out-of-loop and in-loop SSA values for their
 * respective map in the stack.
 */
WalkResult handleFor(scf::ForOp op, VerificationContext& ctx) {
  /// Loop body state.
  ctx.stack.duplicateTop();

  /// Forward out-of-loop and in-loop values.
  const auto initArgs = op.getInitArgs().take_front(ctx.arch->nqubits());
  const auto results = op.getResults().take_front(ctx.arch->nqubits());
  const auto iterArgs = op.getRegionIterArgs().take_front(ctx.arch->nqubits());
  for (const auto [arg, res, iter] : llvm::zip(initArgs, results, iterArgs)) {
    ctx.stack.getItemAtDepth(FOR_PARENT_DEPTH).remapQubitValue(arg, res);
    ctx.stack.top().remapQubitValue(arg, iter);
  }

  return WalkResult::advance();
}

/**
 * @brief Prepares state for nested regions: Pushes two copies of the state on
 * the stack. Forwards the results in the parent state.
 */
WalkResult handleIf(scf::IfOp op, VerificationContext& ctx) {
  /// Prepare stack.
  ctx.stack.duplicateTop(); /// Else
  ctx.stack.duplicateTop(); /// Then.

  /// Forward results for all hardware qubits.
  const auto results = op->getResults().take_front(ctx.arch->nqubits());
  Layout& stateBeforeIf = ctx.stack.getItemAtDepth(IF_PARENT_DEPTH);
  for (const auto [hardwareIdx, res] : llvm::enumerate(results)) {
    const Value q = stateBeforeIf.lookupHardwareValue(hardwareIdx);
    stateBeforeIf.remapQubitValue(q, res);
  }

  return WalkResult::advance();
}

/**
 * @brief Defines the end of a nested region: Pop the top of the stack.
 */
WalkResult handleYield(scf::YieldOp op, VerificationContext& ctx) {
  if (isa<scf::ForOp>(op->getParentOp()) || isa<scf::IfOp>(op->getParentOp())) {
    assert(ctx.stack.size() >= 2 && "expected at least two elements on stack.");

    if (!llvm::equal(ctx.stack.top().getCurrentLayout(),
                     ctx.stack.getItemAtDepth(1).getCurrentLayout())) {
      return op.emitOpError() << "layouts must match after restoration";
    }

    ctx.stack.pop();
  }
  return WalkResult::advance();
}

/**
 * @brief Add hardware qubit with respective program & hardware index to layout.
 */
WalkResult handleQubit(QubitOp op, VerificationContext& ctx) {
  const std::size_t index = op.getIndex();
  ctx.stack.top().add(index, index, op.getQubit());
  return WalkResult::advance();
}

/**
 * @brief Verifies if the unitary acts on either zero, one, or two qubits:
 * - Advances for zero qubit unitaries (Nothing to do)
 * - Forwards SSA values for one qubit.
 * - Verifies executability for two-qubit gates for the given architecture and
 *   forwards SSA values.
 */
WalkResult handleUnitary(UnitaryInterface op, VerificationContext& ctx) {
  const std::vector<Value> inQubits = op.getAllInQubits();
  const std::vector<Value> outQubits = op.getAllOutQubits();
  const std::size_t nacts = inQubits.size();

  if (nacts == 0) {
    return WalkResult::advance();
  }

  if (isa<BarrierOp>(op)) {
    for (const auto [in, out] : llvm::zip(inQubits, outQubits)) {
      ctx.stack.top().remapQubitValue(in, out);
    }
    return WalkResult::advance();
  }

  if (nacts > 2) {
    return op->emitOpError() << "acts on more than two qubits";
  }

  const Value in0 = inQubits[0];
  const Value out0 = outQubits[0];

  Layout& state = ctx.stack.top();

  if (nacts == 1) {
    state.remapQubitValue(in0, out0);
    return WalkResult::advance();
  }

  const Value in1 = inQubits[1];
  const Value out1 = outQubits[1];

  const auto idx0 = state.lookupHardwareIndex(in0);
  const auto idx1 = state.lookupHardwareIndex(in1);

  if (!ctx.arch->areAdjacent(idx0, idx1)) {
    return op->emitOpError() << "(" << idx0 << "," << idx1 << ")"
                             << " is not executable on target architecture '"
                             << ctx.arch->name() << "'";
  }

  if (isa<SWAPOp>(op)) {
    state.swap(in0, in1);
  }

  state.remapQubitValue(in0, out0);
  state.remapQubitValue(in1, out1);

  return WalkResult::advance();
}

/**
 * @brief Update layout.
 */
WalkResult handleReset(ResetOp op, VerificationContext& ctx) {
  ctx.stack.top().remapQubitValue(op.getInQubit(), op.getOutQubit());
  return WalkResult::advance();
}

/**
 * @brief Update layout.
 */
WalkResult handleMeasure(MeasureOp op, VerificationContext& ctx) {
  ctx.stack.top().remapQubitValue(op.getInQubit(), op.getOutQubit());
  return WalkResult::advance();
}

/**
 * @brief This pass verifies that the constraints of a target architecture are
 * met.
 */
struct RoutingVerificationPassSC final
    : impl::RoutingVerificationSCPassBase<RoutingVerificationPassSC> {
  void runOnOperation() override {
    const auto arch = getArchitecture(ArchitectureName::MQTTest);
    VerificationContext ctx(*arch);

    const auto res =
        getOperation()->walk<WalkOrder::PreOrder>([&](Operation* op) {
          return TypeSwitch<Operation*, WalkResult>(op)
              /// built-in Dialect
              .Case<ModuleOp>(
                  [&](ModuleOp /* op */) { return WalkResult::advance(); })
              /// func Dialect
              .Case<func::FuncOp>(
                  [&](func::FuncOp op) { return handleFunc(op, ctx); })
              .Case<func::ReturnOp>(
                  [&](func::ReturnOp /* op */) { return handleReturn(ctx); })
              /// scf Dialect
              .Case<scf::ForOp>(
                  [&](scf::ForOp op) { return handleFor(op, ctx); })
              .Case<scf::IfOp>([&](scf::IfOp op) { return handleIf(op, ctx); })
              .Case<scf::YieldOp>(
                  [&](scf::YieldOp op) { return handleYield(op, ctx); })
              /// mqtopt Dialect
              .Case<QubitOp>([&](QubitOp op) { return handleQubit(op, ctx); })
              .Case<AllocQubitOp, DeallocQubitOp>([&](auto op) {
                return WalkResult(
                    op->emitOpError("not allowed for transpiled program"));
              })
              .Case<ResetOp>([&](ResetOp op) { return handleReset(op, ctx); })
              .Case<MeasureOp>(
                  [&](MeasureOp op) { return handleMeasure(op, ctx); })
              .Case<UnitaryInterface>([&](UnitaryInterface unitary) {
                return handleUnitary(unitary, ctx);
              })
              /// Skip the rest.
              .Default([](auto) { return WalkResult::skip(); });
        });

    if (res.wasInterrupted()) {
      signalPassFailure();
    }
  }
};
} // namespace
} // namespace mqt::ir::opt
