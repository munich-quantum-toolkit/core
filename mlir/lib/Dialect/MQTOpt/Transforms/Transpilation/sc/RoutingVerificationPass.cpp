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
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/SequentialUnit.h"

#include <cassert>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>
#include <memory>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>
#include <queue>
#include <utility>

#define DEBUG_TYPE "routing-verification-sc"

namespace mqt::ir::opt {

#define GEN_PASS_DEF_ROUTINGVERIFICATIONSCPASS
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc"

namespace {
using namespace mlir;

/**
 * @brief Given a layout, validate if the two-qubit unitary op is executable on
 * the targeted architecture.
 *
 * @param op The two-qubit unitary.
 * @param layout The current layout.
 * @param arch The targeted architecture.
 */
[[nodiscard]] bool isExecutable(UnitaryInterface op, const Layout& layout,
                                const Architecture& arch) {
  const auto ins = getIns(op);
  return arch.areAdjacent(layout.lookupHardwareIndex(ins.first),
                          layout.lookupHardwareIndex(ins.second));
}

/**
 * @brief This pass verifies that all two-qubit gates are executable on the
 * target architecture.
 */
struct RoutingVerificationPassSC final
    : impl::RoutingVerificationSCPassBase<RoutingVerificationPassSC> {
  using RoutingVerificationSCPassBase<
      RoutingVerificationPassSC>::RoutingVerificationSCPassBase;

  void runOnOperation() override {
    if (failed(preflight())) {
      signalPassFailure();
      return;
    }

    if (failed(verify())) {
      signalPassFailure();
      return;
    }
  }

private:
  LogicalResult verify() {
    ModuleOp module(getOperation());
    PatternRewriter rewriter(module->getContext());
    std::unique_ptr<Architecture> arch(getArchitecture(archName));

    if (!arch) {
      const Location loc = UnknownLoc::get(&getContext());
      emitError(loc) << "unsupported architecture '" << archName << "'";
      return failure();
    }

    for (auto func : module.getOps<func::FuncOp>()) {
      LLVM_DEBUG(llvm::dbgs() << "handleFunc: " << func.getSymName() << '\n');

      if (!isEntryPoint(func)) {
        LLVM_DEBUG(llvm::dbgs() << "\tskip non entry\n");
        continue;
      }

      /// Iteratively process each unit in the function.
      std::queue<SequentialUnit> units;
      units.emplace(
          SequentialUnit::fromEntryPointFunction(func, arch->nqubits()));
      for (; !units.empty(); units.pop()) {
        SequentialUnit& unit = units.front();

        Layout unmodified(unit.layout());
        SmallVector<QubitIndexPair> history;
        for (Operation& curr : unit) {
          rewriter.setInsertionPoint(&curr);

          const auto res =
              TypeSwitch<Operation*, LogicalResult>(&curr)
                  .Case<UnitaryInterface>([&](UnitaryInterface op)
                                              -> LogicalResult {
                    if (isTwoQubitGate(op)) {
                      /// Verify that the two-qubit gate is executable.
                      if (!isExecutable(op, unit.layout(), *arch)) {
                        const auto ins = getIns(op);
                        const auto hw0 =
                            unit.layout().lookupHardwareIndex(ins.first);
                        const auto hw1 =
                            unit.layout().lookupHardwareIndex(ins.second);

                        return op->emitOpError()
                               << "(" << hw0 << "," << hw1 << ")"
                               << " is not executable on target architecture '"
                               << arch->name() << "'";
                      }
                    }

                    if (isa<SWAPOp>(op)) {
                      const auto ins = getIns(op);
                      unit.layout().swap(ins.first, ins.second);
                      history.push_back(
                          {unit.layout().lookupHardwareIndex(ins.first),
                           unit.layout().lookupHardwareIndex(ins.second)});
                    }

                    remap(op, unit.layout());
                    return success();
                  })
                  .Case<ResetOp>([&](ResetOp op) {
                    remap(op, unit.layout());
                    return success();
                  })
                  .Case<MeasureOp>([&](MeasureOp op) {
                    remap(op, unit.layout());
                    return success();
                  })
                  .Case<scf::YieldOp>([&](scf::YieldOp op) -> LogicalResult {
                    if (!unit.restore()) {
                      return success();
                    }

                    /// Verify that the layouts match at the end.
                    const auto mappingBefore = unmodified.getCurrentLayout();
                    const auto mappingNow = unit.layout().getCurrentLayout();
                    if (llvm::equal(mappingBefore, mappingNow)) {
                      return success();
                    }

                    return op.emitOpError()
                           << "layouts must match after restoration";
                  })
                  .Default([](auto) { return success(); });

          if (failed(res)) {
            return res;
          }
        }

        for (const auto& next : unit.next()) {
          units.emplace(next);
        }
      }
    }

    return success();
  }

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
