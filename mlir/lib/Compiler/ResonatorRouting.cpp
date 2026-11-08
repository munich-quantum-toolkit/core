/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ResonatorRouting.h"

#include "mqt/Compiler/TargetEnvironment.h"
#include "mqt/Dialect/MQT/IR/MQTAttributes.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QC/IR/QCDialect.h"
#include "mqt/Dialect/QC/IR/QCOps.h"

#include "TerminalReadout.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <utility>

namespace mlir {

void forEachVirtualCZ(
    const CompilerTarget& target, ArrayRef<CompilerTarget::SiteId> sites,
    function_ref<void(CompilerTarget::SiteId, CompilerTarget::SiteId,
                      CompilerTarget::SiteId, bool)>
        visit) {
  for (const auto& move : target.operations()) {
    if (move.canonicalName() != "move" ||
        move.arity() != CompilerTarget::OperationCapability::Arity::fixed(2) ||
        move.numParameters() != 0) {
      continue;
    }
    for (const auto& tuple : move.siteTuples()) {
      const auto a = tuple.sites()[0];
      const auto r = tuple.sites()[1];
      if (!llvm::is_contained(sites, a) || llvm::is_contained(sites, r)) {
        continue;
      }
      for (const auto& cz : target.operations()) {
        if (cz.canonicalName() != "cz" ||
            cz.arity() !=
                CompilerTarget::OperationCapability::Arity::fixed(2) ||
            cz.numParameters() != 0) {
          continue;
        }
        for (const auto& pair : cz.siteTuples()) {
          const bool first = pair.sites()[0] == r;
          const auto b = pair.sites()[first ? 1 : 0];
          if (pair.sites()[first ? 0 : 1] == r && a != b &&
              llvm::is_contained(sites, b)) {
            visit(a, b, r, first);
          }
        }
      }
    }
  }
}

static void insertResonatorMoves(ModuleOp moduleOp,
                                 const CompilerTarget& target) {
  if (llvm::none_of(target.operations(), [](const auto& operation) {
        return operation.canonicalName() == "move";
      })) {
    return;
  }
  using SiteId = CompilerTarget::SiteId;
  const auto placeable = target.placeableSites();
  std::map<std::pair<SiteId, SiteId>, std::pair<SiteId, bool>> routes;
  forEachVirtualCZ(target, placeable, [&](auto a, auto b, auto r, bool first) {
    routes.try_emplace({a, b}, r, first);
  });
  if (routes.empty()) {
    return;
  }
  func::FuncOp move;
  OpBuilder builder(moduleOp.getContext());
  moduleOp.walk([&](qc::CtrlOp ctrl) {
    if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1 ||
        ctrl.getNumBodyUnitaries() != 1 ||
        !isa<qc::ZOp>(ctrl.getBodyUnitary(0).getOperation())) {
      return;
    }
    auto a = ctrl.getControl(0).getDefiningOp<qc::StaticOp>();
    auto b = ctrl.getTarget(0).getDefiningOp<qc::StaticOp>();
    if (!a || !b ||
        target.supports(CompilerTarget::GateKind::CZ,
                        std::array<int64_t, 2>{
                            static_cast<int64_t>(a.getIndex()),
                            static_cast<int64_t>(b.getIndex()),
                        })) {
      return;
    }
    auto route = routes.find({a.getIndex(), b.getIndex()});
    if (route == routes.end()) {
      route = routes.find({b.getIndex(), a.getIndex()});
    }
    if (route == routes.end()) {
      return;
    }
    const auto [moving, stationary] = route->first;
    const auto [resonator, first] = route->second;
    if (!move) {
      SmallVector<Type, 2> types(2, qc::QubitType::get(moduleOp.getContext()));
      move = func::FuncOp::create(moduleOp.getLoc(), "move",
                                  builder.getFunctionType(types, {}));
      move.setPrivate();
      move->setAttr(mqt::MQTDialect::SourceNameAttrHelper::getNameStr(),
                    builder.getStringAttr("move"));
      SymbolTable(moduleOp).insert(move);
    }
    builder.setInsertionPoint(ctrl);
    Value r = qc::StaticOp::create(builder, ctrl.getLoc(),
                                   qc::QubitType::get(moduleOp.getContext()),
                                   resonator);
    Value m = a.getIndex() == moving ? a.getResult() : b.getResult();
    Value s = a.getIndex() == stationary ? a.getResult() : b.getResult();
    func::CallOp::create(builder, ctrl.getLoc(), move, ValueRange{m, r});
    qc::CtrlOp::create(builder, ctrl.getLoc(), ValueRange{first ? r : s},
                       ValueRange{first ? s : r}, [&](ValueRange values) {
                         qc::ZOp::create(builder, ctrl.getLoc(),
                                         values.front());
                       });
    func::CallOp::create(builder, ctrl.getLoc(), move, ValueRange{m, r});
    ctrl.erase();
  });
  scheduleTerminalReadout(moduleOp);
}

namespace {

class LowerResonatorCZPass final
    : public PassWrapper<LowerResonatorCZPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerResonatorCZPass)

  [[nodiscard]] StringRef getArgument() const override {
    return "mqt-lower-resonator-cz";
  }

  [[nodiscard]] StringRef getDescription() const override {
    return "Expand virtual QC CZ gates into native MOVE/CZ/MOVE sequences";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<func::FuncDialect, mqt::MQTDialect, qc::QCDialect>();
  }

protected:
  void runOnOperation() override {
    auto moduleOp = getOperation();
    if (!moduleOp->hasAttr(mqt::TargetEnvAttr::name)) {
      return;
    }
    const auto& environment = getAnalysis<TargetEnvironmentAnalysis>();
    if (!environment) {
      moduleOp.emitError(environment.error());
      signalPassFailure();
      return;
    }
    insertResonatorMoves(moduleOp, environment.environment().target());
  }
};

} // namespace

std::unique_ptr<Pass> createLowerResonatorCZPass() {
  return std::make_unique<LowerResonatorCZPass>();
}

} // namespace mlir
