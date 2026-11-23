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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <array>
#include <memory>
#include <optional>
#include <utility>

namespace mlir {

// Return whether a symmetric CZ needs its operands reversed at these sites.
static std::optional<bool> reverseNativeCZ(const CompilerTarget& target,
                                           CompilerTarget::SiteId a,
                                           CompilerTarget::SiteId b) {
  if (target.supports(CompilerTarget::GateKind::CZ, std::array{a, b})) {
    return false;
  }
  if (target.supports(CompilerTarget::GateKind::CZ, std::array{b, a})) {
    return true;
  }
  return std::nullopt;
}

void forEachVirtualCZ(
    const CompilerTarget& target, ArrayRef<CompilerTarget::SiteId> sites,
    function_ref<void(CompilerTarget::SiteId, CompilerTarget::SiteId,
                      CompilerTarget::SiteId, bool)>
        visit) {
  const DenseSet<CompilerTarget::SiteId> placeable(sites.begin(), sites.end());
  for (const auto& move : target.operations()) {
    if (move.canonicalName() != "move" ||
        move.arity() != CompilerTarget::OperationCapability::Arity::fixed(2) ||
        move.numParameters() != 0) {
      continue;
    }
    for (const auto& tuple : move.siteTuples()) {
      const auto a = tuple.sites()[0];
      const auto r = tuple.sites()[1];
      if (!placeable.contains(a) || placeable.contains(r)) {
        continue;
      }
      target.forEachNeighbour(*target.vertexForSite(r), [&](size_t vertex) {
        const auto b = target.siteForVertex(vertex);
        if (a != b && placeable.contains(b)) {
          if (auto reverse = reverseNativeCZ(target, b, r)) {
            visit(a, b, r, *reverse);
          }
        }
      });
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
  DenseMap<std::pair<SiteId, SiteId>, std::pair<SiteId, bool>> routes;
  forEachVirtualCZ(target, placeable, [&](auto a, auto b, auto r, bool first) {
    routes.try_emplace({a, b}, r, first);
  });
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
    if (!a || !b) {
      return;
    }
    if (auto reverse =
            reverseNativeCZ(target, static_cast<SiteId>(a.getIndex()),
                            static_cast<SiteId>(b.getIndex()))) {
      if (*reverse) {
        ctrl->setOperands({b.getResult(), a.getResult()});
      }
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
  if (!routes.empty()) {
    scheduleTerminalReadout(moduleOp);
  }
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
