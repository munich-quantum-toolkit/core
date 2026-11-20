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
#include "llvm/ADT/iterator_range.h"

#include <array>
#include <memory>
#include <utility>

namespace mlir {

// Keep the first reported order when both CZ placements are native.
static DenseSet<CompilerTarget::Coupling>
preferredCZOrientations(const CompilerTarget& target) {
  using Arity = CompilerTarget::OperationCapability::Arity;
  DenseSet<CompilerTarget::Coupling> result;
  for (const auto& op : target.operations()) {
    if (op.canonicalName() != "cz" || op.arity() != Arity::fixed(2) ||
        op.numParameters() != 0) {
      continue;
    }
    for (const auto& tuple : op.siteTuples()) {
      auto sites = tuple.sites();
      if (!result.contains({sites[1], sites[0]})) {
        result.insert({sites[0], sites[1]});
      }
    }
  }
  return result;
}

bool forEachVirtualCZ(
    const CompilerTarget& target,
    function_ref<void(CompilerTarget::SiteId, CompilerTarget::SiteId,
                      CompilerTarget::SiteId, bool)>
        visit) {
  const auto operations = target.operations();
  const auto* firstMove = llvm::find_if(operations, [](const auto& operation) {
    return operation.canonicalName() == "move";
  });
  if (firstMove == operations.end()) {
    return false;
  }
  const auto placeable = target.placeableSites();
  const DenseSet<CompilerTarget::SiteId> placementSites(placeable.begin(),
                                                        placeable.end());
  const auto preferredCZs = preferredCZOrientations(target);
  for (const auto& move : llvm::make_range(firstMove, operations.end())) {
    if (move.canonicalName() != "move" ||
        move.arity() != CompilerTarget::OperationCapability::Arity::fixed(2) ||
        move.numParameters() != 0) {
      continue;
    }
    for (const auto& tuple : move.siteTuples()) {
      const auto a = tuple.sites()[0];
      const auto r = tuple.sites()[1];
      if (!placementSites.contains(a) || placementSites.contains(r) ||
          !target.arePhysicallyCoupled(a, r)) {
        continue;
      }
      // Query the real neighbours of the resonator. The target resolves native
      // CZ aliases, unrestricted placements, and ordered site tuples alike.
      target.forEachNeighbour(*target.vertexForSite(r), [&](size_t vertex) {
        const auto b = target.siteForVertex(vertex);
        if (a == b || !placementSites.contains(b) ||
            !target.arePhysicallyCoupled(r, b)) {
          return;
        }
        bool resonatorFirst = preferredCZs.contains({r, b});
        if (!resonatorFirst &&
            !target.supports(CompilerTarget::GateKind::CZ, std::array{b, r})) {
          if (!target.supports(CompilerTarget::GateKind::CZ,
                               std::array{r, b})) {
            return;
          }
          resonatorFirst = true;
        }
        visit(a, b, r, resonatorFirst);
      });
    }
  }
  return true;
}

static void insertResonatorMoves(ModuleOp moduleOp,
                                 const CompilerTarget& target) {
  using SiteId = CompilerTarget::SiteId;
  DenseMap<std::pair<SiteId, SiteId>, std::pair<SiteId, bool>> routes;
  const bool hasMove =
      forEachVirtualCZ(target, [&](auto a, auto b, auto r, bool first) {
        routes.try_emplace({a, b}, r, first);
      });
  if (!hasMove) {
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
    if (!a || !b) {
      return;
    }
    const auto firstSite = static_cast<SiteId>(a.getIndex());
    const auto secondSite = static_cast<SiteId>(b.getIndex());
    if (target.arePhysicallyCoupled(firstSite, secondSite)) {
      if (target.supports(CompilerTarget::GateKind::CZ,
                          std::array{firstSite, secondSite})) {
        return;
      }
      if (target.supports(CompilerTarget::GateKind::CZ,
                          std::array{secondSite, firstSite})) {
        ctrl->setOperands({b.getResult(), a.getResult()});
        return;
      }
    }
    auto route = routes.find({firstSite, secondSite});
    if (route == routes.end()) {
      route = routes.find({secondSite, firstSite});
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
    Value m = firstSite == moving ? a.getResult() : b.getResult();
    Value s = firstSite == stationary ? a.getResult() : b.getResult();
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
