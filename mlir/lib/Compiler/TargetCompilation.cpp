/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Compiler/TargetCompilation.h"

#include "mqt/Compiler/Target.h"
#include "mqt/Compiler/TargetEnvironment.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/MQT/IR/QubitLayout.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mqt/Dialect/QCO/Transforms/Mapping/Mapping.h"
#include "mqt/Dialect/QCO/Transforms/Passes.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"
#include "mqt/Dialect/QTensor/Transforms/Passes.h"
#include "mqt/Support/Passes.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <utility>

namespace mlir {

/// Retain source order before cleanup removes idle inputs or shrinks tensors.
static LogicalResult prepareLayout(ModuleOp moduleOp,
                                   const TargetEnvironment& environment) {
  moduleOp->removeAttr(mqt::kSourceQubitCountAttr);
  const auto& target = environment.target();
  auto entry = mqt::getEntryPoint(moduleOp);
  if (!entry || !llvm::hasSingleElement(entry.getBody()) ||
      llvm::any_of(entry.getArgumentTypes(), qco::isLinearQubitType)) {
    return success();
  }
  SmallVector<std::pair<Operation*, size_t>> roots;
  size_t count = 0;
  bool invalid = false;
  const auto result = moduleOp.walk([&](Operation* op) {
    if (op->hasAttr(mqt::kSourceQubitIndicesAttr)) {
      op->emitError("layout preparation requires input without source tags");
      invalid = true;
      return WalkResult::interrupt();
    }
    if (!isa<qco::AllocOp, qtensor::AllocOp>(op)) {
      return WalkResult::advance();
    }
    if (op->getBlock() != &entry.getBody().front()) {
      return WalkResult::interrupt();
    }
    size_t size = 1;
    if (auto tensor = dyn_cast<qtensor::AllocOp>(op)) {
      const auto extent = getConstantIntValue(tensor.getSize());
      if (!extent || *extent <= 0) {
        return WalkResult::interrupt();
      }
      size = static_cast<size_t>(*extent);
    }
    if (size > target.numSites() - count) {
      return WalkResult::interrupt();
    }
    roots.emplace_back(op, size);
    count += size;
    return WalkResult::advance();
  });
  if (invalid) {
    return failure();
  }
  if (result.wasInterrupted() || count == 0) {
    return success();
  }
  Builder builder(moduleOp.getContext());
  int64_t offset = 0;
  for (auto [op, size] : roots) {
    SmallVector<int64_t> indices(size);
    std::iota(indices.begin(), indices.end(), offset);
    offset += static_cast<int64_t>(size);
    op->setAttr(mqt::kSourceQubitIndicesAttr,
                builder.getDenseI64ArrayAttr(indices));
  }
  moduleOp->setAttr(mqt::kSourceQubitCountAttr,
                    builder.getI64IntegerAttr(static_cast<int64_t>(count)));
  return success();
}

namespace {

class PrepareTargetCompilationPass
    : public PassWrapper<PrepareTargetCompilationPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareTargetCompilationPass)

  explicit PrepareTargetCompilationPass(TargetEnvironment environment,
                                        bool allToAllOnly = false,
                                        MappingOptions mapping = {})
      : environment_(std::move(environment)), allToAllOnly_(allToAllOnly),
        mapping_(mapping) {}

protected:
  void runOnOperation() override {
    if (getOperation()->hasAttr("mqt.layout")) {
      getOperation().emitError("discard existing layout metadata before target "
                               "compilation");
      signalPassFailure();
      return;
    }
    if (mapping_.trials == 0) {
      getOperation().emitError("mapping trials must be greater than zero");
      signalPassFailure();
      return;
    }
    const auto& target = environment_.target();
    if (target.nativeOperationsKind() ==
            CompilerTarget::NativeOperations::Kind::Explicit &&
        !target.synthesisBasis()) {
      getOperation().emitError(
          "target compilation requires a single-qubit synthesis basis");
      signalPassFailure();
      return;
    }
    if (failed(mqt::verifyQuantumAllocations(getOperation()))) {
      signalPassFailure();
      return;
    }
    bool hasStatic = false;
    const auto staticSites = getOperation().walk([&](qco::StaticOp op) {
      hasStatic = true;
      const auto site = static_cast<CompilerTarget::SiteId>(op.getIndex());
      if (environment_.target().vertexForSite(site)) {
        return WalkResult::advance();
      }
      op.emitError() << "target does not contain static site " << site;
      return WalkResult::interrupt();
    });
    if (staticSites.wasInterrupted()) {
      signalPassFailure();
      return;
    }
    if (allToAllOnly_ &&
        environment_.target().connectivityKind() !=
            CompilerTarget::Connectivity::Kind::AllToAll &&
        !hasStatic) {
      getOperation().emitError(
          "target synthesis requires all-to-all connectivity; use target "
          "compilation for routing");
      signalPassFailure();
      return;
    }
    getAnalysis<TargetEnvironmentAnalysis>().initialize(environment_);
    auto result = getOperation().walk([](Operation* operation) {
      if (operation->getNumSuccessors() == 0) {
        return WalkResult::advance();
      }
      operation->emitError(
          "target compilation requires structured QCO/SCF input; normalize "
          "CFG branches before compilation");
      return WalkResult::interrupt();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }
    if (hasStatic) {
      getOperation()->removeAttr(mqt::kSourceQubitCountAttr);
    } else if (failed(prepareLayout(getOperation(), environment_))) {
      signalPassFailure();
      return;
    }
    markAnalysesPreserved<TargetEnvironmentAnalysis>();
  }

private:
  TargetEnvironment environment_;
  bool allToAllOnly_;
  MappingOptions mapping_;
};

} /* namespace */

void populateTargetCompilationPipeline(OpPassManager& pm,
                                       const TargetEnvironment& environment,
                                       const MappingOptions& mapping) {
  pm.addPass(std::make_unique<PrepareTargetCompilationPass>(environment, false,
                                                            mapping));
  const auto& target = environment.target();
  /// The module cleanup below owns canonicalization and dead symbols.
  pm.addPass(createInlinerPass({}, [](OpPassManager&) {}));
  pm.addPass(createSCCPPass());
  /// Placement changes region results; run liveness once afterwards.
  populateQCOCleanupPipeline(pm, /*removeDeadValues=*/false);
  pm.addPass(qco::createUnrollLoopsForPayload());
  pm.addPass(createSCCPPass());
  /// Unrolling exposes static tensor slots and unreachable callees.
  pm.addPass(createCanonicalizerPass(
      GreedyRewriteConfig{}.setMaxIterations(GreedyRewriteConfig::kNoLimit)));
  pm.addPass(createCSEPass());
  pm.addPass(qtensor::createShrinkQTensorToFitPass());
  pm.addPass(createSymbolDCEPass());
  pm.addPass(qco::createLegalizeControlFlow());
  pm.addPass(qco::createDecomposeMultiControlled(target));
  pm.addPass(qco::createFuseTwoQubitGates(target));
  /// U fusion shrinks runs before routing. Other bases can expand symbolic
  /// runs, so emit them once during native synthesis, after cleanup.
  if (const auto basis = target.synthesisBasis();
      basis && basis->singleQubit == CompilerTarget::SingleQubitBasis::U) {
    pm.addPass(qco::createFuseSingleQubitUnitaryRuns(target));
  }
  switch (target.connectivityKind()) {
  case CompilerTarget::Connectivity::Kind::Explicit: {
    qco::MappingPassOptions mappingOptions;
    if (mapping.trials) {
      mappingOptions.ntrials = *mapping.trials;
    }
    mappingOptions.niterations = mapping.iterations;
    mappingOptions.nlookahead = mapping.lookahead;
    mappingOptions.searchMemoryLimit = mapping.searchMemoryLimit;
    pm.addPass(qco::createMappingPass(mappingOptions));
    break;
  }
  case CompilerTarget::Connectivity::Kind::AllToAll:
    pm.addPass(qco::createPlacementPass(target));
    break;
  }
  qco::populateTargetNativeSynthesisPipeline(pm);
}

void populateTargetSynthesisPipeline(OpPassManager& pm,
                                     const TargetEnvironment& environment,
                                     const MappingOptions& mapping) {
  pm.addPass(std::make_unique<PrepareTargetCompilationPass>(environment, true,
                                                            mapping));
  const auto& target = environment.target();
  /// The module cleanup below owns canonicalization and dead symbols.
  pm.addPass(createInlinerPass({}, [](OpPassManager&) {}));
  /// Placement changes region results; run liveness once afterwards.
  populateQCOCleanupPipeline(pm, /*removeDeadValues=*/false);
  pm.addPass(qco::createLegalizeControlFlow());
  pm.addPass(qco::createDecomposeMultiControlled(target));
  pm.addPass(qco::createFuseTwoQubitGates(target));
  pm.addPass(qco::createPlacementPass(target));
  qco::populateTargetNativeSynthesisPipeline(pm);
}

} // namespace mlir
