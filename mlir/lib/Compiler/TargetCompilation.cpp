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

#include "mqt/Compiler/Programs.h"
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

#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/DenseSet.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace mlir {
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
    if (allToAllOnly_ && environment_.target().connectivityKind() !=
                             CompilerTarget::Connectivity::Kind::AllToAll) {
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
    markAnalysesPreserved<TargetEnvironmentAnalysis>();
  }

private:
  TargetEnvironment environment_;
  bool allToAllOnly_;
  MappingOptions mapping_;
};

} /* namespace */

static FailureOr<mqt::QubitLayout>
compiledLayout(ModuleOp moduleOp, const CompilerTarget& target,
               const qco::LayoutTracking& mapping) {
  const size_t width = target.numSites();
  const size_t sourceCount = mapping.initialLayout.size();
  if (sourceCount > width || mapping.routingPermutation.size() != width) {
    return moduleOp.emitError(
        "native qubit layout does not match target width");
  }
  std::vector<int64_t> placement;
  placement.reserve(width);
  std::vector<bool> used(width, false);
  for (const auto site : mapping.initialLayout) {
    const auto vertex = target.vertexForSite(site);
    if (!vertex || used[*vertex]) {
      return moduleOp.emitError("native qubit layout has an invalid site");
    }
    placement.push_back(static_cast<int64_t>(*vertex));
    used[*vertex] = true;
  }
  for (size_t vertex = 0; vertex < width; ++vertex) {
    if (!used[vertex]) {
      placement.push_back(static_cast<int64_t>(vertex));
    }
  }

  mqt::QubitLayout layout;
  layout.inputCount = static_cast<int64_t>(sourceCount);
  layout.initial = std::move(placement);

  layout.routing.emplace(width);
  for (size_t initial = 0; initial < width; ++initial) {
    (*layout.routing)[initial] =
        static_cast<int64_t>(mapping.routingPermutation[initial]);
  }
  return layout;
}

static void populatePostPlacementPipeline(OpPassManager& pm) {
  /// Placement consumes allocations; native synthesis normalizes phases.
  pm.addPass(createCanonicalizerPass(
      GreedyRewriteConfig{}.setMaxIterations(GreedyRewriteConfig::kNoLimit)));
  /// Reuse unchanged classical reads before native synthesis splits their uses.
  pm.addPass(createCSEPass());
  pm.addPass(createRemoveDeadValuesPass());
  pm.addPass(qco::createTargetNativeSynthesis());
  pm.addPass(createCSEPass());
  pm.addPass(qco::createVerifyTargetConformance());
}

static void populateTargetPipeline(OpPassManager& pm,
                                   const TargetEnvironment& environment,
                                   const MappingOptions& mapping,
                                   qco::LayoutTracking* tracking) {
  pm.addPass(std::make_unique<PrepareTargetCompilationPass>(environment, false,
                                                            mapping));
  const auto& target = environment.target();
  pm.addPass(createInlinerPass());
  pm.addPass(createSymbolDCEPass());
  pm.addPass(createSCCPPass());
  populateQCOCleanupPipeline(pm);
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
  /// Non-U targets fuse during native synthesis, avoiding an intermediate U
  /// representation and its symbolic phase correction.
  if (const auto basis = target.synthesisBasis();
      !basis || basis->singleQubit == CompilerTarget::SingleQubitBasis::U) {
    /// The U optimizer also merges dynamic controlled bodies into native U
    /// gates and preserves isolated gates. Native synthesis does not yet cover
    /// both behaviors; keep this path until their synthesis contracts agree.
    populateDefaultQCOOptimizationPipeline(pm);
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
    pm.addPass(qco::createMappingPass(mappingOptions, tracking));
    break;
  }
  case CompilerTarget::Connectivity::Kind::AllToAll:
    pm.addPass(qco::createPlacementPass(target, tracking));
    break;
  }
  populatePostPlacementPipeline(pm);
}

void populateTargetCompilationPipeline(OpPassManager& pm,
                                       const TargetEnvironment& environment,
                                       const MappingOptions& mapping) {
  populateTargetPipeline(pm, environment, mapping, {});
}

LogicalResult runTargetCompilation(ModuleOp moduleOp, PassManager& pm,
                                   const TargetEnvironment& environment,
                                   const CompilationOptions& options) {
  if (!pm.empty() || pm.getContext() != moduleOp.getContext()) {
    return moduleOp.emitError(
        "target compilation requires an empty pass manager for this context");
  }
  auto entryPoint = mqt::getEntryPoint(moduleOp);
  if (!entryPoint) {
    return moduleOp.emitError("target compilation requires an entry point");
  }
  if (failed(qco::verifyLinearity(moduleOp))) {
    return failure();
  }
  if (options.mapping.trials == 0) {
    return moduleOp.emitError("mapping trials must be greater than zero");
  }
  std::vector<std::optional<int64_t>> inputSegments;
  llvm::SmallDenseSet<int64_t> staticSites;
  for (Operation& operation : entryPoint.getBody().front()) {
    if (auto staticQubit = dyn_cast<qco::StaticOp>(operation)) {
      const auto site = static_cast<int64_t>(staticQubit.getIndex());
      if (!environment.target().vertexForSite(site) ||
          !staticSites.insert(site).second) {
        staticQubit.emitError("preplaced qubit requires a distinct target "
                              "site ID");
        return failure();
      }
      inputSegments.emplace_back(site);
    } else if (isa<qco::AllocOp, qtensor::AllocOp>(operation)) {
      inputSegments.emplace_back(std::nullopt);
    }
  }
  if (moduleOp->hasAttr("mqt.layout")) {
    return moduleOp.emitError("discard existing layout metadata before target "
                              "compilation");
  }
  qco::LayoutTracking tracking;
  const auto prepared =
      qco::prepareLayout(moduleOp, environment.target(), tracking);
  if (failed(prepared)) {
    return failure();
  }
  auto* activeTracking = *prepared ? &tracking : nullptr;
  populateTargetPipeline(pm, environment, options.mapping, activeTracking);
  if (failed(runWithCompilationOptions(pm, moduleOp, options, true)) ||
      failed(qco::verifyLinearity(moduleOp))) {
    moduleOp.walk(
        [](Operation* op) { op->removeAttr(mqt::kSourceQubitIndicesAttr); });
    return moduleOp.emitError(
        "failed to compile the QCO program for the target");
  }
  if (activeTracking == nullptr) {
    return success();
  }
  if (!staticSites.empty()) {
    std::vector<int64_t> combined;
    size_t allocation = 0;
    size_t wire = 0;
    for (const auto site : inputSegments) {
      if (site) {
        combined.push_back(*site);
        continue;
      }
      const auto count = tracking.allocationSizes[allocation++];
      for (size_t index = 0; index < count; ++index, ++wire) {
        combined.push_back(tracking.initialLayout[wire]);
      }
    }
    tracking.initialLayout = std::move(combined);
  }
  if (tracking.initialLayout.empty()) {
    return success();
  }
  auto layout = compiledLayout(moduleOp, environment.target(), tracking);
  if (failed(layout)) {
    return failure();
  }
  moduleOp->setAttr("mqt.layout", layout->toAttr(moduleOp.getContext()));
  return success();
}

bool QCOProgram::compileForTarget(const TargetEnvironment& environment,
                                  const CompilationOptions& options) {
  PassManager pm(mod().getContext());
  return succeeded(runTargetCompilation(mod(), pm, environment, options));
}

void populateTargetSynthesisPipeline(OpPassManager& pm,
                                     const TargetEnvironment& environment,
                                     const MappingOptions& mapping) {
  pm.addPass(std::make_unique<PrepareTargetCompilationPass>(environment, true,
                                                            mapping));
  const auto& target = environment.target();
  pm.addPass(createInlinerPass());
  pm.addPass(createSymbolDCEPass());
  populateQCOCleanupPipeline(pm);
  pm.addPass(qco::createLegalizeControlFlow());
  pm.addPass(qco::createDecomposeMultiControlled(target));
  pm.addPass(qco::createFuseTwoQubitGates(target));
  pm.addPass(qco::createPlacementPass(target));
  populatePostPlacementPipeline(pm);
}

} // namespace mlir
