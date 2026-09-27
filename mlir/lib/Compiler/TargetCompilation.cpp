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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
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
                                        MappingOptions mapping = {},
                                        qco::LayoutTracking* tracking = nullptr)
      : environment_(std::move(environment)), allToAllOnly_(allToAllOnly),
        mapping_(mapping), tracking_(tracking) {}

protected:
  void runOnOperation() override {
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
    if (tracking_ != nullptr &&
        failed(qco::prepareLayout(getOperation(), environment_.target(),
                                  *tracking_))) {
      signalPassFailure();
      return;
    }
    markAnalysesPreserved<TargetEnvironmentAnalysis>();
  }

private:
  TargetEnvironment environment_;
  bool allToAllOnly_;
  MappingOptions mapping_;
  qco::LayoutTracking* tracking_;
};

} /* namespace */

static FailureOr<mqt::QubitLayout>
composeCompiledLayout(ModuleOp moduleOp, const CompilerTarget& target,
                      const MappingResult& mapping,
                      const std::optional<mqt::QubitLayout>& source) {
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
  layout.physicalSize = static_cast<int64_t>(width);
  layout.outputOrder.resize(width);
  std::iota(layout.outputOrder.begin(), layout.outputOrder.end(), 0);
  std::vector<int64_t> previousRouting(width);
  std::iota(previousRouting.begin(), previousRouting.end(), 0);
  if (source) {
    if (std::cmp_not_equal(source->physicalSize, sourceCount) ||
        source->initial.size() != sourceCount ||
        std::ranges::any_of(source->initial,
                            [](int64_t site) { return site < 0; }) ||
        (source->routing &&
         std::ranges::any_of(*source->routing,
                             [](int64_t site) { return site < 0; }))) {
      return moduleOp.emitError(
          "native compilation requires a complete imported qubit layout");
    }
    layout.inputCount = source->inputCount.value_or(
        static_cast<int64_t>(sourceCount - source->ancillas.size()));
    layout.ancillas = source->ancillas;
    layout.registers = source->registers;
    for (const auto initial : source->initial) {
      layout.initial.push_back(placement[initial]);
    }
    if (source->routing) {
      for (size_t wire = 0; wire < sourceCount; ++wire) {
        layout.outputOrder[placement[wire]] =
            placement[source->outputOrder[wire]];
        previousRouting[wire] = (*source->routing)[wire];
      }
    }
  } else {
    layout.inputCount = static_cast<int64_t>(sourceCount);
    layout.initial.assign(placement.begin(),
                          placement.begin() +
                              static_cast<std::ptrdiff_t>(sourceCount));
    if (sourceCount != 0) {
      mqt::LayoutRegister input{.name = "input"};
      input.slots.resize(sourceCount);
      std::iota(input.slots.begin(), input.slots.end(), 0);
      layout.registers.push_back(std::move(input));
    }
  }
  for (size_t index = sourceCount; index < width; ++index) {
    layout.ancillas.push_back(static_cast<int64_t>(layout.initial.size()));
    layout.initial.push_back(placement[index]);
  }

  std::vector<int64_t> inversePlacement(width);
  for (size_t index = 0; index < width; ++index) {
    inversePlacement[placement[index]] = static_cast<int64_t>(index);
  }
  layout.routing.emplace(width);
  for (size_t initial = 0; initial < width; ++initial) {
    const auto oldWire = inversePlacement[initial];
    const auto afterOldRouting = placement[previousRouting[oldWire]];
    (*layout.routing)[initial] =
        static_cast<int64_t>(mapping.routingPermutation[afterOldRouting]);
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
                                                            mapping, tracking));
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

std::optional<MappingResult>
QCOProgram::compileForTargetWithLayout(const TargetEnvironment& environment,
                                       llvm::ArrayRef<int64_t> initialLayout,
                                       const CompilationOptions& options) {
  if (failed(mqt::verifyLayoutEntryPoint(mod())) || !hasValidLinearity()) {
    return std::nullopt;
  }
  auto entryPoint = mqt::getEntryPoint(mod());
  std::vector<std::optional<int64_t>> inputSegments;
  llvm::SmallDenseSet<int64_t> staticSites;
  for (Operation& operation : entryPoint.getBody().front()) {
    if (auto staticQubit = dyn_cast<qco::StaticOp>(operation)) {
      const auto site = static_cast<int64_t>(staticQubit.getIndex());
      if (!environment.target().vertexForSite(site) ||
          !staticSites.insert(site).second) {
        staticQubit.emitError("preplaced qubit requires a distinct target "
                              "site ID");
        return std::nullopt;
      }
      inputSegments.emplace_back(site);
    } else if (isa<qco::AllocOp, qtensor::AllocOp>(operation)) {
      inputSegments.emplace_back(std::nullopt);
    }
  }
  if (!staticSites.empty() && !initialLayout.empty()) {
    entryPoint.emitError("explicit initial layout cannot move preplaced "
                         "static qubits");
    return std::nullopt;
  }
  if (entryPoint->hasAttr("mqt.layout_invalidated")) {
    entryPoint.emitError("discard invalidated qubit layout before native "
                         "compilation");
    return std::nullopt;
  }
  std::optional<mqt::QubitLayout> source;
  if (const auto attribute = entryPoint->getAttr("mqt.layout")) {
    auto parsed = mqt::QubitLayout::fromAttr(
        attribute, [&] { return entryPoint.emitError(); });
    if (failed(parsed)) {
      return std::nullopt;
    }
    source = std::move(*parsed);
  }
  if (source && !staticSites.empty()) {
    entryPoint.emitError("cannot recompose qubit layout for preplaced static "
                         "qubits; import the circuit with dynamic inputs");
    return std::nullopt;
  }
  qco::LayoutTracking tracking{.requested = initialLayout};
  if (failed(runWithPassManager(
          mod(),
          [&](OpPassManager& pm) {
            populateTargetPipeline(pm, environment, options.mapping, &tracking);
          },
          "failed to compile the QCO program with layout tracking", options)) ||
      !hasValidLinearity()) {
    return std::nullopt;
  }
  if (!staticSites.empty()) {
    MappingResult combined;
    size_t allocation = 0;
    size_t wire = 0;
    for (const auto site : inputSegments) {
      if (site) {
        combined.allocationSizes.push_back(1);
        combined.initialLayout.push_back(*site);
        combined.finalLayout.push_back(*site);
        continue;
      }
      const auto count = tracking.result.allocationSizes[allocation++];
      combined.allocationSizes.push_back(count);
      for (size_t index = 0; index < count; ++index, ++wire) {
        combined.initialLayout.push_back(tracking.result.initialLayout[wire]);
        combined.finalLayout.push_back(tracking.result.finalLayout[wire]);
      }
    }
    combined.routingPermutation = std::move(tracking.result.routingPermutation);
    tracking.result = std::move(combined);
  }
  auto layout = composeCompiledLayout(mod(), environment.target(),
                                      tracking.result, source);
  if (failed(layout)) {
    return std::nullopt;
  }
  mqt::discardQubitLayout(mod());
  mqt::getEntryPoint(mod())->setAttr("mqt.layout",
                                     layout->toAttr(mod().getContext()));
  return std::move(tracking.result);
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
