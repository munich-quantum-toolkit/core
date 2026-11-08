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
#include "mqt/Dialect/MQT/IR/MQTAttributes.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/MQT/IR/QubitLayout.h"
#include "mqt/Dialect/MQT/Transforms/Passes.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mqt/Dialect/QCO/Transforms/Mapping/Mapping.h"
#include "mqt/Dialect/QCO/Transforms/Passes.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"
#include "mqt/Dialect/QTensor/Transforms/Passes.h"
#include "mqt/Support/Passes.h"

#include "ResonatorRouting.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <system_error>
#include <utility>
#include <vector>

namespace mlir {

static llvm::Expected<CompilerTarget>
placementTarget(const CompilerTarget& target) {
  auto placeable = target.placeableSites();
  if (placeable.empty()) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "no site supports the operations required for qubit placement");
  }
  if (placeable.size() == target.numSites()) {
    return target;
  }
  std::set<CompilerTarget::Coupling> edges;
  for (const auto& edge : target.couplings()) {
    if (llvm::is_contained(placeable, edge.first) &&
        llvm::is_contained(placeable, edge.second)) {
      edges.insert(edge);
    }
  }
  std::set<CompilerTarget::Coupling> virtualPairs;
  forEachVirtualCZ(target, placeable, [&](auto a, auto b, auto, bool) {
    virtualPairs.emplace(a, b);
    virtualPairs.emplace(b, a);
    edges.emplace(a, b);
  });
  if (target.connectivityKind() !=
      CompilerTarget::Connectivity::Kind::AllToAll) {
    llvm::EquivalenceClasses<CompilerTarget::SiteId> components;
    for (const auto site : placeable) {
      components.insert(site);
    }
    for (const auto& [a, b] : edges) {
      components.unionSets(a, b);
    }
    std::set<CompilerTarget::SiteId> visited;
    auto largest = placeable.front();
    size_t largestSize = 0;
    for (const auto site : placeable) {
      if (visited.insert(components.getLeaderValue(site)).second) {
        const auto size = llvm::range_size(components.members(site));
        if (size > largestSize) {
          largest = site;
          largestSize = size;
        }
      }
    }
    std::erase_if(placeable, [&](auto site) {
      return !components.isEquivalent(largest, site);
    });
    std::erase_if(edges, [&](const auto& edge) {
      return !llvm::is_contained(placeable, edge.first) ||
             !llvm::is_contained(placeable, edge.second);
    });
  }
  std::vector<CompilerTarget::Site> sites;
  for (const auto& site : target.sites()) {
    if (llvm::is_contained(placeable, site.id())) {
      sites.push_back(site);
    }
  }
  using Capability = CompilerTarget::OperationCapability;
  std::vector<Capability> operations;
  for (const auto& op : target.operations()) {
    if (op.arity().value() > placeable.size()) {
      continue;
    }
    std::vector<CompilerTarget::SiteTuple> tuples;
    for (const auto& tuple : op.siteTuples()) {
      if (llvm::all_of(tuple.sites(), [&](auto site) {
            return llvm::is_contained(placeable, site);
          })) {
        tuples.push_back(tuple);
      }
    }
    if (!op.siteTuples().empty() && tuples.empty()) {
      continue;
    }
    operations.push_back(llvm::cantFail(Capability::create(
        op.name().str(), op.arity(), op.numParameters(), std::move(tuples),
        op.duration(), op.fidelity(),
        {op.fixedParameters().begin(), op.fixedParameters().end()},
        op.canonicalName().str(),
        {op.parameterBounds().begin(), op.parameterBounds().end()})));
  }
  std::vector<CompilerTarget::SiteTuple> virtualCZs;
  for (const auto& [a, b] : virtualPairs) {
    if (llvm::is_contained(placeable, a) && llvm::is_contained(placeable, b)) {
      virtualCZs.push_back(
          llvm::cantFail(CompilerTarget::SiteTuple::create({a, b})));
    }
  }
  if (!virtualCZs.empty()) {
    operations.push_back(
        llvm::cantFail(Capability::create("cz", 2, 0, std::move(virtualCZs))));
  }
  const std::vector<CompilerTarget::Coupling> couplings(edges.begin(),
                                                        edges.end());
  const auto connectivity =
      target.connectivityKind() == CompilerTarget::Connectivity::Kind::AllToAll
          ? CompilerTarget::Connectivity::allToAll()
          : CompilerTarget::Connectivity::fromCouplings(couplings);
  const auto native =
      CompilerTarget::NativeOperations::fromOperations(operations);
  return target.name() ? CompilerTarget::create(target.name()->str(),
                                                std::move(sites), connectivity,
                                                native, target.durationUnit())
                       : CompilerTarget::create(std::move(sites), connectivity,
                                                native, target.durationUnit());
}

llvm::Expected<TargetEnvironment>
createTargetCompilationEnvironment(ModuleOp moduleOp,
                                   const TargetEnvironment& environment) {
  if (!environment.target().hasDeviceConstraints()) {
    return environment;
  }
  auto routing = QuantumProgramAnalysis(moduleOp).info().numQubits == 0
                     ? llvm::Expected<CompilerTarget>(environment.target())
                     : placementTarget(environment.target());
  if (!routing) {
    return routing.takeError();
  }
  return TargetEnvironment(*routing, environment.payloadSpecification());
}

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

static void populateTargetPreparationCleanup(OpPassManager& pm) {
  // The module cleanup owns canonicalization and dead symbols.
  pm.addPass(createInlinerPass({}, [](OpPassManager&) {}));
  // Defer tensor shrinking until decomposition and placement.
  populateQCOCleanupPipeline(pm, /*removeUnusedValues=*/false);
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
      if (!target.hasDeviceConstraints()) {
        op.emitError("gate-set synthesis requires allocated program qubits");
      } else {
        op.emitError() << "target does not contain static site " << site;
      }
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
    if (!target.hasDeviceConstraints()) {
      auto entry = mqt::getEntryPoint(getOperation());
      if (!entry || !entry.getBody().hasOneBlock() ||
          llvm::any_of(entry.getArgumentTypes(), qco::isLinearQubitType)) {
        getOperation().emitError(
            "gate-set synthesis requires fixed-width entry-block allocations");
        signalPassFailure();
        return;
      }
      const auto allocations = getOperation().walk([&](Operation* op) {
        if (!isa<qco::AllocOp, qtensor::AllocOp>(op)) {
          return WalkResult::advance();
        }
        if (op->getBlock() != &entry.getBody().front()) {
          op->emitError("gate-set synthesis requires entry-block allocations");
          return WalkResult::interrupt();
        }
        if (auto tensor = dyn_cast<qtensor::AllocOp>(op)) {
          const auto extent = getConstantIntValue(tensor.getSize());
          if (!extent || *extent < 0) {
            tensor.emitError(
                "gate-set synthesis requires fixed-width allocations");
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });
      if (allocations.wasInterrupted()) {
        signalPassFailure();
        return;
      }
    } else if (hasStatic) {
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

class PrepareStagedTargetCompilationPass final
    : public PassWrapper<PrepareStagedTargetCompilationPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      PrepareStagedTargetCompilationPass)

  [[nodiscard]] StringRef getArgument() const override {
    return "mqt-target-prepare";
  }
  [[nodiscard]] StringRef getDescription() const override {
    return "Prepare source wire order and the routing target for custom QCO "
           "compilation";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mqt::MQTDialect>();
    OpPassManager pm(ModuleOp::getOperationName());
    populateTargetPreparationCleanup(pm);
    pm.getDependentDialects(registry);
  }

protected:
  void runOnOperation() override {
    auto moduleOp = getOperation();
    if (moduleOp->hasAttr(mqt::ORIGINAL_TARGET_ENV_ATTR)) {
      moduleOp.emitError("target compilation is already prepared");
      signalPassFailure();
      return;
    }
    const auto& analysis = getAnalysis<TargetEnvironmentAnalysis>();
    if (!analysis) {
      moduleOp.emitError() << "expected a valid mqt.target_env: "
                           << analysis.error();
      signalPassFailure();
      return;
    }
    const TargetEnvironment environment = analysis.environment();
    auto compilationEnvironment =
        createTargetCompilationEnvironment(moduleOp, environment);
    if (!compilationEnvironment) {
      moduleOp.emitError() << llvm::toString(
          compilationEnvironment.takeError());
      signalPassFailure();
      return;
    }
    OpPassManager pm(ModuleOp::getOperationName());
    pm.addPass(std::make_unique<PrepareTargetCompilationPass>(
        *compilationEnvironment));
    populateTargetPreparationCleanup(pm);
    if (failed(runPipeline(pm, moduleOp))) {
      signalPassFailure();
      return;
    }
    moduleOp->setAttr(mqt::ORIGINAL_TARGET_ENV_ATTR,
                      environment.materialize(getContext()));
  }
};

class FinalizeStagedTargetCompilationPass final
    : public PassWrapper<FinalizeStagedTargetCompilationPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      FinalizeStagedTargetCompilationPass)

  [[nodiscard]] StringRef getArgument() const override {
    return "mqt-target-finalize";
  }
  [[nodiscard]] StringRef getDescription() const override {
    return "Synthesize native QCO operations, verify conformance, and restore "
           "the execution target";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    OpPassManager pm(ModuleOp::getOperationName());
    qco::populateTargetNativeSynthesisPipeline(pm);
    pm.getDependentDialects(registry);
  }

protected:
  void runOnOperation() override {
    auto moduleOp = getOperation();
    std::optional<TargetEnvironment> originalEnvironment;
    if (auto attribute = moduleOp->getAttrOfType<mqt::TargetEnvAttr>(
            mqt::ORIGINAL_TARGET_ENV_ATTR)) {
      auto environment = TargetEnvironment::create(attribute);
      if (!environment) {
        moduleOp.emitError() << llvm::toString(environment.takeError());
        signalPassFailure();
        return;
      }
      originalEnvironment = std::move(*environment);
    }
    OpPassManager pm(ModuleOp::getOperationName());
    qco::populateTargetNativeSynthesisPipeline(pm);
    if (failed(runPipeline(pm, moduleOp))) {
      signalPassFailure();
      return;
    }
    if (originalEnvironment) {
      attachTargetEnvironment(moduleOp, *originalEnvironment);
    }
  }
};

} /* namespace */

void registerTargetCompilationPasses() {
  static const auto REGISTERED = [] {
    PassRegistration<PrepareStagedTargetCompilationPass>();
    PassRegistration<FinalizeStagedTargetCompilationPass>();
    registerPass(createLowerResonatorCZPass);
    return true;
  }();
  static_cast<void>(REGISTERED);
}

/// Borrow before placement while allocations still prove clean states.
static void populateDecompositionAndPlacement(OpPassManager& pm,
                                              const CompilerTarget& target) {
  const bool needsSites =
      target.connectivityKind() ==
          CompilerTarget::Connectivity::Kind::AllToAll &&
      llvm::any_of(target.operations(), [](const auto& capability) {
        return !capability.siteTuples().empty() &&
               capability.arity().value() >= 3;
      });
  if (!needsSites) {
    pm.addPass(qco::createDecomposeMultiControlled(target));
  }
  if (!target.hasDeviceConstraints()) {
    return;
  }
  // Drop unused slots before placement checks target capacity.
  pm.addPass(mqt::createRemoveUnusedQubits());
  pm.addPass(qtensor::createShrinkQTensorToFitPass());
  pm.addPass(qco::createElideTerminalSwapsPass());
  if (target.connectivityKind() ==
      CompilerTarget::Connectivity::Kind::AllToAll) {
    pm.addPass(qco::createPlacementPass());
  }
  if (needsSites) {
    pm.addPass(qco::createDecomposeMultiControlled(target));
  }
}

void populateTargetCompilationPipeline(OpPassManager& pm,
                                       const TargetEnvironment& environment,
                                       const MappingOptions& mapping) {
  pm.addPass(std::make_unique<PrepareTargetCompilationPass>(environment, false,
                                                            mapping));
  const auto& target = environment.target();
  populateTargetPreparationCleanup(pm);
  pm.addPass(qco::createCombineControlledGates(target));
  // Discrete targets may not support the gates produced by composition.
  if (target.synthesisBasis()) {
    pm.addPass(mqt::createSimplifyQuantum());
  }
  // Eliminate identity loops before unrolling applies its clone budget.
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createSCCPPass());
  pm.addPass(qco::createUnrollLoopsForPayload());
  pm.addPass(createSCCPPass());
  // Unrolling exposes static register indices and tensor slots.
  populateQCOCleanupPipeline(pm, /*removeUnusedValues=*/false);
  pm.addPass(qco::createLegalizeControlFlow());
  populateDecompositionAndPlacement(pm, target);
  pm.addPass(qco::createFuseTwoQubitGates(target));
  // U fusion shrinks runs before routing. Other bases can expand symbolic
  // runs, so emit them once during native synthesis, after cleanup.
  if (const auto basis = target.synthesisBasis();
      basis && basis->singleQubit == CompilerTarget::SingleQubitBasis::U) {
    pm.addPass(qco::createFuseSingleQubitUnitaryRuns(target));
  }
  // Keep native controlled U bodies; distribute other composites for lowering.
  pm.addPass(mqt::createUnrollModifiers());
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
  populateTargetPreparationCleanup(pm);
  pm.addPass(qco::createCombineControlledGates(target));
  if (target.synthesisBasis()) {
    pm.addPass(mqt::createSimplifyQuantum());
  }
  // Remove identity branches before checking payload control-flow support.
  pm.addPass(createCanonicalizerPass());
  pm.addPass(qco::createLegalizeControlFlow());
  populateDecompositionAndPlacement(pm, target);
  pm.addPass(qco::createFuseTwoQubitGates(target));
  if (const auto basis = target.synthesisBasis();
      basis && basis->singleQubit == CompilerTarget::SingleQubitBasis::U) {
    pm.addPass(qco::createFuseSingleQubitUnitaryRuns(target));
  }
  pm.addPass(mqt::createUnrollModifiers());
  qco::populateTargetNativeSynthesisPipeline(pm);
}

} // namespace mlir
