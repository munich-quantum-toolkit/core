/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Compiler/TargetCompilation.h"

#include "mlir/Compiler/ProgramFormat.h"
#include "mlir/Compiler/Target.h"
#include "mlir/Dialect/CBit/IR/CBitOps.h"
#include "mlir/Dialect/MQT/IR/MQTAttributes.h"
#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Mapping/Mapping.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QCO/Utils/WireIterator.h"
#include "mlir/Support/Passes.h"

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Analysis/CallGraph.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Utils/Utils.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>

namespace mlir {
namespace {

constexpr uint64_t MAX_UNROLLED_OPERATIONS = 65536U;

[[nodiscard]] constexpr llvm::StringRef
programFeatureName(const ProgramFeature feature) {
  switch (feature) {
  case ProgramFeature::MidCircuitMeasurement:
    return "mid-circuit-measurement";
  case ProgramFeature::MeasuredQubitReuse:
    return "measured-qubit-reuse";
  case ProgramFeature::MeasurementResultUse:
    return "measurement-result-use";
  case ProgramFeature::BooleanComputation:
    return "boolean-computation";
  case ProgramFeature::IntegerComputation:
    return "integer-computation";
  case ProgramFeature::FloatComputation:
    return "float-computation";
  case ProgramFeature::ForwardBranching:
    return "forward-branching";
  case ProgramFeature::CountedIteration:
    return "counted-iteration";
  case ProgramFeature::ConditionalLoop:
    return "conditional-loop";
  case ProgramFeature::MultiwayBranching:
    return "multiway-branching";
  case ProgramFeature::IRFunctions:
    return "ir-functions";
  case ProgramFeature::MultipleReturnPoints:
    return "multiple-return-points";
  case ProgramFeature::DynamicQubitManagement:
    return "dynamic-qubit-management";
  case ProgramFeature::DynamicResultManagement:
    return "dynamic-result-management";
  case ProgramFeature::Arrays:
    return "arrays";
  }
  llvm_unreachable("unknown program feature");
}

[[nodiscard]] std::optional<PayloadDescriptor>
payloadForProgramFormat(const ProgramFormat format) {
  switch (format) {
  case ProgramFormat::OpenQASM3:
    return PayloadDescriptor{"openqasm", "3.0.0", "", PayloadEncoding::Text};
  case ProgramFormat::QIRBase:
    return PayloadDescriptor{"qir", "2.1.0", "base", PayloadEncoding::Text};
  case ProgramFormat::QIRAdaptive:
    return PayloadDescriptor{"qir", "2.1.0", "adaptive", PayloadEncoding::Text};
  case ProgramFormat::QCOOptimized:
  case ProgramFormat::QC:
    return PayloadDescriptor{"mqt-qco", "1.0.0", "", PayloadEncoding::Text};
  case ProgramFormat::QCImport:
  case ProgramFormat::QCO:
  case ProgramFormat::Jeff:
    return std::nullopt;
  }
  llvm_unreachable("unknown program format");
}

struct AttachTargetEnvironmentPass final
    : PassWrapper<AttachTargetEnvironmentPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AttachTargetEnvironmentPass)

  AttachTargetEnvironmentPass(const CompilerTarget& targetIn,
                              const PayloadDescriptor& descriptorIn)
      : target(targetIn), descriptor(descriptorIn) {}

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mqt::MQTDialect>();
  }

  void runOnOperation() override {
    const auto profiles = target.executionProfiles();
    const auto* const profile = target.executionProfile(descriptor);
    if (profiles && profile == nullptr && descriptor.id != "mqt-qco") {
      getOperation().emitError()
          << "compiler target does not report support for selected payload '"
          << descriptor.id << " " << descriptor.version << "'";
      signalPassFailure();
      return;
    }

    SmallVector<std::pair<llvm::StringRef, uint64_t>> capabilities;
    if (profile != nullptr) {
      capabilities.reserve(profile->capabilities().size());
      for (const ProgramCapability capability : profile->capabilities()) {
        capabilities.emplace_back(programFeatureName(capability.feature),
                                  capability.value);
      }
    }
    getOperation()->setAttr(
        mqt::TargetEnvAttr::getOperationAttributeName(),
        mqt::TargetEnvAttr::get(
            &getContext(), descriptor.id, descriptor.version,
            descriptor.profile,
            descriptor.encoding == PayloadEncoding::Binary ? "binary" : "text",
            capabilities,
            profile ? profile->optionalFeaturesKnown() : profiles.has_value(),
            target.materialize(&getContext())));
  }

private:
  CompilerTarget target;
  PayloadDescriptor descriptor;
};

struct UnrollUnsupportedCountedIterationPass final
    : PassWrapper<UnrollUnsupportedCountedIterationPass,
                  OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      UnrollUnsupportedCountedIterationPass)

  void runOnOperation() override {
    const auto environment = getOperation()->getAttrOfType<mqt::TargetEnvAttr>(
        mqt::TargetEnvAttr::getOperationAttributeName());
    if (!environment || environment.supports("counted-iteration")) {
      return;
    }
    SmallVector<scf::ForOp> loops;
    getOperation().walk<WalkOrder::PostOrder>([&](scf::ForOp loop) {
      if (loop.getStaticTripCount()) {
        loops.emplace_back(loop);
      }
    });
    for (scf::ForOp loop : loops) {
      const uint64_t tripCount =
          loop.getStaticTripCount()->getLimitedValue(MAX_UNROLLED_OPERATIONS);
      uint64_t bodyOperations = 0U;
      loop.getRegion().walk([&](Operation*) { ++bodyOperations; });
      if (tripCount != 0U &&
          bodyOperations > MAX_UNROLLED_OPERATIONS / tripCount) {
        loop.emitError() << "full unrolling would create more than "
                         << MAX_UNROLLED_OPERATIONS << " operations";
        signalPassFailure();
        return;
      }
      if (failed(loopUnrollFull(loop))) {
        loop.emitError("failed to fully unroll a static counted loop");
        signalPassFailure();
        return;
      }
    }
  }
};

class ResidualLegality final {
public:
  explicit ResidualLegality(const mqt::TargetEnvAttr environmentIn)
      : environment(environmentIn) {}

  [[nodiscard]] bool require(Operation* operation, const ProgramFeature feature,
                             const uint64_t value = 0U) const {
    const llvm::StringRef name = programFeatureName(feature);
    if (environment.supports(name, value)) {
      return true;
    }
    auto diagnostic = operation->emitError()
                      << "selected payload does not support capability '"
                      << name;
    diagnostic << "'";
    if (value != 0U) {
      diagnostic << " with value " << value;
    }
    if (!environment.getOptionalFeaturesKnown()) {
      diagnostic << "; optional capability metadata is unknown";
    }
    return false;
  }

  [[nodiscard]] bool verifyComputation(Operation* operation) const {
    if (operation->hasTrait<OpTrait::ConstantLike>()) {
      return true;
    }
    bool legal = true;
    const auto verifyType = [&](const Type type) {
      if (const auto integer = dyn_cast<IntegerType>(type)) {
        legal &= integer.getWidth() == 1U
                     ? require(operation, ProgramFeature::BooleanComputation)
                     : require(operation, ProgramFeature::IntegerComputation,
                               integer.getWidth());
      } else if (auto floating = dyn_cast<FloatType>(type)) {
        legal &= require(operation, ProgramFeature::FloatComputation,
                         floating.getWidth());
      }
    };
    llvm::for_each(operation->getOperandTypes(), verifyType);
    llvm::for_each(operation->getResultTypes(), verifyType);
    return legal;
  }

private:
  mqt::TargetEnvAttr environment;
};

struct LowerIndexSwitchToIf final : OpConversionPattern<qco::IndexSwitchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(qco::IndexSwitchOp operation, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    SmallVector<Region*> cases;
    cases.reserve(operation.getNumCases());
    llvm::transform(operation.getCaseRegions(), std::back_inserter(cases),
                    [](Region& region) { return &region; });
    Region* const defaultRegion = &operation.getDefaultRegion();

    const auto build = [&](auto&& self, const size_t index,
                           const ValueRange targets) -> qco::IfOp {
      auto constant = arith::ConstantIndexOp::create(
          rewriter, operation.getLoc(), operation.getCases()[index]);
      auto condition = arith::CmpIOp::create(
          rewriter, operation.getLoc(), arith::CmpIPredicate::eq,
          adaptor.getArg(), constant.getResult());
      auto ifOp = qco::IfOp::create(rewriter, operation.getLoc(),
                                    operation.getClassicalResults().getTypes(),
                                    operation.getLinearResults().getTypes(),
                                    condition, targets);
      rewriter.inlineRegionBefore(*cases[index], ifOp.getThenRegion(),
                                  ifOp.getThenRegion().end());
      if (index + 1 == cases.size()) {
        rewriter.inlineRegionBefore(*defaultRegion, ifOp.getElseRegion(),
                                    ifOp.getElseRegion().end());
        return ifOp;
      }

      Block& elseBlock = ifOp.getElseRegion().emplaceBlock();
      elseBlock.addArguments(targets.getTypes(),
                             SmallVector(targets.size(), operation.getLoc()));
      const OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(&elseBlock);
      qco::IfOp nested = self(self, index + 1, elseBlock.getArguments());
      qco::YieldOp::create(rewriter, operation.getLoc(), nested.getResults());
      return ifOp;
    };

    if (cases.empty()) {
      return failure();
    }
    qco::IfOp replacement = build(build, 0, adaptor.getTargets());
    rewriter.replaceOp(operation, replacement.getResults());
    return success();
  }
};

[[nodiscard]] bool isQuantumEvolution(Operation* operation) {
  return isa<qco::UnitaryOpInterface, qco::ResetOp>(operation);
}

struct VerifyTargetEnvironmentPass final
    : PassWrapper<VerifyTargetEnvironmentPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyTargetEnvironmentPass)

  void runOnOperation() override {
    const auto environment = getOperation()->getAttrOfType<mqt::TargetEnvAttr>(
        mqt::TargetEnvAttr::getOperationAttributeName());
    if (!environment) {
      getOperation().emitError("target compilation requires mqt.target_env");
      signalPassFailure();
      return;
    }
    func::FuncOp entryPoint = mqt::getEntryPoint(getOperation());
    if (!entryPoint) {
      getOperation().emitError("no program entry point found");
      signalPassFailure();
      return;
    }

    CallGraph callGraph(getOperation());
    SmallVector<CallGraphNode*> worklist{
        callGraph.lookupNode(&entryPoint.getBody())};
    SmallPtrSet<Operation*, 8> reachable;
    while (!worklist.empty()) {
      CallGraphNode* node = worklist.pop_back_val();
      if (node == nullptr || node->isExternal()) {
        continue;
      }
      Operation* callable = node->getCallableRegion()->getParentOp();
      if (!reachable.insert(callable).second) {
        continue;
      }
      for (const CallGraphNode::Edge& edge : *node) {
        worklist.emplace_back(edge.getTarget());
      }
    }

    ResidualLegality legality(environment);
    bool measurementsLegal = true;
    for (Operation* callable : reachable) {
      DominanceInfo dominance(callable);
      SmallVector<Operation*> quantumEvolution;
      callable->walk([&](Operation* operation) {
        if (isQuantumEvolution(operation)) {
          quantumEvolution.emplace_back(operation);
        }
      });
      callable->walk([&](qco::MeasureOp measurement) {
        if (llvm::any_of(quantumEvolution, [&](Operation* operation) {
              return dominance.properlyDominates(measurement, operation);
            })) {
          measurementsLegal &= legality.require(
              measurement, ProgramFeature::MidCircuitMeasurement);
        }
        for (Operation* user : measurement.getResult().getUsers()) {
          if (!isa<cbit::StoreOp, func::ReturnOp>(user)) {
            measurementsLegal &=
                legality.require(user, ProgramFeature::MeasurementResultUse);
            break;
          }
        }
        auto wire = qco::WireIterator(measurement.getQubitOut());
        for (++wire; wire != std::default_sentinel; ++wire) {
          if (isQuantumEvolution(*wire)) {
            measurementsLegal &=
                legality.require(*wire, ProgramFeature::MeasuredQubitReuse);
            break;
          }
        }
      });
    }
    if (!measurementsLegal) {
      signalPassFailure();
      return;
    }

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerIndexSwitchToIf>(&getContext());
    target.addDynamicallyLegalOp<qco::IfOp, scf::IfOp>(
        [&](Operation* operation) {
          return legality.require(operation, ProgramFeature::ForwardBranching);
        });
    target.addDynamicallyLegalOp<scf::ForOp>([&](Operation* operation) {
      return legality.require(operation, ProgramFeature::CountedIteration);
    });
    target.addDynamicallyLegalOp<scf::WhileOp>([&](Operation* operation) {
      return legality.require(operation, ProgramFeature::ConditionalLoop);
    });
    target.addDynamicallyLegalOp<qco::IndexSwitchOp>([&](Operation* operation) {
      if (environment.supports("forward-branching") &&
          !environment.supports("multiway-branching")) {
        return false;
      }
      return legality.require(operation, ProgramFeature::MultiwayBranching);
    });
    target.addDynamicallyLegalOp<scf::IndexSwitchOp>([&](Operation* operation) {
      return legality.require(operation, ProgramFeature::MultiwayBranching);
    });
    target.markUnknownOpDynamicallyLegal([&](Operation* operation) {
      const std::string_view dialect =
          operation->getName().getDialectNamespace();
      if (dialect == "arith" || dialect == "math") {
        return legality.verifyComputation(operation);
      }
      if (isa<BranchOpInterface>(operation)) {
        return legality.require(operation, ProgramFeature::ForwardBranching);
      }
      if (isa<RegionBranchOpInterface>(operation) &&
          !isa<scf::ExecuteRegionOp>(operation)) {
        operation->emitError(
            "unmodeled structured control remains after cleanup");
        return false;
      }
      return true;
    });
    target.addLegalOp<func::FuncOp>();
    target.markOpRecursivelyLegal<func::FuncOp>([&](func::FuncOp function) {
      return !reachable.contains(function.getOperation());
    });
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

void populateTargetCompilationPipeline(OpPassManager& pm,
                                       const CompilerTarget& target) {
  populateTargetCompilationPipeline(pm, target, ProgramFormat::QCOOptimized);
}

void populateTargetCompilationPipeline(OpPassManager& pm,
                                       const CompilerTarget& target,
                                       const ProgramFormat format) {
  const auto descriptor = payloadForProgramFormat(format);
  assert(descriptor && "target compilation format must have a payload");
  populateTargetCompilationPipeline(pm, target, format, *descriptor);
}

void populateTargetCompilationPipeline(OpPassManager& pm,
                                       const CompilerTarget& target,
                                       const ProgramFormat format,
                                       const PayloadDescriptor& descriptor) {
  assert(isTargetCompilationFormat(format));
  pm.addPass(std::make_unique<AttachTargetEnvironmentPass>(target, descriptor));
  pm.addPass(createSymbolDCEPass());
  pm.addPass(createSCCPPass());
  populateQCOCleanupPipeline(pm);
  pm.addPass(std::make_unique<UnrollUnsupportedCountedIterationPass>());
  pm.addPass(createSCCPPass());
  populateQCOCleanupPipeline(pm);
  pm.addPass(std::make_unique<VerifyTargetEnvironmentPass>());
  populateDecomposeMultiControlledPipeline(pm, 3);
  populateDefaultQCOOptimizationPipeline(pm);
  pm.addPass(qco::createFuseTwoQubitGates());
  pm.addPass(qco::createMappingPass(qco::MappingPassOptions{}));
  populateQCOCleanupPipeline(pm);
  pm.addPass(qco::createTargetNativeSynthesis());
  pm.addPass(createCSEPass());
  pm.addPass(createRemoveDeadValuesPass());
  pm.addPass(qco::createVerifyTargetConformance());
}

} // namespace mlir
