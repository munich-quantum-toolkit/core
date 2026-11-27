/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Support/Passes.h"

#include "mqt/Compiler/TargetEnvironment.h"
#include "mqt/Conversion/CBitToMemRef/CBitToMemRef.h"
#include "mqt/Dialect/MQT/IR/MQTAttributes.h"
#include "mqt/Dialect/MQT/IR/QubitLayout.h"
#include "mqt/Dialect/MQT/Transforms/Passes.h"
#include "mqt/Dialect/QC/Transforms/Passes.h"
#include "mqt/Dialect/QCO/Transforms/Passes.h"
#include "mqt/Dialect/QIR/Transforms/Passes.h"
#include "mqt/Dialect/QTensor/Transforms/Passes.h"
#include "mqt/Support/RandomSeed.h"

#include "mlir/Debug/ExecutionContext.h"
#include "mlir/Debug/Observers/ActionProfiler.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/Timing.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <system_error>

using namespace mlir;

LogicalResult
runWithPassManager(ModuleOp mod,
                   const function_ref<void(OpPassManager&)> populatePasses,
                   const StringRef errorMessage,
                   const CompilationOptions& options, bool preservesLayout) {
  PassManager pm(mod.getContext());
#ifdef NDEBUG
  pm.enableVerifier(false);
#endif
  populatePasses(pm);
  if (failed(runWithCompilationOptions(pm, mod, options, preservesLayout)) ||
      failed(verify(mod))) {
    return mod.emitError(errorMessage);
  }
  return success();
}

void registerMQTCompilerPasses() {
  static const auto REGISTERED = [] {
    registerTransformsPasses();
    registerConvertCBitToMemRef();
    qco::registerCombineControlledGates();
    qco::registerDecomposeMultiControlled();
    qco::registerElidePermutations();
    qco::registerFuseSingleQubitUnitaryRuns();
    qco::registerFuseTwoQubitGates();
    qco::registerHadamardLifting();
    qco::registerLegalizeControlFlow();
    qco::registerMeasurementLifting();
    qco::registerMergeSingleQubitRotationGates();
    qco::registerPauliTwirl2QGates();
    qco::registerMappingPass();
    qco::registerPlacementPass();
    qco::registerQuantumLoopUnroll();
    qco::registerRemoveDeadGates();
    qco::registerVerifyQCOProgram();
    qco::registerReplaceClassicalControls();
    qco::registerReuseQubits();
    qco::registerTargetNativeSynthesis();
    qco::registerUnrollLoopsForPayload();
    qco::registerVerifyTargetConformance();
    mqt::registerRemoveUnusedQubits();
    mqt::registerSimplifyQuantum();
    mqt::registerLowerQuantumModifiers();
    mqt::registerDropGlobalPhases();
    mqt::registerNormalizeGlobalPhases();
    mqt::registerLegalizeControlFlow();
    mqt::registerPromoteClassicalMemory();
    mqt::registerSimplifyClassicalControl();
    mqt::registerUnrollModifiers();
    qc::registerShrinkQubitRegistersPass();
    qtensor::registerShrinkQTensorToFitPass();
    qir::registerQIRPasses();
    PassPipelineRegistration<>("mqt-qco-default",
                               "Run the default MQT QCO optimization pipeline.",
                               populateDefaultQCOOptimizationPipeline);
    PassPipelineRegistration<>(
        "mqt-qubit-reuse",
        "Prepare a QCO program for qubit reuse and reuse eligible qubits.",
        populateQubitReusePipeline);
    // Keeps the shipped pass name working in textual pipelines.
    PassPipelineRegistration<>(
        "elide-terminal-swaps", "Run `elide-permutations{terminal-only=true}`.",
        [](OpPassManager& pm) {
          pm.addPass(qco::createElidePermutations({.terminalOnly = true}));
        });
    return true;
  }();
  static_cast<void>(REGISTERED);
}

void populateDefaultQCOOptimizationPipeline(OpPassManager& pm) {
  // Expose modifier bodies before fusion; simplify generated gates afterwards.
  pm.addPass(mqt::createSimplifyQuantum());
  pm.addPass(qco::createMergeSingleQubitRotationGates());
  pm.addPass(qco::createCombineControlledGates());
  pm.addPass(mqt::createSimplifyQuantum());
  pm.addPass(createCanonicalizerPass(
      CanonicalizerPassOptions{.cseBetweenIterations = true}));
  pm.addPass(mqt::createRemoveUnusedQubits());
}

void populateOpenQASM2PreparationPipeline(OpPassManager& pm,
                                          mqt::PayloadSpecAttr payload) {
  qco::UnrollLoopsForPayloadOptions options;
  llvm::raw_string_ostream stream(options.payloadSpec);
  stream << payload;
  pm.addPass(qco::createUnrollLoopsForPayload(options));
  pm.addPass(mqt::createUnrollModifiers());
  pm.addPass(mqt::createLowerQuantumModifiers());
}

void populateQIRPreparationPipeline(OpPassManager& pm,
                                    mqt::PayloadSpecAttr payload) {
  pm.addPass(createInlinerPass({}, [](OpPassManager&) {}));
  pm.addPass(mqt::createNormalizeGlobalPhases());
  pm.addPass(mqt::createUnrollModifiers());
  if (payload.getFormat().getProfile() == "base") {
    qco::UnrollLoopsForPayloadOptions options;
    llvm::raw_string_ostream stream(options.payloadSpec);
    stream << payload;
    pm.addPass(qco::createUnrollLoopsForPayload(options));
  }
}

void populateQubitReusePipeline(OpPassManager& pm) {
  pm.addPass(qco::createMeasurementLifting());
  pm.addPass(qco::createReplaceClassicalControls());
  pm.addPass(qco::createRemoveDeadGates());
  pm.addPass(qco::createReuseQubits());
}

void populateDecomposeMultiControlledPipeline(OpPassManager& pm,
                                              const uint64_t minQubits) {
  qco::DecomposeMultiControlledOptions options;
  options.minQubits = minQubits;
  pm.addPass(qco::createDecomposeMultiControlled(options));
}

llvm::Error parseModulePassPipeline(StringRef pipeline, OpPassManager& pm) {
  registerMQTCompilerPasses();
  auto passes = pipeline.trim();
  auto [anchor, body] = passes.split('(');
  if (anchor.rtrim() == ModuleOp::getOperationName()) {
    if (!body.consume_back(")")) {
      return llvm::createStringError(std::errc::invalid_argument,
                                     "unclosed builtin.module pass pipeline");
    }
    passes = body;
  }
  std::string diagnostics;
  llvm::raw_string_ostream errorStream(diagnostics);
  if (failed(parsePassPipeline(passes, pm, errorStream))) {
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "failed to parse pass pipeline '" + pipeline + "': " + diagnostics);
  }
  return llvm::Error::success();
}

LogicalResult runPassPipeline(ModuleOp mod, const StringRef pipeline,
                              const CompilationOptions& options,
                              const TargetEnvironment* environment,
                              bool preservesLayout) {
  PassManager pm(mod.getContext());
  if (auto error = parseModulePassPipeline(pipeline, pm)) {
    return mod.emitError(llvm::toString(std::move(error)));
  }
  if (environment != nullptr) {
    attachTargetEnvironment(mod, *environment);
  }
  return runWithCompilationOptions(pm, mod, options, preservesLayout);
}

namespace {

/// Forward output to a buffered stream and ignore flushes.
///
/// tracing::ActionProfiler flushes its stream after each event. With a file,
/// each flush is a system call. On large circuits, these calls make tracing
/// several times slower and add their cost to the traced durations. Remove this
/// class when the profiler no longer flushes after each event.
class FlushIgnoringStream final : public llvm::raw_ostream {
public:
  explicit FlushIgnoringStream(llvm::raw_ostream& os)
      : raw_ostream(/*unbuffered=*/true), os_(os) {}

private:
  void write_impl(const char* ptr, size_t size) override {
    os_.write(ptr, size);
  }
  [[nodiscard]] uint64_t current_pos() const override { return os_.tell(); }

  llvm::raw_ostream& os_;
};

} // namespace

llvm::Error
withActionTrace(MLIRContext& context, const CompilationOptions& options,
                function_ref<llvm::Error(const CompilationOptions&)> run) {
  if (!options.traceFile) {
    return run(options);
  }
  if (context.hasActionHandler()) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "tracing requires a context without an existing action handler");
  }
  const auto path = options.traceFile->string();
  std::error_code error;
  llvm::raw_fd_ostream stream(path, error);
  if (error) {
    return llvm::createStringError(error, "failed to open trace file '%s': %s",
                                   path.c_str(), error.message().c_str());
  }
  auto stepOptions = options;
  stepOptions.traceFile.reset();
  FlushIgnoringStream traceStream(stream);
  auto result = [&] {
    tracing::ActionProfiler profiler(traceStream);
    tracing::ExecutionContext executionContext;
    executionContext.registerObserver(&profiler);
    context.registerActionHandler(std::move(executionContext));
    const llvm::scope_exit clearHandler(
        [&] { context.registerActionHandler(nullptr); });
    return run(stepOptions);
  }();
  // raw_fd_ostream must not close stdout.
  if (path == "-") {
    stream.flush();
  } else {
    stream.close();
  }
  if (!stream.has_error()) {
    return result;
  }
  error = stream.error();
  stream.clear_error();
  return llvm::joinErrors(
      std::move(result),
      llvm::createStringError(error, "failed to write trace file '%s': %s",
                              path.c_str(), error.message().c_str()));
}

LogicalResult runWithCompilationOptions(PassManager& pm, ModuleOp moduleOp,
                                        const CompilationOptions& options,
                                        bool preservesLayout) {
  if (options.traceFile) {
    auto result = failure();
    if (auto error =
            withActionTrace(*moduleOp.getContext(), options,
                            [&](const CompilationOptions& passOptions) {
                              result = runWithCompilationOptions(
                                  pm, moduleOp, passOptions, preservesLayout);
                              return llvm::Error::success();
                            })) {
      moduleOp.emitError(llvm::toString(std::move(error)));
      return failure();
    }
    return result;
  }
  // Embedders need not register command-line options.
  if (llvm::cl::getRegisteredOptions().contains("mlir-print-ir-before-all") &&
      failed(applyPassManagerCLOptions(pm))) {
    return failure();
  }
  if (llvm::cl::getRegisteredOptions().contains("mlir-timing")) {
    applyDefaultTimingPassManagerCLOptions(pm);
  }
  if (!preservesLayout) {
    moduleOp->removeAttr(mqt::kLayoutAttr);
  }
  if (options.enableTiming) {
    pm.enableTiming();
  }
  if (options.enableStatistics) {
    pm.enableStatistics();
  }
  auto previous = moduleOp->getAttr(COMPILATION_SEED_ATTR);
  const auto restoreSeed = llvm::scope_exit([&] {
    if (previous) {
      moduleOp->setAttr(COMPILATION_SEED_ATTR, previous);
    } else {
      moduleOp->removeAttr(COMPILATION_SEED_ATTR);
    }
  });
  if (options.seed) {
    moduleOp->setAttr(
        COMPILATION_SEED_ATTR,
        Builder(moduleOp.getContext())
            .getI64IntegerAttr(std::bit_cast<int64_t>(*options.seed)));
  }
  return pm.run(moduleOp);
}

void populateQCStructuralCleanupPipeline(OpPassManager& pm) {
  pm.addPass(mqt::createNormalizeGlobalPhases());
  pm.addPass(createCanonicalizerPass(GreedyRewriteConfig{}
                                         .enableCSEBetweenIterations()
                                         .setUseTopDownTraversal()));
  pm.addPass(mqt::createRemoveUnusedQubits());
  pm.addPass(qc::createShrinkQubitRegistersPass());
  pm.addPass(createSymbolDCEPass());
}

void populateQCExportPipeline(OpPassManager& pm) {
  pm.addPass(mlir::mqt::createSimplifyClassicalControl());
  pm.addPass(mqt::createPromoteClassicalMemory());
  populateQCStructuralCleanupPipeline(pm);
}

void populateQCCleanupPipeline(OpPassManager& pm) {
  populateQCExportPipeline(pm);
  pm.addPass(createRemoveDeadValuesPass());
}

void populateQCOCleanupPipeline(OpPassManager& pm, bool removeUnusedValues) {
  pm.addPass(mlir::mqt::createSimplifyClassicalControl());
  pm.addPass(mqt::createPromoteClassicalMemory());
  pm.addPass(mqt::createNormalizeGlobalPhases());
  pm.addPass(createCanonicalizerPass(
      CanonicalizerPassOptions{.cseBetweenIterations = true}));
  pm.addPass(mqt::createRemoveUnusedQubits());
  if (removeUnusedValues) {
    pm.addPass(qtensor::createShrinkQTensorToFitPass());
  }
  pm.addPass(createSymbolDCEPass());
  if (removeUnusedValues) {
    pm.addPass(createRemoveDeadValuesPass());
  }
}

void populateQIRCleanupPipeline(OpPassManager& pm, QIRProfile profile) {
  // Region simplification dominates large programs; qir-cleanup erases the
  // unreachable blocks that folded branches leave behind.
  GreedyRewriteConfig config;
  config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
  pm.addPass(createCanonicalizerPass(config));
  pm.addPass(createCSEPass());
  pm.addPass(createMem2Reg());
  pm.addPass(qir::createQIRCleanupPass());
  pm.addPass(qir::createQIRSetAttributesAndMetadata({profile}));
}

void populateJeffCleanupPipeline(OpPassManager& pm) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

[[nodiscard]] LogicalResult runQCCleanupPipeline(ModuleOp mod) {
  return runWithPassManager(mod, populateQCCleanupPipeline,
                            "Failed to run the QC cleanup pipeline.");
}

[[nodiscard]] LogicalResult runQCOCleanupPipeline(ModuleOp mod) {
  return runWithPassManager(
      mod, [](OpPassManager& pm) { populateQCOCleanupPipeline(pm); },
      "Failed to run the QCO cleanup pipeline.");
}

[[nodiscard]] LogicalResult runQIRCleanupPipeline(ModuleOp mod,
                                                  QIRProfile profile) {
  return runWithPassManager(
      mod, [&](OpPassManager& pm) { populateQIRCleanupPipeline(pm, profile); },
      "Failed to run the QIR cleanup pipeline.");
}

[[nodiscard]] LogicalResult runJeffCleanupPipeline(ModuleOp moduleOp) {
  return runWithPassManager(moduleOp, populateJeffCleanupPipeline,
                            "Failed to run the jeff cleanup pipeline.");
}
