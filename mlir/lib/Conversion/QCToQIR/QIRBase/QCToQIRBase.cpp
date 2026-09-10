/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Conversion/QCToQIR/QIRBase/QCToQIRBase.h"

#include "mqt/Conversion/QCToQIR/QIRCommon/QIRCommon.h"
#include "mqt/Dialect/CBit/IR/CBitDialect.h"
#include "mqt/Dialect/CBit/IR/CBitOps.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/MQT/Transforms/GlobalPhaseNormalization.h"
#include "mqt/Dialect/QC/IR/QCDialect.h"
#include "mqt/Dialect/QC/IR/QCOps.h"
#include "mqt/Dialect/QIR/QIRDefinitions.h"
#include "mqt/Dialect/QIR/Utils/QIRUtils.h"

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/DenseSet.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <utility>
#include <variant>

namespace mlir {

using namespace qc;
using namespace qir;

#define GEN_PASS_DEF_QCTOQIRBASE
#include "mqt/Conversion/QCToQIR/QIRBase/QCToQIRBase.h.inc"

/// Returns the result pointer the `qc::MeasureOp` @p op writes to, or
/// a null value if it does not write into a classical register.
static FailureOr<Value> resolveRegisterMeasurement(LoweringState& state,
                                                   Operation* op) {
  const auto it = state.cregMeasurements.find(op);
  if (it == state.cregMeasurements.end()) {
    return Value{};
  }
  auto [registerIndex, index] = it->second;
  const auto indexValue = getConstantIntValue(index);
  if (!indexValue) {
    op->emitError("QIR Base Profile requires constant classical-register "
                  "measurement indices");
    return failure();
  }
  const auto& results = state.cregs[registerIndex].results;
  if (*indexValue < 0 || static_cast<size_t>(*indexValue) >= results.size()) {
    op->emitError("classical-register measurement index is out of bounds");
    return failure();
  }
  return results[static_cast<size_t>(*indexValue)];
}

/// Validates canonical qubit pointers before moving measurements out of order.
static LogicalResult moveTerminalMeasurements(Block& body,
                                              Block& measurements) {
  DenseSet<Value> measuredQubits;
  SmallVector<LLVM::CallOp> measurementCalls;
  for (auto call : body.getOps<LLVM::CallOp>()) {
    if (!call.getCallee() ||
        !call.getCallee()->starts_with("__quantum__qis__")) {
      continue;
    }
    const bool isMeasurement = call.getCallee() == QIR_MEASURE;
    /// Measurement's second pointer identifies a result, not a qubit.
    auto operands = call.getOperands();
    if (isMeasurement) {
      operands = operands.take_front(1);
    }
    for (auto operand : operands) {
      if (measuredQubits.contains(operand)) {
        return call.emitError(
            "QIR Base Profile forbids using a qubit after measurement");
      }
    }
    if (isMeasurement) {
      measuredQubits.insert(call.getOperand(0));
      measurementCalls.push_back(call);
    }
  }
  for (auto call : measurementCalls) {
    call->moveBefore(measurements.getTerminator());
  }
  return success();
}

namespace {

/// Converts `cbit.alloc` to static result
/// pointers represented by `llvm.inttoptr` operations
///
/// Allocate static result pointers for each bit in a classical register.
struct ConvertCBitAllocOp final : StatefulOpConversionPattern<cbit::AllocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(cbit::AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    const auto it = state.cregIndices.find(op.getOperation());
    if (it == state.cregIndices.end()) {
      rewriter.eraseOp(op);
      return success();
    }
    auto& reg = state.cregs[it->second];
    const auto* size = std::get_if<int64_t>(&reg.size);
    if (size == nullptr) {
      op.emitError(
          "QIR Base Profile requires statically sized classical registers");
      return failure();
    }

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(state.entryBlock->getTerminator());
    reg.results.reserve(static_cast<size_t>(*size));
    const auto base = static_cast<int64_t>(state.scalarResults.size());
    for (int64_t i = 0; i < *size; ++i) {
      const auto index = base + i;
      auto result = createPointerFromIndex(rewriter, op.getLoc(), index);
      reg.results.push_back(result);
      // The results are recorded as part of the register
      state.scalarResults.try_emplace(
          index, qir::StaticResult{.pointer = result, .record = false});
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct RejectCBitLoadOp final : OpConversionPattern<cbit::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cbit::LoadOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& /*rewriter*/) const override {
    return op.emitError(
        "QIR Base Profile does not support classical-register loads");
  }
};

struct RejectCBitReadOp final : OpConversionPattern<cbit::ReadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cbit::ReadOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& /*rewriter*/) const override {
    return op.emitError(
        "QIR Base Profile does not support classical-register reads");
  }
};

struct RejectCBitWriteOp final : OpConversionPattern<cbit::WriteOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cbit::WriteOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& /*rewriter*/) const override {
    return op.emitError(
        "QIR Base Profile does not support classical-register writes");
  }
};

struct ConvertMemRefAllocOp final
    : StatefulOpConversionPattern<memref::AllocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    if (failed(getState().ensureAllocationMode(AllocationMode::Dynamic, op))) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

/// Converts a qubit-register `memref.load` to `llvm.inttoptr`
///
/// Converts a load operation to an LLVM pointer by creating a constant with the
/// next available static qubit index and converting it to a pointer. The
/// pointer is cached in the lowering state for reuse.
///
/// @par Example:
/// ```mlir
/// %q0 = memref.load %memref[%c0] : memref<3x!qc.qubit>
/// ```
/// is converted to
/// ```mlir
/// %c0 = llvm.mlir.constant(0 : i64) : i64
/// %q0 = llvm.inttoptr %c0 : i64 to !llvm.ptr
/// ```
struct ConvertMemRefLoadOp final : StatefulOpConversionPattern<memref::LoadOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto shape = op.getMemref().getType().getShape();
    if (shape.size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "Only one-dimensional registers are supported");
    }
    const auto index = getConstantIntValue(op.getIndices().front());
    if (!index || ShapedType::isDynamic(shape.front()) ||
        !op.getMemref().getDefiningOp<memref::AllocOp>()) {
      return op.emitError("QIR Base Profile requires constant indices into "
                          "statically allocated qubit registers");
    }
    if (*index < 0 || *index >= shape.front()) {
      return op.emitError("qubit-register index is out of bounds");
    }
    auto& qubit = state.staticRegisterQubits[{op.getMemref(), *index}];
    if (!qubit) {
      const OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(state.entryBlock->getTerminator());
      const auto id = static_cast<int64_t>(state.staticQubits.size());
      qubit = createPointerFromIndex(rewriter, op.getLoc(), id);
      state.staticQubits.try_emplace(id, qubit);
    }
    rewriter.replaceOp(op, qubit);

    return success();
  }
};

/// Erases memref.dealloc during the QIR Base Profile conversion
struct ConvertMemRefDeallocOp final
    : StatefulOpConversionPattern<memref::DeallocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::DeallocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

/// Converts qc.alloc to llvm.inttoptr
///
/// Converts a qubit allocation to an LLVM pointer by creating a constant
/// with the next available static qubit index and converting it to a pointer.
/// The pointer is cached in the lowering state for reuse.
///
/// @par Example:
/// ```mlir
/// %q = qc.alloc : !qc.qubit
/// ```
/// is converted to
/// ```mlir
/// %c0 = llvm.mlir.constant(0 : i64) : i64
/// %q0 = llvm.inttoptr %c0 : i64 to !llvm.ptr
/// ```
struct ConvertQCAllocOp final : StatefulOpConversionPattern<AllocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    if (failed(state.ensureAllocationMode(AllocationMode::Dynamic, op))) {
      return failure();
    }

    const OpBuilder::InsertionGuard guard(rewriter);

    rewriter.setInsertionPoint(state.entryBlock->getTerminator());

    const auto nqubits = state.staticQubits.size();
    auto qubit = createPointerFromIndex(rewriter, op.getLoc(),
                                        static_cast<int64_t>(nqubits));
    state.staticQubits.try_emplace(static_cast<int64_t>(nqubits), qubit);
    rewriter.replaceOp(op, qubit);

    return success();
  }
};

/// Erases qc.dealloc during the QIR Base Profile conversion
struct ConvertQCDeallocOp final : StatefulOpConversionPattern<DeallocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(DeallocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

/// Converts qc.measure to QIR measurement
///
/// For measurements with register information, a static result is used at
/// the given index + register offset. Otherwise a static result at
/// the next index is used.
///
/// @par Example (without register):
/// ```mlir
/// %result = qc.measure %q : !qc.qubit -> i1
/// ```
/// is converted to
/// ```mlir
/// llvm.call @__quantum__qis__mz__body(%q, %b) : (!llvm.ptr, !llvm.ptr) -> ()
/// ```
struct ConvertQCMeasureOp final : StatefulOpConversionPattern<MeasureOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(MeasureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();

    auto* ctx = getContext();
    auto ptrType = LLVM::LLVMPointerType::get(ctx);
    auto voidType = LLVM::LLVMVoidType::get(ctx);

    OpBuilder::InsertionGuard guard(rewriter);

    auto registerResult = resolveRegisterMeasurement(state, op.getOperation());
    if (failed(registerResult)) {
      return failure();
    }
    auto result = *registerResult;
    if (!result) {
      result = getResultPtr(state, op.getOperation(), rewriter, false);
    }

    /// Preserve instruction order until terminal measurements are verified.
    rewriter.setInsertionPoint(op);
    auto fnSig = LLVM::LLVMFunctionType::get(voidType, {ptrType, ptrType});
    auto fnDec =
        getOrCreateFunctionDeclaration(rewriter, op, QIR_MEASURE, fnSig);
    LLVM::CallOp::create(rewriter, op.getLoc(), fnDec,
                         ValueRange{adaptor.getQubit(), result});

    rewriter.eraseOp(op);

    return success();
  }
};
} // namespace

/// Populates conversion patterns for QC-to-QIR-Base lowering.
static void populateQCToQIRBasePatterns(RewritePatternSet& patterns,
                                        QCToQIRTypeConverter& typeConverter,
                                        MLIRContext* ctx,
                                        LoweringState& state) {
  populateQCToQIRPatterns(patterns, typeConverter, ctx, state);
  patterns.add<ConvertCBitAllocOp, ConvertMemRefAllocOp, ConvertMemRefLoadOp,
               ConvertMemRefDeallocOp, ConvertQCAllocOp, ConvertQCMeasureOp,
               ConvertQCDeallocOp>(typeConverter, ctx, &state);
  patterns.add<RejectCBitLoadOp, RejectCBitReadOp, RejectCBitWriteOp>(
      typeConverter, ctx);
}

namespace {
/// Lower supported QC operations to QIR Base runtime calls and LLVM IR.
/// QIR attributes and module flags are attached by the separate metadata pass.
struct QCToQIRBase final : impl::QCToQIRBaseBase<QCToQIRBase> {
  using QCToQIRBaseBase::QCToQIRBaseBase;

  void getDependentDialects(DialectRegistry& registry) const override {
    QCToQIRBaseBase::getDependentDialects(registry);
    registerQIRClassicalTensorDialects(registry);
  }

  /// Ensures proper block structure for QIR base profile
  ///
  /// The QIR base profile requires a specific 4-block structure:
  /// 1. **Entry block**: Contains constant operations and initialization
  /// 2. **Body block**: Contains reversible quantum operations (gates)
  /// 3. **Measurements block**: Contains irreversible operations (measure
  /// operations)
  /// 4. **Output block**: Contains output recording calls
  ///
  /// Blocks are connected with unconditional jumps (entry, body, measurements,
  /// output). This structure ensures proper QIR Base Profile semantics.
  ///
  /// @param main The main LLVM function to restructure
  static void ensureBlocks(LLVM::LLVMFuncOp& main, LoweringState& state) {
    // Get the existing block
    auto* bodyBlock = &main.front();
    OpBuilder builder(main.getBody());

    // Create the required blocks
    auto* entryBlock = builder.createBlock(&main.getBody());
    // Move the entry block before the body block
    main.getBlocks().splice(Region::iterator(bodyBlock), main.getBlocks(),
                            entryBlock);
    Block* measurementsBlock = builder.createBlock(&main.getBody());
    Block* outputBlock = builder.createBlock(&main.getBody());

    state.entryBlock = entryBlock;
    state.measurementsBlock = measurementsBlock;
    state.outputBlock = outputBlock;

    auto& bodyBlockOps = bodyBlock->getOperations();
    auto& outputBlockOps = outputBlock->getOperations();

    // Move operations to appropriate blocks
    for (auto it = bodyBlock->begin(); it != bodyBlock->end();) {
      // Ensure iterator remains valid after potential move
      if (auto& op = *it++; isa<LLVM::ReturnOp>(op)) {
        // Move return to output block
        outputBlockOps.splice(outputBlock->end(), bodyBlockOps,
                              Block::iterator(op));
      } else if (op.hasTrait<OpTrait::ConstantLike>()) {
        // Move allocations and constant-like operations to entry block
        entryBlock->getOperations().splice(entryBlock->end(), bodyBlockOps,
                                           Block::iterator(op));
      }
      // All other operations (gates, etc.) stay in body block
    }

    // Add unconditional jumps between blocks
    builder.setInsertionPointToEnd(entryBlock);
    LLVM::BrOp::create(builder, main->getLoc(), bodyBlock);

    builder.setInsertionPointToEnd(bodyBlock);
    LLVM::BrOp::create(builder, main->getLoc(), measurementsBlock);

    builder.setInsertionPointToEnd(measurementsBlock);
    LLVM::BrOp::create(builder, main->getLoc(), outputBlock);
  }

protected:
  void runOnOperation() override {
    MLIRContext* ctx = &getContext();
    auto moduleOp = getOperation();
    if (failed(mqt::verifyQuantumAllocations(moduleOp))) {
      signalPassFailure();
      return;
    }
    auto entryPoint = mqt::getEntryPoint(moduleOp);
    if (!entryPoint) {
      moduleOp->emitError("no main function with mqt.entry_point found");
      signalPassFailure();
      return;
    }
    if (!entryPoint.getBody().hasOneBlock()) {
      entryPoint.emitError(
          "QIR Base Profile requires a single-block entry function");
      signalPassFailure();
      return;
    }
    auto entryPointName = entryPoint.getSymNameAttr();
    if (failed(mqt::normalizeGlobalPhases(moduleOp))) {
      signalPassFailure();
      return;
    }
    /// Base Profile uses static resources, so slot ownership has no runtime
    /// effect. Remove its now-unread buffers with the standard memref utility.
    bool removedRelease = false;
    IRRewriter rewriter(ctx);
    moduleOp.walk([&](qc::DeallocRegisterOp op) {
      rewriter.eraseOp(op);
      removedRelease = true;
    });
    if (removedRelease) {
      memref::eraseDeadAllocAndStores(rewriter, moduleOp);
      PassManager cleanup(ctx);
      cleanup.addPass(createCanonicalizerPass());
      if (failed(cleanup.run(moduleOp))) {
        signalPassFailure();
        return;
      }
    }
    ConversionTarget target(*ctx);
    QCToQIRTypeConverter typeConverter(ctx);

    target.addLegalDialect<LLVM::LLVMDialect>();

    LoweringState state;

    // Stage 1.0: Prepare classical result registers
    if (failed(prepareClassicalResults(moduleOp, state))) {
      signalPassFailure();
      return;
    }

    // Stage 1.1: Convert func dialect to LLVM
    {
      RewritePatternSet funcPatterns(ctx);
      target.addIllegalDialect<func::FuncDialect>();
      populateFuncToLLVMConversionPatterns(typeConverter, funcPatterns);

      if (applyPartialConversion(moduleOp, target, std::move(funcPatterns))
              .failed()) {
        signalPassFailure();
        return;
      }
    }

    auto main = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(entryPointName);
    if (!main) {
      moduleOp->emitError("no main function with mqt.entry_point found");
      signalPassFailure();
      return;
    }
    main.setPassthroughAttr(
        ArrayAttr::get(ctx, {StringAttr::get(ctx, ::qir::ENTRY_POINT_ATTR)}));
    mqt::removeEntryPoint(main);

    // Stage 2: Create block structure
    ensureBlocks(main, state);

    // Stage 3: Insert initialize call
    addInitialize(main, ctx, state);

    // Stage 4: Convert QC dialect to LLVM (QIR calls)
    {
      RewritePatternSet patterns(ctx);
      target.addIllegalDialect<cbit::CBitDialect, QCDialect,
                               memref::MemRefDialect>();

      populateQCToQIRBasePatterns(patterns, typeConverter, ctx, state);

      if (applyPartialConversion(moduleOp, target, std::move(patterns))
              .failed()) {
        signalPassFailure();
        return;
      }

      auto& body = *std::next(main.getBody().begin());
      if (failed(moveTerminalMeasurements(body, *state.measurementsBlock))) {
        signalPassFailure();
        return;
      }
      addOutputRecording(main, ctx, state);
    }

    if (failed(finalizeQIRConversion(moduleOp, target, typeConverter))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir
