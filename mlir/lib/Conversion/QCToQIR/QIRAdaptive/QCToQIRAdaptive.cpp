/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/QCToQIR/QIRAdaptive/QCToQIRAdaptive.h"

#include "mlir/Conversion/QCToQIR/Common/CommonQIR.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QIR/Utils/QIRUtils.h"

#include <llvm/Support/ErrorHandling.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir {

using namespace qc;
using namespace qir;

#define GEN_PASS_DEF_QCTOQIRADAPTIVE
#include "mlir/Conversion/QCToQIR/QIRAdaptive/QCToQIRAdaptive.h.inc"

namespace {
/**
 * @brief Converts memref.alloc to QIR qubit-array allocation
 *
 * @par Example:
 * ```mlir
 * %memref = memref.alloc() : memref<3x!qc.qubit>
 * ```
 * becomes:
 * ```mlir
 * %zero = llvm.mlir.zero : !llvm.ptr
 * %alloca = llvm.alloca %c3 x !llvm.ptr : (i64) -> !llvm.ptr
 * llvm.call @"@__quantum__rt__qubit_array_allocate"(%c3, %alloca, %zero) :
 * (i64, !llvm.ptr, !llvm.ptr) -> ()
 * ```
 */
struct ConvertMemRefAllocOp final
    : StatefulOpConversionPattern<memref::AllocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto shape = op.getType().getShape();
    if (shape.size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "Only one-dimensional registers are supported");
    }
    if (failed(getState().ensureAllocationMode(AllocationMode::Dynamic,
                                               op.getOperation()))) {
      return failure();
    }

    auto& state = getState();
    state.useDynamicQubit = true;
    state.useArrays = true;

    auto* ctx = getContext();
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto fnSig =
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                    {rewriter.getI64Type(), ptrType, ptrType});
    auto fnDec = getOrCreateFunctionDeclaration(rewriter, op,
                                                QIR_QUBIT_ARRAY_ALLOC, fnSig);

    Value size;
    if (shape[0] == ShapedType::kDynamic) {
      size = adaptor.getDynamicSizes()[0];
    } else {
      size = LLVM::ConstantOp::create(
                 rewriter, op.getLoc(),
                 rewriter.getI64IntegerAttr(static_cast<int64_t>(shape[0])))
                 .getResult();
    }
    state.memrefSizes.try_emplace(op.getMemref(), size);

    auto array =
        LLVM::AllocaOp::create(rewriter, op.getLoc(), ptrType, ptrType, size);
    auto zero = LLVM::ZeroOp::create(rewriter, op.getLoc(), ptrType);
    LLVM::CallOp::create(rewriter, op.getLoc(), fnDec,
                         ValueRange{size, array.getResult(), zero.getResult()});

    rewriter.replaceOp(op, array.getResult());

    return success();
  }
};

/**
 * @brief Converts memref.load to llvm.load
 *
 * @par Example:
 * ```mlir
 * %q = memref.load %memref[%c1] : memref<3x!qc.qubit>
 * ```
 * is converted to
 * ```mlir
 * %ptr = llvm.getelementptr %alloca[c1] : !llvm.ptr -> !llvm.ptr
 * %q = llvm.load %ptr : !llvm.ptr -> !llvm.ptr
 * ```
 */
struct ConvertMemRefLoadOp final : StatefulOpConversionPattern<memref::LoadOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto shape = op.getMemref().getType().getShape();
    if (shape.size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "Only one-dimensional registers are supported");
    }

    auto* ctx = getContext();
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto array = adaptor.getMemref();
    auto index = adaptor.getIndices()[0];
    auto gep = LLVM::GEPOp::create(rewriter, op.getLoc(), ptrType, ptrType,
                                   array, index);
    auto load =
        LLVM::LoadOp::create(rewriter, op.getLoc(), ptrType, gep.getResult());

    rewriter.replaceOp(op, load.getResult());

    return success();
  }
};

/**
 * @brief Converts memref.dealloc to QIR qubit-array release
 *
 * @par Example:
 * ```mlir
 * memref.dealloc %memref : memref<3x!qc.qubit>
 * ```
 * becomes:
 * ```mlir
 * llvm.call @"@__quantum__rt__qubit_array_release"(%c3, %alloca) : (i64,
 * !llvm.ptr) -> ()
 * ```
 */
struct ConvertMemRefDeallocOp final
    : StatefulOpConversionPattern<memref::DeallocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto shape = op.getMemref().getType().getShape();
    if (shape.size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "Only one-dimensional registers are supported");
    }

    auto& state = getState();
    auto* ctx = getContext();
    auto i64Type = rewriter.getI64Type();
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    // Save current insertion point
    const OpBuilder::InsertionGuard guard(rewriter);

    // Release resources in output block
    rewriter.setInsertionPoint(state.outputBlock->getTerminator());

    auto fnSig = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                             {i64Type, ptrType});
    auto fnDec = getOrCreateFunctionDeclaration(rewriter, op,
                                                QIR_QUBIT_ARRAY_RELEASE, fnSig);

    auto size = state.memrefSizes.lookup(op.getMemref());
    assert(size != nullptr && "Size not found");

    // Create the release call
    LLVM::CallOp::create(rewriter, op.getLoc(), fnDec,
                         ValueRange{size, adaptor.getMemref()});
    rewriter.eraseOp(op);

    return success();
  }
};

/**
 * @brief Converts qc.alloc to QIR qubit allocation
 *
 * @par Example:
 * ```mlir
 * %q = qc.alloc : !qc.qubit
 * ```
 * becomes:
 * ```mlir
 * %zero = llvm.mlir.zero : !llvm.ptr
 * %q = llvm.call @"@__quantum__rt__qubit_allocate"(%zero) : !llvm.ptr ->
 * !llvm.ptr
 * ```
 */
struct ConvertQCAllocOp final : StatefulOpConversionPattern<AllocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    if (failed(state.ensureAllocationMode(AllocationMode::Dynamic,
                                          op.getOperation()))) {
      return failure();
    }
    state.useDynamicQubit = true;

    auto* ctx = getContext();
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto fnSig = LLVM::LLVMFunctionType::get(ptrType, {ptrType});
    auto fnDec =
        getOrCreateFunctionDeclaration(rewriter, op, QIR_QUBIT_ALLOC, fnSig);

    auto zero = LLVM::ZeroOp::create(rewriter, op.getLoc(), ptrType);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDec, zero.getResult());

    return success();
  }
};

/**
 * @brief Converts qc.dealloc to QIR qubit release
 *
 * @par Example:
 * ```mlir
 * qc.dealloc %q : !qc.qubit
 * ```
 * becomes:
 * ```mlir
 * llvm.call @"@__quantum__rt__qubit_release"(%q) : !llvm.ptr -> ()
 * ```
 */
struct ConvertQCDeallocOp final : StatefulOpConversionPattern<DeallocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto* ctx = getContext();
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    // Save current insertion point
    const OpBuilder::InsertionGuard guard(rewriter);

    // Release resources in output block
    rewriter.setInsertionPoint(state.outputBlock->getTerminator());

    auto fnSig =
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), {ptrType});
    auto fnDec =
        getOrCreateFunctionDeclaration(rewriter, op, QIR_QUBIT_RELEASE, fnSig);

    LLVM::CallOp::create(rewriter, op.getLoc(), fnDec, adaptor.getQubit());
    rewriter.eraseOp(op);

    return success();
  }
};

/**
 * @brief Converts qc.reset operation to QIR reset
 *
 * @details
 * Converts qubit reset to a call to the QIR __quantum__qis__reset__body
 * function, which resets a qubit to the |0⟩ state.
 *
 * @par Example:
 * ```mlir
 * qc.reset %q : !qc.qubit
 * ```
 * becomes:
 * ```mlir
 * llvm.call @__quantum__qis__reset__body(%q) : !llvm.ptr -> ()
 * ```
 */
struct ConvertQCResetOp final : StatefulOpConversionPattern<ResetOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(ResetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto* ctx = getContext();

    // Declare QIR function
    const auto fnSignature = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx), LLVM::LLVMPointerType::get(ctx));
    const auto fnDecl =
        getOrCreateFunctionDeclaration(rewriter, op, QIR_RESET, fnSignature);

    // Replace operation with CallOp
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl,
                                              adaptor.getOperands());
    return success();
  }
};

/**
 * @brief Converts qc.measure to QIR measurement
 *
 * @details
 * For measurements with register information, a result array is allocated and
 * all result pointers are loaded.
 *
 * For measurements without register information, a static result pointer is
 * used.
 * If the operation has an user, a read result call operation is created to
 * convert the result !llvm.ptr to an i1 value.
 *
 * @par Example (with register):
 * ```mlir
 * %result = qc.measure("c", 2, 0) %q : !qc.qubit -> i1
 * ```
 * becomes:
 * ```mlir
 * // In entry block:
 * %zero = llvm.mlir.zero : !llvm.ptr
 * %alloca = llvm.alloca %c2 x !llvm.ptr : (i64) -> !llvm.ptr
 * llvm.call @"@__quantum__rt__result_array_allocate"(%c2, %alloca, %zero) :
 * (i64, !llvm.ptr, !llvm.ptr) -> ()
 * %r = llvm.load %alloca : !llvm.ptr -> !llvm.ptr
 *
 * // In measurements block:
 * llvm.call @__quantum__qis__mz__body(%q, %r) : (!llvm.ptr, !llvm.ptr) -> ()
 * ```
 */
struct ConvertQCMeasureOp final : StatefulOpConversionPattern<MeasureOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(MeasureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    state.useDynamicResult = true;

    auto& resultArrays = state.resultArrays;
    auto& loadedResults = state.loadedResults;
    auto& resultPtrs = state.resultPtrs;

    auto* ctx = getContext();
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    // Save current insertion point
    auto savedInsertionPoint = rewriter.saveInsertionPoint();

    // Insert allocations and constants in entry block
    rewriter.setInsertionPoint(state.entryBlock->getTerminator());

    // Get result pointer
    Value result;
    if (op.getRegisterName() && op.getRegisterSize() && op.getRegisterIndex()) {
      state.useArrays = true;
      const auto registerName = op.getRegisterName().value();
      const auto registerSize =
          static_cast<int64_t>(op.getRegisterSize().value());
      const auto registerIndex =
          static_cast<int64_t>(op.getRegisterIndex().value());

      // Create result register if it does not exist yet
      if (!resultArrays.contains(registerName)) {
        auto fnSig = LLVM::LLVMFunctionType::get(
            LLVM::LLVMVoidType::get(ctx),
            {rewriter.getI64Type(), ptrType, ptrType});
        auto fnDec = getOrCreateFunctionDeclaration(
            rewriter, op, QIR_RESULT_ARRAY_ALLOC, fnSig);

        auto size =
            LLVM::ConstantOp::create(rewriter, op.getLoc(),
                                     rewriter.getI64IntegerAttr(registerSize))
                .getResult();
        auto array = LLVM::AllocaOp::create(rewriter, op.getLoc(), ptrType,
                                            ptrType, size);
        auto zero = LLVM::ZeroOp::create(rewriter, op.getLoc(), ptrType);
        LLVM::CallOp::create(
            rewriter, op.getLoc(), fnDec,
            ValueRange{size, array.getResult(), zero.getResult()});
        resultArrays.try_emplace(registerName, array.getResult());

        for (int64_t i = 0; i < registerSize; ++i) {
          auto gep = LLVM::GEPOp::create(
              rewriter, op.getLoc(), ptrType, ptrType, array.getResult(),
              ValueRange{LLVM::ConstantOp::create(
                  rewriter, op.getLoc(), rewriter.getI64IntegerAttr(i))});
          auto load = LLVM::LoadOp::create(rewriter, op.getLoc(), ptrType,
                                           gep.getResult());
          loadedResults.try_emplace({state.stringSaver.save(registerName), i},
                                    load.getResult());
        }
      }

      result = loadedResults.at({registerName, registerIndex});
    } else {
      rewriter.setInsertionPoint(state.entryBlock->getTerminator());
      result = createPointerFromIndex(rewriter, op.getLoc(), resultPtrs.size());
      resultPtrs.try_emplace(resultPtrs.size(), result);
      state.numResults++;
    }

    rewriter.restoreInsertionPoint(savedInsertionPoint);

    // Create measure call
    auto fnSig = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                             {ptrType, ptrType});
    auto fnDec =
        getOrCreateFunctionDeclaration(rewriter, op, QIR_MEASURE, fnSig);

    LLVM::CallOp::create(rewriter, op.getLoc(), fnDec,
                         ValueRange{adaptor.getQubit(), result});

    // Creates a readResult operation if the measure operation has any users
    if (op.getResult().use_empty()) {
      rewriter.eraseOp(op);
    } else {
      // Create read_result call
      auto fnSig = LLVM::LLVMFunctionType::get(rewriter.getI1Type(), {ptrType});
      auto fnDec =
          getOrCreateFunctionDeclaration(rewriter, op, QIR_READ_RESULT, fnSig);

      auto readResult =
          LLVM::CallOp::create(rewriter, op.getLoc(), fnDec, result);
      rewriter.replaceOp(op, readResult.getResult());
    }
    return success();
  }
};
} // namespace

/**
 * @brief Populates conversion patterns for QC-to-QIR-Adaptive lowering.
 */
static void populateQCToQIRAdaptivePatterns(RewritePatternSet& patterns,
                                            QCToQIRTypeConverter& typeConverter,
                                            MLIRContext* ctx,
                                            LoweringState& state) {
  patterns
      .add<ConvertMemRefAllocOp, ConvertMemRefLoadOp, ConvertMemRefDeallocOp>(
          typeConverter, ctx, &state);
  patterns.add<ConvertQCAllocOp>(typeConverter, ctx, &state);
  patterns.add<ConvertQCDeallocOp>(typeConverter, ctx, &state);
  patterns.add<ConvertQCMeasureOp>(typeConverter, ctx, &state);
  patterns.add<ConvertQCResetOp>(typeConverter, ctx, &state);
  populateQCToQIRPatterns(patterns, typeConverter, ctx, state);
}

namespace {

/**
 * @brief Pass for converting QC dialect operations to QIR Adaptive Profile
 *
 * @details
 * Converts QC dialect quantum operations to QIR by lowering them to LLVM
 * dialect operations that call QIR runtime functions.
 *
 * Conversion stages:
 * 1. Convert scf dialect to cf
 * 2. Cpmvert func dialect to LLVM
 * 3. Ensure proper block structure for QIR Adaptive Profile
 * 4. Add QIR initialization call
 * 5. Convert QC and memref operations to QIR calls
 * 6. Set QIR metadata attributes
 * 7. Convert arith and cf dialects to LLVM
 * 8. Reconcile unrealized casts
 *
 */
struct QCToQIRAdaptive final : impl::QCToQIRAdaptiveBase<QCToQIRAdaptive> {
  using QCToQIRAdaptiveBase::QCToQIRAdaptiveBase;

  /**
   * @brief Ensures proper block structure for QIR Adaptive Profile
   *
   * @details
   * The Adaptive Profile requires an entry block and an output block with an
   * arbitrary number of blocks between them.
   * 1. **Entry block**: Contains constant operations and initialization
   * 2. **Intermediate blocks**: Original function structure containing quantum
   * operations
   * 3. **Output block**: Contains output recording calls and qubit release
   * calls
   *
   * @param main The main LLVM function to restructure
   * @param state The LoweringState of the conversion pass
   */
  static void ensureBlocks(LLVM::LLVMFuncOp& main, LoweringState& state) {
    OpBuilder builder(main.getBody());
    auto* firstBlock = &main.front();
    auto* lastBlock = &main.back();

    auto* entryBlock = builder.createBlock(&main.getBody());
    main.getBlocks().splice(Region::iterator(firstBlock), main.getBlocks(),
                            entryBlock);
    Block* finalBlock = builder.createBlock(&main.getBody());
    state.entryBlock = entryBlock;

    state.outputBlock = finalBlock;
    builder.setInsertionPointToEnd(entryBlock);
    LLVM::BrOp::create(builder, main->getLoc(), firstBlock);
    auto* terminatorOp = lastBlock->getTerminator();
    terminatorOp->moveBefore(finalBlock, finalBlock->end());

    builder.setInsertionPointToEnd(lastBlock);
    LLVM::BrOp::create(builder, main->getLoc(), finalBlock);

    // Move up all constants to the beginning
    auto& entryOps = entryBlock->getOperations();
    for (auto& block : main.getBlocks()) {
      if (&block == entryBlock || &block == finalBlock) {
        continue;
      }
      for (auto it = block.begin(); it != block.end();) {
        auto& op = *it++;
        if (op.hasTrait<OpTrait::ConstantLike>()) {
          entryOps.splice(entryBlock->getTerminator()->getIterator(),
                          block.getOperations(), op.getIterator());
        }
      }
    }
  }

  /**
   * @brief Releases all result pointers and arrays in the output block
   */
  static void releaseResults(LLVM::LLVMFuncOp& main, MLIRContext* ctx,
                             LoweringState* state) {
    OpBuilder builder(ctx);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);
    auto voidType = LLVM::LLVMVoidType::get(ctx);

    builder.setInsertionPoint(state->outputBlock->getTerminator());

    for (auto& [_, ptr] : state->resultPtrs) {
      auto sig = LLVM::LLVMFunctionType::get(voidType, {ptrType});
      auto dec = getOrCreateFunctionDeclaration(builder, main,
                                                QIR_RESULT_RELEASE, sig);
      LLVM::CallOp::create(builder, main->getLoc(), dec, ptr);
    }

    for (auto& [_, array] : state->resultArrays) {
      auto sig = LLVM::LLVMFunctionType::get(voidType,
                                             {builder.getI64Type(), ptrType});
      auto dec = getOrCreateFunctionDeclaration(builder, main,
                                                QIR_RESULT_ARRAY_RELEASE, sig);
      auto size = array.getDefiningOp<LLVM::AllocaOp>().getArraySize();
      LLVM::CallOp::create(builder, main->getLoc(), dec,
                           ValueRange{size, array});
    }
  }

  /**
   * @brief Iterates through the module to find any scf.while or scf.for
   * operation to set the appropriate flags before they are converted to cf
   * operations.
   */
  static void setSCFFlags(Operation* op, LoweringState* state) {
    op->walk(
        [&](scf::WhileOp) { state->useConditionalLoopTermination = true; });
    op->walk([&](scf::ForOp) { state->useIteration = true; });
  }

protected:
  /**
   * @brief Executes the QC to QIR conversion pass
   *
   * @details
   * Performs the conversion in seven stages:
   *
   * **Stage 1: scf to cf**
   * Convert scf dialect operation to cf dialect equivalents.
   *
   * **Stage 2: func to LLVM**
   * Convert func dialect operations (main function) to LLVM dialect
   * equivalents.
   *
   * **Stage 3: Block structure**
   * Create proper block structure for QIR Adaptive Profile (entry, intermediate
   * blocks, output).
   *
   * **Stage 4: Initialization**
   * Insert the `__quantum__rt__initialize` call.
   *
   * **Stage 5: QC and memref to LLVM**
   * Convert QC dialect operations and memref operations to QIR calls and add
   * output recording to the output block.
   *
   * **Stage 6: QIR Attributes**
   * Add QIR Profile metadata to the main function, including qubit/result
   * counts and version information.
   *
   * **Stage 7: Standard dialects to LLVM**
   * Convert arith and control flow dialects to LLVM (for index arithmetic and
   * function control flow).
   *
   * **Stage 8: Reconcile casts**
   * Clean up any unrealized cast operations introduced during type conversion.
   */
  void runOnOperation() override {
    MLIRContext* ctx = &getContext();
    auto* moduleOp = getOperation();
    ConversionTarget target(*ctx);
    QCToQIRTypeConverter typeConverter(ctx);
    LoweringState state;
    state.useAdaptive = true;

    setSCFFlags(moduleOp, &state);

    target.addLegalDialect<LLVM::LLVMDialect>();

    // Stage 1: Convert scf dialect to cf
    {
      RewritePatternSet scfPatterns(ctx);
      target.addIllegalDialect<scf::SCFDialect>();
      target.addLegalDialect<cf::ControlFlowDialect>();
      target.addLegalDialect<arith::ArithDialect>();
      populateSCFToControlFlowConversionPatterns(scfPatterns);
      if (applyPartialConversion(moduleOp, target, std::move(scfPatterns))
              .failed()) {
        signalPassFailure();
        return;
      }
    }
    // Stage 2: Convert func dialect to LLVM
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

    auto main = getMainFunction(moduleOp);
    if (!main) {
      moduleOp->emitError("No main function with entry_point attribute found");
      signalPassFailure();
      return;
    }

    // Stage 3: Create block structure
    ensureBlocks(main, state);

    // Stage 4: Insert initialize call
    addInitialize(main, ctx, state);

    // Stage 5: Convert QC dialect to LLVM (QIR calls)
    {
      RewritePatternSet patterns(ctx);
      target.addIllegalDialect<QCDialect, memref::MemRefDialect>();

      populateQCToQIRAdaptivePatterns(patterns, typeConverter, ctx, state);

      if (applyPartialConversion(moduleOp, target, std::move(patterns))
              .failed()) {
        signalPassFailure();
        return;
      }

      addOutputRecording(main, ctx, state);
      releaseResults(main, ctx, &state);
    }

    // Stage 6: Set QIR metadata attributes
    setQIRAttributes(main, state);

    // Stage 7: Convert standard dialects to LLVM
    {
      RewritePatternSet stdPatterns(ctx);
      target.addIllegalDialect<arith::ArithDialect>();
      target.addIllegalDialect<cf::ControlFlowDialect>();

      cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                      stdPatterns);
      arith::populateArithToLLVMConversionPatterns(typeConverter, stdPatterns);

      if (applyPartialConversion(moduleOp, target, std::move(stdPatterns))
              .failed()) {
        signalPassFailure();
        return;
      }
    }

    // Stage 8: Reconcile unrealized casts
    PassManager passManager(ctx);
    passManager.addPass(createReconcileUnrealizedCastsPass());
    if (passManager.run(moduleOp).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir
