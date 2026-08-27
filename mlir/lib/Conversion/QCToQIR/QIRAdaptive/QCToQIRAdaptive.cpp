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

#include "mlir/Conversion/QCToQIR/QIRCommon/QIRCommon.h"
#include "mlir/Dialect/CBit/IR/CBitAttributes.h"
#include "mlir/Dialect/CBit/IR/CBitDialect.h"
#include "mlir/Dialect/CBit/IR/CBitOps.h"
#include "mlir/Dialect/MQT/Transforms/GlobalPhaseNormalization.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QIR/Utils/QIRUtils.h"

#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/MathToLLVM/MathToLLVM.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Region.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>

#include <cassert>
#include <cstdint>
#include <utility>

namespace mlir {

using namespace qc;
using namespace qir;

#define GEN_PASS_DEF_QCTOQIRADAPTIVE
#include "mlir/Conversion/QCToQIR/QIRAdaptive/QCToQIRAdaptive.h.inc"

/**
 * @brief Returns the result pointer the `qc::MeasureOp` @p op writes to, or
 * `nullptr` if it does not write into a classical register.
 */
static Value resolveRegisterMeasurement(LoweringState& state, Operation* op,
                                        ConversionPatternRewriter& rewriter) {
  const auto it = state.cregMeasurements.find(op);
  if (it == state.cregMeasurements.end()) {
    return nullptr;
  }
  auto [registerIndex, index] = it->second;
  auto& reg = state.cregs[registerIndex];
  assert(reg.array && "result array must be allocated");
  auto loc = op->getLoc();
  auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
  auto remappedIndex = rewriter.getRemappedValue(index);
  auto elementptr = LLVM::GEPOp::create(rewriter, loc, ptrType, ptrType,
                                        reg.array, ValueRange{remappedIndex})
                        .getResult();
  return LLVM::LoadOp::create(rewriter, loc, ptrType, elementptr).getResult();
}

/**
 * @brief Converts qubit-register `memref.alloc` to `llvm.alloca`
 */
static LogicalResult
convertQubitMemRefAllocOp(memref::AllocOp op, memref::AllocOp::Adaptor adaptor,
                          LoweringState& state,
                          ConversionPatternRewriter& rewriter) {
  if (failed(state.ensureAllocationMode(AllocationMode::Dynamic,
                                        op.getOperation()))) {
    return failure();
  }

  auto loc = op.getLoc();
  auto* ctx = op.getContext();
  auto ptrType = LLVM::LLVMPointerType::get(ctx);
  auto voidType = LLVM::LLVMVoidType::get(ctx);

  auto fnSig = LLVM::LLVMFunctionType::get(
      voidType, {rewriter.getI64Type(), ptrType, ptrType});
  auto fnDec = getOrCreateFunctionDeclaration(rewriter, op,
                                              QIR_QUBIT_ARRAY_ALLOC, fnSig);

  Value size;
  if (op.getType().getShape()[0] == ShapedType::kDynamic) {
    size = adaptor.getDynamicSizes()[0];
  } else {
    size = LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(),
                                    op.getType().getShape()[0])
               .getResult();
  }
  state.qregSizes.try_emplace(op.getMemref(), size);

  auto array =
      LLVM::AllocaOp::create(rewriter, loc, ptrType, ptrType, size).getResult();
  auto zero = LLVM::ZeroOp::create(rewriter, loc, ptrType).getResult();
  LLVM::CallOp::create(rewriter, loc, fnDec, ValueRange{size, array, zero});

  rewriter.replaceOp(op, array);
  return success();
}

namespace {

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

    auto loc = op.getLoc();
    auto* ctx = op.getContext();
    auto ptrType = LLVM::LLVMPointerType::get(ctx);
    auto voidType = LLVM::LLVMVoidType::get(ctx);

    const OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(state.entryBlock->getTerminator());

    auto size = LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(),
                                         op.getResult().getType().getWidth())
                    .getResult();

    if (!reg.record) {
      auto i1Type = rewriter.getI1Type();
      auto storage =
          LLVM::AllocaOp::create(rewriter, loc, ptrType, i1Type, size)
              .getResult();
      if (op.getInitialization() == cbit::Initialization::Zero) {
        auto zero = LLVM::ConstantOp::create(rewriter, loc, i1Type,
                                             rewriter.getBoolAttr(false));
        for (int64_t index = 0; index < op.getResult().getType().getWidth();
             ++index) {
          auto indexValue = LLVM::ConstantOp::create(
              rewriter, loc, rewriter.getI64Type(), index);
          auto element =
              LLVM::GEPOp::create(rewriter, loc, ptrType, i1Type, storage,
                                  ValueRange{indexValue.getResult()});
          LLVM::StoreOp::create(rewriter, loc, zero, element);
        }
      }
      rewriter.replaceOp(op, storage);
      return success();
    }

    auto fnSig = LLVM::LLVMFunctionType::get(
        voidType, {rewriter.getI64Type(), ptrType, ptrType});
    auto fnDec = getOrCreateFunctionDeclaration(rewriter, op,
                                                QIR_RESULT_ARRAY_ALLOC, fnSig);

    auto array = LLVM::AllocaOp::create(rewriter, loc, ptrType, ptrType, size)
                     .getResult();
    auto zero = LLVM::ZeroOp::create(rewriter, loc, ptrType).getResult();
    LLVM::CallOp::create(rewriter, loc, fnDec, ValueRange{size, array, zero});

    state.resultArrays.insert(array);
    reg.array = array;

    rewriter.replaceOp(op, array);
    return success();
  }
};

struct ConvertCBitLoadOp final : StatefulOpConversionPattern<cbit::LoadOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(cbit::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    const auto ptrType = LLVM::LLVMPointerType::get(getContext());
    if (!state.resultArrays.contains(adaptor.getReg())) {
      auto elementptr = LLVM::GEPOp::create(
          rewriter, op.getLoc(), ptrType, rewriter.getI1Type(),
          adaptor.getReg(), ValueRange{adaptor.getIndex()});
      rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, rewriter.getI1Type(),
                                                elementptr);
      return success();
    }
    auto elementptr =
        LLVM::GEPOp::create(rewriter, op.getLoc(), ptrType, ptrType,
                            adaptor.getReg(), ValueRange{adaptor.getIndex()});
    auto result =
        LLVM::LoadOp::create(rewriter, op.getLoc(), ptrType, elementptr);
    auto fnSig = LLVM::LLVMFunctionType::get(rewriter.getI1Type(), {ptrType});
    auto fnDec =
        getOrCreateFunctionDeclaration(rewriter, op, QIR_READ_RESULT, fnSig);
    auto readResult =
        LLVM::CallOp::create(rewriter, op.getLoc(), fnDec, result.getResult());
    rewriter.replaceOp(op, readResult.getResult());
    return success();
  }
};

struct ConvertCBitStoreOp final : StatefulOpConversionPattern<cbit::StoreOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(cbit::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    if (getState().resultArrays.contains(adaptor.getReg())) {
      return op.emitError(
          "non-measurement stores to returned CBit registers are not "
          "supported by QIR conversion");
    }
    const auto ptrType = LLVM::LLVMPointerType::get(getContext());
    auto elementptr = LLVM::GEPOp::create(
        rewriter, op.getLoc(), ptrType, rewriter.getI1Type(), adaptor.getReg(),
        ValueRange{adaptor.getIndex()});
    LLVM::StoreOp::create(rewriter, op.getLoc(), adaptor.getValue(),
                          elementptr);
    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Converts `memref.alloc` to `llvm.alloca`
 */
struct ConvertMemRefAllocOp final
    : StatefulOpConversionPattern<memref::AllocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    if (op.getType().getShape().size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "Only one-dimensional registers are supported");
    }
    return convertQubitMemRefAllocOp(op, adaptor, getState(), rewriter);
  }
};

/**
 * @brief Converts `memref.load` to `llvm.load`
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
    auto memrefType = op.getMemref().getType();
    if (memrefType.getShape().size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "Only one-dimensional registers are supported");
    }

    auto loc = op.getLoc();
    auto* ctx = getContext();
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto elementptr =
        LLVM::GEPOp::create(rewriter, loc, ptrType, ptrType,
                            adaptor.getMemref(), adaptor.getIndices()[0])
            .getResult();
    auto result =
        LLVM::LoadOp::create(rewriter, loc, ptrType, elementptr).getResult();

    rewriter.replaceOp(op, result);
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
 * is converted to
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
    if (auto shape = op.getMemref().getType().getShape(); shape.size() != 1) {
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

    auto size = state.qregSizes.lookup(op.getMemref());
    if (!size) {
      return rewriter.notifyMatchFailure(op, "unknown qubit register");
    }

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
 * is converted to
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
 * is converted to
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
 * is converted to
 * ```mlir
 * llvm.call @__quantum__qis__reset__body(%q) : !llvm.ptr -> ()
 * ```
 */
struct ConvertQCResetOp final : StatefulOpConversionPattern<ResetOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(ResetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = getContext();

    // Declare QIR function
    const auto fnSignature = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx), LLVM::LLVMPointerType::get(ctx));
    auto fnDecl =
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
 * For measurements without register information, a static result pointer is
 * used.
 * If the operation has an user, a read result call operation is created to
 * convert the result !llvm.ptr to an i1 value.
 *
 * @par Example (with register):
 * ```mlir
 * %result = qc.measure("c", 2, 0) %q : !qc.qubit -> i1
 * ```
 * is converted to
 * ```mlir
 * llvm.call @__quantum__qis__mz__body(%q, %b) : (!llvm.ptr, !llvm.ptr) -> ()
 * ```
 */
struct ConvertQCMeasureOp final : StatefulOpConversionPattern<MeasureOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(MeasureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();

    auto* ctx = getContext();
    auto ptrType = LLVM::LLVMPointerType::get(ctx);
    auto voidType = LLVM::LLVMVoidType::get(ctx);

    auto result =
        resolveRegisterMeasurement(state, op.getOperation(), rewriter);
    if (!result) {
      result = getResultPtr(state, op.getOperation(), rewriter);
    }

    // Create measure call
    auto fnSig = LLVM::LLVMFunctionType::get(voidType, {ptrType, ptrType});
    auto fnDec =
        getOrCreateFunctionDeclaration(rewriter, op, QIR_MEASURE, fnSig);

    LLVM::CallOp::create(rewriter, op.getLoc(), fnDec,
                         ValueRange{adaptor.getQubit(), result});

    // Create read-result call if the result is used
    if (op.getResult().use_empty()) {
      rewriter.eraseOp(op);
    } else {
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
  populateQCToQIRPatterns(patterns, typeConverter, ctx, state);
  patterns.add<ConvertCBitAllocOp, ConvertCBitLoadOp, ConvertCBitStoreOp,
               ConvertMemRefAllocOp, ConvertMemRefLoadOp,
               ConvertMemRefDeallocOp, ConvertQCAllocOp, ConvertQCDeallocOp,
               ConvertQCMeasureOp, ConvertQCResetOp>(typeConverter, ctx,
                                                     &state);
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
   * 2. **Intermediate blocks**: Original function structure containing
   * quantum operations
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
    Block* outputBlock = builder.createBlock(&main.getBody());

    state.entryBlock = entryBlock;
    state.outputBlock = outputBlock;

    builder.setInsertionPointToEnd(entryBlock);
    LLVM::BrOp::create(builder, main->getLoc(), firstBlock);
    auto* terminatorOp = lastBlock->getTerminator();
    terminatorOp->moveBefore(outputBlock, outputBlock->end());

    builder.setInsertionPointToEnd(lastBlock);
    LLVM::BrOp::create(builder, main->getLoc(), outputBlock);

    // Move up all constants to the beginning
    auto& entryOps = entryBlock->getOperations();
    for (auto& block : main.getBlocks()) {
      if (&block == entryBlock || &block == outputBlock) {
        continue;
      }
      for (auto it = block.begin(); it != block.end();) {
        if (auto& op = *it++; op.hasTrait<OpTrait::ConstantLike>()) {
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

    for (auto& [_, result] : state->staticResults) {
      auto sig = LLVM::LLVMFunctionType::get(voidType, {ptrType});
      auto dec = getOrCreateFunctionDeclaration(builder, main,
                                                QIR_RESULT_RELEASE, sig);
      LLVM::CallOp::create(builder, main->getLoc(), dec, result.pointer);
    }

    for (auto array : state->resultArrays) {
      auto sig = LLVM::LLVMFunctionType::get(voidType,
                                             {builder.getI64Type(), ptrType});
      auto dec = getOrCreateFunctionDeclaration(builder, main,
                                                QIR_RESULT_ARRAY_RELEASE, sig);
      auto size = array.getDefiningOp<LLVM::AllocaOp>().getArraySize();
      LLVM::CallOp::create(builder, main->getLoc(), dec,
                           ValueRange{size, array});
    }
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
   * Create proper block structure for QIR Adaptive Profile (entry,
   * intermediate blocks, output).
   *
   * **Stage 4: Initialization**
   * Insert the `__quantum__rt__initialize` call.
   *
   * **Stage 5: QC and memref to LLVM**
   * Convert QC dialect operations and memref operations to QIR calls and add
   * output recording to the output block.
   *
   * **Stage 6: Standard dialects to LLVM**
   * Convert arith and control flow dialects to LLVM (for index arithmetic and
   * function control flow).
   *
   * **Stage 7: Reconcile casts**
   * Clean up any unrealized cast operations introduced during type
   * conversion.
   */
  void runOnOperation() override {
    MLIRContext* ctx = &getContext();
    auto* moduleOp = getOperation();
    if (failed(mqt::normalizeGlobalPhases(cast<ModuleOp>(moduleOp)))) {
      signalPassFailure();
      return;
    }
    ConversionTarget target(*ctx);
    QCToQIRTypeConverter typeConverter(ctx);
    LoweringState state;

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

    // Stage 2.0: Prepare classical result registers
    if (failed(prepareClassicalResults(moduleOp, state))) {
      signalPassFailure();
      return;
    }

    // Stage 2.1: Convert func dialect to LLVM
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
      moduleOp->emitError("no main function with mqt.entry_point found");
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
      target.addIllegalDialect<cbit::CBitDialect, QCDialect,
                               memref::MemRefDialect>();

      populateQCToQIRAdaptivePatterns(patterns, typeConverter, ctx, state);

      if (applyPartialConversion(moduleOp, target, std::move(patterns))
              .failed()) {
        signalPassFailure();
        return;
      }

      addOutputRecording(main, ctx, state);
      releaseResults(main, ctx, &state);
    }

    // Stage 6: Convert standard dialects to LLVM
    {
      RewritePatternSet stdPatterns(ctx);
      target.addIllegalDialect<arith::ArithDialect>();
      target.addIllegalDialect<cf::ControlFlowDialect>();
      target.addIllegalDialect<math::MathDialect>();

      cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                      stdPatterns);
      cf::populateAssertToLLVMConversionPattern(typeConverter, stdPatterns);
      arith::populateArithToLLVMConversionPatterns(typeConverter, stdPatterns);
      populateMathToLLVMConversionPatterns(typeConverter, stdPatterns);

      if (applyPartialConversion(moduleOp, target, std::move(stdPatterns))
              .failed()) {
        signalPassFailure();
        return;
      }
    }

    // Stage 7: Reconcile unrealized casts
    PassManager passManager(ctx);
    passManager.addPass(createReconcileUnrealizedCastsPass());
    if (passManager.run(moduleOp).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir
