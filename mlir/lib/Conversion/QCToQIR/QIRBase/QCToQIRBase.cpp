/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/QCToQIR/QIRBase/QCToQIRBase.h"

#include "mlir/Conversion/QCToQIR/QIRCommon/QIRCommon.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QIR/Utils/QIRUtils.h"

#include <llvm/Support/ErrorHandling.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

#include <cassert>
#include <cstdint>
#include <utility>

namespace mlir {

using namespace qc;
using namespace qir;

#define GEN_PASS_DEF_QCTOQIRBASE
#include "mlir/Conversion/QCToQIR/QIRBase/QCToQIRBase.h.inc"

/**
 * @brief Returns the result pointer the `qc::MeasureOp` @p op writes to, or
 * `nullptr` if it does not write into a returned classical register.
 */
static Value resolveRegisterMeasurement(LoweringState& state, Operation* op) {
  const auto it = state.returnedCregs.find(op);
  if (it == state.returnedCregs.end()) {
    return nullptr;
  }
  const auto [allocOp, index] = it->second;
  const auto indexValue = getConstantIntValue(index);
  assert(indexValue && "index must be constant");
  return state.cregs[allocOp].results[*indexValue];
}

namespace {

/**
 * @brief Converts a classical-bit-register `memref.alloc` to static result
 * pointers represented by `llvm.inttoptr` operations
 *
 * @details
 * Static qubit pointers are allocated during by `ConvertMemRefLoadOp`.
 */
struct ConvertMemRefAllocOp final
    : StatefulOpConversionPattern<memref::AllocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    const auto it = state.cregs.find(op.getOperation());
    if (it == state.cregs.end()) {
      rewriter.eraseOp(op);
      return success();
    }
    auto& reg = it->second;

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(state.entryBlock->getTerminator());
    const auto base = static_cast<int64_t>(state.staticResults.size());
    const auto size = std::get<int64_t>(reg.size);
    for (int64_t i = 0; i < size; ++i) {
      const auto index = base + i;
      auto result = createPointerFromIndex(rewriter, op.getLoc(), index);
      reg.results[i] = result;
      // The results are recorded as part of the register
      state.staticResults.try_emplace(
          index, qir::StaticResult{.pointer = result, .record = false});
    }

    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Converts a qubit-register `memref.load` to `llvm.inttoptr`
 *
 * @details
 * Converts a load operation to an LLVM pointer by creating a constant with the
 * next available static qubit index and converting it to a pointer. The pointer
 * is cached in the lowering state for reuse.
 *
 * @par Example:
 * ```mlir
 * %q0 = memref.load %memref[%c0] : memref<3x!qc.qubit>
 * ```
 * is converted to
 * ```mlir
 * %c0 = llvm.mlir.constant(0 : i64) : i64
 * %q0 = llvm.inttoptr %c0 : i64 to !llvm.ptr
 * ```
 */
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
    // Save current insertion point
    const OpBuilder::InsertionGuard guard(rewriter);

    // Switch to entry block
    rewriter.setInsertionPoint(state.entryBlock->getTerminator());

    auto nqubits = state.staticQubits.size();
    auto qubit = createPointerFromIndex(rewriter, op.getLoc(),
                                        static_cast<int64_t>(nqubits));
    state.staticQubits.try_emplace(static_cast<int64_t>(nqubits), qubit);
    rewriter.replaceOp(op, qubit);

    return success();
  }
};

/**
 * @brief Erases `memref.store` operations
 *
 * @details
 * Measurement results are implicitly stored by `__quantum__qis__mz__body`.
 */
struct ConvertMemRefStoreOp final
    : StatefulOpConversionPattern<memref::StoreOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Erases memref.dealloc during the QIR Base Profile conversion
 */
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

/**
 * @brief Converts qc.alloc to llvm.inttoptr
 *
 * @details
 * Converts a qubit allocation to an LLVM pointer by creating a constant
 * with the next available static qubit index and converting it to a pointer.
 * The pointer is cached in the lowering state for reuse.
 *
 * @par Example:
 * ```mlir
 * %q = qc.alloc : !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * %c0 = llvm.mlir.constant(0 : i64) : i64
 * %q0 = llvm.inttoptr %c0 : i64 to !llvm.ptr
 * ```
 */
struct ConvertQCAllocOp final : StatefulOpConversionPattern<AllocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();

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

/**
 * @brief Erases qc.dealloc during the QIR Base Profile conversion
 */
struct ConvertQCDeallocOp final : StatefulOpConversionPattern<DeallocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(DeallocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Converts qc.measure to QIR measurement
 *
 * @details
 * For measurements with register information, a static result is used at
 * the given index + register offset. Otherwise a static result at
 * the next index is used.
 *
 * @par Example (without register):
 * ```mlir
 * %result = qc.measure %q : !qc.qubit -> i1
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

    OpBuilder::InsertionGuard guard(rewriter);

    auto result = resolveRegisterMeasurement(state, op.getOperation());
    if (!result) {
      result = getResultPtr(state, op.getOperation(), rewriter);
    }

    // Emit the measurement in the measurements block
    rewriter.setInsertionPoint(state.measurementsBlock->getTerminator());
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

/**
 * @brief Populates conversion patterns for QC-to-QIR-Base lowering.
 */
static void populateQCToQIRBasePatterns(RewritePatternSet& patterns,
                                        QCToQIRTypeConverter& typeConverter,
                                        MLIRContext* ctx,
                                        LoweringState& state) {
  populateQCToQIRPatterns(patterns, typeConverter, ctx, state);
  patterns.add<ConvertMemRefAllocOp, ConvertMemRefLoadOp, ConvertMemRefStoreOp,
               ConvertMemRefDeallocOp, ConvertQCAllocOp, ConvertQCMeasureOp,
               ConvertQCDeallocOp>(typeConverter, ctx, &state);
}

namespace {
/**
 * @brief Pass for converting QC dialect operations to QIR
 *
 * @details
 * This pass converts QC dialect quantum operations to QIR (Quantum
 * Intermediate Representation) by lowering them to LLVM dialect operations
 * that call QIR runtime functions.
 *
 * Conversion stages:
 * 1. Convert func dialect to LLVM
 * 2. Ensure proper block structure for QIR base profile
 * 3. Add QIR initialization call
 * 4. Convert QC operations to QIR calls
 * 5. Set QIR metadata attributes
 * 6. Convert arith and cf dialects to LLVM
 * 7. Reconcile unrealized casts
 *
 * @pre
 * The input entry function must consist of a single block. The pass will
 * restructure it into four blocks. Multi-block input functions are
 * currently not supported.
 */
struct QCToQIRBase final : impl::QCToQIRBaseBase<QCToQIRBase> {
  using QCToQIRBaseBase::QCToQIRBaseBase;

  /**
   * @brief Ensures proper block structure for QIR base profile
   *
   * @details
   * The QIR base profile requires a specific 4-block structure:
   * 1. **Entry block**: Contains constant operations and initialization
   * 2. **Body block**: Contains reversible quantum operations (gates)
   * 3. **Measurements block**: Contains irreversible operations (measure
   * operations)
   * 4. **Output block**: Contains output recording calls
   *
   * Blocks are connected with unconditional jumps (entry, body, measurements,
   * output). This structure ensures proper QIR Base Profile semantics.
   *
   * @param main The main LLVM function to restructure
   */
  static void ensureBlocks(LLVM::LLVMFuncOp& main, LoweringState& state) {
    if (main.getBlocks().size() > 1) {
      llvm::reportFatalInternalError(
          "Modules with multiple blocks are not supported in the Base Profile");
    }

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
  /**
   * @brief Executes the QC to QIR conversion pass
   *
   * @details
   * Performs the conversion in six stages:
   *
   * **Stage 1: Func to LLVM**
   * Convert func dialect operations (main function) to LLVM dialect
   * equivalents.
   *
   * **Stage 2: Block structure**
   * Create proper 4-block structure for QIR base profile (entry, main,
   * irreversible, output).
   *
   * **Stage 3: Initialization**
   * Insert the `__quantum__rt__initialize` call.
   *
   * **Stage 4: QC to LLVM**
   * Convert QC dialect operations to QIR calls and add output recording to the
   * output block.
   *
   * **Stage 5: Standard dialects to LLVM**
   * Convert arith and control flow dialects to LLVM (for index arithmetic and
   * function control flow).
   *
   * **Stage 6: Reconcile casts**
   * Clean up any unrealized cast operations introduced during type conversion.
   */
  void runOnOperation() override {
    MLIRContext* ctx = &getContext();
    auto* moduleOp = getOperation();
    ConversionTarget target(*ctx);
    QCToQIRTypeConverter typeConverter(ctx);

    target.addLegalDialect<LLVM::LLVMDialect>();

    LoweringState state;

    // Stage 1.0: Strip returned measurements from func::ReturnOp
    stripReturnedMeasurements(moduleOp, state);

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

    auto main = getMainFunction(moduleOp);
    if (!main) {
      moduleOp->emitError("No main function with entry_point attribute found");
      signalPassFailure();
      return;
    }

    // Stage 2: Create block structure
    ensureBlocks(main, state);

    // Stage 3: Insert initialize call
    addInitialize(main, ctx, state);

    // Stage 4: Convert QC dialect to LLVM (QIR calls)
    {
      RewritePatternSet patterns(ctx);
      target.addIllegalDialect<QCDialect, memref::MemRefDialect>();

      populateQCToQIRBasePatterns(patterns, typeConverter, ctx, state);

      if (applyPartialConversion(moduleOp, target, std::move(patterns))
              .failed()) {
        signalPassFailure();
        return;
      }

      addOutputRecording(main, ctx, state);
    }

    // Stage 5: Convert standard dialects to LLVM
    {
      RewritePatternSet stdPatterns(ctx);
      target.addIllegalDialect<arith::ArithDialect>();
      target.addIllegalDialect<cf::ControlFlowDialect>();

      cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                      stdPatterns);
      cf::populateAssertToLLVMConversionPattern(typeConverter, stdPatterns);
      arith::populateArithToLLVMConversionPatterns(typeConverter, stdPatterns);

      if (applyPartialConversion(moduleOp, target, std::move(stdPatterns))
              .failed()) {
        signalPassFailure();
        return;
      }
    }

    // Stage 6: Reconcile unrealized casts
    PassManager passManager(ctx);
    passManager.addPass(createReconcileUnrealizedCastsPass());
    if (passManager.run(moduleOp).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir
