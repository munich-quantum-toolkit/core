/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/QCToQIR/QCToQIR.h"

#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QIR/Utils/QIRMetadata.h"
#include "mlir/Dialect/QIR/Utils/QIRUtils.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>
#include <utility>

namespace mlir {

using namespace qc;
using namespace qir;

#define GEN_PASS_DEF_QCTOQIR
#include "mlir/Conversion/QCToQIR/QCToQIR.h.inc"

namespace {

/**
 * @brief State object for tracking lowering information during QIR conversion
 *
 * @details
 * This struct maintains state during the conversion of QC dialect
 * operations to QIR (Quantum Intermediate Representation). It tracks:
 * - Qubit and result counts for QIR metadata
 * - Pointer value caching for reuse
 * - Whether dynamic memory management is needed
 * - Sequence of measurements for output recording
 */
struct LoweringState : QIRMetadata {
  /// Map from index to qubit pointer
  DenseMap<int64_t, Value> ptrMap;

  /// Map from register name to result-array pointer
  llvm::StringMap<Value> resultArrays;

  /// Map from (register name, index) to loaded result
  llvm::DenseMap<std::pair<llvm::StringRef, int64_t>, Value> loadedResults;

  /// Map from index to result pointer for non-register results
  DenseMap<int64_t, Value> resultPtrs;

  /// Modifier information
  int64_t inCtrlOp = 0;
  DenseMap<int64_t, SmallVector<Value>> controls;

  // Block information
  Block* entryBlock{};
  Block* measurementsBlock{};
};

/**
 * @brief Base class for conversion patterns that need access to lowering state
 *
 * @details
 * Extends OpConversionPattern to provide access to a shared LoweringState
 * object, which tracks qubit/result counts and caches values across multiple
 * pattern applications.
 *
 * @tparam OpType The operation type to convert
 */
template <typename OpType>
class StatefulOpConversionPattern : public OpConversionPattern<OpType> {
  using OpConversionPattern<OpType>::OpConversionPattern;

public:
  StatefulOpConversionPattern(TypeConverter& typeConverter, MLIRContext* ctx,
                              LoweringState* state)
      : OpConversionPattern<OpType>(typeConverter, ctx), state_(state) {}

  /// Returns the shared lowering state object
  [[nodiscard]] LoweringState& getState() const { return *state_; }

private:
  LoweringState* state_;
};

} // namespace

/**
 * @brief Helper to convert a QC operation to a LLVM CallOp
 *
 * @tparam QCOpType The operation type of the QC operation
 * @tparam QCOpAdaptorType The OpAdaptor type of the QC operation
 * @param op The QC operation instance to convert
 * @param adaptor The OpAdaptor of the QC operation
 * @param rewriter The pattern rewriter
 * @param ctx The MLIR context
 * @param state The lowering state
 * @param fnName The name of the QIR function to call
 * @param numTargets The number of targets
 * @param numParams The number of parameters
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QCOpType, typename QCOpAdaptorType>
static LogicalResult
convertUnitaryToCallOp(QCOpType& op, QCOpAdaptorType& adaptor,
                       ConversionPatternRewriter& rewriter, MLIRContext* ctx,
                       LoweringState& state, StringRef fnName,
                       size_t numTargets, size_t numParams) {
  // Query state for modifier information
  const auto inCtrlOp = state.inCtrlOp;
  const SmallVector<Value> controls =
      inCtrlOp != 0 ? state.controls[inCtrlOp] : SmallVector<Value>{};
  const size_t numCtrls = controls.size();

  // Define argument types
  SmallVector<Type> argumentTypes;
  argumentTypes.reserve(numParams + numCtrls + numTargets);
  const auto ptrType = LLVM::LLVMPointerType::get(ctx);
  const auto floatType = Float64Type::get(ctx);
  // Add control pointers
  for (size_t i = 0; i < numCtrls; ++i) {
    argumentTypes.push_back(ptrType);
  }
  // Add target pointers
  for (size_t i = 0; i < numTargets; ++i) {
    argumentTypes.push_back(ptrType);
  }
  // Add parameter types
  for (size_t i = 0; i < numParams; ++i) {
    argumentTypes.push_back(floatType);
  }

  // Define function signature
  const auto fnSignature =
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), argumentTypes);

  // Declare QIR function
  const auto fnDecl =
      getOrCreateFunctionDeclaration(rewriter, op, fnName, fnSignature);

  SmallVector<Value> operands;
  operands.reserve(numParams + numCtrls + numTargets);
  operands.append(controls.begin(), controls.end());
  operands.append(adaptor.getOperands().begin(), adaptor.getOperands().end());

  // Clean up modifier information
  if (inCtrlOp != 0) {
    state.controls.erase(inCtrlOp);
    state.inCtrlOp--;
  }

  // Replace operation with CallOp
  rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl, operands);
  return success();
}

namespace {

/**
 * @brief Type converter for lowering QC dialect types to LLVM types
 *
 * @details
 * Converts QC dialect types to their LLVM equivalents for QIR emission.
 *
 * Type conversions:
 * - `!qc.qubit` -> `!llvm.ptr` (opaque pointer to qubit in QIR)
 */
struct QCToQIRTypeConverter final : LLVMTypeConverter {
  explicit QCToQIRTypeConverter(MLIRContext* ctx) : LLVMTypeConverter(ctx) {
    // Convert QubitType to LLVM pointer (QIR uses opaque pointers for qubits)
    addConversion(
        [ctx](QubitType /*type*/) { return LLVM::LLVMPointerType::get(ctx); });

    addConversion(
        [ctx](MemRefType /*type*/) { return LLVM::LLVMPointerType::get(ctx); });
  }
};

struct ConvertMemRefAllocOp final
    : StatefulOpConversionPattern<memref::AllocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    state.useDynamicQubit = true;

    auto* ctx = getContext();
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto fnSig =
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                    {rewriter.getI64Type(), ptrType, ptrType});
    auto fnDec = getOrCreateFunctionDeclaration(rewriter, op,
                                                QIR_QUBIT_ARRAY_ALLOC, fnSig);

    auto shape = op.getType().getShape();
    if (shape.size() != 1) {
      return failure();
    }

    Value size;
    if (shape[0] == ShapedType::kDynamic) {
      size = adaptor.getDynamicSizes()[0];
    } else {
      size = LLVM::ConstantOp::create(
                 rewriter, op.getLoc(),
                 rewriter.getI64IntegerAttr(static_cast<int64_t>(shape[0])))
                 .getResult();
    }

    auto array =
        LLVM::AllocaOp::create(rewriter, op.getLoc(), ptrType, ptrType, size);
    auto zero = LLVM::ZeroOp::create(rewriter, op.getLoc(), ptrType);
    LLVM::CallOp::create(rewriter, op.getLoc(), fnDec,
                         ValueRange{size, array.getResult(), zero.getResult()});

    rewriter.replaceOp(op, array.getResult());

    return success();
  }
};

struct ConvertMemRefLoadOp final : StatefulOpConversionPattern<memref::LoadOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
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

struct ConvertMemRefDeallocOp final
    : StatefulOpConversionPattern<memref::DeallocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = getContext();
    auto i64Type = rewriter.getI64Type();
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto shape = op.getMemref().getType().getShape();
    if (shape.size() != 1) {
      return failure();
    }

    // Save current insertion point
    const OpBuilder::InsertionGuard guard(rewriter);

    // Switch to measurements block
    rewriter.setInsertionPoint(getState().measurementsBlock->getTerminator());

    Value size;
    if (shape[0] == ShapedType::kDynamic) {
      size =
          op.getMemref().getDefiningOp<memref::AllocOp>().getDynamicSizes()[0];
    } else {
      size = LLVM::ConstantOp::create(
                 rewriter, op.getLoc(),
                 rewriter.getI64IntegerAttr(static_cast<int64_t>(shape[0])))
                 .getResult();
    }

    auto fnSig = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                             {i64Type, ptrType});
    auto fnDec = getOrCreateFunctionDeclaration(rewriter, op,
                                                QIR_QUBIT_ARRAY_RELEASE, fnSig);

    // Create the release call
    LLVM::CallOp::create(rewriter, op.getLoc(), fnDec,
                         ValueRange{size, adaptor.getMemref()});
    rewriter.eraseOp(op);

    return success();
  }
};

/**
 * @brief Converts qc.alloc operation to static QIR qubit allocations
 *
 * @details
 * QIR 2.0 does not support dynamic qubit allocation. Therefore, qc.alloc
 * operations are converted to static qubit references using inttoptr with a
 * constant index.
 *
 * Register metadata (register_name, register_size, register_index) is used to
 * provide a reasonable guess for a static qubit index that is still free.
 *
 * @par Example:
 * ```mlir
 * %q = qc.alloc : !qc.qubit
 * ```
 * becomes:
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
 * @brief Erases qc.dealloc operations
 *
 * @details
 * Since QIR 2.0 does not support dynamic qubit allocation, dynamic
 * allocations are converted to static allocations. Therefore, deallocation
 * operations become no-ops and are simply removed from the IR.
 *
 * @par Example:
 * ```mlir
 * qc.dealloc %q : !qc.qubit
 * ```
 * becomes:
 * ```mlir
 * // (removed)
 * ```
 */
struct ConvertQCDeallocOp final : StatefulOpConversionPattern<DeallocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = getContext();
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    // Save current insertion point
    const OpBuilder::InsertionGuard guard(rewriter);

    // Switch to measurements block
    rewriter.setInsertionPoint(getState().measurementsBlock->getTerminator());

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
 * @brief Converts qc.static operation to QIR inttoptr
 *
 * @details
 * Converts a static qubit reference to an LLVM pointer by creating a constant
 * with the qubit index and converting it to a pointer. The pointer is cached
 * in the lowering state for reuse.
 *
 * @par Example:
 * ```mlir
 * %q0 = qc.static 0 : !qc.qubit
 * ```
 * becomes:
 * ```mlir
 * %c0 = llvm.mlir.constant(0 : i64) : i64
 * %q0 = llvm.inttoptr %c0 : i64 to !llvm.ptr
 * ```
 */
struct ConvertQCStaticOp final : StatefulOpConversionPattern<StaticOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(StaticOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    const auto index = static_cast<int64_t>(op.getIndex());
    auto& state = getState();

    // Get or create a pointer to the qubit
    Value qubit;
    if (const auto it = state.ptrMap.find(index); it != state.ptrMap.end()) {
      // Reuse existing pointer
      qubit = it->second;
    } else {
      // Create and cache for reuse
      qubit = createPointerFromIndex(rewriter, op.getLoc(), index);
      state.ptrMap.try_emplace(index, qubit);
    }
    rewriter.replaceOp(op, qubit);

    // Track maximum qubit index
    if (std::cmp_greater_equal(index, state.numQubits)) {
      state.numQubits = index + 1;
    }

    return success();
  }
};

/**
 * @brief Converts qc.measure operation to QIR measurement
 *
 * @details
 * Converts qubit measurement to a QIR call to `__quantum__qis__mz__body`.
 * Unlike the previous implementation, this does NOT immediately record
 * output. Instead, it tracks measurements in the lowering state for deferred
 * output recording in a separate output block, as required by the QIR Base
 * Profile.
 *
 * For measurements with register information, the result pointer is mapped
 * to (register_name, register_index) for later retrieval. For measurements
 * without register information, a sequential result pointer is assigned.
 *
 * @par Example (with register):
 * ```mlir
 * %result = qc.measure("c", 2, 0) %q : !qc.qubit -> i1
 * ```
 * becomes:
 * ```mlir
 * %c0_i64 = llvm.mlir.constant(0 : i64) : i64
 * %result_ptr = llvm.inttoptr %c0_i64 : i64 to !llvm.ptr
 * llvm.call @__quantum__qis__mz__body(%q, %result_ptr) : (!llvm.ptr,
 * !llvm.ptr)
 * -> ()
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
    const OpBuilder::InsertionGuard guard(rewriter);

    // Insert allocations and constants in entry block
    rewriter.setInsertionPoint(state.entryBlock->getTerminator());

    // Get result pointer
    Value result;
    if (op.getRegisterName() && op.getRegisterSize() && op.getRegisterIndex()) {
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
          loadedResults.try_emplace({registerName, i}, load.getResult());
        }
      }

      result = loadedResults.at({registerName, registerIndex});
    } else {
      auto fnSig =
          LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), {ptrType});
      auto fnDec =
          getOrCreateFunctionDeclaration(rewriter, op, QIR_RESULT_ALLOC, fnSig);

      auto zero = LLVM::ZeroOp::create(rewriter, op.getLoc(), ptrType);
      result =
          LLVM::CallOp::create(rewriter, op.getLoc(), fnDec, zero.getResult())
              .getResult();

      resultPtrs.try_emplace(resultPtrs.size(), result);
    }

    // Switch to measurements block
    rewriter.setInsertionPoint(state.measurementsBlock->getTerminator());

    // Create measure call
    auto fnSig = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                             {ptrType, ptrType});
    auto fnDec =
        getOrCreateFunctionDeclaration(rewriter, op, QIR_MEASURE, fnSig);

    LLVM::CallOp::create(rewriter, op.getLoc(), fnDec,
                         ValueRange{adaptor.getQubit(), result});

    rewriter.replaceOp(op, result);

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
 * llvm.call @__quantum__qis__reset__body(%q) : (!llvm.ptr) -> ()
 * ```
 */
struct ConvertQCResetOp final : StatefulOpConversionPattern<ResetOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(ResetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = getContext();

    // Save current insertion point
    const OpBuilder::InsertionGuard guard(rewriter);

    // Switch to measurements block
    rewriter.setInsertionPoint(getState().measurementsBlock->getTerminator());

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

// GPhaseOp

/**
 * @brief Converts qc.gphase to QIR gphase
 *
 * @par Example:
 * ```mlir
 * qc.gphase(%theta)
 * ```
 * is converted to
 * ```mlir
 * llvm.call @__quantum__qis__gphase__body(%theta) : (f64) -> ()
 * ```
 */
struct ConvertQCGPhaseOp final : StatefulOpConversionPattern<GPhaseOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(GPhaseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    if (state.inCtrlOp != 0) {
      return op.emitError("Controlled GPhaseOps cannot be converted to QIR");
    }
    return convertUnitaryToCallOp(op, adaptor, rewriter, getContext(), state,
                                  QIR_GPHASE, 0, 1);
  }
};

// OneTargetZeroParameter

#define DEFINE_ONE_TARGET_ZERO_PARAMETER(OP_CLASS, OP_NAME_BIG, OP_NAME_SMALL, \
                                         QIR_NAME)                             \
  /**                                                                          \
   * @brief Converts qc.OP_NAME_SMALL operation to QIR QIR_NAME                \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * qc.OP_NAME_SMALL %q : !qc.qubit                                           \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * llvm.call @__quantum__qis__QIR_NAME__body(%q) : (!llvm.ptr) -> ()         \
   * ```                                                                       \
   */                                                                          \
  struct ConvertQC##OP_CLASS final : StatefulOpConversionPattern<OP_CLASS> {   \
    using StatefulOpConversionPattern::StatefulOpConversionPattern;            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(OP_CLASS op, OpAdaptor adaptor,                            \
                    ConversionPatternRewriter& rewriter) const override {      \
      auto& state = getState();                                                \
      const auto inCtrlOp = state.inCtrlOp;                                    \
      const size_t numCtrls =                                                  \
          inCtrlOp != 0 ? state.controls[inCtrlOp].size() : 0;                 \
      const auto fnName = getFnName##OP_NAME_BIG(numCtrls);                    \
      return convertUnitaryToCallOp(op, adaptor, rewriter, getContext(),       \
                                    state, fnName, 1, 0);                      \
    }                                                                          \
  };

DEFINE_ONE_TARGET_ZERO_PARAMETER(IdOp, I, id, i)
DEFINE_ONE_TARGET_ZERO_PARAMETER(XOp, X, x, x)
DEFINE_ONE_TARGET_ZERO_PARAMETER(YOp, Y, y, y)
DEFINE_ONE_TARGET_ZERO_PARAMETER(ZOp, Z, z, z)
DEFINE_ONE_TARGET_ZERO_PARAMETER(HOp, H, h, h)
DEFINE_ONE_TARGET_ZERO_PARAMETER(SOp, S, s, s)
DEFINE_ONE_TARGET_ZERO_PARAMETER(SdgOp, SDG, sdg, sdg)
DEFINE_ONE_TARGET_ZERO_PARAMETER(TOp, T, t, t)
DEFINE_ONE_TARGET_ZERO_PARAMETER(TdgOp, TDG, tdg, tdg)
DEFINE_ONE_TARGET_ZERO_PARAMETER(SXOp, SX, sx, sx)
DEFINE_ONE_TARGET_ZERO_PARAMETER(SXdgOp, SXDG, sxdg, sxdg)

#undef DEFINE_ONE_TARGET_ZERO_PARAMETER

// OneTargetOneParameter

#define DEFINE_ONE_TARGET_ONE_PARAMETER(OP_CLASS, OP_NAME_BIG, OP_NAME_SMALL,  \
                                        QIR_NAME, PARAM)                       \
  /**                                                                          \
   * @brief Converts qc.OP_NAME_SMALL operation to QIR QIR_NAME                \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * qc.OP_NAME_SMALL(%PARAM) %q : !qc.qubit                                   \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * llvm.call @__quantum__qis__QIR_NAME__body(%q, %PARAM) : (!llvm.ptr, f64)  \
   * -> ()                                                                     \
   * ```                                                                       \
   */                                                                          \
  struct ConvertQC##OP_CLASS final : StatefulOpConversionPattern<OP_CLASS> {   \
    using StatefulOpConversionPattern::StatefulOpConversionPattern;            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(OP_CLASS op, OpAdaptor adaptor,                            \
                    ConversionPatternRewriter& rewriter) const override {      \
      auto& state = getState();                                                \
      const auto inCtrlOp = state.inCtrlOp;                                    \
      const size_t numCtrls =                                                  \
          inCtrlOp != 0 ? state.controls[inCtrlOp].size() : 0;                 \
      const auto fnName = getFnName##OP_NAME_BIG(numCtrls);                    \
      return convertUnitaryToCallOp(op, adaptor, rewriter, getContext(),       \
                                    state, fnName, 1, 1);                      \
    }                                                                          \
  };

DEFINE_ONE_TARGET_ONE_PARAMETER(RXOp, RX, rx, rx, theta)
DEFINE_ONE_TARGET_ONE_PARAMETER(RYOp, RY, ry, ry, theta)
DEFINE_ONE_TARGET_ONE_PARAMETER(RZOp, RZ, rz, rz, theta)
DEFINE_ONE_TARGET_ONE_PARAMETER(POp, P, p, p, theta)

#undef DEFINE_ONE_TARGET_ONE_PARAMETER

// OneTargetTwoParameter

#define DEFINE_ONE_TARGET_TWO_PARAMETER(OP_CLASS, OP_NAME_BIG, OP_NAME_SMALL,  \
                                        QIR_NAME, PARAM1, PARAM2)              \
  /**                                                                          \
   * @brief Converts qc.OP_NAME_SMALL operation to QIR QIR_NAME                \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * qc.OP_NAME_SMALL(%PARAM1, %PARAM2) %q : !qc.qubit                         \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * llvm.call @__quantum__qis__QIR_NAME__body(%q, %PARAM1, %PARAM2) :         \
   * (!llvm.ptr, f64, f64) -> ()                                               \
   * ```                                                                       \
   */                                                                          \
  struct ConvertQC##OP_CLASS final : StatefulOpConversionPattern<OP_CLASS> {   \
    using StatefulOpConversionPattern::StatefulOpConversionPattern;            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(OP_CLASS op, OpAdaptor adaptor,                            \
                    ConversionPatternRewriter& rewriter) const override {      \
      auto& state = getState();                                                \
      const auto inCtrlOp = state.inCtrlOp;                                    \
      const size_t numCtrls =                                                  \
          inCtrlOp != 0 ? state.controls[inCtrlOp].size() : 0;                 \
      const auto fnName = getFnName##OP_NAME_BIG(numCtrls);                    \
      return convertUnitaryToCallOp(op, adaptor, rewriter, getContext(),       \
                                    state, fnName, 1, 2);                      \
    }                                                                          \
  };

DEFINE_ONE_TARGET_TWO_PARAMETER(ROp, R, r, r, theta, phi)
DEFINE_ONE_TARGET_TWO_PARAMETER(U2Op, U2, u2, u2, phi, lambda)

#undef DEFINE_ONE_TARGET_TWO_PARAMETER

// OneTargetThreeParameter

#define DEFINE_ONE_TARGET_THREE_PARAMETER(OP_CLASS, OP_NAME_BIG,               \
                                          OP_NAME_SMALL, QIR_NAME)             \
  /**                                                                          \
   * @brief Converts qc.OP_NAME_SMALL operation to QIR QIR_NAME                \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * qc.OP_NAME_SMALL(%PARAM1, %PARAM2, %PARAM3) %q : !qc.qubit                \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * llvm.call @__quantum__qis__QIR_NAME__body(%q, %PARAM1, %PARAM2, %PARAM3)  \
   * : (!llvm.ptr, f64, f64, f64) -> ()                                        \
   * ```                                                                       \
   */                                                                          \
  struct ConvertQC##OP_CLASS final : StatefulOpConversionPattern<OP_CLASS> {   \
    using StatefulOpConversionPattern::StatefulOpConversionPattern;            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(OP_CLASS op, OpAdaptor adaptor,                            \
                    ConversionPatternRewriter& rewriter) const override {      \
      auto& state = getState();                                                \
      const auto inCtrlOp = state.inCtrlOp;                                    \
      const size_t numCtrls =                                                  \
          inCtrlOp != 0 ? state.controls[inCtrlOp].size() : 0;                 \
      const auto fnName = getFnName##OP_NAME_BIG(numCtrls);                    \
      return convertUnitaryToCallOp<OP_CLASS>(                                 \
          op, adaptor, rewriter, getContext(), state, fnName, 1, 3);           \
    }                                                                          \
  };

DEFINE_ONE_TARGET_THREE_PARAMETER(UOp, U, u, u3)

#undef DEFINE_ONE_TARGET_THREE_PARAMETER

// TwoTargetZeroParameter

#define DEFINE_TWO_TARGET_ZERO_PARAMETER(OP_CLASS, OP_NAME_BIG, OP_NAME_SMALL, \
                                         QIR_NAME)                             \
  /**                                                                          \
   * @brief Converts qc.OP_NAME_SMALL operation to QIR QIR_NAME                \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * qc.OP_NAME_SMALL %q1, %q2 : !qc.qubit, !qc.qubit                          \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * llvm.call @__quantum__qis__QIR_NAME__body(%q1, %q2) : (!llvm.ptr,         \
   * !llvm.ptr) -> ()                                                          \
   * ```                                                                       \
   */                                                                          \
  struct ConvertQC##OP_CLASS final : StatefulOpConversionPattern<OP_CLASS> {   \
    using StatefulOpConversionPattern::StatefulOpConversionPattern;            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(OP_CLASS op, OpAdaptor adaptor,                            \
                    ConversionPatternRewriter& rewriter) const override {      \
      auto& state = getState();                                                \
      const auto inCtrlOp = state.inCtrlOp;                                    \
      const size_t numCtrls =                                                  \
          inCtrlOp != 0 ? state.controls[inCtrlOp].size() : 0;                 \
      const auto fnName = getFnName##OP_NAME_BIG(numCtrls);                    \
      return convertUnitaryToCallOp(op, adaptor, rewriter, getContext(),       \
                                    state, fnName, 2, 0);                      \
    }                                                                          \
  };

DEFINE_TWO_TARGET_ZERO_PARAMETER(SWAPOp, SWAP, swap, swap)
DEFINE_TWO_TARGET_ZERO_PARAMETER(iSWAPOp, ISWAP, iswap, iswap)
DEFINE_TWO_TARGET_ZERO_PARAMETER(DCXOp, DCX, dcx, dcx)
DEFINE_TWO_TARGET_ZERO_PARAMETER(ECROp, ECR, ecr, ecr)

#undef DEFINE_TWO_TARGET_ZERO_PARAMETER

// TwoTargetOneParameter

#define DEFINE_TWO_TARGET_ONE_PARAMETER(OP_CLASS, OP_NAME_BIG, OP_NAME_SMALL,  \
                                        QIR_NAME, PARAM)                       \
  /**                                                                          \
   * @brief Converts qc.OP_NAME_SMALL operation to QIR QIR_NAME                \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * qc.OP_NAME_SMALL(%PARAM) %q1, %q2 : !qc.qubit, !qc.qubit                  \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * llvm.call @__quantum__qis__QIR_NAME__body(%q1, %q2, %PARAM) :             \
   * (!llvm.ptr, !llvm.ptr, f64) -> ()                                         \
   * ```                                                                       \
   */                                                                          \
  struct ConvertQC##OP_CLASS final : StatefulOpConversionPattern<OP_CLASS> {   \
    using StatefulOpConversionPattern::StatefulOpConversionPattern;            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(OP_CLASS op, OpAdaptor adaptor,                            \
                    ConversionPatternRewriter& rewriter) const override {      \
      auto& state = getState();                                                \
      const auto inCtrlOp = state.inCtrlOp;                                    \
      const size_t numCtrls =                                                  \
          inCtrlOp != 0 ? state.controls[inCtrlOp].size() : 0;                 \
      const auto fnName = getFnName##OP_NAME_BIG(numCtrls);                    \
      return convertUnitaryToCallOp(op, adaptor, rewriter, getContext(),       \
                                    state, fnName, 2, 1);                      \
    }                                                                          \
  };

DEFINE_TWO_TARGET_ONE_PARAMETER(RXXOp, RXX, rxx, rxx, theta)
DEFINE_TWO_TARGET_ONE_PARAMETER(RYYOp, RYY, ryy, ryy, theta)
DEFINE_TWO_TARGET_ONE_PARAMETER(RZXOp, RZX, rzx, rzx, theta)
DEFINE_TWO_TARGET_ONE_PARAMETER(RZZOp, RZZ, rzz, rzz, theta)

#undef DEFINE_TWO_TARGET_ONE_PARAMETER

// TwoTargetTwoParameter

#define DEFINE_TWO_TARGET_TWO_PARAMETER(OP_CLASS, OP_NAME_BIG, OP_NAME_SMALL,  \
                                        QIR_NAME, PARAM1, PARAM2)              \
  /**                                                                          \
   * @brief Converts qc.OP_NAME_SMALL operation to QIR QIR_NAME                \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * qc.OP_NAME_SMALL(%PARAM1, %PARAM2) %q1, %q2 : !qc.qubit,                  \
   * !qc.qubit                                                                 \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * llvm.call @__quantum__qis__QIR_NAME__body(%q1, %q2, %PARAM1, %PARAM2) :   \
   * (!llvm.ptr, !llvm.ptr, f64, f64) -> ()                                    \
   * ```                                                                       \
   */                                                                          \
  struct ConvertQC##OP_CLASS final : StatefulOpConversionPattern<OP_CLASS> {   \
    using StatefulOpConversionPattern::StatefulOpConversionPattern;            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(OP_CLASS op, OpAdaptor adaptor,                            \
                    ConversionPatternRewriter& rewriter) const override {      \
      auto& state = getState();                                                \
      const auto inCtrlOp = state.inCtrlOp;                                    \
      const size_t numCtrls =                                                  \
          inCtrlOp != 0 ? state.controls[inCtrlOp].size() : 0;                 \
      const auto fnName = getFnName##OP_NAME_BIG(numCtrls);                    \
      return convertUnitaryToCallOp(op, adaptor, rewriter, getContext(),       \
                                    state, fnName, 2, 2);                      \
    }                                                                          \
  };

DEFINE_TWO_TARGET_TWO_PARAMETER(XXPlusYYOp, XXPLUSYY, xx_plus_yy, xx_plus_yy,
                                theta, beta)
DEFINE_TWO_TARGET_TWO_PARAMETER(XXMinusYYOp, XXMINUSYY, xx_minus_yy,
                                xx_minus_yy, theta, beta)

#undef DEFINE_TWO_TARGET_TWO_PARAMETER

// BarrierOp

/**
 * @brief Erases qc.barrier operation, as it is a no-op in QIR
 */
struct ConvertQCBarrierOp final : StatefulOpConversionPattern<BarrierOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(BarrierOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Inlines qc.ctrl region removes the operation
 */
struct ConvertQCCtrlOp final : StatefulOpConversionPattern<CtrlOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(CtrlOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Update modifier information
    auto& state = getState();
    state.inCtrlOp++;
    const SmallVector<Value> controls(adaptor.getControls().begin(),
                                      adaptor.getControls().end());
    state.controls[state.inCtrlOp] = controls;

    // Inline region and remove operation
    rewriter.inlineBlockBefore(&op.getRegion().front(), op->getBlock(),
                               op->getIterator());
    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Erases qc.yield operation
 */
struct ConvertQCYieldOp final : StatefulOpConversionPattern<YieldOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(YieldOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

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
 * 2. Ensure proper block structure for QIR base profile and add initialization
 * 3. Convert QC operations to QIR calls
 * 4. Set QIR metadata attributes
 * 5. Convert arith and cf dialects to LLVM
 * 6. Reconcile unrealized casts
 *
 * @pre
 * The input entry function must consist of a single block. The pass will
 * restructure it into four blocks. Multi-block input functions are
 * currently not supported.
 */
struct QCToQIR final : impl::QCToQIRBase<QCToQIR> {
  using QCToQIRBase::QCToQIRBase;

  /**
   * @brief Ensures proper block structure for QIR base profile
   *
   * @details
   * The QIR base profile requires a specific 4-block structure:
   * 1. **Entry block**: Contains constant operations and initialization
   * 2. **Body block**: Contains reversible quantum operations (gates)
   * 3. **Measurements block**: Contains irreversible operations (measure,
   * reset, dealloc)
   * 4. **Output block**: Contains output recording calls
   *
   * Blocks are connected with unconditional jumps (entry, body, measurements,
   * output). This structure ensures proper QIR Base Profile semantics.
   *
   * If the function already has multiple blocks, this function does nothing.
   *
   * @param main The main LLVM function to restructure
   */
  static void ensureBlocks(LLVM::LLVMFuncOp& main, LoweringState& state) {
    // Return if there are already multiple blocks
    if (main.getBlocks().size() > 1) {
      return;
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

    auto& bodyBlockOps = bodyBlock->getOperations();
    auto& outputBlockOps = outputBlock->getOperations();

    // Move operations to appropriate blocks
    for (auto it = bodyBlock->begin(); it != bodyBlock->end();) {
      // Ensure iterator remains valid after potential move
      if (auto& op = *it++; isa<LLVM::ReturnOp>(op)) {
        // Move return to output block
        outputBlockOps.splice(outputBlock->end(), bodyBlockOps,
                              Block::iterator(op));
      } else if (isa<memref::AllocOp>(op) || isa<memref::LoadOp>(op) ||
                 isa<AllocOp>(op) || op.hasTrait<OpTrait::ConstantLike>()) {
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

  /**
   * @brief Adds QIR initialization call to the entry block
   *
   * @details
   * Inserts a call to `__quantum__rt__initialize` at the end of the entry block
   * (before the jump to main block). This QIR runtime function initializes the
   * quantum execution environment and takes a null pointer as argument.
   *
   * @param main The main LLVM function
   * @param ctx The MLIR context
   */
  static void addInitialize(LLVM::LLVMFuncOp& main, MLIRContext* ctx) {
    auto moduleOp = main->getParentOfType<ModuleOp>();
    auto& firstBlock = *(main.getBlocks().begin());
    OpBuilder builder(main.getBody());

    // Create a zero (null) pointer for the initialize call
    builder.setInsertionPointToStart(&firstBlock);
    auto zeroOp = LLVM::ZeroOp::create(builder, main->getLoc(),
                                       LLVM::LLVMPointerType::get(ctx));

    // Insert the initialize call before the jump to main block
    const auto insertPoint = std::prev(firstBlock.getOperations().end(), 1);
    builder.setInsertionPoint(&*insertPoint);

    // Get or create the initialize function declaration
    auto* fnDecl = SymbolTable::lookupNearestSymbolFrom(
        main, builder.getStringAttr(QIR_INITIALIZE));
    if (fnDecl == nullptr) {
      const PatternRewriter::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(moduleOp.getBody());
      auto fnSignature = LLVM::LLVMFunctionType::get(
          LLVM::LLVMVoidType::get(ctx), LLVM::LLVMPointerType::get(ctx));
      fnDecl = LLVM::LLVMFuncOp::create(builder, main->getLoc(), QIR_INITIALIZE,
                                        fnSignature);
    }

    // Create the initialization call
    LLVM::CallOp::create(builder, main->getLoc(),
                         cast<LLVM::LLVMFuncOp>(fnDecl), zeroOp->getResult(0));
  }

  /**
   * @brief Adds output recording calls to the output block
   *
   * @details
   * Generates output recording calls in the output block based on the
   * measurements tracked during conversion. Follows the QIR Base Profile
   * specification for labeled output schema.
   *
   * For each classical register, creates:
   * 1. An array_record_output call with the register size and label
   * 2. Individual result_record_output calls for each measurement in the
   * register
   *
   * Labels follow the format: "{registerName}{resultIndex}r"
   * - registerName: Name of the classical register (e.g., "c")
   * - resultIndex: Index within the array
   * - 'r' suffix: Indicates this is a result record
   *
   * Example output:
   * ```
   * @0 = internal constant [3 x i8] c"c\00"
   * @1 = internal constant [5 x i8] c"c0r\00"
   * @2 = internal constant [5 x i8] c"c1r\00"
   * call void @__quantum__rt__array_record_output(i64 2, ptr @0)
   * call void @__quantum__rt__result_record_output(ptr %result0, ptr @1)
   * call void @__quantum__rt__result_record_output(ptr %result1, ptr @2)
   * ```
   *
   * Any output recording calls that are not part of registers (i.e.,
   * measurements without register info) are grouped under a default label "c"
   * and recorded similarly.
   *
   * @param main The main LLVM function
   * @param ctx The MLIR context
   * @param state The lowering state containing measurement information
   */
  static void addOutputRecording(LLVM::LLVMFuncOp& main, MLIRContext* ctx,
                                 LoweringState* state) {
    auto& resultArrays = state->resultArrays;
    auto& resultPtrs = state->resultPtrs;

    if (resultArrays.empty() && resultPtrs.empty()) {
      return; // No measurements to record
    }

    OpBuilder builder(ctx);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);
    auto voidType = LLVM::LLVMVoidType::get(ctx);

    // Find the output block
    auto& outputBlock = main.getBlocks().back();

    // Insert before the branch to output block
    builder.setInsertionPoint(&outputBlock.back());

    if (!resultPtrs.empty()) {
      // Sort result pointers for deterministic output
      llvm::SmallVector<std::pair<int64_t, Value>> sortedPtrs;
      for (const auto& [index, resultPtr] : resultPtrs) {
        sortedPtrs.emplace_back(index, resultPtr);
      }
      llvm::sort(sortedPtrs, [](const auto& a, const auto& b) {
        return a.first < b.first;
      });

      // Create output recording for each result pointer
      auto fnSig = LLVM::LLVMFunctionType::get(voidType, {ptrType, ptrType});
      auto fnDec = getOrCreateFunctionDeclaration(builder, main,
                                                  QIR_RECORD_OUTPUT, fnSig);

      for (const auto& [index, ptr] : sortedPtrs) {
        auto label = createResultLabel(builder, main,
                                       "__unnamed__" + std::to_string(index))
                         .getResult();
        LLVM::CallOp::create(builder, main->getLoc(), fnDec,
                             ValueRange{ptr, label});
      }
    }

    if (!resultArrays.empty()) {
      // Sort registers by name for deterministic output
      SmallVector<std::pair<StringRef, Value>> sortedRegisters;
      for (auto& [name, results] : resultArrays) {
        sortedRegisters.emplace_back(name, results);
      }
      llvm::sort(sortedRegisters, [](const auto& a, const auto& b) {
        return a.first < b.first;
      });

      auto fnSig = LLVM::LLVMFunctionType::get(
          voidType, {builder.getI64Type(), ptrType, ptrType});
      auto fnDec = getOrCreateFunctionDeclaration(
          builder, main, QIR_ARRAY_RECORD_OUTPUT, fnSig);

      // Generate output recording for each register
      for (auto& [name, results] : sortedRegisters) {
        auto size = results.getDefiningOp<LLVM::AllocaOp>().getArraySize();
        auto label = createResultLabel(builder, main, name).getResult();

        LLVM::CallOp::create(builder, main->getLoc(), fnDec,
                             ValueRange{size, results, label});
      }
    }
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
   * **Stage 2: Block structure and initialization**
   * Create proper 4-block structure for QIR base profile (entry, main,
   * irreversible, output) and insert the `__quantum__rt__initialize` call in
   * the entry block.
   *
   * **Stage 3: QC to LLVM**
   * Convert QC dialect operations to QIR calls and add output recording to the
   * output block.
   *
   * **Stage 4: QIR attributes**
   * Add QIR base profile metadata to the main function, including qubit/result
   * counts and version information.
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

    // Stage 1: Convert func dialect to LLVM
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

    LoweringState state;

    // Stage 2: Create block structure
    ensureBlocks(main, state);

    // Stage 3: Convert QC dialect to LLVM (QIR calls)
    {
      RewritePatternSet patterns(ctx);
      target.addIllegalDialect<QCDialect, memref::MemRefDialect>();

      patterns.add<ConvertMemRefAllocOp, ConvertMemRefLoadOp,
                   ConvertMemRefDeallocOp, ConvertQCAllocOp, ConvertQCDeallocOp,
                   ConvertQCStaticOp, ConvertQCMeasureOp, ConvertQCResetOp,
                   ConvertQCGPhaseOp, ConvertQCIdOp, ConvertQCXOp, ConvertQCYOp,
                   ConvertQCZOp, ConvertQCHOp, ConvertQCSOp, ConvertQCSdgOp,
                   ConvertQCTOp, ConvertQCTdgOp, ConvertQCSXOp, ConvertQCSXdgOp,
                   ConvertQCRXOp, ConvertQCRYOp, ConvertQCRZOp, ConvertQCPOp,
                   ConvertQCROp, ConvertQCU2Op, ConvertQCUOp, ConvertQCSWAPOp,
                   ConvertQCiSWAPOp, ConvertQCDCXOp, ConvertQCECROp,
                   ConvertQCRXXOp, ConvertQCRYYOp, ConvertQCRZXOp,
                   ConvertQCRZZOp, ConvertQCXXPlusYYOp, ConvertQCXXMinusYYOp,
                   ConvertQCBarrierOp, ConvertQCCtrlOp, ConvertQCYieldOp>(
          typeConverter, ctx, &state);

      if (applyPartialConversion(moduleOp, target, std::move(patterns))
              .failed()) {
        signalPassFailure();
        return;
      }

      addOutputRecording(main, ctx, &state);
    }

    // Stage ?: Insert initialize call
    addInitialize(main, ctx);

    // Stage 4: Set QIR metadata attributes
    setQIRAttributes(main, state);

    // Stage 5: Convert standard dialects to LLVM
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
