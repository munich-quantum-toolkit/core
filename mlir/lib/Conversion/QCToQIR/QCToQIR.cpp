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

#include "mlir/Conversion/GateTable.h"
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
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
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
  /// Map from register name to register start index
  DenseMap<StringRef, int64_t> registerStartIndexMap;

  /// Map from index to pointer value for reuse
  DenseMap<int64_t, Value> ptrMap;

  /// Map from (register_name, register_index) to result pointer
  /// This allows caching result pointers for measurements with register info
  DenseMap<std::pair<StringRef, int64_t>, Value> registerResultMap;

  /// Modifier information
  int64_t inCtrlOp = 0;
  DenseMap<int64_t, SmallVector<Value>> controls;
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

/**
 * @brief Generic converter for unitary QC ops to QIR calls.
 *
 * @details
 * Many QC gates lower to a QIR runtime call where the callee name depends on
 * the number of active controls. This helper factors out that boilerplate
 * without relying on preprocessor macros.
 *
 * @par Examples
 * The examples below illustrate the shapes that were previously documented via
 * `DEFINE_ONE_TARGET_ZERO_PARAMETER`, `DEFINE_ONE_TARGET_ONE_PARAMETER`, etc.
 *
 * @par One target, zero parameters
 * ```mlir
 * qc.x %q : !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * llvm.call @__quantum__qis__x__body(%q) : (!llvm.ptr) -> ()
 * ```
 *
 * @par One target, one parameter
 * ```mlir
 * qc.rx(%theta) %q : !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * llvm.call @__quantum__qis__rx__body(%q, %theta) : (!llvm.ptr, f64) -> ()
 * ```
 *
 * @par One target, two parameters
 * ```mlir
 * qc.r(%theta, %phi) %q : !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * llvm.call @__quantum__qis__r__body(%q, %theta, %phi) : (!llvm.ptr, f64, f64)
 * -> ()
 * ```
 *
 * @par One target, three parameters
 * ```mlir
 * qc.u(%theta, %phi, %lambda) %q : !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * llvm.call @__quantum__qis__u3__body(%q, %theta, %phi, %lambda)
 *     : (!llvm.ptr, f64, f64, f64) -> ()
 * ```
 *
 * @par Two targets, zero parameters
 * ```mlir
 * qc.swap %q0, %q1 : !qc.qubit, !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * llvm.call @__quantum__qis__swap__body(%q0, %q1) : (!llvm.ptr, !llvm.ptr) ->
 * ()
 * ```
 *
 * @par Two targets, one parameter
 * ```mlir
 * qc.rxx(%theta) %q0, %q1 : !qc.qubit, !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * llvm.call @__quantum__qis__rxx__body(%q0, %q1, %theta)
 *     : (!llvm.ptr, !llvm.ptr, f64) -> ()
 * ```
 *
 * @par Two targets, two parameters
 * ```mlir
 * qc.xx_plus_yy(%theta, %beta) %q0, %q1 : !qc.qubit, !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * llvm.call @__quantum__qis__xx_plus_yy__body(%q0, %q1, %theta, %beta)
 *     : (!llvm.ptr, !llvm.ptr, f64, f64) -> ()
 * ```
 *
 * @tparam OpType The QC operation type to convert
 * @tparam NumTargets Number of target qubits for this operation
 * @tparam NumParams Number of floating-point parameters for this operation
 * @tparam GetFnName Function that maps numCtrls -> QIR function name
 */
template <typename OpType, std::size_t NumTargets, std::size_t NumParams,
          auto GetFnName>
struct ConvertQCUnitaryOpQIR : StatefulOpConversionPattern<OpType> {
  using StatefulOpConversionPattern<OpType>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = this->getState();
    const auto inCtrlOp = state.inCtrlOp;
    const size_t numCtrls = inCtrlOp != 0 ? state.controls[inCtrlOp].size() : 0;
    const auto fnName = GetFnName(numCtrls);
    return convertUnitaryToCallOp(op, adaptor, rewriter, this->getContext(),
                                  state, fnName, NumTargets, NumParams);
  }
};

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
struct ConvertQCAllocQIR final : StatefulOpConversionPattern<AllocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    const auto numQubits = static_cast<int64_t>(state.numQubits);
    auto& ptrMap = state.ptrMap;
    auto& registerMap = state.registerStartIndexMap;

    // Get or create pointer value
    if (op.getRegisterName() && op.getRegisterSize() && op.getRegisterIndex()) {
      const auto registerName = op.getRegisterName().value();
      const auto registerSize =
          static_cast<int64_t>(op.getRegisterSize().value());
      const auto registerIndex =
          static_cast<int64_t>(op.getRegisterIndex().value());

      if (const auto it = registerMap.find(registerName);
          it != registerMap.end()) {
        // Register is already tracked
        // The pointer was created by the step below
        const auto globalIndex = it->second + registerIndex;
        if (!ptrMap.contains(globalIndex)) {
          return op.emitError("Pointer not found");
        }
        rewriter.replaceOp(op, ptrMap.at(globalIndex));
        return success();
      }

      // Allocate the entire register as static qubits
      registerMap[registerName] = numQubits;
      SmallVector<Value> pointers;
      pointers.reserve(registerSize);
      for (int64_t i = 0; i < registerSize; ++i) {
        Value val{};
        if (const auto it = ptrMap.find(numQubits + i); it != ptrMap.end()) {
          val = it->second;
        } else {
          val = createPointerFromIndex(rewriter, op.getLoc(), numQubits + i);
          ptrMap[numQubits + i] = val;
        }
        pointers.push_back(val);
      }
      rewriter.replaceOp(op, pointers[registerIndex]);
      state.numQubits += registerSize;
      return success();
    }

    // no register info, check if ptr has already been allocated (as a Result)
    Value val{};
    if (const auto it = ptrMap.find(numQubits); it != ptrMap.end()) {
      val = it->second;
    } else {
      val = createPointerFromIndex(rewriter, op.getLoc(), numQubits);
      ptrMap[numQubits] = val;
    }
    rewriter.replaceOp(op, val);
    state.numQubits++;
    return success();
  }
};

/**
 * @brief Erases qc.dealloc operations
 *
 * @details
 * Since QIR 2.0 does not support dynamic qubit allocation, dynamic allocations
 * are converted to static allocations. Therefore, deallocation operations
 * become no-ops and are simply removed from the IR.
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
struct ConvertQCDeallocQIR final : OpConversionPattern<DeallocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DeallocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
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
struct ConvertQCStaticQIR final : StatefulOpConversionPattern<StaticOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(StaticOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    const auto index = static_cast<int64_t>(op.getIndex());
    auto& state = getState();
    // Get or create a pointer to the qubit
    Value val{};
    if (const auto it = state.ptrMap.find(index); it != state.ptrMap.end()) {
      // Reuse existing pointer
      val = it->second;
    } else {
      // Create and cache for reuse
      val = createPointerFromIndex(rewriter, op.getLoc(), index);
      state.ptrMap.try_emplace(index, val);
    }
    rewriter.replaceOp(op, val);

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
 * Unlike the previous implementation, this does NOT immediately record output.
 * Instead, it tracks measurements in the lowering state for deferred output
 * recording in a separate output block, as required by the QIR Base Profile.
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
 * llvm.call @__quantum__qis__mz__body(%q, %result_ptr) : (!llvm.ptr, !llvm.ptr)
 * -> ()
 * ```
 */
struct ConvertQCMeasureQIR final : StatefulOpConversionPattern<MeasureOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(MeasureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = getContext();
    const auto ptrType = LLVM::LLVMPointerType::get(ctx);
    auto& state = getState();
    const auto numResults = static_cast<int64_t>(state.numResults);
    auto& ptrMap = state.ptrMap;
    auto& registerResultMap = state.registerResultMap;

    // Get or create result pointer value
    Value resultValue;
    if (op.getRegisterName() && op.getRegisterSize() && op.getRegisterIndex()) {
      const auto registerName = op.getRegisterName().value();
      const auto registerSize =
          static_cast<int64_t>(op.getRegisterSize().value());
      const auto registerIndex =
          static_cast<int64_t>(op.getRegisterIndex().value());
      const auto key = std::make_pair(registerName, registerIndex);

      if (const auto it = registerResultMap.find(key);
          it != registerResultMap.end()) {
        resultValue = it->second;
      } else {
        // Allocate the entire register as static results
        for (int64_t i = 0; i < registerSize; ++i) {
          Value val{};
          if (const auto ptrIt = ptrMap.find(numResults + i);
              ptrIt != ptrMap.end()) {
            val = ptrIt->second;
          } else {
            val = createPointerFromIndex(rewriter, op.getLoc(), numResults + i);
            ptrMap[numResults + i] = val;
          }
          registerResultMap.try_emplace({registerName, i}, val);
        }
        state.numResults += registerSize;
        resultValue = registerResultMap.at(key);
      }
    } else {
      // Choose a safe default register name
      StringRef defaultRegName = "c";
      if (llvm::any_of(registerResultMap, [](const auto& entry) {
            return entry.first.first == "c";
          })) {
        defaultRegName = "__unnamed__";
      }
      // No register info, check if ptr has already been allocated (as a Qubit)
      if (const auto it = ptrMap.find(numResults); it != ptrMap.end()) {
        resultValue = it->second;
      } else {
        resultValue = createPointerFromIndex(rewriter, op.getLoc(), numResults);
        ptrMap[numResults] = resultValue;
      }
      registerResultMap.insert({{defaultRegName, numResults}, resultValue});
      state.numResults++;
    }

    // Declare QIR function
    const auto fnSignature = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx), {ptrType, ptrType});
    const auto fnDecl =
        getOrCreateFunctionDeclaration(rewriter, op, QIR_MEASURE, fnSignature);

    // Create CallOp and replace qc.measure with result pointer
    LLVM::CallOp::create(rewriter, op.getLoc(), fnDecl,
                         ValueRange{adaptor.getQubit(), resultValue});
    rewriter.replaceOp(op, resultValue);
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
struct ConvertQCResetQIR final : OpConversionPattern<ResetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ResetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
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
struct ConvertQCGPhaseOpQIR final : StatefulOpConversionPattern<GPhaseOp> {
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

// BarrierOp

/**
 * @brief Erases qc.barrier operation, as it is a no-op in QIR
 */
struct ConvertQCBarrierQIR final : StatefulOpConversionPattern<BarrierOp> {
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
struct ConvertQCCtrlQIR final : StatefulOpConversionPattern<CtrlOp> {
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
struct ConvertQCYieldQIR final : StatefulOpConversionPattern<YieldOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(YieldOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Populates conversion patterns for QC-to-QIR lowering.
 *
 * @details
 * Centralizes pattern registration so adding a new QC gate typically only
 * requires adding a new `ConvertQCUnitaryOpQIR<...>` specialization to the
 * list of unitary gates below.
 */
void populateQCToQIRPatterns(RewritePatternSet& patterns,
                             QCToQIRTypeConverter& typeConverter,
                             MLIRContext* ctx, LoweringState& state) {
  patterns.add<ConvertQCAllocQIR>(typeConverter, ctx, &state);
  patterns.add<ConvertQCDeallocQIR>(typeConverter, ctx);
  patterns.add<ConvertQCStaticQIR>(typeConverter, ctx, &state);
  patterns.add<ConvertQCMeasureQIR>(typeConverter, ctx, &state);
  patterns.add<ConvertQCResetQIR>(typeConverter, ctx);
  patterns.add<ConvertQCGPhaseOpQIR>(typeConverter, ctx, &state);

  // Note: `MQT_GATE_TABLE` is defined in `mlir/Conversion/GateTable.h`.
#define MQT_ADD_QC_TO_QIR_UNITARY(                                             \
    KEY, TARGETS, PARAMS, QCO_OP, QC_OP, JEFF_KIND, JEFF_OP,                   \
    JEFF_BASE_ADJOINT, JEFF_CUSTOM_NAME, JEFF_PPR, QIR_KIND, QIR_FN)           \
  patterns.add<ConvertQCUnitaryOpQIR<QC_OP, (TARGETS), (PARAMS), &(QIR_FN)>>(  \
      typeConverter, ctx, &state);
  MQT_GATE_TABLE(MQT_ADD_QC_TO_QIR_UNITARY)
#undef MQT_ADD_QC_TO_QIR_UNITARY

  patterns.add<ConvertQCBarrierQIR, ConvertQCCtrlQIR, ConvertQCYieldQIR>(
      typeConverter, ctx, &state);
}

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
 * 2. Ensure proper block structure for QIR base profile and add
 * initialization
 * 3. Convert QC operations to QIR calls
 * 4. Set QIR metadata attributes
 * 5. Convert arith and cf dialects to LLVM
 * 6. Reconcile unrealized casts
 *
 * @pre
 * The input entry function must consist of a single block. The pass will
 * restructure it into four blocks. Multi-block input functions are currently
 * not supported.
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
   *    reset, dealloc)
   * 4. **Output block**: Contains output recording calls
   *
   * Blocks are connected with unconditional jumps (entry, body, measurements,
   * output). This structure ensures proper QIR Base Profile semantics.
   *
   * If the function already has multiple blocks, this function does nothing.
   *
   * @param main The main LLVM function to restructure
   */
  static void ensureBlocks(LLVM::LLVMFuncOp& main) {
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

    auto& bodyBlockOps = bodyBlock->getOperations();
    auto& outputBlockOps = outputBlock->getOperations();
    auto& measurementsBlockOps = measurementsBlock->getOperations();

    // Move operations to appropriate blocks
    for (auto it = bodyBlock->begin(); it != bodyBlock->end();) {
      // Ensure iterator remains valid after potential move
      if (auto& op = *it++;
          isa<DeallocOp>(op) || isa<ResetOp>(op) || isa<MeasureOp>(op)) {
        // Move irreversible quantum operations to measurements block
        measurementsBlockOps.splice(measurementsBlock->end(), bodyBlockOps,
                                    Block::iterator(op));
      } else if (isa<LLVM::ReturnOp>(op)) {
        // Move return to output block
        outputBlockOps.splice(outputBlock->end(), bodyBlockOps,
                              Block::iterator(op));
      } else if (op.hasTrait<OpTrait::ConstantLike>()) {
        // Move constant like operations to the entry block
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
   * Inserts a call to `__quantum__rt__initialize` at the end of the entry
   * block (before the jump to main block). This QIR runtime function
   * initializes the quantum execution environment and takes a null pointer as
   * argument.
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
                         cast<LLVM::LLVMFuncOp>(fnDecl),
                         ValueRange{zeroOp->getResult(0)});
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
   * measurements without register info) are grouped under a default label
   * "c" and recorded similarly.
   *
   * @param main The main LLVM function
   * @param ctx The MLIR context
   * @param state The lowering state containing measurement information
   */
  static void addOutputRecording(LLVM::LLVMFuncOp& main, MLIRContext* ctx,
                                 LoweringState* state) {
    if (state->registerResultMap.empty()) {
      return; // No measurements to record
    }

    OpBuilder builder(ctx);
    const auto ptrType = LLVM::LLVMPointerType::get(ctx);

    // Find the output block
    auto& outputBlock = main.getBlocks().back();

    // Insert before the branch to output block
    builder.setInsertionPoint(&outputBlock.back());

    // Group measurements by register
    llvm::StringMap<SmallVector<std::pair<int64_t, Value>>> registerGroups;
    for (const auto& [key, resultPtr] : state->registerResultMap) {
      const auto& [registerName, registerIndex] = key;
      registerGroups[registerName].emplace_back(registerIndex, resultPtr);
    }

    // Sort registers by name for deterministic output
    SmallVector<std::pair<StringRef, SmallVector<std::pair<int64_t, Value>>>>
        sortedRegisters;
    for (auto& [name, measurements] : registerGroups) {
      sortedRegisters.emplace_back(name, std::move(measurements));
    }
    llvm::sort(sortedRegisters,
               [](const auto& a, const auto& b) { return a.first < b.first; });

    // create function declarations for output recording
    const auto arrayRecordSig = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx), {builder.getI64Type(), ptrType});
    const auto arrayRecordDecl = getOrCreateFunctionDeclaration(
        builder, main, QIR_ARRAY_RECORD_OUTPUT, arrayRecordSig);

    const auto resultRecordSig = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx), {ptrType, ptrType});
    const auto resultRecordDecl = getOrCreateFunctionDeclaration(
        builder, main, QIR_RECORD_OUTPUT, resultRecordSig);

    // Generate output recording for each register
    for (auto& [registerName, measurements] : sortedRegisters) {
      // Sort measurements by register index
      llvm::sort(measurements, [](const auto& a, const auto& b) {
        return a.first < b.first;
      });

      const auto arraySize = measurements.size();
      auto arrayLabelOp = createResultLabel(builder, main, registerName);
      auto arraySizeConst = LLVM::ConstantOp::create(
          builder, main->getLoc(),
          builder.getI64IntegerAttr(static_cast<int64_t>(arraySize)));

      LLVM::CallOp::create(
          builder, main->getLoc(), arrayRecordDecl,
          ValueRange{arraySizeConst.getResult(), arrayLabelOp.getResult()});

      // Create result_record_output calls for each measurement
      for (auto [regIdx, resultPtr] : measurements) {
        // Create label for result: "{arrayCounter+1+i}_{registerName}{i}r"
        const std::string resultLabel =
            registerName.str() + std::to_string(regIdx) + "r";
        auto resultLabelOp = createResultLabel(builder, main, resultLabel);

        LLVM::CallOp::create(builder, main->getLoc(), resultRecordDecl,
                             ValueRange{resultPtr, resultLabelOp.getResult()});
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
   * irreversible, output) and insert the `__quantum__rt__initialize` call
   * in the entry block.
   *
   * **Stage 3: QC to LLVM**
   * Convert QC dialect operations to QIR calls and add output recording to
   * the output block.
   *
   * **Stage 4: QIR attributes**
   * Add QIR base profile metadata to the main function, including
   * qubit/result counts and version information.
   *
   * **Stage 5: Standard dialects to LLVM**
   * Convert arith and control flow dialects to LLVM (for index arithmetic and
   * function control flow).
   *
   * **Stage 6: Reconcile casts**
   * Clean up any unrealized cast operations introduced during type
   * conversion.
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

    // Stage 2: Ensure proper block structure and add initialization
    auto main = getMainFunction(moduleOp);
    if (!main) {
      moduleOp->emitError("No main function with entry_point attribute found");
      signalPassFailure();
      return;
    }

    ensureBlocks(main);
    addInitialize(main, ctx);

    LoweringState state;

    // Stage 3: Convert QC dialect to LLVM (QIR calls)
    {
      RewritePatternSet qcPatterns(ctx);
      target.addIllegalDialect<QCDialect>();

      populateQCToQIRPatterns(qcPatterns, typeConverter, ctx, state);

      if (applyPartialConversion(moduleOp, target, std::move(qcPatterns))
              .failed()) {
        signalPassFailure();
        return;
      }

      addOutputRecording(main, ctx, &state);
    }

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
