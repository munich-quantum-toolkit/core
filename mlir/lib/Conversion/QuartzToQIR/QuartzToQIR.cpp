/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/QuartzToQIR/QuartzToQIR.h"

#include "mlir/Dialect/QIR/Utils/QIRUtils.h"
#include "mlir/Dialect/Quartz/IR/QuartzDialect.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
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
#include <string>
#include <utility>

namespace mlir {

using namespace quartz;
using namespace qir;

#define GEN_PASS_DEF_QUARTZTOQIR
#include "mlir/Conversion/QuartzToQIR/QuartzToQIR.h.inc"

namespace {

/**
 * @brief State object for tracking lowering information during QIR conversion
 *
 * @details
 * This struct maintains state during the conversion of Quartz dialect
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
  DenseMap<int64_t, SmallVector<Value>> posCtrls;
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

/**
 * @brief Converts a one-target, zero-parameter Quartz operation to QIR
 *
 * @tparam QuartzOpType The operation type of the Quartz operation
 * @tparam QuartzOpAdaptorType The OpAdaptor type of the Quartz operation
 * @param op The Quartz operation instance to convert
 * @param adaptor The OpAdaptor of the Quartz operation
 * @param rewriter The pattern rewriter
 * @param ctx The MLIR context
 * @param state The lowering state
 * @param fnName The name of the QIR function to call
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QuartzOpType, typename QuartzOpAdaptorType>
LogicalResult
convertOneTargetZeroParameter(QuartzOpType& op, QuartzOpAdaptorType& adaptor,
                              ConversionPatternRewriter& rewriter,
                              MLIRContext* ctx, LoweringState& state,
                              StringRef fnName) {
  // Query state for modifier information
  const auto inCtrlOp = state.inCtrlOp;
  const SmallVector<Value> posCtrls =
      inCtrlOp != 0 ? state.posCtrls[inCtrlOp] : SmallVector<Value>{};
  const size_t numCtrls = posCtrls.size();

  // Define argument types
  SmallVector<Type> argumentTypes;
  argumentTypes.reserve(numCtrls + 1);
  const auto ptrType = LLVM::LLVMPointerType::get(ctx);
  // Add control pointers
  for (size_t i = 0; i < numCtrls; ++i) {
    argumentTypes.push_back(ptrType);
  }
  // Add target pointer
  argumentTypes.push_back(ptrType);

  // Define function signature
  const auto fnSignature =
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), argumentTypes);

  // Declare QIR function
  const auto fnDecl =
      getOrCreateFunctionDeclaration(rewriter, op, fnName, fnSignature);

  SmallVector<Value> operands;
  operands.reserve(numCtrls + 1);
  operands.append(posCtrls.begin(), posCtrls.end());
  operands.append(adaptor.getOperands().begin(), adaptor.getOperands().end());

  // Clean up modifier information
  if (inCtrlOp != 0) {
    state.posCtrls.erase(inCtrlOp);
    state.inCtrlOp--;
  }

  // Replace operation with CallOp
  rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl, operands);
  return success();
}

/**
 * @brief Converts a one-target, one-parameter Quartz operation to QIR
 *
 * @tparam QuartzOpType The operation type of the Quartz operation
 * @tparam QuartzOpAdaptorType The OpAdaptor type of the Quartz operation
 * @param op The Quartz operation instance to convert
 * @param adaptor The OpAdaptor of the Quartz operation
 * @param rewriter The pattern rewriter
 * @param ctx The MLIR context
 * @param state The lowering state
 * @param fnName The name of the QIR function to call
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QuartzOpType, typename QuartzOpAdaptorType>
LogicalResult
convertOneTargetOneParameter(QuartzOpType& op, QuartzOpAdaptorType& adaptor,
                             ConversionPatternRewriter& rewriter,
                             MLIRContext* ctx, LoweringState& state,
                             StringRef fnName) {
  // Query state for modifier information
  const auto inCtrlOp = state.inCtrlOp;
  const SmallVector<Value> posCtrls =
      inCtrlOp != 0 ? state.posCtrls[inCtrlOp] : SmallVector<Value>{};
  const size_t numCtrls = posCtrls.size();

  // Define argument types
  SmallVector<Type> argumentTypes;
  argumentTypes.reserve(numCtrls + 2);
  const auto ptrType = LLVM::LLVMPointerType::get(ctx);
  // Add control pointers
  for (size_t i = 0; i < numCtrls; ++i) {
    argumentTypes.push_back(ptrType);
  }
  // Add target pointer
  argumentTypes.push_back(ptrType);
  // Add parameter type
  argumentTypes.push_back(Float64Type::get(ctx));

  // Define function signature
  const auto fnSignature =
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), argumentTypes);

  // Declare QIR function
  const auto fnDecl =
      getOrCreateFunctionDeclaration(rewriter, op, fnName, fnSignature);

  SmallVector<Value> operands;
  operands.reserve(numCtrls + 2);
  operands.append(posCtrls.begin(), posCtrls.end());
  operands.append(adaptor.getOperands().begin(), adaptor.getOperands().end());

  // Clean up modifier information
  if (inCtrlOp != 0) {
    state.posCtrls.erase(inCtrlOp);
    state.inCtrlOp--;
  }

  // Replace operation with CallOp
  rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl, operands);
  return success();
}

/**
 * @brief Converts a one-target, two-parameter Quartz operation to QIR
 *
 * @tparam QuartzOpType The operation type of the Quartz operation
 * @tparam QuartzOpAdaptorType The OpAdaptor type of the Quartz operation
 * @param op The Quartz operation instance to convert
 * @param adaptor The OpAdaptor of the Quartz operation
 * @param rewriter The pattern rewriter
 * @param ctx The MLIR context
 * @param state The lowering state
 * @param fnName The name of the QIR function to call
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QuartzOpType, typename QuartzOpAdaptorType>
LogicalResult
convertOneTargetTwoParameter(QuartzOpType& op, QuartzOpAdaptorType& adaptor,
                             ConversionPatternRewriter& rewriter,
                             MLIRContext* ctx, LoweringState& state,
                             StringRef fnName) {
  // Query state for modifier information
  const auto inCtrlOp = state.inCtrlOp;
  const SmallVector<Value> posCtrls =
      inCtrlOp != 0 ? state.posCtrls[inCtrlOp] : SmallVector<Value>{};
  const size_t numCtrls = posCtrls.size();

  // Define argument types
  SmallVector<Type> argumentTypes;
  argumentTypes.reserve(numCtrls + 3);
  const auto ptrType = LLVM::LLVMPointerType::get(ctx);
  const auto floatType = Float64Type::get(ctx);
  // Add control pointers
  for (size_t i = 0; i < numCtrls; ++i) {
    argumentTypes.push_back(ptrType);
  }
  // Add target pointer
  argumentTypes.push_back(ptrType);
  // Add parameter types
  argumentTypes.push_back(floatType);
  argumentTypes.push_back(floatType);

  // Define function signature
  const auto fnSignature =
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), argumentTypes);

  // Declare QIR function
  const auto fnDecl =
      getOrCreateFunctionDeclaration(rewriter, op, fnName, fnSignature);

  SmallVector<Value> operands;
  operands.reserve(numCtrls + 3);
  operands.append(posCtrls.begin(), posCtrls.end());
  operands.append(adaptor.getOperands().begin(), adaptor.getOperands().end());

  // Clean up modifier information
  if (inCtrlOp != 0) {
    state.posCtrls.erase(inCtrlOp);
    state.inCtrlOp--;
  }

  // Replace operation with CallOp
  rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl, operands);
  return success();
}

/**
 * @brief Converts a one-target, three-parameter Quartz operation to QIR
 *
 * @tparam QuartzOpType The operation type of the Quartz operation
 * @tparam QuartzOpAdaptorType The OpAdaptor type of the Quartz operation
 * @param op The Quartz operation instance to convert
 * @param adaptor The OpAdaptor of the Quartz operation
 * @param rewriter The pattern rewriter
 * @param ctx The MLIR context
 * @param state The lowering state
 * @param fnName The name of the QIR function to call
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QuartzOpType, typename QuartzOpAdaptorType>
LogicalResult
convertOneTargetThreeParameter(QuartzOpType& op, QuartzOpAdaptorType& adaptor,
                               ConversionPatternRewriter& rewriter,
                               MLIRContext* ctx, LoweringState& state,
                               StringRef fnName) {
  // Query state for modifier information
  const auto inCtrlOp = state.inCtrlOp;
  const SmallVector<Value> posCtrls =
      inCtrlOp != 0 ? state.posCtrls[inCtrlOp] : SmallVector<Value>{};
  const size_t numCtrls = posCtrls.size();

  // Define argument types
  SmallVector<Type> argumentTypes;
  argumentTypes.reserve(numCtrls + 4);
  const auto ptrType = LLVM::LLVMPointerType::get(ctx);
  const auto floatType = Float64Type::get(ctx);
  // Add control pointers
  for (size_t i = 0; i < numCtrls; ++i) {
    argumentTypes.push_back(ptrType);
  }
  // Add target pointer
  argumentTypes.push_back(ptrType);
  // Add parameter types
  argumentTypes.push_back(floatType);
  argumentTypes.push_back(floatType);
  argumentTypes.push_back(floatType);

  // Define function signature
  const auto fnSignature =
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), argumentTypes);

  // Declare QIR function
  const auto fnDecl =
      getOrCreateFunctionDeclaration(rewriter, op, fnName, fnSignature);

  SmallVector<Value> operands;
  operands.reserve(numCtrls + 4);
  operands.append(posCtrls.begin(), posCtrls.end());
  operands.append(adaptor.getOperands().begin(), adaptor.getOperands().end());

  // Clean up modifier information
  if (inCtrlOp != 0) {
    state.posCtrls.erase(inCtrlOp);
    state.inCtrlOp--;
  }

  // Replace operation with CallOp
  rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl, operands);
  return success();
}

/**
 * @brief Converts a two-target, zero-parameter Quartz operation to QIR
 *
 * @tparam QuartzOpType The operation type of the Quartz operation
 * @tparam QuartzOpAdaptorType The OpAdaptor type of the Quartz operation
 * @param op The Quartz operation instance to convert
 * @param adaptor The OpAdaptor of the Quartz operation
 * @param rewriter The pattern rewriter
 * @param ctx The MLIR context
 * @param state The lowering state
 * @param fnName The name of the QIR function to call
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QuartzOpType, typename QuartzOpAdaptorType>
LogicalResult
convertTwoTargetZeroParameter(QuartzOpType& op, QuartzOpAdaptorType& adaptor,
                              ConversionPatternRewriter& rewriter,
                              MLIRContext* ctx, LoweringState& state,
                              StringRef fnName) {
  // Query state for modifier information
  const auto inCtrlOp = state.inCtrlOp;
  const SmallVector<Value> posCtrls =
      inCtrlOp != 0 ? state.posCtrls[inCtrlOp] : SmallVector<Value>{};
  const size_t numCtrls = posCtrls.size();

  // Define argument types
  SmallVector<Type> argumentTypes;
  argumentTypes.reserve(numCtrls + 2);
  const auto ptrType = LLVM::LLVMPointerType::get(ctx);
  // Add control pointers
  for (size_t i = 0; i < numCtrls; ++i) {
    argumentTypes.push_back(ptrType);
  }
  // Add target pointers
  argumentTypes.push_back(ptrType);
  argumentTypes.push_back(ptrType);

  // Define function signature
  const auto fnSignature =
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), argumentTypes);

  // Declare QIR function
  const auto fnDecl =
      getOrCreateFunctionDeclaration(rewriter, op, fnName, fnSignature);

  SmallVector<Value> operands;
  operands.reserve(numCtrls + 2);
  operands.append(posCtrls.begin(), posCtrls.end());
  operands.append(adaptor.getOperands().begin(), adaptor.getOperands().end());

  // Clean up modifier information
  if (inCtrlOp != 0) {
    state.posCtrls.erase(inCtrlOp);
    state.inCtrlOp--;
  }

  // Replace operation with CallOp
  rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl, operands);
  return success();
}

/**
 * @brief Converts a two-target, one-parameter Quartz operation to QIR
 *
 * @tparam QuartzOpType The operation type of the Quartz operation
 * @tparam QuartzOpAdaptorType The OpAdaptor type of the Quartz operation
 * @param op The Quartz operation instance to convert
 * @param adaptor The OpAdaptor of the Quartz operation
 * @param rewriter The pattern rewriter
 * @param ctx The MLIR context
 * @param state The lowering state
 * @param fnName The name of the QIR function to call
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QuartzOpType, typename QuartzOpAdaptorType>
LogicalResult
convertTwoTargetOneParameter(QuartzOpType& op, QuartzOpAdaptorType& adaptor,
                             ConversionPatternRewriter& rewriter,
                             MLIRContext* ctx, LoweringState& state,
                             StringRef fnName) {
  // Query state for modifier information
  const auto inCtrlOp = state.inCtrlOp;
  const SmallVector<Value> posCtrls =
      inCtrlOp != 0 ? state.posCtrls[inCtrlOp] : SmallVector<Value>{};
  const size_t numCtrls = posCtrls.size();

  // Define argument types
  SmallVector<Type> argumentTypes;
  argumentTypes.reserve(numCtrls + 3);
  const auto ptrType = LLVM::LLVMPointerType::get(ctx);
  // Add control pointers
  for (size_t i = 0; i < numCtrls; ++i) {
    argumentTypes.push_back(ptrType);
  }
  // Add target pointers
  argumentTypes.push_back(ptrType);
  argumentTypes.push_back(ptrType);
  // Add parameter type
  argumentTypes.push_back(Float64Type::get(ctx));

  // Define function signature
  const auto fnSignature =
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), argumentTypes);

  // Declare QIR function
  const auto fnDecl =
      getOrCreateFunctionDeclaration(rewriter, op, fnName, fnSignature);

  SmallVector<Value> operands;
  operands.reserve(numCtrls + 3);
  operands.append(posCtrls.begin(), posCtrls.end());
  operands.append(adaptor.getOperands().begin(), adaptor.getOperands().end());

  // Clean up modifier information
  if (inCtrlOp != 0) {
    state.posCtrls.erase(inCtrlOp);
    state.inCtrlOp--;
  }

  // Replace operation with CallOp
  rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl, operands);
  return success();
}

} // namespace

/**
 * @brief Type converter for lowering Quartz dialect types to LLVM types
 *
 * @details
 * Converts Quartz dialect types to their LLVM equivalents for QIR emission.
 *
 * Type conversions:
 * - `!quartz.qubit` -> `!llvm.ptr` (opaque pointer to qubit in QIR)
 */
struct QuartzToQIRTypeConverter final : LLVMTypeConverter {
  explicit QuartzToQIRTypeConverter(MLIRContext* ctx) : LLVMTypeConverter(ctx) {
    // Convert QubitType to LLVM pointer (QIR uses opaque pointers for qubits)
    addConversion(
        [ctx](QubitType /*type*/) { return LLVM::LLVMPointerType::get(ctx); });
  }
};

namespace {

/**
 * @brief Converts quartz.alloc operation to static QIR qubit allocations
 *
 * @details
 * QIR 2.0 does not support dynamic qubit allocation. Therefore, quartz.alloc
 * operations are converted to static qubit references using inttoptr with a
 * constant index.
 *
 * Register metadata (register_name, register_size, register_index) is used to
 * provide a reasonable guess for a static qubit index that is still free.
 *
 * @par Example:
 * ```mlir
 * %q = quartz.alloc : !quartz.qubit
 * ```
 * becomes:
 * ```mlir
 * %c0 = llvm.mlir.constant(0 : i64) : i64
 * %q0 = llvm.inttoptr %c0 : i64 to !llvm.ptr
 * ```
 */
struct ConvertQuartzAllocQIR final : StatefulOpConversionPattern<AllocOp> {
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
        assert(ptrMap.contains(globalIndex));
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
 * @brief Erases quartz.dealloc operations
 *
 * @details
 * Since QIR 2.0 does not support dynamic qubit allocation, dynamic allocations
 * are converted to static allocations. Therefore, deallocation operations
 * become no-ops and are simply removed from the IR.
 *
 * @par Example:
 * ```mlir
 * quartz.dealloc %q : !quartz.qubit
 * ```
 * becomes:
 * ```mlir
 * // (removed)
 * ```
 */
struct ConvertQuartzDeallocQIR final : OpConversionPattern<DeallocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DeallocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Converts quartz.static operation to QIR inttoptr
 *
 * @details
 * Converts a static qubit reference to an LLVM pointer by creating a constant
 * with the qubit index and converting it to a pointer. The pointer is cached
 * in the lowering state for reuse.
 *
 * @par Example:
 * ```mlir
 * %q0 = quartz.static 0 : !quartz.qubit
 * ```
 * becomes:
 * ```mlir
 * %c0 = llvm.mlir.constant(0 : i64) : i64
 * %q0 = llvm.inttoptr %c0 : i64 to !llvm.ptr
 * ```
 */
struct ConvertQuartzStaticQIR final : StatefulOpConversionPattern<StaticOp> {
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
 * @brief Converts quartz.measure operation to QIR measurement
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
 * %result = quartz.measure("c", 2, 0) %q : !quartz.qubit -> i1
 * ```
 * becomes:
 * ```mlir
 * %c0_i64 = llvm.mlir.constant(0 : i64) : i64
 * %result_ptr = llvm.inttoptr %c0_i64 : i64 to !llvm.ptr
 * llvm.call @__quantum__qis__mz__body(%q, %result_ptr) : (!llvm.ptr, !llvm.ptr)
 * -> ()
 * ```
 */
struct ConvertQuartzMeasureQIR final : StatefulOpConversionPattern<MeasureOp> {
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
          if (const auto it = ptrMap.find(numResults + i); it != ptrMap.end()) {
            val = it->second;
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
      // no register info, check if ptr has already been allocated (as a Qubit)
      if (const auto it = ptrMap.find(numResults); it != ptrMap.end()) {
        resultValue = it->second;
      } else {
        resultValue = createPointerFromIndex(rewriter, op.getLoc(), numResults);
        ptrMap[numResults] = resultValue;
      }
      registerResultMap.insert({{"c", numResults}, resultValue});
      state.numResults++;
    }

    // Declare QIR function
    const auto fnSignature = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx), {ptrType, ptrType});
    const auto fnDecl =
        getOrCreateFunctionDeclaration(rewriter, op, QIR_MEASURE, fnSignature);

    // Create CallOp and replace quartz.measure with result pointer
    rewriter.create<LLVM::CallOp>(op.getLoc(), fnDecl,
                                  ValueRange{adaptor.getQubit(), resultValue});
    rewriter.replaceOp(op, resultValue);
    return success();
  }
};

/**
 * @brief Converts quartz.reset operation to QIR reset
 *
 * @details
 * Converts qubit reset to a call to the QIR __quantum__qis__reset__body
 * function, which resets a qubit to the |0⟩ state.
 *
 * @par Example:
 * ```mlir
 * quartz.reset %q : !quartz.qubit
 * ```
 * becomes:
 * ```mlir
 * llvm.call @__quantum__qis__reset__body(%q) : (!llvm.ptr) -> ()
 * ```
 */
struct ConvertQuartzResetQIR final : OpConversionPattern<ResetOp> {
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

// OneTargetOneParameter

#define DEFINE_ONE_TARGET_ZERO_PARAMETER(OP_CLASS, OP_NAME_SMALL, OP_NAME_BIG, \
                                         QIR_NAME)                             \
  /**                                                                          \
   * @brief Converts quartz.OP_NAME_SMALL operation to QIR QIR_NAME            \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * quartz.OP_NAME_SMALL %q : !quartz.qubit                                   \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * llvm.call @__quantum__qis__QIR_NAME__body(%q) : (!llvm.ptr) -> ()         \
   * ```                                                                       \
   */                                                                          \
  struct ConvertQuartz##OP_CLASS##QIR final                                    \
      : StatefulOpConversionPattern<OP_CLASS> {                                \
    using StatefulOpConversionPattern::StatefulOpConversionPattern;            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(OP_CLASS op, OpAdaptor adaptor,                            \
                    ConversionPatternRewriter& rewriter) const override {      \
      auto& state = getState();                                                \
                                                                               \
      /* Query state for modifier information */                               \
      const auto inCtrlOp = state.inCtrlOp;                                    \
      const size_t numCtrls =                                                  \
          inCtrlOp != 0 ? state.posCtrls[inCtrlOp].size() : 0;                 \
                                                                               \
      /* Define function name */                                               \
      StringRef fnName;                                                        \
      if (inCtrlOp == 0) {                                                     \
        fnName = QIR_C##OP_NAME_BIG;                                           \
      } else {                                                                 \
        if (numCtrls == 1) {                                                   \
          fnName = QIR_C##OP_NAME_BIG;                                         \
        } else if (numCtrls == 2) {                                            \
          fnName = QIR_CC##OP_NAME_BIG;                                        \
        } else if (numCtrls == 3) {                                            \
          fnName = QIR_CCC##OP_NAME_BIG;                                       \
        } else {                                                               \
          return failure();                                                    \
        }                                                                      \
      }                                                                        \
                                                                               \
      return convertOneTargetZeroParameter(op, adaptor, rewriter,              \
                                           getContext(), state, fnName);       \
    }                                                                          \
  };

DEFINE_ONE_TARGET_ZERO_PARAMETER(IdOp, id, ID, i)
DEFINE_ONE_TARGET_ZERO_PARAMETER(XOp, x, X, x)
DEFINE_ONE_TARGET_ZERO_PARAMETER(YOp, y, Y, y)
DEFINE_ONE_TARGET_ZERO_PARAMETER(ZOp, z, Z, z)
DEFINE_ONE_TARGET_ZERO_PARAMETER(HOp, h, H, h)
DEFINE_ONE_TARGET_ZERO_PARAMETER(SOp, s, S, s)
DEFINE_ONE_TARGET_ZERO_PARAMETER(SdgOp, sdg, SDG, sdg)
DEFINE_ONE_TARGET_ZERO_PARAMETER(TOp, t, T, t)
DEFINE_ONE_TARGET_ZERO_PARAMETER(TdgOp, tdg, TDG, tdg)
DEFINE_ONE_TARGET_ZERO_PARAMETER(SXOp, sx, SX, sx)
DEFINE_ONE_TARGET_ZERO_PARAMETER(SXdgOp, sxdg, SXDG, sxdg)

#undef DEFINE_ONE_TARGET_ZERO_PARAMETER

// OneTargetOneParameter

#define DEFINE_ONE_TARGET_ONE_PARAMETER(OP_CLASS, OP_NAME_SMALL, OP_NAME_BIG,  \
                                        QIR_NAME, PARAM)                       \
  /**                                                                          \
   * @brief Converts quartz.OP_NAME_SMALL operation to QIR QIR_NAME            \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * quartz.OP_NAME_SMALL(%PARAM) %q : !quartz.qubit                           \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * llvm.call @__quantum__qis__QIR_NAME__body(%q, %PARAM) : (!llvm.ptr, f64)  \
   * -> ()                                                                     \
   * ```                                                                       \
   */                                                                          \
  struct ConvertQuartz##OP_CLASS##QIR final                                    \
      : StatefulOpConversionPattern<OP_CLASS> {                                \
    using StatefulOpConversionPattern::StatefulOpConversionPattern;            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(OP_CLASS op, OpAdaptor adaptor,                            \
                    ConversionPatternRewriter& rewriter) const override {      \
      auto& state = getState();                                                \
                                                                               \
      /* Query state for modifier information */                               \
      const auto inCtrlOp = state.inCtrlOp;                                    \
      const size_t numCtrls =                                                  \
          inCtrlOp != 0 ? state.posCtrls[inCtrlOp].size() : 0;                 \
                                                                               \
      /* Define function name */                                               \
      StringRef fnName;                                                        \
      if (inCtrlOp == 0) {                                                     \
        fnName = QIR_##OP_NAME_BIG;                                            \
      } else {                                                                 \
        if (numCtrls == 1) {                                                   \
          fnName = QIR_C##OP_NAME_BIG;                                         \
        } else if (numCtrls == 2) {                                            \
          fnName = QIR_CC##OP_NAME_BIG;                                        \
        } else if (numCtrls == 3) {                                            \
          fnName = QIR_CCC##OP_NAME_BIG;                                       \
        } else {                                                               \
          return failure();                                                    \
        }                                                                      \
      }                                                                        \
                                                                               \
      return convertOneTargetOneParameter(op, adaptor, rewriter, getContext(), \
                                          state, fnName);                      \
    }                                                                          \
  };

DEFINE_ONE_TARGET_ONE_PARAMETER(RXOp, rx, RX, rx, theta)
DEFINE_ONE_TARGET_ONE_PARAMETER(RYOp, ry, RY, ry, theta)
DEFINE_ONE_TARGET_ONE_PARAMETER(RZOp, rz, RZ, rz, theta)
DEFINE_ONE_TARGET_ONE_PARAMETER(POp, p, P, p, theta)

#undef DEFINE_ONE_TARGET_ONE_PARAMETER

// OneTargetTwoParameter

#define DEFINE_ONE_TARGET_TWO_PARAMETER(OP_CLASS, OP_NAME_SMALL, OP_NAME_BIG,  \
                                        QIR_NAME, PARAM1, PARAM2)              \
  /**                                                                          \
   * @brief Converts quartz.OP_NAME_SMALL operation to QIR QIR_NAME            \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * quartz.OP_NAME_SMALL(%PARAM1, %PARAM2) %q : !quartz.qubit                 \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * llvm.call @__quantum__qis__QIR_NAME__body(%q, %PARAM1, %PARAM2) :         \
   * (!llvm.ptr, f64, f64) -> ()                                               \
   * ```                                                                       \
   */                                                                          \
  struct ConvertQuartz##OP_CLASS##QIR final                                    \
      : StatefulOpConversionPattern<OP_CLASS> {                                \
    using StatefulOpConversionPattern::StatefulOpConversionPattern;            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(OP_CLASS op, OpAdaptor adaptor,                            \
                    ConversionPatternRewriter& rewriter) const override {      \
      auto& state = getState();                                                \
                                                                               \
      /* Query state for modifier information */                               \
      const auto inCtrlOp = state.inCtrlOp;                                    \
      const size_t numCtrls =                                                  \
          inCtrlOp != 0 ? state.posCtrls[inCtrlOp].size() : 0;                 \
                                                                               \
      /* Define function name */                                               \
      StringRef fnName;                                                        \
      if (inCtrlOp == 0) {                                                     \
        fnName = QIR_##OP_NAME_BIG;                                            \
      } else {                                                                 \
        if (numCtrls == 1) {                                                   \
          fnName = QIR_C##OP_NAME_BIG;                                         \
        } else if (numCtrls == 2) {                                            \
          fnName = QIR_CC##OP_NAME_BIG;                                        \
        } else if (numCtrls == 3) {                                            \
          fnName = QIR_CCC##OP_NAME_BIG;                                       \
        } else {                                                               \
          return failure();                                                    \
        }                                                                      \
      }                                                                        \
                                                                               \
      return convertOneTargetTwoParameter(op, adaptor, rewriter, getContext(), \
                                          state, fnName);                      \
    }                                                                          \
  };

DEFINE_ONE_TARGET_TWO_PARAMETER(ROp, r, R, r, theta, phi)
DEFINE_ONE_TARGET_TWO_PARAMETER(U2Op, u2, U2, u2, phi, lambda)

#undef DEFINE_ONE_TARGET_TWO_PARAMETER

// OneTargetThreeParameter

#define DEFINE_ONE_TARGET_THREE_PARAMETER(OP_CLASS, OP_NAME_SMALL,             \
                                          OP_NAME_BIG, QIR_NAME)               \
  /**                                                                          \
   * @brief Converts quartz.OP_NAME_SMALL operation to QIR QIR_NAME            \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * quartz.OP_NAME_SMALL(%PARAM1, %PARAM2, %PARAM3) %q : !quartz.qubit        \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * llvm.call @__quantum__qis__QIR_NAME__body(%q, %PARAM1, %PARAM2, %PARAM3)  \
   * : (!llvm.ptr, f64, f64, f64) -> ()                                        \
   * ```                                                                       \
   */                                                                          \
  struct ConvertQuartz##OP_CLASS##QIR final                                    \
      : StatefulOpConversionPattern<OP_CLASS> {                                \
    using StatefulOpConversionPattern::StatefulOpConversionPattern;            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(OP_CLASS op, OpAdaptor adaptor,                            \
                    ConversionPatternRewriter& rewriter) const override {      \
      auto& state = getState();                                                \
                                                                               \
      /* Query state for modifier information */                               \
      const auto inCtrlOp = state.inCtrlOp;                                    \
      const size_t numCtrls =                                                  \
          inCtrlOp != 0 ? state.posCtrls[inCtrlOp].size() : 0;                 \
                                                                               \
      /* Define function name */                                               \
      StringRef fnName;                                                        \
      if (inCtrlOp == 0) {                                                     \
        fnName = QIR_##OP_NAME_BIG;                                            \
      } else {                                                                 \
        if (numCtrls == 1) {                                                   \
          fnName = QIR_C##OP_NAME_BIG;                                         \
        } else if (numCtrls == 2) {                                            \
          fnName = QIR_CC##OP_NAME_BIG;                                        \
        } else if (numCtrls == 3) {                                            \
          fnName = QIR_CCC##OP_NAME_BIG;                                       \
        } else {                                                               \
          return failure();                                                    \
        }                                                                      \
      }                                                                        \
                                                                               \
      return convertOneTargetThreeParameter<OP_CLASS>(                         \
          op, adaptor, rewriter, getContext(), state, fnName);                 \
    }                                                                          \
  };

DEFINE_ONE_TARGET_THREE_PARAMETER(UOp, u, U, u3)

#undef DEFINE_ONE_TARGET_THREE_PARAMETER

// TwoTargetZeroParameter

#define DEFINE_TWO_TARGET_ZERO_PARAMETER(OP_CLASS, OP_NAME_SMALL, OP_NAME_BIG, \
                                         QIR_NAME)                             \
  /**                                                                          \
   * @brief Converts quartz.OP_NAME_SMALL operation to QIR QIR_NAME            \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * quartz.OP_NAME_SMALL %q1, %q2 : !quartz.qubit, !quartz.qubit              \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * llvm.call @__quantum__qis__QIR_NAME__body(%q1, %q2) : (!llvm.ptr,         \
   * !llvm.ptr) -> ()                                                          \
   * ```                                                                       \
   */                                                                          \
  struct ConvertQuartz##OP_CLASS##QIR final                                    \
      : StatefulOpConversionPattern<OP_CLASS> {                                \
    using StatefulOpConversionPattern::StatefulOpConversionPattern;            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(OP_CLASS op, OpAdaptor adaptor,                            \
                    ConversionPatternRewriter& rewriter) const override {      \
      auto& state = getState();                                                \
                                                                               \
      /* Query state for modifier information */                               \
      const auto inCtrlOp = state.inCtrlOp;                                    \
      const size_t numCtrls =                                                  \
          inCtrlOp != 0 ? state.posCtrls[inCtrlOp].size() : 0;                 \
                                                                               \
      /* Define function name */                                               \
      StringRef fnName;                                                        \
      if (inCtrlOp == 0) {                                                     \
        fnName = QIR_##OP_NAME_BIG;                                            \
      } else {                                                                 \
        if (numCtrls == 1) {                                                   \
          fnName = QIR_C##OP_NAME_BIG;                                         \
        } else if (numCtrls == 2) {                                            \
          fnName = QIR_CC##OP_NAME_BIG;                                        \
        } else if (numCtrls == 3) {                                            \
          fnName = QIR_CCC##OP_NAME_BIG;                                       \
        } else {                                                               \
          return failure();                                                    \
        }                                                                      \
      }                                                                        \
                                                                               \
      return convertTwoTargetZeroParameter(op, adaptor, rewriter,              \
                                           getContext(), state, fnName);       \
    }                                                                          \
  };

DEFINE_TWO_TARGET_ZERO_PARAMETER(SWAPOp, swap, SWAP, swap)
DEFINE_TWO_TARGET_ZERO_PARAMETER(iSWAPOp, iswap, ISWAP, iswap)
DEFINE_TWO_TARGET_ZERO_PARAMETER(DCXOp, dcx, DCX, dcx)
DEFINE_TWO_TARGET_ZERO_PARAMETER(ECROp, ecr, ECR, ecr)

#undef DEFINE_TWO_TARGET_ZERO_PARAMETER

// TwoTargetOneParameter

#define DEFINE_TWO_TARGET_ONE_PARAMETER(OP_CLASS, OP_NAME_SMALL, OP_NAME_BIG,  \
                                        QIR_NAME, PARAM)                       \
  /**                                                                          \
   * @brief Converts quartz.OP_NAME_SMALL operation to QIR QIR_NAME            \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * quartz.OP_NAME_SMALL(%PARAM) %q1, %q2 : !quartz.qubit, !quartz.qubit      \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * llvm.call @__quantum__qis__QIR_NAME__body(%q1, %q2, %PARAM) :             \
   * (!llvm.ptr, !llvm.ptr, f64) -> ()                                         \
   * ```                                                                       \
   */                                                                          \
  struct ConvertQuartz##OP_CLASS##QIR final                                    \
      : StatefulOpConversionPattern<OP_CLASS> {                                \
    using StatefulOpConversionPattern::StatefulOpConversionPattern;            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(OP_CLASS op, OpAdaptor adaptor,                            \
                    ConversionPatternRewriter& rewriter) const override {      \
      auto& state = getState();                                                \
                                                                               \
      /* Query state for modifier information */                               \
      const auto inCtrlOp = state.inCtrlOp;                                    \
      const size_t numCtrls =                                                  \
          inCtrlOp != 0 ? state.posCtrls[inCtrlOp].size() : 0;                 \
                                                                               \
      /* Define function name */                                               \
      StringRef fnName;                                                        \
      if (inCtrlOp == 0) {                                                     \
        fnName = QIR_##OP_NAME_BIG;                                            \
      } else {                                                                 \
        if (numCtrls == 1) {                                                   \
          fnName = QIR_C##OP_NAME_BIG;                                         \
        } else if (numCtrls == 2) {                                            \
          fnName = QIR_CC##OP_NAME_BIG;                                        \
        } else if (numCtrls == 3) {                                            \
          fnName = QIR_CCC##OP_NAME_BIG;                                       \
        } else {                                                               \
          return failure();                                                    \
        }                                                                      \
      }                                                                        \
                                                                               \
      return convertTwoTargetOneParameter(op, adaptor, rewriter, getContext(), \
                                          state, fnName);                      \
    }                                                                          \
  };

DEFINE_TWO_TARGET_ONE_PARAMETER(RXXOp, rxx, RXX, rxx, theta)

#undef DEFINE_TWO_TARGET_ONE_PARAMETER

/**
 * @brief Inlines quartz.ctrl region removes the operation
 */
struct ConvertQuartzCtrlQIR final : StatefulOpConversionPattern<CtrlOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(CtrlOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Update modifier information
    auto& state = getState();
    state.inCtrlOp++;
    const SmallVector<Value> posCtrls(adaptor.getControls().begin(),
                                      adaptor.getControls().end());
    state.posCtrls[state.inCtrlOp] = posCtrls;

    // Inline region and remove operation
    rewriter.inlineBlockBefore(&op.getRegion().front(), op->getBlock(),
                               op->getIterator());
    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Erases quartz.yield operation
 */
struct ConvertQuartzYieldQIR final : StatefulOpConversionPattern<YieldOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(YieldOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

/**
 * @brief Pass for converting Quartz dialect operations to QIR
 *
 * @details
 * This pass converts Quartz dialect quantum operations to QIR (Quantum
 * Intermediate Representation) by lowering them to LLVM dialect operations
 * that call QIR runtime functions.
 *
 * Conversion stages:
 * 1. Convert func dialect to LLVM
 * 2. Ensure proper block structure for QIR base profile and add
 * initialization
 * 3. Convert Quartz operations to QIR calls
 * 4. Set QIR metadata attributes
 * 5. Convert arith and cf dialects to LLVM
 * 6. Reconcile unrealized casts
 *
 * Current scope:
 * - Quartz operations: static, alloc, dealloc, measure, reset
 * - Gate operations will be added as the dialect expands
 * - Supports both static and dynamic qubit management
 */
struct QuartzToQIR final : impl::QuartzToQIRBase<QuartzToQIR> {
  using QuartzToQIRBase::QuartzToQIRBase;

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
   * Blocks are connected with unconditional jumps (entry → body →
   * measurements → output). This structure ensures proper QIR Base
   * Profile semantics.
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
    builder.create<LLVM::BrOp>(main->getLoc(), bodyBlock);

    builder.setInsertionPointToEnd(bodyBlock);
    builder.create<LLVM::BrOp>(main->getLoc(), measurementsBlock);

    builder.setInsertionPointToEnd(measurementsBlock);
    builder.create<LLVM::BrOp>(main->getLoc(), outputBlock);
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
  static void addInitialize(LLVM::LLVMFuncOp& main, MLIRContext* ctx,
                            LoweringState* /*state*/) {
    auto moduleOp = main->getParentOfType<ModuleOp>();
    auto& firstBlock = *(main.getBlocks().begin());
    OpBuilder builder(main.getBody());

    // Create a zero (null) pointer for the initialize call
    builder.setInsertionPointToStart(&firstBlock);
    auto zeroOp = builder.create<LLVM::ZeroOp>(main->getLoc(),
                                               LLVM::LLVMPointerType::get(ctx));

    // Insert the initialize call before the jump to main block
    const auto insertPoint = std::prev(firstBlock.getOperations().end(), 1);
    builder.setInsertionPoint(&*insertPoint);

    // Get or create the initialize function declaration
    auto* fnDecl = SymbolTable::lookupNearestSymbolFrom(
        main, builder.getStringAttr(QIR_INITIALIZE));
    if (fnDecl == nullptr) {
      const PatternRewriter::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToEnd(moduleOp.getBody());
      auto fnSignature = LLVM::LLVMFunctionType::get(
          LLVM::LLVMVoidType::get(ctx), LLVM::LLVMPointerType::get(ctx));
      fnDecl = builder.create<LLVM::LLVMFuncOp>(main->getLoc(), QIR_INITIALIZE,
                                                fnSignature);
    }

    // Create the initialization call
    builder.create<LLVM::CallOp>(main->getLoc(), cast<LLVM::LLVMFuncOp>(fnDecl),
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

    // Find the output block (4th block: entry, body, measurements, output,
    // end)
    auto& outputBlock = main.getBlocks().back();

    // Insert before the branch to end block
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
      auto arraySizeConst = builder.create<LLVM::ConstantOp>(
          main->getLoc(),
          builder.getI64IntegerAttr(static_cast<int64_t>(arraySize)));

      builder.create<LLVM::CallOp>(
          main->getLoc(), arrayRecordDecl,
          ValueRange{arraySizeConst.getResult(), arrayLabelOp.getResult()});

      // Create result_record_output calls for each measurement
      for (auto [regIdx, resultPtr] : measurements) {
        // Create label for result: "{arrayCounter+1+i}_{registerName}{i}r"
        const std::string resultLabel =
            registerName.str() + std::to_string(regIdx) + "r";
        auto resultLabelOp = createResultLabel(builder, main, resultLabel);

        builder.create<LLVM::CallOp>(
            main->getLoc(), resultRecordDecl,
            ValueRange{resultPtr, resultLabelOp.getResult()});
      }
    }
  }

  /**
   * @brief Executes the Quartz to QIR conversion pass
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
   * irreversible, end blocks) and insert the `__quantum__rt__initialize` call
   * in the entry block.
   *
   * **Stage 3: Quartz to LLVM**
   * Convert Quartz dialect operations to QIR calls (static, alloc, dealloc,
   * measure, reset) and add output recording to the output block.
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
    QuartzToQIRTypeConverter typeConverter(ctx);

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
    LoweringState state;
    addInitialize(main, ctx, &state);

    // Stage 3: Convert Quartz dialect to LLVM (QIR calls)
    {
      RewritePatternSet quartzPatterns(ctx);
      target.addIllegalDialect<QuartzDialect>();

      // Add conversion patterns for Quartz operations
      quartzPatterns.add<ConvertQuartzAllocQIR>(typeConverter, ctx, &state);
      quartzPatterns.add<ConvertQuartzDeallocQIR>(typeConverter, ctx);
      quartzPatterns.add<ConvertQuartzStaticQIR>(typeConverter, ctx, &state);
      quartzPatterns.add<ConvertQuartzMeasureQIR>(typeConverter, ctx, &state);
      quartzPatterns.add<ConvertQuartzResetQIR>(typeConverter, ctx);
      quartzPatterns.add<ConvertQuartzIdOpQIR>(typeConverter, ctx, &state);
      quartzPatterns.add<ConvertQuartzXOpQIR>(typeConverter, ctx, &state);
      quartzPatterns.add<ConvertQuartzYOpQIR>(typeConverter, ctx, &state);
      quartzPatterns.add<ConvertQuartzZOpQIR>(typeConverter, ctx, &state);
      quartzPatterns.add<ConvertQuartzHOpQIR>(typeConverter, ctx, &state);
      quartzPatterns.add<ConvertQuartzSOpQIR>(typeConverter, ctx, &state);
      quartzPatterns.add<ConvertQuartzSdgOpQIR>(typeConverter, ctx, &state);
      quartzPatterns.add<ConvertQuartzTOpQIR>(typeConverter, ctx, &state);
      quartzPatterns.add<ConvertQuartzTdgOpQIR>(typeConverter, ctx, &state);
      quartzPatterns.add<ConvertQuartzSXOpQIR>(typeConverter, ctx, &state);
      quartzPatterns.add<ConvertQuartzSXdgOpQIR>(typeConverter, ctx, &state);
      quartzPatterns.add<ConvertQuartzRXOpQIR>(typeConverter, ctx, &state);
      quartzPatterns.add<ConvertQuartzRYOpQIR>(typeConverter, ctx, &state);
      quartzPatterns.add<ConvertQuartzRZOpQIR>(typeConverter, ctx, &state);
      quartzPatterns.add<ConvertQuartzPOpQIR>(typeConverter, ctx, &state);
      quartzPatterns.add<ConvertQuartzROpQIR>(typeConverter, ctx, &state);
      quartzPatterns.add<ConvertQuartzU2OpQIR>(typeConverter, ctx, &state);
      quartzPatterns.add<ConvertQuartzUOpQIR>(typeConverter, ctx, &state);
      quartzPatterns.add<ConvertQuartzSWAPOpQIR>(typeConverter, ctx, &state);
      quartzPatterns.add<ConvertQuartziSWAPOpQIR>(typeConverter, ctx, &state);
      quartzPatterns.add<ConvertQuartzDCXOpQIR>(typeConverter, ctx, &state);
      quartzPatterns.add<ConvertQuartzECROpQIR>(typeConverter, ctx, &state);
      quartzPatterns.add<ConvertQuartzRXXOpQIR>(typeConverter, ctx, &state);
      quartzPatterns.add<ConvertQuartzCtrlQIR>(typeConverter, ctx, &state);
      quartzPatterns.add<ConvertQuartzYieldQIR>(typeConverter, ctx, &state);
      // Gate operations will be added here as the dialect expands

      if (applyPartialConversion(moduleOp, target, std::move(quartzPatterns))
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

} // namespace mlir
