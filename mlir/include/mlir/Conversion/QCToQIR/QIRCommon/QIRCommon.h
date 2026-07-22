/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "mlir/Dialect/QIR/Utils/QIRUtils.h"

#include <llvm/ADT/StringMap.h>
#include <llvm/Support/Allocator.h>
#include <llvm/Support/StringSaver.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace mlir {

/// Qubit allocation mode
enum class AllocationMode : std::uint8_t {
  Unset,  //!< No allocation mode has been established yet.
  Static, //!< The module uses static qubit allocation.
  Dynamic //!< The module uses dynamic qubit allocation.
};

/// State object for tracking lowering information during QIR conversion
struct LoweringState {
  /// Result-array pointers to be deallocated at the end of the program
  DenseSet<Value> resultArrays;

  /// Cache static qubit pointers for reuse
  DenseMap<int64_t, Value> staticQubits;

  /// Cache qubit register sizes for reuse
  DenseMap<Value, Value> qregSizes;

  /// Map from `memref::AllocOp` to `ClassicalRegister`
  DenseMap<Operation*, qir::ClassicalRegister> cregs;

  /// Metadata for returned classical registers. Each entry is a defining
  /// `qc::MeasureOp` and its corresponding (`memref::AllocOp`, index) pair
  DenseMap<Operation*, std::pair<Operation*, Value>> returnedCregs;

  /// Map from index to `StaticResult`
  DenseMap<int64_t, qir::StaticResult> staticResults;

  /// Metadata for returned static measurement results. Each entry is a defining
  /// `qc::MeasureOp`
  DenseSet<Operation*> returnedStaticResults;

  /// Modifier information
  size_t inCtrlOp = 0;
  SmallVector<Value> controls;

  /// Allocator and StringSaver for stable StringRefs
  llvm::BumpPtrAllocator allocator;
  llvm::StringSaver stringSaver{allocator};

  /// Block information
  Block* entryBlock{};
  Block* measurementsBlock{};
  Block* outputBlock{};

  /// The qubit allocation mode used in the module
  AllocationMode allocationMode = AllocationMode::Unset;

  /// Sets or validates the allocation mode, or emits an error if it conflicts.
  [[nodiscard]] LogicalResult ensureAllocationMode(AllocationMode requestedMode,
                                                   Operation* op);
};

struct QCToQIRTypeConverter final : LLVMTypeConverter {
  explicit QCToQIRTypeConverter(MLIRContext* ctx);
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
public:
  StatefulOpConversionPattern(TypeConverter& tc, MLIRContext* ctx,
                              LoweringState* state)
      : OpConversionPattern<OpType>(tc, ctx), state_(state) {}

  [[nodiscard]] LoweringState& getState() const { return *state_; }

private:
  LoweringState* state_;
};

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

/**
 * @brief Adds QIR initialization call to the entry block
 *
 * @details
 * This QIR runtime function initializes the quantum execution environment.
 *
 * @param main The main LLVM function
 * @param ctx The MLIR context
 * @param state The lowering state
 */
void addInitialize(LLVM::LLVMFuncOp& main, MLIRContext* ctx,
                   LoweringState& state);

/**
 * @brief Populates common conversion patterns for QC-to-QIR lowering.
 *
 * @details
 * Centralizes pattern registration so adding a new QC gate typically only
 * requires adding a new `ConvertQCUnitaryOpQIR<...>` specialization.
 */
void populateQCToQIRPatterns(RewritePatternSet& patterns,
                             QCToQIRTypeConverter& typeConverter,
                             MLIRContext* ctx, LoweringState& state);

/**
 * @brief Adds output recording calls to the output block
 *
 * @details
 * Generates output recording calls in the output block based on the
 * measurements tracked during conversion. Follows the QIR specification for
 * labeled output schema.
 *
 * Results that are part of registers are recorded via
 * `__quantum__rt__result_array_record_output`.
 *
 * Results that are not part of registers (i.e., measurements without register
 * info) are grouped under a default `__unnamed__` label recorded via
 * `__quantum__rt__result_record_output`.
 *
 * @param main The main LLVM function
 * @param ctx The MLIR context
 * @param state The lowering state containing measurement information
 */
void addOutputRecording(LLVM::LLVMFuncOp& main, MLIRContext* ctx,
                        LoweringState& state);

/**
 * @brief Strips returned measurement results from function return statements
 *
 * @details
 * Walks all `func::ReturnOp` operations in the module to identify operands
 * that correspond to measurement results. For each such operand:
 * - The defining operations are added to the `state` so that they are included
 * in the output recording.
 * - The operand is removed from the return statement.
 *
 * Non-measurement return values are preserved. After stripping, the enclosing
 * `func::FuncOp` function type is updated to match the new return operands.
 *
 * This must be called **before** func-to-LLVM conversion, while
 * `func::ReturnOp` and `qc::MeasureOp` are still in the IR.
 *
 * Return values that are indirectly computed from measurement outcomes remain
 * unaffected.
 *
 * @param moduleOp The top-level module operation to walk
 * @param state The lowering state; `returnedStaticResults` is populated
 */
void stripReturnedMeasurements(Operation* moduleOp, LoweringState& state);

/**
 * @brief Returns a result pointer for a measurement that does not write into a
 * returned classical bit register
 */
Value getResultPtr(LoweringState& state, Operation* op,
                   ConversionPatternRewriter& rewriter);

} // namespace mlir
