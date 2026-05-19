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

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/QIR/Utils/QIRMetadata.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Support/Allocator.h>
#include <llvm/Support/StringSaver.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

#include <cstdint>
#include <utility>

namespace mlir {
using namespace qir;

/** @brief Qubit allocation mode */
enum class AllocationMode : std::uint8_t {
  Unset,  //!< No allocation mode has been established yet.
  Static, //!< The module uses static qubit allocation.
  Dynamic //!< The module uses dynamic qubit allocation.
};

/**
 * @brief State object for tracking lowering information during QIR conversion
 */
struct LoweringState : QIRMetadata {
  /// Cache static qubit pointers for reuse
  DenseMap<int64_t, Value> staticQubits;

  /// Cache MemRef sizes for reuse
  DenseMap<Value, Value> memrefSizes;

  /// Map from register name to result-array pointer
  llvm::StringMap<Value> resultArrays;

  /// Map from (register name, index) to loaded result
  DenseMap<std::pair<StringRef, int64_t>, Value> loadedResults;

  // Map from register name to its offset
  DenseMap<StringRef, int64_t> registerOffsets;

  /// Map from index to result pointer for non-register results
  DenseMap<int64_t, Value> resultPtrs;

  /// Modifier information
  int64_t inCtrlOp = 0;
  DenseMap<int64_t, SmallVector<Value>> controls;

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
 * requires adding a new `ConvertQCUnitaryOpQIR<...>` specialization to the
 * list of unitary gates below.
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

} // namespace mlir
